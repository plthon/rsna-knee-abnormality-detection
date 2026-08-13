"""Guarded Kaggle publication bundle for the gold-only reference baseline."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess

from .kaggle import _percent_source_to_notebook
from .kaggle_run import (
    WheelDataset,
    _require_pure_python_wheel,
    build_wheel_dataset,
)

type CommandRunner = Callable[[list[str]], CompletedProcess[str]]
type Waiter = Callable[[float], object]

_KAGGLE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_-]*")
_KERNEL_TITLE = "RSNA Knee: A Reproducible MRI Baseline"
_KERNEL_SLUG = "rsna-knee-a-reproducible-mri-baseline"
_NOTEBOOK_NAME = "public_gold_baseline.ipynb"
_CONFIG_NAME = "kaggle-baseline.toml"
_GPU_MACHINE_SHAPE = "NvidiaTeslaT4"
_WHEEL_TITLE = "RSNA Knee Gold Baseline Package"
_WHEEL_DESCRIPTION = (
    "Private review input containing the tested rsna-knee wheel and frozen "
    "gold-baseline configuration. Publication requires a separate human action."
)


@dataclass(frozen=True, slots=True)
class PublicBaselineBundle:
    root: Path
    wheel_dataset: WheelDataset
    kernel_root: Path
    notebook_path: Path
    kernel_metadata_path: Path


def build_public_baseline_bundle(
    project_root: Path,
    output_root: Path,
    *,
    wheel_dataset_id: str,
    weight_dataset_id: str,
    kernel_id: str,
    gpu_approved: bool = False,
) -> PublicBaselineBundle:
    """Render the public-facing notebook locally, private by default."""
    _require_kaggle_id(wheel_dataset_id)
    _require_kaggle_id(weight_dataset_id)
    _require_public_kernel_id(kernel_id)
    expected_weight_dataset_id = _weight_dataset_id_from_contract(
        (project_root / "contracts" / "resnet18-imagenet1k-v1.json").read_bytes()
    )
    if weight_dataset_id != expected_weight_dataset_id:
        raise ValueError("Weight Dataset ID differs from initialization contract")
    if not gpu_approved:
        raise PermissionError(
            "GPU notebook metadata requires explicit GPU approval; no bundle was written"
        )

    wheel_dataset = build_wheel_dataset(
        project_root,
        output_root / "wheel-dataset",
        dataset_id=wheel_dataset_id,
        config_path=project_root / "configs" / _CONFIG_NAME,
        description=_WHEEL_DESCRIPTION,
        title=_WHEEL_TITLE,
    )
    if weight_dataset_id != _weight_dataset_id_from_wheel(wheel_dataset.wheel_path):
        raise ValueError("Weight Dataset ID differs from initialization contract")
    provenance = json.loads(wheel_dataset.provenance_path.read_text())
    wheel_record = provenance.get("wheel")
    if not isinstance(wheel_record, dict):
        raise TypeError("Wheel provenance is malformed")

    metadata = _kernel_metadata(
        wheel_dataset_id=wheel_dataset_id,
        weight_dataset_id=weight_dataset_id,
        kernel_id=kernel_id,
    )
    kernel_root = output_root / "kernel"
    kernel_root.mkdir(parents=True, exist_ok=True)
    kernel_metadata_path = kernel_root / "kernel-metadata.json"
    kernel_metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    source_path = project_root / "notebooks" / "public_gold_baseline.py"
    source = source_path.read_text(encoding="utf-8")
    config_bytes = wheel_dataset.config_path.read_bytes()
    replacements = {
        "__RSNA_KNEE_APPROVED_GPU__": "true",
        "__RSNA_KNEE_BASELINE_CONFIG__": base64.b64encode(config_bytes).decode("ascii"),
        "__RSNA_KNEE_CONFIG_SHA256__": hashlib.sha256(config_bytes).hexdigest(),
        "__RSNA_KNEE_GIT_COMMIT__": str(provenance["git_commit"]),
        "__RSNA_KNEE_GIT_DIRTY__": json.dumps(bool(provenance["git_dirty"])),
        "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__": hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest(),
        "__RSNA_KNEE_KERNEL_ID__": kernel_id,
        "__RSNA_KNEE_WEIGHT_DATASET_ID__": weight_dataset_id,
        "__RSNA_KNEE_WHEEL_DATASET_ID__": wheel_dataset_id,
        "__RSNA_KNEE_WHEEL_DATASET_SLUG__": wheel_dataset_id.partition("/")[2],
        "__RSNA_KNEE_WHEEL_FILENAME__": str(wheel_record["filename"]),
        "__RSNA_KNEE_WHEEL_SHA256__": str(wheel_record["sha256"]),
    }
    for marker, value in replacements.items():
        if source.count(marker) < 1:
            raise ValueError(f"Public baseline notebook must contain {marker}")
        source = source.replace(marker, value)
    if "__RSNA_KNEE_" in source:
        raise ValueError("Public baseline notebook contains an unresolved marker")

    notebook_path = kernel_root / _NOTEBOOK_NAME
    notebook_path.write_text(
        json.dumps(_percent_source_to_notebook(source)), encoding="utf-8"
    )
    bundle = PublicBaselineBundle(
        root=output_root,
        wheel_dataset=wheel_dataset,
        kernel_root=kernel_root,
        notebook_path=notebook_path,
        kernel_metadata_path=kernel_metadata_path,
    )
    _validate_public_baseline_bundle(bundle)
    return bundle


def load_public_baseline_bundle(root: Path) -> PublicBaselineBundle:
    """Load a locally rendered publication bundle for guarded execution."""
    dataset_root = root / "wheel-dataset"
    wheels = list(dataset_root.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected one wheel, found {len(wheels)}")
    wheel_dataset = WheelDataset(
        root=dataset_root,
        wheel_path=wheels[0],
        provenance_path=dataset_root / "wheel-provenance.json",
        metadata_path=dataset_root / "dataset-metadata.json",
        config_path=dataset_root / _CONFIG_NAME,
    )
    kernel_root = root / "kernel"
    return PublicBaselineBundle(
        root=root,
        wheel_dataset=wheel_dataset,
        kernel_root=kernel_root,
        notebook_path=kernel_root / _NOTEBOOK_NAME,
        kernel_metadata_path=kernel_root / "kernel-metadata.json",
    )


def push_public_baseline_bundle(
    bundle: PublicBaselineBundle,
    *,
    approved: bool = False,
    gpu_approved: bool = False,
    reuse_ready_dataset: bool = False,
    runner: CommandRunner | None = None,
    wait: Waiter | None = None,
) -> tuple[CompletedProcess[str] | None, CompletedProcess[str]]:
    """Upload inputs and execute privately after remote and GPU approvals."""
    if not approved:
        raise PermissionError(
            "Kaggle execution requires explicit remote approval; no command was executed"
        )
    if not gpu_approved:
        raise PermissionError(
            "Kaggle GPU execution requires separate explicit GPU approval; "
            "no command was executed"
        )
    _validate_public_baseline_bundle(bundle)
    run = runner or _run_command
    status_run = runner or _run_status_command
    dataset_result = None
    if not reuse_ready_dataset:
        dataset_result = run(
            ["kaggle", "datasets", "create", "-p", str(bundle.wheel_dataset.root)]
        )
    dataset_id = json.loads(bundle.wheel_dataset.metadata_path.read_text())["id"]
    if not isinstance(dataset_id, str):
        raise TypeError("Wheel Dataset ID is malformed")
    _wait_for_dataset_ready(
        dataset_id,
        runner=status_run,
        wait=wait or time.sleep,
    )
    kernel_result = run(["kaggle", "kernels", "push", "-p", str(bundle.kernel_root)])
    return dataset_result, kernel_result


def _validate_public_baseline_bundle(bundle: PublicBaselineBundle) -> None:
    dataset_metadata = json.loads(bundle.wheel_dataset.metadata_path.read_text())
    dataset_id = dataset_metadata.get("id")
    if not isinstance(dataset_id, str):
        raise TypeError("Wheel Dataset ID is malformed")
    if dataset_metadata != {
        "description": _WHEEL_DESCRIPTION,
        "id": dataset_id,
        "licenses": [{"name": "other"}],
        "title": _WHEEL_TITLE,
    }:
        raise ValueError("Wheel Dataset metadata violates the review policy")

    metadata = json.loads(bundle.kernel_metadata_path.read_text())
    kernel_id = metadata.get("id")
    sources = metadata.get("dataset_sources")
    if (
        not isinstance(kernel_id, str)
        or not isinstance(sources, list)
        or len(sources) != 2
        or not isinstance(sources[1], str)
    ):
        raise TypeError("Public baseline kernel metadata is malformed")
    if metadata != _kernel_metadata(
        wheel_dataset_id=dataset_id,
        weight_dataset_id=sources[1],
        kernel_id=kernel_id,
    ):
        raise ValueError("Public baseline kernel metadata violates the safe policy")
    if sources[1] != _weight_dataset_id_from_wheel(bundle.wheel_dataset.wheel_path):
        raise ValueError("Weight Dataset ID differs from initialization contract")

    provenance = json.loads(bundle.wheel_dataset.provenance_path.read_text())
    wheel_record = provenance.get("wheel")
    expected_wheel_hash = hashlib.sha256(
        bundle.wheel_dataset.wheel_path.read_bytes()
    ).hexdigest()
    if (
        provenance.get("dataset_id") != dataset_id
        or not isinstance(wheel_record, dict)
        or wheel_record.get("filename") != bundle.wheel_dataset.wheel_path.name
        or wheel_record.get("sha256") != expected_wheel_hash
    ):
        raise ValueError("Public baseline wheel provenance does not match exact bytes")

    notebook = json.loads(bundle.notebook_path.read_text())
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if isinstance(cell, dict)
    )
    required = (
        "pip",
        "--no-index",
        "--no-deps",
        "load_reference_baseline_summary",
        "reproduce_reference_baseline",
        "Patient-level validation",
        "54 patients",
        "4 different patients",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "PatientID",
        "ImagePositionPatient",
        "/kaggle/input/competitions/rsna-knee-abnormality-detection",
        "teaching_slice_indices",
        "preprocess_series_result",
        "volume_middle",
        "valid_slice_features",
        "series_features",
        "study_features",
        "binary_cross_entropy_with_logits",
        "Evidence for model selection",
        "Inadequate",
        "90.65%",
        "Complete notebook run time (minutes)",
        "Reproducibility details",
        dataset_id,
        sources[1],
        kernel_id,
        expected_wheel_hash,
        hashlib.sha256(bundle.wheel_dataset.config_path.read_bytes()).hexdigest(),
    )
    if "__RSNA_KNEE_" in source or any(value not in source for value in required):
        raise ValueError("Public baseline notebook omits a publication safeguard")
    forbidden = (
        "report-derived supervision is used",
        "leaderboard score",
        "issue #",
        "issue_",
        "validated experiment",
        "evidence judgment",
        "adequacy_for_issue",
        "artifact_type",
        "private_kaggle_gpu_reproduction",
        "private_review_kaggle_gpu_publication_reproduction",
        "discover_submission_inputs",
    )
    if any(value in source.casefold() for value in forbidden):
        raise ValueError("Public baseline notebook contains a prohibited claim")
    _require_pure_python_wheel(bundle.wheel_dataset.wheel_path)


def _weight_dataset_id_from_wheel(wheel_path: Path) -> str:
    with zipfile.ZipFile(wheel_path) as wheel:
        raw = wheel.read("rsna_knee/resnet18-imagenet1k-v1.json")
    return _weight_dataset_id_from_contract(raw)


def _weight_dataset_id_from_contract(raw: bytes) -> str:
    contract = json.loads(raw)
    if not isinstance(contract, dict):
        raise TypeError("Packaged initialization contract is malformed")
    offline_input = contract.get("offline_kaggle_input")
    if not isinstance(offline_input, dict):
        raise TypeError("Packaged initialization input contract is malformed")
    dataset_id = offline_input.get("dataset_id")
    if not isinstance(dataset_id, str):
        raise TypeError("Packaged initialization Dataset ID is malformed")
    return dataset_id


def _kernel_metadata(
    *, wheel_dataset_id: str, weight_dataset_id: str, kernel_id: str
) -> dict[str, object]:
    _require_kaggle_id(wheel_dataset_id)
    _require_kaggle_id(weight_dataset_id)
    _require_public_kernel_id(kernel_id)
    return {
        "code_file": _NOTEBOOK_NAME,
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": [wheel_dataset_id, weight_dataset_id],
        "enable_gpu": "true",
        "enable_internet": "false",
        "enable_tpu": "false",
        "id": kernel_id,
        "is_private": "true",
        "kernel_sources": [],
        "kernel_type": "notebook",
        "language": "python",
        "machine_shape": _GPU_MACHINE_SHAPE,
        "model_sources": [],
        "title": _KERNEL_TITLE,
    }


def _wait_for_dataset_ready(
    dataset_id: str,
    *,
    runner: CommandRunner,
    wait: Waiter,
    attempts: int = 60,
    interval_seconds: float = 2,
) -> None:
    command = ["kaggle", "datasets", "status", dataset_id]
    last_status = "unavailable"
    for attempt in range(attempts):
        try:
            result = runner(command)
        except CalledProcessError:
            last_status = "unavailable"
        else:
            last_status = (result.stdout or "").strip().lower()
            if last_status == "ready":
                return
            if last_status in {"error", "failed", "failure"}:
                raise RuntimeError(
                    f"Kaggle wheel Dataset entered terminal status: {last_status}"
                )
        if attempt < attempts - 1:
            wait(interval_seconds)
    raise TimeoutError(
        f"Kaggle wheel Dataset was not ready after {attempts} checks: {last_status}"
    )


def _require_public_kernel_id(value: str) -> None:
    _require_kaggle_id(value)
    if value.partition("/")[2] != _KERNEL_SLUG:
        raise ValueError(f"Public baseline kernel slug must be {_KERNEL_SLUG}")


def _require_kaggle_id(value: str) -> None:
    if _KAGGLE_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("Kaggle ID must use resolved owner/slug form")


def _run_command(command: list[str]) -> CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True)


def _run_status_command(command: list[str]) -> CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)
