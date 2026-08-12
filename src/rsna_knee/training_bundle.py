"""Guarded private Kaggle bundle for the issue #8 gold-training smoke run."""

import base64
import hashlib
import json
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess

from .kaggle import _package_archive, _percent_source_to_notebook
from .training import verify_initialization_artifact

type CommandRunner = Callable[[list[str]], CompletedProcess[str]]
type Waiter = Callable[[float], object]

_KAGGLE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_-]*")
_NOTEBOOK_NAME = "train_gold_model.ipynb"
_CONFIG_NAME = "kaggle-training.toml"
_WEIGHT_CONTRACT_NAME = "resnet18-imagenet1k-v1.json"
_GPU_MACHINE_SHAPE = "NvidiaTeslaT4"


@dataclass(frozen=True, slots=True)
class WeightDataset:
    root: Path
    weight_path: Path
    contract_path: Path
    provenance_path: Path
    metadata_path: Path


@dataclass(frozen=True, slots=True)
class TrainingBundle:
    root: Path
    weight_dataset: WeightDataset
    kernel_root: Path
    notebook_path: Path
    kernel_metadata_path: Path


def build_training_bundle(
    project_root: Path,
    output_root: Path,
    *,
    weight_path: Path,
    dataset_id: str,
    kernel_id: str,
    gpu_approved: bool = False,
) -> TrainingBundle:
    """Build exact private input bytes and GPU metadata only after GPU approval."""
    _require_kaggle_id(dataset_id)
    _require_kaggle_id(kernel_id)
    if not gpu_approved:
        raise PermissionError(
            "GPU training bundle generation requires explicit GPU approval; "
            "no bundle was written"
        )
    initialization = verify_initialization_artifact(weight_path)
    expected_dataset_id = initialization.contract["offline_kaggle_input"]["dataset_id"]
    if dataset_id != expected_dataset_id:
        raise ValueError("Weight Dataset ID differs from initialization contract")

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )

    dataset_root = output_root / "weight-dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    copied_weight = dataset_root / weight_path.name
    copied_contract = dataset_root / _WEIGHT_CONTRACT_NAME
    shutil.copy2(weight_path, copied_weight)
    shutil.copy2(project_root / "contracts" / _WEIGHT_CONTRACT_NAME, copied_contract)
    provenance_path = dataset_root / "weight-provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": dataset_id,
                "visibility": "private",
                "versioning": "immutable Kaggle Dataset version",
                "artifact": {
                    "filename": copied_weight.name,
                    "sha256": initialization.sha256,
                    "size_bytes": initialization.size_bytes,
                },
                "initialization_contract_sha256": initialization.contract_sha256,
                "upstream": initialization.contract["upstream"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    metadata_path = dataset_root / "dataset-metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "description": (
                    "Private, versioned offline copy of the exact torchvision "
                    "ResNet18 IMAGENET1K_V1 state dict for RSNA Knee training."
                ),
                "id": dataset_id,
                "licenses": [{"name": "other"}],
                "title": initialization.contract["offline_kaggle_input"]["title"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    weight_dataset = WeightDataset(
        root=dataset_root,
        weight_path=copied_weight,
        contract_path=copied_contract,
        provenance_path=provenance_path,
        metadata_path=metadata_path,
    )

    kernel_root = output_root / "kernel"
    kernel_root.mkdir(parents=True, exist_ok=True)
    kernel_metadata = _kernel_metadata(dataset_id, kernel_id)
    kernel_metadata_path = kernel_root / "kernel-metadata.json"
    kernel_metadata_path.write_text(
        json.dumps(kernel_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_path = project_root / "notebooks" / "train_gold_model.py"
    config_path = project_root / "configs" / _CONFIG_NAME
    source = source_path.read_text(encoding="utf-8")
    archive = _package_archive(
        project_root / "src" / "rsna_knee",
        additional_contracts=(
            "data-index.json",
            "gold-supervision.json",
            _WEIGHT_CONTRACT_NAME,
        ),
    )
    replacements = {
        "__RSNA_KNEE_PACKAGE_ARCHIVE__": base64.b64encode(archive).decode("ascii"),
        "__RSNA_KNEE_CONFIG__": base64.b64encode(config_path.read_bytes()).decode(
            "ascii"
        ),
        "__RSNA_KNEE_CONFIG_SHA256__": hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest(),
        "__RSNA_KNEE_DATASET_ID__": dataset_id,
        "__RSNA_KNEE_GIT_COMMIT__": git_commit,
        "__RSNA_KNEE_GIT_DIRTY__": json.dumps(git_dirty),
        "__RSNA_KNEE_GPU_APPROVED__": "true",
        "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__": hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest(),
        "__RSNA_KNEE_KERNEL_ID__": kernel_id,
    }
    for marker, value in replacements.items():
        if source.count(marker) < 1:
            raise ValueError(f"Training notebook must contain {marker}")
        source = source.replace(marker, value)
    if "__RSNA_KNEE_" in source:
        raise ValueError("Training notebook contains an unresolved marker")
    notebook_path = kernel_root / _NOTEBOOK_NAME
    notebook_path.write_text(
        json.dumps(_percent_source_to_notebook(source)), encoding="utf-8"
    )
    bundle = TrainingBundle(
        root=output_root,
        weight_dataset=weight_dataset,
        kernel_root=kernel_root,
        notebook_path=notebook_path,
        kernel_metadata_path=kernel_metadata_path,
    )
    _validate_training_bundle(bundle)
    return bundle


def load_training_bundle(root: Path) -> TrainingBundle:
    """Load a previously rendered training bundle for guarded remote execution."""
    dataset_root = root / "weight-dataset"
    contract_path = dataset_root / _WEIGHT_CONTRACT_NAME
    if not contract_path.is_file():
        raise FileNotFoundError(f"Missing initialization contract: {contract_path}")
    contract = json.loads(contract_path.read_text())
    filename = contract.get("artifact", {}).get("filename")
    if not isinstance(filename, str):
        raise TypeError("Initialization contract artifact filename is malformed")
    weight_dataset = WeightDataset(
        root=dataset_root,
        weight_path=dataset_root / filename,
        contract_path=contract_path,
        provenance_path=dataset_root / "weight-provenance.json",
        metadata_path=dataset_root / "dataset-metadata.json",
    )
    kernel_root = root / "kernel"
    return TrainingBundle(
        root=root,
        weight_dataset=weight_dataset,
        kernel_root=kernel_root,
        notebook_path=kernel_root / _NOTEBOOK_NAME,
        kernel_metadata_path=kernel_root / "kernel-metadata.json",
    )


def push_training_bundle(
    bundle: TrainingBundle,
    *,
    approved: bool = False,
    gpu_approved: bool = False,
    reuse_ready_dataset: bool = False,
    runner: CommandRunner | None = None,
    wait: Waiter | None = None,
) -> tuple[CompletedProcess[str] | None, CompletedProcess[str]]:
    """Create the private weight Dataset and execute only after both approvals."""
    if not approved:
        raise PermissionError(
            "Kaggle execution requires explicit remote approval; no command was executed"
        )
    if not gpu_approved:
        raise PermissionError(
            "Kaggle GPU execution requires separate explicit GPU approval; "
            "no command was executed"
        )
    _validate_training_bundle(bundle)
    run_command = runner or _run_command
    status_runner = runner or _run_status_command
    dataset_result = None
    if not reuse_ready_dataset:
        dataset_result = run_command(
            ["kaggle", "datasets", "create", "-p", str(bundle.weight_dataset.root)]
        )
    _wait_for_dataset_ready(
        bundle.weight_dataset,
        runner=status_runner,
        wait=wait or time.sleep,
    )
    kernel_result = run_command(
        ["kaggle", "kernels", "push", "-p", str(bundle.kernel_root)]
    )
    return dataset_result, kernel_result


def _validate_training_bundle(bundle: TrainingBundle) -> None:
    initialization = verify_initialization_artifact(
        bundle.weight_dataset.weight_path,
        contract_path=bundle.weight_dataset.contract_path,
    )
    provenance = json.loads(bundle.weight_dataset.provenance_path.read_text())
    metadata = json.loads(bundle.weight_dataset.metadata_path.read_text())
    kernel_metadata = json.loads(bundle.kernel_metadata_path.read_text())
    dataset_id = metadata.get("id")
    if not isinstance(dataset_id, str):
        raise TypeError("Weight Dataset ID is malformed")
    if provenance.get("visibility") != "private":
        raise ValueError("Weight Dataset provenance must declare private visibility")
    if provenance.get("artifact") != {
        "filename": initialization.path.name,
        "sha256": initialization.sha256,
        "size_bytes": initialization.size_bytes,
    }:
        raise ValueError("Weight Dataset provenance does not match exact bytes")
    if kernel_metadata != _kernel_metadata(
        dataset_id, str(kernel_metadata.get("id", ""))
    ):
        raise ValueError("Training kernel metadata violates the safe GPU policy")
    if not bundle.notebook_path.is_file():
        raise FileNotFoundError(f"Missing training notebook: {bundle.notebook_path}")
    notebook = json.loads(bundle.notebook_path.read_text())
    raw_cells = notebook.get("cells")
    if not isinstance(raw_cells, list):
        raise TypeError("Training notebook cell list is malformed")
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in raw_cells if isinstance(cell, dict)
    )
    if "__RSNA_KNEE_" in source:
        raise ValueError("Training notebook contains an unresolved marker")
    required_fragments = (
        "run_gold_smoke_training",
        "approved_gpu=GPU_APPROVED",
        "verify_training_artifact",
        'Path("/kaggle/input")',
        'Path("/kaggle/working")',
    )
    if any(fragment not in source for fragment in required_fragments):
        raise ValueError("Training notebook omits a required guarded execution step")


def _kernel_metadata(dataset_id: str, kernel_id: str) -> dict[str, object]:
    _require_kaggle_id(dataset_id)
    _require_kaggle_id(kernel_id)
    return {
        "code_file": _NOTEBOOK_NAME,
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": [dataset_id],
        "enable_gpu": "true",
        "enable_internet": "false",
        "enable_tpu": "false",
        "id": kernel_id,
        "is_private": "true",
        "kernel_sources": ["plthon/rsna-knee-data-index"],
        "kernel_type": "notebook",
        "language": "python",
        "machine_shape": _GPU_MACHINE_SHAPE,
        "model_sources": [],
        "title": "RSNA Knee Gold Model Smoke",
    }


def _wait_for_dataset_ready(
    dataset: WeightDataset,
    *,
    runner: CommandRunner,
    wait: Waiter,
    attempts: int = 60,
    interval_seconds: float = 2,
) -> None:
    metadata = json.loads(dataset.metadata_path.read_text())
    dataset_id = metadata.get("id")
    if not isinstance(dataset_id, str):
        raise TypeError("Weight Dataset metadata ID is malformed")
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
                    f"Kaggle weight Dataset entered terminal status: {last_status}"
                )
        if attempt < attempts - 1:
            wait(interval_seconds)
    raise TimeoutError(
        f"Kaggle weight Dataset was not ready after {attempts} checks: {last_status}"
    )


def _require_kaggle_id(value: str) -> None:
    if _KAGGLE_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("Kaggle ID must use resolved owner/slug form")


def _run_command(command: list[str]) -> CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True)


def _run_status_command(command: list[str]) -> CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)
