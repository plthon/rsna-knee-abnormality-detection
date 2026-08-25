"""One guarded boundary for rendering and optionally pushing Kaggle notebooks."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
import subprocess
import time
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess
from typing import Any

type CommandRunner = Callable[[list[str]], CompletedProcess[str]]
type Waiter = Callable[[float], object]

_PACKAGE_MARKER = "__RSNA_KNEE_PACKAGE_ARCHIVE__"
_KAGGLE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_-]*")
_KAGGLE_MODEL_SOURCE_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_.-]*/"
    r"[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_.-]*/[1-9][0-9]*"
)
_PLACEHOLDER_PARTS = {
    "DATASET_ID",
    "DATASET_SLUG",
    "INSERT_SLUG_HERE",
    "INSERT_USERNAME_HERE",
    "KERNEL_ID",
    "KERNEL_SLUG",
    "OWNER",
    "USERNAME",
    "YOUR_USERNAME",
}
_PUBLIC_ORDERING_FORBIDDEN = (
    ".artifacts/",
    "build/kaggle/",
    "credentials/",
    "kaggle.json",
    "rsna-knee-offline-wheel",
    "rsna_knee_source.zip",
    "run-manifest.json",
    "index-manifest.json",
    "pseudo-label",
    "pseudo_label",
    "report mapping",
    "report-language",
    "report language",
    "validation split",
    "leaderboard",
    "checkpoint",
    "attention/mil",
    "2.5d",
    "ensemble strategy",
    "calibration strateg",
    "plthon/rsna-knee-real-data-inspection",
    "plthon/rsna-knee-data-index",
)
_PUBLIC_BASELINE_FORBIDDEN = (
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
_KERNEL_TITLE_MAX_LENGTH = 50
_KERNEL_SLUG_MAX_LENGTH = 50
_GPU_MACHINE_SHAPE = "NvidiaTeslaT4"
_PUBLIC_BASELINE_KERNEL = "rsna-knee-a-reproducible-mri-baseline"
_PUBLIC_BASELINE_TITLE = "RSNA Knee: A Reproducible MRI Baseline"
_PUBLIC_BASELINE_CONTRACTS = (
    "data-index.json",
    "gold-baseline.json",
    "gold-supervision.json",
    "gold-training.json",
    "resnet18-imagenet1k-v1.json",
)


@dataclass(frozen=True, slots=True)
class KaggleSpec:
    source_name: str
    notebook_name: str
    kernel_id: str
    title: str
    competition_sources: tuple[str, ...] = ()
    dataset_sources: tuple[str, ...] = ()
    kernel_sources: tuple[str, ...] = ()
    model_sources: tuple[str, ...] = ()
    gpu: bool = False
    machine_shape: str = ""
    replacements: Mapping[str, str] = field(default_factory=dict)
    embed_package: bool = False
    package_contracts: tuple[str, ...] = ()
    validator: str = ""


@dataclass(frozen=True, slots=True)
class WheelDataset:
    root: Path
    wheel_path: Path
    provenance_path: Path
    metadata_path: Path
    config_path: Path


@dataclass(frozen=True, slots=True)
class KaggleBundle:
    root: Path
    kernel_root: Path
    notebook_path: Path
    kernel_metadata_path: Path
    provenance_path: Path
    dataset: WheelDataset | None = None


def build_kaggle_bundle(
    project_root: Path,
    output_root: Path,
    spec: KaggleSpec,
    *,
    dataset: WheelDataset | None = None,
    gpu_approved: bool = False,
) -> KaggleBundle:
    """Render one validated notebook bundle without contacting Kaggle."""
    _validate_spec(spec, gpu_approved=gpu_approved)
    source_path = project_root / "notebooks" / spec.source_name
    source = source_path.read_text(encoding="utf-8")
    if spec.validator == "ordering":
        _validate_public_ordering_source(source)
    if spec.embed_package or spec.package_contracts:
        if source.count(_PACKAGE_MARKER) != 1:
            raise ValueError(
                "Packaged notebook source must contain exactly one package marker"
            )
        source = source.replace(
            _PACKAGE_MARKER,
            base64.b64encode(
                _package_archive(
                    project_root / "src" / "rsna_knee",
                    additional_contracts=spec.package_contracts,
                )
            ).decode("ascii"),
        )
    for marker, value in spec.replacements.items():
        if marker not in source:
            raise ValueError(f"Notebook source does not contain {marker}")
        source = source.replace(marker, value)
    if "__RSNA_KNEE_" in source:
        raise ValueError("Notebook contains an unresolved Kaggle placeholder")
    if spec.validator == "baseline":
        _validate_public_baseline_source(source)
    elif spec.validator == "reports":
        _validate_reports_source(source)

    metadata = _kernel_metadata(spec)
    kernel_root = output_root / "kernel"
    kernel_root.mkdir(parents=True, exist_ok=True)
    notebook_path = kernel_root / spec.notebook_name
    metadata_path = kernel_root / "kernel-metadata.json"
    notebook_path.write_text(
        json.dumps(_percent_source_to_notebook(source)), encoding="utf-8"
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    provenance_path = output_root / "bundle-provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": {
                    "path": spec.source_name,
                    "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                },
                "kernel": {
                    "id": spec.kernel_id,
                    "metadata_sha256": _sha256(metadata_path),
                },
                "config_sha256": (
                    _sha256(dataset.config_path) if dataset is not None else None
                ),
                "dataset_id": _dataset_id(dataset),
                "validator": spec.validator,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    bundle = KaggleBundle(
        root=output_root,
        kernel_root=kernel_root,
        notebook_path=notebook_path,
        kernel_metadata_path=metadata_path,
        provenance_path=provenance_path,
        dataset=dataset,
    )
    _validate_bundle(bundle)
    return bundle


def load_kaggle_bundle(root: Path) -> KaggleBundle:
    """Load a rendered bundle for the guarded push command."""
    kernel_root = root / "kernel"
    metadata = json.loads((kernel_root / "kernel-metadata.json").read_text())
    code_file = metadata.get("code_file")
    if not isinstance(code_file, str):
        raise TypeError("Kaggle metadata code_file is malformed")
    dataset_root = root / "wheel-dataset"
    dataset = _load_wheel_dataset(dataset_root) if dataset_root.is_dir() else None
    return KaggleBundle(
        root=root,
        kernel_root=kernel_root,
        notebook_path=kernel_root / code_file,
        kernel_metadata_path=kernel_root / "kernel-metadata.json",
        provenance_path=root / "bundle-provenance.json",
        dataset=dataset,
    )


def push_kaggle_bundle(
    bundle: KaggleBundle,
    *,
    approved: bool = False,
    gpu_approved: bool = False,
    reuse_ready_dataset: bool = False,
    runner: CommandRunner | None = None,
    wait: Waiter | None = None,
) -> tuple[CompletedProcess[str] | None, CompletedProcess[str]]:
    """Push only after explicit remote approval and, separately, GPU approval."""
    if not approved:
        raise PermissionError(
            "Kaggle push requires explicit remote approval; no command was executed"
        )
    gpu_enabled = _validate_bundle(bundle)
    if gpu_enabled and not gpu_approved:
        raise PermissionError(
            "Kaggle GPU execution requires separate explicit GPU approval; "
            "no command was executed"
        )
    run = runner or _run_command
    status_run = runner or _run_status_command
    dataset_result: CompletedProcess[str] | None = None
    if bundle.dataset is not None:
        if not reuse_ready_dataset:
            dataset_result = run(
                ["kaggle", "datasets", "create", "-p", str(bundle.dataset.root)]
            )
        _wait_for_dataset_ready(
            bundle.dataset,
            runner=status_run,
            wait=wait or time.sleep,
        )
    kernel_result = run(["kaggle", "kernels", "push", "-p", str(bundle.kernel_root)])
    return dataset_result, kernel_result


def pull_kaggle_output(
    bundle: KaggleBundle,
    output_root: Path,
    *,
    artifact_root: Path,
    filenames: tuple[str, ...],
    runner: CommandRunner | None = None,
) -> tuple[CompletedProcess[str], ...]:
    """Retrieve selected kernel outputs into the ignored local artifact tree."""
    _validate_bundle(bundle)
    resolved_artifacts = artifact_root.resolve()
    resolved_output = output_root.resolve()
    if artifact_root.name != ".artifacts" or not resolved_output.is_relative_to(
        resolved_artifacts
    ):
        raise ValueError("Kaggle output must stay inside the ignored artifact root")
    if not filenames:
        raise ValueError("At least one Kaggle output filename is required")
    private_row_outputs = {
        "public-label-row-validation.csv",
        "report-labels-v1.csv",
        "report-evidence-v1.jsonl",
        "report-llm-labels-v1.csv",
        "report-llm-study-status-v1.csv",
        "report-llm-checkpoint.json",
        "report-llm-stability-first.json",
        "report-llm-stability-second.json",
        "source-comparison-tables.json",
    }
    if private_row_outputs & set(filenames):
        raise PermissionError(
            "Private row-level report artifacts must remain on Kaggle"
        )
    if any(Path(filename).name != filename for filename in filenames):
        raise ValueError("Kaggle output filenames must be basenames")
    metadata = json.loads(bundle.kernel_metadata_path.read_text())
    kernel_id = metadata.get("id")
    if not isinstance(kernel_id, str):
        raise TypeError("Kaggle kernel ID is malformed")
    output_root.mkdir(parents=True, exist_ok=True)
    run = runner or _run_command
    return tuple(
        run(
            [
                "kaggle",
                "kernels",
                "output",
                kernel_id,
                "-p",
                str(output_root),
                "--file-pattern",
                f"^{re.escape(filename)}$",
            ]
        )
        for filename in filenames
    )


def build_wheel_dataset(
    project_root: Path,
    output_root: Path,
    *,
    dataset_id: str,
    config_path: Path | None = None,
    contract_names: tuple[str, ...] = (),
    description: str = "Private offline package for RSNA Knee experiments.",
    title: str = "RSNA Knee Offline Wheel",
) -> WheelDataset:
    """Build a pure-Python wheel and attach only explicitly requested files."""
    _require_resolved_kaggle_id(dataset_id)
    git_commit, git_dirty = _git_state(project_root)
    output_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "uv",
            "build",
            "--wheel",
            "--out-dir",
            str(output_root),
            str(project_root),
        ],
        check=True,
        text=True,
    )
    wheels = list(output_root.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected one wheel, found {len(wheels)}")
    wheel_path = wheels[0]
    _require_pure_python_wheel(wheel_path)
    provenance_path = output_root / "wheel-provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "git_commit": git_commit,
                "git_dirty": git_dirty,
                "schema_version": 1,
                "wheel": {
                    "filename": wheel_path.name,
                    "sha256": _sha256(wheel_path),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    metadata_path = output_root / "dataset-metadata.json"
    metadata_path.write_text(
        json.dumps(
            _dataset_metadata(dataset_id, description=description, title=title),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    source_config_path = config_path or project_root / "configs" / "kaggle.toml"
    copied_config_path = output_root / source_config_path.name
    shutil.copyfile(source_config_path, copied_config_path)
    for name in contract_names:
        if Path(name).name != name:
            raise ValueError(f"Contract name must be a basename: {name}")
        source_contract = project_root / "contracts" / name
        if not source_contract.is_file():
            raise FileNotFoundError(f"Missing requested contract: {source_contract}")
        shutil.copy2(source_contract, output_root / name)
    return WheelDataset(
        root=output_root,
        wheel_path=wheel_path,
        provenance_path=provenance_path,
        metadata_path=metadata_path,
        config_path=copied_config_path,
    )


def build_public_notebook_bundle(project_root: Path, output_root: Path) -> KaggleBundle:
    return build_kaggle_bundle(
        project_root,
        output_root,
        KaggleSpec(
            source_name="public_dicom_ordering.py",
            notebook_name="rsna_knee_dicom_ordering.ipynb",
            kernel_id="plthon/rsna-knee-mri-correct-dicom-ordering-sampling",
            title="RSNA Knee MRI | Correct DICOM Ordering & Sampling",
            competition_sources=("rsna-knee-abnormality-detection",),
            validator="ordering",
        ),
    )


def build_reports_bundle(
    project_root: Path, output_root: Path, *, stage: str
) -> KaggleBundle:
    """Build the private CPU report audit or extraction notebook."""
    if stage not in {"audit", "extract"}:
        raise ValueError("Report stage must be audit or extract")
    contract_names: tuple[str, ...] = ()
    if stage == "extract":
        contract_path = project_root / "contracts" / "report-extractor-v1.json"
        if not contract_path.is_file():
            raise FileNotFoundError(
                "The extract stage requires frozen contracts/report-extractor-v1.json"
            )
        contract_names = (contract_path.name,)
    source_path = project_root / "notebooks" / "report_labels.py"
    git_commit, git_dirty = _git_state(project_root)
    bundle = build_kaggle_bundle(
        project_root,
        output_root,
        KaggleSpec(
            source_name=source_path.name,
            notebook_name="report_labels.ipynb",
            kernel_id="plthon/rsna-knee-report-labels",
            title="RSNA Knee Report Labels",
            competition_sources=("rsna-knee-abnormality-detection",),
            replacements={
                "__RSNA_KNEE_GIT_COMMIT__": git_commit,
                "__RSNA_KNEE_GIT_DIRTY__": json.dumps(git_dirty),
                "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__": hashlib.sha256(
                    source_path.read_bytes()
                ).hexdigest(),
                "__RSNA_KNEE_RENDERED_NOTEBOOK_SHA256__": "0" * 64,
                "__RSNA_KNEE_REPORT_STAGE__": stage,
            },
            embed_package=True,
            package_contracts=contract_names,
            validator="reports",
        ),
    )
    return _record_rendered_notebook_hash(bundle, label="Report")


def build_public_label_audit_bundle(
    project_root: Path, output_root: Path, *, stage: str
) -> KaggleBundle:
    """Build the private CPU audit of the frozen public label dataset."""
    if stage != "audit":
        raise ValueError("Public label audit stage must be audit")
    contract_path = project_root / "contracts" / "public-report-label-dataset-v1.json"
    if not contract_path.is_file():
        raise FileNotFoundError(
            "Public label audit requires its frozen dataset contract"
        )
    source_path = project_root / "notebooks" / "public_label_audit.py"
    git_commit, git_dirty = _git_state(project_root)
    bundle = build_kaggle_bundle(
        project_root,
        output_root,
        KaggleSpec(
            source_name=source_path.name,
            notebook_name="public_label_audit.ipynb",
            kernel_id="plthon/rsna-knee-public-label-audit",
            title="RSNA Knee Public Label Audit",
            competition_sources=("rsna-knee-abnormality-detection",),
            dataset_sources=("pilkwang/rsna-knee-llm-labels",),
            kernel_sources=("plthon/rsna-knee-report-labels",),
            replacements={
                "__RSNA_KNEE_GIT_COMMIT__": git_commit,
                "__RSNA_KNEE_GIT_DIRTY__": json.dumps(git_dirty),
                "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__": hashlib.sha256(
                    source_path.read_bytes()
                ).hexdigest(),
                "__RSNA_KNEE_RENDERED_NOTEBOOK_SHA256__": "0" * 64,
                "__RSNA_KNEE_PUBLIC_LABEL_STAGE__": stage,
            },
            embed_package=True,
            package_contracts=(contract_path.name,),
            validator="public_labels",
        ),
    )
    return _record_rendered_notebook_hash(bundle, label="Public label audit")


def build_report_llm_bundle(
    project_root: Path, output_root: Path, *, stage: str
) -> KaggleBundle:
    """Build the private offline Qwen report-label notebook."""
    if stage not in {"smoke", "extract"}:
        raise ValueError("Report LLM stage must be smoke or extract")
    contract_path = project_root / "contracts" / "report-llm-extractor-v1.json"
    if not contract_path.is_file():
        raise FileNotFoundError(
            "Report LLM requires contracts/report-llm-extractor-v1.json"
        )
    source_path = project_root / "notebooks" / "report_llm_labels.py"
    git_commit, git_dirty = _git_state(project_root)
    bundle = build_kaggle_bundle(
        project_root,
        output_root,
        KaggleSpec(
            source_name=source_path.name,
            notebook_name="report_llm_labels.ipynb",
            kernel_id="plthon/rsna-knee-report-llm-labels",
            title="RSNA Knee Report LLM Labels",
            competition_sources=("rsna-knee-abnormality-detection",),
            kernel_sources=("plthon/rsna-knee-report-labels",),
            model_sources=("qwen-lm/qwen2.5/transformers/3b-instruct/1",),
            gpu=True,
            machine_shape=_GPU_MACHINE_SHAPE,
            replacements={
                "__RSNA_KNEE_GIT_COMMIT__": git_commit,
                "__RSNA_KNEE_GIT_DIRTY__": json.dumps(git_dirty),
                "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__": hashlib.sha256(
                    source_path.read_bytes()
                ).hexdigest(),
                "__RSNA_KNEE_RENDERED_NOTEBOOK_SHA256__": "0" * 64,
                "__RSNA_KNEE_REPORT_LLM_STAGE__": stage,
            },
            embed_package=True,
            package_contracts=(contract_path.name,),
            validator="report_llm",
        ),
        gpu_approved=True,
    )
    return _record_rendered_notebook_hash(bundle, label="Report LLM")


def build_report_source_comparison_bundle(
    project_root: Path, output_root: Path, *, stage: str
) -> KaggleBundle:
    """Build the private CPU three-way report-source comparison notebook."""
    if stage != "compare":
        raise ValueError("Report source comparison stage must be compare")
    contract_names = (
        "report-extractor-v1.json",
        "report-llm-extractor-v1.json",
        "public-report-label-dataset-v1.json",
        "report-source-comparison-v1.json",
    )
    for name in (*contract_names,):
        contract_path = project_root / "contracts" / name
        if not contract_path.is_file():
            raise FileNotFoundError(f"Source comparison requires contracts/{name}")
    source_path = project_root / "notebooks" / "report_source_comparison.py"
    git_commit, git_dirty = _git_state(project_root)
    bundle = build_kaggle_bundle(
        project_root,
        output_root,
        KaggleSpec(
            source_name=source_path.name,
            notebook_name="report_source_comparison.ipynb",
            kernel_id="plthon/rsna-knee-report-source-comparison",
            title="RSNA Knee Report Source Comparison",
            competition_sources=("rsna-knee-abnormality-detection",),
            dataset_sources=("pilkwang/rsna-knee-llm-labels",),
            kernel_sources=(
                "plthon/rsna-knee-report-labels",
                "plthon/rsna-knee-report-llm-labels",
            ),
            replacements={
                "__RSNA_KNEE_GIT_COMMIT__": git_commit,
                "__RSNA_KNEE_GIT_DIRTY__": json.dumps(git_dirty),
                "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__": hashlib.sha256(
                    source_path.read_bytes()
                ).hexdigest(),
                "__RSNA_KNEE_RENDERED_NOTEBOOK_SHA256__": "0" * 64,
                "__RSNA_KNEE_REPORT_SOURCE_STAGE__": stage,
            },
            embed_package=True,
            package_contracts=contract_names,
            validator="source_comparison",
        ),
    )
    return _record_rendered_notebook_hash(bundle, label="Report source comparison")


def _record_rendered_notebook_hash(bundle: KaggleBundle, *, label: str) -> KaggleBundle:
    rendered = bundle.notebook_path.read_text(encoding="utf-8")
    zero_hash = "0" * 64
    if zero_hash not in rendered:
        raise ValueError(f"{label} notebook must contain its rendered-hash field")
    canonical_sha256 = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    bundle.notebook_path.write_text(
        rendered.replace(zero_hash, canonical_sha256), encoding="utf-8"
    )
    provenance = json.loads(bundle.provenance_path.read_text(encoding="utf-8"))
    provenance["rendered_notebook"] = {
        "sha256": canonical_sha256,
        "canonicalization": (
            "rendered notebook with its recorded SHA-256 replaced by 64 zeroes"
        ),
    }
    bundle.provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _validate_bundle(bundle)
    return bundle


def build_public_baseline_bundle(
    project_root: Path,
    output_root: Path,
    *,
    wheel_dataset_id: str,
    weight_dataset_id: str,
    kernel_id: str,
    gpu_approved: bool = False,
) -> KaggleBundle:
    _require_resolved_kaggle_id(wheel_dataset_id)
    _require_resolved_kaggle_id(weight_dataset_id)
    if kernel_id.partition("/")[2] != _PUBLIC_BASELINE_KERNEL:
        raise ValueError(
            f"Public baseline kernel slug must be {_PUBLIC_BASELINE_KERNEL}"
        )
    expected_weight_dataset_id = _weight_dataset_id_from_contract(
        (project_root / "contracts" / "resnet18-imagenet1k-v1.json").read_bytes()
    )
    if weight_dataset_id != expected_weight_dataset_id:
        raise ValueError("Weight Dataset ID differs from initialization contract")
    if not gpu_approved:
        raise PermissionError(
            "GPU notebook metadata requires explicit GPU approval; no bundle was written"
        )
    config_path = project_root / "configs" / "kaggle-baseline.toml"
    dataset = build_wheel_dataset(
        project_root,
        output_root / "wheel-dataset",
        dataset_id=wheel_dataset_id,
        config_path=config_path,
        contract_names=_PUBLIC_BASELINE_CONTRACTS,
        description=(
            "Private review input containing the tested rsna-knee wheel and "
            "frozen gold-baseline configuration."
        ),
        title="RSNA Knee Gold Baseline Package",
    )
    provenance = json.loads(dataset.provenance_path.read_text())
    wheel_record = provenance["wheel"]
    config_bytes = dataset.config_path.read_bytes()
    source_path = project_root / "notebooks" / "public_gold_baseline.py"
    return build_kaggle_bundle(
        project_root,
        output_root,
        KaggleSpec(
            source_name=source_path.name,
            notebook_name="public_gold_baseline.ipynb",
            kernel_id=kernel_id,
            title=_PUBLIC_BASELINE_TITLE,
            competition_sources=("rsna-knee-abnormality-detection",),
            dataset_sources=(wheel_dataset_id, weight_dataset_id),
            gpu=True,
            machine_shape=_GPU_MACHINE_SHAPE,
            replacements={
                "__RSNA_KNEE_APPROVED_GPU__": "true",
                "__RSNA_KNEE_BASELINE_CONFIG__": base64.b64encode(config_bytes).decode(
                    "ascii"
                ),
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
            },
            validator="baseline",
        ),
        dataset=dataset,
        gpu_approved=True,
    )


def _validate_spec(spec: KaggleSpec, *, gpu_approved: bool) -> None:
    _require_resolved_kaggle_id(spec.kernel_id)
    if Path(spec.notebook_name).name != spec.notebook_name:
        raise ValueError("Kaggle notebook name must be a basename")
    for dataset_id in spec.dataset_sources:
        _require_resolved_kaggle_id(dataset_id)
    for kernel_source in spec.kernel_sources:
        _require_resolved_kaggle_id(kernel_source)
    for model_source in spec.model_sources:
        if _KAGGLE_MODEL_SOURCE_PATTERN.fullmatch(model_source) is None:
            raise ValueError(
                f"Invalid or unresolved Kaggle model source: {model_source}"
            )
    if len(spec.title) > _KERNEL_TITLE_MAX_LENGTH:
        raise ValueError("Kaggle kernel title is too long")
    slug = _kernel_title_slug(spec.title)
    if len(slug) > _KERNEL_SLUG_MAX_LENGTH:
        raise ValueError("Kaggle kernel slug is too long")
    if spec.kernel_id.partition("/")[2] != slug:
        raise ValueError(f"Kaggle kernel id slug must resolve from title: {slug}")
    if spec.gpu and not gpu_approved:
        raise PermissionError(
            "GPU metadata requires a separate GPU approval; no bundle was written"
        )
    if not spec.gpu and spec.machine_shape:
        raise ValueError("CPU metadata cannot select an accelerator shape")


def _kernel_metadata(spec: KaggleSpec) -> dict[str, object]:
    if spec.gpu and not spec.machine_shape:
        machine_shape = ""
    else:
        machine_shape = spec.machine_shape
    return {
        "code_file": spec.notebook_name,
        "competition_sources": list(spec.competition_sources),
        "dataset_sources": list(spec.dataset_sources),
        "enable_gpu": str(spec.gpu).lower(),
        "enable_internet": "false",
        "enable_tpu": "false",
        "id": spec.kernel_id,
        "is_private": "true",
        "kernel_sources": list(spec.kernel_sources),
        "kernel_type": "notebook",
        "language": "python",
        "machine_shape": machine_shape,
        "model_sources": list(spec.model_sources),
        "title": spec.title,
    }


def _validate_bundle(bundle: KaggleBundle) -> bool:
    metadata = json.loads(bundle.kernel_metadata_path.read_text())
    kernel_id = metadata.get("id")
    if not isinstance(kernel_id, str):
        raise TypeError("Kaggle kernel ID is malformed")
    _require_resolved_kaggle_id(kernel_id)
    if (
        metadata.get("is_private") != "true"
        or metadata.get("enable_internet") != "false"
        or metadata.get("enable_tpu") != "false"
        or metadata.get("kernel_type") != "notebook"
        or metadata.get("language") != "python"
    ):
        raise ValueError("Kaggle metadata violates the private offline policy")
    gpu_value = metadata.get("enable_gpu")
    if not isinstance(gpu_value, str) or gpu_value not in {"true", "false"}:
        raise ValueError("Kaggle GPU metadata must be true or false")
    code_file = metadata.get("code_file")
    if (
        not isinstance(code_file, str)
        or Path(code_file).name != code_file
        or bundle.kernel_root / code_file != bundle.notebook_path
    ):
        raise ValueError("Kaggle metadata code_file does not match the bundle")
    if not bundle.notebook_path.is_file():
        raise FileNotFoundError(f"Missing rendered notebook: {bundle.notebook_path}")
    source = bundle.notebook_path.read_text()
    if "__RSNA_KNEE_" in source:
        raise ValueError("Rendered notebook contains an unresolved placeholder")
    if not bundle.provenance_path.is_file():
        raise FileNotFoundError(f"Missing bundle provenance: {bundle.provenance_path}")
    provenance = json.loads(bundle.provenance_path.read_text())
    kernel_provenance = provenance.get("kernel")
    source_provenance = provenance.get("source")
    if (
        not isinstance(kernel_provenance, dict)
        or kernel_provenance.get("id") != kernel_id
        or kernel_provenance.get("metadata_sha256")
        != _sha256(bundle.kernel_metadata_path)
        or not isinstance(source_provenance, dict)
        or not isinstance(source_provenance.get("path"), str)
        or not isinstance(source_provenance.get("sha256"), str)
    ):
        raise ValueError("Bundle provenance does not match the rendered metadata")
    validator = provenance.get("validator")
    if validator == "ordering":
        _validate_public_ordering_source(_notebook_source(source))
    elif validator == "baseline":
        _validate_public_baseline_source(_notebook_source(source))
    elif validator == "reports":
        _validate_reports_source(_notebook_source(source))
    elif validator == "report_llm":
        _validate_report_llm_source(_notebook_source(source))
    elif validator == "public_labels":
        _validate_public_label_source(_notebook_source(source))
    elif validator == "source_comparison":
        _validate_source_comparison_source(_notebook_source(source))
    if bundle.dataset is not None:
        if provenance.get("config_sha256") != _sha256(bundle.dataset.config_path):
            raise ValueError("Configuration digest does not match the rendered bundle")
        if _dataset_id(bundle.dataset) not in metadata.get("dataset_sources", []):
            raise ValueError("Attached Dataset is missing from kernel metadata")
        _validate_wheel_dataset(bundle.dataset)
    return gpu_value == "true"


def _validate_wheel_dataset(dataset: WheelDataset) -> None:
    metadata = json.loads(dataset.metadata_path.read_text())
    dataset_id = metadata.get("id")
    if not isinstance(dataset_id, str):
        raise TypeError("Dataset metadata ID is malformed")
    _require_resolved_kaggle_id(dataset_id)
    if metadata.get("isPublic") is not None:
        raise ValueError("Dataset metadata must not set public visibility")
    provenance = json.loads(dataset.provenance_path.read_text())
    record = provenance.get("wheel")
    if (
        provenance.get("dataset_id") != dataset_id
        or not isinstance(record, dict)
        or record.get("filename") != dataset.wheel_path.name
        or record.get("sha256") != _sha256(dataset.wheel_path)
    ):
        raise ValueError("Wheel provenance does not match the exact wheel bytes")
    _require_pure_python_wheel(dataset.wheel_path)


def _load_wheel_dataset(root: Path) -> WheelDataset:
    wheels = list(root.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected one wheel, found {len(wheels)}")
    config_candidates = [path for path in root.glob("*.toml") if path.is_file()]
    if len(config_candidates) != 1:
        raise ValueError("Expected one attached Kaggle configuration")
    return WheelDataset(
        root=root,
        wheel_path=wheels[0],
        provenance_path=root / "wheel-provenance.json",
        metadata_path=root / "dataset-metadata.json",
        config_path=config_candidates[0],
    )


def _dataset_id(dataset: WheelDataset | None) -> str | None:
    if dataset is None:
        return None
    value = json.loads(dataset.metadata_path.read_text()).get("id")
    if not isinstance(value, str):
        raise TypeError("Dataset metadata ID is malformed")
    return value


def _git_state(project_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def _dataset_metadata(
    dataset_id: str,
    *,
    description: str,
    title: str,
) -> dict[str, object]:
    return {
        "description": description,
        "id": dataset_id,
        "licenses": [{"name": "other"}],
        "title": title,
    }


def _wait_for_dataset_ready(
    dataset: WheelDataset,
    *,
    runner: CommandRunner,
    wait: Waiter,
    attempts: int = 60,
    interval_seconds: float = 2.0,
) -> None:
    dataset_id = _dataset_id(dataset)
    if dataset_id is None:
        raise TypeError("Dataset metadata ID is missing")
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
                    f"Kaggle Dataset entered terminal status: {last_status}"
                )
        if attempt < attempts - 1:
            wait(interval_seconds)
    raise TimeoutError(
        f"Kaggle Dataset was not ready after {attempts} checks: {last_status}"
    )


def _validate_public_ordering_source(source: str) -> None:
    if _PACKAGE_MARKER in source or "from rsna_knee" in source:
        raise ValueError("Public ordering notebook must not embed the private package")
    folded = source.casefold()
    for detail in _PUBLIC_ORDERING_FORBIDDEN:
        if detail.casefold() in folded:
            raise ValueError(
                f"Notebook contains forbidden public-notebook detail: {detail}"
            )


def _validate_public_baseline_source(source: str) -> None:
    required = (
        "--no-index",
        "--no-deps",
        "load_reference_baseline_summary",
        "reproduce_reference_baseline",
        "Patient-level validation",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "PatientID",
        "ImagePositionPatient",
        "teaching_slice_indices",
        "preprocess_series_result",
        "valid_slice_features",
        "series_features",
        "study_features",
        "binary_cross_entropy_with_logits",
        "Evidence for model selection",
        "Inadequate",
        "Reproducibility details",
    )
    if any(value not in source for value in required):
        raise ValueError("Public baseline notebook omits a reproducibility safeguard")
    if any(value in source.casefold() for value in _PUBLIC_BASELINE_FORBIDDEN):
        raise ValueError("Public baseline notebook contains a prohibited claim")


def _validate_reports_source(source: str) -> None:
    required = (
        "/kaggle/input/competitions/rsna-knee-abnormality-detection",
        "train.csv",
        '("StudyInstanceUID", "Report")',
        "read_report_csv",
        "audit_report_corpus",
        "report-audit.json",
        'os.environ["CUDA_VISIBLE_DEVICES"] = ""',
    )
    if any(value not in source for value in required):
        raise ValueError(
            "Report notebook omits a required privacy or runtime safeguard"
        )
    folded = source.casefold()
    if any(value in folded for value in ("pydicom", ".dcm", "train_series.csv")):
        raise ValueError("Report notebook must not read DICOM or train_series.csv")


def _validate_report_llm_source(source: str) -> None:
    required = (
        "/kaggle/input/competitions/rsna-knee-abnormality-detection",
        "/kaggle/input/kernels/plthon/rsna-knee-report-labels",
        "train.csv",
        'requested_columns=("StudyInstanceUID", "Report")',
        'os.environ["HF_HUB_OFFLINE"] = "1"',
        'os.environ["TRANSFORMERS_OFFLINE"] = "1"',
        "read_non_gold_study_uids",
        'model_asset["mount_paths"]',
        "resolve_private_kernel_root",
        "resolve_dataset_root",
        "verify_model_asset",
        "local_files_only=True",
        "do_sample=False",
        'salt="report-llm-smoke-v1"',
        "count=32",
        'salt="report-llm-stability-v1"',
        "count=64",
        'expected_study_count = 32 if STAGE == "smoke" else 4_407',
        "write_llm_artifacts",
    )
    if any(value not in source for value in required):
        raise ValueError("Report LLM notebook omits a frozen privacy or runtime guard")
    folded = source.casefold()
    forbidden = (
        "openai",
        "anthropic",
        "gemini",
        "openrouter",
        "inference api",
        "requests.",
        "leaderboard",
        "pydicom",
        ".dcm",
        "train_series.csv",
    )
    if any(value in folded for value in forbidden):
        raise ValueError("Report LLM notebook contains a prohibited external path")


def _validate_public_label_source(source: str) -> None:
    required = (
        "/kaggle/input/competitions/rsna-knee-abnormality-detection",
        "/kaggle/input/datasets/pilkwang/rsna-knee-llm-labels",
        "/kaggle/input/rsna-knee-llm-labels",
        "/kaggle/input/kernels/plthon/rsna-knee-report-labels",
        "resolve_private_kernel_root",
        "resolve_dataset_root",
        'usecols=["StudyInstanceUID", *target_order]',
        "summarize_coverage",
        "compare_label_sources",
        "public_safe_summary",
        "public-label-audit-manifest.json",
        "public-label-comparison-summary.json",
        "public-label-audit-summary.json",
        'os.environ["CUDA_VISIBLE_DEVICES"] = ""',
    )
    if any(value not in source for value in required):
        raise ValueError("Public label audit omits a privacy or runtime safeguard")
    folded = source.casefold()
    if any(value in folded for value in ("pydicom", ".dcm", "train_series.csv")):
        raise ValueError("Public label audit must not read DICOM or series metadata")


def _validate_source_comparison_source(source: str) -> None:
    required = (
        "/kaggle/input/competitions/rsna-knee-abnormality-detection",
        "/kaggle/input/datasets/pilkwang/rsna-knee-llm-labels",
        "/kaggle/input/kernels/plthon/rsna-knee-report-labels",
        "/kaggle/input/kernels/plthon/rsna-knee-report-llm-labels",
        "/kaggle/input/plthon/rsna-knee-report-llm-labels",
        "resolve_private_kernel_root",
        "resolve_dataset_root",
        "read_competition_cohorts",
        "verify_artifact_identity",
        "join_three_sources",
        "count_triple_combinations",
        "partition_strata",
        "compare_pairwise",
        "compute_deep_dives",
        "build_quota_feasibility",
        "build_public_safe_summary",
        "source-comparison-manifest.json",
        "source-comparison-tables.json",
        "source-comparison-summary.json",
        'os.environ["CUDA_VISIBLE_DEVICES"] = ""',
    )
    if any(value not in source for value in required):
        raise ValueError(
            "Source comparison notebook omits a privacy or runtime safeguard"
        )
    folded = source.casefold()
    if any(
        value in folded
        for value in (
            "pydicom",
            ".dcm",
            "train_series.csv",
            "torch.cuda",
            "openai",
            "anthropic",
        )
    ):
        raise ValueError(
            "Source comparison must stay CPU-only and offline without external APIs"
        )


def _notebook_source(rendered: str) -> str:
    notebook = json.loads(rendered)
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise TypeError("Rendered notebook cell list is malformed")
    return "\n".join(
        "".join(cell.get("source", [])) for cell in cells if isinstance(cell, dict)
    )


def _kernel_title_slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-")


def _package_archive(
    package_root: Path, *, additional_contracts: tuple[str, ...] = ()
) -> bytes:
    archive_buffer = BytesIO()
    with zipfile.ZipFile(
        archive_buffer, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for source_path in sorted(package_root.rglob("*.py")):
            relative_path = source_path.relative_to(package_root.parent).as_posix()
            _write_archive_entry(archive, relative_path, source_path)
        for contract_name in dict.fromkeys(("competition.json", *additional_contracts)):
            contract_path = package_root.parents[1] / "contracts" / contract_name
            if not contract_path.is_file():
                raise FileNotFoundError(f"Missing package contract: {contract_path}")
            _write_archive_entry(archive, f"rsna_knee/{contract_name}", contract_path)
    return archive_buffer.getvalue()


def _write_archive_entry(
    archive: zipfile.ZipFile, archive_name: str, source_path: Path
) -> None:
    info = zipfile.ZipInfo(archive_name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, source_path.read_bytes())


def _percent_source_to_notebook(source: str) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    cell_kind: str | None = None
    cell_lines: list[str] = []

    def append_cell() -> None:
        if cell_kind is None:
            return
        cell_id = f"cell-{len(cells) + 1:03d}"
        if cell_kind == "markdown":
            cells.append(
                {
                    "cell_type": "markdown",
                    "id": cell_id,
                    "metadata": {},
                    "source": [_markdown_line(line) for line in cell_lines],
                }
            )
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "id": cell_id,
                    "metadata": {},
                    "outputs": [],
                    "source": cell_lines.copy(),
                }
            )

    for line in source.splitlines(keepends=True):
        if line.startswith("# %%"):
            append_cell()
            cell_kind = "markdown" if "[markdown]" in line else "code"
            cell_lines = []
        elif cell_kind is not None:
            cell_lines.append(line)
    append_cell()
    if not cells:
        raise ValueError("Notebook source contains no Jupytext percent cells")
    return {
        "cells": cells,
        "metadata": {
            "jupytext": {
                "formats": "py:percent",
                "text_representation": {
                    "extension": ".py",
                    "format_name": "percent",
                    "format_version": "1.3",
                    "jupytext_version": "1.17",
                },
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _markdown_line(line: str) -> str:
    if line.startswith("# "):
        return line[2:]
    if line in {"#\n", "#"}:
        return "\n" if line.endswith("\n") else ""
    return line


def _require_resolved_kaggle_id(value: str) -> None:
    parts = value.split("/")
    if (
        any(character in value for character in "<>{}")
        or {part.upper() for part in parts} & _PLACEHOLDER_PARTS
        or _KAGGLE_ID_PATTERN.fullmatch(value) is None
    ):
        raise ValueError(f"Invalid or unresolved Kaggle ID: {value}")


def _weight_dataset_id_from_contract(raw: bytes) -> str:
    contract = json.loads(raw)
    if not isinstance(contract, dict) or not isinstance(
        contract.get("offline_kaggle_input"), dict
    ):
        raise TypeError("Initialization contract is malformed")
    dataset_id = contract["offline_kaggle_input"].get("dataset_id")
    if not isinstance(dataset_id, str):
        raise TypeError("Initialization Dataset ID is malformed")
    return dataset_id


def _require_pure_python_wheel(wheel_path: Path) -> None:
    with zipfile.ZipFile(wheel_path) as wheel:
        metadata_paths = [
            name for name in wheel.namelist() if name.endswith(".dist-info/WHEEL")
        ]
        if len(metadata_paths) != 1:
            raise ValueError("Wheel must contain exactly one WHEEL metadata file")
        metadata = wheel.read(metadata_paths[0]).decode()
    if "Root-Is-Purelib: true" not in metadata or "Tag: py3-none-any" not in metadata:
        raise ValueError("Kaggle package must be a pure Python py3-none-any wheel")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_command(command: list[str]) -> CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True)


def _run_status_command(command: list[str]) -> CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)
