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
_COMBINED_CHECKPOINT_DATASET_ID = "plthon/rsna-knee-combined-baseline-checkpoint"
_PLANE_GATING_DATASET_SOURCES = (
    "pilkwang/rsna-knee-llm-labels",
    "plthon/torchvision-resnet18-imagenet1k-v1",
    _COMBINED_CHECKPOINT_DATASET_ID,
)
_PLANE_GATING_KERNEL_SOURCES = ("plthon/rsna-knee-data-index",)
_DINOV2_TRAINING_DATASET_SOURCES = (
    "pilkwang/rsna-knee-llm-labels",
    "plthon/dinov2-vits14-lvd142m",
    _COMBINED_CHECKPOINT_DATASET_ID,
)
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
class WeightDataset:
    root: Path
    weight_path: Path
    contract_path: Path
    provenance_path: Path
    metadata_path: Path


type AttachedDataset = WheelDataset | WeightDataset


@dataclass(frozen=True, slots=True)
class KaggleBundle:
    root: Path
    kernel_root: Path
    notebook_path: Path
    kernel_metadata_path: Path
    provenance_path: Path
    dataset: AttachedDataset | None = None


def build_kaggle_bundle(
    project_root: Path,
    output_root: Path,
    spec: KaggleSpec,
    *,
    dataset: AttachedDataset | None = None,
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
    elif spec.validator == "combined_baseline":
        _validate_combined_baseline_source(source)
    elif spec.validator in {
        "plane_gating",
        "physical_crop",
        "adjacent_slice",
        "clipped_window",
        "slot_attention",
    }:
        _validate_controlled_training_source(
            source,
            physical_crop=spec.validator
            in {"physical_crop", "adjacent_slice", "clipped_window", "slot_attention"},
            adjacent_slice=spec.validator in {"adjacent_slice", "clipped_window"},
            clipped_window=spec.validator == "clipped_window",
            slot_attention=spec.validator == "slot_attention",
        )
    elif spec.validator in {"resolution_smoke", "resolution_full"}:
        _validate_resolution_source(source, smoke=spec.validator == "resolution_smoke")
    elif spec.validator in {"plane_gated_scoring", "physical_crop_scoring"}:
        _validate_plane_gated_scoring_source(
            source, physical_crop=spec.validator == "physical_crop_scoring"
        )
    elif spec.validator == "dinov2_training":
        _validate_dinov2_training_source(source)
    elif spec.validator == "dinov2_scoring":
        _validate_dinov2_scoring_source(source)
    elif spec.validator == "raptor_parity":
        _validate_raptor_parity_source(source)
    elif spec.validator == "raptor_scoring":
        _validate_raptor_scoring_source(source)
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
                    _sha256(dataset.config_path)
                    if isinstance(dataset, WheelDataset)
                    else None
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
    wheel_dataset_root = root / "wheel-dataset"
    weight_dataset_root = root / "weight-dataset"
    if wheel_dataset_root.is_dir() and weight_dataset_root.is_dir():
        raise ValueError("Kaggle bundle cannot contain two managed Datasets")
    dataset: AttachedDataset | None = None
    if wheel_dataset_root.is_dir():
        dataset = _load_wheel_dataset(wheel_dataset_root)
    elif weight_dataset_root.is_dir():
        dataset = _load_weight_dataset(weight_dataset_root)
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
        "validation-predictions.csv",
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


def build_dinov2_weight_dataset(
    project_root: Path,
    output_root: Path,
    *,
    weight_path: Path,
) -> WeightDataset:
    """Package the exact offline DINOv2 bytes as a private Kaggle Dataset."""
    contract_source = project_root / "contracts" / "dinov2-vits14-lvd142m.json"
    dataset_id, weight_record = _weight_dataset_contract(contract_source)
    if (
        not weight_path.is_file()
        or weight_path.stat().st_size != weight_record["size_bytes"]
        or _sha256(weight_path) != weight_record["sha256"]
    ):
        raise ValueError("DINOv2 weight artifact differs from the frozen contract")

    output_root.mkdir(parents=True, exist_ok=True)
    copied_weight_path = output_root / str(weight_record["filename"])
    copied_contract_path = output_root / contract_source.name
    shutil.copy2(weight_path, copied_weight_path)
    shutil.copy2(contract_source, copied_contract_path)
    metadata_path = output_root / "dataset-metadata.json"
    metadata_path.write_text(
        json.dumps(
            _dataset_metadata(
                dataset_id,
                description=(
                    "Private offline Meta DINOv2 ViT-S/14 LVD-142M initialization "
                    "artifact for the controlled RSNA Knee experiment."
                ),
                title="DINOv2 ViT-S/14 LVD-142M",
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    provenance_path = output_root / "weight-provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": dataset_id,
                "contract_sha256": _sha256(copied_contract_path),
                "weight": weight_record,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    dataset = WeightDataset(
        root=output_root,
        weight_path=copied_weight_path,
        contract_path=copied_contract_path,
        provenance_path=provenance_path,
        metadata_path=metadata_path,
    )
    _validate_weight_dataset(dataset)
    return dataset


def build_dinov2_resume_dataset(
    output_root: Path,
    *,
    checkpoint_path: Path,
    dataset_id: str = "plthon/rsna-knee-dinov2-resume",
) -> Path:
    """Package a validated DINOv2 intermediate checkpoint as a Kaggle resume dataset."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    output_root.mkdir(parents=True, exist_ok=True)
    copied_checkpoint = output_root / "checkpoint.pt"
    shutil.copy2(checkpoint_path, copied_checkpoint)
    metadata_path = output_root / "dataset-metadata.json"
    metadata_path.write_text(
        json.dumps(
            _dataset_metadata(
                dataset_id,
                description=(
                    "Intermediate DINOv2 ViT-S/14 checkpoint for training resumption."
                ),
                title="RSNA Knee DINOv2 Resume Checkpoint",
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return output_root


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


def build_combined_scoring_bundle(
    project_root: Path,
    output_root: Path,
    *,
    kernel_id: str = "plthon/rsna-knee-gold-public-scoring",
    gpu_approved: bool = False,
) -> KaggleBundle:
    """Build the private GPU inference-only combined baseline scoring bundle."""
    contract_names = ("competition.json", "gold-public-baseline.json")
    for name in contract_names:
        contract_path = project_root / "contracts" / name
        if not contract_path.is_file():
            raise FileNotFoundError(f"Combined scoring requires contracts/{name}")
    source_path = project_root / "notebooks" / "score_combined_baseline.py"
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing notebook source: {source_path}")

    git_commit, git_dirty = _git_state(project_root)
    bundle = build_kaggle_bundle(
        project_root,
        output_root,
        KaggleSpec(
            source_name=source_path.name,
            notebook_name="score_combined_baseline.ipynb",
            kernel_id=kernel_id,
            title="RSNA Knee Gold + Public Scoring",
            competition_sources=("rsna-knee-abnormality-detection",),
            dataset_sources=(_COMBINED_CHECKPOINT_DATASET_ID,),
            gpu=True,
            machine_shape=_GPU_MACHINE_SHAPE,
            replacements={
                "__RSNA_KNEE_APPROVED_GPU__": json.dumps(gpu_approved),
                "__RSNA_KNEE_GIT_COMMIT__": git_commit,
                "__RSNA_KNEE_GIT_DIRTY__": json.dumps(git_dirty),
                "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__": hashlib.sha256(
                    source_path.read_bytes()
                ).hexdigest(),
                "__RSNA_KNEE_RENDERED_NOTEBOOK_SHA256__": "0" * 64,
            },
            embed_package=True,
            package_contracts=contract_names,
            validator="combined_baseline",
        ),
        gpu_approved=gpu_approved,
    )
    return _record_rendered_notebook_hash(bundle, label="Combined scoring")


def build_combined_baseline_bundle(
    project_root: Path,
    output_root: Path,
    *,
    kernel_id: str = "plthon/rsna-knee-gold-public-scoring",
    gpu_approved: bool = False,
) -> KaggleBundle:
    """Backward-compatible name for the inference-only combined bundle."""
    return build_combined_scoring_bundle(
        project_root,
        output_root,
        kernel_id=kernel_id,
        gpu_approved=gpu_approved,
    )


def build_plane_gating_bundle(
    project_root: Path,
    output_root: Path,
    *,
    kernel_id: str | None = None,
    gpu_approved: bool = False,
    slot_attention: bool = False,
    physical_crop: bool = False,
    adjacent_slice: bool = False,
    clipped_window: bool = False,
    resolution_stage: str | None = None,
    resource_smoke_path: Path | None = None,
) -> KaggleBundle:
    """Build a guarded controlled-training and paired-validation bundle."""
    is_resolution = resolution_stage is not None
    if is_resolution and resolution_stage not in {"smoke", "full"}:
        raise ValueError("Resolution stage must be smoke or full")
    if (
        sum(
            (
                physical_crop,
                adjacent_slice,
                clipped_window,
                slot_attention,
                is_resolution,
            )
        )
        > 1
    ):
        raise ValueError("Choose one controlled representation experiment")
    name = (
        "resolution-336"
        if is_resolution
        else "slot-attention"
        if slot_attention
        else "clipped-window"
        if clipped_window
        else "adjacent-slice"
        if adjacent_slice
        else "physical-crop"
        if physical_crop
        else "plane-gating"
    )
    source_name = (
        "train_resolution" if is_resolution else f"train_{name.replace('-', '_')}"
    )
    config_path = project_root / "configs" / f"{name}.toml"
    source_path = project_root / "notebooks" / f"{source_name}.py"
    contract_names = (
        "competition.json",
        "data-index.json",
        "public-report-label-dataset-v1.json",
        "resnet18-imagenet1k-v1.json",
    )
    config_bytes = config_path.read_bytes()
    git_commit, git_dirty = _git_state(project_root)
    guarded = name != "plane-gating"
    if guarded and gpu_approved and git_dirty:
        raise ValueError(
            "Controlled experiment GPU execution requires a clean committed revision"
        )
    replacements = {
        "__RSNA_KNEE_APPROVED_GPU__": json.dumps(gpu_approved),
        "__RSNA_KNEE_CONFIG__": base64.b64encode(config_bytes).decode("ascii"),
        "__RSNA_KNEE_GIT_COMMIT__": git_commit,
        "__RSNA_KNEE_GIT_DIRTY__": json.dumps(git_dirty),
        "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__": _sha256(source_path),
        "__RSNA_KNEE_RENDERED_NOTEBOOK_SHA256__": "0" * 64,
    }
    if is_resolution:
        from dataclasses import replace

        from .combined_supervision import (
            _expected_resolution_configuration,
            load_combined_baseline_config,
        )
        from .resolution import require_resolution_smoke

        if resolution_stage == "full" and resource_smoke_path is None:
            raise ValueError("Full bundle requires successful resource smoke evidence")
        if resolution_stage == "smoke" and resource_smoke_path is not None:
            raise ValueError(
                "Smoke starts at batch 16 and does not accept prior evidence"
            )
        replacements["__RSNA_KNEE_RESOLUTION_STAGE__"] = str(resolution_stage)
        replacements["__RSNA_KNEE_RESOURCE_SMOKE__"] = ""
        if resource_smoke_path is not None:
            smoke_bytes = resource_smoke_path.read_bytes()
            batch_size = json.loads(smoke_bytes)["configuration"]["training"][
                "batch_size"
            ]
            _expected_resolution_configuration(batch_size)
            config = replace(
                load_combined_baseline_config(config_path),
                batch_size=batch_size,
                gradient_accumulation_steps=16 // batch_size,
            )
            require_resolution_smoke(
                resource_smoke_path,
                config,
                {"git_commit": git_commit, "git_dirty": git_dirty},
            )
            replacements["__RSNA_KNEE_RESOURCE_SMOKE__"] = base64.b64encode(
                smoke_bytes
            ).decode("ascii")
    elif resource_smoke_path is not None:
        raise ValueError("Resource smoke applies only to the resolution experiment")
    bundle = build_kaggle_bundle(
        project_root,
        output_root,
        KaggleSpec(
            source_name=source_path.name,
            notebook_name=f"{source_name}.ipynb",
            kernel_id=kernel_id
            or (
                f"plthon/rsna-knee-resolution-336-{resolution_stage}"
                if is_resolution
                else "plthon/rsna-knee-plane-gating"
            ),
            title=f"RSNA Knee Resolution 336 {str(resolution_stage).title()}"
            if is_resolution
            else f"RSNA Knee {name.replace('-', ' ').title()}",
            competition_sources=("rsna-knee-abnormality-detection",),
            dataset_sources=_PLANE_GATING_DATASET_SOURCES,
            kernel_sources=(
                (*_PLANE_GATING_KERNEL_SOURCES, "plthon/rsna-knee-physical-crop")
                if adjacent_slice or clipped_window or slot_attention or is_resolution
                else _PLANE_GATING_KERNEL_SOURCES
            ),
            gpu=gpu_approved if guarded else True,
            machine_shape=_GPU_MACHINE_SHAPE if gpu_approved or not guarded else "",
            replacements=replacements,
            embed_package=True,
            package_contracts=contract_names,
            validator=f"resolution_{resolution_stage}"
            if is_resolution
            else name.replace("-", "_"),
        ),
        gpu_approved=gpu_approved,
    )
    return _record_rendered_notebook_hash(bundle, label=name.replace("-", " ").title())


def _validate_resolution_source(source: str, *, smoke: bool) -> None:
    required = (
        f'STAGE = "{"smoke" if smoke else "full"}"',
        'if STAGE == "smoke":',
        'elif STAGE == "full":',
        "run_resolution_smoke(",
        "run_combined_baseline(",
        "approved_gpu=GPU_APPROVED",
        "source_provenance=source_provenance",
        '"git_dirty"',
        '"config_sha256"',
        "load_combined_baseline_config(CONFIG_PATH)",
        "resource_smoke_path=SMOKE_PATH",
        "run_test_inference=False",
        "verify_physical_crop_artifact(",
        "resolution_336=True",
    )
    if any(text not in source for text in required) or any(
        text in source
        for text in (
            "run_combined_scoring(",
            "competitions submit",
            "run_test_inference=True",
        )
    ):
        raise ValueError("Resolution notebook omits a required execution safeguard")


def build_plane_gated_scoring_bundle(
    project_root: Path,
    output_root: Path,
    *,
    checkpoint_dataset_id: str,
    checkpoint_sha256: str,
    validation_kept: bool = False,
    kernel_id: str = "plthon/rsna-knee-plane-gated-scoring",
    gpu_approved: bool = False,
) -> KaggleBundle:
    """Build hidden inference only for a validation-retained #28 checkpoint."""
    if not validation_kept:
        raise PermissionError("Plane-gated scoring requires a recorded keep decision")
    _require_resolved_kaggle_id(checkpoint_dataset_id)
    if re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256) is None:
        raise ValueError("Plane-gated checkpoint SHA-256 is malformed")
    source_path = project_root / "notebooks" / "score_plane_gating.py"
    git_commit, git_dirty = _git_state(project_root)
    bundle = build_kaggle_bundle(
        project_root,
        output_root,
        KaggleSpec(
            source_name=source_path.name,
            notebook_name="score_plane_gating.ipynb",
            kernel_id=kernel_id,
            title="RSNA Knee Plane-Gated Scoring",
            competition_sources=("rsna-knee-abnormality-detection",),
            dataset_sources=(checkpoint_dataset_id,),
            gpu=True,
            machine_shape=_GPU_MACHINE_SHAPE,
            replacements={
                "__RSNA_KNEE_APPROVED_GPU__": json.dumps(gpu_approved),
                "__RSNA_KNEE_CHECKPOINT_SHA256__": checkpoint_sha256,
                "__RSNA_KNEE_GIT_COMMIT__": git_commit,
                "__RSNA_KNEE_GIT_DIRTY__": json.dumps(git_dirty),
                "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__": hashlib.sha256(
                    source_path.read_bytes()
                ).hexdigest(),
                "__RSNA_KNEE_RENDERED_NOTEBOOK_SHA256__": "0" * 64,
                "__RSNA_KNEE_KERNEL_ID__": kernel_id,
            },
            embed_package=True,
            package_contracts=("competition.json",),
            validator="plane_gated_scoring",
        ),
        gpu_approved=gpu_approved,
    )
    return _record_rendered_notebook_hash(bundle, label="Plane-gated scoring")


def build_physical_crop_scoring_bundle(
    project_root: Path,
    output_root: Path,
    *,
    kernel_id: str = "plthon/rsna-knee-physical-crop-scoring",
    gpu_approved: bool = False,
) -> KaggleBundle:
    """Score with the exact #31 checkpoint retained by paired validation."""
    source_path = project_root / "notebooks" / "score_physical_crop.py"
    git_commit, git_dirty = _git_state(project_root)
    if gpu_approved and git_dirty:
        raise ValueError("Physical crop scoring requires a clean committed revision")
    bundle = build_kaggle_bundle(
        project_root,
        output_root,
        KaggleSpec(
            source_name=source_path.name,
            notebook_name="score_physical_crop.ipynb",
            kernel_id=kernel_id,
            title="RSNA Knee Physical Crop Scoring",
            competition_sources=("rsna-knee-abnormality-detection",),
            kernel_sources=("plthon/rsna-knee-physical-crop",),
            gpu=gpu_approved,
            machine_shape=_GPU_MACHINE_SHAPE if gpu_approved else "",
            replacements={
                "__RSNA_KNEE_APPROVED_GPU__": json.dumps(gpu_approved),
                "__RSNA_KNEE_CHECKPOINT_SHA256__": "55e9da7854d284a91c2569e25c799281d09301bae61dc5adcecebbd17209f013",
                "__RSNA_KNEE_GIT_COMMIT__": git_commit,
                "__RSNA_KNEE_GIT_DIRTY__": json.dumps(git_dirty),
                "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__": hashlib.sha256(
                    source_path.read_bytes()
                ).hexdigest(),
                "__RSNA_KNEE_RENDERED_NOTEBOOK_SHA256__": "0" * 64,
                "__RSNA_KNEE_KERNEL_ID__": kernel_id,
            },
            embed_package=True,
            package_contracts=("competition.json",),
            validator="physical_crop_scoring",
        ),
        gpu_approved=gpu_approved,
    )
    return _record_rendered_notebook_hash(bundle, label="Physical crop scoring")


def build_dinov2_training_bundle(
    project_root: Path,
    output_root: Path,
    *,
    weight_path: Path,
    kernel_id: str = "plthon/rsna-knee-dinov2-vit-s-14",
    gpu_approved: bool = False,
    resume: bool = False,
    resume_dataset_id: str | None = None,
) -> KaggleBundle:
    """Build the clean-revision #30 training and paired-validation bundle."""
    config_path = project_root / "configs" / "dinov2.toml"
    source_path = project_root / "notebooks" / "train_dinov2.py"
    contract_names = (
        "competition.json",
        "data-index.json",
        "public-report-label-dataset-v1.json",
        "dinov2-vits14-lvd142m.json",
    )
    config_bytes = config_path.read_bytes()
    git_commit, git_dirty = _git_state(project_root)
    if git_dirty:
        raise ValueError("DINOv2 training requires a clean committed Git revision")
    weight_dataset = build_dinov2_weight_dataset(
        project_root,
        output_root / "weight-dataset",
        weight_path=weight_path,
    )
    kernel_sources = (
        (*_PLANE_GATING_KERNEL_SOURCES, kernel_id)
        if resume and not resume_dataset_id
        else _PLANE_GATING_KERNEL_SOURCES
    )
    dataset_sources = (
        (*_DINOV2_TRAINING_DATASET_SOURCES, resume_dataset_id)
        if resume_dataset_id
        else _DINOV2_TRAINING_DATASET_SOURCES
    )
    bundle = build_kaggle_bundle(
        project_root,
        output_root,
        KaggleSpec(
            source_name=source_path.name,
            notebook_name="train_dinov2.ipynb",
            kernel_id=kernel_id,
            title="RSNA Knee DINOv2 ViT-S/14",
            competition_sources=("rsna-knee-abnormality-detection",),
            dataset_sources=dataset_sources,
            kernel_sources=kernel_sources,
            gpu=True,
            machine_shape=_GPU_MACHINE_SHAPE,
            replacements={
                "__RSNA_KNEE_APPROVED_GPU__": json.dumps(gpu_approved),
                "__RSNA_KNEE_CONFIG__": base64.b64encode(config_bytes).decode("ascii"),
                "__RSNA_KNEE_GIT_COMMIT__": git_commit,
                "__RSNA_KNEE_GIT_DIRTY__": json.dumps(git_dirty),
                "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__": hashlib.sha256(
                    source_path.read_bytes()
                ).hexdigest(),
                "__RSNA_KNEE_RENDERED_NOTEBOOK_SHA256__": "0" * 64,
            },
            embed_package=True,
            package_contracts=contract_names,
            validator="dinov2_training",
        ),
        dataset=weight_dataset,
        gpu_approved=gpu_approved,
    )
    return _record_rendered_notebook_hash(bundle, label="DINOv2 training")


def build_dinov2_scoring_bundle(
    project_root: Path,
    output_root: Path,
    *,
    checkpoint_dataset_id: str,
    checkpoint_sha256: str,
    validation_kept: bool = False,
    kernel_id: str = "plthon/rsna-knee-dinov2-scoring",
    gpu_approved: bool = False,
) -> KaggleBundle:
    """Build inference-only scoring for a validation-retained #30 checkpoint."""
    if not validation_kept:
        raise PermissionError("DINOv2 scoring requires a recorded keep decision")
    _require_resolved_kaggle_id(checkpoint_dataset_id)
    if re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256) is None:
        raise ValueError("DINOv2 checkpoint SHA-256 is malformed")
    source_path = project_root / "notebooks" / "score_dinov2.py"
    git_commit, git_dirty = _git_state(project_root)
    if git_dirty:
        raise ValueError("DINOv2 scoring requires a clean committed Git revision")
    bundle = build_kaggle_bundle(
        project_root,
        output_root,
        KaggleSpec(
            source_name=source_path.name,
            notebook_name="score_dinov2.ipynb",
            kernel_id=kernel_id,
            title="RSNA Knee DINOv2 Scoring",
            competition_sources=("rsna-knee-abnormality-detection",),
            dataset_sources=(checkpoint_dataset_id,),
            gpu=True,
            machine_shape=_GPU_MACHINE_SHAPE,
            replacements={
                "__RSNA_KNEE_APPROVED_GPU__": json.dumps(gpu_approved),
                "__RSNA_KNEE_CHECKPOINT_SHA256__": checkpoint_sha256,
                "__RSNA_KNEE_GIT_COMMIT__": git_commit,
                "__RSNA_KNEE_GIT_DIRTY__": json.dumps(git_dirty),
                "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__": hashlib.sha256(
                    source_path.read_bytes()
                ).hexdigest(),
                "__RSNA_KNEE_RENDERED_NOTEBOOK_SHA256__": "0" * 64,
                "__RSNA_KNEE_KERNEL_ID__": kernel_id,
            },
            embed_package=True,
            package_contracts=(
                "competition.json",
                "dinov2-vits14-lvd142m.json",
            ),
            validator="dinov2_scoring",
        ),
        gpu_approved=gpu_approved,
    )
    return _record_rendered_notebook_hash(bundle, label="DINOv2 scoring")


def build_raptor_parity_bundle(
    project_root: Path,
    output_root: Path,
    *,
    kernel_id: str = "plthon/rsna-knee-raptor-reference-parity",
    gpu_approved: bool = False,
) -> KaggleBundle:
    """Build approved CPU/T4 parity and organizer-gold diagnostics for Raptor."""
    contract_names = (
        "competition.json",
        "data-index.json",
        "gold-supervision.json",
        "raptor-reference-v1.json",
    )
    for name in contract_names:
        if not (project_root / "contracts" / name).is_file():
            raise FileNotFoundError(f"Raptor parity requires contracts/{name}")
    source_path = project_root / "notebooks" / "raptor_reference_parity.py"
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing notebook source: {source_path}")
    git_commit, git_dirty = _git_state(project_root)
    if git_dirty:
        raise ValueError("Raptor parity requires a clean committed Git revision")
    bundle = build_kaggle_bundle(
        project_root,
        output_root,
        KaggleSpec(
            source_name=source_path.name,
            notebook_name="raptor_reference_parity.ipynb",
            kernel_id=kernel_id,
            title="RSNA Knee Raptor Reference Parity",
            competition_sources=("rsna-knee-abnormality-detection",),
            dataset_sources=("dreaddevelopment/raptor-knee-widedense",),
            kernel_sources=("plthon/rsna-knee-data-index",),
            gpu=True,
            machine_shape=_GPU_MACHINE_SHAPE,
            replacements={
                "__RSNA_KNEE_APPROVED_GPU__": json.dumps(gpu_approved),
                "__RSNA_KNEE_GIT_COMMIT__": git_commit,
                "__RSNA_KNEE_GIT_DIRTY__": json.dumps(git_dirty),
                "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__": hashlib.sha256(
                    source_path.read_bytes()
                ).hexdigest(),
                "__RSNA_KNEE_RENDERED_NOTEBOOK_SHA256__": "0" * 64,
                "__RSNA_KNEE_KERNEL_ID__": kernel_id,
            },
            embed_package=True,
            package_contracts=contract_names,
            validator="raptor_parity",
        ),
        gpu_approved=gpu_approved,
    )
    return _record_rendered_notebook_hash(bundle, label="Raptor parity")


def build_raptor_scoring_bundle(
    project_root: Path,
    output_root: Path,
    *,
    kernel_id: str = "plthon/rsna-knee-raptor-reference-scoring",
    gpu_approved: bool = False,
) -> KaggleBundle:
    """Build approved hidden inference for the exact public Raptor checkpoint."""
    contract_names = ("competition.json", "raptor-reference-v1.json")
    for name in contract_names:
        if not (project_root / "contracts" / name).is_file():
            raise FileNotFoundError(f"Raptor scoring requires contracts/{name}")
    source_path = project_root / "notebooks" / "score_raptor_reference.py"
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing notebook source: {source_path}")
    git_commit, git_dirty = _git_state(project_root)
    if git_dirty:
        raise ValueError("Raptor scoring requires a clean committed Git revision")
    bundle = build_kaggle_bundle(
        project_root,
        output_root,
        KaggleSpec(
            source_name=source_path.name,
            notebook_name="score_raptor_reference.ipynb",
            kernel_id=kernel_id,
            title="RSNA Knee Raptor Reference Scoring",
            competition_sources=("rsna-knee-abnormality-detection",),
            dataset_sources=("dreaddevelopment/raptor-knee-widedense",),
            gpu=True,
            machine_shape=_GPU_MACHINE_SHAPE,
            replacements={
                "__RSNA_KNEE_APPROVED_GPU__": json.dumps(gpu_approved),
                "__RSNA_KNEE_GIT_COMMIT__": git_commit,
                "__RSNA_KNEE_GIT_DIRTY__": json.dumps(git_dirty),
                "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__": hashlib.sha256(
                    source_path.read_bytes()
                ).hexdigest(),
                "__RSNA_KNEE_RENDERED_NOTEBOOK_SHA256__": "0" * 64,
                "__RSNA_KNEE_KERNEL_ID__": kernel_id,
            },
            embed_package=True,
            package_contracts=contract_names,
            validator="raptor_scoring",
        ),
        gpu_approved=gpu_approved,
    )
    return _record_rendered_notebook_hash(bundle, label="Raptor scoring")


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
    elif validator == "combined_baseline":
        _validate_combined_baseline_source(_notebook_source(source))
        if metadata.get("dataset_sources") != [_COMBINED_CHECKPOINT_DATASET_ID]:
            raise ValueError(
                "Combined scoring must attach the frozen checkpoint Dataset"
            )
        if metadata.get("kernel_sources") != []:
            raise ValueError("Combined scoring must not attach the frozen test index")
    elif validator in {
        "plane_gating",
        "physical_crop",
        "adjacent_slice",
        "clipped_window",
        "slot_attention",
    }:
        _validate_controlled_training_source(
            _notebook_source(source),
            physical_crop=validator
            in {"physical_crop", "adjacent_slice", "clipped_window", "slot_attention"},
            adjacent_slice=validator in {"adjacent_slice", "clipped_window"},
            clipped_window=validator == "clipped_window",
            slot_attention=validator == "slot_attention",
        )
        if tuple(metadata.get("dataset_sources", ())) != _PLANE_GATING_DATASET_SOURCES:
            raise ValueError("Plane gating attached unexpected Dataset inputs")
        expected_kernels = (
            (*_PLANE_GATING_KERNEL_SOURCES, "plthon/rsna-knee-physical-crop")
            if validator in {"adjacent_slice", "clipped_window", "slot_attention"}
            else _PLANE_GATING_KERNEL_SOURCES
        )
        if tuple(metadata.get("kernel_sources", ())) != expected_kernels:
            raise ValueError("Plane gating must attach the frozen index kernel output")
    elif validator in {"resolution_smoke", "resolution_full"}:
        _validate_resolution_source(
            _notebook_source(source), smoke=validator == "resolution_smoke"
        )
        if tuple(
            metadata.get("dataset_sources", ())
        ) != _PLANE_GATING_DATASET_SOURCES or tuple(
            metadata.get("kernel_sources", ())
        ) != (*_PLANE_GATING_KERNEL_SOURCES, "plthon/rsna-knee-physical-crop"):
            raise ValueError("Resolution bundles require the frozen #31 inputs")
    elif validator == "physical_crop_scoring":
        _validate_plane_gated_scoring_source(
            _notebook_source(source), physical_crop=True
        )
        if (
            metadata.get("kernel_sources") != ["plthon/rsna-knee-physical-crop"]
            or metadata.get("dataset_sources") != []
        ):
            raise ValueError(
                "Physical crop scoring requires the retained training output only"
            )
    elif validator == "plane_gated_scoring":
        _validate_plane_gated_scoring_source(_notebook_source(source))
        dataset_sources = metadata.get("dataset_sources")
        if not isinstance(dataset_sources, list) or len(dataset_sources) != 1:
            raise ValueError(
                "Plane-gated scoring requires one retained checkpoint input"
            )
    elif validator == "dinov2_training":
        _validate_dinov2_training_source(_notebook_source(source))
        actual_datasets = tuple(metadata.get("dataset_sources", ()))
        is_standard = actual_datasets == _DINOV2_TRAINING_DATASET_SOURCES
        is_resumed = (
            len(actual_datasets) == len(_DINOV2_TRAINING_DATASET_SOURCES) + 1
            and actual_datasets[:-1] == _DINOV2_TRAINING_DATASET_SOURCES
            and bool(_KAGGLE_ID_PATTERN.fullmatch(actual_datasets[-1]))
        )
        if not (is_standard or is_resumed):
            raise ValueError("DINOv2 training attached unexpected Dataset inputs")
        if tuple(metadata.get("kernel_sources", ())) not in {
            _PLANE_GATING_KERNEL_SOURCES,
            (*_PLANE_GATING_KERNEL_SOURCES, kernel_id),
        }:
            raise ValueError("DINOv2 training must attach the frozen index output")
    elif validator == "dinov2_scoring":
        _validate_dinov2_scoring_source(_notebook_source(source))
        dataset_sources = metadata.get("dataset_sources")
        if not isinstance(dataset_sources, list) or len(dataset_sources) != 1:
            raise ValueError("DINOv2 scoring requires one retained checkpoint input")
    elif validator in {"raptor_parity", "raptor_scoring"}:
        notebook_source = _notebook_source(source)
        if validator == "raptor_parity":
            _validate_raptor_parity_source(notebook_source)
            raptor_expected_kernels: list[str] = ["plthon/rsna-knee-data-index"]
        else:
            _validate_raptor_scoring_source(notebook_source)
            raptor_expected_kernels = []
        if (
            metadata.get("competition_sources") != ["rsna-knee-abnormality-detection"]
            or metadata.get("dataset_sources")
            != ["dreaddevelopment/raptor-knee-widedense"]
            or metadata.get("kernel_sources") != raptor_expected_kernels
            or metadata.get("model_sources") != []
        ):
            raise ValueError("Raptor bundle attached unexpected inputs")
    elif validator == "reports":
        _validate_reports_source(_notebook_source(source))
    elif validator == "report_llm":
        _validate_report_llm_source(_notebook_source(source))
    elif validator == "public_labels":
        _validate_public_label_source(_notebook_source(source))
    elif validator == "source_comparison":
        _validate_source_comparison_source(_notebook_source(source))
    if bundle.dataset is not None:
        if _dataset_id(bundle.dataset) not in metadata.get("dataset_sources", []):
            raise ValueError("Attached Dataset is missing from kernel metadata")
        if isinstance(bundle.dataset, WheelDataset):
            if provenance.get("config_sha256") != _sha256(bundle.dataset.config_path):
                raise ValueError(
                    "Configuration digest does not match the rendered bundle"
                )
            _validate_wheel_dataset(bundle.dataset)
        else:
            _validate_weight_dataset(bundle.dataset)
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


def _validate_weight_dataset(dataset: WeightDataset) -> None:
    metadata = json.loads(dataset.metadata_path.read_text())
    dataset_id, weight_record = _weight_dataset_contract(dataset.contract_path)
    provenance = json.loads(dataset.provenance_path.read_text())
    if (
        metadata.get("id") != dataset_id
        or metadata.get("isPublic") is not None
        or provenance
        != {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "contract_sha256": _sha256(dataset.contract_path),
            "weight": weight_record,
        }
        or dataset.weight_path.name != weight_record["filename"]
        or not dataset.weight_path.is_file()
        or dataset.weight_path.stat().st_size != weight_record["size_bytes"]
        or _sha256(dataset.weight_path) != weight_record["sha256"]
    ):
        raise ValueError("DINOv2 weight Dataset differs from the frozen contract")


def _weight_dataset_contract(path: Path) -> tuple[str, dict[str, object]]:
    contract = json.loads(path.read_text())
    artifact = contract.get("artifact")
    offline_input = contract.get("offline_kaggle_input")
    if not isinstance(artifact, dict) or not isinstance(offline_input, dict):
        raise TypeError("DINOv2 weight contract is malformed")
    filename = artifact.get("filename")
    sha256 = artifact.get("sha256")
    size_bytes = artifact.get("size_bytes")
    dataset_id = offline_input.get("dataset_id")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or not isinstance(sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
        or type(size_bytes) is not int
        or size_bytes <= 0
        or not isinstance(dataset_id, str)
    ):
        raise TypeError("DINOv2 weight contract fields are malformed")
    _require_resolved_kaggle_id(dataset_id)
    return dataset_id, {
        "filename": filename,
        "sha256": sha256,
        "size_bytes": size_bytes,
    }


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


def _load_weight_dataset(root: Path) -> WeightDataset:
    contract_path = root / "dinov2-vits14-lvd142m.json"
    _, weight_record = _weight_dataset_contract(contract_path)
    return WeightDataset(
        root=root,
        weight_path=root / str(weight_record["filename"]),
        contract_path=contract_path,
        provenance_path=root / "weight-provenance.json",
        metadata_path=root / "dataset-metadata.json",
    )


def _dataset_id(dataset: AttachedDataset | None) -> str | None:
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
    dataset: AttachedDataset,
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


def _validate_combined_baseline_source(source: str) -> None:
    required = (
        "COMBINED_BASELINE_CHECKPOINT_SHA256",
        "discover_combined_checkpoint",
        "load_verified_combined_checkpoint",
        "run_combined_scoring",
        "discover_submission_inputs",
        "build_competition_index",
        'splits=("test",)',
        "verify_index_artifact",
        "load_indexed_studies",
        "write_submission",
        "validate_submission",
        "inference_only",
        "submission.csv",
    )
    if any(value not in source for value in required):
        raise ValueError("Combined scoring notebook omits an inference safeguard")
    forbidden = (
        "run_combined_baseline",
        "derive_combined_cohort",
        "derive_patient_split",
        "train_epoch",
        "predict_partition",
        "rsna-knee-data-index",
        "kernel_sources",
    )
    folded = source.casefold()
    if any(value.casefold() in folded for value in forbidden):
        raise ValueError(
            "Combined scoring notebook contains training or frozen-index work"
        )


def _validate_controlled_training_source(
    source: str,
    *,
    slot_attention: bool = False,
    physical_crop: bool = False,
    adjacent_slice: bool = False,
    clipped_window: bool = False,
) -> None:
    verifier = (
        "verify_slot_attention_artifact"
        if slot_attention
        else "verify_adjacent_slice_artifact"
        if adjacent_slice
        else "verify_physical_crop_artifact"
        if physical_crop
        else "verify_plane_gating_artifact"
    )
    if clipped_window and (
        'config.preprocessing.sampling_layout != "central_80_three_anchor"'
        not in source
        or "clipped_window=True" not in source
    ):
        raise ValueError("Clipped-window notebook omits layout safeguard")
    aggregation = (
        "target_slot_attention"
        if slot_attention
        else "mean"
        if physical_crop
        else "target_plane_gate"
    )
    if (
        physical_crop
        and 'config.preprocessing.spatial_mode != "physical_crop"' not in source
    ):
        raise ValueError("Physical crop notebook omits spatial safeguard")
    if slot_attention and (
        'config.preprocessing.input_representation != "same_slice_rgb"' not in source
        or "PHYSICAL_CROP_CHECKPOINT_SHA256" not in source
    ):
        raise ValueError("Slot attention notebook omits representation safeguard")
    if adjacent_slice and (
        'config.preprocessing.input_representation != "adjacent_slice_2_5d"'
        not in source
        or "PHYSICAL_CROP_CHECKPOINT_SHA256" not in source
    ):
        raise ValueError("Adjacent-slice notebook omits representation safeguard")
    required = (
        "discover_combined_checkpoint",
        "load_combined_baseline_config",
        "run_combined_baseline",
        verifier,
        f'config.aggregation != "{aggregation}"',
        "control_checkpoint_path=control_checkpoint",
        "run_test_inference=False",
        '"competition_submission": False',
        "Paired macro AUC delta",
        "Hidden-test inference: not run",
    )
    if any(value not in source for value in required):
        raise ValueError(
            "Plane-gating notebook omits a controlled-experiment safeguard"
        )
    forbidden = (
        "run_combined_scoring",
        "write_submission",
        "validate_submission",
        "kaggle competitions submit",
    )
    if any(value in source for value in forbidden):
        raise ValueError("Plane-gating training notebook contains hidden scoring work")


def _validate_plane_gated_scoring_source(
    source: str, *, physical_crop: bool = False
) -> None:
    verifier = (
        "load_verified_physical_crop_checkpoint"
        if physical_crop
        else "load_verified_plane_gated_checkpoint"
    )
    aggregation = "mean" if physical_crop else "target_plane_gate"
    if physical_crop and "physical_crop=True" not in source:
        raise ValueError("Physical crop scoring omits the crop safeguard")
    required = (
        "EXPECTED_CHECKPOINT_SHA256",
        verifier,
        "run_combined_scoring",
        f'aggregation="{aggregation}"',
        "expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256",
        "build_competition_index",
        "verify_index_artifact",
        "load_indexed_studies",
        "write_submission",
        "validate_submission",
        '"competition_submission": False',
        "Competition submission: not run",
    )
    if any(value not in source for value in required):
        raise ValueError("Plane-gated scoring notebook omits an inference safeguard")
    forbidden = ("run_combined_baseline", "train_epoch", "derive_combined_cohort")
    if any(value in source for value in forbidden):
        raise ValueError("Plane-gated scoring notebook contains training work")


def _validate_dinov2_training_source(source: str) -> None:
    required = (
        'os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"',
        'config.architecture != "dinov2_vits14"',
        "discover_combined_checkpoint",
        "run_combined_baseline",
        "verify_dinov2_artifact",
        "control_checkpoint_path=control_checkpoint",
        "run_test_inference=False",
        '"competition_submission": False',
        "Hidden-test inference: not run",
        "Competition submission: not run",
    )
    if any(value not in source for value in required):
        raise ValueError("DINOv2 training notebook omits an experiment safeguard")
    forbidden = (
        "run_combined_scoring",
        "write_submission",
        "validate_submission",
        "kaggle competitions submit",
    )
    if any(value in source for value in forbidden):
        raise ValueError("DINOv2 training notebook contains hidden scoring work")


def _validate_dinov2_scoring_source(source: str) -> None:
    required = (
        "EXPECTED_CHECKPOINT_SHA256",
        "load_verified_dinov2_checkpoint",
        "run_combined_scoring",
        'backbone="dinov2_vits14"',
        "expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256",
        "build_competition_index",
        "verify_index_artifact",
        "load_indexed_studies",
        "write_submission",
        "validate_submission",
        '"competition_submission": False',
        "Competition submission: not run",
    )
    if any(value not in source for value in required):
        raise ValueError("DINOv2 scoring notebook omits an inference safeguard")
    forbidden = ("run_combined_baseline", "train_epoch", "derive_combined_cohort")
    if any(value in source for value in forbidden):
        raise ValueError("DINOv2 scoring notebook contains training work")


def _validate_raptor_parity_source(source: str) -> None:
    required = (
        "derive_gold_training_cohort",
        "evaluate_point_predictions",
        "load_raptor_model",
        "torch.backends.cudnn.benchmark = True",
        "torch.backends.cuda.matmul.allow_tf32 = True",
        'COMPETITION_ROOT / "train_series"',
        '"hidden_test_inference": False',
        '"competition_submission": False',
        '"training": False',
        "Hidden-test inference: not run",
        "Competition submission: not run",
        "Training: not run",
    )
    if any(value not in source for value in required):
        raise ValueError("Raptor parity notebook omits a phase-boundary safeguard")
    forbidden = (
        'COMPETITION_ROOT / "test_series"',
        "run_raptor_reference",
        "write_submission",
        "kaggle competitions submit",
        "train_epoch",
    )
    if any(value in source for value in forbidden):
        raise ValueError("Raptor parity notebook crosses an unapproved boundary")


def _validate_raptor_scoring_source(source: str) -> None:
    required = (
        "CHECKPOINT_FILENAME",
        "CHECKPOINT_SHA256",
        "CHECKPOINT_SIZE_BYTES",
        "run_raptor_reference",
        "torch.backends.cudnn.benchmark = True",
        "torch.backends.cuda.matmul.allow_tf32 = True",
        '"hidden_test_inference": True',
        '"competition_submission": False',
        '"training": False',
        '"submission.csv"',
        '"submission-fallbacks.csv"',
        '"submission-validation.json"',
        '"raptor-reference-run.json"',
        '"run-manifest.json"',
        "Hidden-test inference: completed",
        "Competition submission: not run",
        "Training: not run",
    )
    if any(value not in source for value in required):
        raise ValueError("Raptor scoring notebook omits an inference safeguard")
    forbidden = (
        "derive_gold_training_cohort",
        "evaluate_point_predictions",
        'COMPETITION_ROOT / "train_series"',
        "write_submission",
        "run_combined_baseline",
        "run_combined_scoring",
        "train_epoch",
        "kaggle competitions submit",
        "competitions.submit",
    )
    if any(value in source for value in forbidden):
        raise ValueError("Raptor scoring notebook crosses an unapproved boundary")


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
