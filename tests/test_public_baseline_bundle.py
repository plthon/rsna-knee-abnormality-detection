import json
import zipfile
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from rsna_knee.public_baseline_bundle import (
    build_public_baseline_bundle,
    load_public_baseline_bundle,
    push_public_baseline_bundle,
)


def _notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_public_baseline_bundle_is_offline_private_and_wheel_backed(
    tmp_path: Path,
) -> None:
    bundle = build_public_baseline_bundle(
        Path.cwd(),
        tmp_path / "bundle",
        wheel_dataset_id="plthon/rsna-knee-gold-baseline-package",
        weight_dataset_id="plthon/torchvision-resnet18-imagenet1k-v1",
        kernel_id="plthon/rsna-knee-a-reproducible-mri-baseline",
        gpu_approved=True,
    )

    metadata = json.loads(bundle.kernel_metadata_path.read_text())
    source = _notebook_source(bundle.notebook_path)
    with zipfile.ZipFile(bundle.wheel_dataset.wheel_path) as wheel:
        names = set(wheel.namelist())

    assert metadata == {
        "code_file": "public_gold_baseline.ipynb",
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": [
            "plthon/rsna-knee-gold-baseline-package",
            "plthon/torchvision-resnet18-imagenet1k-v1",
        ],
        "enable_gpu": "true",
        "enable_internet": "false",
        "enable_tpu": "false",
        "id": "plthon/rsna-knee-a-reproducible-mri-baseline",
        "is_private": "true",
        "kernel_sources": [],
        "kernel_type": "notebook",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "model_sources": [],
        "title": "RSNA Knee: A Reproducible MRI Baseline",
    }
    assert {
        "rsna_knee/publication.py",
        "rsna_knee/training.py",
        "rsna_knee/baseline.py",
        "rsna_knee/gold-baseline.json",
        "rsna_knee/resnet18-imagenet1k-v1.json",
    } <= names
    assert all(
        token in source
        for token in (
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
            "Training uses only labels supplied by the competition",
        )
    )
    assert all(
        token not in source.casefold()
        for token in (
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
    )
    assert "__RSNA_KNEE_" not in source
    assert not list(bundle.root.rglob("*.pt"))
    assert not list(bundle.root.rglob("*.pth"))
    assert not list(bundle.root.rglob("submission.csv"))


def test_public_baseline_build_and_push_require_explicit_approvals(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bundle"
    with pytest.raises(PermissionError, match="explicit GPU approval"):
        build_public_baseline_bundle(
            Path.cwd(),
            output,
            wheel_dataset_id="plthon/rsna-knee-gold-baseline-package",
            weight_dataset_id="plthon/torchvision-resnet18-imagenet1k-v1",
            kernel_id="plthon/rsna-knee-a-reproducible-mri-baseline",
        )
    assert not output.exists()

    bundle = build_public_baseline_bundle(
        Path.cwd(),
        output,
        wheel_dataset_id="plthon/rsna-knee-gold-baseline-package",
        weight_dataset_id="plthon/torchvision-resnet18-imagenet1k-v1",
        kernel_id="plthon/rsna-knee-a-reproducible-mri-baseline",
        gpu_approved=True,
    )
    bundle = load_public_baseline_bundle(output)
    calls: list[list[str]] = []

    def record(command: list[str]) -> CompletedProcess[str]:
        calls.append(command)
        if command[:3] == ["kaggle", "datasets", "status"]:
            return CompletedProcess(command, 0, stdout="ready")
        return CompletedProcess(command, 0)

    with pytest.raises(PermissionError, match="explicit remote approval"):
        push_public_baseline_bundle(bundle, runner=record)
    with pytest.raises(PermissionError, match="separate explicit GPU approval"):
        push_public_baseline_bundle(bundle, approved=True, runner=record)
    assert calls == []

    dataset, kernel = push_public_baseline_bundle(
        bundle,
        approved=True,
        gpu_approved=True,
        reuse_ready_dataset=True,
        runner=record,
        wait=lambda _: None,
    )
    assert dataset is None
    assert kernel.returncode == 0
    assert calls == [
        [
            "kaggle",
            "datasets",
            "status",
            "plthon/rsna-knee-gold-baseline-package",
        ],
        ["kaggle", "kernels", "push", "-p", str(bundle.kernel_root)],
    ]


def test_public_baseline_rejects_a_different_weight_dataset_before_building(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="differs from initialization contract"):
        build_public_baseline_bundle(
            Path.cwd(),
            output,
            wheel_dataset_id="plthon/rsna-knee-gold-baseline-package",
            weight_dataset_id="plthon/different-resnet-weights",
            kernel_id="plthon/rsna-knee-a-reproducible-mri-baseline",
            gpu_approved=True,
        )
    assert not output.exists()
