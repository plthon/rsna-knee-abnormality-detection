import base64
import hashlib
import json
import re
import zipfile
from io import BytesIO
from pathlib import Path
from subprocess import CompletedProcess

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

import rsna_knee.training_bundle as training_bundle_module
from rsna_knee.training import InitializationArtifact
from rsna_knee.training_bundle import build_training_bundle, push_training_bundle

pytestmark = pytest.mark.training


def test_training_bundle_is_private_gpu_offline_and_exactly_initialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weight_path = tmp_path / "fake-weight.pth"
    weight_path.write_bytes(b"exact-offline-weight")
    _mock_initialization(monkeypatch, weight_path)

    bundle = build_training_bundle(
        Path.cwd(),
        tmp_path / "bundle",
        weight_path=weight_path,
        dataset_id="plthon/torchvision-resnet18-imagenet1k-v1",
        kernel_id="plthon/rsna-knee-gold-model-smoke",
        gpu_approved=True,
    )

    metadata = json.loads(bundle.kernel_metadata_path.read_text())
    assert metadata == {
        "code_file": "train_gold_model.ipynb",
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": ["plthon/torchvision-resnet18-imagenet1k-v1"],
        "enable_gpu": "true",
        "enable_internet": "false",
        "enable_tpu": "false",
        "id": "plthon/rsna-knee-gold-model-smoke",
        "is_private": "true",
        "kernel_sources": ["plthon/rsna-knee-data-index"],
        "kernel_type": "notebook",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "model_sources": [],
        "title": "RSNA Knee Gold Model Smoke",
    }
    dataset_metadata = json.loads(bundle.weight_dataset.metadata_path.read_text())
    assert dataset_metadata["id"] == "plthon/torchvision-resnet18-imagenet1k-v1"
    assert "isPublic" not in dataset_metadata
    provenance = json.loads(bundle.weight_dataset.provenance_path.read_text())
    assert provenance["visibility"] == "private"
    assert (
        provenance["artifact"]["sha256"]
        == hashlib.sha256(weight_path.read_bytes()).hexdigest()
    )

    notebook = json.loads(bundle.notebook_path.read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "__RSNA_KNEE_" not in source
    assert "run_gold_smoke_training" in source
    assert "verify_training_artifact" in source
    assert "approved_gpu=GPU_APPROVED" in source
    assert 'json.loads("true")' in source
    assert "report-derived labels are never read" in source
    archive_match = re.search(r'base64\.b64decode\("([A-Za-z0-9+/=]+)"\)', source)
    assert archive_match is not None
    with zipfile.ZipFile(BytesIO(base64.b64decode(archive_match.group(1)))) as archive:
        assert "rsna_knee/training.py" in archive.namelist()
        assert "rsna_knee/gold-supervision.json" in archive.namelist()
        assert "rsna_knee/resnet18-imagenet1k-v1.json" in archive.namelist()


def test_training_bundle_and_push_refuse_missing_gpu_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weight_path = tmp_path / "fake-weight.pth"
    weight_path.write_bytes(b"exact-offline-weight")
    _mock_initialization(monkeypatch, weight_path)
    output_root = tmp_path / "bundle"

    with pytest.raises(PermissionError, match="explicit GPU approval"):
        build_training_bundle(
            Path.cwd(),
            output_root,
            weight_path=weight_path,
            dataset_id="plthon/torchvision-resnet18-imagenet1k-v1",
            kernel_id="plthon/rsna-knee-gold-model-smoke",
        )
    assert not output_root.exists()

    bundle = build_training_bundle(
        Path.cwd(),
        output_root,
        weight_path=weight_path,
        dataset_id="plthon/torchvision-resnet18-imagenet1k-v1",
        kernel_id="plthon/rsna-knee-gold-model-smoke",
        gpu_approved=True,
    )
    calls: list[list[str]] = []

    def record(command: list[str]) -> CompletedProcess[str]:
        calls.append(command)
        return CompletedProcess(command, 0, stdout="ready")

    with pytest.raises(PermissionError, match="explicit remote approval"):
        push_training_bundle(bundle, gpu_approved=True, runner=record)
    with pytest.raises(PermissionError, match="separate explicit GPU approval"):
        push_training_bundle(bundle, approved=True, runner=record)
    assert calls == []

    dataset_result, kernel_result = push_training_bundle(
        bundle,
        approved=True,
        gpu_approved=True,
        runner=record,
        wait=lambda _: None,
    )
    assert dataset_result is not None
    assert kernel_result.returncode == 0
    assert calls == [
        ["kaggle", "datasets", "create", "-p", str(bundle.weight_dataset.root)],
        [
            "kaggle",
            "datasets",
            "status",
            "plthon/torchvision-resnet18-imagenet1k-v1",
        ],
        ["kaggle", "kernels", "push", "-p", str(bundle.kernel_root)],
    ]


def _mock_initialization(
    monkeypatch: pytest.MonkeyPatch, original_weight: Path
) -> None:
    contract = {
        "offline_kaggle_input": {
            "dataset_id": "plthon/torchvision-resnet18-imagenet1k-v1",
            "title": "Torchvision ResNet18 ImageNet1K V1",
        },
        "upstream": {"repository": "https://github.com/pytorch/vision"},
    }

    def verify(
        path: Path, *, contract_path: Path | None = None
    ) -> InitializationArtifact:
        del contract_path
        source = original_weight if path == original_weight else path
        return InitializationArtifact(
            path=path,
            sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            size_bytes=source.stat().st_size,
            contract_sha256="contract-sha256",
            contract=contract,
        )

    monkeypatch.setattr(
        training_bundle_module, "verify_initialization_artifact", verify
    )
