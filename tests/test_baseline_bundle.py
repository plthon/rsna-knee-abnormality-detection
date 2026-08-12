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

import rsna_knee.baseline_bundle as bundle_module
from rsna_knee.baseline_bundle import build_baseline_bundle, push_baseline_bundle
from rsna_knee.training import InitializationArtifact


def test_baseline_bundle_is_private_offline_gpu_and_contains_frozen_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weight_path = tmp_path / "fake-weight.pth"
    weight_path.write_bytes(b"exact-offline-weight")
    _mock_initialization(monkeypatch, weight_path)

    bundle = build_baseline_bundle(
        Path.cwd(),
        tmp_path / "bundle",
        weight_path=weight_path,
        dataset_id="plthon/torchvision-resnet18-imagenet1k-v1",
        kernel_id="plthon/rsna-knee-gold-holdout-baseline",
        gpu_approved=True,
    )

    metadata = json.loads(bundle.kernel_metadata_path.read_text())
    assert metadata["enable_gpu"] == "true"
    assert metadata["enable_internet"] == "false"
    assert metadata["is_private"] == "true"
    assert metadata["competition_sources"] == ["rsna-knee-abnormality-detection"]
    assert metadata["kernel_sources"] == ["plthon/rsna-knee-data-index"]
    notebook = json.loads(bundle.notebook_path.read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "__RSNA_KNEE_" not in source
    assert "run_gold_holdout_baseline" in source
    assert "verify_baseline_artifact" in source
    assert "approved_gpu=GPU_APPROVED" in source
    assert "report-derived labels are never read" in source
    assert "patient-id-nonparametric-bootstrap-v1" not in source
    archive_match = re.search(r'base64\.b64decode\("([A-Za-z0-9+/=]+)"\)', source)
    assert archive_match is not None
    with zipfile.ZipFile(BytesIO(base64.b64decode(archive_match.group(1)))) as archive:
        assert "rsna_knee/baseline.py" in archive.namelist()
        assert "rsna_knee/holdout.py" in archive.namelist()
        assert "rsna_knee/gold-supervision.json" in archive.namelist()
        assert "rsna_knee/gold-training.json" in archive.namelist()


def test_baseline_bundle_and_push_require_explicit_approvals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weight_path = tmp_path / "fake-weight.pth"
    weight_path.write_bytes(b"exact-offline-weight")
    _mock_initialization(monkeypatch, weight_path)
    output = tmp_path / "bundle"
    with pytest.raises(PermissionError, match="explicit GPU approval"):
        build_baseline_bundle(
            Path.cwd(),
            output,
            weight_path=weight_path,
            dataset_id="plthon/torchvision-resnet18-imagenet1k-v1",
            kernel_id="plthon/rsna-knee-gold-holdout-baseline",
        )
    assert not output.exists()

    bundle = build_baseline_bundle(
        Path.cwd(),
        output,
        weight_path=weight_path,
        dataset_id="plthon/torchvision-resnet18-imagenet1k-v1",
        kernel_id="plthon/rsna-knee-gold-holdout-baseline",
        gpu_approved=True,
    )
    calls: list[list[str]] = []

    def record(command: list[str]) -> CompletedProcess[str]:
        calls.append(command)
        if command[:3] == ["kaggle", "datasets", "status"]:
            return CompletedProcess(command, 0, stdout="ready")
        return CompletedProcess(command, 0)

    with pytest.raises(PermissionError, match="explicit remote approval"):
        push_baseline_bundle(bundle, runner=record)
    with pytest.raises(PermissionError, match="separate explicit GPU approval"):
        push_baseline_bundle(bundle, approved=True, runner=record)
    assert calls == []
    dataset, kernel = push_baseline_bundle(
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

    monkeypatch.setattr(bundle_module, "verify_initialization_artifact", verify)
