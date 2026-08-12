import base64
import json
import re
import zipfile
from io import BytesIO
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from rsna_knee.kaggle import build_gold_audit_bundle, push_gold_audit_bundle


def test_gold_audit_bundle_is_private_cpu_only_and_attaches_accepted_index(
    tmp_path: Path,
) -> None:
    bundle = build_gold_audit_bundle(Path.cwd(), tmp_path / "bundle")
    metadata = json.loads(bundle.metadata_path.read_text())
    notebook = json.loads(bundle.notebook_path.read_text())
    notebook_source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert metadata == {
        "code_file": "gold_supervision_audit.ipynb",
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": [],
        "enable_gpu": "false",
        "enable_internet": "false",
        "enable_tpu": "false",
        "id": "plthon/rsna-knee-gold-supervision-audit",
        "is_private": "true",
        "kernel_sources": ["plthon/rsna-knee-data-index"],
        "kernel_type": "notebook",
        "language": "python",
        "machine_shape": "",
        "model_sources": [],
        "title": "RSNA Knee Gold Supervision Audit",
    }
    assert notebook["nbformat"] == 4
    assert "__RSNA_KNEE_" not in notebook_source
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = ""' in notebook_source
    assert "discover_gold_audit_inputs" in notebook_source
    assert "audit_gold_supervision" in notebook_source
    assert "verify_gold_audit_artifact" in notebook_source
    assert "does not process or export report" in notebook_source
    assert "DICOM" in notebook_source

    encoded_archive = re.search(
        r'base64\.b64decode\("([A-Za-z0-9+/=]+)"\)', notebook_source
    )
    assert encoded_archive is not None
    with zipfile.ZipFile(
        BytesIO(base64.b64decode(encoded_archive.group(1)))
    ) as archive:
        archive_names = archive.namelist()
        assert "rsna_knee/gold_audit.py" in archive_names
        assert (
            archive.read("rsna_knee/competition.json")
            == Path("contracts/competition.json").read_bytes()
        )
        assert (
            archive.read("rsna_knee/data-index.json")
            == Path("contracts/data-index.json").read_bytes()
        )
        assert "rsna_knee/gold-supervision.json" not in archive_names


def test_gold_audit_push_refuses_remote_execution_without_approval(
    tmp_path: Path,
) -> None:
    bundle = build_gold_audit_bundle(Path.cwd(), tmp_path / "bundle")
    calls: list[list[str]] = []

    def record_call(command: list[str]) -> CompletedProcess[str]:
        calls.append(command)
        return CompletedProcess(command, 0)

    with pytest.raises(PermissionError, match="explicit remote approval"):
        push_gold_audit_bundle(bundle.root, runner=record_call)

    assert calls == []
