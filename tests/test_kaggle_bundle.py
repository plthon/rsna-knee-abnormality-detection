import base64
import json
import re
import zipfile
from io import BytesIO
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from rsna_knee.kaggle import (
    build_index_bundle,
    build_inspection_bundle,
    push_index_bundle,
    push_inspection_bundle,
)


def test_inspection_bundle_is_private_cpu_only_and_contains_package(
    tmp_path: Path,
) -> None:
    bundle = build_inspection_bundle(Path.cwd(), tmp_path / "bundle")
    metadata = json.loads(bundle.metadata_path.read_text())
    notebook = json.loads(bundle.notebook_path.read_text())
    notebook_source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert {
        "metadata": metadata,
        "is_notebook": notebook["nbformat"] == 4,
        "package_marker_resolved": "__RSNA_KNEE_PACKAGE_ARCHIVE__"
        not in notebook_source,
        "calls_inspection_interface": (
            "from rsna_knee import inspect_competition" in notebook_source
        ),
    } == {
        "metadata": {
            "code_file": "inspect_real_data.ipynb",
            "competition_sources": ["rsna-knee-abnormality-detection"],
            "dataset_sources": [],
            "enable_gpu": "false",
            "enable_internet": "false",
            "enable_tpu": "false",
            "id": "plthon/rsna-knee-real-data-inspection",
            "is_private": "true",
            "kernel_sources": [],
            "kernel_type": "notebook",
            "language": "python",
            "machine_shape": "",
            "model_sources": [],
            "title": "RSNA Knee Real Data Inspection",
        },
        "is_notebook": True,
        "package_marker_resolved": True,
        "calls_inspection_interface": True,
    }


def test_inspection_push_refuses_to_call_kaggle_without_approval(
    tmp_path: Path,
) -> None:
    bundle = build_inspection_bundle(Path.cwd(), tmp_path / "bundle")
    calls: list[list[str]] = []

    def record_call(command: list[str]) -> CompletedProcess[str]:
        calls.append(command)
        return CompletedProcess(command, 0)

    with pytest.raises(PermissionError, match="explicit remote approval"):
        push_inspection_bundle(bundle.root, runner=record_call)

    assert calls == []


def test_index_bundle_is_private_cpu_only_and_runs_shared_contract(
    tmp_path: Path,
) -> None:
    bundle = build_index_bundle(Path.cwd(), tmp_path / "bundle")
    metadata = json.loads(bundle.metadata_path.read_text())
    notebook = json.loads(bundle.notebook_path.read_text())
    notebook_source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert metadata == {
        "code_file": "index_real_data.ipynb",
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": [],
        "enable_gpu": "false",
        "enable_internet": "false",
        "enable_tpu": "false",
        "id": "plthon/rsna-knee-data-index",
        "is_private": "true",
        "kernel_sources": [],
        "kernel_type": "notebook",
        "language": "python",
        "machine_shape": "",
        "model_sources": [],
        "title": "RSNA Knee Data Index",
    }
    assert notebook["nbformat"] == 4
    assert "__RSNA_KNEE_PACKAGE_ARCHIVE__" not in notebook_source
    assert "build_competition_index" in notebook_source
    assert "verify_index_artifact" in notebook_source
    assert "slice_count=8" in notebook_source
    assert "image_size=224" in notebook_source
    encoded_archive = re.search(
        r'base64\.b64decode\("([A-Za-z0-9+/=]+)"\)', notebook_source
    )
    assert encoded_archive is not None
    with zipfile.ZipFile(
        BytesIO(base64.b64decode(encoded_archive.group(1)))
    ) as archive:
        assert (
            archive.read("rsna_knee/competition.json")
            == Path("contracts/competition.json").read_bytes()
        )


def test_index_push_refuses_to_call_kaggle_without_approval(tmp_path: Path) -> None:
    bundle = build_index_bundle(Path.cwd(), tmp_path / "bundle")
    calls: list[list[str]] = []

    def record_call(command: list[str]) -> CompletedProcess[str]:
        calls.append(command)
        return CompletedProcess(command, 0)

    with pytest.raises(PermissionError, match="explicit remote approval"):
        push_index_bundle(bundle.root, runner=record_call)

    assert calls == []
