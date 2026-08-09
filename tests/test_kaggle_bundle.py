import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from rsna_knee.kaggle import build_inspection_bundle, push_inspection_bundle


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
