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
    build_public_notebook_bundle,
    push_index_bundle,
    push_inspection_bundle,
    push_public_notebook_bundle,
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


def test_public_notebook_bundle_is_standalone_private_cpu_and_offline(
    tmp_path: Path,
) -> None:
    bundle = build_public_notebook_bundle(Path.cwd(), tmp_path / "bundle")
    metadata = json.loads(bundle.metadata_path.read_text())
    notebook = json.loads(bundle.notebook_path.read_text())
    notebook_source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert metadata == {
        "code_file": "rsna_knee_dicom_ordering.ipynb",
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": [],
        "enable_gpu": "false",
        "enable_internet": "false",
        "enable_tpu": "false",
        "id": "plthon/rsna-knee-mri-correct-dicom-ordering-sampling",
        "is_private": "true",
        "kernel_sources": [],
        "kernel_type": "notebook",
        "language": "python",
        "machine_shape": "",
        "model_sources": [],
        "title": "RSNA Knee MRI | Correct DICOM Ordering & Sampling",
    }
    title_slug = re.sub(r"[^a-z0-9]+", "-", metadata["title"].casefold()).strip("-")
    assert len(metadata["title"]) <= 50
    assert len(title_slug) <= 50
    assert metadata["id"].split("/", 1)[1] == title_slug
    assert notebook["nbformat"] == 4
    cell_ids = [cell["id"] for cell in notebook["cells"]]
    assert len(cell_ids) == len(set(cell_ids))
    assert all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cell_id) for cell_id in cell_ids)
    assert all(
        cell.get("outputs", []) == [] and cell.get("execution_count") is None
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "__RSNA_KNEE_PACKAGE_ARCHIVE__" not in notebook_source
    assert "from rsna_knee" not in notebook_source
    assert "/kaggle/input/competitions/rsna-knee-abnormality-detection" not in (
        notebook_source
    )
    assert "discover_competition_root" in notebook_source
    assert "StudyInstanceUID" in notebook_source
    assert "SeriesInstanceUID" in notebook_source
    assert "ImageOrientationPatient" in notebook_source
    assert "ImagePositionPatient" in notebook_source
    assert "InstanceNumber" in notebook_source
    assert "MONOCHROME1" in notebook_source
    assert "uniform_indices" in notebook_source
    assert "195,088 unique positions" in notebook_source
    assert "all three" in notebook_source
    assert "two of three" in notebook_source
    assert "four-series check" not in notebook_source
    assert "4/4" not in notebook_source
    assert "3/4" not in notebook_source
    assert "Visual comparison: filename order versus geometry" in notebook_source
    assert "Lightweight preprocessing and three-plane visualization" in notebook_source
    assert "TARGETS =" not in notebook_source
    assert "sample_submission_columns" in notebook_source
    assert "range(1, len(target_names) + 1)" in notebook_source
    assert "224" not in notebook_source
    assert "native shapes" in notebook_source
    assert "preserves each slice's native row and column resolution" in notebook_source
    assert "computed jointly across the deterministically selected slices" in (
        notebook_source
    )
    assert "series-level" not in notebook_source
    assert "spatially consistent sequence" in notebook_source
    assert "does not define a universal clinical anatomical direction" in (
        notebook_source
    )
    assert "In this notebook run" in notebook_source
    assert "label_figure_rows" in notebook_source
    assert '("Filename order", "Geometry order")' in notebook_source
    assert '"selected_file_count"' in notebook_source
    assert '"selected_files"' not in notebook_source
    assert "Practical takeaways" in notebook_source


def test_public_notebook_push_refuses_remote_execution_without_approval(
    tmp_path: Path,
) -> None:
    bundle = build_public_notebook_bundle(Path.cwd(), tmp_path / "bundle")
    calls: list[list[str]] = []

    def record_call(command: list[str]) -> CompletedProcess[str]:
        calls.append(command)
        return CompletedProcess(command, 0)

    with pytest.raises(PermissionError, match="explicit remote approval"):
        push_public_notebook_bundle(bundle.root, runner=record_call)

    assert calls == []


def test_public_notebook_builder_rejects_private_repository_details(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    source_path = project_root / "notebooks" / "public_dicom_ordering.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        Path("notebooks/public_dicom_ordering.py").read_text()
        + '\nPRIVATE_PATH = ".artifacts/experiments/registry.jsonl"\n'
    )

    with pytest.raises(ValueError, match="forbidden public-notebook detail"):
        build_public_notebook_bundle(project_root, tmp_path / "bundle")


def test_public_notebook_push_revalidates_the_rendered_notebook(
    tmp_path: Path,
) -> None:
    bundle = build_public_notebook_bundle(Path.cwd(), tmp_path / "bundle")
    notebook = json.loads(bundle.notebook_path.read_text())
    notebook["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ['PRIVATE_PATH = ".artifacts/experiments/registry.jsonl"\n'],
        }
    )
    bundle.notebook_path.write_text(json.dumps(notebook))
    calls: list[list[str]] = []

    def record_call(command: list[str]) -> CompletedProcess[str]:
        calls.append(command)
        return CompletedProcess(command, 0)

    with pytest.raises(ValueError, match="forbidden public-notebook detail"):
        push_public_notebook_bundle(bundle.root, approved=True, runner=record_call)

    assert calls == []
