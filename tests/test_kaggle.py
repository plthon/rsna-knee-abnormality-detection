import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from rsna_knee.kaggle import (
    KaggleSpec,
    WheelDataset,
    build_kaggle_bundle,
    build_public_baseline_bundle,
    build_reports_bundle,
    build_wheel_dataset,
    pull_kaggle_output,
    push_kaggle_bundle,
)


def _attached_dataset(
    root: Path,
    *,
    dataset_id: str = "alice/dataset",
    config_name: str = "config.toml",
    config_text: str = "epochs = 1\n",
) -> tuple[WheelDataset, Path]:
    root.mkdir(parents=True, exist_ok=True)
    wheel_path = root / "rsna_knee-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        wheel.writestr(
            "rsna_knee-0.1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
    config_path = root / config_name
    config_path.write_text(config_text)
    metadata_path = root / "dataset-metadata.json"
    metadata_path.write_text(json.dumps({"id": dataset_id}))
    provenance_path = root / "wheel-provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "git_commit": "test-commit",
                "git_dirty": False,
                "wheel": {
                    "filename": wheel_path.name,
                    "sha256": hashlib.sha256(wheel_path.read_bytes()).hexdigest(),
                },
            }
        )
    )
    return (
        WheelDataset(
            root=root,
            wheel_path=wheel_path,
            provenance_path=provenance_path,
            metadata_path=metadata_path,
            config_path=config_path,
        ),
        config_path,
    )


def test_kaggle_boundary_renders_safe_metadata_and_requires_remote_approval(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    source_path = project / "notebooks" / "experiment.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("# %% [markdown]\n# Example\n# %%\nprint('ok')\n")
    spec = KaggleSpec(
        source_name="experiment.py",
        notebook_name="example.ipynb",
        kernel_id="alice/example",
        title="Example",
        competition_sources=("rsna-knee-abnormality-detection",),
    )

    bundle = build_kaggle_bundle(project, tmp_path / "bundle", spec)
    metadata = json.loads(bundle.kernel_metadata_path.read_text())
    assert metadata["is_private"] == "true"
    assert metadata["enable_internet"] == "false"
    assert metadata["enable_gpu"] == "false"
    assert json.loads(bundle.notebook_path.read_text())["nbformat"] == 4

    calls: list[list[str]] = []

    def runner(command: list[str]) -> CompletedProcess[str]:
        calls.append(command)
        return CompletedProcess(command, 0)

    with pytest.raises(PermissionError, match="explicit remote approval"):
        push_kaggle_bundle(bundle, runner=runner)
    assert calls == []


def test_kaggle_boundary_requires_gpu_approval_before_writing(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source_path = project / "notebooks" / "experiment.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("# %%\nprint('ok')\n")
    spec = KaggleSpec(
        source_name="experiment.py",
        notebook_name="example.ipynb",
        kernel_id="alice/gpu-example",
        title="GPU Example",
        gpu=True,
        machine_shape="NvidiaTeslaT4",
    )

    with pytest.raises(PermissionError, match="GPU approval"):
        build_kaggle_bundle(project, tmp_path / "bundle", spec)
    assert not (tmp_path / "bundle").exists()


def test_remote_push_rejects_configuration_changed_after_render(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    source_path = project / "notebooks" / "experiment.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("# %%\nprint('ok')\n")

    dataset, config_path = _attached_dataset(tmp_path / "dataset")
    bundle = build_kaggle_bundle(
        project,
        tmp_path / "bundle",
        KaggleSpec(
            source_name="experiment.py",
            notebook_name="example.ipynb",
            kernel_id="alice/example",
            title="Example",
            dataset_sources=("alice/dataset",),
        ),
        dataset=dataset,
    )
    config_path.write_text("epochs = 2\n")
    calls: list[list[str]] = []

    def runner(command: list[str]) -> CompletedProcess[str]:
        calls.append(command)
        return CompletedProcess(command, 0, stdout="ready")

    with pytest.raises(ValueError, match="Configuration digest"):
        push_kaggle_bundle(
            bundle,
            approved=True,
            runner=runner,
            wait=lambda _: None,
        )
    assert calls == []


def test_remote_push_rejects_wheel_changed_after_render(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source_path = project / "notebooks" / "experiment.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("# %%\nprint('ok')\n")
    dataset, _ = _attached_dataset(tmp_path / "dataset")
    bundle = build_kaggle_bundle(
        project,
        tmp_path / "bundle",
        KaggleSpec(
            source_name="experiment.py",
            notebook_name="example.ipynb",
            kernel_id="alice/example",
            title="Example",
            dataset_sources=("alice/dataset",),
        ),
        dataset=dataset,
    )
    dataset.wheel_path.write_bytes(dataset.wheel_path.read_bytes() + b"drift")
    calls: list[list[str]] = []

    def runner(command: list[str]) -> CompletedProcess[str]:
        calls.append(command)
        return CompletedProcess(command, 0, stdout="ready")

    with pytest.raises(ValueError, match="Wheel provenance"):
        push_kaggle_bundle(
            bundle,
            approved=True,
            runner=runner,
            wait=lambda _: None,
        )
    assert calls == []


def test_wheel_provenance_captures_source_status_before_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    (project / "configs" / "kaggle.toml").write_text("epochs = 1\n")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> CompletedProcess[str]:
        calls.append(command)
        if command[0] == "uv":
            output_root = Path(command[command.index("--out-dir") + 1])
            output_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(
                output_root / "rsna_knee-0.1.0-py3-none-any.whl", "w"
            ) as wheel:
                wheel.writestr(
                    "rsna_knee-0.1.0.dist-info/WHEEL",
                    "Root-Is-Purelib: true\nTag: py3-none-any\n",
                )
            return CompletedProcess(command, 0)
        if command[1:3] == ["rev-parse", "HEAD"]:
            return CompletedProcess(command, 0, stdout="test-commit\n")
        if command[1:3] == ["status", "--porcelain"]:
            return CompletedProcess(
                command, 0, stdout="?? src/rsna_knee/untracked_marker.py\n"
            )
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr("rsna_knee.kaggle.subprocess.run", fake_run)

    dataset = build_wheel_dataset(
        project,
        tmp_path / "wheel-dataset",
        dataset_id="alice/offline-wheel",
    )

    assert calls[0] == ["git", "rev-parse", "HEAD"]
    assert calls[1] == ["git", "status", "--porcelain", "--untracked-files=all"]
    assert calls[2][0] == "uv"
    assert json.loads(dataset.provenance_path.read_text())["git_dirty"] is True


def test_wheel_provenance_includes_untracked_package_sources(tmp_path: Path) -> None:
    project = tmp_path / "project"
    package = project / "src" / "rsna_knee"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n")
    (project / "configs").mkdir()
    (project / "configs" / "kaggle.toml").write_text("epochs = 1\n")
    (project / "pyproject.toml").write_text(
        """
[project]
name = "rsna-knee"
version = "0.1.0"
requires-python = ">=3.12"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/rsna_knee"]
""".lstrip()
    )
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    subprocess.run(["git", "add", "."], cwd=project, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=project,
        check=True,
    )
    (package / "untracked_marker.py").write_text("MARKER = True\n")

    dataset = build_wheel_dataset(
        project,
        tmp_path / "wheel-dataset",
        dataset_id="alice/offline-wheel",
    )

    with zipfile.ZipFile(dataset.wheel_path) as wheel:
        assert "rsna_knee/untracked_marker.py" in wheel.namelist()
    assert json.loads(dataset.provenance_path.read_text())["git_dirty"] is True


def test_kaggle_boundary_rejects_unresolved_id_before_writing(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source_path = project / "notebooks" / "experiment.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("# %%\nprint('ok')\n")

    with pytest.raises(ValueError, match="Invalid or unresolved Kaggle ID"):
        build_kaggle_bundle(
            project,
            tmp_path / "bundle",
            KaggleSpec(
                source_name="experiment.py",
                notebook_name="example.ipynb",
                kernel_id="alice/KERNEL_SLUG",
                title="Example",
            ),
        )
    assert not (tmp_path / "bundle").exists()


def test_kaggle_cli_builds_validated_public_ordering_bundle(tmp_path: Path) -> None:
    project = Path(__file__).parents[1]
    output = tmp_path / "ordering"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/kaggle.py",
            "build",
            "ordering",
            "--output",
            str(output),
        ],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == str(output)
    metadata = json.loads((output / "kernel" / "kernel-metadata.json").read_text())
    provenance = json.loads((output / "bundle-provenance.json").read_text())
    notebook = json.loads(
        (output / "kernel" / "rsna_knee_dicom_ordering.ipynb").read_text()
    )

    assert metadata["is_private"] == "true"
    assert metadata["enable_internet"] == "false"
    assert metadata["enable_gpu"] == "false"
    assert provenance["validator"] == "ordering"
    assert "__RSNA_KNEE_" not in "\n".join(notebook["cells"][0]["source"])


def test_public_baseline_builder_attaches_runtime_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = Path(__file__).parents[1]

    def fake_build_wheel_dataset(
        project_root: Path,
        output_root: Path,
        *,
        dataset_id: str,
        config_path: Path | None,
        contract_names: tuple[str, ...],
        description: str,
        title: str,
    ) -> WheelDataset:
        del config_path, description, title
        dataset, _ = _attached_dataset(
            output_root,
            dataset_id=dataset_id,
            config_name="kaggle-baseline.toml",
            config_text=(project_root / "configs" / "kaggle-baseline.toml").read_text(),
        )
        for contract_name in contract_names:
            shutil.copy2(
                project_root / "contracts" / contract_name, output_root / contract_name
            )
        return dataset

    monkeypatch.setattr(
        "rsna_knee.kaggle.build_wheel_dataset", fake_build_wheel_dataset
    )
    bundle = build_public_baseline_bundle(
        project,
        tmp_path / "baseline",
        wheel_dataset_id="alice/gold",
        weight_dataset_id="plthon/torchvision-resnet18-imagenet1k-v1",
        kernel_id="alice/rsna-knee-a-reproducible-mri-baseline",
        gpu_approved=True,
    )
    metadata = json.loads(bundle.kernel_metadata_path.read_text())
    assert bundle.dataset is not None
    attached = {path.name for path in bundle.dataset.root.iterdir()}

    assert metadata["enable_gpu"] == "true"
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert {
        "data-index.json",
        "gold-baseline.json",
        "gold-supervision.json",
        "gold-training.json",
        "resnet18-imagenet1k-v1.json",
    } <= attached


def test_reports_audit_bundle_is_private_offline_cpu_and_stage_resolved(
    tmp_path: Path,
) -> None:
    project = Path(__file__).parents[1]

    bundle = build_reports_bundle(project, tmp_path / "reports-audit", stage="audit")
    metadata = json.loads(bundle.kernel_metadata_path.read_text())
    notebook = json.loads(bundle.notebook_path.read_text())
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert bundle.dataset is None
    assert metadata == {
        "code_file": "report_labels.ipynb",
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": [],
        "enable_gpu": "false",
        "enable_internet": "false",
        "enable_tpu": "false",
        "id": "plthon/rsna-knee-report-labels",
        "is_private": "true",
        "kernel_sources": [],
        "kernel_type": "notebook",
        "language": "python",
        "machine_shape": "",
        "model_sources": [],
        "title": "RSNA Knee Report Labels",
    }
    assert 'STAGE = "audit"' in source
    assert "__RSNA_KNEE_" not in source
    assert "rsna_knee_source.zip" in source

    calls: list[list[str]] = []

    def runner(command: list[str]) -> CompletedProcess[str]:
        calls.append(command)
        return CompletedProcess(command, 0)

    with pytest.raises(PermissionError, match="explicit remote approval"):
        push_kaggle_bundle(bundle, runner=runner)
    assert calls == []


def test_reports_builder_rejects_invalid_stage_before_writing(tmp_path: Path) -> None:
    project = Path(__file__).parents[1]

    with pytest.raises(ValueError, match="audit or extract"):
        build_reports_bundle(project, tmp_path / "reports", stage="discover")

    assert not (tmp_path / "reports").exists()


def test_kaggle_output_pull_is_read_only_and_confined_to_artifacts(
    tmp_path: Path,
) -> None:
    project = Path(__file__).parents[1]
    bundle = build_reports_bundle(project, tmp_path / "reports-audit", stage="audit")
    artifact_root = tmp_path / ".artifacts"
    output = artifact_root / "report-audit"
    calls: list[list[str]] = []

    def runner(command: list[str]) -> CompletedProcess[str]:
        calls.append(command)
        return CompletedProcess(command, 0)

    pull_kaggle_output(
        bundle,
        output,
        artifact_root=artifact_root,
        filenames=("report-audit.json", "report-audit-manifest.json"),
        runner=runner,
    )

    assert calls == [
        [
            "kaggle",
            "kernels",
            "output",
            "plthon/rsna-knee-report-labels",
            "-p",
            str(output),
            "--file-pattern",
            r"^report\-audit\.json$",
        ],
        [
            "kaggle",
            "kernels",
            "output",
            "plthon/rsna-knee-report-labels",
            "-p",
            str(output),
            "--file-pattern",
            r"^report\-audit\-manifest\.json$",
        ],
    ]

    with pytest.raises(ValueError, match="ignored artifact root"):
        pull_kaggle_output(
            bundle,
            tmp_path / "committed-output",
            artifact_root=artifact_root,
            filenames=("report-audit.json",),
            runner=runner,
        )
