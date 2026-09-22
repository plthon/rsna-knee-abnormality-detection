import base64
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from rsna_knee.kaggle import (
    KaggleSpec,
    WeightDataset,
    WheelDataset,
    build_dinov2_resume_dataset,
    build_dinov2_scoring_bundle,
    build_dinov2_training_bundle,
    build_dinov2_weight_dataset,
    build_kaggle_bundle,
    build_plane_gated_scoring_bundle,
    build_plane_gating_bundle,
    build_public_baseline_bundle,
    build_raptor_parity_bundle,
    build_report_llm_bundle,
    build_report_source_comparison_bundle,
    build_reports_bundle,
    build_wheel_dataset,
    load_kaggle_bundle,
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


def _attached_weight_dataset(root: Path) -> WeightDataset:
    root.mkdir(parents=True, exist_ok=True)
    weight_path = root / "dinov2_vits14_pretrain.pth"
    weight_path.write_bytes(b"verified-test-weights")
    weight_sha256 = hashlib.sha256(weight_path.read_bytes()).hexdigest()
    contract_path = root / "dinov2-vits14-lvd142m.json"
    contract_path.write_text(
        json.dumps(
            {
                "artifact": {
                    "filename": weight_path.name,
                    "sha256": weight_sha256,
                    "size_bytes": weight_path.stat().st_size,
                },
                "offline_kaggle_input": {"dataset_id": "plthon/dinov2-vits14-lvd142m"},
            }
        )
    )
    metadata_path = root / "dataset-metadata.json"
    metadata_path.write_text(json.dumps({"id": "plthon/dinov2-vits14-lvd142m"}))
    provenance_path = root / "weight-provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "plthon/dinov2-vits14-lvd142m",
                "contract_sha256": hashlib.sha256(
                    contract_path.read_bytes()
                ).hexdigest(),
                "weight": {
                    "filename": weight_path.name,
                    "sha256": weight_sha256,
                    "size_bytes": weight_path.stat().st_size,
                },
            }
        )
    )
    return WeightDataset(
        root=root,
        weight_path=weight_path,
        contract_path=contract_path,
        provenance_path=provenance_path,
        metadata_path=metadata_path,
    )


def test_dinov2_weight_dataset_requires_and_packages_exact_contract_bytes(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    contracts = project / "contracts"
    contracts.mkdir(parents=True)
    source_path = tmp_path / "dinov2_vits14_pretrain.pth"
    source_path.write_bytes(b"verified-test-weights")
    contract = {
        "artifact": {
            "filename": source_path.name,
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "size_bytes": source_path.stat().st_size,
        },
        "offline_kaggle_input": {"dataset_id": "plthon/dinov2-vits14-lvd142m"},
    }
    (contracts / "dinov2-vits14-lvd142m.json").write_text(json.dumps(contract))

    dataset = build_dinov2_weight_dataset(
        project, tmp_path / "dataset", weight_path=source_path
    )

    assert dataset.weight_path.read_bytes() == source_path.read_bytes()
    assert json.loads(dataset.metadata_path.read_text())["id"] == (
        "plthon/dinov2-vits14-lvd142m"
    )
    assert (
        json.loads(dataset.provenance_path.read_text())["weight"]
        == contract["artifact"]
    )

    source_path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="DINOv2 weight artifact"):
        build_dinov2_weight_dataset(
            project, tmp_path / "rejected", weight_path=source_path
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


def test_reports_extract_bundle_embeds_frozen_contract_and_canonical_hash(
    tmp_path: Path,
) -> None:
    project = Path(__file__).parents[1]

    bundle = build_reports_bundle(
        project, tmp_path / "reports-extract", stage="extract"
    )
    rendered = bundle.notebook_path.read_text(encoding="utf-8")
    notebook = json.loads(rendered)
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    provenance = json.loads(bundle.provenance_path.read_text(encoding="utf-8"))
    rendered_record = provenance["rendered_notebook"]
    rendered_sha256 = rendered_record["sha256"]
    canonical = rendered.replace(rendered_sha256, "0" * 64)
    archive_match = re.search(r'base64\.b64decode\("([A-Za-z0-9+/=]+)"\)', source)

    assert 'STAGE = "extract"' in source
    assert "write_extraction_artifacts" in source
    assert rendered_record["canonicalization"] == (
        "rendered notebook with its recorded SHA-256 replaced by 64 zeroes"
    )
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == rendered_sha256
    assert archive_match is not None
    with zipfile.ZipFile(BytesIO(base64.b64decode(archive_match.group(1)))) as archive:
        assert "rsna_knee/report-extractor-v1.json" in archive.namelist()


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
    with pytest.raises(PermissionError, match="row-level"):
        pull_kaggle_output(
            bundle,
            output,
            artifact_root=artifact_root,
            filenames=("report-labels-v1.csv",),
            runner=runner,
        )


@pytest.mark.parametrize("stage", ["smoke", "extract"])
def test_report_llm_bundle_is_private_offline_gpu_and_frozen(
    tmp_path: Path, stage: str
) -> None:
    project = Path(__file__).parents[1]

    bundle = build_report_llm_bundle(
        project, tmp_path / f"report-llm-{stage}", stage=stage
    )
    metadata = json.loads(bundle.kernel_metadata_path.read_text())
    notebook = json.loads(bundle.notebook_path.read_text())
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert metadata["competition_sources"] == ["rsna-knee-abnormality-detection"]
    assert metadata["kernel_sources"] == ["plthon/rsna-knee-report-labels"]
    assert metadata["model_sources"] == ["qwen-lm/qwen2.5/transformers/3b-instruct/1"]
    assert metadata["enable_gpu"] == "true"
    assert metadata["enable_internet"] == "false"
    assert metadata["is_private"] == "true"
    assert f'STAGE = "{stage}"' in source
    assert "Qwen/Qwen2.5-3B-Instruct" not in source
    assert "verify_model_asset" in source
    assert "read_non_gold_study_uids" in source
    assert 'model_asset["mount_paths"]' in source
    assert "/kaggle/input/kernels/plthon/rsna-knee-report-labels" in source
    assert "/kaggle/input/plthon/rsna-knee-report-labels" in source
    assert "resolve_dataset_root" in source
    assert "resolve_private_kernel_root" in source
    assert 'requested_columns=("StudyInstanceUID", "Report")' in source
    assert 'expected_study_count = 32 if STAGE == "smoke" else 4_407' in source
    assert "if len(selected_rows) != expected_study_count:" in source
    assert "__RSNA_KNEE_" not in source

    calls: list[list[str]] = []

    def runner(command: list[str]) -> CompletedProcess[str]:
        calls.append(command)
        return CompletedProcess(command, 0)

    with pytest.raises(PermissionError, match="explicit remote approval"):
        push_kaggle_bundle(bundle, runner=runner)
    with pytest.raises(PermissionError, match="GPU approval"):
        push_kaggle_bundle(bundle, approved=True, runner=runner)
    assert calls == []


@pytest.mark.parametrize(
    "filename",
    ("report-llm-stability-first.json", "report-llm-stability-second.json"),
)
def test_report_llm_stability_checkpoints_cannot_be_pulled(
    tmp_path: Path, filename: str
) -> None:
    project = Path(__file__).parents[1]
    bundle = build_report_llm_bundle(
        project, tmp_path / "report-llm-extract", stage="extract"
    )

    with pytest.raises(PermissionError, match="row-level"):
        pull_kaggle_output(
            bundle,
            tmp_path / ".artifacts" / "report-llm",
            artifact_root=tmp_path / ".artifacts",
            filenames=(filename,),
            runner=lambda command: CompletedProcess(command, 0),
        )


def test_report_llm_cli_builds_smoke_without_remote_action(tmp_path: Path) -> None:
    project = Path(__file__).parents[1]
    output = tmp_path / "report-llm-smoke"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/kaggle.py",
            "build",
            "report-llm",
            "--stage",
            "smoke",
            "--output",
            str(output),
        ],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == str(output)
    assert (output / "kernel" / "report_llm_labels.ipynb").is_file()


def test_report_llm_builder_rejects_invalid_stage_before_writing(
    tmp_path: Path,
) -> None:
    project = Path(__file__).parents[1]

    with pytest.raises(ValueError, match="smoke or extract"):
        build_report_llm_bundle(project, tmp_path / "report-llm", stage="acquire")

    assert not (tmp_path / "report-llm").exists()


def test_source_comparison_bundle_is_private_offline_cpu_and_frozen(
    tmp_path: Path,
) -> None:
    project = Path(__file__).parents[1]

    bundle = build_report_source_comparison_bundle(
        project, tmp_path / "report-source-comparison", stage="compare"
    )
    metadata = json.loads(bundle.kernel_metadata_path.read_text())
    notebook = json.loads(bundle.notebook_path.read_text())
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert metadata["competition_sources"] == ["rsna-knee-abnormality-detection"]
    assert metadata["dataset_sources"] == ["pilkwang/rsna-knee-llm-labels"]
    assert metadata["kernel_sources"] == [
        "plthon/rsna-knee-report-labels",
        "plthon/rsna-knee-report-llm-labels",
    ]
    assert metadata["enable_gpu"] == "false"
    assert metadata["enable_internet"] == "false"
    assert metadata["is_private"] == "true"
    assert metadata["id"] == "plthon/rsna-knee-report-source-comparison"
    assert 'STAGE = "compare"' in source
    assert "__RSNA_KNEE_" not in source
    for required in (
        "resolve_dataset_root",
        "resolve_private_kernel_root",
        "read_competition_cohorts",
        "verify_artifact_identity",
        "join_three_sources",
        "count_triple_combinations",
        "partition_strata",
        "compare_pairwise",
        "compute_deep_dives",
        "build_quota_feasibility",
        "build_public_safe_summary",
        "/kaggle/input/kernels/plthon/rsna-knee-report-labels",
        "/kaggle/input/kernels/plthon/rsna-knee-report-llm-labels",
        "/kaggle/input/datasets/pilkwang/rsna-knee-llm-labels",
    ):
        assert required in source
    assert "__RSNA_KNEE_" not in source

    calls: list[list[str]] = []

    def runner(command: list[str]) -> CompletedProcess[str]:
        calls.append(command)
        return CompletedProcess(command, 0)

    with pytest.raises(PermissionError, match="explicit remote approval"):
        push_kaggle_bundle(bundle, runner=runner)
    assert calls == []


def test_source_comparison_boundary_rejects_wrong_stage_and_private_pull(
    tmp_path: Path,
) -> None:
    project = Path(__file__).parents[1]
    with pytest.raises(ValueError, match="stage must be compare"):
        build_report_source_comparison_bundle(
            project, tmp_path / "wrong-stage", stage="extract"
        )
    assert not (tmp_path / "wrong-stage").exists()

    bundle = build_report_source_comparison_bundle(
        project, tmp_path / "report-source-comparison", stage="compare"
    )
    with pytest.raises(PermissionError, match="row-level"):
        pull_kaggle_output(
            bundle,
            tmp_path / ".artifacts" / "comparison",
            artifact_root=tmp_path / ".artifacts",
            filenames=("source-comparison-tables.json",),
            runner=lambda command: CompletedProcess(command, 0),
        )
    results = pull_kaggle_output(
        bundle,
        tmp_path / ".artifacts" / "comparison",
        artifact_root=tmp_path / ".artifacts",
        filenames=("source-comparison-summary.json",),
        runner=lambda command: CompletedProcess(command, 0),
    )
    assert len(results) == 1


def test_source_comparison_cli_builds_bundle_without_remote_action(
    tmp_path: Path,
) -> None:
    project = Path(__file__).parents[1]
    output = tmp_path / "report-source-comparison"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/kaggle.py",
            "build",
            "report-source-comparison",
            "--stage",
            "compare",
            "--output",
            str(output),
        ],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == str(output)
    assert (output / "kernel" / "report_source_comparison.ipynb").is_file()


def test_combined_baseline_bundle_requires_gpu_approval(tmp_path: Path) -> None:
    from rsna_knee.kaggle import build_combined_baseline_bundle

    project = Path(__file__).parents[1]
    output = tmp_path / "combined-bundle"

    with pytest.raises(PermissionError, match="GPU approval"):
        build_combined_baseline_bundle(
            project,
            output,
            gpu_approved=False,
        )


def test_combined_baseline_bundle_builds_and_verifies_metadata(tmp_path: Path) -> None:
    from rsna_knee.kaggle import build_combined_baseline_bundle

    project = Path(__file__).parents[1]
    output = tmp_path / "combined-bundle"

    bundle = build_combined_baseline_bundle(
        project,
        output,
        gpu_approved=True,
    )

    metadata = json.loads(bundle.kernel_metadata_path.read_text())
    assert metadata["is_private"] == "true"
    assert metadata["enable_internet"] == "false"
    assert metadata["enable_gpu"] == "true"
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["competition_sources"] == ["rsna-knee-abnormality-detection"]
    assert metadata["dataset_sources"] == [
        "plthon/rsna-knee-combined-baseline-checkpoint"
    ]
    assert metadata["kernel_sources"] == []
    assert (output / "kernel" / "score_combined_baseline.ipynb").is_file()
    assert (output / "bundle-provenance.json").is_file()


def test_plane_gating_bundle_is_training_only_and_requires_gpu_approval(
    tmp_path: Path,
) -> None:
    project = Path(__file__).parents[1]
    output = tmp_path / "plane-gating"

    with pytest.raises(PermissionError, match="GPU approval"):
        build_plane_gating_bundle(project, output)

    bundle = build_plane_gating_bundle(project, output, gpu_approved=True)
    metadata = json.loads(bundle.kernel_metadata_path.read_text())
    assert metadata["enable_gpu"] == "true"
    assert metadata["dataset_sources"] == [
        "pilkwang/rsna-knee-llm-labels",
        "plthon/torchvision-resnet18-imagenet1k-v1",
        "plthon/rsna-knee-combined-baseline-checkpoint",
    ]
    assert metadata["kernel_sources"] == ["plthon/rsna-knee-data-index"]
    notebook = json.loads(bundle.notebook_path.read_text())
    source = "\n".join(
        cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        for cell in notebook["cells"]
    )
    assert "run_test_inference=False" in source
    assert "run_combined_scoring" not in source


def test_plane_gated_scoring_bundle_requires_keep_decision(tmp_path: Path) -> None:
    project = Path(__file__).parents[1]
    output = tmp_path / "plane-scoring"
    with pytest.raises(PermissionError, match="recorded keep decision"):
        build_plane_gated_scoring_bundle(
            project,
            output,
            checkpoint_dataset_id="plthon/rsna-knee-plane-gating-checkpoint",
            checkpoint_sha256="a" * 64,
            gpu_approved=True,
        )

    bundle = build_plane_gated_scoring_bundle(
        project,
        output,
        checkpoint_dataset_id="plthon/rsna-knee-plane-gating-checkpoint",
        checkpoint_sha256="a" * 64,
        validation_kept=True,
        gpu_approved=True,
    )
    metadata = json.loads(bundle.kernel_metadata_path.read_text())
    assert metadata["dataset_sources"] == ["plthon/rsna-knee-plane-gating-checkpoint"]
    source = json.dumps(json.loads(bundle.notebook_path.read_text()))
    assert "target_plane_gate" in source
    assert "Competition submission: not run" in source


def test_dinov2_training_bundle_is_clean_offline_and_training_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = Path(__file__).parents[1]
    output = tmp_path / "dinov2-training"
    weight_path = tmp_path / "dinov2_vits14_pretrain.pth"
    weight_path.write_bytes(b"verified-test-weights")
    monkeypatch.setattr("rsna_knee.kaggle._git_state", lambda _: ("a" * 40, True))
    with pytest.raises(ValueError, match="clean committed Git revision"):
        build_dinov2_training_bundle(
            project, output, weight_path=weight_path, gpu_approved=True
        )
    assert not output.exists()

    monkeypatch.setattr("rsna_knee.kaggle._git_state", lambda _: ("a" * 40, False))
    monkeypatch.setattr(
        "rsna_knee.kaggle.build_dinov2_weight_dataset",
        lambda project_root, output_root, *, weight_path: _attached_weight_dataset(
            output_root
        ),
    )
    bundle = build_dinov2_training_bundle(
        project, output, weight_path=weight_path, gpu_approved=True
    )
    assert isinstance(bundle.dataset, WeightDataset)
    assert bundle.dataset.weight_path.name == "dinov2_vits14_pretrain.pth"
    metadata = json.loads(bundle.kernel_metadata_path.read_text())
    assert metadata["enable_internet"] == "false"
    assert metadata["dataset_sources"] == [
        "pilkwang/rsna-knee-llm-labels",
        "plthon/dinov2-vits14-lvd142m",
        "plthon/rsna-knee-combined-baseline-checkpoint",
    ]
    assert metadata["kernel_sources"] == ["plthon/rsna-knee-data-index"]

    resume_bundle = build_dinov2_training_bundle(
        project,
        tmp_path / "dinov2-resume",
        weight_path=weight_path,
        gpu_approved=True,
        resume=True,
    )
    resume_metadata = json.loads(resume_bundle.kernel_metadata_path.read_text())
    assert resume_metadata["kernel_sources"] == [
        "plthon/rsna-knee-data-index",
        "plthon/rsna-knee-dinov2-vit-s-14",
    ]

    resume_dataset_bundle = build_dinov2_training_bundle(
        project,
        tmp_path / "dinov2-resume-ds",
        weight_path=weight_path,
        gpu_approved=True,
        resume_dataset_id="plthon/rsna-knee-dinov2-resume",
    )
    resume_ds_metadata = json.loads(
        resume_dataset_bundle.kernel_metadata_path.read_text()
    )
    assert resume_ds_metadata["kernel_sources"] == ["plthon/rsna-knee-data-index"]
    assert "plthon/rsna-knee-dinov2-resume" in resume_ds_metadata["dataset_sources"]

    fake_ckpt = tmp_path / "checkpoint.pt"
    fake_ckpt.write_bytes(b"dummy")
    dataset_out = build_dinov2_resume_dataset(
        tmp_path / "resume-ds-out",
        checkpoint_path=fake_ckpt,
        dataset_id="plthon/rsna-knee-dinov2-resume",
    )
    assert (dataset_out / "checkpoint.pt").read_bytes() == b"dummy"
    assert (dataset_out / "dataset-metadata.json").is_file()

    notebook = json.loads(bundle.notebook_path.read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "run_test_inference=False" in source
    assert "run_combined_scoring" not in source
    assert 'GIT_DIRTY = json.loads("false")' in source
    allocator_config = 'os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"'
    assert allocator_config in source
    assert source.index(allocator_config) < source.index(
        "from rsna_knee.combined_supervision import"
    )

    loaded = load_kaggle_bundle(output)
    assert isinstance(loaded.dataset, WeightDataset)
    calls: list[list[str]] = []

    def runner(command: list[str]) -> CompletedProcess[str]:
        calls.append(command)
        return CompletedProcess(command, 0, stdout="ready")

    push_kaggle_bundle(
        loaded,
        approved=True,
        gpu_approved=True,
        runner=runner,
        wait=lambda _: None,
    )
    assert calls == [
        ["kaggle", "datasets", "create", "-p", str(loaded.dataset.root)],
        ["kaggle", "datasets", "status", "plthon/dinov2-vits14-lvd142m"],
        ["kaggle", "kernels", "push", "-p", str(loaded.kernel_root)],
    ]


def test_dinov2_scoring_bundle_requires_retained_validation_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = Path(__file__).parents[1]
    with pytest.raises(PermissionError, match="recorded keep decision"):
        build_dinov2_scoring_bundle(
            project,
            tmp_path / "dinov2-scoring",
            checkpoint_dataset_id="plthon/rsna-knee-dinov2-checkpoint",
            checkpoint_sha256="a" * 64,
            gpu_approved=True,
        )

    monkeypatch.setattr("rsna_knee.kaggle._git_state", lambda _: ("a" * 40, True))
    with pytest.raises(ValueError, match="clean committed Git revision"):
        build_dinov2_scoring_bundle(
            project,
            tmp_path / "dinov2-scoring",
            checkpoint_dataset_id="plthon/rsna-knee-dinov2-checkpoint",
            checkpoint_sha256="a" * 64,
            validation_kept=True,
            gpu_approved=True,
        )

    monkeypatch.setattr("rsna_knee.kaggle._git_state", lambda _: ("a" * 40, False))
    bundle = build_dinov2_scoring_bundle(
        project,
        tmp_path / "dinov2-scoring",
        checkpoint_dataset_id="plthon/rsna-knee-dinov2-checkpoint",
        checkpoint_sha256="a" * 64,
        validation_kept=True,
        gpu_approved=True,
    )
    metadata = json.loads(bundle.kernel_metadata_path.read_text())
    assert metadata["dataset_sources"] == ["plthon/rsna-knee-dinov2-checkpoint"]
    source = json.dumps(json.loads(bundle.notebook_path.read_text()))
    assert 'backbone=\\"dinov2_vits14\\"' in source
    assert "Competition submission: not run" in source


def test_raptor_parity_bundle_is_clean_offline_and_gold_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = Path(__file__).parents[1]
    output = tmp_path / "raptor-parity"

    monkeypatch.setattr("rsna_knee.kaggle._git_state", lambda _: ("a" * 40, True))
    with pytest.raises(ValueError, match="clean committed Git revision"):
        build_raptor_parity_bundle(project, output, gpu_approved=True)

    monkeypatch.setattr("rsna_knee.kaggle._git_state", lambda _: ("a" * 40, False))
    with pytest.raises(PermissionError, match="GPU approval"):
        build_raptor_parity_bundle(project, output)

    bundle = build_raptor_parity_bundle(project, output, gpu_approved=True)
    metadata = json.loads(bundle.kernel_metadata_path.read_text())
    assert metadata["competition_sources"] == ["rsna-knee-abnormality-detection"]
    assert metadata["dataset_sources"] == ["dreaddevelopment/raptor-knee-widedense"]
    assert metadata["kernel_sources"] == ["plthon/rsna-knee-data-index"]
    assert metadata["enable_gpu"] == "true"
    assert metadata["enable_internet"] == "false"
    assert metadata["is_private"] == "true"

    notebook = json.loads(bundle.notebook_path.read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "derive_gold_training_cohort" in source
    assert "torch.backends.cudnn.benchmark = True" in source
    assert "torch.backends.cuda.matmul.allow_tf32 = True" in source
    assert 'COMPETITION_ROOT / "train_series"' in source
    assert 'COMPETITION_ROOT / "test_series"' not in source
    assert "write_submission" not in source
    assert "Competition submission: not run" in source
    assert "Training: not run" in source
