import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from rsna_knee.kaggle_run import (
    build_run_bundle,
    build_wheel_dataset,
    download_and_register_run,
    push_run_bundle,
)
from rsna_knee.provenance import create_run_id, write_run_manifest


def _read_notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_wheel_dataset_records_a_pure_python_build(tmp_path: Path) -> None:
    dataset = build_wheel_dataset(
        Path.cwd(),
        tmp_path / "wheel-dataset",
        dataset_id="plthon/rsna-knee-offline-wheel",
    )

    with zipfile.ZipFile(dataset.wheel_path) as wheel:
        wheel_metadata_path = next(
            name for name in wheel.namelist() if name.endswith(".dist-info/WHEEL")
        )
        wheel_metadata = wheel.read(wheel_metadata_path).decode()
    provenance = json.loads(dataset.provenance_path.read_text())
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_dirty = (
        subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=Path.cwd(),
            check=False,
        ).returncode
        != 0
    )

    assert {
        "filename": dataset.wheel_path.name,
        "is_pure_python": "Root-Is-Purelib: true" in wheel_metadata,
        "tag": "Tag: py3-none-any" in wheel_metadata,
        "provenance": provenance,
    } == {
        "filename": "rsna_knee-0.1.0-py3-none-any.whl",
        "is_pure_python": True,
        "tag": True,
        "provenance": {
            "dataset_id": "plthon/rsna-knee-offline-wheel",
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "schema_version": 1,
            "wheel": {
                "filename": "rsna_knee-0.1.0-py3-none-any.whl",
                "sha256": hashlib.sha256(dataset.wheel_path.read_bytes()).hexdigest(),
            },
        },
    }


def test_run_bundle_renders_private_offline_cpu_notebook(tmp_path: Path) -> None:
    bundle = build_run_bundle(
        Path.cwd(),
        tmp_path / "run-bundle",
        dataset_id="plthon/rsna-knee-offline-wheel",
        kernel_id="plthon/rsna-knee-synthetic-pipeline",
    )

    dataset_metadata = json.loads(bundle.wheel_dataset.metadata_path.read_text())
    kernel_metadata = json.loads(bundle.kernel_metadata_path.read_text())
    notebook_source = _read_notebook_source(bundle.notebook_path)
    tracked_notebooks = subprocess.run(
        ["git", "ls-files", "*.ipynb"],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    assert {
        "dataset_metadata": dataset_metadata,
        "kernel_metadata": kernel_metadata,
        "offline_install": all(
            token in notebook_source
            for token in ("pip", "install", "--no-index", "--no-deps")
        ),
        "same_pipeline": "preprocess_series" in notebook_source,
        "unresolved_marker": "__RSNA_KNEE_" in notebook_source,
        "tracked_generated_notebooks": tracked_notebooks,
    } == {
        "dataset_metadata": {
            "description": "Private offline package for the RSNA Knee synthetic run.",
            "id": "plthon/rsna-knee-offline-wheel",
            "licenses": [{"name": "other"}],
            "title": "RSNA Knee Offline Wheel",
        },
        "kernel_metadata": {
            "code_file": "run_synthetic_pipeline.ipynb",
            "competition_sources": [],
            "dataset_sources": ["plthon/rsna-knee-offline-wheel"],
            "enable_gpu": "false",
            "enable_internet": "false",
            "enable_tpu": "false",
            "id": "plthon/rsna-knee-synthetic-pipeline",
            "is_private": "true",
            "kernel_sources": [],
            "kernel_type": "notebook",
            "language": "python",
            "machine_shape": "",
            "model_sources": [],
            "title": "RSNA Knee Synthetic Pipeline",
        },
        "offline_install": True,
        "same_pipeline": True,
        "unresolved_marker": False,
        "tracked_generated_notebooks": [],
    }


@pytest.mark.parametrize(
    ("dataset_id", "kernel_id"),
    [
        ("YOUR_USERNAME/rsna-knee-wheel", "plthon/rsna-knee-run"),
        ("OWNER/rsna-knee-wheel", "plthon/rsna-knee-run"),
        ("plthon/DATASET_SLUG", "plthon/rsna-knee-run"),
        ("plthon/rsna-knee-wheel", "<username>/rsna-knee-run"),
    ],
)
def test_run_bundle_rejects_unresolved_kaggle_ids(
    tmp_path: Path,
    dataset_id: str,
    kernel_id: str,
) -> None:
    with pytest.raises(ValueError, match="unresolved Kaggle placeholder"):
        build_run_bundle(
            Path.cwd(),
            tmp_path / "run-bundle",
            dataset_id=dataset_id,
            kernel_id=kernel_id,
        )

    assert not (tmp_path / "run-bundle").exists()


def test_remote_push_requires_explicit_remote_and_gpu_approvals(
    tmp_path: Path,
) -> None:
    with pytest.raises(PermissionError, match="separate GPU approval"):
        build_run_bundle(
            Path.cwd(),
            tmp_path / "gpu-run-bundle",
            dataset_id="plthon/rsna-knee-offline-wheel",
            kernel_id="plthon/rsna-knee-synthetic-pipeline",
            enable_gpu=True,
        )
    assert not (tmp_path / "gpu-run-bundle").exists()

    bundle = build_run_bundle(
        Path.cwd(),
        tmp_path / "run-bundle",
        dataset_id="plthon/rsna-knee-offline-wheel",
        kernel_id="plthon/rsna-knee-synthetic-pipeline",
    )
    calls: list[list[str]] = []

    def record_call(command: list[str]) -> CompletedProcess[str]:
        calls.append(command)
        return CompletedProcess(command, 0)

    with pytest.raises(PermissionError, match="explicit remote approval"):
        push_run_bundle(bundle, runner=record_call)

    metadata = json.loads(bundle.kernel_metadata_path.read_text())
    metadata["enable_gpu"] = "true"
    bundle.kernel_metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(PermissionError, match="separate GPU approval"):
        push_run_bundle(bundle, approved=True, runner=record_call)

    push_run_bundle(
        bundle,
        approved=True,
        gpu_approved=True,
        runner=record_call,
    )

    assert calls == [
        ["kaggle", "datasets", "create", "-p", str(bundle.wheel_dataset.root)],
        ["kaggle", "kernels", "push", "-p", str(bundle.kernel_root)],
    ]


def test_run_manifest_records_source_config_runtime_notebook_and_artifacts(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "synthetic-output.npz"
    artifact_path.write_bytes(b"abc")
    manifest_path = write_run_manifest(
        tmp_path / "run-manifest.json",
        run_id="synthetic-deadbeef1234",
        source={
            "git_commit": "deadbeef",
            "wheel": {"filename": "rsna_knee.whl", "sha256": "wheel-sha"},
        },
        configuration={"slice_count": 4, "config_sha256": "config-sha"},
        notebook={
            "dataset_id": "plthon/rsna-knee-offline-wheel",
            "kernel_id": "plthon/rsna-knee-synthetic-pipeline",
        },
        artifacts=[artifact_path],
        artifact_root=tmp_path,
    )

    manifest = json.loads(manifest_path.read_text())

    assert set(manifest) == {
        "artifacts",
        "configuration",
        "notebook",
        "run_id",
        "runtime",
        "schema_version",
        "source",
    }
    assert manifest["source"] == {
        "git_commit": "deadbeef",
        "wheel": {"filename": "rsna_knee.whl", "sha256": "wheel-sha"},
    }
    assert manifest["configuration"] == {
        "slice_count": 4,
        "config_sha256": "config-sha",
    }
    assert manifest["notebook"] == {
        "dataset_id": "plthon/rsna-knee-offline-wheel",
        "kernel_id": "plthon/rsna-knee-synthetic-pipeline",
    }
    assert manifest["artifacts"] == [
        {
            "path": "synthetic-output.npz",
            "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "size_bytes": 3,
        }
    ]
    assert manifest["run_id"] == "synthetic-deadbeef1234"
    assert manifest["schema_version"] == 1
    assert set(manifest["runtime"]) == {
        "captured_at_utc",
        "cpu_count",
        "cuda_visible_devices",
        "machine",
        "packages",
        "platform",
        "python",
    }


def test_run_ids_preserve_kaggle_versions_and_avoid_local_collisions() -> None:
    assert create_run_id("synthetic", external_id="version-42") == (
        "synthetic-version-42"
    )
    assert create_run_id("synthetic") != create_run_id("synthetic")


def test_rendered_notebook_writes_build_and_artifact_provenance(
    tmp_path: Path,
) -> None:
    bundle = build_run_bundle(
        Path.cwd(),
        tmp_path / "run-bundle",
        dataset_id="plthon/rsna-knee-offline-wheel",
        kernel_id="plthon/rsna-knee-synthetic-pipeline",
    )
    wheel_provenance = json.loads(bundle.wheel_dataset.provenance_path.read_text())
    config_digest = hashlib.sha256(
        bundle.wheel_dataset.config_path.read_bytes()
    ).hexdigest()
    source_digest = hashlib.sha256(
        (Path.cwd() / "notebooks" / "run_synthetic_pipeline.py").read_bytes()
    ).hexdigest()
    notebook_source = _read_notebook_source(bundle.notebook_path)

    assert all(
        evidence in notebook_source
        for evidence in (
            "write_run_manifest",
            '"run-manifest.json"',
            '"synthetic-output.npz"',
            wheel_provenance["git_commit"],
            wheel_provenance["wheel"]["sha256"],
            config_digest,
            source_digest,
            "plthon/rsna-knee-offline-wheel",
            "plthon/rsna-knee-synthetic-pipeline",
            '"kaggle_image_git_commit"',
            '"kernel_run_id"',
        )
    )


def test_selected_kernel_output_is_verified_and_registered_locally(
    tmp_path: Path,
) -> None:
    download_root = tmp_path / "download"
    registry_root = tmp_path / "registry"
    calls: list[list[str]] = []

    def download_output(command: list[str]) -> CompletedProcess[str]:
        calls.append(command)
        output_root = download_root / "rsna-knee-run"
        output_root.mkdir(parents=True)
        (output_root / "synthetic-output.npz").write_bytes(b"abc")
        (output_root / "run-manifest.json").write_text(
            json.dumps(
                {
                    "artifacts": [
                        {
                            "path": "synthetic-output.npz",
                            "sha256": (
                                "ba7816bf8f01cfea414140de5dae2223"
                                "b00361a396177a9cb410ff61f20015ad"
                            ),
                            "size_bytes": 3,
                        }
                    ],
                    "configuration": {"slice_count": 4},
                    "notebook": {"kernel_id": "plthon/rsna-knee-synthetic-pipeline"},
                    "run_id": "synthetic-deadbeef1234",
                    "runtime": {"python": "3.12.0"},
                    "schema_version": 1,
                    "source": {"git_commit": "deadbeef"},
                }
            )
        )
        return CompletedProcess(command, 0)

    registered = download_and_register_run(
        kernel_id="plthon/rsna-knee-synthetic-pipeline",
        download_root=download_root,
        registry_root=registry_root,
        selected_artifacts=["synthetic-output.npz"],
        runner=download_output,
    )

    registry_entries = [
        json.loads(line) for line in registered.registry_path.read_text().splitlines()
    ]
    assert calls == [
        [
            "kaggle",
            "kernels",
            "output",
            "plthon/rsna-knee-synthetic-pipeline",
            "-p",
            str(download_root),
        ]
    ]
    assert (registered.root / "run-manifest.json").is_file()
    assert (registered.root / "synthetic-output.npz").read_bytes() == b"abc"
    assert registry_entries == [
        {
            "artifacts": ["runs/synthetic-deadbeef1234/synthetic-output.npz"],
            "kernel_id": "plthon/rsna-knee-synthetic-pipeline",
            "manifest": "runs/synthetic-deadbeef1234/run-manifest.json",
            "run_id": "synthetic-deadbeef1234",
        }
    ]
