import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess

from .kaggle import _percent_source_to_notebook

type CommandRunner = Callable[[list[str]], CompletedProcess[str]]

_KAGGLE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_-]*")
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_PLACEHOLDER_PARTS = {
    "DATASET_ID",
    "DATASET_SLUG",
    "INSERT_SLUG_HERE",
    "INSERT_USERNAME_HERE",
    "KERNEL_ID",
    "KERNEL_SLUG",
    "OWNER",
    "USERNAME",
    "YOUR_USERNAME",
}


@dataclass(frozen=True, slots=True)
class WheelDataset:
    root: Path
    wheel_path: Path
    provenance_path: Path
    metadata_path: Path
    config_path: Path


@dataclass(frozen=True, slots=True)
class KaggleRunBundle:
    root: Path
    wheel_dataset: WheelDataset
    kernel_root: Path
    notebook_path: Path
    kernel_metadata_path: Path


@dataclass(frozen=True, slots=True)
class RegisteredRun:
    run_id: str
    root: Path
    manifest_path: Path
    artifact_paths: tuple[Path, ...]
    registry_path: Path


def _dataset_metadata(dataset_id: str) -> dict[str, object]:
    return {
        "description": "Private offline package for the RSNA Knee synthetic run.",
        "id": dataset_id,
        "licenses": [{"name": "other"}],
        "title": "RSNA Knee Offline Wheel",
    }


def _kernel_metadata(
    dataset_id: str,
    kernel_id: str,
    *,
    enable_gpu: bool,
) -> dict[str, object]:
    return {
        "code_file": "run_synthetic_pipeline.ipynb",
        "competition_sources": [],
        "dataset_sources": [dataset_id],
        "enable_gpu": str(enable_gpu).lower(),
        "enable_internet": "false",
        "enable_tpu": "false",
        "id": kernel_id,
        "is_private": "true",
        "kernel_sources": [],
        "kernel_type": "notebook",
        "language": "python",
        "machine_shape": "",
        "model_sources": [],
        "title": "RSNA Knee Synthetic Pipeline",
    }


def build_wheel_dataset(
    project_root: Path,
    output_root: Path,
    *,
    dataset_id: str,
) -> WheelDataset:
    """Build the project wheel and record its local source provenance."""
    _require_resolved_kaggle_id(dataset_id)
    output_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "uv",
            "build",
            "--wheel",
            "--out-dir",
            str(output_root),
            str(project_root),
        ],
        check=True,
        text=True,
    )
    wheels = list(output_root.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected one wheel, found {len(wheels)}")
    wheel_path = wheels[0]
    _require_pure_python_wheel(wheel_path)

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    provenance_path = output_root / "wheel-provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "git_commit": git_commit,
                "git_dirty": git_dirty,
                "schema_version": 1,
                "wheel": {
                    "filename": wheel_path.name,
                    "sha256": hashlib.sha256(wheel_path.read_bytes()).hexdigest(),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    metadata_path = output_root / "dataset-metadata.json"
    metadata_path.write_text(
        json.dumps(_dataset_metadata(dataset_id), indent=2, sort_keys=True) + "\n"
    )
    config_path = output_root / "kaggle-synthetic.toml"
    shutil.copyfile(project_root / "configs" / config_path.name, config_path)
    return WheelDataset(
        root=output_root,
        wheel_path=wheel_path,
        provenance_path=provenance_path,
        metadata_path=metadata_path,
        config_path=config_path,
    )


def build_run_bundle(
    project_root: Path,
    output_root: Path,
    *,
    dataset_id: str,
    kernel_id: str,
    enable_gpu: bool = False,
    gpu_approved: bool = False,
) -> KaggleRunBundle:
    """Build an offline wheel Dataset and its private Kaggle notebook."""
    _require_resolved_kaggle_id(dataset_id)
    _require_resolved_kaggle_id(kernel_id)
    if enable_gpu and not gpu_approved:
        raise PermissionError(
            "GPU metadata requires a separate GPU approval; no bundle was written"
        )
    wheel_dataset = build_wheel_dataset(
        project_root,
        output_root / "wheel-dataset",
        dataset_id=dataset_id,
    )
    kernel_root = output_root / "kernel"
    kernel_root.mkdir(parents=True, exist_ok=True)
    kernel_metadata = _kernel_metadata(
        dataset_id,
        kernel_id,
        enable_gpu=enable_gpu,
    )
    kernel_metadata_path = kernel_root / "kernel-metadata.json"
    kernel_metadata_path.write_text(
        json.dumps(kernel_metadata, indent=2, sort_keys=True) + "\n"
    )

    wheel_provenance = json.loads(wheel_dataset.provenance_path.read_text())
    wheel_digest = str(wheel_provenance["wheel"]["sha256"])
    config_digest = hashlib.sha256(wheel_dataset.config_path.read_bytes()).hexdigest()
    jupytext_source_path = project_root / "notebooks" / "run_synthetic_pipeline.py"
    source = jupytext_source_path.read_text()
    replacements = {
        "__RSNA_KNEE_CONFIG_SHA256__": config_digest,
        "__RSNA_KNEE_DATASET_ID__": dataset_id,
        "__RSNA_KNEE_DATASET_SLUG__": dataset_id.partition("/")[2],
        "__RSNA_KNEE_GIT_COMMIT__": str(wheel_provenance["git_commit"]),
        "__RSNA_KNEE_GPU_ENABLED__": str(enable_gpu).lower(),
        "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__": hashlib.sha256(
            jupytext_source_path.read_bytes()
        ).hexdigest(),
        "__RSNA_KNEE_KERNEL_ID__": kernel_id,
        "__RSNA_KNEE_WHEEL_FILENAME__": wheel_dataset.wheel_path.name,
        "__RSNA_KNEE_WHEEL_SHA256__": wheel_digest,
    }
    for marker, value in replacements.items():
        source = source.replace(marker, value)
    notebook_path = kernel_root / str(kernel_metadata["code_file"])
    notebook_path.write_text(json.dumps(_percent_source_to_notebook(source)))
    return KaggleRunBundle(
        root=output_root,
        wheel_dataset=wheel_dataset,
        kernel_root=kernel_root,
        notebook_path=notebook_path,
        kernel_metadata_path=kernel_metadata_path,
    )


def load_run_bundle(root: Path) -> KaggleRunBundle:
    """Load a previously built bundle for guarded pushing."""
    wheel_dataset_root = root / "wheel-dataset"
    wheels = list(wheel_dataset_root.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected one wheel, found {len(wheels)}")
    wheel_dataset = WheelDataset(
        root=wheel_dataset_root,
        wheel_path=wheels[0],
        provenance_path=wheel_dataset_root / "wheel-provenance.json",
        metadata_path=wheel_dataset_root / "dataset-metadata.json",
        config_path=wheel_dataset_root / "kaggle-synthetic.toml",
    )
    kernel_root = root / "kernel"
    return KaggleRunBundle(
        root=root,
        wheel_dataset=wheel_dataset,
        kernel_root=kernel_root,
        notebook_path=kernel_root / "run_synthetic_pipeline.ipynb",
        kernel_metadata_path=kernel_root / "kernel-metadata.json",
    )


def push_run_bundle(
    bundle: KaggleRunBundle,
    *,
    approved: bool = False,
    gpu_approved: bool = False,
    runner: CommandRunner | None = None,
) -> tuple[CompletedProcess[str], CompletedProcess[str]]:
    """Create the private Dataset and execute its notebook after approval."""
    if not approved:
        raise PermissionError(
            "Kaggle pushes require explicit remote approval; no command was executed"
        )
    gpu_enabled = _validate_run_bundle(bundle)
    if gpu_enabled and not gpu_approved:
        raise PermissionError(
            "GPU metadata requires a separate GPU approval; no command was executed"
        )

    run_command = runner or _run_command
    dataset_result = run_command(
        ["kaggle", "datasets", "create", "-p", str(bundle.wheel_dataset.root)]
    )
    kernel_result = run_command(
        ["kaggle", "kernels", "push", "-p", str(bundle.kernel_root)]
    )
    return dataset_result, kernel_result


def download_and_register_run(
    *,
    kernel_id: str,
    download_root: Path,
    registry_root: Path,
    selected_artifacts: Sequence[str],
    runner: CommandRunner | None = None,
) -> RegisteredRun:
    """Download kernel outputs and register selected, verified artifacts locally."""
    _require_resolved_kaggle_id(kernel_id)
    download_root.mkdir(parents=True, exist_ok=True)
    run_command = runner or _run_command
    run_command(
        [
            "kaggle",
            "kernels",
            "output",
            kernel_id,
            "-p",
            str(download_root),
        ]
    )

    manifests = list(download_root.rglob("run-manifest.json"))
    if len(manifests) != 1:
        raise ValueError(
            f"Expected one downloaded run manifest, found {len(manifests)}"
        )
    source_manifest_path = manifests[0]
    manifest: dict[str, object] = json.loads(source_manifest_path.read_text())
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("Run manifest contains an invalid run_id")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported run manifest schema")
    notebook = manifest.get("notebook")
    if not isinstance(notebook, dict) or notebook.get("kernel_id") != kernel_id:
        raise ValueError("Run manifest kernel ID does not match the download request")
    artifact_records = _artifact_records_by_path(manifest)

    relative_run_root = Path("runs") / run_id
    registered_root = registry_root / relative_run_root
    registered_root.mkdir(parents=True, exist_ok=True)
    registered_manifest_path = registered_root / source_manifest_path.name
    shutil.copy2(source_manifest_path, registered_manifest_path)

    registered_artifacts: list[Path] = []
    for selected_path in selected_artifacts:
        record = artifact_records.get(selected_path)
        if record is None:
            raise ValueError(
                f"Selected artifact is not recorded in manifest: {selected_path}"
            )
        source_path = (source_manifest_path.parent / selected_path).resolve()
        source_path.relative_to(source_manifest_path.parent.resolve())
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing selected kernel output: {selected_path}")
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if (
            record["sha256"] != digest
            or record["size_bytes"] != source_path.stat().st_size
        ):
            raise ValueError(
                f"Selected artifact does not match manifest: {selected_path}"
            )
        destination_path = registered_root / selected_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        registered_artifacts.append(destination_path)

    registry_path = registry_root / "registry.jsonl"
    entry: dict[str, object] = {
        "artifacts": [
            (relative_run_root / path.relative_to(registered_root)).as_posix()
            for path in registered_artifacts
        ],
        "kernel_id": kernel_id,
        "manifest": (relative_run_root / registered_manifest_path.name).as_posix(),
        "run_id": run_id,
    }
    _append_registry_entry(registry_path, entry)
    return RegisteredRun(
        run_id=run_id,
        root=registered_root,
        manifest_path=registered_manifest_path,
        artifact_paths=tuple(registered_artifacts),
        registry_path=registry_path,
    )


def _run_command(command: list[str]) -> CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True)


def _artifact_records_by_path(
    manifest: dict[str, object],
) -> dict[str, dict[str, object]]:
    values = manifest.get("artifacts")
    if not isinstance(values, list):
        raise TypeError("Run manifest artifacts must be a list")
    records: dict[str, dict[str, object]] = {}
    for value in values:
        if not isinstance(value, dict):
            raise TypeError("Run manifest contains an invalid artifact record")
        path = value.get("path")
        sha256 = value.get("sha256")
        size_bytes = value.get("size_bytes")
        if (
            not isinstance(path, str)
            or not isinstance(sha256, str)
            or not isinstance(size_bytes, int)
        ):
            raise TypeError("Run manifest contains an invalid artifact record")
        records[path] = {
            "sha256": sha256,
            "size_bytes": size_bytes,
        }
    return records


def _append_registry_entry(path: Path, entry: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_entries = (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )
    matching_entries = [
        existing
        for existing in existing_entries
        if existing.get("run_id") == entry["run_id"]
    ]
    if matching_entries:
        if matching_entries != [entry]:
            raise ValueError(
                f"Registry already contains a different run: {entry['run_id']}"
            )
        return
    with path.open("a") as registry:
        registry.write(json.dumps(entry, sort_keys=True) + "\n")


def _require_resolved_kaggle_id(value: str) -> None:
    parts = value.split("/")
    normalized_parts = {part.upper() for part in parts}
    if (
        any(character in value for character in "<>{}")
        or normalized_parts & _PLACEHOLDER_PARTS
    ):
        raise ValueError(
            f"Kaggle ID contains an unresolved Kaggle placeholder: {value}"
        )
    if _KAGGLE_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"Invalid Kaggle ID; expected owner/slug: {value}")


def _validate_run_bundle(bundle: KaggleRunBundle) -> bool:
    dataset_metadata = json.loads(bundle.wheel_dataset.metadata_path.read_text())
    dataset_id = dataset_metadata.get("id")
    if not isinstance(dataset_id, str):
        raise TypeError("Dataset metadata must contain an ID")
    _require_resolved_kaggle_id(dataset_id)
    expected_dataset_metadata = _dataset_metadata(dataset_id)
    if dataset_metadata != expected_dataset_metadata:
        raise ValueError("Dataset metadata does not match the private wheel policy")

    kernel_metadata = json.loads(bundle.kernel_metadata_path.read_text())
    kernel_id = kernel_metadata.get("id")
    if not isinstance(kernel_id, str):
        raise TypeError("Kernel metadata must contain an ID")
    _require_resolved_kaggle_id(kernel_id)
    gpu_value = kernel_metadata.get("enable_gpu")
    if not isinstance(gpu_value, str) or gpu_value not in {"false", "true"}:
        raise ValueError("Kernel GPU metadata must be true or false")
    expected_kernel_metadata = _kernel_metadata(
        dataset_id,
        kernel_id,
        enable_gpu=gpu_value == "true",
    )
    if kernel_metadata != expected_kernel_metadata:
        raise ValueError("Kernel metadata does not match the private offline policy")
    code_file = bundle.kernel_root / str(kernel_metadata["code_file"])
    if not code_file.is_file():
        raise FileNotFoundError(f"Missing rendered notebook: {code_file}")
    if "__RSNA_KNEE_" in code_file.read_text():
        raise ValueError("Rendered notebook contains an unresolved Kaggle placeholder")

    provenance = json.loads(bundle.wheel_dataset.provenance_path.read_text())
    if provenance.get("dataset_id") != dataset_id:
        raise ValueError("Wheel provenance Dataset ID does not match metadata")
    expected_digest = hashlib.sha256(
        bundle.wheel_dataset.wheel_path.read_bytes()
    ).hexdigest()
    if provenance.get("wheel", {}).get("sha256") != expected_digest:
        raise ValueError("Wheel digest does not match recorded provenance")
    _require_pure_python_wheel(bundle.wheel_dataset.wheel_path)
    return gpu_value == "true"


def _require_pure_python_wheel(wheel_path: Path) -> None:
    with zipfile.ZipFile(wheel_path) as wheel:
        metadata_paths = [
            name for name in wheel.namelist() if name.endswith(".dist-info/WHEEL")
        ]
        if len(metadata_paths) != 1:
            raise ValueError("Wheel must contain exactly one WHEEL metadata file")
        metadata = wheel.read(metadata_paths[0]).decode()
    if "Root-Is-Purelib: true" not in metadata or "Tag: py3-none-any" not in metadata:
        raise ValueError("Kaggle package must be a pure Python py3-none-any wheel")
