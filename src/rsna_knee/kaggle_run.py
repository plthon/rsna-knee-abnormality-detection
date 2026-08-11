import hashlib
import json
import re
import shutil
import subprocess
import time
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess

from .kaggle import _percent_source_to_notebook

type CommandRunner = Callable[[list[str]], CompletedProcess[str]]
type Waiter = Callable[[float], object]

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
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
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
    return render_run_kernel(
        project_root,
        output_root,
        wheel_dataset=wheel_dataset,
        kernel_id=kernel_id,
        enable_gpu=enable_gpu,
        gpu_approved=gpu_approved,
    )


def render_run_kernel(
    project_root: Path,
    output_root: Path,
    *,
    wheel_dataset: WheelDataset,
    kernel_id: str,
    enable_gpu: bool = False,
    gpu_approved: bool = False,
) -> KaggleRunBundle:
    """Render a kernel while preserving an existing wheel Dataset."""
    _require_resolved_kaggle_id(kernel_id)
    if enable_gpu and not gpu_approved:
        raise PermissionError(
            "GPU metadata requires a separate GPU approval; no bundle was written"
        )
    dataset_metadata = json.loads(wheel_dataset.metadata_path.read_text())
    dataset_id = dataset_metadata.get("id")
    if not isinstance(dataset_id, str):
        raise TypeError("Dataset metadata must contain an ID")
    _require_resolved_kaggle_id(dataset_id)
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
    reuse_ready_dataset: bool = False,
    runner: CommandRunner | None = None,
    wait: Waiter | None = None,
) -> tuple[CompletedProcess[str] | None, CompletedProcess[str]]:
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
    status_command_runner = runner or _run_status_command
    dataset_result = None
    if not reuse_ready_dataset:
        dataset_result = run_command(
            ["kaggle", "datasets", "create", "-p", str(bundle.wheel_dataset.root)]
        )
    _wait_for_dataset_ready(
        bundle,
        runner=status_command_runner,
        wait=wait or time.sleep,
    )
    kernel_result = run_command(
        ["kaggle", "kernels", "push", "-p", str(bundle.kernel_root)]
    )
    return dataset_result, kernel_result


def _wait_for_dataset_ready(
    bundle: KaggleRunBundle,
    *,
    runner: CommandRunner,
    wait: Waiter,
    attempts: int = 60,
    interval_seconds: float = 2.0,
) -> None:
    metadata = json.loads(bundle.wheel_dataset.metadata_path.read_text())
    dataset_id = metadata["id"]
    if not isinstance(dataset_id, str):
        raise TypeError("Dataset metadata must contain an ID")
    status_command = ["kaggle", "datasets", "status", dataset_id]
    last_status = "unavailable"
    for attempt in range(attempts):
        try:
            result = runner(status_command)
        except CalledProcessError:
            last_status = "unavailable"
        else:
            last_status = (result.stdout or "").strip().lower()
            if last_status == "ready":
                return
            if last_status in {"error", "failed", "failure"}:
                raise RuntimeError(
                    f"Kaggle Dataset entered terminal status: {last_status}"
                )
        if attempt < attempts - 1:
            wait(interval_seconds)
    raise TimeoutError(
        f"Kaggle Dataset was not ready after {attempts} checks: {last_status}"
    )


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
    if len(set(selected_artifacts)) != len(selected_artifacts):
        raise ValueError("Selected artifacts must not contain duplicate paths")

    relative_run_root = Path("runs") / run_id
    registered_root = registry_root / relative_run_root
    registered_manifest_path = registered_root / source_manifest_path.name

    artifact_sources: list[tuple[Path, Path]] = []
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
        artifact_sources.append((source_path, registered_root / selected_path))

    registry_path = registry_root / "registry.jsonl"
    entry: dict[str, object] = {
        "artifacts": [
            (relative_run_root / destination.relative_to(registered_root)).as_posix()
            for _, destination in artifact_sources
        ],
        "kernel_id": kernel_id,
        "manifest": (relative_run_root / registered_manifest_path.name).as_posix(),
        "run_id": run_id,
    }
    existing_entry = _registry_entry_for_run(registry_path, run_id)
    if existing_entry is not None:
        if existing_entry != entry or not _registered_run_matches(
            source_manifest_path,
            registered_manifest_path,
            artifact_sources,
        ):
            raise ValueError(f"Registry already contains a different run: {run_id}")
        return RegisteredRun(
            run_id=run_id,
            root=registered_root,
            manifest_path=registered_manifest_path,
            artifact_paths=tuple(destination for _, destination in artifact_sources),
            registry_path=registry_path,
        )
    if registered_root.exists():
        raise ValueError(
            f"Run directory already exists without a registry entry: {run_id}"
        )

    registered_root.mkdir(parents=True)
    shutil.copy2(source_manifest_path, registered_manifest_path)
    for source_path, destination_path in artifact_sources:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
    _append_registry_entry(registry_path, entry)
    return RegisteredRun(
        run_id=run_id,
        root=registered_root,
        manifest_path=registered_manifest_path,
        artifact_paths=tuple(destination for _, destination in artifact_sources),
        registry_path=registry_path,
    )


def _run_command(command: list[str]) -> CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True)


def _run_status_command(command: list[str]) -> CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


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
        if path in records:
            raise ValueError(f"Run manifest contains a duplicate artifact path: {path}")
        records[path] = {
            "sha256": sha256,
            "size_bytes": size_bytes,
        }
    return records


def _registry_entry_for_run(path: Path, run_id: str) -> dict[str, object] | None:
    if not path.exists():
        return None
    matching_entries = [
        entry
        for entry in (json.loads(line) for line in path.read_text().splitlines())
        if isinstance(entry, dict) and entry.get("run_id") == run_id
    ]
    if len(matching_entries) > 1:
        raise ValueError(f"Registry contains duplicate run entries: {run_id}")
    return matching_entries[0] if matching_entries else None


def _registered_run_matches(
    source_manifest_path: Path,
    registered_manifest_path: Path,
    artifact_sources: Sequence[tuple[Path, Path]],
) -> bool:
    if not registered_manifest_path.is_file() or (
        registered_manifest_path.read_bytes() != source_manifest_path.read_bytes()
    ):
        return False
    return all(
        destination.is_file() and destination.read_bytes() == source.read_bytes()
        for source, destination in artifact_sources
    )


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
    wheel_record = provenance.get("wheel")
    if not isinstance(wheel_record, dict) or (
        wheel_record.get("filename") != bundle.wheel_dataset.wheel_path.name
    ):
        raise ValueError("Wheel filename does not match recorded provenance")
    if wheel_record.get("sha256") != expected_digest:
        raise ValueError("Wheel digest does not match recorded provenance")
    notebook_source = code_file.read_text()
    config_digest = hashlib.sha256(
        bundle.wheel_dataset.config_path.read_bytes()
    ).hexdigest()
    if config_digest not in notebook_source:
        raise ValueError("Configuration digest does not match the rendered notebook")
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
