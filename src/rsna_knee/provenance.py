import hashlib
import importlib.metadata
import json
import os
import platform
import re
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

_RUN_ID_PART_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def create_run_id(prefix: str, *, external_id: str | None = None) -> str:
    """Create a registry-safe run ID, preserving an external version when present."""
    suffix = external_id or uuid.uuid4().hex
    if (
        _RUN_ID_PART_PATTERN.fullmatch(prefix) is None
        or _RUN_ID_PART_PATTERN.fullmatch(suffix) is None
    ):
        raise ValueError("Run ID parts may contain only letters, digits, '.', '_', '-'")
    return f"{prefix}-{suffix}"


def write_run_manifest(
    path: Path,
    *,
    run_id: str,
    source: Mapping[str, object],
    configuration: Mapping[str, object],
    notebook: Mapping[str, object],
    artifacts: Sequence[Path],
    artifact_root: Path,
    runtime: Mapping[str, object] | None = None,
) -> Path:
    """Write compact provenance for one portable pipeline run."""
    artifact_root = artifact_root.resolve()
    artifact_records: list[dict[str, object]] = []
    for artifact_path in artifacts:
        resolved_path = artifact_path.resolve()
        relative_path = resolved_path.relative_to(artifact_root)
        artifact_records.append(
            {
                "path": relative_path.as_posix(),
                "sha256": hashlib.sha256(resolved_path.read_bytes()).hexdigest(),
                "size_bytes": resolved_path.stat().st_size,
            }
        )

    runtime_record = _capture_runtime()
    for key, value in (runtime or {}).items():
        current = runtime_record.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            current.update(value)
        else:
            runtime_record[key] = value
    manifest = {
        "artifacts": artifact_records,
        "configuration": dict(configuration),
        "notebook": dict(notebook),
        "run_id": run_id,
        "runtime": runtime_record,
        "schema_version": 1,
        "source": dict(source),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def _capture_runtime() -> dict[str, object]:
    return {
        "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "cpu_count": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "machine": platform.machine(),
        "packages": {
            package: _distribution_version(package)
            for package in ("numpy", "pydicom", "rsna-knee")
        },
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def _distribution_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None
