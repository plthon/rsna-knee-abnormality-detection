import base64
import json
import subprocess
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

type CommandRunner = Callable[[list[str]], CompletedProcess[str]]

_PACKAGE_MARKER = "__RSNA_KNEE_PACKAGE_ARCHIVE__"


@dataclass(frozen=True, slots=True)
class _KernelMetadata:
    kernel_id: str
    title: str
    code_file: str
    competition_slug: str = "rsna-knee-abnormality-detection"

    def as_json(self) -> dict[str, Any]:
        return {
            "code_file": self.code_file,
            "competition_sources": [self.competition_slug],
            "dataset_sources": [],
            "enable_gpu": "false",
            "enable_internet": "false",
            "enable_tpu": "false",
            "id": self.kernel_id,
            "is_private": "true",
            "kernel_sources": [],
            "kernel_type": "notebook",
            "language": "python",
            "machine_shape": "",
            "model_sources": [],
            "title": self.title,
        }


_INSPECTION_METADATA = _KernelMetadata(
    kernel_id="plthon/rsna-knee-real-data-inspection",
    title="RSNA Knee Real Data Inspection",
    code_file="inspect_real_data.ipynb",
)
_INDEX_METADATA = _KernelMetadata(
    kernel_id="plthon/rsna-knee-data-index",
    title="RSNA Knee Data Index",
    code_file="index_real_data.ipynb",
)


@dataclass(frozen=True, slots=True)
class KaggleSourceBundle:
    root: Path
    notebook_path: Path
    metadata_path: Path


type InspectionBundle = KaggleSourceBundle
type IndexBundle = KaggleSourceBundle


def build_inspection_bundle(project_root: Path, output_root: Path) -> InspectionBundle:
    """Build the reviewable, offline inspection notebook and Kaggle metadata."""
    return _build_source_bundle(
        project_root,
        output_root,
        source_name="inspect_real_data.py",
        metadata=_INSPECTION_METADATA.as_json(),
    )


def build_index_bundle(project_root: Path, output_root: Path) -> IndexBundle:
    """Build the reviewable, offline real-data index notebook."""
    return _build_source_bundle(
        project_root,
        output_root,
        source_name="index_real_data.py",
        metadata=_INDEX_METADATA.as_json(),
    )


def _build_source_bundle(
    project_root: Path,
    output_root: Path,
    *,
    source_name: str,
    metadata: dict[str, Any],
) -> KaggleSourceBundle:
    source_path = project_root / "notebooks" / source_name
    package_root = project_root / "src" / "rsna_knee"

    source = source_path.read_text()
    if source.count(_PACKAGE_MARKER) != 1:
        raise ValueError(
            "Notebook source must contain exactly one package archive marker"
        )
    archive = _package_archive(package_root)
    rendered_source = source.replace(
        _PACKAGE_MARKER, base64.b64encode(archive).decode("ascii")
    )

    output_root.mkdir(parents=True, exist_ok=True)
    notebook_path = output_root / str(metadata["code_file"])
    metadata_path = output_root / "kernel-metadata.json"
    notebook_path.write_text(json.dumps(_percent_source_to_notebook(rendered_source)))
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    _validate_source_bundle(output_root, metadata)
    return KaggleSourceBundle(
        root=output_root,
        notebook_path=notebook_path,
        metadata_path=metadata_path,
    )


def push_inspection_bundle(
    bundle_root: Path,
    *,
    approved: bool = False,
    runner: CommandRunner | None = None,
) -> CompletedProcess[str]:
    """Push a validated bundle only after an explicit remote approval gate."""
    return _push_source_bundle(
        bundle_root,
        metadata=_INSPECTION_METADATA.as_json(),
        approved=approved,
        runner=runner,
    )


def push_index_bundle(
    bundle_root: Path,
    *,
    approved: bool = False,
    runner: CommandRunner | None = None,
) -> CompletedProcess[str]:
    """Push a validated index bundle only after explicit remote approval."""
    return _push_source_bundle(
        bundle_root,
        metadata=_INDEX_METADATA.as_json(),
        approved=approved,
        runner=runner,
    )


def _push_source_bundle(
    bundle_root: Path,
    *,
    metadata: dict[str, Any],
    approved: bool,
    runner: CommandRunner | None,
) -> CompletedProcess[str]:
    if not approved:
        raise PermissionError(
            "Kaggle push requires explicit remote approval; no command was executed"
        )
    _validate_source_bundle(bundle_root, metadata)
    command = ["kaggle", "kernels", "push", "-p", str(bundle_root)]
    return (runner or _run_command)(command)


def _run_command(command: list[str]) -> CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True)


def _package_archive(package_root: Path) -> bytes:
    archive_buffer = BytesIO()
    with zipfile.ZipFile(
        archive_buffer, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for source_path in sorted(package_root.rglob("*.py")):
            relative_path = source_path.relative_to(package_root.parent).as_posix()
            info = zipfile.ZipInfo(relative_path, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, source_path.read_bytes())
        contract_path = package_root.parents[1] / "contracts" / "competition.json"
        if not contract_path.is_file():
            raise FileNotFoundError(
                f"Missing canonical competition contract: {contract_path}"
            )
        info = zipfile.ZipInfo(
            "rsna_knee/competition.json", date_time=(1980, 1, 1, 0, 0, 0)
        )
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        archive.writestr(info, contract_path.read_bytes())
    return archive_buffer.getvalue()


def _percent_source_to_notebook(source: str) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    cell_kind: str | None = None
    cell_lines: list[str] = []

    def append_cell() -> None:
        if cell_kind is None:
            return
        if cell_kind == "markdown":
            rendered_lines = [_markdown_line(line) for line in cell_lines]
            cells.append(
                {"cell_type": "markdown", "metadata": {}, "source": rendered_lines}
            )
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": cell_lines.copy(),
                }
            )

    for line in source.splitlines(keepends=True):
        if line.startswith("# %%"):
            append_cell()
            cell_kind = "markdown" if "[markdown]" in line else "code"
            cell_lines = []
            continue
        if cell_kind is not None:
            cell_lines.append(line)
    append_cell()
    if not cells:
        raise ValueError("Notebook source contains no Jupytext percent cells")
    return {
        "cells": cells,
        "metadata": {
            "jupytext": {
                "formats": "py:percent",
                "text_representation": {
                    "extension": ".py",
                    "format_name": "percent",
                    "format_version": "1.3",
                    "jupytext_version": "1.17",
                },
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _markdown_line(line: str) -> str:
    if line.startswith("# "):
        return line[2:]
    if line in {"#\n", "#"}:
        return "\n" if line.endswith("\n") else ""
    return line


def _validate_source_bundle(
    bundle_root: Path, expected_metadata: dict[str, Any]
) -> None:
    metadata_path = bundle_root / "kernel-metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing Kaggle metadata: {metadata_path}")
    metadata: dict[str, Any] = json.loads(metadata_path.read_text())
    if metadata != expected_metadata:
        raise ValueError("Kaggle metadata does not match the safe source policy")
    code_file = metadata.get("code_file")
    if not isinstance(code_file, str) or not (bundle_root / code_file).is_file():
        raise FileNotFoundError("Kaggle metadata code_file does not exist")
