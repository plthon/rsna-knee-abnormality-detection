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
class _InspectionKernelMetadata:
    kernel_id: str = "plthon/rsna-knee-real-data-inspection"
    title: str = "RSNA Knee Real Data Inspection"
    code_file: str = "inspect_real_data.ipynb"
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


_INSPECTION_METADATA = _InspectionKernelMetadata()


@dataclass(frozen=True, slots=True)
class InspectionBundle:
    root: Path
    notebook_path: Path
    metadata_path: Path


def build_inspection_bundle(project_root: Path, output_root: Path) -> InspectionBundle:
    """Build the reviewable, offline inspection notebook and Kaggle metadata."""
    source_path = project_root / "notebooks" / "inspect_real_data.py"
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

    metadata = _INSPECTION_METADATA.as_json()
    output_root.mkdir(parents=True, exist_ok=True)
    notebook_path = output_root / str(metadata["code_file"])
    metadata_path = output_root / "kernel-metadata.json"
    notebook_path.write_text(json.dumps(_percent_source_to_notebook(rendered_source)))
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    _validate_bundle(output_root)
    return InspectionBundle(
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
    if not approved:
        raise PermissionError(
            "Kaggle push requires explicit remote approval; no command was executed"
        )
    _validate_bundle(bundle_root)
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


def _validate_bundle(bundle_root: Path) -> None:
    metadata_path = bundle_root / "kernel-metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing Kaggle metadata: {metadata_path}")
    metadata: dict[str, Any] = json.loads(metadata_path.read_text())
    if metadata != _INSPECTION_METADATA.as_json():
        raise ValueError("Kaggle metadata does not match the safe inspection policy")
    code_file = metadata.get("code_file")
    if not isinstance(code_file, str) or not (bundle_root / code_file).is_file():
        raise FileNotFoundError("Kaggle metadata code_file does not exist")
