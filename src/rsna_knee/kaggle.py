import base64
import hashlib
import json
import re
import subprocess
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

type CommandRunner = Callable[[list[str]], CompletedProcess[str]]

_PACKAGE_MARKER = "__RSNA_KNEE_PACKAGE_ARCHIVE__"
_KAGGLE_KERNEL_TITLE_MAX_LENGTH = 50
_KAGGLE_KERNEL_SLUG_MAX_LENGTH = 50
_FORBIDDEN_PUBLIC_NOTEBOOK_DETAILS = (
    ".artifacts/",
    "build/kaggle/",
    "credentials/",
    "kaggle.json",
    "rsna-knee-offline-wheel",
    "rsna_knee_source.zip",
    "run-manifest.json",
    "index-manifest.json",
    "pseudo-label",
    "pseudo_label",
    "report mapping",
    "report-language",
    "report language",
    "validation split",
    "leaderboard",
    "checkpoint",
    "attention/mil",
    "2.5d",
    "ensemble strategy",
    "calibration strateg",
    "plthon/rsna-knee-real-data-inspection",
    "plthon/rsna-knee-data-index",
)


def _kernel_title_slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-")


@dataclass(frozen=True, slots=True)
class _KernelMetadata:
    kernel_id: str
    title: str
    code_file: str
    competition_slug: str = "rsna-knee-abnormality-detection"
    kernel_sources: tuple[str, ...] = ()

    def as_json(self) -> dict[str, Any]:
        owner_and_slug = self.kernel_id.split("/", 1)
        expected_slug = _kernel_title_slug(self.title)
        if len(self.title) > _KAGGLE_KERNEL_TITLE_MAX_LENGTH:
            raise ValueError(
                "Kaggle kernel title must be no longer than "
                f"{_KAGGLE_KERNEL_TITLE_MAX_LENGTH} characters"
            )
        if len(expected_slug) > _KAGGLE_KERNEL_SLUG_MAX_LENGTH:
            raise ValueError(
                "Kaggle kernel slug must be no longer than "
                f"{_KAGGLE_KERNEL_SLUG_MAX_LENGTH} characters"
            )
        if len(owner_and_slug) != 2 or owner_and_slug[1] != expected_slug:
            raise ValueError(
                "Kaggle kernel id slug must resolve from its title: "
                f"expected {expected_slug}"
            )
        return {
            "code_file": self.code_file,
            "competition_sources": [self.competition_slug],
            "dataset_sources": [],
            "enable_gpu": "false",
            "enable_internet": "false",
            "enable_tpu": "false",
            "id": self.kernel_id,
            "is_private": "true",
            "kernel_sources": list(self.kernel_sources),
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
_SUBMISSION_METADATA = _KernelMetadata(
    kernel_id="plthon/rsna-knee-constant-baseline",
    title="RSNA Knee Constant Baseline",
    code_file="constant_baseline_submission.ipynb",
)
_GOLD_AUDIT_METADATA = _KernelMetadata(
    kernel_id="plthon/rsna-knee-gold-supervision-audit",
    title="RSNA Knee Gold Supervision Audit",
    code_file="gold_supervision_audit.ipynb",
    kernel_sources=("plthon/rsna-knee-data-index",),
)
_PUBLIC_NOTEBOOK_METADATA = _KernelMetadata(
    kernel_id="plthon/rsna-knee-mri-correct-dicom-ordering-sampling",
    title="RSNA Knee MRI | Correct DICOM Ordering & Sampling",
    code_file="rsna_knee_dicom_ordering.ipynb",
)


@dataclass(frozen=True, slots=True)
class KaggleSourceBundle:
    root: Path
    notebook_path: Path
    metadata_path: Path


type InspectionBundle = KaggleSourceBundle
type IndexBundle = KaggleSourceBundle
type SubmissionBundle = KaggleSourceBundle
type GoldAuditBundle = KaggleSourceBundle
type PublicNotebookBundle = KaggleSourceBundle


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


def build_submission_bundle(project_root: Path, output_root: Path) -> SubmissionBundle:
    """Build the private, offline constant-baseline submission notebook."""
    source_path = project_root / "notebooks" / "constant_baseline_submission.py"
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
    return _build_source_bundle(
        project_root,
        output_root,
        source_name="constant_baseline_submission.py",
        metadata=_SUBMISSION_METADATA.as_json(),
        replacements={
            "__RSNA_KNEE_GIT_COMMIT__": git_commit,
            "__RSNA_KNEE_GIT_DIRTY__": json.dumps(git_dirty),
            "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__": hashlib.sha256(
                source_path.read_bytes()
            ).hexdigest(),
        },
    )


def build_gold_audit_bundle(project_root: Path, output_root: Path) -> GoldAuditBundle:
    """Build the private, offline, CPU-only organizer-gold audit notebook."""
    source_path = project_root / "notebooks" / "gold_supervision_audit.py"
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
    return _build_source_bundle(
        project_root,
        output_root,
        source_name="gold_supervision_audit.py",
        metadata=_GOLD_AUDIT_METADATA.as_json(),
        replacements={
            "__RSNA_KNEE_GIT_COMMIT__": git_commit,
            "__RSNA_KNEE_GIT_DIRTY__": json.dumps(git_dirty),
            "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__": hashlib.sha256(
                source_path.read_bytes()
            ).hexdigest(),
        },
        package_contracts=("data-index.json",),
    )


def build_public_notebook_bundle(
    project_root: Path, output_root: Path
) -> PublicNotebookBundle:
    """Build the standalone public-facing notebook as a private review bundle."""
    source_path = project_root / "notebooks" / "public_dicom_ordering.py"
    source = source_path.read_text()
    _validate_public_notebook_source(source)

    metadata = _PUBLIC_NOTEBOOK_METADATA.as_json()
    output_root.mkdir(parents=True, exist_ok=True)
    notebook_path = output_root / str(metadata["code_file"])
    metadata_path = output_root / "kernel-metadata.json"
    notebook_path.write_text(json.dumps(_percent_source_to_notebook(source)))
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    _validate_public_notebook_bundle(output_root, metadata)
    return KaggleSourceBundle(
        root=output_root,
        notebook_path=notebook_path,
        metadata_path=metadata_path,
    )


def _validate_public_notebook_source(source: str) -> None:
    if _PACKAGE_MARKER in source or "from rsna_knee" in source:
        raise ValueError("Public notebook must not embed the private package")
    folded_source = source.casefold()
    for detail in _FORBIDDEN_PUBLIC_NOTEBOOK_DETAILS:
        if detail.casefold() in folded_source:
            raise ValueError(
                f"Notebook contains forbidden public-notebook detail: {detail}"
            )


def _validate_public_notebook_bundle(
    bundle_root: Path, expected_metadata: dict[str, Any]
) -> None:
    _validate_source_bundle(bundle_root, expected_metadata)
    notebook_path = bundle_root / str(expected_metadata["code_file"])
    notebook: dict[str, Any] = json.loads(notebook_path.read_text())
    raw_cells = notebook.get("cells")
    if not isinstance(raw_cells, list):
        raise TypeError("Public notebook has no cell list")
    source_parts: list[str] = []
    for raw_cell in raw_cells:
        if not isinstance(raw_cell, dict):
            raise TypeError("Public notebook contains a malformed cell")
        raw_source = raw_cell.get("source", [])
        if isinstance(raw_source, str):
            source_parts.append(raw_source)
        elif isinstance(raw_source, list) and all(
            isinstance(line, str) for line in raw_source
        ):
            source_parts.append("".join(raw_source))
        else:
            raise TypeError("Public notebook cell source is malformed")
    _validate_public_notebook_source("\n".join(source_parts))


def _build_source_bundle(
    project_root: Path,
    output_root: Path,
    *,
    source_name: str,
    metadata: dict[str, Any],
    replacements: Mapping[str, str] | None = None,
    package_contracts: tuple[str, ...] = (),
) -> KaggleSourceBundle:
    source_path = project_root / "notebooks" / source_name
    package_root = project_root / "src" / "rsna_knee"

    source = source_path.read_text()
    for marker, value in (replacements or {}).items():
        if source.count(marker) != 1:
            raise ValueError(f"Notebook source must contain exactly one {marker}")
        source = source.replace(marker, value)
    if source.count(_PACKAGE_MARKER) != 1:
        raise ValueError(
            "Notebook source must contain exactly one package archive marker"
        )
    archive = _package_archive(package_root, additional_contracts=package_contracts)
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


def push_submission_bundle(
    bundle_root: Path,
    *,
    approved: bool = False,
    runner: CommandRunner | None = None,
) -> CompletedProcess[str]:
    """Execute the constant baseline only after explicit remote approval."""
    return _push_source_bundle(
        bundle_root,
        metadata=_SUBMISSION_METADATA.as_json(),
        approved=approved,
        runner=runner,
    )


def push_gold_audit_bundle(
    bundle_root: Path,
    *,
    approved: bool = False,
    runner: CommandRunner | None = None,
) -> CompletedProcess[str]:
    """Execute the gold audit only after explicit remote CPU approval."""
    return _push_source_bundle(
        bundle_root,
        metadata=_GOLD_AUDIT_METADATA.as_json(),
        approved=approved,
        runner=runner,
    )


def push_public_notebook_bundle(
    bundle_root: Path,
    *,
    approved: bool = False,
    runner: CommandRunner | None = None,
) -> CompletedProcess[str]:
    """Execute the review notebook privately after explicit remote approval."""
    if approved:
        _validate_public_notebook_bundle(
            bundle_root, _PUBLIC_NOTEBOOK_METADATA.as_json()
        )
    return _push_source_bundle(
        bundle_root,
        metadata=_PUBLIC_NOTEBOOK_METADATA.as_json(),
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


def _package_archive(
    package_root: Path, *, additional_contracts: tuple[str, ...] = ()
) -> bytes:
    archive_buffer = BytesIO()
    with zipfile.ZipFile(
        archive_buffer, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for source_path in sorted(package_root.rglob("*.py")):
            relative_path = source_path.relative_to(package_root.parent).as_posix()
            _write_archive_entry(archive, relative_path, source_path)
        for contract_name in ("competition.json", *additional_contracts):
            contract_path = package_root.parents[1] / "contracts" / contract_name
            if not contract_path.is_file():
                raise FileNotFoundError(f"Missing package contract: {contract_path}")
            _write_archive_entry(archive, f"rsna_knee/{contract_name}", contract_path)
    return archive_buffer.getvalue()


def _write_archive_entry(
    archive: zipfile.ZipFile, archive_name: str, source_path: Path
) -> None:
    info = zipfile.ZipInfo(archive_name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, source_path.read_bytes())


def _percent_source_to_notebook(source: str) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    cell_kind: str | None = None
    cell_lines: list[str] = []

    def append_cell() -> None:
        if cell_kind is None:
            return
        cell_id = f"cell-{len(cells) + 1:03d}"
        if cell_kind == "markdown":
            rendered_lines = [_markdown_line(line) for line in cell_lines]
            cells.append(
                {
                    "cell_type": "markdown",
                    "id": cell_id,
                    "metadata": {},
                    "source": rendered_lines,
                }
            )
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "id": cell_id,
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
