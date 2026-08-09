import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Literal, cast

from .inspection_imaging import (
    ImagingInspection,
    SeriesInventory,
    inspect_imaging_sample,
)
from .inspection_runtime import capture_runtime

type AssumptionStatus = Literal["verified", "rejected", "unresolved"]
type IdentifierRole = Literal["study", "series", "image", "other"]

_DICOM_SUFFIXES = {".dcm", ".dicom"}
_RASTER_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
_REPORT_COLUMN_NAMES = {"findings", "impression", "report", "report_text", "text"}
_IMAGE_PATH_SAMPLE_LIMIT = 24
_PROVISIONAL_DATASET_SIZE_BYTES = 600_000_000_000
_DATASET_SIZE_RELATIVE_TOLERANCE = 0.25


@dataclass(frozen=True, slots=True)
class InspectionOutcome:
    artifact_path: Path
    findings: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _CsvSummary:
    columns: list[str]
    row_count: int
    sampled_values: dict[str, set[str]]
    sampled_study_series_pairs: set[tuple[str, str]]
    identifier_rows: set[tuple[str, ...]]
    duplicate_identifier_rows: int
    missing_identifier_rows: int


@dataclass(frozen=True, slots=True)
class _Inventory:
    counts: dict[str, int]
    paths: dict[str, list[str]]
    complete: dict[str, bool]
    image_roots: dict[str, int]
    series: list[SeriesInventory]
    total_bytes: int


@dataclass(frozen=True, slots=True)
class _ReportSummary:
    paths: list[str]
    structures: dict[str, list[str]]
    language_values: list[str]

    @property
    def available(self) -> bool:
        return bool(self.paths)

    def as_json(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "language_values": self.language_values,
            "paths": self.paths,
            "structures": self.structures,
            "text_exported": False,
        }


@dataclass(frozen=True, slots=True)
class _HierarchySummary:
    image_count: int
    images_per_series: dict[str, int | float | None]
    relationships_verified: bool
    series_count: int
    series_per_study: dict[str, int | float | None]
    study_count: int
    study_ids: set[str]
    series_ids: set[str]
    study_series_pairs: set[tuple[str, str]]
    identifier_samples: list[dict[str, str]]

    def as_json(self) -> dict[str, Any]:
        return {
            "identifier_samples": self.identifier_samples,
            "image_count": self.image_count,
            "images_per_series": self.images_per_series,
            "relationships_verified": self.relationships_verified,
            "series_count": self.series_count,
            "series_per_study": self.series_per_study,
            "study_count": self.study_count,
        }


def inspect_competition(input_root: Path, work_root: Path) -> InspectionOutcome:
    """Inspect a bounded real-data sample and persist a compact JSON contract."""
    competition_root = _discover_competition_root(input_root)

    inventory_started = time.perf_counter()
    inventory = _inventory_files(competition_root)
    csv_paths = sorted(competition_root / path for path in inventory.paths["csv"])
    csv_summaries = {path: _summarize_csv(path) for path in csv_paths}
    inventory_seconds = time.perf_counter() - inventory_started

    sample_submission_path = competition_root / "sample_submission.csv"
    if sample_submission_path not in csv_summaries:
        raise FileNotFoundError("Competition root has no sample_submission.csv")
    sample_submission = csv_summaries[sample_submission_path]
    submission_identifier_columns = _identifier_columns(sample_submission.columns)
    targets = [
        column
        for column in sample_submission.columns
        if column not in submission_identifier_columns
    ]

    artifact_dir = work_root / "rsna-knee-inspection"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    imaging = inspect_imaging_sample(inventory.series, competition_root, artifact_dir)
    reports = _summarize_reports(
        competition_root, inventory.paths["report"], csv_summaries
    )
    hierarchy = _summarize_path_hierarchy(inventory.series)
    table_relationships = _summarize_table_relationships(
        competition_root, csv_summaries, hierarchy
    )
    runtime = capture_runtime(work_root)
    assumptions = _classify_assumptions(
        total_bytes=inventory.total_bytes,
        targets=targets,
        submission_identifier_columns=submission_identifier_columns,
        sample_submission=sample_submission,
        csv_summaries=csv_summaries,
        sample_submission_path=sample_submission_path,
        hierarchy=hierarchy,
        table_relationships=table_relationships,
        reports=reports,
        imaging=imaging,
        runtime=runtime,
    )

    inventory_paths = {kind: list(paths) for kind, paths in inventory.paths.items()}
    inventory_paths["report"] = reports.paths
    inventory_counts = dict(inventory.counts)
    inventory_counts["report"] = len(reports.paths)
    findings: dict[str, Any] = {
        "schema_version": 2,
        "dataset": {
            "provisional_size_bytes": _PROVISIONAL_DATASET_SIZE_BYTES,
            "provisional_size_relative_tolerance": _DATASET_SIZE_RELATIVE_TOLERANCE,
            "root": str(competition_root),
            "root_discovery": "unique sample_submission.csv below input_root",
            "total_bytes": inventory.total_bytes,
        },
        "inventory": {
            "complete": inventory.complete,
            "counts": inventory_counts,
            "image_roots": inventory.image_roots,
            "paths": inventory_paths,
        },
        "targets": {
            "count": len(targets),
            "label_representation": _target_label_representation(
                csv_summaries, targets, sample_submission_path
            ),
            "names": targets,
            "ordering_source": "sample_submission.csv",
        },
        "sample_submission": {
            "columns": sample_submission.columns,
            "identifier_columns": submission_identifier_columns,
            "path": "sample_submission.csv",
            "row_count": sample_submission.row_count,
        },
        "identifiers": {
            "path_hierarchy": hierarchy.as_json(),
            "table_columns": {
                path.relative_to(competition_root).as_posix(): _identifier_columns(
                    summary.columns
                )
                for path, summary in csv_summaries.items()
            },
            "table_relationships": table_relationships,
        },
        "tables": {
            path.relative_to(competition_root).as_posix(): {
                "columns": summary.columns,
                "identifier_columns": _identifier_columns(summary.columns),
                "row_count": summary.row_count,
            }
            for path, summary in csv_summaries.items()
        },
        "dicom": imaging.dicom,
        "preprocessing": imaging.preprocessing.as_json(),
        "reports": reports.as_json(),
        "runtime": runtime,
        "timings_seconds": {
            "header_indexing": round(imaging.header_seconds, 6),
            "inventory": round(inventory_seconds, 6),
            "pixel_decoding_and_preprocessing": round(imaging.preprocessing_seconds, 6),
        },
        "visualizations": {
            "paths": imaging.visualization_paths,
            "planes": imaging.visualized_planes,
            "study_ids": imaging.visualized_study_ids,
        },
        "assumptions": assumptions,
        "limits": {
            "full_corpus_pixel_decoding": False,
            "gpu_used": False,
            "plane_discovery_header_reads_per_series": 1,
            "representative_series_limit": 4,
            "representative_series_candidate_limit": 24,
            "slices_per_representative_series_limit": 256,
        },
    }

    artifact_path = artifact_dir / "inspection_findings.json"
    artifact_path.write_text(json.dumps(findings, indent=2, sort_keys=True) + "\n")
    return InspectionOutcome(artifact_path=artifact_path, findings=findings)


def _discover_competition_root(input_root: Path) -> Path:
    candidates = sorted(input_root.rglob("sample_submission.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No sample_submission.csv found below input root: {input_root}"
        )
    if len(candidates) > 1:
        candidate_roots = ", ".join(str(path.parent) for path in candidates)
        raise RuntimeError(f"Ambiguous competition roots: {candidate_roots}")
    return candidates[0].parent


def _inventory_files(root: Path) -> _Inventory:
    counts = {
        "all_files": 0,
        "csv": 0,
        "image": 0,
        "json": 0,
        "report": 0,
        "submission": 0,
    }
    paths: dict[str, list[str]] = {
        "csv": [],
        "image": [],
        "json": [],
        "report": [],
        "submission": [],
    }
    image_roots: dict[str, int] = {}
    series: list[SeriesInventory] = []
    total_bytes = 0

    for directory, child_directories, filenames in os.walk(root):
        child_directories.sort()
        filenames.sort()
        directory_path = Path(directory)
        dicom_names: list[str] = []
        for filename in filenames:
            path = directory_path / filename
            relative_path = path.relative_to(root).as_posix()
            suffix = path.suffix.casefold()
            counts["all_files"] += 1
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass

            if suffix == ".csv":
                counts["csv"] += 1
                paths["csv"].append(relative_path)
            if suffix == ".json":
                counts["json"] += 1
                paths["json"].append(relative_path)
            if "submission" in filename.casefold():
                counts["submission"] += 1
                paths["submission"].append(relative_path)
            if "report" in relative_path.casefold():
                counts["report"] += 1
                paths["report"].append(relative_path)
            if suffix in _DICOM_SUFFIXES or suffix in _RASTER_SUFFIXES:
                counts["image"] += 1
                if len(paths["image"]) < _IMAGE_PATH_SAMPLE_LIMIT:
                    paths["image"].append(relative_path)
                image_root = Path(relative_path).parts[0]
                image_roots[image_root] = image_roots.get(image_root, 0) + 1
            if suffix in _DICOM_SUFFIXES:
                dicom_names.append(filename)

        if dicom_names:
            series.append(_series_inventory(root, directory_path, dicom_names))

    return _Inventory(
        counts=counts,
        paths=paths,
        complete={
            "csv_paths": True,
            "image_paths": counts["image"] <= len(paths["image"]),
            "image_roots": True,
            "json_paths": True,
            "report_paths": True,
            "submission_paths": True,
        },
        image_roots=dict(sorted(image_roots.items())),
        series=series,
        total_bytes=total_bytes,
    )


def _series_inventory(
    root: Path, directory: Path, dicom_names: list[str]
) -> SeriesInventory:
    parts = directory.relative_to(root).parts
    if len(parts) >= 3:
        image_root = "/".join(parts[:-2])
        study_id = parts[-2]
        series_id = parts[-1]
    else:
        image_root = None
        study_id = None
        series_id = None
    return SeriesInventory(
        path=directory,
        slice_count=len(dicom_names),
        image_root=image_root,
        study_id=study_id,
        series_id=series_id,
        sample_image_id=Path(dicom_names[0]).stem,
    )


def _summarize_csv(path: Path, sample_limit: int = 1_000) -> _CsvSummary:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        columns = list(reader.fieldnames or [])
        sampled_values: dict[str, set[str]] = {column: set() for column in columns}
        sampled_study_series_pairs: set[tuple[str, str]] = set()
        identifier_columns = _identifier_columns(columns)
        study_column = _column_for_identifier_role(columns, "study")
        series_column = _column_for_identifier_role(columns, "series")
        identifier_rows: set[tuple[str, ...]] = set()
        duplicate_identifier_rows = 0
        missing_identifier_rows = 0
        row_count = 0
        for row_count, row in enumerate(reader, start=1):
            if row_count <= sample_limit:
                for column in columns:
                    value = row.get(column)
                    if value not in {None, ""}:
                        sampled_values[column].add(cast(str, value))
                if study_column is not None and series_column is not None:
                    study_id = row.get(study_column)
                    series_id = row.get(series_column)
                    if study_id and series_id:
                        sampled_study_series_pairs.add((study_id, series_id))
            identifier_row = tuple(
                (row.get(column) or "").strip() for column in identifier_columns
            )
            if identifier_columns:
                if any(not value for value in identifier_row):
                    missing_identifier_rows += 1
                elif identifier_row in identifier_rows:
                    duplicate_identifier_rows += 1
                else:
                    identifier_rows.add(identifier_row)
    return _CsvSummary(
        columns=columns,
        row_count=row_count,
        sampled_values=sampled_values,
        sampled_study_series_pairs=sampled_study_series_pairs,
        identifier_rows=identifier_rows,
        duplicate_identifier_rows=duplicate_identifier_rows,
        missing_identifier_rows=missing_identifier_rows,
    )


def _identifier_columns(columns: list[str]) -> list[str]:
    return [column for column in columns if _identifier_role(column) is not None]


def _column_for_identifier_role(columns: list[str], role: IdentifierRole) -> str | None:
    return next(
        (column for column in columns if _identifier_role(column) == role), None
    )


def _identifier_role(column: str) -> IdentifierRole | None:
    normalized = "".join(
        character for character in column.casefold() if character.isalnum()
    )
    known_roles: dict[str, IdentifierRole] = {
        "studyid": "study",
        "studyinstanceuid": "study",
        "seriesid": "series",
        "seriesinstanceuid": "series",
        "imageid": "image",
        "sopinstanceuid": "image",
    }
    if normalized in known_roles:
        return known_roles[normalized]
    if column.casefold() == "id" or column.casefold().endswith("_id"):
        return "other"
    return None


def _target_label_representation(
    csv_summaries: dict[Path, _CsvSummary],
    targets: list[str],
    sample_submission_path: Path,
) -> dict[str, list[str]]:
    training_tables = [
        summary
        for path, summary in csv_summaries.items()
        if path != sample_submission_path
        and path.name.startswith("train")
        and all(target in summary.columns for target in targets)
    ]
    if not training_tables:
        return {target: [] for target in targets}
    training_table = training_tables[0]
    return {target: sorted(training_table.sampled_values[target]) for target in targets}


def _summarize_reports(
    root: Path,
    report_paths: list[str],
    csv_summaries: dict[Path, _CsvSummary],
) -> _ReportSummary:
    discovered_paths = set(report_paths)
    structures: dict[str, list[str]] = {}
    language_values: set[str] = set()
    for path, summary in csv_summaries.items():
        relative_path = path.relative_to(root).as_posix()
        normalized_columns = {column.casefold() for column in summary.columns}
        if normalized_columns & _REPORT_COLUMN_NAMES:
            discovered_paths.add(relative_path)
        if relative_path in discovered_paths:
            structures[relative_path] = summary.columns
            for column in summary.columns:
                normalized = column.casefold()
                if "language" in normalized or normalized in {"lang", "locale"}:
                    language_values.update(summary.sampled_values[column])
    return _ReportSummary(
        paths=sorted(discovered_paths),
        structures=dict(sorted(structures.items())),
        language_values=sorted(language_values),
    )


def _summarize_path_hierarchy(
    series: list[SeriesInventory],
) -> _HierarchySummary:
    valid_series = [item for item in series if item.relationships_valid]
    study_to_series: dict[tuple[str, str], set[str]] = {}
    identifier_samples: list[dict[str, str]] = []
    for item in valid_series:
        image_root = cast(str, item.image_root)
        study_id = cast(str, item.study_id)
        series_id = cast(str, item.series_id)
        study_to_series.setdefault((image_root, study_id), set()).add(series_id)
        if len(identifier_samples) < 12:
            identifier_samples.append(
                {
                    "image_id": cast(str, item.sample_image_id),
                    "image_root": image_root,
                    "series_id": series_id,
                    "study_id": study_id,
                }
            )

    series_per_study = [len(values) for values in study_to_series.values()]
    images_per_series = [item.slice_count for item in series]
    return _HierarchySummary(
        image_count=sum(images_per_series),
        images_per_series=_distribution(images_per_series),
        relationships_verified=bool(series) and len(valid_series) == len(series),
        series_count=len(series),
        series_per_study=_distribution(series_per_study),
        study_count=len(study_to_series),
        study_ids={cast(str, item.study_id) for item in valid_series},
        series_ids={cast(str, item.series_id) for item in valid_series},
        study_series_pairs={
            (cast(str, item.study_id), cast(str, item.series_id))
            for item in valid_series
        },
        identifier_samples=identifier_samples,
    )


def _distribution(values: list[int]) -> dict[str, int | float | None]:
    return {
        "maximum": max(values) if values else None,
        "median": median(values) if values else None,
        "minimum": min(values) if values else None,
    }


def _summarize_table_relationships(
    root: Path,
    csv_summaries: dict[Path, _CsvSummary],
    hierarchy: _HierarchySummary,
) -> dict[str, dict[str, int]]:
    relationships: dict[str, dict[str, int]] = {}
    for path, summary in csv_summaries.items():
        study_column = _column_for_identifier_role(summary.columns, "study")
        series_column = _column_for_identifier_role(summary.columns, "series")
        if study_column is None or series_column is None:
            continue
        study_values = summary.sampled_values[study_column]
        series_values = summary.sampled_values[series_column]
        relationships[path.relative_to(root).as_posix()] = {
            "series_ids_matched": len(series_values & hierarchy.series_ids),
            "series_ids_sampled": len(series_values),
            "study_series_pairs_matched": len(
                summary.sampled_study_series_pairs & hierarchy.study_series_pairs
            ),
            "study_series_pairs_sampled": len(summary.sampled_study_series_pairs),
            "study_ids_matched": len(study_values & hierarchy.study_ids),
            "study_ids_sampled": len(study_values),
        }
    return dict(sorted(relationships.items()))


def _classify_assumptions(
    *,
    total_bytes: int,
    targets: list[str],
    submission_identifier_columns: list[str],
    sample_submission: _CsvSummary,
    csv_summaries: dict[Path, _CsvSummary],
    sample_submission_path: Path,
    hierarchy: _HierarchySummary,
    table_relationships: dict[str, dict[str, int]],
    reports: _ReportSummary,
    imaging: ImagingInspection,
    runtime: dict[str, Any],
) -> dict[str, AssumptionStatus]:
    table_relationships_verified = bool(table_relationships) and all(
        values["series_ids_matched"] == values["series_ids_sampled"]
        and values["study_series_pairs_matched"] == values["study_series_pairs_sampled"]
        and values["study_ids_matched"] == values["study_ids_sampled"]
        for values in table_relationships.values()
    )
    preprocessing = imaging.preprocessing
    decoded_sample_verified = (
        preprocessing.decoded_series == imaging.ordered_series_count
        and preprocessing.decoded_series > 0
    )
    return {
        "competition_metric": "unresolved",
        "competition_rules_for_external_data": "unresolved",
        "competition_rules_for_generative_ai": "unresolved",
        "dataset_size": _dataset_size_status(total_bytes),
        "decoder_supports_observed_sample": (
            "verified" if decoded_sample_verified else "unresolved"
        ),
        "deterministic_eight_slice_sampling_executes": (
            "verified"
            if preprocessing.deterministic_sampling and preprocessing.decoded_series > 0
            else "unresolved"
        ),
        "external_report_processing_rule": "unresolved",
        "geometric_ordering_available": (
            "verified" if imaging.geometry_available else "unresolved"
        ),
        "identifier_hierarchy_is_study_series_image": (
            "verified"
            if hierarchy.relationships_verified and table_relationships_verified
            else "unresolved"
        ),
        "internet_access_rule": "unresolved",
        "monochrome1_interface_on_observed_sample": (
            "verified" if preprocessing.monochrome1.verified else "unresolved"
        ),
        "multilingual_reports": (
            "verified" if len(reports.language_values) > 1 else "unresolved"
        ),
        "percentile_normalization_is_suitable": "unresolved",
        "pretrained_weights_are_permitted": "unresolved",
        "report_pseudo_labelling_is_needed": "unresolved",
        "reports_are_available": "verified" if reports.available else "rejected",
        "rescale_interface_on_observed_sample": (
            "verified" if preprocessing.rescale.verified else "unresolved"
        ),
        "runtime_environment_captured": "verified" if runtime else "unresolved",
        "runtime_limits": "unresolved",
        "submission_contract": _submission_contract_status(
            sample_submission=sample_submission,
            sample_submission_path=sample_submission_path,
            submission_identifier_columns=submission_identifier_columns,
            targets=targets,
            csv_summaries=csv_summaries,
        ),
        "target_schema_from_sample_submission": "verified" if targets else "unresolved",
    }


def _dataset_size_status(total_bytes: int) -> AssumptionStatus:
    if total_bytes <= 0:
        return "unresolved"
    relative_difference = abs(total_bytes - _PROVISIONAL_DATASET_SIZE_BYTES) / (
        _PROVISIONAL_DATASET_SIZE_BYTES
    )
    return (
        "verified"
        if relative_difference <= _DATASET_SIZE_RELATIVE_TOLERANCE
        else "rejected"
    )


def _submission_contract_status(
    *,
    sample_submission: _CsvSummary,
    sample_submission_path: Path,
    submission_identifier_columns: list[str],
    targets: list[str],
    csv_summaries: dict[Path, _CsvSummary],
) -> AssumptionStatus:
    if (
        not submission_identifier_columns
        or not targets
        or sample_submission.row_count == 0
    ):
        return "unresolved"
    if (
        sample_submission.missing_identifier_rows > 0
        or sample_submission.duplicate_identifier_rows > 0
    ):
        return "rejected"

    compatible_test_tables = [
        summary
        for path, summary in csv_summaries.items()
        if path != sample_submission_path
        and path.name.casefold().startswith("test")
        and _identifier_columns(summary.columns) == submission_identifier_columns
    ]
    if not compatible_test_tables:
        return "unresolved"
    return (
        "verified"
        if any(
            summary.row_count == sample_submission.row_count
            and summary.missing_identifier_rows == 0
            and summary.duplicate_identifier_rows == 0
            and summary.identifier_rows == sample_submission.identifier_rows
            for summary in compatible_test_tables
        )
        else "rejected"
    )
