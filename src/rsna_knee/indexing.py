import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from pydicom import dcmread
from pydicom.dataset import Dataset

from .competition_contract import SeriesTableContract, load_verified_index_contract
from .config import PreprocessingConfig
from .preprocessing import preprocess_series_result, uniform_indices

_SCHEMA_VERSION = 1
_ARTIFACT_DIRECTORY = "rsna-knee-index-v1"
_DICOM_SUFFIXES = {".dcm", ".dicom"}
_HEADER_TAGS = (
    "SOPInstanceUID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "PatientID",
    "Rows",
    "Columns",
    "InstanceNumber",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "PhotometricInterpretation",
    "RescaleSlope",
    "RescaleIntercept",
)

type _OrderingMethod = Literal["none", "geometry", "instance_number", "filename"]
type _MetadataAttributeStatus = Literal["present", "missing", "invalid"]


@dataclass(frozen=True, slots=True)
class IndexOutcome:
    artifact_dir: Path
    manifest_path: Path
    statistics_path: Path
    statistics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _SeriesSource:
    split: str
    image_root: str
    study_uid: str
    series_uid: str
    fluid_sensitive: str
    fat_suppression: str
    anatomical_plane: str


@dataclass(frozen=True, slots=True)
class _Header:
    path: Path
    sop_uid: str
    patient_id: str
    rows: int | None
    columns: int | None
    instance_number: int | None
    orientation: tuple[float, ...] | None
    position: tuple[float, ...] | None
    transfer_syntax: str
    photometric_interpretation: str
    rescale_slope: float
    rescale_intercept: float


@dataclass(frozen=True, slots=True)
class _Ordering:
    headers: tuple[_Header, ...]
    method: _OrderingMethod
    geometry_coordinates: dict[Path, float]


@dataclass(frozen=True, slots=True)
class _SelectedSlice:
    slot: int
    slice_index: int
    ordered_position: int
    path: Path


@dataclass(frozen=True, slots=True)
class _RepresentativeCandidate:
    source: _SeriesSource
    series_index: int
    selected: tuple[_SelectedSlice, ...]


@dataclass(slots=True)
class _StudyAggregate:
    split: str
    study_uid: str
    series_count: int = 0
    usable_series_count: int = 0
    slice_count: int = 0
    usable_slice_count: int = 0
    patient_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class _ErrorRecord:
    severity: str
    scope: str
    code: str
    split: str = ""
    study_index: int | None = None
    series_index: int | None = None
    slice_index: int | None = None
    relative_path: str = ""
    detail: str = ""


def verify_index_artifact(artifact_dir: Path) -> dict[str, Any]:
    """Verify a completed local or Kaggle index without reading private pixels."""
    manifest_path = artifact_dir / "index-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing index manifest: {manifest_path}")
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("Unsupported index schema version")
    contract = load_verified_index_contract()
    if manifest.get("competition_contract") != {
        "schema_version": contract.schema_version,
        "verified_at": contract.verified_at,
        "sha256": contract.sha256,
    }:
        raise ValueError("Index does not match the canonical competition contract")
    if manifest.get("decoded_images_persisted") is not False:
        raise ValueError("Index manifest must prohibit persisted decoded images")
    if manifest.get("decoder_dependencies") != ["pydicom"]:
        raise ValueError("Index decoder dependency contract has drifted")

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise TypeError("Index manifest has no file inventory")
    expected_file_columns: dict[str, list[str] | None] = {
        "studies.csv": _study_fields(),
        "series.csv": _series_fields(),
        "slices.csv": _slice_fields(),
        "selected_slices.csv": _selected_fields(),
        "errors.csv": _error_fields(),
        "statistics.json": None,
    }
    if set(files) != set(expected_file_columns):
        raise ValueError("Index manifest file inventory is incomplete or unexpected")
    for filename, raw_metadata in files.items():
        if not isinstance(filename, str) or not isinstance(raw_metadata, dict):
            raise TypeError("Index manifest file inventory is malformed")
        path = artifact_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing index artifact file: {path}")
        if raw_metadata.get("sha256") != _sha256(path):
            raise ValueError(f"Index artifact hash mismatch: {filename}")
        if raw_metadata.get("size_bytes") != path.stat().st_size:
            raise ValueError(f"Index artifact size mismatch: {filename}")
        columns = expected_file_columns[filename]
        if raw_metadata.get("columns") != columns and columns is not None:
            raise ValueError(f"Index artifact columns are malformed: {filename}")
        if columns is not None:
            with path.open(newline="", encoding="utf-8") as file:
                if next(csv.reader(file), None) != columns:
                    raise ValueError(f"Index artifact columns do not match: {filename}")
        elif "columns" in raw_metadata:
            raise ValueError(f"Non-tabular artifact records columns: {filename}")

    forbidden_suffixes = {".dcm", ".dicom", ".npy", ".npz", ".png"}
    persisted_pixels = [
        path
        for path in artifact_dir.rglob("*")
        if path.is_file() and path.suffix.casefold() in forbidden_suffixes
    ]
    if persisted_pixels:
        raise ValueError("Index artifact contains decoded or source pixel files")

    statistics_path = artifact_dir / "statistics.json"
    statistics: dict[str, Any] = json.loads(statistics_path.read_text(encoding="utf-8"))
    if statistics.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("Statistics schema does not match the index")
    preprocessing = statistics.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise TypeError("Index has no representative preprocessing evidence")
    checked = preprocessing.get("representative_series_checked")
    succeeded = preprocessing.get("representative_series_succeeded")
    checks = preprocessing.get("checks")
    manifest_preprocessing = manifest.get("preprocessing")
    if not isinstance(manifest_preprocessing, dict):
        raise TypeError("Index manifest has no preprocessing contract")
    slice_count = manifest_preprocessing.get("slice_count")
    image_size = manifest_preprocessing.get("image_size")
    if (
        not isinstance(checked, int)
        or isinstance(checked, bool)
        or checked <= 0
        or not isinstance(succeeded, int)
        or isinstance(succeeded, bool)
        or not isinstance(checks, list)
        or len(checks) != checked
        or not isinstance(slice_count, int)
        or isinstance(slice_count, bool)
        or slice_count <= 0
        or not isinstance(image_size, int)
        or isinstance(image_size, bool)
        or image_size <= 0
    ):
        raise ValueError("Representative preprocessing evidence is incomplete")
    successful_checks = sum(
        _representative_check_succeeded(check, slice_count, image_size)
        for check in checks
    )
    if succeeded != successful_checks or successful_checks != checked:
        raise ValueError("Representative preprocessing checks did not pass")
    return statistics


def build_competition_index(
    input_root: Path,
    work_root: Path,
    config: PreprocessingConfig,
    *,
    representative_series_per_split: int = 4,
    contract_path: Path | None = None,
) -> IndexOutcome:
    """Index verified competition series, then exercise selected real pixels.

    The header/index phase is completed and its files are closed before the first
    representative pixel array is decoded.
    """
    if representative_series_per_split < 0:
        raise ValueError("representative_series_per_split must be non-negative")
    contract = load_verified_index_contract(contract_path)
    competition_root = _discover_competition_root(input_root)
    errors: list[_ErrorRecord] = []
    sources = _load_series_sources(competition_root, contract.series_tables, errors)
    artifact_dir = work_root / _ARTIFACT_DIRECTORY
    artifact_dir.mkdir(parents=True, exist_ok=True)

    studies_path = artifact_dir / "studies.csv"
    series_path = artifact_dir / "series.csv"
    slices_path = artifact_dir / "slices.csv"
    selected_path = artifact_dir / "selected_slices.csv"
    errors_path = artifact_dir / "errors.csv"
    statistics_path = artifact_dir / "statistics.json"
    manifest_path = artifact_dir / "index-manifest.json"

    study_lookup: dict[tuple[str, str], int] = {}
    studies: list[_StudyAggregate] = []
    candidates: list[_RepresentativeCandidate] = []
    transfer_syntax_counts: Counter[str] = Counter()
    ordering_counts: Counter[str] = Counter()
    dimension_counts: Counter[str] = Counter()
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    next_slice_index = 0
    total_slice_count = 0

    with (
        series_path.open("w", newline="", encoding="utf-8") as series_file,
        slices_path.open("w", newline="", encoding="utf-8") as slices_file,
        selected_path.open("w", newline="", encoding="utf-8") as selected_file,
    ):
        series_writer = csv.DictWriter(series_file, fieldnames=_series_fields())
        slices_writer = csv.DictWriter(slices_file, fieldnames=_slice_fields())
        selected_writer = csv.DictWriter(selected_file, fieldnames=_selected_fields())
        series_writer.writeheader()
        slices_writer.writeheader()
        selected_writer.writeheader()

        for series_index, source in enumerate(sources):
            study_key = (source.split, source.study_uid)
            study_index = study_lookup.get(study_key)
            if study_index is None:
                study_index = len(studies)
                study_lookup[study_key] = study_index
                studies.append(
                    _StudyAggregate(split=source.split, study_uid=source.study_uid)
                )
            study = studies[study_index]
            study.series_count += 1

            relative_series_path = (
                Path(source.image_root) / source.study_uid / source.series_uid
            )
            absolute_series_path = competition_root / relative_series_path
            paths = _dicom_paths(absolute_series_path)
            total_slice_count += len(paths)
            headers: list[_Header] = []
            for path in paths:
                header = _read_header(
                    path,
                    competition_root,
                    source,
                    study_index,
                    series_index,
                    errors,
                )
                if header is not None:
                    headers.append(header)

            if not paths:
                errors.append(
                    _ErrorRecord(
                        severity="error",
                        scope="series",
                        code="empty_series",
                        split=source.split,
                        study_index=study_index,
                        series_index=series_index,
                        relative_path=relative_series_path.as_posix(),
                    )
                )
            elif not headers:
                errors.append(
                    _ErrorRecord(
                        severity="error",
                        scope="series",
                        code="unusable_series",
                        split=source.split,
                        study_index=study_index,
                        series_index=series_index,
                        relative_path=relative_series_path.as_posix(),
                        detail="no_valid_headers",
                    )
                )

            ordering = _order_headers(headers)
            ordering_counts[ordering.method] += 1
            if ordering.method != "geometry" and ordering.headers:
                errors.append(
                    _ErrorRecord(
                        severity="warning",
                        scope="series",
                        code=f"ordering_{ordering.method}_fallback",
                        split=source.split,
                        study_index=study_index,
                        series_index=series_index,
                        relative_path=relative_series_path.as_posix(),
                        detail=(
                            "geometry_unavailable"
                            if ordering.method == "instance_number"
                            else "geometry_and_instance_unavailable"
                        ),
                    )
                )

            dimensions = {
                (header.rows, header.columns)
                for header in ordering.headers
                if header.rows is not None and header.columns is not None
            }
            for rows, columns in dimensions:
                dimension_counts[f"{rows}x{columns}"] += 1
            if len(dimensions) > 1:
                errors.append(
                    _ErrorRecord(
                        severity="warning",
                        scope="series",
                        code="inconsistent_dimensions",
                        split=source.split,
                        study_index=study_index,
                        series_index=series_index,
                        relative_path=relative_series_path.as_posix(),
                        detail="|".join(
                            f"{rows}x{columns}" for rows, columns in sorted(dimensions)
                        ),
                    )
                )

            selected_positions = uniform_indices(
                len(ordering.headers), config.slice_count
            )
            selected_position_set = set(selected_positions)
            selected: list[_SelectedSlice] = []
            patient_ids: set[str] = set()
            for ordered_position, header in enumerate(ordering.headers):
                slice_index = next_slice_index
                next_slice_index += 1
                patient_ids.update([header.patient_id] if header.patient_id else [])
                transfer_syntax_counts[header.transfer_syntax or "missing"] += 1
                slices_writer.writerow(
                    {
                        "slice_index": slice_index,
                        "series_index": series_index,
                        "SOPInstanceUID": header.sop_uid,
                        "filename": header.path.name,
                        "ordered_position": ordered_position,
                        "rows": _csv_value(header.rows),
                        "columns": _csv_value(header.columns),
                        "instance_number": _csv_value(header.instance_number),
                        "geometry_coordinate": _csv_float(
                            ordering.geometry_coordinates.get(header.path)
                        ),
                        "transfer_syntax": header.transfer_syntax,
                        "photometric_interpretation": (
                            header.photometric_interpretation
                        ),
                        "rescale_slope": header.rescale_slope,
                        "rescale_intercept": header.rescale_intercept,
                    }
                )
                if ordered_position in selected_position_set:
                    slot = selected_positions.index(ordered_position)
                    item = _SelectedSlice(
                        slot=slot,
                        slice_index=slice_index,
                        ordered_position=ordered_position,
                        path=header.path,
                    )
                    selected.append(item)
                    selected_writer.writerow(
                        {
                            "series_index": series_index,
                            "slot": slot,
                            "slice_index": slice_index,
                            "ordered_position": ordered_position,
                        }
                    )

            if len(patient_ids) > 1:
                errors.append(
                    _ErrorRecord(
                        severity="error",
                        scope="series",
                        code="inconsistent_patient_id",
                        split=source.split,
                        study_index=study_index,
                        series_index=series_index,
                        relative_path=relative_series_path.as_posix(),
                    )
                )
            study.patient_ids.update(patient_ids)
            study.slice_count += len(paths)
            study.usable_slice_count += len(ordering.headers)
            if ordering.headers:
                study.usable_series_count += 1
                candidates.append(
                    _RepresentativeCandidate(
                        source=source,
                        series_index=series_index,
                        selected=tuple(selected),
                    )
                )

            split_counts[source.split]["series"] += 1
            split_counts[source.split]["slices"] += len(paths)
            split_counts[source.split]["usable_slices"] += len(ordering.headers)
            split_counts[source.split]["selected_slices"] += len(selected)
            series_writer.writerow(
                {
                    "series_index": series_index,
                    "study_index": study_index,
                    "SeriesInstanceUID": source.series_uid,
                    "relative_path": relative_series_path.as_posix(),
                    "Fluid_Sensitive": source.fluid_sensitive,
                    "Fat_Suppression": source.fat_suppression,
                    "Anatomical_Plane": source.anatomical_plane,
                    "slice_count": len(paths),
                    "usable_slice_count": len(ordering.headers),
                    "ordering_method": ordering.method,
                    "selection_mask": "".join(
                        "1" if slot < len(selected) else "0"
                        for slot in range(config.slice_count)
                    ),
                }
            )

    _write_studies(studies_path, studies, errors)

    representative_results = _run_representative_checks(
        candidates,
        competition_root,
        config,
        representative_series_per_split,
        study_lookup,
        errors,
    )
    _write_errors(errors_path, errors)

    for study in studies:
        split_counts[study.split]["studies"] += 1
    error_counts = Counter(error.code for error in errors)
    statistics: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "counts": {
            "studies": len(studies),
            "series": len(sources),
            "slices": total_slice_count,
            "usable_slices": next_slice_index,
            "selected_slices": sum(
                counter["selected_slices"] for counter in split_counts.values()
            ),
            "errors": len(errors),
        },
        "by_split": {
            split: dict(sorted(counter.items()))
            for split, counter in sorted(split_counts.items())
        },
        "dimensions": dict(sorted(dimension_counts.items())),
        "errors_by_code": dict(sorted(error_counts.items())),
        "ordering_methods": dict(sorted(ordering_counts.items())),
        "transfer_syntaxes": dict(sorted(transfer_syntax_counts.items())),
        "decoder": {
            "required_dependency": "pydicom",
            "optional_plugins": [],
            "sampled_decode_failures": error_counts["pixel_decode_failed"],
        },
        "preprocessing": representative_results,
    }
    statistics_path.write_text(
        json.dumps(statistics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    artifact_columns = {
        studies_path.name: _study_fields(),
        series_path.name: _series_fields(),
        slices_path.name: _slice_fields(),
        selected_path.name: _selected_fields(),
        errors_path.name: _error_fields(),
    }
    artifact_files: dict[str, dict[str, Any]] = {}
    for path in (
        studies_path,
        series_path,
        slices_path,
        selected_path,
        errors_path,
        statistics_path,
    ):
        metadata: dict[str, Any] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        if path.name in artifact_columns:
            metadata["columns"] = artifact_columns[path.name]
        artifact_files[path.name] = metadata
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "artifact_type": "rsna_knee_compact_index",
        "competition_contract": {
            "schema_version": contract.schema_version,
            "verified_at": contract.verified_at,
            "sha256": contract.sha256,
        },
        "source_root": str(competition_root),
        "join_fields": list(contract.series_tables[0].columns[:2]),
        "series_metadata_fields": list(contract.series_tables[0].columns[2:]),
        "preprocessing": {
            "slice_count": config.slice_count,
            "lower_percentile": config.lower_percentile,
            "upper_percentile": config.upper_percentile,
            "image_size": config.image_size,
            "normalization_status": "provisional",
            "resize": "bilinear_aspect_preserving_center_pad",
        },
        "decoder_dependencies": ["pydicom"],
        "decoded_images_persisted": False,
        "files": artifact_files,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return IndexOutcome(
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        statistics_path=statistics_path,
        statistics=statistics,
    )


def _discover_competition_root(input_root: Path) -> Path:
    candidates = sorted(input_root.rglob("sample_submission.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No sample_submission.csv found below input root: {input_root}"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            "More than one competition root contains sample_submission.csv"
        )
    return candidates[0].parent


def _load_series_sources(
    root: Path,
    table_contracts: tuple[SeriesTableContract, ...],
    errors: list[_ErrorRecord],
) -> list[_SeriesSource]:
    sources: list[_SeriesSource] = []
    seen: set[tuple[str, str, str]] = set()
    for table_contract in table_contracts:
        split = table_contract.split
        image_root = table_contract.image_root
        filename = table_contract.filename
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing verified series table: {path}")
        with path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            columns = tuple(reader.fieldnames or ())
            missing = [
                column for column in table_contract.columns if column not in columns
            ]
            if missing:
                raise ValueError(
                    f"{filename} is missing verified columns: {', '.join(missing)}"
                )
            for row_number, row in enumerate(reader, start=2):
                study_uid = (row.get("StudyInstanceUID") or "").strip()
                series_uid = (row.get("SeriesInstanceUID") or "").strip()
                if not study_uid or not series_uid:
                    errors.append(
                        _ErrorRecord(
                            severity="error",
                            scope="table_row",
                            code="missing_join_identifier",
                            split=split,
                            relative_path=filename,
                            detail=str(row_number),
                        )
                    )
                    continue
                key = (split, study_uid, series_uid)
                if key in seen:
                    errors.append(
                        _ErrorRecord(
                            severity="error",
                            scope="table_row",
                            code="duplicate_series_row",
                            split=split,
                            relative_path=filename,
                            detail=str(row_number),
                        )
                    )
                    continue
                seen.add(key)
                sources.append(
                    _SeriesSource(
                        split=split,
                        image_root=image_root,
                        study_uid=study_uid,
                        series_uid=series_uid,
                        fluid_sensitive=(row.get("Fluid_Sensitive") or "").strip(),
                        fat_suppression=(row.get("Fat_Suppression") or "").strip(),
                        anatomical_plane=(row.get("Anatomical_Plane") or "").strip(),
                    )
                )
    return sources


def _dicom_paths(path: Path) -> list[Path]:
    try:
        return sorted(
            item
            for item in path.iterdir()
            if item.is_file() and item.suffix.casefold() in _DICOM_SUFFIXES
        )
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return []


def _read_header(
    path: Path,
    root: Path,
    source: _SeriesSource,
    study_index: int,
    series_index: int,
    errors: list[_ErrorRecord],
) -> _Header | None:
    relative_path = path.relative_to(root).as_posix()
    try:
        dataset = dcmread(
            path, stop_before_pixels=True, specific_tags=list(_HEADER_TAGS)
        )
    except Exception as error:  # noqa: BLE001 - heterogeneous data errors are indexed
        errors.append(
            _ErrorRecord(
                severity="error",
                scope="slice",
                code="corrupt_dicom_header",
                split=source.split,
                study_index=study_index,
                series_index=series_index,
                relative_path=relative_path,
                detail=type(error).__name__,
            )
        )
        return None

    header_study_uid = _text_attribute(dataset, "StudyInstanceUID")
    header_series_uid = _text_attribute(dataset, "SeriesInstanceUID")
    if (header_study_uid and header_study_uid != source.study_uid) or (
        header_series_uid and header_series_uid != source.series_uid
    ):
        errors.append(
            _ErrorRecord(
                severity="error",
                scope="slice",
                code="identifier_mismatch",
                split=source.split,
                study_index=study_index,
                series_index=series_index,
                relative_path=relative_path,
            )
        )
        return None
    if not header_study_uid or not header_series_uid:
        errors.append(
            _ErrorRecord(
                severity="warning",
                scope="slice",
                code="missing_header_identifier",
                split=source.split,
                study_index=study_index,
                series_index=series_index,
                relative_path=relative_path,
                detail="path_table_join_used",
            )
        )

    rows = _integer_attribute(dataset, "Rows")
    columns = _integer_attribute(dataset, "Columns")
    if rows is None or columns is None or rows <= 0 or columns <= 0:
        errors.append(
            _ErrorRecord(
                severity="warning",
                scope="slice",
                code="invalid_dimensions",
                split=source.split,
                study_index=study_index,
                series_index=series_index,
                relative_path=relative_path,
            )
        )
    rescale_slope, slope_status = _float_attribute_with_status(
        dataset, "RescaleSlope", 1.0
    )
    rescale_intercept, intercept_status = _float_attribute_with_status(
        dataset, "RescaleIntercept", 0.0
    )
    rescale_fallbacks = [
        f"{attribute}:{status}:{format(default, '.9g')}"
        for attribute, status, default in (
            ("RescaleSlope", slope_status, 1.0),
            ("RescaleIntercept", intercept_status, 0.0),
        )
        if status != "present"
    ]
    if rescale_fallbacks:
        errors.append(
            _ErrorRecord(
                severity="warning",
                scope="slice",
                code="rescale_metadata_fallback",
                split=source.split,
                study_index=study_index,
                series_index=series_index,
                relative_path=relative_path,
                detail=";".join(rescale_fallbacks),
            )
        )
    return _Header(
        path=path,
        sop_uid=_text_attribute(dataset, "SOPInstanceUID"),
        patient_id=_text_attribute(dataset, "PatientID"),
        rows=rows,
        columns=columns,
        instance_number=_integer_attribute(dataset, "InstanceNumber"),
        orientation=_float_sequence(dataset, "ImageOrientationPatient", 6),
        position=_float_sequence(dataset, "ImagePositionPatient", 3),
        transfer_syntax=str(getattr(dataset.file_meta, "TransferSyntaxUID", "")),
        photometric_interpretation=_text_attribute(
            dataset, "PhotometricInterpretation"
        ),
        rescale_slope=rescale_slope,
        rescale_intercept=rescale_intercept,
    )


def _order_headers(headers: Sequence[_Header]) -> _Ordering:
    if not headers:
        return _Ordering(headers=(), method="none", geometry_coordinates={})
    geometry_coordinates = _geometry_coordinates(headers)
    if geometry_coordinates is not None:
        return _Ordering(
            headers=tuple(
                sorted(headers, key=lambda header: geometry_coordinates[header.path])
            ),
            method="geometry",
            geometry_coordinates=geometry_coordinates,
        )
    instance_values = [header.instance_number for header in headers]
    if all(value is not None for value in instance_values) and len(
        set(instance_values)
    ) == len(headers):
        return _Ordering(
            headers=tuple(
                sorted(headers, key=lambda header: cast(int, header.instance_number))
            ),
            method="instance_number",
            geometry_coordinates={},
        )
    return _Ordering(
        headers=tuple(sorted(headers, key=lambda header: header.path.name)),
        method="filename",
        geometry_coordinates={},
    )


def _geometry_coordinates(
    headers: Sequence[_Header],
) -> dict[Path, float] | None:
    normals: list[np.ndarray] = []
    for header in headers:
        if header.orientation is None or header.position is None:
            return None
        row = np.asarray(header.orientation[:3], dtype=np.float64)
        column = np.asarray(header.orientation[3:], dtype=np.float64)
        normal = np.cross(row, column)
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(normal).all() or norm < 1e-6:
            return None
        normals.append(normal / norm)
        if not np.isfinite(np.asarray(header.position, dtype=np.float64)).all():
            return None
    reference = normals[0]
    if any(float(np.dot(reference, normal)) < 0.999 for normal in normals[1:]):
        return None
    coordinates = {
        header.path: float(np.dot(reference, np.asarray(header.position)))
        for header in headers
    }
    rounded = [round(value, 6) for value in coordinates.values()]
    return coordinates if len(set(rounded)) == len(rounded) else None


def _run_representative_checks(
    candidates: list[_RepresentativeCandidate],
    root: Path,
    config: PreprocessingConfig,
    per_split_limit: int,
    study_lookup: dict[tuple[str, str], int],
    errors: list[_ErrorRecord],
) -> dict[str, Any]:
    selected_candidates: list[_RepresentativeCandidate] = []
    for split in ("train", "test"):
        split_candidates = [item for item in candidates if item.source.split == split]
        selected_candidates.extend(
            _diverse_representatives(split_candidates, per_split_limit)
        )

    checks: list[dict[str, Any]] = []
    for candidate in selected_candidates:
        paths = [item.path for item in candidate.selected]
        result = preprocess_series_result(paths, config)
        study_index = study_lookup[(candidate.source.split, candidate.source.study_uid)]
        by_path = {item.path: item for item in candidate.selected}
        for failure in result.failures:
            selected_slice = by_path[failure.path]
            errors.append(
                _ErrorRecord(
                    severity="error",
                    scope="slice",
                    code=failure.code,
                    split=candidate.source.split,
                    study_index=study_index,
                    series_index=candidate.series_index,
                    slice_index=selected_slice.slice_index,
                    relative_path=failure.path.relative_to(root).as_posix(),
                    detail=failure.exception_type,
                )
            )
        visible = result.pixels[result.mask]
        checks.append(
            {
                "split": candidate.source.split,
                "series_index": candidate.series_index,
                "study_index": study_index,
                "plane": candidate.source.anatomical_plane,
                "selected_count": len(paths),
                "decoded_count": int(result.mask.sum()),
                "mask": result.mask.astype(int).tolist(),
                "output_shape": list(result.pixels.shape),
                "all_finite": bool(np.isfinite(visible).all())
                if visible.size
                else False,
                "normalized_range": (
                    [float(visible.min()), float(visible.max())] if visible.size else []
                ),
                "lower_value": result.lower_value,
                "upper_value": result.upper_value,
            }
        )
    return {
        "contract": "shared_train_inference_preprocess_series_result",
        "normalization_status": "provisional",
        "representative_studies_checked": len(
            {(check["split"], check["study_index"]) for check in checks}
        ),
        "representative_series_checked": len(checks),
        "representative_series_succeeded": sum(
            _representative_check_succeeded(
                check,
                config.slice_count,
                config.image_size,
            )
            for check in checks
        ),
        "checks": checks,
    }


def _representative_check_succeeded(
    check: object, slice_count: int, image_size: int
) -> bool:
    if not isinstance(check, dict) or check.get("all_finite") is not True:
        return False
    selected_count = check.get("selected_count")
    decoded_count = check.get("decoded_count")
    mask = check.get("mask")
    output_shape = check.get("output_shape")
    normalized_range = check.get("normalized_range")
    lower_value = check.get("lower_value")
    upper_value = check.get("upper_value")
    return bool(
        isinstance(selected_count, int)
        and not isinstance(selected_count, bool)
        and 0 < selected_count <= slice_count
        and isinstance(decoded_count, int)
        and not isinstance(decoded_count, bool)
        and decoded_count == selected_count
        and isinstance(mask, list)
        and len(mask) == slice_count
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value in (0, 1)
            for value in mask
        )
        and sum(mask) == decoded_count
        and output_shape == [slice_count, image_size, image_size]
        and isinstance(normalized_range, list)
        and len(normalized_range) == 2
        and all(_is_finite_number(value) for value in normalized_range)
        and 0 <= normalized_range[0] <= normalized_range[1] <= 1
        and _is_finite_number(lower_value)
        and _is_finite_number(upper_value)
        and cast(float, lower_value) <= cast(float, upper_value)
    )


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and bool(np.isfinite(value))
    )


def _diverse_representatives(
    candidates: list[_RepresentativeCandidate], limit: int
) -> list[_RepresentativeCandidate]:
    remaining = list(candidates)
    selected: list[_RepresentativeCandidate] = []
    studies: set[str] = set()
    planes: set[str] = set()
    while remaining and len(selected) < limit:
        best_index = max(
            range(len(remaining)),
            key=lambda index: (
                int(remaining[index].source.study_uid not in studies)
                + int(remaining[index].source.anatomical_plane not in planes),
                int(remaining[index].source.study_uid not in studies),
                int(remaining[index].source.anatomical_plane not in planes),
                -remaining[index].series_index,
            ),
        )
        candidate = remaining.pop(best_index)
        selected.append(candidate)
        studies.add(candidate.source.study_uid)
        planes.add(candidate.source.anatomical_plane)
    return selected


def _write_studies(
    path: Path, studies: list[_StudyAggregate], errors: list[_ErrorRecord]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=_study_fields())
        writer.writeheader()
        for study_index, study in enumerate(studies):
            if len(study.patient_ids) > 1:
                errors.append(
                    _ErrorRecord(
                        severity="error",
                        scope="study",
                        code="inconsistent_patient_id",
                        split=study.split,
                        study_index=study_index,
                    )
                )
            writer.writerow(
                {
                    "study_index": study_index,
                    "split": study.split,
                    "StudyInstanceUID": study.study_uid,
                    "PatientID": (
                        next(iter(study.patient_ids))
                        if len(study.patient_ids) == 1
                        else ""
                    ),
                    "series_count": study.series_count,
                    "usable_series_count": study.usable_series_count,
                    "slice_count": study.slice_count,
                    "usable_slice_count": study.usable_slice_count,
                }
            )


def _write_errors(path: Path, errors: list[_ErrorRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=_error_fields())
        writer.writeheader()
        for error_id, error in enumerate(errors):
            writer.writerow(
                {
                    "error_id": error_id,
                    "severity": error.severity,
                    "scope": error.scope,
                    "code": error.code,
                    "split": error.split,
                    "study_index": _csv_value(error.study_index),
                    "series_index": _csv_value(error.series_index),
                    "slice_index": _csv_value(error.slice_index),
                    "relative_path": error.relative_path,
                    "detail": error.detail,
                }
            )


def _text_attribute(dataset: Dataset, attribute: str) -> str:
    value = getattr(dataset, attribute, "")
    return str(value).strip() if value is not None else ""


def _integer_attribute(dataset: Dataset, attribute: str) -> int | None:
    value: object = getattr(dataset, attribute, None)
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return None


def _float_attribute_with_status(
    dataset: Dataset, attribute: str, default: float
) -> tuple[float, _MetadataAttributeStatus]:
    value: object = getattr(dataset, attribute, None)
    if value is None:
        return default, "missing"
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError):
        return default, "invalid"
    if not np.isfinite(result):
        return default, "invalid"
    return result, "present"


def _float_sequence(
    dataset: Dataset, attribute: str, length: int
) -> tuple[float, ...] | None:
    value: object = getattr(dataset, attribute, None)
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return result if len(result) == length else None


def _csv_value(value: int | None) -> int | str:
    return "" if value is None else value


def _csv_float(value: float | None) -> str:
    return "" if value is None else format(value, ".9g")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _study_fields() -> list[str]:
    return [
        "study_index",
        "split",
        "StudyInstanceUID",
        "PatientID",
        "series_count",
        "usable_series_count",
        "slice_count",
        "usable_slice_count",
    ]


def _series_fields() -> list[str]:
    return [
        "series_index",
        "study_index",
        "SeriesInstanceUID",
        "relative_path",
        "Fluid_Sensitive",
        "Fat_Suppression",
        "Anatomical_Plane",
        "slice_count",
        "usable_slice_count",
        "ordering_method",
        "selection_mask",
    ]


def _slice_fields() -> list[str]:
    return [
        "slice_index",
        "series_index",
        "SOPInstanceUID",
        "filename",
        "ordered_position",
        "rows",
        "columns",
        "instance_number",
        "geometry_coordinate",
        "transfer_syntax",
        "photometric_interpretation",
        "rescale_slope",
        "rescale_intercept",
    ]


def _selected_fields() -> list[str]:
    return ["series_index", "slot", "slice_index", "ordered_position"]


def _error_fields() -> list[str]:
    return [
        "error_id",
        "severity",
        "scope",
        "code",
        "split",
        "study_index",
        "series_index",
        "slice_index",
        "relative_path",
        "detail",
    ]
