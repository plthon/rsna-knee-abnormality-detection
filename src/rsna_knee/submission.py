import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .competition_contract import (
    VerifiedSubmissionContract,
    load_verified_submission_contract,
)
from .dicom import decode_pixels
from .provenance import create_run_id, write_run_manifest

_VALIDATION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SubmissionInputs:
    competition_root: Path
    entity_ids: tuple[str, ...]
    contract: VerifiedSubmissionContract


@dataclass(frozen=True, slots=True)
class SubmissionOutcome:
    submission_path: Path
    fallbacks_path: Path
    validation_report_path: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class SubmissionArtifacts:
    submission_path: Path
    fallbacks_path: Path
    validation_report_path: Path


def discover_submission_inputs(
    input_root: Path,
    *,
    contract_path: Path | None = None,
) -> SubmissionInputs:
    """Discover and verify the dynamic study-level submission inputs."""
    contract = load_verified_submission_contract(contract_path)
    competition_root = _discover_competition_root(input_root)
    test_ids = _read_identifiers(
        competition_root / "test.csv", (contract.identifier_column,)
    )
    sample_ids = _read_identifiers(
        competition_root / "sample_submission.csv", contract.ordered_columns
    )
    if set(test_ids) != set(sample_ids):
        raise ValueError("Test and sample submission identifiers do not match")
    return SubmissionInputs(
        competition_root=competition_root,
        entity_ids=test_ids,
        contract=contract,
    )


def run_constant_baseline(
    *,
    input_root: Path,
    work_root: Path,
    probability: float = 0.5,
    contract_path: Path | None = None,
    run_id: str | None = None,
    notebook: Mapping[str, object] | None = None,
    source_provenance: Mapping[str, object] | None = None,
) -> SubmissionOutcome:
    """Write and validate a CPU-only constant prediction for every test study."""
    inputs = discover_submission_inputs(input_root, contract_path=contract_path)
    failures = discover_unusable_studies(inputs)
    predictions = {
        entity_id: values
        for entity_id, values in constant_predictions(
            inputs, probability=probability
        ).items()
        if entity_id not in failures
    }
    artifacts = write_submission(
        inputs=inputs,
        work_root=work_root,
        predictions=predictions,
        failures=failures,
        fallback_probability=probability,
    )
    manifest_path = work_root / "run-manifest.json"
    manifest_source = dict(source_provenance or {})
    manifest_source.update(
        {
            "competition_contract": {
                "schema_version": inputs.contract.schema_version,
                "verified_at": inputs.contract.verified_at,
                "sha256": inputs.contract.sha256,
            },
            "competition_root": str(inputs.competition_root),
            "inputs": {
                filename: _file_record(inputs.competition_root / filename)
                for filename in (
                    "sample_submission.csv",
                    "test.csv",
                    inputs.contract.test_series_filename,
                )
            },
        }
    )
    write_run_manifest(
        manifest_path,
        run_id=run_id or create_run_id("constant-baseline"),
        source=manifest_source,
        configuration={
            "fallback_probability": probability,
            "probability": probability,
            "target_order": list(inputs.contract.target_names),
            "usability_probe": "first_decodable_slice_per_study",
        },
        notebook=notebook or {"kind": "local_or_kaggle_constant_baseline"},
        artifacts=[
            artifacts.submission_path,
            artifacts.fallbacks_path,
            artifacts.validation_report_path,
        ],
        artifact_root=work_root,
    )
    return SubmissionOutcome(
        submission_path=artifacts.submission_path,
        fallbacks_path=artifacts.fallbacks_path,
        validation_report_path=artifacts.validation_report_path,
        manifest_path=manifest_path,
    )


def constant_predictions(
    inputs: SubmissionInputs, *, probability: float
) -> dict[str, tuple[float, ...]]:
    """Produce one fixed CPU prediction vector for every required entity."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Constant probability must be within [0,1]")
    values = (probability,) * len(inputs.contract.target_names)
    return {entity_id: values for entity_id in inputs.entity_ids}


def discover_unusable_studies(inputs: SubmissionInputs) -> dict[str, str]:
    """Record studies with no DICOM slice decodable by the inference decoder."""
    series_by_study = _read_test_series(inputs)
    failures: dict[str, str] = {}
    for entity_id in inputs.entity_ids:
        series_ids = series_by_study.get(entity_id, ())
        if not series_ids:
            failures[entity_id] = "no_declared_series"
            continue
        found_dicom = False
        found_decodable_slice = False
        for series_id in series_ids:
            series_root = (
                inputs.competition_root
                / inputs.contract.test_series_image_root
                / entity_id
                / series_id
            )
            for path in _dicom_paths(series_root):
                found_dicom = True
                if _is_decodable_slice(path):
                    found_decodable_slice = True
                    break
            if found_decodable_slice:
                break
        if not found_decodable_slice:
            failures[entity_id] = (
                "no_decodable_slices" if found_dicom else "no_dicom_files"
            )
    return failures


def write_submission(
    *,
    inputs: SubmissionInputs,
    work_root: Path,
    predictions: Mapping[str, Sequence[float]],
    failures: Mapping[str, str],
    fallback_probability: float,
) -> SubmissionArtifacts:
    """Complete, record, write, and validate study-level predictions.

    Every required entity must have either a prediction or an explicit failure.
    Failed entities receive the fixed fallback and remain in the submission.
    """
    if not 0.0 <= fallback_probability <= 1.0:
        raise ValueError("Fallback probability must be within [0,1]")
    expected_ids = set(inputs.entity_ids)
    prediction_ids = set(predictions)
    failure_ids = set(failures)
    unexpected_ids = (prediction_ids | failure_ids) - expected_ids
    if unexpected_ids:
        raise ValueError("Predictions or failures contain unexpected identifiers")
    if prediction_ids & failure_ids:
        raise ValueError("An entity cannot have both a prediction and a failure")
    missing_ids = expected_ids - prediction_ids - failure_ids
    if missing_ids:
        raise ValueError("Missing predictions require an explicit failure record")
    if any(not reason.strip() for reason in failures.values()):
        raise ValueError("Fallback reasons must be nonempty")

    work_root.mkdir(parents=True, exist_ok=True)
    submission_path = work_root / inputs.contract.filename
    fallbacks_path = work_root / "submission-fallbacks.csv"
    validation_report_path = work_root / "submission-validation.json"
    fallback_values = (fallback_probability,) * len(inputs.contract.target_names)

    with submission_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(inputs.contract.ordered_columns)
        for entity_id in inputs.entity_ids:
            values = predictions.get(entity_id, fallback_values)
            if len(values) != len(inputs.contract.target_names):
                raise ValueError(
                    f"Prediction for {entity_id} has the wrong target count"
                )
            writer.writerow((entity_id, *values))
    with fallbacks_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow((inputs.contract.identifier_column, "reason"))
        for entity_id in inputs.entity_ids:
            if entity_id in failures:
                writer.writerow((entity_id, failures[entity_id]))

    validation = validate_submission(submission_path, inputs=inputs)
    validation["fallbacks"] = {
        "count": len(failures),
        "path": fallbacks_path.name,
    }
    validation_report_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return SubmissionArtifacts(
        submission_path=submission_path,
        fallbacks_path=fallbacks_path,
        validation_report_path=validation_report_path,
    )


def validate_submission(path: Path, *, inputs: SubmissionInputs) -> dict[str, Any]:
    """Validate a submission against the discovered entities and verified schema."""
    return _validate_submission(
        path,
        expected_entity_ids=inputs.entity_ids,
        contract=inputs.contract,
    )


def _validate_submission(
    path: Path,
    *,
    expected_entity_ids: Sequence[str],
    contract: VerifiedSubmissionContract,
) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.reader(file))
    if not rows or tuple(rows[0]) != contract.ordered_columns:
        raise ValueError("Submission columns or order do not match the contract")
    data_rows = rows[1:]
    if len(data_rows) != len(expected_entity_ids):
        raise ValueError("Submission row count does not match required entities")

    identifiers: list[str] = []
    probabilities: list[float] = []
    for row_number, row in enumerate(data_rows, start=2):
        if len(row) != len(contract.ordered_columns):
            raise ValueError(f"Submission row {row_number} has the wrong column count")
        identifier = row[0].strip()
        if not identifier:
            raise ValueError(f"Submission row {row_number} has a missing identifier")
        identifiers.append(identifier)
        for raw_value in row[1:]:
            try:
                value = float(raw_value)
            except ValueError as error:
                raise ValueError(
                    f"Submission row {row_number} has a nonnumeric probability"
                ) from error
            if not math.isfinite(value):
                raise ValueError(
                    f"Submission row {row_number} has a non-finite probability"
                )
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Submission row {row_number} has a probability outside [0,1]"
                )
            probabilities.append(value)

    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Submission contains duplicate identifiers")
    if set(identifiers) != set(expected_entity_ids):
        raise ValueError("Submission identifiers do not match required entities")
    return {
        "columns": list(contract.ordered_columns),
        "competition_contract_sha256": contract.sha256,
        "identifier_column": contract.identifier_column,
        "probabilities": {
            "maximum": max(probabilities),
            "minimum": min(probabilities),
        },
        "row_count": len(data_rows),
        "schema_version": _VALIDATION_SCHEMA_VERSION,
        "submission_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "target_count": len(contract.target_names),
        "valid": True,
    }


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


def _read_identifiers(path: Path, expected_columns: tuple[str, ...]) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing verified submission input: {path}")
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.reader(file))
    if not rows or tuple(rows[0]) != expected_columns:
        raise ValueError(
            f"Input columns do not match the verified contract: {path.name}"
        )
    identifiers: list[str] = []
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != len(expected_columns):
            raise ValueError(f"Malformed row {row_number} in {path.name}")
        identifier = row[0].strip()
        if not identifier:
            raise ValueError(f"Missing identifier at row {row_number} in {path.name}")
        identifiers.append(identifier)
    if not identifiers:
        raise ValueError(f"Verified submission input contains no entities: {path.name}")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"Duplicate identifiers in {path.name}")
    return tuple(identifiers)


def _file_record(path: Path) -> dict[str, object]:
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _read_test_series(inputs: SubmissionInputs) -> dict[str, tuple[str, ...]]:
    path = inputs.competition_root / inputs.contract.test_series_filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing verified submission input: {path}")
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.reader(file))
    if not rows or tuple(rows[0]) != inputs.contract.test_series_columns:
        raise ValueError("Test series columns do not match the verified contract")

    expected_ids = set(inputs.entity_ids)
    series_by_study: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != len(inputs.contract.test_series_columns):
            raise ValueError(f"Malformed row {row_number} in {path.name}")
        values = tuple(value.strip() for value in row)
        if any(not value for value in values):
            raise ValueError(f"Missing value at row {row_number} in {path.name}")
        study_id, series_id = values[:2]
        if study_id not in expected_ids:
            raise ValueError("Test series contains an unexpected study identifier")
        key = (study_id, series_id)
        if key in seen:
            raise ValueError("Test series contains duplicate study-series identifiers")
        seen.add(key)
        series_by_study.setdefault(study_id, []).append(series_id)
    return {
        study_id: tuple(series_ids) for study_id, series_ids in series_by_study.items()
    }


def _dicom_paths(series_root: Path) -> tuple[Path, ...]:
    try:
        return tuple(
            sorted(
                path
                for path in series_root.iterdir()
                if path.is_file() and path.suffix.casefold() in {".dcm", ".dicom"}
            )
        )
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return ()


def _is_decodable_slice(path: Path) -> bool:
    try:
        pixels = decode_pixels(path)
        return pixels.ndim == 2 and bool(pixels.size)
    except Exception:  # noqa: BLE001 - heterogeneous data failures become fallbacks
        return False
