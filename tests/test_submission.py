import csv
import json
from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    SecondaryCaptureImageStorage,
    generate_uid,
)

from rsna_knee.submission import (
    discover_submission_inputs,
    run_constant_baseline,
    validate_submission,
    write_submission,
)

PROJECT_ROOT = Path(__file__).parents[1]
CONTRACT_PATH = PROJECT_ROOT / "contracts" / "competition.json"
TARGETS = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)


def _write_submission_inputs(
    root: Path,
    study_ids: tuple[str, ...],
    *,
    unusable_ids: tuple[str, ...] = (),
) -> Path:
    competition_root = root / "competitions" / "rsna-knee-abnormality-detection"
    competition_root.mkdir(parents=True)
    with (competition_root / "test.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["StudyInstanceUID"])
        writer.writerows((study_id,) for study_id in study_ids)
    with (competition_root / "sample_submission.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("StudyInstanceUID", *TARGETS))
        writer.writerows((study_id, *(0.5 for _ in TARGETS)) for study_id in study_ids)
    with (competition_root / "test_series.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            (
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "Fluid_Sensitive",
                "Fat_Suppression",
                "Anatomical_Plane",
            )
        )
        for study_id in study_ids:
            series_id = f"series-{study_id}"
            writer.writerow((study_id, series_id, "Yes", "No", "Sagittal"))
            series_root = competition_root / "test_series" / study_id / series_id
            series_root.mkdir(parents=True)
            dicom_path = series_root / "slice.dcm"
            if study_id in unusable_ids:
                dicom_path.write_bytes(b"not a DICOM")
            else:
                _write_dicom(dicom_path)
    return competition_root


def _write_dicom(path: Path) -> None:
    sop_instance_uid = generate_uid()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = sop_instance_uid
    dataset.Rows = 2
    dataset.Columns = 2
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = np.arange(4, dtype=np.uint16).tobytes()
    dataset.save_as(path, enforce_file_format=True)


def _prediction_row(study_id: str, value: object = 0.5) -> list[object]:
    return [study_id, *(value for _ in TARGETS)]


def _write_raw_submission(
    path: Path, header: tuple[str, ...], rows: list[list[object]]
) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def test_constant_baseline_discovers_entities_and_writes_validated_artifacts(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    work_root = tmp_path / "working"
    _write_submission_inputs(
        input_root,
        ("study-c", "study-a", "study-d", "study-b"),
        unusable_ids=("study-d",),
    )

    outcome = run_constant_baseline(
        input_root=input_root,
        work_root=work_root,
        probability=0.25,
        contract_path=CONTRACT_PATH,
        run_id="constant-baseline-local-test",
    )

    with outcome.submission_path.open(newline="") as file:
        submission_rows = list(csv.reader(file))
    validation = json.loads(outcome.validation_report_path.read_text())
    manifest = json.loads(outcome.manifest_path.read_text())

    assert submission_rows == [
        ["StudyInstanceUID", *TARGETS],
        ["study-c", *("0.25" for _ in TARGETS)],
        ["study-a", *("0.25" for _ in TARGETS)],
        ["study-d", *("0.25" for _ in TARGETS)],
        ["study-b", *("0.25" for _ in TARGETS)],
    ]
    assert validation == {
        "columns": ["StudyInstanceUID", *TARGETS],
        "competition_contract_sha256": validation["competition_contract_sha256"],
        "fallbacks": {"count": 1, "path": "submission-fallbacks.csv"},
        "identifier_column": "StudyInstanceUID",
        "probabilities": {"maximum": 0.25, "minimum": 0.25},
        "row_count": 4,
        "schema_version": 1,
        "submission_sha256": validation["submission_sha256"],
        "target_count": 12,
        "valid": True,
    }
    assert validation["competition_contract_sha256"]
    assert validation["submission_sha256"]
    assert outcome.fallbacks_path.read_text().splitlines() == [
        "StudyInstanceUID,reason",
        "study-d,no_decodable_slices",
    ]
    assert manifest["run_id"] == "constant-baseline-local-test"
    assert manifest["configuration"]["probability"] == 0.25
    assert manifest["configuration"]["target_order"] == list(TARGETS)
    assert set(manifest["source"]["inputs"]) == {
        "sample_submission.csv",
        "test.csv",
        "test_series.csv",
    }
    assert all(
        record["sha256"] and record["size_bytes"] > 0
        for record in manifest["source"]["inputs"].values()
    )
    assert {record["path"] for record in manifest["artifacts"]} == {
        "submission.csv",
        "submission-fallbacks.csv",
        "submission-validation.json",
    }


def test_unusable_studies_receive_an_explicit_recorded_fixed_fallback(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    _write_submission_inputs(input_root, ("usable-a", "unusable", "usable-b"))
    inputs = discover_submission_inputs(input_root, contract_path=CONTRACT_PATH)

    result = write_submission(
        inputs=inputs,
        work_root=tmp_path / "working",
        predictions={
            "usable-a": (0.8,) * len(TARGETS),
            "usable-b": (0.2,) * len(TARGETS),
        },
        failures={"unusable": "no_usable_series"},
        fallback_probability=0.1,
    )

    with result.submission_path.open(newline="") as file:
        rows = list(csv.reader(file))
    assert rows[2] == ["unusable", *("0.1" for _ in TARGETS)]
    assert result.fallbacks_path.read_text().splitlines() == [
        "StudyInstanceUID,reason",
        "unusable,no_usable_series",
    ]
    report = json.loads(result.validation_report_path.read_text())
    assert report["valid"] is True
    assert report["fallbacks"] == {
        "count": 1,
        "path": "submission-fallbacks.csv",
    }


@pytest.mark.parametrize(
    ("header", "rows", "error"),
    [
        (
            ("StudyInstanceUID", *TARGETS),
            [_prediction_row("study-a"), _prediction_row("study-c")],
            "identifiers do not match",
        ),
        (
            ("StudyInstanceUID", *TARGETS),
            [_prediction_row("study-a"), _prediction_row("study-a")],
            "duplicate identifiers",
        ),
        (
            ("StudyInstanceUID", TARGETS[1], TARGETS[0], *TARGETS[2:]),
            [_prediction_row("study-a"), _prediction_row("study-b")],
            "columns or order",
        ),
        (
            ("StudyInstanceUID", *TARGETS),
            [_prediction_row("study-a")],
            "row count",
        ),
        (
            ("StudyInstanceUID", *TARGETS),
            [_prediction_row("study-a", "nan"), _prediction_row("study-b")],
            "finite",
        ),
        (
            ("StudyInstanceUID", *TARGETS),
            [_prediction_row("study-a", "inf"), _prediction_row("study-b")],
            "finite",
        ),
        (
            ("StudyInstanceUID", *TARGETS),
            [_prediction_row("study-a", 1.01), _prediction_row("study-b")],
            "outside \\[0,1\\]",
        ),
    ],
    ids=(
        "missing-identifier",
        "duplicate-identifier",
        "wrong-columns-order",
        "wrong-row-count",
        "nan",
        "infinity",
        "out-of-bounds",
    ),
)
def test_submission_validation_rejects_every_invalid_contract_class(
    tmp_path: Path,
    header: tuple[str, ...],
    rows: list[list[object]],
    error: str,
) -> None:
    input_root = tmp_path / "input"
    _write_submission_inputs(input_root, ("study-a", "study-b"))
    inputs = discover_submission_inputs(input_root, contract_path=CONTRACT_PATH)
    submission_path = tmp_path / "submission.csv"
    _write_raw_submission(submission_path, header, rows)

    with pytest.raises(ValueError, match=error):
        validate_submission(submission_path, inputs=inputs)
