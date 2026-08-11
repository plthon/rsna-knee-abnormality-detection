import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydicom import dcmread
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from rsna_knee import (
    PreprocessingConfig,
    build_competition_index,
    indexing,
    preprocess_series,
    preprocess_series_result,
    verify_index_artifact,
)
from rsna_knee.competition_contract import load_verified_index_contract

_STUDY_1 = "1.2.826.0.1.3680043.8.498.1001"
_STUDY_2 = "1.2.826.0.1.3680043.8.498.1002"
_TEST_STUDY = "1.2.826.0.1.3680043.8.498.1003"
_GEOMETRY_SERIES = "1.2.826.0.1.3680043.8.498.2001"
_INSTANCE_SERIES = "1.2.826.0.1.3680043.8.498.2002"
_EMPTY_SERIES = "1.2.826.0.1.3680043.8.498.2003"
_FILENAME_SERIES = "1.2.826.0.1.3680043.8.498.2004"


def test_competition_index_is_ordered_compact_and_header_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "input"
    root = input_root / "competition"
    root.mkdir(parents=True)
    (root / "sample_submission.csv").write_text(
        f"StudyInstanceUID,ACL\n{_TEST_STUDY},0.5\n", encoding="utf-8"
    )
    _write_series_table(
        root / "train_series.csv",
        [
            (_STUDY_1, _GEOMETRY_SERIES, "Yes", "No", "Axial"),
            (_STUDY_1, _INSTANCE_SERIES, "Yes", "No", "Sagittal"),
            (_STUDY_2, _EMPTY_SERIES, "No", "No", "Coronal"),
        ],
    )
    _write_series_table(
        root / "test_series.csv",
        [(_TEST_STUDY, _FILENAME_SERIES, "Yes", "No", "Axial")],
    )

    geometry_path = root / "train_series" / _STUDY_1 / _GEOMETRY_SERIES
    geometry_path.mkdir(parents=True)
    for position in range(10):
        shape = (2, 4) if position == 5 else (2, 2)
        _write_dicom(
            geometry_path / f"{9 - position:02}.dcm",
            study_uid=_STUDY_1,
            series_uid=_GEOMETRY_SERIES,
            patient_id="P1",
            pixels=np.arange(np.prod(shape), dtype=np.uint16).reshape(shape) + position,
            instance_number=10 - position,
            position=(0.0, 0.0, float(position)),
        )

    instance_path = root / "train_series" / _STUDY_1 / _INSTANCE_SERIES
    instance_path.mkdir(parents=True)
    for filename, instance_number in (("c.dcm", 1), ("a.dcm", 3), ("b.dcm", 2)):
        _write_dicom(
            instance_path / filename,
            study_uid=_STUDY_1,
            series_uid=_INSTANCE_SERIES,
            patient_id="P1",
            pixels=np.arange(4, dtype=np.uint16).reshape(2, 2),
            instance_number=instance_number,
        )
    (instance_path / "corrupt.dcm").write_bytes(b"not a dicom")

    filename_path = root / "test_series" / _TEST_STUDY / _FILENAME_SERIES
    filename_path.mkdir(parents=True)
    for filename in ("b.dcm", "a.dcm"):
        _write_dicom(
            filename_path / filename,
            study_uid=_TEST_STUDY,
            series_uid=_FILENAME_SERIES,
            patient_id="PT1",
            pixels=np.arange(4, dtype=np.uint16).reshape(2, 2),
            include_rescale_metadata=filename != "a.dcm",
        )

    config = PreprocessingConfig(
        slice_count=8, lower_percentile=0, upper_percentile=100, image_size=8
    )
    work_root = tmp_path / "work"
    original_preprocess = indexing.preprocess_series_result  # type: ignore[attr-defined]
    preprocessing_calls = 0

    def assert_index_complete_before_decode(
        paths: list[Path], passed_config: PreprocessingConfig
    ) -> Any:
        nonlocal preprocessing_calls
        preprocessing_calls += 1
        artifact_dir = work_root / "rsna-knee-index-v1"
        assert _read_csv(artifact_dir / "slices.csv")
        assert _read_csv(artifact_dir / "selected_slices.csv")
        return original_preprocess(paths, passed_config)

    monkeypatch.setattr(
        indexing, "preprocess_series_result", assert_index_complete_before_decode
    )
    outcome = build_competition_index(input_root, work_root, config)

    studies = _read_csv(outcome.artifact_dir / "studies.csv")
    series = _read_csv(outcome.artifact_dir / "series.csv")
    slices = _read_csv(outcome.artifact_dir / "slices.csv")
    selected = _read_csv(outcome.artifact_dir / "selected_slices.csv")
    errors = _read_csv(outcome.artifact_dir / "errors.csv")
    manifest = json.loads(outcome.manifest_path.read_text())

    assert preprocessing_calls == 3
    assert [(row["split"], row["StudyInstanceUID"]) for row in studies] == [
        ("train", _STUDY_1),
        ("train", _STUDY_2),
        ("test", _TEST_STUDY),
    ]
    assert [row["ordering_method"] for row in series] == [
        "geometry",
        "instance_number",
        "none",
        "filename",
    ]
    assert [row["selection_mask"] for row in series] == [
        "11111111",
        "11100000",
        "00000000",
        "11000000",
    ]

    geometry_slices = [row for row in slices if row["series_index"] == "0"]
    assert [row["filename"] for row in geometry_slices] == [
        "09.dcm",
        "08.dcm",
        "07.dcm",
        "06.dcm",
        "05.dcm",
        "04.dcm",
        "03.dcm",
        "02.dcm",
        "01.dcm",
        "00.dcm",
    ]
    assert [
        int(row["ordered_position"]) for row in selected if row["series_index"] == "0"
    ] == [0, 1, 2, 3, 5, 6, 7, 9]
    assert {row["code"] for row in errors} >= {
        "corrupt_dicom_header",
        "empty_series",
        "inconsistent_dimensions",
        "ordering_instance_number_fallback",
        "ordering_filename_fallback",
        "rescale_metadata_fallback",
    }
    assert [
        row["detail"] for row in errors if row["code"] == "rescale_metadata_fallback"
    ] == ["RescaleSlope:missing:1;RescaleIntercept:missing:0"]
    assert outcome.statistics["transfer_syntaxes"] == {str(ExplicitVRLittleEndian): 15}
    assert outcome.statistics["preprocessing"]["representative_series_checked"] == 3
    assert outcome.statistics["preprocessing"]["representative_studies_checked"] == 2
    assert outcome.statistics["preprocessing"]["representative_series_succeeded"] == 3
    assert manifest["join_fields"] == ["StudyInstanceUID", "SeriesInstanceUID"]
    assert manifest["competition_contract"] == {
        "schema_version": 1,
        "verified_at": "2026-08-09",
        "sha256": hashlib.sha256(
            Path("contracts/competition.json").read_bytes()
        ).hexdigest(),
    }
    assert manifest["decoder_dependencies"] == ["pydicom"]
    assert manifest["decoded_images_persisted"] is False
    assert set(manifest["files"]) == {
        "studies.csv",
        "series.csv",
        "slices.csv",
        "selected_slices.csv",
        "errors.csv",
        "statistics.json",
    }
    assert manifest["files"]["selected_slices.csv"]["columns"] == [
        "series_index",
        "slot",
        "slice_index",
        "ordered_position",
    ]
    assert manifest["files"]["errors.csv"]["columns"] == [
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
    assert not list(outcome.artifact_dir.glob("*.npy"))
    assert not list(outcome.artifact_dir.glob("*.npz"))
    assert not list(outcome.artifact_dir.glob("*.png"))
    assert verify_index_artifact(outcome.artifact_dir) == outcome.statistics


def test_index_contract_rejects_schema_drift(tmp_path: Path) -> None:
    contract = json.loads(Path("contracts/competition.json").read_text())
    contract["data"]["tables"]["train_series.csv"]["columns"].append("PatientSex")
    path = tmp_path / "drifted-contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected columns"):
        load_verified_index_contract(path)


def test_shared_preprocessing_resizes_pads_and_masks_decode_failures(
    tmp_path: Path,
) -> None:
    valid_path = tmp_path / "valid.dcm"
    broken_path = tmp_path / "broken.dcm"
    pixels = np.arange(8, dtype=np.uint16).reshape(2, 4)
    _write_dicom(
        valid_path,
        study_uid=_STUDY_1,
        series_uid=_GEOMETRY_SERIES,
        patient_id="P1",
        pixels=pixels,
        slope=2.0,
        intercept=-10.0,
        photometric_interpretation="MONOCHROME1",
    )
    _write_dicom(
        broken_path,
        study_uid=_STUDY_1,
        series_uid=_GEOMETRY_SERIES,
        patient_id="P1",
        pixels=pixels,
    )
    broken = dcmread(broken_path)
    broken.PixelData = b"\x00\x00"
    broken.save_as(broken_path)

    config = PreprocessingConfig(
        slice_count=4, lower_percentile=0, upper_percentile=100, image_size=8
    )
    result = preprocess_series_result([valid_path, broken_path], config)
    ordinary_pixels, ordinary_mask = preprocess_series(
        [valid_path, broken_path], config
    )

    assert result.pixels.shape == (4, 8, 8)
    np.testing.assert_array_equal(result.mask, [True, False, False, False])
    assert [(failure.path, failure.code) for failure in result.failures] == [
        (broken_path, "pixel_decode_failed")
    ]
    assert not result.pixels[0, :2].any()
    assert not result.pixels[0, 6:].any()
    assert result.pixels[0, 2, 0] == 1.0
    assert result.pixels[0, 5, -1] == 0.0
    np.testing.assert_array_equal(ordinary_pixels, result.pixels)
    np.testing.assert_array_equal(ordinary_mask, result.mask)


def _write_series_table(path: Path, rows: list[tuple[str, str, str, str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "Fluid_Sensitive",
                "Fat_Suppression",
                "Anatomical_Plane",
            ]
        )
        writer.writerows(rows)


def _write_dicom(
    path: Path,
    *,
    study_uid: str,
    series_uid: str,
    patient_id: str,
    pixels: np.ndarray,
    instance_number: int | None = None,
    position: tuple[float, float, float] | None = None,
    slope: float = 1.0,
    intercept: float = 0.0,
    photometric_interpretation: str = "MONOCHROME2",
    include_rescale_metadata: bool = True,
) -> None:
    sop_uid = generate_uid()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = MRImageStorage
    dataset.SOPInstanceUID = sop_uid
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.PatientID = patient_id
    dataset.Modality = "MR"
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = photometric_interpretation
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    if include_rescale_metadata:
        dataset.RescaleSlope = slope
        dataset.RescaleIntercept = intercept
    if instance_number is not None:
        dataset.InstanceNumber = instance_number
    if position is not None:
        dataset.ImagePositionPatient = list(position)
        dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.PixelData = pixels.astype(np.uint16).tobytes()
    dataset.save_as(path, enforce_file_format=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))
