import json
from pathlib import Path

import numpy as np
from pydicom import dcmread
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    MRImageStorage,
    generate_uid,
)

from rsna_knee.inspection import inspect_competition


def test_inspection_discovers_schema_and_writes_compact_findings(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    competition_root = input_root / "mounted-with-an-arbitrary-name"
    competition_root.mkdir(parents=True)
    (competition_root / "sample_submission.csv").write_text(
        "study_id,acl_tear,effusion\nS3,0.5,0.5\nS4,0.5,0.5\n"
    )
    (competition_root / "train.csv").write_text(
        "study_id,acl_tear,effusion\nS1,1,0\nS2,0,1\n"
    )
    (competition_root / "train_series.csv").write_text(
        "study_id,series_id,series_description\nS1,SE1,Sagittal PD\nS1,SE2,Coronal T2\n"
    )

    outcome = inspect_competition(input_root, tmp_path / "working")
    persisted = json.loads(outcome.artifact_path.read_text())

    projection = {
        "root": persisted["dataset"]["root"],
        "csv_paths": persisted["inventory"]["paths"]["csv"],
        "inventory_complete": persisted["inventory"]["complete"],
        "targets": persisted["targets"],
        "submission": persisted["sample_submission"],
        "identifier_columns": persisted["identifiers"]["table_columns"],
    }
    assert projection == {
        "root": str(competition_root),
        "csv_paths": [
            "sample_submission.csv",
            "train.csv",
            "train_series.csv",
        ],
        "inventory_complete": {
            "csv_paths": True,
            "image_paths": True,
            "image_roots": True,
            "json_paths": True,
            "report_paths": True,
            "submission_paths": True,
        },
        "targets": {
            "count": 2,
            "label_representation": {
                "acl_tear": ["0", "1"],
                "effusion": ["0", "1"],
            },
            "names": ["acl_tear", "effusion"],
            "ordering_source": "sample_submission.csv",
        },
        "submission": {
            "columns": ["study_id", "acl_tear", "effusion"],
            "identifier_columns": ["study_id"],
            "path": "sample_submission.csv",
            "row_count": 2,
        },
        "identifier_columns": {
            "sample_submission.csv": ["study_id"],
            "train.csv": ["study_id"],
            "train_series.csv": ["study_id", "series_id"],
        },
    }


def test_inspection_exercises_representative_dicom_and_report_data(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    competition_root = input_root / "competition-mount"
    competition_root.mkdir(parents=True)
    (competition_root / "sample_submission.csv").write_text(
        "study_id,acl_tear\nS9,0.5\n"
    )
    (competition_root / "train.csv").write_text("study_id,acl_tear\nS1,1\n")
    (competition_root / "test.csv").write_text("study_id\nS9\n")
    (competition_root / "train_series.csv").write_text(
        "study_id,series_id,series_description\nS1,SE1,Axial PD\nS2,SE2,Coronal T2\n"
    )
    (competition_root / "train_reports.csv").write_text(
        "study_id,language,report\nS1,en,No acute tear.\nS2,es,Sin rotura.\n"
    )

    axial = competition_root / "train_series" / "S1" / "SE1"
    coronal = competition_root / "train_series" / "S2" / "SE2"
    axial.mkdir(parents=True)
    coronal.mkdir(parents=True)
    _write_inspection_dicom(
        axial / "z.dcm",
        instance_number=1,
        position=(0.0, 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        pixels=np.array([[0, 1], [2, 3]], dtype=np.uint16),
    )
    _write_inspection_dicom(
        axial / "a.dcm",
        instance_number=2,
        position=(0.0, 0.0, 1.0),
        orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        pixels=np.array([[1, 2], [3, 4]], dtype=np.uint16),
    )
    _write_inspection_dicom(
        axial / "m.dcm",
        instance_number=3,
        position=(0.0, 0.0, 2.0),
        orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        pixels=np.array([[2, 3], [4, 5]], dtype=np.uint16),
    )
    for index, name in enumerate(("1.dcm", "2.dcm"), start=1):
        _write_inspection_dicom(
            coronal / name,
            instance_number=index,
            position=(0.0, float(index), 0.0),
            orientation=(1.0, 0.0, 0.0, 0.0, 0.0, -1.0),
            pixels=np.array([[index, index + 1], [index + 2, index + 3]]),
            photometric_interpretation="MONOCHROME1",
            slope=2.0,
            intercept=-10.0,
        )

    outcome = inspect_competition(input_root, tmp_path / "working")
    findings = outcome.findings
    visualization_paths = [
        outcome.artifact_path.parent / path
        for path in findings["visualizations"]["paths"]
    ]
    projection = {
        "image_count": findings["inventory"]["counts"]["image"],
        "image_roots": findings["inventory"]["image_roots"],
        "hierarchy": {
            name: findings["identifiers"]["path_hierarchy"][name]
            for name in (
                "image_count",
                "images_per_series",
                "relationships_verified",
                "series_count",
                "series_per_study",
                "study_count",
            )
        },
        "table_relationships": findings["identifiers"]["table_relationships"],
        "dimensions": findings["dicom"]["dimensions"],
        "photometric": findings["dicom"]["photometric_interpretations"],
        "slice_counts": findings["dicom"]["slice_counts"],
        "ordering": findings["dicom"]["ordering_comparisons"],
        "first_header": findings["dicom"]["header_samples"][0],
        "tables": findings["tables"],
        "preprocessing": {
            name: findings["preprocessing"][name]
            for name in (
                "all_finite",
                "decode_failures",
                "decoded_series",
                "deterministic_sampling",
                "monochrome1_tested",
                "normalized_range",
                "rescale_tested",
                "transform_evidence",
            )
        },
        "reports": findings["reports"],
        "visualizations_exist": all(path.is_file() for path in visualization_paths),
        "visualized_planes": findings["visualizations"]["planes"],
        "visualized_studies": findings["visualizations"]["study_ids"],
        "assumptions": {
            name: findings["assumptions"][name]
            for name in (
                "decoder_supports_observed_sample",
                "geometric_ordering_available",
                "identifier_hierarchy_is_study_series_image",
                "multilingual_reports",
                "percentile_normalization_is_suitable",
                "reports_are_available",
                "target_schema_from_sample_submission",
                "competition_rules_for_generative_ai",
                "dataset_size",
                "external_report_processing_rule",
                "internet_access_rule",
                "runtime_environment_captured",
                "runtime_limits",
                "submission_contract",
            )
        },
    }
    assert projection == {
        "image_count": 5,
        "image_roots": {"train_series": 5},
        "hierarchy": {
            "image_count": 5,
            "images_per_series": {"maximum": 3, "median": 2.5, "minimum": 2},
            "relationships_verified": True,
            "series_count": 2,
            "series_per_study": {"maximum": 1, "median": 1.0, "minimum": 1},
            "study_count": 2,
        },
        "table_relationships": {
            "train_series.csv": {
                "series_ids_matched": 2,
                "series_ids_sampled": 2,
                "study_series_pairs_matched": 2,
                "study_series_pairs_sampled": 2,
                "study_ids_matched": 2,
                "study_ids_sampled": 2,
            }
        },
        "dimensions": [[2, 2]],
        "photometric": ["MONOCHROME1", "MONOCHROME2"],
        "slice_counts": {"maximum": 3, "median": 2.5, "minimum": 2},
        "ordering": [
            {
                "filename_agrees_with_geometry": False,
                "geometry_agrees_with_instance": True,
                "orders": {
                    "filename": ["a.dcm", "m.dcm", "z.dcm"],
                    "geometry": ["z.dcm", "a.dcm", "m.dcm"],
                    "instance_number": ["z.dcm", "a.dcm", "m.dcm"],
                },
                "plane": "axial",
                "selected_method": "geometry",
                "series": "train_series/S1/SE1",
                "slice_count": 3,
            },
            {
                "filename_agrees_with_geometry": True,
                "geometry_agrees_with_instance": True,
                "orders": {
                    "filename": ["1.dcm", "2.dcm"],
                    "geometry": ["1.dcm", "2.dcm"],
                    "instance_number": ["1.dcm", "2.dcm"],
                },
                "plane": "coronal",
                "selected_method": "geometry",
                "series": "train_series/S2/SE2",
                "slice_count": 2,
            },
        ],
        "first_header": {
            "columns": 2,
            "instance_number": 2,
            "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "path": "train_series/S1/SE1/a.dcm",
            "photometric_interpretation": "MONOCHROME2",
            "position": [0.0, 0.0, 1.0],
            "rescale_intercept": 0.0,
            "rescale_slope": 1.0,
            "rows": 2,
            "transfer_syntax": "1.2.840.10008.1.2.1",
        },
        "tables": {
            "sample_submission.csv": {
                "columns": ["study_id", "acl_tear"],
                "identifier_columns": ["study_id"],
                "row_count": 1,
            },
            "train.csv": {
                "columns": ["study_id", "acl_tear"],
                "identifier_columns": ["study_id"],
                "row_count": 1,
            },
            "test.csv": {
                "columns": ["study_id"],
                "identifier_columns": ["study_id"],
                "row_count": 1,
            },
            "train_reports.csv": {
                "columns": ["study_id", "language", "report"],
                "identifier_columns": ["study_id"],
                "row_count": 2,
            },
            "train_series.csv": {
                "columns": ["study_id", "series_id", "series_description"],
                "identifier_columns": ["study_id", "series_id"],
                "row_count": 2,
            },
        },
        "preprocessing": {
            "all_finite": True,
            "decode_failures": [],
            "decoded_series": 2,
            "deterministic_sampling": True,
            "monochrome1_tested": True,
            "normalized_range": [0.0, 1.0],
            "rescale_tested": True,
            "transform_evidence": {
                "monochrome1": {
                    "observed_in_headers": 2,
                    "successful_files": 1,
                    "tested_files": 1,
                },
                "rescale": {
                    "observed_in_headers": 2,
                    "successful_files": 1,
                    "tested_files": 1,
                },
            },
        },
        "reports": {
            "available": True,
            "language_values": ["en", "es"],
            "paths": ["train_reports.csv"],
            "structures": {"train_reports.csv": ["study_id", "language", "report"]},
            "text_exported": False,
        },
        "visualizations_exist": True,
        "visualized_planes": ["axial", "coronal"],
        "visualized_studies": ["S1", "S2"],
        "assumptions": {
            "competition_rules_for_generative_ai": "unresolved",
            "decoder_supports_observed_sample": "verified",
            "dataset_size": "rejected",
            "external_report_processing_rule": "unresolved",
            "geometric_ordering_available": "verified",
            "identifier_hierarchy_is_study_series_image": "verified",
            "internet_access_rule": "unresolved",
            "multilingual_reports": "verified",
            "percentile_normalization_is_suitable": "unresolved",
            "reports_are_available": "verified",
            "runtime_environment_captured": "verified",
            "runtime_limits": "unresolved",
            "submission_contract": "verified",
            "target_schema_from_sample_submission": "verified",
        },
    }


def test_inspection_recognizes_dicom_uid_identifier_columns(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    competition_root = input_root / "competition-mount"
    competition_root.mkdir(parents=True)
    train_study = "1.2.826.0.1.1"
    train_series = "1.2.826.0.2.1"
    test_study = "1.2.826.0.1.2"
    test_series = "1.2.826.0.2.2"
    (competition_root / "sample_submission.csv").write_text(
        f"StudyInstanceUID,ACL,MCL\n{test_study},0.5,0.5\n"
    )
    (competition_root / "train.csv").write_text(
        f"StudyInstanceUID,Report,ACL,MCL\n{train_study},No tear.,1,0\n"
    )
    (competition_root / "test.csv").write_text(f"StudyInstanceUID\n{test_study}\n")
    (competition_root / "train_series.csv").write_text(
        "StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane\n"
        f"{train_study},{train_series},Axial\n"
    )
    (competition_root / "test_series.csv").write_text(
        "StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane\n"
        f"{test_study},{test_series},Coronal\n"
    )
    for image_root, study_id, series_id in (
        ("train_series", train_study, train_series),
        ("test_series", test_study, test_series),
    ):
        series_path = competition_root / image_root / study_id / series_id
        series_path.mkdir(parents=True)
        _write_inspection_dicom(
            series_path / "1.2.826.0.3.1.dcm",
            instance_number=1,
            position=(0.0, 0.0, 0.0),
            orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            pixels=np.array([[0, 1], [2, 3]], dtype=np.uint16),
        )

    findings = inspect_competition(input_root, tmp_path / "working").findings

    assert findings["targets"] == {
        "count": 2,
        "label_representation": {"ACL": ["1"], "MCL": ["0"]},
        "names": ["ACL", "MCL"],
        "ordering_source": "sample_submission.csv",
    }
    assert findings["sample_submission"]["identifier_columns"] == ["StudyInstanceUID"]
    assert findings["identifiers"]["table_columns"] == {
        "sample_submission.csv": ["StudyInstanceUID"],
        "test.csv": ["StudyInstanceUID"],
        "test_series.csv": ["StudyInstanceUID", "SeriesInstanceUID"],
        "train.csv": ["StudyInstanceUID"],
        "train_series.csv": ["StudyInstanceUID", "SeriesInstanceUID"],
    }
    assert findings["identifiers"]["table_relationships"] == {
        "test_series.csv": {
            "series_ids_matched": 1,
            "series_ids_sampled": 1,
            "study_series_pairs_matched": 1,
            "study_series_pairs_sampled": 1,
            "study_ids_matched": 1,
            "study_ids_sampled": 1,
        },
        "train_series.csv": {
            "series_ids_matched": 1,
            "series_ids_sampled": 1,
            "study_series_pairs_matched": 1,
            "study_series_pairs_sampled": 1,
            "study_ids_matched": 1,
            "study_ids_sampled": 1,
        },
    }
    assert (
        findings["assumptions"]["identifier_hierarchy_is_study_series_image"]
        == "verified"
    )
    assert findings["assumptions"]["submission_contract"] == "verified"


def test_inspection_discovers_rare_plane_without_csv_hints(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    competition_root = input_root / "competition-mount"
    competition_root.mkdir(parents=True)
    (competition_root / "sample_submission.csv").write_text(
        "study_id,acl_tear\nTEST,0.5\n"
    )
    for index in range(101):
        series_path = (
            competition_root / "train_series" / f"S{index:03}" / f"SE{index:03}"
        )
        series_path.mkdir(parents=True)
        is_coronal = index == 50
        _write_inspection_dicom(
            series_path / "1.dcm",
            instance_number=1,
            position=(0.0, 0.0, 0.0),
            orientation=(
                (1.0, 0.0, 0.0, 0.0, 0.0, -1.0)
                if is_coronal
                else (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
            ),
            pixels=np.array([[0, 1], [2, 3]], dtype=np.uint16),
        )

    outcome = inspect_competition(input_root, tmp_path / "working")

    assert outcome.findings["visualizations"]["planes"] == ["axial", "coronal"]


def test_representative_visualizations_span_available_studies(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    competition_root = input_root / "competition-mount"
    competition_root.mkdir(parents=True)
    (competition_root / "sample_submission.csv").write_text(
        "study_id,acl_tear\nTEST,0.5\n"
    )
    series = (
        ("S1", "SE1", "axial"),
        ("S1", "SE2", "coronal"),
        ("S1", "SE3", "sagittal"),
        ("S1", "SE4", "axial"),
        ("S2", "SE1", "axial"),
    )
    orientations = {
        "axial": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        "coronal": (1.0, 0.0, 0.0, 0.0, 0.0, -1.0),
        "sagittal": (0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    }
    for study_id, series_id, plane in series:
        series_path = competition_root / "train_series" / study_id / series_id
        series_path.mkdir(parents=True)
        _write_inspection_dicom(
            series_path / "1.dcm",
            instance_number=1,
            position=(0.0, 0.0, 0.0),
            orientation=orientations[plane],
            pixels=np.array([[0, 1], [2, 3]], dtype=np.uint16),
        )

    outcome = inspect_competition(input_root, tmp_path / "working")

    assert outcome.findings["visualizations"]["study_ids"] == ["S1", "S2"]
    assert outcome.findings["visualizations"]["planes"] == [
        "axial",
        "coronal",
        "sagittal",
    ]


def test_failed_series_is_not_reported_as_visualized(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    competition_root = input_root / "competition-mount"
    competition_root.mkdir(parents=True)
    (competition_root / "sample_submission.csv").write_text(
        "study_id,acl_tear\nTEST,0.5\n"
    )
    (competition_root / "train_series.csv").write_text(
        "study_id,series_id,series_description\nS1,SE1,Axial PD\nS2,SE2,Coronal T2\n"
    )
    valid_series = competition_root / "train_series" / "S1" / "SE1"
    invalid_series = competition_root / "train_series" / "S2" / "SE2"
    valid_series.mkdir(parents=True)
    invalid_series.mkdir(parents=True)
    _write_inspection_dicom(
        valid_series / "1.dcm",
        instance_number=1,
        position=(0.0, 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        pixels=np.array([[0, 1], [2, 3]], dtype=np.uint16),
    )
    (invalid_series / "1.dcm").write_bytes(b"not a DICOM")

    outcome = inspect_competition(input_root, tmp_path / "working")

    assert outcome.findings["visualizations"] == {
        "paths": ["ordered_slices_01_axial.png"],
        "planes": ["axial"],
        "study_ids": ["S1"],
    }


def test_failed_decode_uses_an_alternate_series_for_the_same_plane(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    competition_root = input_root / "competition-mount"
    competition_root.mkdir(parents=True)
    (competition_root / "sample_submission.csv").write_text(
        "study_id,acl_tear\nTEST,0.5\n"
    )
    planes = ("axial", "coronal", "sagittal", "axial", "sagittal", "coronal")
    broken_path: Path | None = None
    for index, plane in enumerate(planes):
        series_path = (
            competition_root / "train_series" / f"S{index:03}" / f"SE{index:03}"
        )
        series_path.mkdir(parents=True)
        orientation = {
            "axial": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            "coronal": (1.0, 0.0, 0.0, 0.0, 0.0, -1.0),
            "sagittal": (0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        }[plane]
        dicom_path = series_path / "1.dcm"
        _write_inspection_dicom(
            dicom_path,
            instance_number=1,
            position=(0.0, 0.0, 0.0),
            orientation=orientation,
            pixels=np.array([[0, 1], [2, 3]], dtype=np.uint16),
        )
        if index == 1:
            broken_path = dicom_path
    assert broken_path is not None
    broken = dcmread(broken_path)
    broken.PixelData = b"\x00\x00"
    broken.save_as(broken_path)

    outcome = inspect_competition(input_root, tmp_path / "working")

    assert outcome.findings["visualizations"]["planes"] == [
        "axial",
        "sagittal",
        "coronal",
    ]
    assert outcome.findings["preprocessing"]["decode_failures"]


def test_unobserved_relationships_and_mismatched_submission_are_not_verified(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    competition_root = input_root / "competition-mount"
    series_path = competition_root / "train_series" / "S1" / "SE1"
    series_path.mkdir(parents=True)
    (competition_root / "sample_submission.csv").write_text(
        "study_id,acl_tear\nEXPECTED,0.5\n"
    )
    (competition_root / "test.csv").write_text("study_id\nOTHER\n")
    _write_inspection_dicom(
        series_path / "1.dcm",
        instance_number=1,
        position=(0.0, 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        pixels=np.array([[0, 1], [2, 3]], dtype=np.uint16),
    )

    outcome = inspect_competition(input_root, tmp_path / "working")

    assert (
        outcome.findings["assumptions"]["identifier_hierarchy_is_study_series_image"]
        == "unresolved"
    )
    assert outcome.findings["assumptions"]["submission_contract"] == "rejected"


def _write_inspection_dicom(
    path: Path,
    *,
    instance_number: int,
    position: tuple[float, float, float],
    orientation: tuple[float, float, float, float, float, float],
    pixels: np.ndarray,
    photometric_interpretation: str = "MONOCHROME2",
    slope: float = 1.0,
    intercept: float = 0.0,
) -> None:
    sop_instance_uid = generate_uid()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dataset = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = MRImageStorage
    dataset.SOPInstanceUID = sop_instance_uid
    dataset.Modality = "MR"
    dataset.SeriesDescription = "Synthetic knee MRI"
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = photometric_interpretation
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.InstanceNumber = instance_number
    dataset.ImagePositionPatient = list(position)
    dataset.ImageOrientationPatient = list(orientation)
    dataset.RescaleSlope = slope
    dataset.RescaleIntercept = intercept
    dataset.PixelData = pixels.astype(np.uint16).tobytes()
    dataset.save_as(path, enforce_file_format=True)
