import csv
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    SecondaryCaptureImageStorage,
    generate_uid,
)

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


from rsna_knee.combined_supervision import (
    PUBLIC_VERDICT_TO_LABEL,
    derive_combined_cohort,
    derive_patient_split,
    load_combined_baseline_config,
)

pytestmark = pytest.mark.training

PROJECT_ROOT = Path(__file__).parents[1]
COMPETITION_CONTRACT = PROJECT_ROOT / "contracts" / "competition.json"
TARGETS = tuple(json.loads(COMPETITION_CONTRACT.read_text())["targets"]["order"])
TRAIN_COLUMNS = tuple(
    json.loads(COMPETITION_CONTRACT.read_text())["data"]["tables"]["train.csv"][
        "columns"
    ]
)
STUDY_COLUMNS = (
    "study_index",
    "split",
    "StudyInstanceUID",
    "PatientID",
    "series_count",
    "usable_series_count",
    "slice_count",
    "usable_slice_count",
)
SERIES_COLUMNS = (
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
)
SLICE_COLUMNS = (
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
)
SELECTED_COLUMNS = ("series_index", "slot", "slice_index", "ordered_position")


def test_public_verdict_mapping() -> None:
    assert PUBLIC_VERDICT_TO_LABEL["YES"] == 1
    assert PUBLIC_VERDICT_TO_LABEL["NO"] == 0
    assert PUBLIC_VERDICT_TO_LABEL["UNK"] is None
    assert "MAYBE" not in PUBLIC_VERDICT_TO_LABEL


def test_combined_cohort_construction_and_gold_precedence(tmp_path: Path) -> None:
    comp_root, pub_root, idx_root, pub_contract, idx_contract = _combined_fixture(
        tmp_path
    )

    cohort = derive_combined_cohort(
        comp_root,
        pub_root,
        idx_root,
        public_dataset_contract_path=pub_contract,
        data_index_contract_path=idx_contract,
    )

    assert cohort.total_study_count == 3
    assert cohort.gold_study_count == 1
    assert cohort.non_gold_study_count == 2
    assert cohort.target_names == TARGETS

    by_uid = {s.study_uid: s for s in cohort.studies}

    # study-gold has organizer 1 for target 0, but public verdict has NO. Organizer gold must win!
    gold_study = by_uid["study-gold"]
    assert gold_study.provenance == "gold"
    assert gold_study.labels[0] == 1
    assert gold_study.patient_id == "patient-gold"

    # study-public-1 is non-gold, uses public verdicts (YES for target 0, UNK for target 1)
    pub1 = by_uid["study-public-1"]
    assert pub1.provenance == "public"
    assert pub1.labels[0] == 1
    assert pub1.labels[1] is None
    assert pub1.patient_id == "patient-pub1"

    # study-public-2 is non-gold, uses public verdicts (NO for target 0, YES for target 1)
    pub2 = by_uid["study-public-2"]
    assert pub2.provenance == "public"
    assert pub2.labels[0] == 0
    assert pub2.labels[1] == 1
    assert pub2.patient_id == "patient-pub2"


def test_patient_disjoint_split_guarantee(tmp_path: Path) -> None:
    comp_root, pub_root, idx_root, pub_contract, idx_contract = _combined_fixture(
        tmp_path
    )
    cohort = derive_combined_cohort(
        comp_root,
        pub_root,
        idx_root,
        public_dataset_contract_path=pub_contract,
        data_index_contract_path=idx_contract,
    )

    partition = derive_patient_split(cohort, split_seed=20260825, val_fraction=0.4)

    assert len(partition.train_studies) + len(partition.val_studies) == len(
        cohort.studies
    )
    assert partition.train_patients.isdisjoint(partition.val_patients)

    # Every study in train has patient in train_patients
    assert all(
        s.patient_id in partition.train_patients for s in partition.train_studies
    )
    # Every study in val has patient in val_patients
    assert all(s.patient_id in partition.val_patients for s in partition.val_studies)


def test_public_label_drift_rejected(tmp_path: Path) -> None:
    comp_root, pub_root, idx_root, pub_contract, idx_contract = _combined_fixture(
        tmp_path
    )
    csv_file = pub_root / "report_labels_v2.csv"
    csv_file.write_text(csv_file.read_text() + "\nextra_bad_data")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        derive_combined_cohort(
            comp_root,
            pub_root,
            idx_root,
            public_dataset_contract_path=pub_contract,
            data_index_contract_path=idx_contract,
        )


def test_missing_non_gold_study_in_public_dataset_raises(tmp_path: Path) -> None:
    comp_root, pub_root, idx_root, pub_contract, idx_contract = _combined_fixture(
        tmp_path
    )
    # Remove study-public-1 from public dataset
    csv_file = pub_root / "report_labels_v2.csv"
    lines = [
        line
        for line in csv_file.read_text().splitlines()
        if "study-public-1" not in line
    ]
    csv_file.write_text("\n".join(lines))
    # Update contract hash to bypass sha check
    contract_data = json.loads(pub_contract.read_text())
    contract_data["file_hashes"]["report_labels_v2.csv"] = _sha256(csv_file)
    pub_contract.write_text(json.dumps(contract_data))

    with pytest.raises(ValueError, match="missing from public label dataset"):
        derive_combined_cohort(
            comp_root,
            pub_root,
            idx_root,
            public_dataset_contract_path=pub_contract,
            data_index_contract_path=idx_contract,
        )


def test_invalid_verdict_raises(tmp_path: Path) -> None:
    comp_root, pub_root, idx_root, pub_contract, idx_contract = _combined_fixture(
        tmp_path
    )
    csv_file = pub_root / "report_labels_v2.csv"
    content = csv_file.read_text().replace("YES", "INVALID_VALUE")
    csv_file.write_text(content)
    contract_data = json.loads(pub_contract.read_text())
    contract_data["file_hashes"]["report_labels_v2.csv"] = _sha256(csv_file)
    pub_contract.write_text(json.dumps(contract_data))

    with pytest.raises(ValueError, match="Invalid public verdict"):
        derive_combined_cohort(
            comp_root,
            pub_root,
            idx_root,
            public_dataset_contract_path=pub_contract,
            data_index_contract_path=idx_contract,
        )


def test_public_score_and_confidence_are_ignored(tmp_path: Path) -> None:
    comp_root, pub_root, idx_root, pub_contract, idx_contract = _combined_fixture(
        tmp_path
    )
    cohort1 = derive_combined_cohort(
        comp_root,
        pub_root,
        idx_root,
        public_dataset_contract_path=pub_contract,
        data_index_contract_path=idx_contract,
    )

    # Change score and confidence values in public CSV
    csv_file = pub_root / "report_labels_v2.csv"
    content = csv_file.read_text().replace("0.82", "0.01").replace("0.95", "0.02")
    csv_file.write_text(content)
    contract_data = json.loads(pub_contract.read_text())
    contract_data["file_hashes"]["report_labels_v2.csv"] = _sha256(csv_file)
    pub_contract.write_text(json.dumps(contract_data))

    cohort2 = derive_combined_cohort(
        comp_root,
        pub_root,
        idx_root,
        public_dataset_contract_path=pub_contract,
        data_index_contract_path=idx_contract,
    )

    # Labels and digest must be completely identical
    assert cohort1.digest == cohort2.digest
    assert [s.labels for s in cohort1.studies] == [s.labels for s in cohort2.studies]


def test_duplicate_study_rejected_in_train_csv(tmp_path: Path) -> None:
    comp_root, pub_root, idx_root, pub_contract, idx_contract = _combined_fixture(
        tmp_path
    )
    train_csv = comp_root / "train.csv"
    dup_row = _train_row("study-gold", [1] * len(TARGETS))
    with train_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRAIN_COLUMNS)
        writer.writerow(dup_row)

    with pytest.raises(ValueError, match="Duplicate study in competition train.csv"):
        derive_combined_cohort(
            comp_root,
            pub_root,
            idx_root,
            public_dataset_contract_path=pub_contract,
            data_index_contract_path=idx_contract,
        )


def test_duplicate_study_rejected_in_public_csv(tmp_path: Path) -> None:
    comp_root, pub_root, idx_root, pub_contract, idx_contract = _combined_fixture(
        tmp_path
    )
    pub_csv = pub_root / "report_labels_v2.csv"
    lines = pub_csv.read_text().splitlines()
    lines.append(lines[1])  # duplicate first data row
    pub_csv.write_text("\n".join(lines))
    contract_data = json.loads(pub_contract.read_text())
    contract_data["file_hashes"]["report_labels_v2.csv"] = _sha256(pub_csv)
    pub_contract.write_text(json.dumps(contract_data))

    with pytest.raises(ValueError, match="Duplicate study in public label dataset"):
        derive_combined_cohort(
            comp_root,
            pub_root,
            idx_root,
            public_dataset_contract_path=pub_contract,
            data_index_contract_path=idx_contract,
        )


def test_target_order_mismatch_rejected(tmp_path: Path) -> None:
    comp_root, pub_root, idx_root, pub_contract, idx_contract = _combined_fixture(
        tmp_path
    )
    contract_data = json.loads(pub_contract.read_text())
    contract_data["target_order"] = list(reversed(TARGETS))
    pub_contract.write_text(json.dumps(contract_data))

    with pytest.raises(ValueError, match="Public dataset target order differs"):
        derive_combined_cohort(
            comp_root,
            pub_root,
            idx_root,
            public_dataset_contract_path=pub_contract,
            data_index_contract_path=idx_contract,
        )


def test_load_combined_baseline_config() -> None:
    config = load_combined_baseline_config(
        PROJECT_ROOT / "configs/combined-baseline.toml"
    )
    assert config.epochs == 3
    assert config.batch_size == 16
    assert config.learning_rate == 0.0001
    assert config.weight_decay == 0.0001
    assert config.mixed_precision is True
    assert config.architecture == "torchvision.models.resnet18"
    assert config.weight_identity == "ResNet18_Weights.IMAGENET1K_V1"
    assert config.val_fraction == 0.10
    assert config.split_seed == 20260825
    assert config.reload_absolute_tolerance == 0.00001


def test_verify_combined_baseline_artifact(tmp_path: Path) -> None:
    from rsna_knee.combined_supervision import verify_combined_baseline_artifact

    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()

    (artifact_dir / "checkpoint.pt").write_bytes(b"dummy_checkpoint_data")
    (artifact_dir / "submission.csv").write_text("StudyInstanceUID,ACL\nstudy-1,0.5\n")
    (artifact_dir / "submission-fallbacks.csv").write_text("StudyInstanceUID,reason\n")
    (artifact_dir / "submission-validation.json").write_text(
        json.dumps({"valid": True})
    )
    (artifact_dir / "training-report.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "cohort": {"digest": "test-digest", "total_studies": 100},
                "checkpoint": {"reload_reproducible": True},
                "training_execution": {"epochs_completed": 3},
            }
        )
    )
    (artifact_dir / "run-manifest.json").write_text(
        json.dumps({"run_id": "test-run-123", "schema_version": 1})
    )

    verified = verify_combined_baseline_artifact(artifact_dir)
    assert verified["valid"] is True
    assert verified["run_id"] == "test-run-123"
    assert verified["cohort_digest"] == "test-digest"
    assert verified["total_studies"] == 100
    assert verified["completed_epochs"] == 3


def test_load_combined_checkpoint_validates_structure(tmp_path: Path) -> None:
    from rsna_knee.combined_supervision import load_combined_checkpoint
    from rsna_knee.training import _new_uninitialized_model

    model = _new_uninitialized_model(len(TARGETS))
    ckpt_path = tmp_path / "checkpoint.pt"
    valid_payload = {
        "checkpoint_schema_version": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "scaler_state_dict": {},
        "completed_epochs": 3,
        "targets": list(TARGETS),
        "priors": [0.5] * len(TARGETS),
        "preprocessing": {},
        "cohort": {},
        "partition": {},
        "initialization": {},
        "configuration": {},
        "code": {},
        "runtime": {},
        "artifact_hashes": {},
        "rng_state": {},
    }
    torch.save(valid_payload, ckpt_path)

    loaded_model, loaded_ckpt = load_combined_checkpoint(
        ckpt_path, device=torch.device("cpu")
    )
    assert loaded_model is not None
    assert len(loaded_ckpt["targets"]) == len(TARGETS)
    assert loaded_ckpt["completed_epochs"] == 3

    # Incomplete keys rejected
    bad_payload = dict(valid_payload)
    del bad_payload["partition"]
    torch.save(bad_payload, ckpt_path)
    with pytest.raises(ValueError, match="Combined checkpoint fields are incomplete"):
        load_combined_checkpoint(ckpt_path, device=torch.device("cpu"))


def test_run_combined_baseline_with_pretrained_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rsna_knee.combined_supervision import (
        load_combined_baseline_config,
        run_combined_baseline,
    )
    from rsna_knee.training import _new_uninitialized_model

    comp_root, pub_root, idx_root, pub_contract, idx_contract = _combined_fixture(
        tmp_path
    )
    model = _new_uninitialized_model(len(TARGETS))
    ckpt_path = tmp_path / "pretrained_checkpoint.pt"
    valid_payload = {
        "checkpoint_schema_version": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "scaler_state_dict": {},
        "completed_epochs": 3,
        "targets": list(TARGETS),
        "priors": [0.5] * len(TARGETS),
        "preprocessing": {
            "slice_count": 1,
            "lower_percentile": 1.0,
            "upper_percentile": 99.0,
            "image_size": 8,
            "normalization_status": "provisional",
            "resize": "bilinear_aspect_preserving_center_pad",
            "shared_contract": "preprocess_series_result",
            "channel_adapter": "grayscale_repeat_to_rgb",
            "imagenet_mean": [0.485, 0.456, 0.406],
            "imagenet_std": [0.229, 0.224, 0.225],
        },
        "cohort": {},
        "partition": {},
        "initialization": {"artifact_sha256": "0" * 64},
        "configuration": {},
        "code": {},
        "runtime": {},
        "artifact_hashes": {},
        "rng_state": {},
    }
    torch.save(valid_payload, ckpt_path)

    # Make dummy weight file for discover_combined_training_inputs
    weight_dir = tmp_path / "torchvision-resnet18-imagenet1k-v1"
    weight_dir.mkdir()
    torch.save(model.state_dict(), weight_dir / "resnet18-f37072fd.pth")

    input_root = tmp_path / "kaggle_input"
    input_root.mkdir()
    # Symlink or copy competition root into input_root / competitions / ...
    import shutil

    shutil.copytree(
        comp_root,
        input_root / "competitions" / "rsna-knee-abnormality-detection",
        dirs_exist_ok=True,
    )
    shutil.copytree(
        pub_root,
        input_root / "datasets" / "pilkwang" / "rsna-knee-llm-labels",
        dirs_exist_ok=True,
    )
    shutil.copytree(
        idx_root,
        input_root / "datasets" / "plthon" / "rsna-knee-data-index",
        dirs_exist_ok=True,
    )
    shutil.copytree(
        weight_dir,
        input_root / "datasets" / "plthon" / "torchvision-resnet18-imagenet1k-v1",
        dirs_exist_ok=True,
    )

    raw_config = load_combined_baseline_config(
        PROJECT_ROOT / "configs/combined-baseline.toml"
    )
    from dataclasses import replace

    config = replace(raw_config, val_fraction=0.3)

    monkeypatch.setattr(
        "rsna_knee.combined_supervision.require_approved_cuda",
        lambda approved: {"approved": True, "device_name": "CPU"},
    )

    outcome = run_combined_baseline(
        input_root=input_root,
        work_root=tmp_path / "working",
        config=config,
        approved_gpu=True,
        pretrained_checkpoint_path=ckpt_path,
        index_artifact_dir=idx_root,
        public_dataset_root=pub_root,
        weight_path=weight_dir / "resnet18-f37072fd.pth",
        public_dataset_contract_path=pub_contract,
        data_index_contract_path=idx_contract,
    )
    assert outcome.checkpoint_path.is_file()
    assert outcome.submission_path.is_file()
    assert outcome.report_path.is_file()


def _write_dicom(path: Path) -> None:
    sop_instance_uid = generate_uid()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = sop_instance_uid
    dataset.Rows = 8
    dataset.Columns = 8
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = np.arange(64, dtype=np.uint16).tobytes()
    dataset.save_as(path, enforce_file_format=True)


def _combined_fixture(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    competition_root = root / "competition"
    competition_root.mkdir()
    train_rows = [
        _train_row("study-gold", [1] * len(TARGETS)),
        _train_row("study-public-1", [None] * len(TARGETS)),
        _train_row("study-public-2", [None] * len(TARGETS)),
    ]
    pub_rows = [
        _public_row("study-gold", ["NO"] * len(TARGETS)),
        _public_row("study-public-1", ["YES", "UNK"] + ["NO"] * (len(TARGETS) - 2)),
        _public_row("study-public-2", ["NO", "YES"] + ["NO"] * (len(TARGETS) - 2)),
    ]
    study_rows = [
        _study_row(0, "study-gold", "patient-gold", "train"),
        _study_row(1, "study-public-1", "patient-pub1", "train"),
        _study_row(2, "study-public-2", "patient-pub2", "train"),
        _study_row(3, "test-study-1", "patient-test1", "test"),
    ]
    series_specs = [
        (0, "gold", "train"),
        (1, "pub1", "train"),
        (2, "pub2", "train"),
        (3, "test1", "test"),
    ]

    _write_csv(competition_root / "train.csv", TRAIN_COLUMNS, train_rows)
    _write_csv(
        competition_root / "test.csv",
        ("StudyInstanceUID",),
        [{"StudyInstanceUID": "test-study-1"}],
    )
    sample_sub_row: dict[str, object] = {"StudyInstanceUID": "test-study-1"}
    sample_sub_row.update({t: "0.5" for t in TARGETS})
    _write_csv(
        competition_root / "sample_submission.csv",
        ("StudyInstanceUID", *TARGETS),
        [sample_sub_row],
    )
    _write_csv(
        competition_root / "test_series.csv",
        (
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "Fluid_Sensitive",
            "Fat_Suppression",
            "Anatomical_Plane",
        ),
        [
            {
                "StudyInstanceUID": "test-study-1",
                "SeriesInstanceUID": "series-test1",
                "Fluid_Sensitive": "Yes",
                "Fat_Suppression": "No",
                "Anatomical_Plane": "Sagittal",
            }
        ],
    )

    public_root = root / "public_dataset"
    public_root.mkdir()
    public_cols = ["StudyInstanceUID"]
    for t in TARGETS:
        public_cols.extend([t, f"{t}__conf", f"{t}__verdict"])
    pub_csv = public_root / "report_labels_v2.csv"
    _write_csv(pub_csv, public_cols, pub_rows)

    index_root = root / "index"
    index_root.mkdir()
    series_rows: list[dict[str, object]] = []
    slice_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for series_index, (study_index, suffix, split) in enumerate(series_specs):
        series_rows.append(
            {
                "series_index": series_index,
                "study_index": study_index,
                "SeriesInstanceUID": f"series-{suffix}",
                "relative_path": f"{split}_series/study-{suffix}/series-{suffix}",
                "Fluid_Sensitive": "",
                "Fat_Suppression": "",
                "Anatomical_Plane": "Sagittal",
                "slice_count": 1,
                "usable_slice_count": 1,
                "ordering_method": "geometry",
                "selection_mask": "10000000",
            }
        )
        slice_rows.append(
            {
                "slice_index": series_index,
                "series_index": series_index,
                "SOPInstanceUID": f"sop-{suffix}",
                "filename": f"slice-{suffix}.dcm",
                "ordered_position": 0,
                "rows": 8,
                "columns": 8,
                "instance_number": 1,
                "geometry_coordinate": 0,
                "transfer_syntax": "1.2.840.10008.1.2.1",
                "photometric_interpretation": "MONOCHROME2",
                "rescale_slope": 1,
                "rescale_intercept": 0,
            }
        )
        dcm_dir = (
            competition_root
            / f"{split}_series"
            / f"study-{suffix}"
            / f"series-{suffix}"
        )
        dcm_dir.mkdir(parents=True, exist_ok=True)
        _write_dicom(dcm_dir / f"slice-{suffix}.dcm")
        selected_rows.append(
            {
                "series_index": series_index,
                "slot": 0,
                "slice_index": series_index,
                "ordered_position": 0,
            }
        )
    error_cols = (
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
    )
    paths_and_columns: tuple[
        tuple[Path, Sequence[str], Sequence[dict[str, object]]], ...
    ] = (
        (index_root / "studies.csv", STUDY_COLUMNS, study_rows),
        (index_root / "series.csv", SERIES_COLUMNS, series_rows),
        (index_root / "slices.csv", SLICE_COLUMNS, slice_rows),
        (index_root / "selected_slices.csv", SELECTED_COLUMNS, selected_rows),
        (index_root / "errors.csv", error_cols, []),
    )
    for path, columns, rows in paths_and_columns:
        _write_csv(path, columns, rows)
    prep_record = {
        "slice_count": 8,
        "lower_percentile": 1.0,
        "upper_percentile": 99.0,
        "image_size": 224,
        "normalization_status": "provisional",
        "resize": "bilinear_aspect_preserving_center_pad",
    }
    stats_path = index_root / "statistics.json"
    stats_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "preprocessing": {
                    **prep_record,
                    "representative_series_checked": 1,
                    "representative_series_succeeded": 1,
                    "checks": [
                        {
                            "all_finite": True,
                            "selected_count": 8,
                            "decoded_count": 8,
                            "mask": [1] * 8,
                            "output_shape": [8, 224, 224],
                            "normalized_range": [0.0, 1.0],
                            "lower_value": 0.0,
                            "upper_value": 255.0,
                        }
                    ],
                },
                "studies": {},
            }
        )
    )
    manifest_files = {
        path.name: {
            "columns": list(columns),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path, columns, _ in paths_and_columns
    }
    manifest_files["statistics.json"] = {
        "sha256": _sha256(stats_path),
        "size_bytes": stats_path.stat().st_size,
    }
    manifest = {
        "schema_version": 1,
        "decoded_images_persisted": False,
        "decoder_dependencies": ["pydicom"],
        "competition_contract": {
            "schema_version": 1,
            "verified_at": "2026-08-11",
            "sha256": _sha256(COMPETITION_CONTRACT),
        },
        "preprocessing": prep_record,
        "files": manifest_files,
    }
    manifest_path = index_root / "index-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    data_index = {
        "schema_version": 1,
        "current_competition_contract": {"sha256": _sha256(COMPETITION_CONTRACT)},
        "run_provenance": {"outputs": {"manifest_sha256": _sha256(manifest_path)}},
    }
    (root / "data-index.json").write_text(json.dumps(data_index))

    pub_contract = {
        "schema_version": 1,
        "target_order": list(TARGETS),
        "file_hashes": {
            "report_labels_v2.csv": _sha256(pub_csv),
        },
    }
    pub_contract_path = root / "public-report-label-dataset-v1.json"
    pub_contract_path.write_text(json.dumps(pub_contract))

    index_contract_path = root / "data-index.json"
    return (
        competition_root,
        public_root,
        index_root,
        pub_contract_path,
        index_contract_path,
    )


def _train_row(study_uid: str, labels: Sequence[int | None]) -> dict[str, object]:
    row: dict[str, object] = {column: "" for column in TRAIN_COLUMNS}
    row["StudyInstanceUID"] = study_uid
    row["Report"] = "private report text must not be consumed"
    row.update(
        {
            target: "" if label is None else label
            for target, label in zip(TARGETS, labels, strict=True)
        }
    )
    return row


def _public_row(study_uid: str, verdicts: Sequence[str]) -> dict[str, object]:
    row: dict[str, object] = {"StudyInstanceUID": study_uid}
    for target, verdict in zip(TARGETS, verdicts, strict=True):
        row[target] = "0.82" if verdict == "YES" else "0.08"
        row[f"{target}__conf"] = "0.95" if verdict != "UNK" else "0.05"
        row[f"{target}__verdict"] = verdict
    return row


def _study_row(
    index: int, study_uid: str, patient_id: str, split: str
) -> dict[str, object]:
    return {
        "study_index": index,
        "split": split,
        "StudyInstanceUID": study_uid,
        "PatientID": patient_id,
        "series_count": 1,
        "usable_series_count": 1,
        "slice_count": 1,
        "usable_slice_count": 1,
    }


def _write_csv(
    path: Path, columns: Sequence[str], rows: Sequence[dict[str, object]]
) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
