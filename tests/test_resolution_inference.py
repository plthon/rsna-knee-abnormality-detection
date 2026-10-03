import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from dicom_support import write_dicom as _write_dicom
from pydicom import dcmread


def test_real_336_checkpoint_drives_live_hidden_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from rsna_knee.combined_supervision import (
        PHYSICAL_CROP_CHECKPOINT_SHA256,
        RESOLUTION_SCHEMA,
        _combined_config_record,
        load_combined_baseline_config,
        load_verified_combined_checkpoint,
        load_verified_physical_crop_checkpoint,
        run_combined_scoring,
        verify_physical_crop_artifact,
    )
    from rsna_knee.competition_contract import load_verified_submission_contract
    from rsna_knee.training import _new_uninitialized_model

    root = Path(__file__).parents[1]
    config = load_combined_baseline_config(root / "configs/resolution-336.toml")
    targets = list(load_verified_submission_contract().target_names)
    artifact = tmp_path / "candidate"
    artifact.mkdir()
    model = _new_uninitialized_model(12)
    payload: dict[str, Any] = {
        "checkpoint_schema_version": RESOLUTION_SCHEMA,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "scaler_state_dict": {},
        "completed_epochs": 3,
        "targets": targets,
        "priors": [0.5] * 12,
        "preprocessing": {
            **config.preprocessing.record(),
            "shared_contract": "preprocess_series_result",
            "channel_adapter": "grayscale_repeat_to_rgb",
            "imagenet_mean": [0.485, 0.456, 0.406],
            "imagenet_std": [0.229, 0.224, 0.225],
        },
        "configuration": _combined_config_record(config),
        "cohort": {
            "selection_rule_version": "gold-plus-public-verdicts-v1",
            "digest": "b066a7e6cd7119f4bef330a006a86a0e08e664d7b5347a111ec781e72abe1fdc",
            "gold_study_count": 58,
            "non_gold_study_count": 4349,
            "total_study_count": 4407,
        },
        "partition": {
            "split_rule_version": "patient-hash-disjoint-v1",
            "split_seed": 20260825,
            "val_fraction": 0.1,
            "train_study_count": 3957,
            "val_study_count": 450,
        },
        "initialization": {
            "artifact_sha256": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
            "artifact_size_bytes": 46830571,
            "contract_sha256": "cbd3639197247bf8b4bd83f11047f5cac5bf8b13ba92e881c423f7c61fabc4ca",
        },
        "code": {},
        "runtime": {},
        "artifact_hashes": {},
        "rng_state": {},
    }
    checkpoint = artifact / "checkpoint.pt"
    torch.save(payload, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    load_verified_combined_checkpoint(
        checkpoint,
        device=torch.device("cpu"),
        expected_sha256=digest,
        resolution_336=True,
    )
    with pytest.raises(ValueError, match="schema"):
        load_verified_physical_crop_checkpoint(
            checkpoint, device=torch.device("cpu"), expected_sha256=digest
        )
    report: dict[str, Any] = {
        "status": "complete",
        "checkpoint": {"reload_reproducible": True},
        "configuration": payload["configuration"],
        "physical_crop_evidence": {"synthetic": True},
        "baseline_reproducibility": {"control_predictions_reused": True},
        "resource_smoke": {"status": "safe_for_separate_full_approval"},
        "controlled_comparison": {
            "validation_membership_digest": "8b1240fc2456779ba64fa115b23956adb131fb387f01c0f0424a933a2096ee88",
            "baseline_checkpoint_sha256": PHYSICAL_CROP_CHECKPOINT_SHA256,
            "decision": {"result": "keep"},
            "metrics": {
                "delta": {
                    "official_macro_auc": 0.006,
                    "per_target": {
                        target: {"delta_roc_auc": 0.006} for target in targets
                    },
                },
                "bootstrap": {
                    "uncertainty": {
                        "official_macro_auc_delta": {"fraction_positive": 0.96}
                    }
                },
            },
        },
    }
    report_path = artifact / "training-report.json"
    report_path.write_text(json.dumps(report))
    manifest = {
        "configuration": payload["configuration"],
        "artifacts": [
            {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
            for path in (checkpoint, report_path)
        ],
    }
    (artifact / "run-manifest.json").write_text(json.dumps(manifest))
    assert (
        verify_physical_crop_artifact(artifact, resolution_336=True)["decision"][
            "result"
        ]
        == "keep"
    )

    competition = tmp_path / "input/competitions/rsna-knee-abnormality-detection"
    competition.mkdir(parents=True)
    for name, columns, rows in (
        ("test.csv", ["StudyInstanceUID"], [["hidden"]]),
        (
            "sample_submission.csv",
            ["StudyInstanceUID", *targets],
            [["hidden", *([0.5] * 12)]],
        ),
        (
            "test_series.csv",
            [
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "Fluid_Sensitive",
                "Fat_Suppression",
                "Anatomical_Plane",
            ],
            [["hidden", "series", "Yes", "No", "Sagittal"]],
        ),
    ):
        with (competition / name).open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            writer.writerows(rows)
    dicom = competition / "test_series/hidden/series/slice.dcm"
    dicom.parent.mkdir(parents=True)
    _write_dicom(dicom, np.arange(160 * 160).reshape(160, 160))
    ds = dcmread(dicom)
    ds.PixelSpacing = [1, 1]
    ds.save_as(dicom, enforce_file_format=True)
    monkeypatch.setattr(
        "rsna_knee.combined_supervision.require_approved_cuda",
        lambda approved: {"approved": approved, "device_name": "CPU-test"},
    )
    outcome = run_combined_scoring(
        input_root=tmp_path / "input",
        work_root=tmp_path / "scored",
        approved_gpu=True,
        checkpoint_path=checkpoint,
        expected_checkpoint_sha256=digest,
        physical_crop=True,
        resolution_336=True,
        retained_artifact_dir=artifact,
    )
    manifest = json.loads(outcome.manifest_path.read_text())
    assert manifest["configuration"]["preprocessing"]["image_size"] == 336
    assert manifest["configuration"]["preprocessing"]["physical_fov_mm"] == 130.0
    scoring = json.loads(outcome.scoring_report_path.read_text())
    assert scoring["inference"]["spatial_evidence"]["counts"]["shape:130x130"] == 1
    with outcome.submission_path.open() as file:
        rows = list(csv.reader(file))
    assert rows[0] == ["StudyInstanceUID", *targets]
    assert rows[1][0] == "hidden" and len(rows) == 2
    assert all(0 <= float(value) <= 1 for value in rows[1][1:])
