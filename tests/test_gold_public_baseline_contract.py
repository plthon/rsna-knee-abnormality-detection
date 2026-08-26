import json
from pathlib import Path


def test_gold_public_baseline_contract_records_verified_real_data_evidence() -> None:
    root = Path(__file__).parents[1]
    contract = json.loads(
        (root / "contracts" / "gold-public-baseline.json").read_text()
    )

    assert contract["schema_version"] == 1
    assert contract["artifact_type"] == "rsna_knee_gold_public_baseline"

    supervision = contract["supervision"]
    assert supervision["gold_studies"] == 58
    assert supervision["non_gold_studies"] == 4349
    assert supervision["total_studies"] == 4407
    assert supervision["gold_precedence"] is True
    assert supervision["public_mapping"] == {
        "YES": 1,
        "NO": 0,
        "UNK": "masked",
    }
    assert len(supervision["target_order"]) == 12

    evaluation = contract["evaluation"]
    assert 0.80 <= evaluation["val_mean_roc_auc"] <= 0.90
    assert len(evaluation["val_per_target"]) == 12
    for metrics in evaluation["val_per_target"].values():
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert metrics["known"] > 0
        assert metrics["positive"] >= 0
        assert metrics["negative"] >= 0

    checkpoint = contract["checkpoint"]
    assert checkpoint["reload_reproducible"] is True
    assert checkpoint["max_reload_absolute_diff"] == 0.0
    assert checkpoint["size_bytes"] > 100_000_000

    assert contract["execution"]["optimizer_train_study_count"] == 3957
    assert contract["execution"]["supervision_cohort_study_count"] == 4407

    scoring = contract["scoring"]
    assert scoring == {
        "checkpoint_sha256": checkpoint["sha256"],
        "evidence_scope": "final inference bundle contract; not verified by the mounted three-study example",
        "frozen_test_index_used": False,
        "live_test_index": True,
        "mode": "inference_only",
        "retrain": False,
        "strict_submission_validation": True,
    }

    submission = contract["submission"]
    assert submission["strictly_validated"] is True
    assert submission["fallback_count"] == 0
    assert submission["prediction_count"] == 3
    assert submission["hidden_scoring_verified"] is False
    assert submission["sigmoid_outputs_calibrated"] is False

    assert contract["privacy"] == {
        "competition_data_downloaded": False,
        "private_artifacts_registered_under_ignored_path": True,
        "row_level_membership_labels_patient_ids_or_predictions_committed": False,
    }
