import json
from pathlib import Path


def test_gold_baseline_contract_records_verified_real_data_evidence() -> None:
    root = Path(__file__).parents[1]
    contract = json.loads((root / "contracts" / "gold-baseline.json").read_text())

    assert contract["schema_version"] == 1
    assert contract["artifact_type"] == "rsna_knee_gold_holdout_baseline"
    assert contract["supervision"]["report_derived_labels_used"] is False
    assert contract["exclusions"] == {
        "leaderboard_evidence_used": False,
        "report_derived_labels_used": False,
    }
    split = contract["split"]
    assert split["grouping_entity"] == "PatientID"
    assert split["report_hash_grouping_applied"] is False
    assert split["patient_overlap_count"] == 0
    assert split["assignment_digest"] == (
        "8dccc31852a3e26ec677062a289301004fdce29eb64a556bea8922dfd2bb263d"
    )
    assert split["partitions"]["train"]["patient_count"] == 54
    assert split["partitions"]["holdout"]["patient_count"] == 4
    for partition in ("train", "holdout"):
        for counts in split["partitions"][partition]["per_target"].values():
            assert counts["positive"] >= 1
            assert counts["negative"] >= 1

    evaluation = contract["evaluation"]
    assert evaluation["metric"]["point_estimate"]["official_macro_auc"] == (
        0.611111111111111
    )
    bootstrap = evaluation["bootstrap"]
    assert bootstrap["method_version"] == "patient-id-nonparametric-bootstrap-v1"
    assert bootstrap["seed"] == 20260812
    assert bootstrap["replicate_count"] == 10000
    assert bootstrap["confidence_level"] == 0.95
    assert bootstrap["undefined_metric_policy"] == {
        "default_scores_assigned": False,
        "defined_target_only_macro_average": False,
        "macro_auc": "undefined when any required target AUC is undefined",
        "target_auc": ("undefined when a replicate lacks either organizer-gold class"),
    }
    macro = bootstrap["uncertainty"]["official_macro_auc"]
    assert macro["valid_replicate_count"] == 935
    assert macro["undefined_replicate_count"] == 9065
    for record in bootstrap["uncertainty"]["per_target"].values():
        assert (
            record["valid_replicate_count"] + record["undefined_replicate_count"]
            == 10000
        )

    assert contract["checkpoints"]["retained_count"] == 2
    assert contract["checkpoints"]["reload_predictions_within_tolerance"] is True
    assert contract["verification"]["valid"] is True
    assert contract["adequacy_for_issue_11"]["decision"] == "inadequate"
    assert (
        contract["adequacy_for_issue_11"]["claims_general_statistical_sufficiency"]
        is False
    )
    assert contract["privacy"] == {
        "competition_data_downloaded": False,
        "private_artifacts_registered_under_ignored_path": True,
        "row_level_membership_labels_patient_ids_or_predictions_committed": False,
    }
