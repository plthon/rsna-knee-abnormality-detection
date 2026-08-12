import json
from pathlib import Path


def test_gold_supervision_contract_records_verified_real_data_decisions() -> None:
    contract = json.loads(Path("contracts/gold-supervision.json").read_text())
    targets = contract["label_contract"]["target_order"]

    assert contract["schema_version"] == 1
    assert contract["label_contract"]["status"] == "verified"
    assert contract["label_contract"]["target_count"] == 12
    assert contract["label_contract"]["invalid_value_count"] == 0
    assert contract["source_gold_labels"]["rowwise_completeness"] == {
        "all_targets_known": 58,
        "all_targets_missing": 4349,
        "known_target_count_histogram": {"0": 4349, "12": 58},
        "partially_known": 0,
        "status": "verified",
    }

    cohort = contract["canonical_cohort"]
    assert cohort["cohort_size"] == 58
    assert cohort["uses_all_eligible_studies"] is True
    assert cohort["selection_optimizes_class_balance"] is False
    assert cohort["all_selected_studies_resolve_to_exactly_one_usable_index_study"]
    assert cohort["excluded_study_counts_by_reason"] == {"no_verified_gold_label": 4349}
    assert (
        cohort["digest"]["value"]
        == "ea739a412709b0ac65429ca47f29161614ba384e2e92643301f817712cd6b7bc"
    )
    assert all(
        counts["known"] == 58
        and counts["missing"] == 0
        and counts["negative"] >= 1
        and counts["positive"] >= 1
        for counts in cohort["per_target"].values()
    )

    smoke = contract["smoke_sufficiency"]
    assert smoke["decision"] == "smoke-sufficient"
    assert smoke["unsupported_targets"] == []
    assert smoke["pseudo_label_work_activated"] is False
    assert smoke["target_contract_weakened"] is False
    assert smoke["substitute_cohort_used"] is False

    evaluation = contract["evaluation_eligibility"]
    assert evaluation["decision"] == "Candidate"
    assert evaluation["grouping_entity"] == "PatientID"
    assert evaluation["seed"] == 20260812
    assert evaluation["uses_predictions"] is False
    assert evaluation["uses_leaderboard_results"] is False
    assert evaluation["unsupported_findings"] == []
    assert (
        evaluation["assignment_digest"]["value"]
        == "8dccc31852a3e26ec677062a289301004fdce29eb64a556bea8922dfd2bb263d"
    )
    assert evaluation["partitions"]["train"]["patient_count"] == 54
    assert evaluation["partitions"]["holdout"]["patient_count"] == 4
    for partition in ("train", "holdout"):
        counts_by_target = evaluation["partitions"][partition]["per_target"]
        assert set(counts_by_target) == set(targets)
        assert all(
            counts["positive"] >= 1 and counts["negative"] >= 1
            for counts in counts_by_target.values()
        )

    assert contract["privacy"] == {
        "patient_ids_committed": False,
        "private_files_remain_in_kaggle_artifact": True,
        "public_safe_summary": True,
        "row_level_assignments_committed": False,
        "row_level_labels_committed": False,
        "row_level_membership_committed": False,
    }
    recommendation = contract["report_derived_supervision_recommendation"]
    assert recommendation["status"] == "informational-only"
    assert recommendation["activated"] is False
    assert recommendation["changes_smoke_decision"] is False
    assert recommendation["changes_evaluation_decision"] is False
    assert recommendation["treats_report_values_as_gold"] is False
