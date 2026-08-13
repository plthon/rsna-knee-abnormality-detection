import json
from pathlib import Path

import pytest

from rsna_knee.publication import (
    load_public_baseline_evidence,
    load_reference_baseline_summary,
    reproduce_reference_baseline,
)


def test_reference_baseline_summary_uses_reader_facing_terms() -> None:
    summary = load_reference_baseline_summary()

    assert len(summary["targets"]) == 12
    assert summary["supervision"]["source"] == (
        "Organizer-provided train.csv labels only"
    )
    assert summary["split"]["partitions"]["train"]["patient_count"] == 54
    assert summary["limitations"] == {
        "holdout_patient_count": 4,
        "bootstrap_replicate_count": 10_000,
        "defined_macro_replicate_count": 935,
        "undefined_macro_replicate_count": 9_065,
        "undefined_macro_replicate_rate": pytest.approx(0.9065),
        "supports_model_comparison": False,
    }
    assert "adequacy_for_issue_11" not in summary


def test_public_baseline_evidence_preserves_limited_gold_only_judgment() -> None:
    evidence = load_public_baseline_evidence()

    assert evidence["supervision"]["source"] == "organizer gold train.csv only"
    assert evidence["supervision"]["report_derived_labels_used"] is False
    assert evidence["split"]["grouping_entity"] == "PatientID"
    assert evidence["split"]["patient_overlap_count"] == 0
    assert evidence["evaluation"]["metric"]["point_estimate"][
        "official_macro_auc"
    ] == pytest.approx(0.611111111111111)
    macro = evidence["evaluation"]["bootstrap"]["uncertainty"]["official_macro_auc"]
    assert macro["valid_replicate_count"] == 935
    assert macro["undefined_replicate_count"] == 9065
    assert evidence["adequacy_for_issue_11"]["decision"] == "inadequate"
    assert (
        evidence["adequacy_for_issue_11"]["claims_general_statistical_sufficiency"]
        is False
    )


def test_public_baseline_evidence_rejects_upgraded_adequacy_claim(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gold-baseline.json"
    evidence = json.loads(Path("contracts/gold-baseline.json").read_text())
    evidence["adequacy_for_issue_11"]["decision"] = "adequate"
    path.write_text(json.dumps(evidence))

    with pytest.raises(ValueError, match="must not claim support for model comparison"):
        load_public_baseline_evidence(path)


def test_public_reproduction_rejects_empty_kernel_id_before_work(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Kaggle kernel ID"):
        reproduce_reference_baseline(
            input_root=tmp_path / "input",
            work_parent=tmp_path,
            config_bytes=b"not toml",
            approved_gpu=False,
            source_provenance={},
            kernel_id="",
        )
