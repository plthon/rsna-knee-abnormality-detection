import csv
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from rsna_knee import (
    audit_gold_supervision,
    discover_gold_audit_inputs,
    export_public_gold_audit_contract,
    verify_gold_audit_artifact,
)

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


def test_gold_audit_freezes_all_eligible_cohort_and_candidate_witness(
    tmp_path: Path,
) -> None:
    train_rows = [
        _train_row("study-a", [0] * len(TARGETS)),
        _train_row("study-b", [0] * len(TARGETS)),
        _train_row("study-c", [1] * len(TARGETS)),
        _train_row("study-d", [1] * len(TARGETS)),
        _train_row("study-partial", [1, *([None] * (len(TARGETS) - 1))]),
        _train_row("study-unlabeled", [None] * len(TARGETS)),
        _train_row("study-unusable", [1] * len(TARGETS)),
    ]
    index_rows = [
        _index_row(index, row["StudyInstanceUID"], f"patient-{index}")
        for index, row in enumerate(train_rows)
    ]
    index_rows[-1]["usable_series_count"] = "0"
    index_rows[-1]["usable_slice_count"] = "0"
    competition_root, index_root, index_contract = _prepare_inputs(
        tmp_path, train_rows, index_rows
    )

    first = audit_gold_supervision(
        competition_root,
        index_root,
        tmp_path / "first",
        data_index_contract_path=index_contract,
        source_provenance={"test": "candidate"},
    )
    second = audit_gold_supervision(
        competition_root,
        index_root,
        tmp_path / "second",
        data_index_contract_path=index_contract,
        source_provenance={"test": "candidate"},
    )

    summary = first.summary
    assert summary["label_contract"]["status"] == "verified"
    assert summary["source_gold_labels"]["rowwise_completeness"] == {
        "status": "verified",
        "all_targets_known": 5,
        "partially_known": 1,
        "all_targets_missing": 1,
        "known_target_count_histogram": {
            "0": 1,
            "1": 1,
            str(len(TARGETS)): 5,
        },
    }
    assert summary["canonical_cohort"]["cohort_size"] == 5
    assert summary["canonical_cohort"]["excluded_study_counts_by_reason"] == {
        "index_study_unusable": 1,
        "no_verified_gold_label": 1,
    }
    assert summary["smoke_sufficiency"]["decision"] == "smoke-sufficient"
    assert summary["smoke_sufficiency"]["unsupported_targets"] == []
    assert summary["smoke_sufficiency"]["target_contract_weakened"] is False
    assert summary["smoke_sufficiency"]["substitute_cohort_used"] is False
    assert summary["smoke_sufficiency"]["pseudo_label_work_activated"] is False
    assert summary["evaluation_eligibility"]["decision"] == "Candidate"
    assert summary["evaluation_eligibility"]["unsupported_findings"] == []
    assert summary["evaluation_eligibility"]["minimum_class_support_per_partition"] == 1
    assert summary["canonical_cohort"]["per_target"][TARGETS[1]]["missing"] == 1
    assert (
        summary["canonical_cohort"]["digest"]
        == second.summary["canonical_cohort"]["digest"]
    )
    assert (
        summary["evaluation_eligibility"]["assignment_digest"]
        == second.summary["evaluation_eligibility"]["assignment_digest"]
    )

    for partition in ("train", "holdout"):
        per_target = summary["evaluation_eligibility"]["partitions"][partition][
            "per_target"
        ]
        assert all(counts["positive"] >= 1 for counts in per_target.values())
        assert all(counts["negative"] >= 1 for counts in per_target.values())
    assert summary["report_derived_supervision_recommendation"] == {
        "status": "informational-only",
        "plausibly_addresses_observed_training_limitation": True,
        "observed_limitation": (
            "Organizer-provided gold supervision covers fewer studies than the "
            "usable indexed training corpus."
        ),
        "usable_indexed_train_studies": 6,
        "organizer_gold_cohort_studies": 5,
        "activated": False,
        "changes_smoke_decision": False,
        "changes_evaluation_decision": False,
        "treats_report_values_as_gold": False,
        "downstream_ticket_created": False,
    }
    assert verify_gold_audit_artifact(first.artifact_dir) == summary
    private_cohort = _read_csv(first.artifact_dir / "gold-cohort-private.csv")
    partial = next(
        row for row in private_cohort if row["StudyInstanceUID"] == "study-partial"
    )
    assert partial[TARGETS[0]] == "1"
    assert all(partial[target] == "" for target in TARGETS[1:])

    public_contract = tmp_path / "gold-supervision.json"
    export_public_gold_audit_contract(first.artifact_dir, public_contract)
    public_text = public_contract.read_text()
    assert json.loads(public_text) == summary
    assert "study-a" not in public_text
    assert "patient-0" not in public_text
    assert "Report" not in "\n".join(
        first.artifact_dir.joinpath(filename).read_text().splitlines()[0]
        for filename in (
            "gold-cohort-private.csv",
            "gold-exclusions-private.csv",
            "invalid-labels-private.csv",
            "evaluation-assignment-private.csv",
        )
    )


@pytest.mark.parametrize(
    ("invalid_value", "category"),
    [
        ("2", "outside_binary_domain"),
        ("NaN", "non_finite"),
        ("not-a-label", "not_numeric"),
    ],
)
def test_gold_audit_rejects_and_reports_invalid_label_values(
    tmp_path: Path, invalid_value: str, category: str
) -> None:
    train_rows = [
        _train_row("study-a", [0] * len(TARGETS)),
        _train_row("study-b", [0] * len(TARGETS)),
        _train_row("study-c", [1] * len(TARGETS)),
        _train_row("study-d", [1] * len(TARGETS)),
    ]
    train_rows[-1][TARGETS[0]] = invalid_value
    index_rows = [
        _index_row(index, row["StudyInstanceUID"], f"patient-{index}")
        for index, row in enumerate(train_rows)
    ]
    competition_root, index_root, index_contract = _prepare_inputs(
        tmp_path, train_rows, index_rows
    )

    outcome = audit_gold_supervision(
        competition_root,
        index_root,
        tmp_path / "work",
        data_index_contract_path=index_contract,
    )

    assert outcome.summary["label_contract"]["status"] == "rejected"
    assert outcome.summary["label_contract"]["invalid_value_count"] == 1
    assert outcome.summary["smoke_sufficiency"]["decision"] == "not-smoke-sufficient"
    assert outcome.summary["evaluation_eligibility"]["decision"] == "Ineligible"
    assert {
        finding["code"]
        for finding in outcome.summary["evaluation_eligibility"]["unsupported_findings"]
    } >= {"input_contract_rejected"}
    invalid_rows = _read_csv(outcome.artifact_dir / "invalid-labels-private.csv")
    assert invalid_rows == [
        {
            "source_row": "5",
            "StudyInstanceUID": "study-d",
            "target": TARGETS[0],
            "raw_value": invalid_value,
            "category": category,
        }
    ]
    public_summary = json.dumps(outcome.summary)
    assert "raw_value" not in public_summary
    assert "study-d" not in public_summary


def test_gold_audit_reports_every_unsupported_smoke_target(tmp_path: Path) -> None:
    labels = [
        [0] * len(TARGETS),
        [0] * len(TARGETS),
        [1] * len(TARGETS),
        [1] * len(TARGETS),
    ]
    for row_labels in labels:
        row_labels[-1] = 0
    train_rows = [
        _train_row(f"study-{index}", row_labels)
        for index, row_labels in enumerate(labels)
    ]
    index_rows = [
        _index_row(index, row["StudyInstanceUID"], f"patient-{index}")
        for index, row in enumerate(train_rows)
    ]
    competition_root, index_root, index_contract = _prepare_inputs(
        tmp_path, train_rows, index_rows
    )

    outcome = audit_gold_supervision(
        competition_root,
        index_root,
        tmp_path / "work",
        data_index_contract_path=index_contract,
    )

    assert outcome.summary["smoke_sufficiency"]["decision"] == "not-smoke-sufficient"
    assert outcome.summary["smoke_sufficiency"]["unsupported_targets"] == [
        {"target": TARGETS[-1], "missing_classes": ["positive"]}
    ]
    assert outcome.summary["evaluation_eligibility"]["decision"] == "Ineligible"
    assert {
        (finding.get("target"), finding.get("class"))
        for finding in outcome.summary["evaluation_eligibility"]["unsupported_findings"]
    } >= {(TARGETS[-1], "positive")}
    assert (
        outcome.summary["report_derived_supervision_recommendation"]["activated"]
        is False
    )


def test_gold_audit_exact_search_proves_ineligible_non_bipartite_constraints(
    tmp_path: Path,
) -> None:
    labels = {
        "study-a1": [0] * len(TARGETS),
        "study-a2": [1] * len(TARGETS),
        "study-b1": [0] * len(TARGETS),
        "study-c1": [0] * len(TARGETS),
        "study-c2": [1] * len(TARGETS),
    }
    labels["study-a1"][0] = 1
    labels["study-a2"][0] = 0
    labels["study-b1"][0] = 1
    labels["study-c1"][0] = 0
    labels["study-c2"][0] = 0
    labels["study-a1"][1] = 0
    labels["study-a2"][1] = 0
    labels["study-b1"][1] = 1
    labels["study-c1"][1] = 1
    labels["study-c2"][1] = 0
    patients = {
        "study-a1": "patient-a",
        "study-a2": "patient-a",
        "study-b1": "patient-b",
        "study-c1": "patient-c",
        "study-c2": "patient-c",
    }
    train_rows = [_train_row(study, values) for study, values in labels.items()]
    index_rows = [
        _index_row(index, row["StudyInstanceUID"], patients[row["StudyInstanceUID"]])
        for index, row in enumerate(train_rows)
    ]
    competition_root, index_root, index_contract = _prepare_inputs(
        tmp_path, train_rows, index_rows
    )

    outcome = audit_gold_supervision(
        competition_root,
        index_root,
        tmp_path / "work",
        data_index_contract_path=index_contract,
    )

    assert outcome.summary["smoke_sufficiency"]["decision"] == "smoke-sufficient"
    assert outcome.summary["evaluation_eligibility"]["decision"] == "Ineligible"
    assert outcome.summary["evaluation_eligibility"]["unsupported_findings"] == [
        {
            "code": "no_patient_disjoint_assignment_satisfies_all_class_constraints",
            "search_complete": True,
            "constraint_count": 2 * len(TARGETS),
            "patient_count": 3,
        }
    ]
    assert _read_csv(outcome.artifact_dir / "evaluation-assignment-private.csv") == []


def test_gold_audit_discovers_only_the_accepted_private_index(tmp_path: Path) -> None:
    train_rows = [
        _train_row("study-a", [0] * len(TARGETS)),
        _train_row("study-b", [1] * len(TARGETS)),
    ]
    index_rows = [
        _index_row(index, row["StudyInstanceUID"], f"patient-{index}")
        for index, row in enumerate(train_rows)
    ]
    competition_root, index_root, index_contract = _prepare_inputs(
        tmp_path, train_rows, index_rows
    )
    (competition_root / "sample_submission.csv").write_text("StudyInstanceUID\n")
    (competition_root / "train_series.csv").write_text("StudyInstanceUID\n")

    discovered = discover_gold_audit_inputs(
        tmp_path,
        data_index_contract_path=index_contract,
    )

    assert discovered.competition_root == competition_root
    assert discovered.index_artifact_dir == index_root


def test_gold_audit_rejects_truncated_train_rows(tmp_path: Path) -> None:
    train_rows = [
        _train_row("study-a", [0] * len(TARGETS)),
        _train_row("study-b", [0] * len(TARGETS)),
        _train_row("study-c", [1] * len(TARGETS)),
        _train_row("study-d", [1] * len(TARGETS)),
    ]
    index_rows = [
        _index_row(index, row["StudyInstanceUID"], f"patient-{index}")
        for index, row in enumerate(train_rows)
    ]
    competition_root, index_root, index_contract = _prepare_inputs(
        tmp_path, train_rows, index_rows
    )
    train_path = competition_root / "train.csv"
    lines = train_path.read_text().splitlines()
    lines[1] = "study-a"
    train_path.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="Malformed CSV row in train.csv"):
        audit_gold_supervision(
            competition_root,
            index_root,
            tmp_path / "work",
            data_index_contract_path=index_contract,
        )


def test_gold_audit_verifier_rejects_private_assignment_tampering(
    tmp_path: Path,
) -> None:
    train_rows = [
        _train_row("study-a", [0] * len(TARGETS)),
        _train_row("study-b", [0] * len(TARGETS)),
        _train_row("study-c", [1] * len(TARGETS)),
        _train_row("study-d", [1] * len(TARGETS)),
    ]
    index_rows = [
        _index_row(index, row["StudyInstanceUID"], f"patient-{index}")
        for index, row in enumerate(train_rows)
    ]
    competition_root, index_root, index_contract = _prepare_inputs(
        tmp_path, train_rows, index_rows
    )
    outcome = audit_gold_supervision(
        competition_root,
        index_root,
        tmp_path / "work",
        data_index_contract_path=index_contract,
    )
    assignment_path = outcome.artifact_dir / "evaluation-assignment-private.csv"
    assignment_path.write_text(
        assignment_path.read_text().replace("train", "holdout", 1)
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        verify_gold_audit_artifact(outcome.artifact_dir)


def test_gold_audit_verifier_rejects_patient_crossing_after_integrity_check(
    tmp_path: Path,
) -> None:
    train_rows = [
        _train_row("study-a1", [0] * len(TARGETS)),
        _train_row("study-a2", [0] * len(TARGETS)),
        _train_row("study-b", [0] * len(TARGETS)),
        _train_row("study-c", [1] * len(TARGETS)),
        _train_row("study-d", [1] * len(TARGETS)),
    ]
    patient_by_study = {
        "study-a1": "patient-a",
        "study-a2": "patient-a",
        "study-b": "patient-b",
        "study-c": "patient-c",
        "study-d": "patient-d",
    }
    index_rows = [
        _index_row(
            index,
            row["StudyInstanceUID"],
            patient_by_study[row["StudyInstanceUID"]],
        )
        for index, row in enumerate(train_rows)
    ]
    competition_root, index_root, index_contract = _prepare_inputs(
        tmp_path, train_rows, index_rows
    )
    outcome = audit_gold_supervision(
        competition_root,
        index_root,
        tmp_path / "work",
        data_index_contract_path=index_contract,
    )
    assignment_path = outcome.artifact_dir / "evaluation-assignment-private.csv"
    assignment_rows = _read_csv(assignment_path)
    patient_rows = [row for row in assignment_rows if row["PatientID"] == "patient-a"]
    assert len(patient_rows) == 2
    patient_rows[1]["partition"] = (
        "holdout" if patient_rows[0]["partition"] == "train" else "train"
    )
    _write_csv(
        assignment_path,
        ("StudyInstanceUID", "PatientID", "partition"),
        assignment_rows,
    )

    manifest_path = outcome.manifest_path
    manifest = json.loads(manifest_path.read_text())
    assignment_record = manifest["files"]["evaluation-assignment-private.csv"]
    assignment_record["sha256"] = _sha256(assignment_path)
    assignment_record["size_bytes"] = assignment_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="PatientID crosses"):
        verify_gold_audit_artifact(outcome.artifact_dir)


def _train_row(study_uid: str, labels: Sequence[int | str | None]) -> dict[str, Any]:
    row: dict[str, Any] = {column: "" for column in TRAIN_COLUMNS}
    row["StudyInstanceUID"] = study_uid
    row["Report"] = "private report text"
    row.update(
        {
            target: "" if label is None else label
            for target, label in zip(TARGETS, labels, strict=True)
        }
    )
    return row


def _index_row(index: int, study_uid: str, patient_id: str) -> dict[str, Any]:
    return {
        "study_index": index,
        "split": "train",
        "StudyInstanceUID": study_uid,
        "PatientID": patient_id,
        "series_count": 1,
        "usable_series_count": 1,
        "slice_count": 8,
        "usable_slice_count": 8,
    }


def _prepare_inputs(
    root: Path,
    train_rows: list[dict[str, Any]],
    index_rows: list[dict[str, Any]],
) -> tuple[Path, Path, Path]:
    competition_root = root / "competition"
    competition_root.mkdir(parents=True)
    _write_csv(competition_root / "train.csv", TRAIN_COLUMNS, train_rows)

    index_root = root / "attached-index" / "rsna-knee-index-v1"
    index_root.mkdir(parents=True)
    studies_path = index_root / "studies.csv"
    _write_csv(studies_path, STUDY_COLUMNS, index_rows)
    studies_record = {
        "columns": list(STUDY_COLUMNS),
        "sha256": _sha256(studies_path),
        "size_bytes": studies_path.stat().st_size,
    }
    manifest = {
        "schema_version": 1,
        "artifact_type": "rsna_knee_compact_index",
        "decoded_images_persisted": False,
        "files": {"studies.csv": studies_record},
    }
    manifest_path = index_root / "index-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    contract_path = root / "data-index.json"
    contract_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "verified_at": "2026-08-12",
                "current_competition_contract": {
                    "sha256": _sha256(COMPETITION_CONTRACT)
                },
                "artifact": {
                    "decoded_images_persisted": False,
                    "files": {
                        "studies.csv": {
                            "sha256": _sha256(studies_path),
                            "size_bytes": studies_path.stat().st_size,
                        }
                    },
                },
                "run_provenance": {
                    "outputs": {"manifest_sha256": _sha256(manifest_path)}
                },
            },
            sort_keys=True,
        )
    )
    return competition_root, index_root, contract_path


def _write_csv(
    path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]
) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
