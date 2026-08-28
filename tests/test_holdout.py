import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from rsna_knee.holdout import (
    ALLOCATION_RULE_VERSION,
    ASSIGNMENT_DIGEST_VERSION,
    BOOTSTRAP_METHOD_VERSION,
    PAIRED_BOOTSTRAP_METHOD_VERSION,
    Label,
    PatientBootstrapConfig,
    derive_frozen_evaluation_partition,
    evaluate_holdout_predictions,
    evaluate_paired_auc_delta,
)

TARGETS = ("target-a", "target-b")
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


@dataclass(frozen=True)
class _Study:
    study_uid: str
    labels: tuple[int | None, ...]


@dataclass(frozen=True)
class _Cohort:
    studies: tuple[_Study, ...]
    target_names: tuple[str, ...]


def test_frozen_partition_is_independently_reproduced_and_digest_checked(
    tmp_path: Path,
) -> None:
    seed = 20260812
    cohort = _Cohort(
        studies=tuple(
            _Study(f"study-{index}", (label, label))
            for index, label in enumerate((0, 0, 1, 1))
        ),
        target_names=TARGETS,
    )
    index_root = tmp_path / "index"
    index_root.mkdir()
    rows = [
        {
            "study_index": index,
            "split": "train",
            "StudyInstanceUID": study.study_uid,
            "PatientID": f"patient-{index}",
            "series_count": 1,
            "usable_series_count": 1,
            "slice_count": 1,
            "usable_slice_count": 1,
        }
        for index, study in enumerate(cohort.studies)
    ]
    _write_csv(index_root / "studies.csv", STUDY_COLUMNS, rows)
    assignments = _expected_assignments(seed)
    per_partition = _partition_counts(cohort, assignments)
    contract = {
        "schema_version": 1,
        "evaluation_eligibility": {
            "decision": "Candidate",
            "grouping_entity": "PatientID",
            "allocation_rule_version": ALLOCATION_RULE_VERSION,
            "seed": seed,
            "uses_predictions": False,
            "uses_leaderboard_results": False,
            "prediction_independent": True,
            "assignment_digest": {
                "canonicalization_version": ASSIGNMENT_DIGEST_VERSION,
                "value": _assignment_digest(assignments, seed),
            },
            "partitions": per_partition,
        },
    }
    contract_path = tmp_path / "gold-supervision.json"
    contract_path.write_text(json.dumps(contract))

    partition = derive_frozen_evaluation_partition(
        cohort,
        index_root,
        supervision_contract_path=contract_path,
    )

    evaluation = cast(dict[str, Any], contract["evaluation_eligibility"])
    digest = cast(dict[str, str], evaluation["assignment_digest"])
    assert partition.assignment_digest == digest["value"]
    assert partition.per_partition == per_partition
    assert {study.patient_id for study in partition.train}.isdisjoint(
        study.patient_id for study in partition.holdout
    )
    assert len(partition.train) == 2
    assert len(partition.holdout) == 2

    digest["value"] = "0" * 64
    contract_path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="assignment digest mismatch"):
        derive_frozen_evaluation_partition(
            cohort,
            index_root,
            supervision_contract_path=contract_path,
        )


def test_patient_bootstrap_is_deterministic_and_preserves_undefined_metrics() -> None:
    configuration = PatientBootstrapConfig(
        method_version=BOOTSTRAP_METHOD_VERSION,
        seed=314159,
        replicate_count=1000,
        confidence_level=0.95,
    )
    patient_ids = ["p0", "p1", "p2", "p3"]
    labels: list[tuple[Label, ...]] = [(0, 0), (0, 1), (1, 0), (1, 1)]
    predictions = [(0.1, 0.2), (0.2, 0.8), (0.8, 0.1), (0.9, 0.9)]
    first = evaluate_holdout_predictions(
        patient_ids=patient_ids,
        labels=labels,
        predictions=predictions,
        target_names=TARGETS,
        bootstrap=configuration,
    )
    second = evaluate_holdout_predictions(
        patient_ids=patient_ids,
        labels=labels,
        predictions=predictions,
        target_names=TARGETS,
        bootstrap=configuration,
    )

    assert first == second
    point = first["metric"]["point_estimate"]
    assert point["official_macro_auc"] == 1.0
    assert all(point["per_target"][target]["roc_auc"] == 1.0 for target in TARGETS)
    uncertainty = first["bootstrap"]["uncertainty"]
    macro = uncertainty["official_macro_auc"]
    assert 0 < macro["undefined_replicate_count"] < 1000
    assert macro["valid_replicate_count"] + macro["undefined_replicate_count"] == 1000
    assert macro["interval"] == {"lower": 1.0, "upper": 1.0}
    for target in TARGETS:
        record = uncertainty["per_target"][target]
        assert (
            record["valid_replicate_count"] + record["undefined_replicate_count"]
            == 1000
        )
        assert record["undefined_replicate_count"] > 0
        assert macro["undefined_replicate_count"] >= record["undefined_replicate_count"]
    policy = first["bootstrap"]["undefined_metric_policy"]
    assert policy["default_scores_assigned"] is False
    assert policy["defined_target_only_macro_average"] is False


def test_patient_bootstrap_resamples_patient_groups_not_individual_studies() -> None:
    result = evaluate_holdout_predictions(
        patient_ids=["p0", "p0", "p1", "p1"],
        labels=[(0,), (0,), (1,), (1,)],
        predictions=[(0.1,), (0.2,), (0.8,), (0.9,)],
        target_names=("target",),
        bootstrap=PatientBootstrapConfig(
            method_version=BOOTSTRAP_METHOD_VERSION,
            seed=7,
            replicate_count=200,
            confidence_level=0.95,
        ),
    )

    record = result["bootstrap"]["uncertainty"]["per_target"]["target"]
    assert record["undefined_replicate_count"] > 0
    assert result["bootstrap"]["sampling_unit"] == "PatientID"


def test_paired_bootstrap_reports_candidate_minus_baseline_delta() -> None:
    result = evaluate_paired_auc_delta(
        patient_ids=["p0", "p1", "p2", "p3"],
        labels=[(0,), (0,), (1,), (1,)],
        baseline_predictions=[(0.4,), (0.6,), (0.5,), (0.7,)],
        candidate_predictions=[(0.1,), (0.2,), (0.8,), (0.9,)],
        target_names=("target",),
        bootstrap=PatientBootstrapConfig(
            method_version=PAIRED_BOOTSTRAP_METHOD_VERSION,
            seed=7,
            replicate_count=200,
            confidence_level=0.95,
        ),
    )

    assert result["baseline"]["official_macro_auc"] == 0.75
    assert result["candidate"]["official_macro_auc"] == 1.0
    assert result["delta"]["official_macro_auc"] == 0.25
    uncertainty = result["bootstrap"]["uncertainty"]["official_macro_auc_delta"]
    assert uncertainty["valid_replicate_count"] > 0
    assert uncertainty["fraction_nonnegative"] == 1.0


def _expected_assignments(seed: int) -> dict[str, str]:
    patients_by_label = {
        0: ["patient-0", "patient-1"],
        1: ["patient-2", "patient-3"],
    }
    assignments: dict[str, str] = {}
    for patients in patients_by_label.values():
        ordered = sorted(
            patients,
            key=lambda patient: (
                hashlib.sha256(f"{seed}\0{patient}".encode()).digest(),
                patient.encode(),
            ),
        )
        assignments[ordered[0]] = "train"
        assignments[ordered[1]] = "holdout"
    return assignments


def _partition_counts(
    cohort: _Cohort, assignments: dict[str, str]
) -> dict[str, object]:
    result: dict[str, object] = {}
    for partition in ("train", "holdout"):
        selected = [
            study
            for index, study in enumerate(cohort.studies)
            if assignments[f"patient-{index}"] == partition
        ]
        result[partition] = {
            "study_count": len(selected),
            "patient_count": len(selected),
            "per_target": {
                target: {
                    "known": len(selected),
                    "missing": 0,
                    "positive": sum(study.labels[index] == 1 for study in selected),
                    "negative": sum(study.labels[index] == 0 for study in selected),
                }
                for index, target in enumerate(TARGETS)
            },
        }
    return result


def _assignment_digest(assignments: dict[str, str], seed: int) -> str:
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "schema": ASSIGNMENT_DIGEST_VERSION,
                "allocation_rule": ALLOCATION_RULE_VERSION,
                "seed": seed,
            }
        )
        + b"\n"
    )
    for index in sorted(
        range(4),
        key=lambda value: (
            f"patient-{value}".encode(),
            f"study-{value}".encode(),
        ),
    ):
        digest.update(
            _canonical_json(
                [
                    f"patient-{index}",
                    f"study-{index}",
                    assignments[f"patient-{index}"],
                ]
            )
            + b"\n"
        )
    return digest.hexdigest()


def _write_csv(
    path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]
) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
