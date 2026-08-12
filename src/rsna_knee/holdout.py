"""Frozen PatientID holdout derivation and shared ROC AUC uncertainty."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np

type Label = Literal[0, 1] | None
type Partition = Literal["train", "holdout"]

BOOTSTRAP_METHOD_VERSION = "patient-id-nonparametric-bootstrap-v1"
ALLOCATION_RULE_VERSION = "patient-disjoint-exact-backtracking-v1"
ASSIGNMENT_DIGEST_VERSION = "patient-study-partition-jsonl-v1"

_PARTITIONS: tuple[Partition, Partition] = ("train", "holdout")
_STUDY_COLUMNS = (
    "study_index",
    "split",
    "StudyInstanceUID",
    "PatientID",
    "series_count",
    "usable_series_count",
    "slice_count",
    "usable_slice_count",
)


class GoldCohortLike(Protocol):
    @property
    def studies(self) -> Sequence[Any]: ...

    @property
    def target_names(self) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class PartitionedGoldStudy:
    study_uid: str
    patient_id: str
    labels: tuple[Label, ...]
    partition: Partition


@dataclass(frozen=True, slots=True)
class GoldEvaluationPartition:
    studies: tuple[PartitionedGoldStudy, ...]
    target_names: tuple[str, ...]
    assignment_digest: str
    allocation_rule_version: str
    allocation_seed: int
    per_partition: dict[str, dict[str, Any]]
    supervision_contract_sha256: str

    @property
    def train(self) -> tuple[PartitionedGoldStudy, ...]:
        return tuple(study for study in self.studies if study.partition == "train")

    @property
    def holdout(self) -> tuple[PartitionedGoldStudy, ...]:
        return tuple(study for study in self.studies if study.partition == "holdout")


@dataclass(frozen=True, slots=True)
class PatientBootstrapConfig:
    method_version: str
    seed: int
    replicate_count: int
    confidence_level: float


def derive_frozen_evaluation_partition(
    cohort: GoldCohortLike,
    index_artifact_dir: Path,
    *,
    supervision_contract_path: Path | None = None,
) -> GoldEvaluationPartition:
    """Rebuild #7's assignment from authoritative cohort and index rows."""
    contract, contract_sha256 = _load_supervision_contract(supervision_contract_path)
    evaluation = _mapping(contract, "evaluation_eligibility")
    if evaluation.get("decision") != "Candidate":
        raise ValueError("#7 did not classify the evaluation witness as Candidate")
    if evaluation.get("grouping_entity") != "PatientID":
        raise ValueError("#7 evaluation witness is not grouped by PatientID")
    if evaluation.get("allocation_rule_version") != ALLOCATION_RULE_VERSION:
        raise ValueError("#7 allocation rule version is unsupported")
    if (
        evaluation.get("uses_predictions") is not False
        or evaluation.get("uses_leaderboard_results") is not False
        or evaluation.get("prediction_independent") is not True
    ):
        raise ValueError("#7 evaluation witness is not prediction-independent")
    seed = _integer(evaluation, "seed")

    patient_by_study = _patient_ids(index_artifact_dir / "studies.csv")
    rows: list[tuple[str, str, tuple[Label, ...]]] = []
    for study in cohort.studies:
        patient_id = patient_by_study.get(study.study_uid)
        if patient_id is None:
            raise ValueError("Canonical cohort study is absent from the train index")
        if not patient_id:
            raise ValueError("Canonical cohort study has an empty PatientID")
        rows.append((study.study_uid, patient_id, study.labels))
    if len({study_uid for study_uid, _, _ in rows}) != len(rows):
        raise ValueError("Canonical cohort contains duplicate study identifiers")

    constraints: list[frozenset[str]] = []
    for target_index, target in enumerate(cohort.target_names):
        for label in (0, 1):
            patients = frozenset(
                patient_id
                for _, patient_id, labels in rows
                if labels[target_index] == label
            )
            if len(patients) < 2:
                raise ValueError(
                    f"Cannot reproduce both partitions for target {target!r} class {label}"
                )
            constraints.append(patients)
    patient_ids = sorted({patient_id for _, patient_id, _ in rows})
    assignments = _solve_two_partition_constraints(patient_ids, constraints, seed)
    if assignments is None:
        raise ValueError("Cannot reproduce #7's exact PatientID-disjoint assignment")

    partitioned = tuple(
        PartitionedGoldStudy(
            study_uid=study_uid,
            patient_id=patient_id,
            labels=labels,
            partition=assignments[patient_id],
        )
        for study_uid, patient_id, labels in sorted(
            rows,
            key=lambda item: (item[1].encode("utf-8"), item[0].encode("utf-8")),
        )
    )
    if any(
        len({study.partition for study in partitioned if study.patient_id == patient})
        != 1
        for patient in patient_ids
    ):
        raise ValueError("PatientID crosses the independently derived partition")

    digest = _assignment_digest(partitioned, seed)
    digest_contract = _mapping(evaluation, "assignment_digest")
    if digest_contract.get("canonicalization_version") != ASSIGNMENT_DIGEST_VERSION:
        raise ValueError("#7 assignment canonicalization version is unsupported")
    if digest != digest_contract.get("value"):
        raise ValueError("Independently derived evaluation assignment digest mismatch")
    per_partition = _partition_summary(partitioned, cohort.target_names)
    if per_partition != evaluation.get("partitions"):
        raise ValueError("Independently derived partition aggregates mismatch #7")
    for partition in _PARTITIONS:
        per_target = cast(
            dict[str, dict[str, int]], per_partition[partition]["per_target"]
        )
        for target in cohort.target_names:
            if per_target[target]["positive"] < 1 or per_target[target]["negative"] < 1:
                raise ValueError("Frozen partition does not contain both gold classes")
    return GoldEvaluationPartition(
        studies=partitioned,
        target_names=cohort.target_names,
        assignment_digest=digest,
        allocation_rule_version=ALLOCATION_RULE_VERSION,
        allocation_seed=seed,
        per_partition=per_partition,
        supervision_contract_sha256=contract_sha256,
    )


def evaluate_holdout_predictions(
    *,
    patient_ids: Sequence[str],
    labels: Sequence[Sequence[Label]],
    predictions: Sequence[Sequence[float]],
    target_names: Sequence[str],
    bootstrap: PatientBootstrapConfig,
) -> dict[str, Any]:
    """Compute official AUCs and one PatientID-level nonparametric bootstrap."""
    if bootstrap.method_version != BOOTSTRAP_METHOD_VERSION:
        raise ValueError("Unsupported PatientID bootstrap method version")
    if bootstrap.seed < 0:
        raise ValueError("Bootstrap seed must be nonnegative")
    if bootstrap.replicate_count <= 0:
        raise ValueError("Bootstrap replicate count must be positive")
    if not 0 < bootstrap.confidence_level < 1:
        raise ValueError("Bootstrap confidence level must be between zero and one")
    if not patient_ids or any(not patient_id for patient_id in patient_ids):
        raise ValueError("PatientID bootstrap requires nonempty PatientIDs")
    label_values = np.asarray(
        [[np.nan if value is None else value for value in row] for row in labels],
        dtype=np.float64,
    )
    prediction_values = np.asarray(predictions, dtype=np.float64)
    expected_shape = (len(patient_ids), len(target_names))
    if (
        label_values.shape != expected_shape
        or prediction_values.shape != expected_shape
    ):
        raise ValueError("Holdout labels and predictions do not match target shape")
    if (
        not np.isfinite(prediction_values).all()
        or ((prediction_values < 0) | (prediction_values > 1)).any()
    ):
        raise ValueError("Holdout predictions must be finite probabilities")
    known_labels = label_values[~np.isnan(label_values)]
    if ((known_labels != 0) & (known_labels != 1)).any():
        raise ValueError("Holdout labels must be binary or missing")

    point = _metric_record(label_values, prediction_values, target_names)
    unique_patients = sorted(set(patient_ids), key=lambda value: value.encode("utf-8"))
    row_indices = {
        patient: np.asarray(
            [index for index, value in enumerate(patient_ids) if value == patient],
            dtype=np.int64,
        )
        for patient in unique_patients
    }
    target_replicates: list[list[float]] = [[] for _ in target_names]
    macro_replicates: list[float] = []
    target_undefined = [0 for _ in target_names]
    macro_undefined = 0
    rng = np.random.default_rng(bootstrap.seed)
    for _ in range(bootstrap.replicate_count):
        sampled = rng.integers(0, len(unique_patients), size=len(unique_patients))
        indices = np.concatenate(
            [row_indices[unique_patients[int(index)]] for index in sampled]
        )
        aucs: list[float] = []
        macro_defined = True
        for target_index in range(len(target_names)):
            auc = _roc_auc(
                label_values[indices, target_index],
                prediction_values[indices, target_index],
            )
            if auc is None:
                target_undefined[target_index] += 1
                macro_defined = False
            else:
                target_replicates[target_index].append(auc)
                aucs.append(auc)
        if macro_defined:
            macro_replicates.append(float(np.mean(aucs)))
        else:
            macro_undefined += 1

    uncertainty: dict[str, Any] = {"per_target": {}}
    for index, target in enumerate(target_names):
        uncertainty["per_target"][target] = _interval_record(
            target_replicates[index],
            undefined_count=target_undefined[index],
            bootstrap=bootstrap,
        )
    uncertainty["official_macro_auc"] = _interval_record(
        macro_replicates,
        undefined_count=macro_undefined,
        bootstrap=bootstrap,
    )
    return {
        "metric": {
            "name": "macro_averaged_roc_auc",
            "aggregation": "unweighted_mean_across_all_required_targets",
            "point_estimate": point,
        },
        "bootstrap": {
            "method_version": bootstrap.method_version,
            "sampling_unit": "PatientID",
            "sampling": (
                "sample the frozen holdout PatientIDs with replacement using the "
                "original distinct-PatientID count; include every study for each draw"
            ),
            "seed": bootstrap.seed,
            "replicate_count": bootstrap.replicate_count,
            "confidence_level": bootstrap.confidence_level,
            "interval": "two-sided percentile interval over defined replicates",
            "quantile_method": "linear",
            "undefined_metric_policy": {
                "target_auc": (
                    "undefined when a replicate lacks either organizer-gold class"
                ),
                "macro_auc": ("undefined when any required target AUC is undefined"),
                "default_scores_assigned": False,
                "defined_target_only_macro_average": False,
            },
            "uncertainty": uncertainty,
        },
    }


def evaluate_point_predictions(
    *,
    labels: Sequence[Sequence[Label]],
    predictions: Sequence[Sequence[float]],
    target_names: Sequence[str],
) -> dict[str, Any]:
    """Compute per-target and official macro AUC without resampling."""
    label_values = np.asarray(
        [[np.nan if value is None else value for value in row] for row in labels],
        dtype=np.float64,
    )
    prediction_values = np.asarray(predictions, dtype=np.float64)
    expected_shape = (len(labels), len(target_names))
    if (
        label_values.shape != expected_shape
        or prediction_values.shape != expected_shape
    ):
        raise ValueError("Holdout labels and predictions do not match target shape")
    if (
        not np.isfinite(prediction_values).all()
        or ((prediction_values < 0) | (prediction_values > 1)).any()
    ):
        raise ValueError("Holdout predictions must be finite probabilities")
    return _metric_record(label_values, prediction_values, target_names)


def counts_with_prevalence(
    counts: Mapping[str, Mapping[str, int]], target_names: Sequence[str]
) -> dict[str, dict[str, int | float | None]]:
    """Add positive prevalence to the complete known/missing count record."""
    result: dict[str, dict[str, int | float | None]] = {}
    for target in target_names:
        values = counts[target]
        known = int(values["known"])
        result[target] = {
            "known": known,
            "missing": int(values["missing"]),
            "positive": int(values["positive"]),
            "negative": int(values["negative"]),
            "positive_prevalence": (
                float(values["positive"]) / known if known else None
            ),
        }
    return result


def _metric_record(
    labels: np.ndarray, predictions: np.ndarray, targets: Sequence[str]
) -> dict[str, Any]:
    per_target: dict[str, Any] = {}
    aucs: list[float] = []
    macro_defined = True
    for index, target in enumerate(targets):
        target_labels = labels[:, index]
        known = ~np.isnan(target_labels)
        positives = int((target_labels[known] == 1).sum())
        negatives = int((target_labels[known] == 0).sum())
        auc = _roc_auc(target_labels, predictions[:, index])
        per_target[target] = {
            "roc_auc": auc,
            "known": int(known.sum()),
            "missing": int((~known).sum()),
            "positive": positives,
            "negative": negatives,
            "positive_prevalence": (
                positives / int(known.sum()) if bool(known.any()) else None
            ),
        }
        if auc is None:
            macro_defined = False
        else:
            aucs.append(auc)
    return {
        "official_macro_auc": float(np.mean(aucs)) if macro_defined else None,
        "per_target": per_target,
    }


def _roc_auc(labels: np.ndarray, predictions: np.ndarray) -> float | None:
    known = ~np.isnan(labels)
    target_labels = labels[known]
    target_predictions = predictions[known]
    positive = target_predictions[target_labels == 1]
    negative = target_predictions[target_labels == 0]
    if not len(positive) or not len(negative):
        return None
    comparisons = (positive[:, None] > negative[None, :]).astype(np.float64)
    comparisons += 0.5 * (positive[:, None] == negative[None, :])
    return float(comparisons.mean())


def _interval_record(
    values: Sequence[float],
    *,
    undefined_count: int,
    bootstrap: PatientBootstrapConfig,
) -> dict[str, Any]:
    valid_count = len(values)
    if valid_count + undefined_count != bootstrap.replicate_count:
        raise ValueError("Bootstrap replicate accounting is inconsistent")
    interval: dict[str, float] | None = None
    if values:
        tail = (1 - bootstrap.confidence_level) / 2
        lower, upper = np.quantile(values, [tail, 1 - tail], method="linear")
        interval = {"lower": float(lower), "upper": float(upper)}
    return {
        "interval": interval,
        "valid_replicate_count": valid_count,
        "undefined_replicate_count": undefined_count,
        "valid_replicate_rate": valid_count / bootstrap.replicate_count,
        "undefined_replicate_rate": undefined_count / bootstrap.replicate_count,
    }


def _patient_ids(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != _STUDY_COLUMNS:
            raise ValueError("Unexpected CSV columns for studies.csv")
        result: dict[str, str] = {}
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed CSV row in studies.csv")
            if row["split"] != "train":
                continue
            study_uid = row["StudyInstanceUID"]
            if study_uid in result:
                raise ValueError("Index contains a duplicate training study")
            result[study_uid] = row["PatientID"]
    return result


def _solve_two_partition_constraints(
    patient_ids: Sequence[str],
    constraints: Sequence[frozenset[str]],
    seed: int,
) -> dict[str, Partition] | None:
    order = sorted(
        patient_ids,
        key=lambda patient: (
            hashlib.sha256(f"{seed}\0{patient}".encode()).digest(),
            patient.encode("utf-8"),
        ),
    )

    def propagate(
        assignment: dict[str, Partition],
    ) -> dict[str, Partition] | None:
        changed = True
        while changed:
            changed = False
            for constraint in constraints:
                assigned = {
                    assignment[patient]
                    for patient in constraint
                    if patient in assignment
                }
                if len(assigned) == 2:
                    continue
                unassigned = [
                    patient for patient in constraint if patient not in assignment
                ]
                if not unassigned:
                    return None
                if len(unassigned) == 1 and len(assigned) == 1:
                    forced: Partition = "holdout" if "train" in assigned else "train"
                    assignment[unassigned[0]] = forced
                    changed = True
        return assignment

    def search(assignment: dict[str, Partition]) -> dict[str, Partition] | None:
        propagated = propagate(assignment.copy())
        if propagated is None:
            return None
        next_patient = next(
            (patient for patient in order if patient not in propagated), None
        )
        if next_patient is None:
            return propagated
        for partition in _PARTITIONS:
            candidate = propagated.copy()
            candidate[next_patient] = partition
            result = search(candidate)
            if result is not None:
                return result
        return None

    return search({})


def _assignment_digest(rows: Sequence[PartitionedGoldStudy], seed: int) -> str:
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
    for row in sorted(
        rows,
        key=lambda item: (
            item.patient_id.encode("utf-8"),
            item.study_uid.encode("utf-8"),
        ),
    ):
        digest.update(
            _canonical_json([row.patient_id, row.study_uid, row.partition]) + b"\n"
        )
    return digest.hexdigest()


def _partition_summary(
    rows: Sequence[PartitionedGoldStudy], targets: Sequence[str]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for partition in _PARTITIONS:
        selected = [row for row in rows if row.partition == partition]
        per_target: dict[str, dict[str, int]] = {}
        for index, target in enumerate(targets):
            values = [row.labels[index] for row in selected]
            positive = sum(value == 1 for value in values)
            negative = sum(value == 0 for value in values)
            per_target[target] = {
                "known": positive + negative,
                "missing": sum(value is None for value in values),
                "positive": positive,
                "negative": negative,
            }
        result[partition] = {
            "study_count": len(selected),
            "patient_count": len({row.patient_id for row in selected}),
            "per_target": per_target,
        }
    return result


def _load_supervision_contract(path: Path | None) -> tuple[dict[str, Any], str]:
    if path is not None:
        raw = path.read_bytes()
    else:
        source_path = (
            Path(__file__).resolve().parents[2] / "contracts" / "gold-supervision.json"
        )
        raw = (
            source_path.read_bytes()
            if source_path.is_file()
            else files("rsna_knee").joinpath("gold-supervision.json").read_bytes()
        )
    contract = cast(dict[str, Any], json.loads(raw))
    if contract.get("schema_version") != 1:
        raise ValueError("Unsupported gold-supervision contract schema")
    return contract, hashlib.sha256(raw).hexdigest()


def _mapping(parent: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Contract field {key} must be an object")
    return cast(dict[str, Any], value)


def _integer(parent: Mapping[str, Any], key: str) -> int:
    value = parent.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"Contract field {key} must be an integer")
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
