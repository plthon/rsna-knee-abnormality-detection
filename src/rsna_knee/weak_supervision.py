"""Frozen direct weak-supervision candidates and their validation boundary."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from importlib.resources import files
from pathlib import Path
from typing import Literal, cast

from .competition_contract import (
    load_verified_gold_contract,
    load_verified_submission_contract,
)

type Label = int | None
type CandidateId = Literal["gold_only", "public_direct_v1", "qwen_direct_v1"]
type Provenance = Literal["gold", "public_direct_v1", "qwen_direct_v1"]
type GoldMembership = Literal["gold", "non_gold"]

PUBLIC_VERDICT_TO_LABEL: Mapping[str, Literal[0, 1] | None] = {
    "YES": 1,
    "NO": 0,
    "UNK": None,
}

QWEN_STATE_TO_LABEL: Mapping[str, Label] = {
    "positive": 1,
    "negative": 0,
    "uncertain": None,
    "unknown": None,
}

_PUBLIC_VERDICTS = frozenset(PUBLIC_VERDICT_TO_LABEL)
_QWEN_STATES = frozenset(QWEN_STATE_TO_LABEL)
_CANDIDATE_IDS = frozenset({"gold_only", "public_direct_v1", "qwen_direct_v1"})
_POLICY_VERSION = "report-weak-supervision-v1"
_QWEN_LABEL_COLUMNS = (
    "StudyInstanceUID",
    "target",
    "state",
    "extractor_version",
    "model_revision",
    "prompt_version",
    "normalized_report_sha256",
    "parse_failed",
    "input_truncated",
    "provenance_key",
)
_QWEN_STATUS_COLUMNS = (
    "StudyInstanceUID",
    "extractor_version",
    "model_revision",
    "prompt_version",
    "normalized_report_sha256",
    "parse_failed",
    "retry_count",
    "input_truncated",
    "checkpoint_sequence",
)
_QWEN_EXTRACTOR_VERSION = "report-llm-qwen2.5-3b-v1"
_QWEN_MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
_QWEN_PROMPT_VERSION = "rsna-knee-12target-fourstate-v1"

_EXPECTED_SOURCE_RECORDS: dict[str, dict[str, object]] = {
    "rules": {
        "source_name": "report-extractor-v1",
        "version": "report-extractor-v1",
        "provenance_status": "diagnostic_reference",
        "role_in_v1": "diagnostic_reference",
        "training_activation": False,
        "raw_state_domain": ("positive", "negative", "uncertain", "unknown"),
        "training_mapping": {
            "positive": 1,
            "negative": 0,
            "uncertain": None,
            "unknown": None,
        },
        "masked_states": ("uncertain", "unknown"),
        "parse_failure_policy": "not_applicable",
        "artifact_identities": {
            "contracts/report-extractor-v1.json": {
                "sha256": "901e8a53173fef0d85e12ed93cbea05dd5b8cfc900ef111e0125cb751634156a",
                "size_bytes": 15896,
            },
            "report-labels-v1.csv": {
                "sha256": "6651f60b00e5bec292e8c98e7fa83c75e8aa5af5931414a9a68c8ce10a52d527",
                "size_bytes": 12409688,
            },
        },
    },
    "qwen": {
        "source_name": "report-llm-qwen2.5-3b-v1",
        "version": "report-llm-qwen2.5-3b-v1",
        "provenance_status": "gold_independent_reproducible_candidate",
        "role_in_v1": "weak_supervision_candidate",
        "raw_state_domain": ("positive", "negative", "uncertain", "unknown"),
        "training_mapping": {
            "positive": 1,
            "negative": 0,
            "uncertain": None,
            "unknown": None,
        },
        "masked_states": ("uncertain", "unknown"),
        "parse_failure_policy": "parse_failed_true_always_masked",
        "artifact_identities": {
            "contracts/report-llm-extractor-v1.json": {
                "sha256": "c428d1b44bb6c87f84e7423feeaefa6c635a5ba1cb5dd277468128d7df0bbe51",
                "size_bytes": 4364,
            },
            "report-llm-labels-v1.csv": {
                "sha256": "d78c7f1d27437f6d2870e48fbd730a57c60d34380720cd4e0e07811a699db52a",
                "size_bytes": 17063200,
            },
            "report-llm-study-status-v1.csv": {
                "sha256": "151d0554cbd7622028df1bd58a4e2c37d18796c5860502cf3eb0917ec6baf923",
                "size_bytes": 1079236,
            },
        },
    },
    "public": {
        "source_name": "pilkwang/rsna-knee-llm-labels",
        "version": 1,
        "provenance_status": "external_benchmark_with_possible_gold_influence",
        "role_in_v1": "weak_supervision_candidate",
        "training_columns": "verdict columns only",
        "score_columns_training_effect": False,
        "confidence_columns_training_effect": False,
        "raw_state_domain": ("YES", "NO", "UNK"),
        "training_mapping": {"YES": 1, "NO": 0, "UNK": None},
        "masked_states": ("UNK",),
        "parse_failure_policy": "not_applicable",
        "artifact_identities": {
            "contracts/public-report-label-dataset-v1.json": {
                "sha256": "3b607558a5bea66d2d10dd3884bd6fa9e2ac81b4617c6ea372d7feb0c00068e9",
                "size_bytes": 5901,
            },
            "report_labels_v2.csv": {
                "sha256": "6f704a7bdb2f894cc49445b19ba7c4378c3f548d3449e00361e10044bee40920",
                "size_bytes": 1004839,
            },
        },
    },
}

_EXPECTED_CANDIDATE_RECORDS: dict[str, dict[str, object]] = {
    "gold_only": {
        "source": None,
        "positive_mapping": 1,
        "negative_mapping": 0,
        "masked_mapping": None,
        "gold_study_policy": (
            "Use organizer-gold labels only; no report-derived source contributes to loss"
        ),
        "provenance_classification": "gold_control",
    },
    "public_direct_v1": {
        "source": "public",
        "positive_mapping": 1,
        "negative_mapping": 0,
        "masked_mapping": None,
        "gold_study_policy": (
            "Use authoritative organizer-gold labels for gold studies; "
            "exclude gold studies from public supervision"
        ),
        "provenance_classification": "external_benchmark_with_possible_gold_influence",
    },
    "qwen_direct_v1": {
        "source": "qwen",
        "positive_mapping": 1,
        "negative_mapping": 0,
        "masked_mapping": None,
        "gold_study_policy": (
            "Use authoritative organizer-gold labels for gold studies; "
            "exclude gold studies from Qwen supervision"
        ),
        "provenance_classification": "gold_independent_reproducible_candidate",
    },
}


@dataclass(frozen=True, slots=True)
class WeakSupervisionCandidate:
    candidate_id: str
    source: str | None
    target_order: tuple[str, ...]
    positive_mapping: object
    negative_mapping: object
    masked_mapping: object
    gold_study_policy: str
    provenance_classification: str


@dataclass(frozen=True, slots=True)
class WeakSupervisionPolicy:
    schema_version: int
    policy_version: str
    target_order: tuple[str, ...]
    gold_firewall: Mapping[str, object]
    sources: Mapping[str, Mapping[str, object]]
    candidates: Mapping[str, WeakSupervisionCandidate]
    forbidden_v1: tuple[str, ...]
    selection_owner: str
    inventory: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class TrainingLabel:
    """One deterministic study-target label at the image-model training seam."""

    study_uid: str
    target: str
    value: Label
    observed: bool
    provenance: Provenance
    source_state: str

    @property
    def StudyInstanceUID(self) -> str:
        return self.study_uid

    def as_dict(self) -> dict[str, object]:
        return {
            "StudyInstanceUID": self.study_uid,
            "target": self.target,
            "value": self.value,
            "observed": self.observed,
            "provenance": self.provenance,
            "source_state": self.source_state,
        }


def load_weak_supervision_policy(
    path: Path | None = None,
    *,
    competition_contract_path: Path | None = None,
) -> WeakSupervisionPolicy:
    """Load the frozen policy and reject target, source, or candidate drift."""
    raw_bytes = _policy_bytes(path)
    raw_value = json.loads(raw_bytes)
    if not isinstance(raw_value, dict):
        raise TypeError("Weak-supervision policy must be a JSON object")
    raw = cast(dict[str, object], raw_value)

    required = {
        "schema_version",
        "policy_version",
        "target_order",
        "gold_firewall",
        "sources",
        "candidates",
        "forbidden_v1",
        "selection_owner",
        "inventory",
    }
    missing = required - set(raw)
    if missing:
        raise ValueError(
            f"Weak-supervision policy is missing fields: {sorted(missing)}"
        )
    if raw.get("schema_version") != 1:
        raise ValueError("Unsupported weak-supervision policy schema")
    if raw.get("policy_version") != _POLICY_VERSION:
        raise ValueError("Weak-supervision policy version is not frozen v1")
    if raw.get("selection_owner") != "issue-10":
        raise ValueError("Weak-supervision selection owner must remain issue-10")

    target_order = _string_tuple(raw.get("target_order"), "target_order")
    competition_contract = load_verified_submission_contract(competition_contract_path)
    expected_targets = tuple(competition_contract.target_names)
    if target_order != expected_targets:
        raise ValueError(
            "Weak-supervision target order differs from competition contract"
        )
    target_order_source = {
        "path": "contracts/competition.json",
        "sha256": "3c6d42d4a4883a9626294c79af02f2ff687ba9413da8f316db9b877ebb1e2292",
    }
    if raw.get("target_order_source") != target_order_source:
        raise ValueError("Weak-supervision target-order source identity differs")
    if competition_contract.sha256 != target_order_source["sha256"]:
        raise ValueError("Competition contract identity differs from frozen policy")

    firewall = _mapping(raw, "gold_firewall")
    if firewall.get("identifier_column") != "StudyInstanceUID":
        raise ValueError("Weak-supervision gold firewall identifier is invalid")
    if (
        firewall.get("gold_study_count") != 58
        or firewall.get("non_gold_study_count") != 4349
    ):
        raise ValueError("Weak-supervision gold firewall cohort counts are not frozen")
    if firewall.get("partial_label_policy") != "reject":
        raise ValueError("Weak-supervision partial-label policy must reject")

    raw_sources = _mapping(raw, "sources")
    if set(raw_sources) != set(_EXPECTED_SOURCE_RECORDS):
        raise ValueError("Weak-supervision source set is not frozen")
    sources: dict[str, Mapping[str, object]] = {}
    for source_name, expected in _EXPECTED_SOURCE_RECORDS.items():
        source = _mapping(raw_sources, source_name)
        _validate_source_record(source_name, source, expected)
        sources[source_name] = source
    _verify_policy_contract_artifacts(sources)

    raw_candidates = _mapping(raw, "candidates")
    if set(raw_candidates) != _CANDIDATE_IDS:
        raise ValueError(
            "Weak-supervision candidates are not exactly the frozen v1 set"
        )
    candidates: dict[str, WeakSupervisionCandidate] = {}
    for candidate_id in sorted(_CANDIDATE_IDS):
        candidate = _mapping(raw_candidates, candidate_id)
        if candidate.get("candidate_id") != candidate_id:
            raise ValueError("Weak-supervision candidate ID is inconsistent")
        candidate_source = candidate.get("source")
        if candidate_source is not None and not isinstance(candidate_source, str):
            raise TypeError(f"Candidate {candidate_id} source must be a string or null")
        if candidate_source is not None and candidate_source not in sources:
            raise ValueError(f"Candidate {candidate_id} references an unknown source")
        candidate_targets = _string_tuple(
            candidate.get("target_order"), f"candidate {candidate_id} target_order"
        )
        if candidate_targets != target_order:
            raise ValueError(
                f"Candidate {candidate_id} target order differs from policy"
            )
        expected_candidate = _EXPECTED_CANDIDATE_RECORDS[candidate_id]
        if set(candidate) != {"candidate_id", "target_order", *expected_candidate}:
            raise ValueError(f"Frozen candidate {candidate_id} fields differ")
        for field, expected_value in expected_candidate.items():
            if field not in candidate or candidate[field] != expected_value:
                raise ValueError(
                    f"Frozen candidate {candidate_id} field differs: {field}"
                )
        candidates[candidate_id] = WeakSupervisionCandidate(
            candidate_id=candidate_id,
            source=candidate_source,
            target_order=candidate_targets,
            positive_mapping=candidate.get("positive_mapping"),
            negative_mapping=candidate.get("negative_mapping"),
            masked_mapping=candidate.get("masked_mapping"),
            gold_study_policy=_required_text(candidate, "gold_study_policy"),
            provenance_classification=_required_text(
                candidate, "provenance_classification"
            ),
        )

    forbidden = _string_tuple(raw.get("forbidden_v1"), "forbidden_v1")
    inventory = _mapping(raw, "inventory") if "inventory" in raw else {}
    return WeakSupervisionPolicy(
        schema_version=1,
        policy_version=_POLICY_VERSION,
        target_order=target_order,
        gold_firewall=firewall,
        sources=sources,
        candidates=candidates,
        forbidden_v1=forbidden,
        selection_owner="issue-10",
        inventory=inventory,
    )


def verify_source_identity(
    path: Path,
    expected_sha256: str,
    *,
    expected_size_bytes: int | None = None,
    description: str = "Source artifact",
) -> str:
    """Verify an attached source artifact before reading its labels."""
    _validate_sha256(expected_sha256, "expected_sha256")
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    if expected_size_bytes is not None and path.stat().st_size != expected_size_bytes:
        raise ValueError(
            f"{description} size mismatch; expected {expected_size_bytes}, "
            f"got {path.stat().st_size}"
        )
    actual_sha256 = _file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{description} SHA-256 mismatch; expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    return actual_sha256


def public_label_columns(target_order: Sequence[str]) -> tuple[str, ...]:
    """Return the one frozen wide-column layout used by #24 and #27."""
    columns = ["StudyInstanceUID"]
    for target in target_order:
        columns.extend((target, f"{target}__conf", f"{target}__verdict"))
    return tuple(columns)


def classify_gold_membership(values: Sequence[object]) -> GoldMembership:
    """Apply the shared all-present/all-missing cohort boundary."""
    present = [value is not None and bool(str(value).strip()) for value in values]
    if not any(present):
        return "non_gold"
    if all(present):
        return "gold"
    raise ValueError("Partially labeled study is not allowed by the gold firewall")


def resolve_weak_labels(
    candidate_id: str,
    *,
    gold_rows: Sequence[Mapping[str, object]] = (),
    source_rows: Sequence[Mapping[str, object]] = (),
    qwen_status_rows: Sequence[Mapping[str, object]] = (),
    policy: WeakSupervisionPolicy | None = None,
    gold_study_uids: Collection[str] | None = None,
) -> tuple[TrainingLabel, ...]:
    """Resolve one registered candidate into common study-target labels."""
    resolved_policy = policy or load_weak_supervision_policy()
    candidate = resolved_policy.candidates.get(candidate_id)
    if candidate is None or candidate_id not in _CANDIDATE_IDS:
        raise ValueError(
            f"Unknown or unregistered weak-supervision candidate: {candidate_id}"
        )

    gold_labels, derived_gold_uids, known_study_uids = _parse_gold_rows(
        gold_rows, resolved_policy.target_order
    )
    declared_gold_uids = _normalize_uid_set(gold_study_uids)
    if (
        derived_gold_uids
        and declared_gold_uids
        and derived_gold_uids != declared_gold_uids
    ):
        raise ValueError("Explicit gold-study membership differs from train.csv")
    firewall_gold_uids = declared_gold_uids or derived_gold_uids

    if candidate_id == "gold_only":
        if source_rows or qwen_status_rows:
            raise ValueError("gold_only cannot be combined with a weak source")
        return _ordered_labels(gold_labels, resolved_policy.target_order)

    if candidate.source not in {"public", "qwen"}:
        raise ValueError(f"Candidate {candidate_id} has no registered direct source")
    if known_study_uids:
        unknown_source_uids = {_row_uid(row) for row in source_rows} - known_study_uids
        if unknown_source_uids:
            raise ValueError("Weak source contains a study outside the verified cohort")

    if candidate.source == "public":
        weak_labels = _parse_public_rows(source_rows, resolved_policy.target_order)
    else:
        weak_labels = _parse_qwen_rows(
            source_rows,
            qwen_status_rows,
            resolved_policy.target_order,
        )

    if known_study_uids:
        source_uids = {label.study_uid for label in weak_labels}
        expected_non_gold_uids = known_study_uids - firewall_gold_uids
        if source_uids - firewall_gold_uids != expected_non_gold_uids:
            raise ValueError("Weak source coverage differs from the verified cohort")

    filtered_weak = tuple(
        label for label in weak_labels if label.study_uid not in firewall_gold_uids
    )
    return _ordered_labels((*gold_labels, *filtered_weak), resolved_policy.target_order)


def load_candidate_labels(
    candidate_id: str,
    train_csv_path: Path,
    *,
    public_labels_path: Path | None = None,
    qwen_labels_path: Path | None = None,
    qwen_status_path: Path | None = None,
    policy_path: Path | None = None,
    competition_contract_path: Path | None = None,
) -> tuple[TrainingLabel, ...]:
    """Load a candidate from attached CSVs after contract and hash validation."""
    policy = load_weak_supervision_policy(
        policy_path, competition_contract_path=competition_contract_path
    )
    if candidate_id not in policy.candidates:
        raise ValueError(
            f"Unknown or unregistered weak-supervision candidate: {candidate_id}"
        )
    gold_contract = load_verified_gold_contract(competition_contract_path)
    gold_rows = _read_csv(train_csv_path, gold_contract.train_columns)

    if candidate_id == "gold_only":
        if public_labels_path or qwen_labels_path or qwen_status_path:
            raise ValueError("gold_only cannot be combined with a weak source")
        return resolve_weak_labels(candidate_id, gold_rows=gold_rows, policy=policy)
    if candidate_id == "public_direct_v1":
        if public_labels_path is None or qwen_labels_path or qwen_status_path:
            raise ValueError("public_direct_v1 requires only the frozen public labels")
        public_identity = _artifact_identity(policy, "public", "report_labels_v2.csv")
        verify_source_identity(
            public_labels_path,
            _required_text(public_identity, "sha256"),
            expected_size_bytes=_required_int(public_identity, "size_bytes"),
            description="Public label file",
        )
        public_rows = _read_csv(
            public_labels_path, public_label_columns(policy.target_order)
        )
        return resolve_weak_labels(
            candidate_id,
            gold_rows=gold_rows,
            source_rows=public_rows,
            policy=policy,
        )
    if candidate_id == "qwen_direct_v1":
        if qwen_labels_path is None or qwen_status_path is None or public_labels_path:
            raise ValueError("qwen_direct_v1 requires both frozen Qwen CSVs only")
        label_identity = _artifact_identity(policy, "qwen", "report-llm-labels-v1.csv")
        status_identity = _artifact_identity(
            policy, "qwen", "report-llm-study-status-v1.csv"
        )
        verify_source_identity(
            qwen_labels_path,
            _required_text(label_identity, "sha256"),
            expected_size_bytes=_required_int(label_identity, "size_bytes"),
            description="Qwen label file",
        )
        verify_source_identity(
            qwen_status_path,
            _required_text(status_identity, "sha256"),
            expected_size_bytes=_required_int(status_identity, "size_bytes"),
            description="Qwen study-status file",
        )
        return resolve_weak_labels(
            candidate_id,
            gold_rows=gold_rows,
            source_rows=_read_csv(qwen_labels_path, _QWEN_LABEL_COLUMNS),
            qwen_status_rows=_read_csv(qwen_status_path, _QWEN_STATUS_COLUMNS),
            policy=policy,
        )
    raise ValueError(
        f"Unknown or unregistered weak-supervision candidate: {candidate_id}"
    )


def _parse_gold_rows(
    rows: Sequence[Mapping[str, object]], target_order: Sequence[str]
) -> tuple[tuple[TrainingLabel, ...], set[str], set[str]]:
    labels: list[TrainingLabel] = []
    gold_uids: set[str] = set()
    known_uids: set[str] = set()
    seen_uids: set[str] = set()
    for row in rows:
        if any(target not in row for target in target_order):
            raise ValueError("Gold rows are missing required target columns")
        study_uid = _row_uid(row)
        if study_uid in seen_uids:
            raise ValueError(f"Duplicate study-target label in gold rows: {study_uid}")
        seen_uids.add(study_uid)
        known_uids.add(study_uid)
        values = [row[target] for target in target_order]
        membership = classify_gold_membership(values)
        if membership != "gold":
            continue
        gold_uids.add(study_uid)
        for target, raw_value in zip(target_order, values, strict=True):
            value = _parse_gold_value(raw_value)
            labels.append(
                TrainingLabel(
                    study_uid=study_uid,
                    target=target,
                    value=value,
                    observed=value is not None,
                    provenance="gold",
                    source_state=str(raw_value).strip(),
                )
            )
    return tuple(labels), gold_uids, known_uids


def _parse_public_rows(
    rows: Sequence[Mapping[str, object]], target_order: Sequence[str]
) -> tuple[TrainingLabel, ...]:
    expected_columns = public_label_columns(target_order)
    labels: list[TrainingLabel] = []
    seen_keys: set[tuple[str, str]] = set()
    for row in rows:
        _validate_columns(row, expected_columns, "public source")
        study_uid = _row_uid(row)
        for target in target_order:
            key = (study_uid, target)
            if key in seen_keys:
                raise ValueError(
                    f"Duplicate study-target label in public source: {key}"
                )
            seen_keys.add(key)
            source_state = _required_text_value(
                row.get(f"{target}__verdict"),
                f"public verdict for {study_uid} target {target}",
            )
            if source_state not in _PUBLIC_VERDICTS:
                raise ValueError(
                    f"Invalid public verdict {source_state!r} for "
                    f"study {study_uid} target {target}"
                )
            value = PUBLIC_VERDICT_TO_LABEL[source_state]
            labels.append(
                TrainingLabel(
                    study_uid=study_uid,
                    target=target,
                    value=value,
                    observed=value is not None,
                    provenance="public_direct_v1",
                    source_state=source_state,
                )
            )
    return tuple(labels)


def _parse_qwen_rows(
    rows: Sequence[Mapping[str, object]],
    status_rows: Sequence[Mapping[str, object]],
    target_order: Sequence[str],
) -> tuple[TrainingLabel, ...]:
    if rows and not status_rows:
        raise ValueError("Qwen study-status rows are required")
    statuses = _parse_qwen_status_rows(status_rows)
    labels: list[TrainingLabel] = []
    seen_keys: set[tuple[str, str]] = set()
    target_indices = {target: index for index, target in enumerate(target_order)}
    targets_by_uid: dict[str, list[str]] = {}
    for row in rows:
        _validate_columns(row, _QWEN_LABEL_COLUMNS, "Qwen label source")
        study_uid = _row_uid(row)
        target = _required_text_value(row.get("target"), "Qwen target")
        if target not in target_indices:
            raise ValueError(f"Unknown target name in Qwen source: {target}")
        key = (study_uid, target)
        if key in seen_keys:
            raise ValueError(f"Duplicate study-target label in Qwen source: {key}")
        seen_keys.add(key)
        state = _required_text_value(row.get("state"), f"Qwen state for {key}")
        if state not in _QWEN_STATES:
            raise ValueError(f"Invalid Qwen state {state!r} for {key}")
        _validate_qwen_metadata(row)
        parse_failed = _parse_bool(row.get("parse_failed"), "Qwen parse_failed")
        status = statuses.get(study_uid)
        if status is None:
            raise ValueError(f"Qwen label has no matching study status: {study_uid}")
        if parse_failed != status["parse_failed"]:
            raise ValueError(f"Qwen parse_failed identity mismatch for {study_uid}")
        if str(row["normalized_report_sha256"]).strip() != status["report_sha256"]:
            raise ValueError(f"Qwen report identity mismatch for {study_uid}")
        _validate_sha256(
            str(row["normalized_report_sha256"]).strip(),
            "normalized_report_sha256",
        )
        _validate_sha256(
            _required_text_value(row.get("provenance_key"), "Qwen provenance_key"),
            "provenance_key",
        )
        _parse_bool(row.get("input_truncated"), "Qwen input_truncated")
        value = None if parse_failed else QWEN_STATE_TO_LABEL[state]
        labels.append(
            TrainingLabel(
                study_uid=study_uid,
                target=target,
                value=value,
                observed=value is not None,
                provenance="qwen_direct_v1",
                source_state=state,
            )
        )
        targets_by_uid.setdefault(study_uid, []).append(target)

    for study_uid, observed_targets in targets_by_uid.items():
        if tuple(observed_targets) != tuple(target_order):
            raise ValueError(f"Qwen target order drift for study {study_uid}")
    if set(targets_by_uid) != set(statuses):
        raise ValueError("Qwen label and study-status study identities differ")
    return tuple(labels)


def _parse_qwen_status_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    statuses: dict[str, dict[str, object]] = {}
    for row in rows:
        _validate_columns(row, _QWEN_STATUS_COLUMNS, "Qwen study-status source")
        study_uid = _row_uid(row)
        if study_uid in statuses:
            raise ValueError(f"Duplicate Qwen study status: {study_uid}")
        _validate_qwen_metadata(row)
        parse_failed = _parse_bool(row.get("parse_failed"), "Qwen parse_failed")
        report_sha256 = _required_text_value(
            row.get("normalized_report_sha256"),
            f"Qwen report identity for {study_uid}",
        )
        _validate_sha256(report_sha256, "normalized_report_sha256")
        statuses[study_uid] = {
            "parse_failed": parse_failed,
            "report_sha256": report_sha256,
        }
    return statuses


def _validate_qwen_metadata(row: Mapping[str, object]) -> None:
    expected = {
        "extractor_version": _QWEN_EXTRACTOR_VERSION,
        "model_revision": _QWEN_MODEL_REVISION,
        "prompt_version": _QWEN_PROMPT_VERSION,
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise ValueError(f"Qwen {field} differs from the frozen contract")


def _validate_columns(
    row: Mapping[str, object], expected: Sequence[str], source_name: str
) -> None:
    actual = tuple(row)
    if actual != tuple(expected):
        actual_set = set(actual)
        expected_set = set(expected)
        if actual_set - expected_set:
            raise ValueError(f"Unknown target names or columns in {source_name}")
        if actual_set == expected_set:
            raise ValueError(f"{source_name} target order drift")
        raise ValueError(f"{source_name} required columns differ from frozen schema")


def _ordered_labels(
    labels: Sequence[TrainingLabel], target_order: Sequence[str]
) -> tuple[TrainingLabel, ...]:
    target_indices = {target: index for index, target in enumerate(target_order)}
    return tuple(
        sorted(
            labels,
            key=lambda label: (
                label.study_uid.encode("utf-8"),
                target_indices[label.target],
            ),
        )
    )


def _row_uid(row: Mapping[str, object]) -> str:
    value = row.get("StudyInstanceUID")
    uid = "" if value is None else str(value).strip()
    if not uid:
        raise ValueError("StudyInstanceUID is missing or blank")
    return uid


def _normalize_uid_set(values: Collection[str] | None) -> set[str]:
    if values is None:
        return set()
    normalized = {str(value).strip() for value in values}
    if "" in normalized:
        raise ValueError("Gold-study membership contains a blank StudyInstanceUID")
    return normalized


def _parse_gold_value(raw: object) -> Label:
    stripped = "" if raw is None else str(raw).strip()
    if not stripped:
        return None
    try:
        value = Decimal(stripped)
    except InvalidOperation as error:
        raise ValueError("Organizer label is not numeric") from error
    if not value.is_finite() or value not in {Decimal(0), Decimal(1)}:
        raise ValueError("Organizer label is outside {0,1,missing}")
    return 1 if value == 1 else 0


def _required_text_value(value: object, field: str) -> str:
    if value is None:
        raise ValueError(f"{field} is missing")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} is blank")
    return text


def _parse_bool(value: object, field: str) -> bool:
    if isinstance(value, bool):
        return value
    text = _required_text_value(value, field)
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"{field} must be exactly true or false")


def _read_csv(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != tuple(columns):
            raise ValueError(
                f"Unexpected CSV columns for {path.name}: "
                f"{reader.fieldnames} vs expected {columns}"
            )
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV row in {path.name}")
    return cast(list[dict[str, str]], rows)


def _policy_bytes(path: Path | None) -> bytes:
    if path is not None:
        return path.read_bytes()
    repository_path = (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / ("report-weak-supervision-v1.json")
    )
    if repository_path.is_file():
        return repository_path.read_bytes()
    return files("rsna_knee").joinpath("report-weak-supervision-v1.json").read_bytes()


def _artifact_identity(
    policy: WeakSupervisionPolicy, source_name: str, filename: str
) -> dict[str, object]:
    source = policy.sources.get(source_name)
    if source is None:
        raise ValueError(f"Policy has no source: {source_name}")
    identities = _mapping(source, "artifact_identities")
    return _mapping(identities, filename)


def _verify_policy_contract_artifacts(
    sources: Mapping[str, Mapping[str, object]],
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    for source in sources.values():
        identities = _mapping(source, "artifact_identities")
        for filename in identities:
            if not filename.startswith("contracts/"):
                continue
            identity = _mapping(identities, filename)
            path = repository_root / filename
            if path.is_file():
                verify_source_identity(
                    path,
                    _required_text(identity, "sha256"),
                    expected_size_bytes=_required_int(identity, "size_bytes"),
                    description=f"Frozen source contract {filename}",
                )


def _validate_source_record(
    source_name: str,
    source: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    if set(source) != set(expected):
        raise ValueError(f"Frozen {source_name} source fields differ")
    for field in (
        "source_name",
        "version",
        "provenance_status",
        "role_in_v1",
        "training_activation",
        "training_columns",
        "score_columns_training_effect",
        "confidence_columns_training_effect",
        "parse_failure_policy",
    ):
        if field in expected and source.get(field) != expected[field]:
            raise ValueError(f"Frozen {source_name} source field differs: {field}")
    if _string_tuple(
        source.get("raw_state_domain"), f"{source_name} raw_state_domain"
    ) != tuple(cast(tuple[str, ...], expected["raw_state_domain"])):
        raise ValueError(f"Frozen {source_name} state domain differs")
    if source.get("training_mapping") != expected.get("training_mapping"):
        raise ValueError(f"Frozen {source_name} training mapping differs")
    if _string_tuple(
        source.get("masked_states"), f"{source_name} masked_states"
    ) != tuple(cast(tuple[str, ...], expected["masked_states"])):
        raise ValueError(f"Frozen {source_name} masked states differ")
    actual_artifacts = _mapping(source, "artifact_identities")
    expected_artifacts = cast(Mapping[str, object], expected["artifact_identities"])
    if set(actual_artifacts) != set(expected_artifacts):
        raise ValueError(f"Frozen {source_name} artifact set differs")
    for filename, expected_identity in expected_artifacts.items():
        actual_identity = _mapping(actual_artifacts, filename)
        expected_record = cast(Mapping[str, object], expected_identity)
        if actual_identity != dict(expected_record):
            raise ValueError(
                f"Frozen {source_name} artifact identity differs: {filename}"
            )
        _validate_sha256(
            _required_text(actual_identity, "sha256"), f"{filename} sha256"
        )
        _required_int(actual_identity, "size_bytes")


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise TypeError(f"Field {field} must be a list of nonempty strings")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise ValueError(f"Field {field} contains duplicates")
    return result


def _mapping(parent: Mapping[str, object], key: str) -> dict[str, object]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Field {key} must be an object")
    return cast(dict[str, object], value)


def _required_text(parent: Mapping[str, object], key: str) -> str:
    return _required_text_value(parent.get(key), key)


def _required_int(parent: Mapping[str, object], key: str) -> int:
    value = parent.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"Field {key} must be an integer")
    return value


def _validate_sha256(value: str, field: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"Field {field} must be a lowercase SHA-256 digest")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
