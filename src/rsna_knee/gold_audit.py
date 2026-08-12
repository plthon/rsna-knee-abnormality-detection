import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, cast

from .competition_contract import VerifiedGoldContract, load_verified_gold_contract

type Label = Literal[0, 1] | None
type Partition = Literal["train", "holdout"]

_SCHEMA_VERSION = 1
_ARTIFACT_DIRECTORY = "rsna-knee-gold-audit-v1"
_COHORT_RULE_VERSION = "organizer-gold-all-eligible-v1"
_COHORT_DIGEST_VERSION = "study-label-jsonl-v1"
_ALLOCATION_RULE_VERSION = "patient-disjoint-exact-backtracking-v1"
_ASSIGNMENT_DIGEST_VERSION = "patient-study-partition-jsonl-v1"
_DEFAULT_ALLOCATION_SEED = 20260812
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
_COHORT_PRIVATE = "gold-cohort-private.csv"
_EXCLUSIONS_PRIVATE = "gold-exclusions-private.csv"
_INVALID_LABELS_PRIVATE = "invalid-labels-private.csv"
_ASSIGNMENT_PRIVATE = "evaluation-assignment-private.csv"
_SUMMARY = "gold-audit-summary.json"
_MANIFEST = "gold-audit-manifest.json"


@dataclass(frozen=True, slots=True)
class GoldAuditInputs:
    competition_root: Path
    index_artifact_dir: Path


@dataclass(frozen=True, slots=True)
class GoldAuditOutcome:
    artifact_dir: Path
    manifest_path: Path
    summary_path: Path
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _CohortRow:
    source_row: int
    study_uid: str
    patient_id: str
    labels: tuple[Label, ...]


@dataclass(frozen=True, slots=True)
class _ExcludedRow:
    source_row: int
    study_uid: str
    reason: str


@dataclass(frozen=True, slots=True)
class _InvalidLabel:
    source_row: int
    study_uid: str
    target: str
    raw_value: str
    category: str


def discover_gold_audit_inputs(
    input_root: Path,
    *,
    competition_contract_path: Path | None = None,
    data_index_contract_path: Path | None = None,
) -> GoldAuditInputs:
    """Discover the competition mount and exact accepted private index output."""
    gold_contract = load_verified_gold_contract(competition_contract_path)
    competition_candidates = sorted(
        path.parent
        for path in input_root.rglob(gold_contract.train_filename)
        if (path.parent / "sample_submission.csv").is_file()
        and (path.parent / "train_series.csv").is_file()
    )
    if len(competition_candidates) != 1:
        raise ValueError(
            "Expected exactly one competition root containing train.csv, "
            "train_series.csv, and sample_submission.csv"
        )

    index_contract, _ = _load_data_index_contract(data_index_contract_path)
    _validate_data_index_compatibility(index_contract, gold_contract)
    artifact = _required_mapping(index_contract, "artifact")
    artifact_files = _required_mapping(artifact, "files")
    studies_record = _required_mapping(artifact_files, "studies.csv")
    expected_size = _required_int(studies_record, "size_bytes")
    expected_sha256 = _required_text(studies_record, "sha256")
    outputs = _required_mapping(
        _required_mapping(index_contract, "run_provenance"), "outputs"
    )
    expected_manifest_sha256 = _required_text(outputs, "manifest_sha256")

    index_candidates: list[Path] = []
    for studies_path in sorted(input_root.rglob("studies.csv")):
        if (
            studies_path.stat().st_size == expected_size
            and _sha256(studies_path) == expected_sha256
            and (studies_path.parent / "index-manifest.json").is_file()
            and _sha256(studies_path.parent / "index-manifest.json")
            == expected_manifest_sha256
        ):
            index_candidates.append(studies_path.parent)
    if len(index_candidates) != 1:
        raise ValueError(
            "Expected exactly one attached index artifact matching the accepted "
            "studies.csv and index-manifest.json digests"
        )
    return GoldAuditInputs(
        competition_root=competition_candidates[0],
        index_artifact_dir=index_candidates[0],
    )


def audit_gold_supervision(
    competition_root: Path,
    index_artifact_dir: Path,
    work_root: Path,
    *,
    competition_contract_path: Path | None = None,
    data_index_contract_path: Path | None = None,
    allocation_seed: int = _DEFAULT_ALLOCATION_SEED,
    source_provenance: Mapping[str, Any] | None = None,
) -> GoldAuditOutcome:
    """Audit organizer gold labels and freeze one exact eligibility witness."""
    if isinstance(allocation_seed, bool) or not isinstance(allocation_seed, int):
        raise TypeError("allocation_seed must be an integer")
    gold_contract = load_verified_gold_contract(competition_contract_path)
    index_contract, index_contract_sha256 = _load_data_index_contract(
        data_index_contract_path
    )
    _validate_data_index_compatibility(index_contract, gold_contract)
    index_manifest = _verify_accepted_index(
        index_artifact_dir, index_contract, gold_contract
    )

    artifact_dir = work_root / _ARTIFACT_DIRECTORY
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(f"Gold audit artifact already exists: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    index_rows = _read_csv(index_artifact_dir / "studies.csv", _STUDY_COLUMNS)
    index_by_study: dict[str, list[dict[str, str]]] = defaultdict(list)
    usable_train_study_count = 0
    for row in index_rows:
        index_by_study[row["StudyInstanceUID"]].append(row)
        if row["split"] == "train" and _index_row_is_usable(row):
            usable_train_study_count += 1

    train_path = competition_root / gold_contract.train_filename
    train_rows = _read_csv(train_path, gold_contract.train_columns)
    identifier_counts = Counter(
        row[gold_contract.identifier_column].strip() for row in train_rows
    )

    source_target_counts = _empty_target_counts(gold_contract.target_names)
    known_histogram: Counter[int] = Counter()
    cohort: list[_CohortRow] = []
    excluded: list[_ExcludedRow] = []
    invalid_labels: list[_InvalidLabel] = []
    contract_violations: Counter[str] = Counter()

    for source_row, row in enumerate(train_rows, start=2):
        study_uid = row[gold_contract.identifier_column].strip()
        labels: list[Label] = []
        row_invalid: list[_InvalidLabel] = []
        for target in gold_contract.target_names:
            label, invalid_category = _parse_label(row[target])
            labels.append(label)
            if invalid_category is not None:
                item = _InvalidLabel(
                    source_row=source_row,
                    study_uid=study_uid,
                    target=target,
                    raw_value=row[target],
                    category=invalid_category,
                )
                invalid_labels.append(item)
                row_invalid.append(item)
            elif label is None:
                source_target_counts[target]["missing"] += 1
            elif label == 1:
                source_target_counts[target]["known"] += 1
                source_target_counts[target]["positive"] += 1
            else:
                source_target_counts[target]["known"] += 1
                source_target_counts[target]["negative"] += 1
        known_count = sum(label is not None for label in labels)
        known_histogram[known_count] += 1

        reason: str | None = None
        if row_invalid:
            reason = "invalid_label_value"
            contract_violations[reason] += 1
        elif not study_uid:
            reason = "missing_study_identifier"
            contract_violations[reason] += 1
        elif identifier_counts[study_uid] != 1:
            reason = "duplicate_study_identifier"
            contract_violations[reason] += 1
        elif known_count == 0:
            reason = "no_verified_gold_label"
        else:
            linked_rows = index_by_study.get(study_uid, [])
            if not linked_rows:
                reason = "index_study_not_found"
            elif len(linked_rows) != 1:
                reason = "index_study_ambiguous"
            elif linked_rows[0]["split"] != "train":
                reason = "index_study_not_train"
            elif not _index_row_is_usable(linked_rows[0]):
                reason = "index_study_unusable"
            else:
                cohort.append(
                    _CohortRow(
                        source_row=source_row,
                        study_uid=study_uid,
                        patient_id=linked_rows[0][
                            gold_contract.patient_group_column
                        ].strip(),
                        labels=tuple(labels),
                    )
                )
        if reason is not None:
            excluded.append(
                _ExcludedRow(
                    source_row=source_row,
                    study_uid=study_uid,
                    reason=reason,
                )
            )

    cohort.sort(key=lambda row: row.study_uid.encode("utf-8"))
    cohort_counts = _target_counts(cohort, gold_contract.target_names)
    unsupported_smoke = _unsupported_smoke_targets(
        cohort_counts, gold_contract.target_names
    )
    cohort_digest = _cohort_digest(cohort, gold_contract.target_names)
    input_contract_valid = not contract_violations
    smoke_sufficient = input_contract_valid and not unsupported_smoke

    assignments, evaluation_findings = _evaluation_assignment(
        cohort,
        gold_contract.target_names,
        allocation_seed,
        input_contract_valid=input_contract_valid,
    )
    evaluation_decision = "Candidate" if assignments is not None else "Ineligible"
    assignment_rows = _assignment_rows(cohort, assignments)
    assignment_digest = (
        _assignment_digest(assignment_rows, allocation_seed)
        if assignments is not None
        else None
    )
    partition_summary = (
        _partition_summary(cohort, assignments, gold_contract.target_names)
        if assignments is not None
        else {}
    )

    invalid_counts = Counter(item.target for item in invalid_labels)
    exclusion_counts = Counter(item.reason for item in excluded)
    provenance = dict(source_provenance or {})
    summary: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "artifact_type": "rsna_knee_gold_supervision_audit",
        "privacy": {
            "public_safe_summary": True,
            "row_level_membership_committed": False,
            "row_level_labels_committed": False,
            "patient_ids_committed": False,
            "row_level_assignments_committed": False,
            "private_files_remain_in_kaggle_artifact": True,
        },
        "source_contracts": {
            "competition": {
                "schema_version": gold_contract.schema_version,
                "verified_at": gold_contract.verified_at,
                "sha256": gold_contract.sha256,
            },
            "data_index": {
                "schema_version": _required_int(index_contract, "schema_version"),
                "verified_at": _required_text(index_contract, "verified_at"),
                "sha256": index_contract_sha256,
                "accepted_manifest_sha256": _sha256(
                    index_artifact_dir / "index-manifest.json"
                ),
                "accepted_studies_sha256": _sha256(index_artifact_dir / "studies.csv"),
                "verification_scope": [
                    "accepted external manifest digest",
                    "accepted studies.csv digest and size",
                    "studies.csv columns",
                    "competition-contract compatibility",
                ],
            },
        },
        "source_provenance": provenance,
        "label_contract": {
            "status": "verified" if input_contract_valid else "rejected",
            "target_count": len(gold_contract.target_names),
            "target_order": list(gold_contract.target_names),
            "allowed_values": [0, 1, "missing"],
            "source_missing_encoding": "empty CSV field after whitespace trimming",
            "invalid_value_count": len(invalid_labels),
            "invalid_value_counts_by_target": {
                target: invalid_counts.get(target, 0)
                for target in gold_contract.target_names
            },
            "contract_violation_counts": dict(sorted(contract_violations.items())),
        },
        "source_gold_labels": {
            "row_count": len(train_rows),
            "rowwise_completeness": {
                "status": "verified" if input_contract_valid else "rejected",
                "all_targets_known": known_histogram.get(
                    len(gold_contract.target_names), 0
                ),
                "partially_known": sum(
                    count
                    for known, count in known_histogram.items()
                    if 0 < known < len(gold_contract.target_names)
                ),
                "all_targets_missing": known_histogram.get(0, 0),
                "known_target_count_histogram": {
                    str(known): count
                    for known, count in sorted(known_histogram.items())
                },
            },
            "per_target": source_target_counts,
        },
        "canonical_cohort": {
            "selection_rule": (
                "Every organizer-labeled study with at least one nonmissing "
                "verified gold target that resolves through the authoritative "
                "index to one usable study; missing targets remain masked."
            ),
            "selection_rule_version": _COHORT_RULE_VERSION,
            "uses_all_eligible_studies": True,
            "selection_optimizes_class_balance": False,
            "label_contract_version": {
                "schema_version": gold_contract.schema_version,
                "sha256": gold_contract.sha256,
            },
            "cohort_size": len(cohort),
            "excluded_study_count": len(excluded),
            "excluded_study_counts_by_reason": dict(sorted(exclusion_counts.items())),
            "per_target": cohort_counts,
            "all_selected_studies_resolve_to_exactly_one_usable_index_study": True,
            "digest": {
                "algorithm": "sha256",
                "value": cohort_digest,
                "canonicalization_version": _COHORT_DIGEST_VERSION,
                "ordering": "StudyInstanceUID UTF-8 byte order",
                "identifier_representation": "UTF-8 JSON string",
                "label_representation": "ordered JSON array of 0, 1, or null",
                "missing_value_encoding": "JSON null",
                "record_framing": (
                    "UTF-8 newline-delimited canonical JSON with a final newline"
                ),
                "header_record": {
                    "schema": _COHORT_DIGEST_VERSION,
                    "targets": list(gold_contract.target_names),
                },
                "row_record": "[StudyInstanceUID, ordered_label_array]",
                "json_serialization": (
                    "ensure_ascii=false, separators=(',', ':'), sort_keys=true"
                ),
            },
        },
        "smoke_sufficiency": {
            "decision": (
                "smoke-sufficient" if smoke_sufficient else "not-smoke-sufficient"
            ),
            "rule": "Each verified target has at least one gold positive and one gold negative in the canonical cohort.",
            "unsupported_targets": unsupported_smoke,
            "input_contract_valid": input_contract_valid,
            "target_contract_weakened": False,
            "substitute_cohort_used": False,
            "pseudo_label_work_activated": False,
            "bounded_smoke_conflict": {
                "status": "none_identified",
                "basis": (
                    "Issue #8 requires one epoch over this all-eligible cohort and "
                    "declares no smaller cohort-size limit."
                ),
            },
        },
        "evaluation_eligibility": {
            "decision": evaluation_decision,
            "candidate_scope": (
                "A valid gold-only attempt is possible; no claim of statistical "
                "adequacy is made."
            ),
            "allocation_rule_version": _ALLOCATION_RULE_VERSION,
            "seed": allocation_seed,
            "grouping_entity": gold_contract.patient_group_column,
            "prediction_independent": True,
            "uses_predictions": False,
            "uses_leaderboard_results": False,
            "minimum_class_support_per_partition": 1,
            "missing_labels_remain_masked": True,
            "unsupported_findings": evaluation_findings,
            "assignment_digest": (
                {
                    "algorithm": "sha256",
                    "value": assignment_digest,
                    "canonicalization_version": _ASSIGNMENT_DIGEST_VERSION,
                    "ordering": "PatientID then StudyInstanceUID UTF-8 byte order",
                    "identifier_representation": "UTF-8 JSON strings",
                    "partition_values": ["train", "holdout"],
                    "record_framing": (
                        "UTF-8 newline-delimited canonical JSON with a final newline"
                    ),
                    "header_record": {
                        "schema": _ASSIGNMENT_DIGEST_VERSION,
                        "allocation_rule": _ALLOCATION_RULE_VERSION,
                        "seed": allocation_seed,
                    },
                    "row_record": ("[PatientID, StudyInstanceUID, partition]"),
                    "json_serialization": (
                        "ensure_ascii=false, separators=(',', ':'), sort_keys=true"
                    ),
                }
                if assignment_digest is not None
                else None
            ),
            "partitions": partition_summary,
        },
        "report_derived_supervision_recommendation": {
            "status": "informational-only",
            "plausibly_addresses_observed_training_limitation": (
                len(cohort) < usable_train_study_count
            ),
            "observed_limitation": (
                "Organizer-provided gold supervision covers fewer studies than "
                "the usable indexed training corpus."
                if len(cohort) < usable_train_study_count
                else "No organizer-gold study-coverage gap was observed."
            ),
            "usable_indexed_train_studies": usable_train_study_count,
            "organizer_gold_cohort_studies": len(cohort),
            "activated": False,
            "changes_smoke_decision": False,
            "changes_evaluation_decision": False,
            "treats_report_values_as_gold": False,
            "downstream_ticket_created": False,
        },
    }

    _write_cohort(artifact_dir / _COHORT_PRIVATE, cohort, gold_contract.target_names)
    _write_exclusions(artifact_dir / _EXCLUSIONS_PRIVATE, excluded)
    _write_invalid_labels(artifact_dir / _INVALID_LABELS_PRIVATE, invalid_labels)
    _write_assignments(artifact_dir / _ASSIGNMENT_PRIVATE, assignment_rows)
    summary_path = artifact_dir / _SUMMARY
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    file_columns: dict[str, Sequence[str] | None] = {
        _COHORT_PRIVATE: (
            gold_contract.identifier_column,
            gold_contract.patient_group_column,
            *gold_contract.target_names,
        ),
        _EXCLUSIONS_PRIVATE: ("source_row", "StudyInstanceUID", "reason"),
        _INVALID_LABELS_PRIVATE: (
            "source_row",
            "StudyInstanceUID",
            "target",
            "raw_value",
            "category",
        ),
        _ASSIGNMENT_PRIVATE: (
            "StudyInstanceUID",
            "PatientID",
            "partition",
        ),
        _SUMMARY: None,
    }
    manifest_files: dict[str, dict[str, Any]] = {}
    for filename, columns in file_columns.items():
        path = artifact_dir / filename
        record: dict[str, Any] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "visibility": (
                "public_safe" if filename == _SUMMARY else "private_kaggle_only"
            ),
        }
        if columns is not None:
            record["columns"] = list(columns)
        manifest_files[filename] = record
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "artifact_type": "rsna_knee_gold_supervision_audit",
        "contains_private_row_level_data": True,
        "private_artifact_must_not_be_committed": True,
        "index_manifest_schema_version": index_manifest.get("schema_version"),
        "files": manifest_files,
    }
    manifest_path = artifact_dir / _MANIFEST
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verified_summary = verify_gold_audit_artifact(artifact_dir)
    return GoldAuditOutcome(
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        summary_path=summary_path,
        summary=verified_summary,
    )


def verify_gold_audit_artifact(artifact_dir: Path) -> dict[str, Any]:
    """Verify hashes, privacy boundaries, counts, digests, and witness invariants."""
    manifest_path = artifact_dir / _MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing gold audit manifest: {manifest_path}")
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text()))
    if (
        manifest.get("schema_version") != _SCHEMA_VERSION
        or manifest.get("artifact_type") != "rsna_knee_gold_supervision_audit"
        or manifest.get("contains_private_row_level_data") is not True
        or manifest.get("private_artifact_must_not_be_committed") is not True
    ):
        raise ValueError("Gold audit manifest contract is invalid")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, dict):
        raise TypeError("Gold audit manifest has no file inventory")
    expected_files = {
        _COHORT_PRIVATE,
        _EXCLUSIONS_PRIVATE,
        _INVALID_LABELS_PRIVATE,
        _ASSIGNMENT_PRIVATE,
        _SUMMARY,
    }
    if set(raw_files) != expected_files:
        raise ValueError("Gold audit file inventory is incomplete or unexpected")
    expected_columns: dict[str, list[str] | None] = {
        _COHORT_PRIVATE: None,
        _EXCLUSIONS_PRIVATE: ["source_row", "StudyInstanceUID", "reason"],
        _INVALID_LABELS_PRIVATE: [
            "source_row",
            "StudyInstanceUID",
            "target",
            "raw_value",
            "category",
        ],
        _ASSIGNMENT_PRIVATE: ["StudyInstanceUID", "PatientID", "partition"],
        _SUMMARY: None,
    }
    for filename, raw_record in raw_files.items():
        if not isinstance(filename, str) or not isinstance(raw_record, dict):
            raise TypeError("Gold audit file inventory is malformed")
        path = artifact_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing gold audit file: {path}")
        if raw_record.get("sha256") != _sha256(path):
            raise ValueError(f"Gold audit hash mismatch: {filename}")
        if raw_record.get("size_bytes") != path.stat().st_size:
            raise ValueError(f"Gold audit size mismatch: {filename}")
        expected_visibility = (
            "public_safe" if filename == _SUMMARY else "private_kaggle_only"
        )
        if raw_record.get("visibility") != expected_visibility:
            raise ValueError(f"Gold audit visibility mismatch: {filename}")
        columns = raw_record.get("columns")
        if filename != _COHORT_PRIVATE and columns != expected_columns[filename]:
            if filename == _SUMMARY and columns is None:
                continue
            raise ValueError(f"Gold audit columns mismatch: {filename}")

    summary = cast(dict[str, Any], json.loads((artifact_dir / _SUMMARY).read_text()))
    label_contract = _required_mapping(summary, "label_contract")
    raw_targets = label_contract.get("target_order")
    if not isinstance(raw_targets, list) or not all(
        isinstance(target, str) and target for target in raw_targets
    ):
        raise TypeError("Gold audit summary target order is malformed")
    targets = tuple(raw_targets)
    cohort_columns = ("StudyInstanceUID", "PatientID", *targets)
    cohort_manifest = cast(dict[str, Any], raw_files[_COHORT_PRIVATE])
    if cohort_manifest.get("columns") != list(cohort_columns):
        raise ValueError("Gold cohort manifest columns do not match target order")
    cohort_raw = _read_csv(artifact_dir / _COHORT_PRIVATE, cohort_columns)
    cohort = [
        _CohortRow(
            source_row=0,
            study_uid=row["StudyInstanceUID"],
            patient_id=row["PatientID"],
            labels=tuple(_private_label(row[target]) for target in targets),
        )
        for row in cohort_raw
    ]
    if cohort != sorted(cohort, key=lambda row: row.study_uid.encode("utf-8")):
        raise ValueError("Private gold cohort is not canonically ordered")
    canonical = _required_mapping(summary, "canonical_cohort")
    if canonical.get("cohort_size") != len(cohort):
        raise ValueError("Gold cohort size does not match private rows")
    digest = _required_mapping(canonical, "digest")
    if digest.get("value") != _cohort_digest(cohort, targets):
        raise ValueError("Gold cohort membership/label digest mismatch")
    if canonical.get("per_target") != _target_counts(cohort, targets):
        raise ValueError("Gold cohort target counts do not match private labels")

    exclusions = _read_csv(
        artifact_dir / _EXCLUSIONS_PRIVATE,
        ("source_row", "StudyInstanceUID", "reason"),
    )
    invalid = _read_csv(
        artifact_dir / _INVALID_LABELS_PRIVATE,
        ("source_row", "StudyInstanceUID", "target", "raw_value", "category"),
    )
    if canonical.get("excluded_study_count") != len(exclusions):
        raise ValueError("Gold exclusion count does not match private rows")
    if label_contract.get("invalid_value_count") != len(invalid):
        raise ValueError("Invalid-label count does not match private rows")

    assignment_rows = _read_csv(
        artifact_dir / _ASSIGNMENT_PRIVATE,
        ("StudyInstanceUID", "PatientID", "partition"),
    )
    evaluation = _required_mapping(summary, "evaluation_eligibility")
    decision = evaluation.get("decision")
    if decision == "Candidate":
        if len(assignment_rows) != len(cohort):
            raise ValueError("Candidate assignment does not cover the cohort")
        assignment_lookup: dict[str, Partition] = {}
        patient_partitions: dict[str, set[str]] = defaultdict(set)
        cohort_lookup = {row.study_uid: row for row in cohort}
        for row in assignment_rows:
            study_uid = row["StudyInstanceUID"]
            partition = row["partition"]
            if partition not in {"train", "holdout"}:
                raise ValueError("Candidate assignment has an invalid partition")
            if study_uid in assignment_lookup or study_uid not in cohort_lookup:
                raise ValueError("Candidate assignment identifiers do not match cohort")
            if row["PatientID"] != cohort_lookup[study_uid].patient_id:
                raise ValueError("Candidate assignment PatientID mismatch")
            typed_partition = cast(Partition, partition)
            assignment_lookup[study_uid] = typed_partition
            patient_partitions[row["PatientID"]].add(partition)
        if any(len(partitions) != 1 for partitions in patient_partitions.values()):
            raise ValueError("PatientID crosses candidate partitions")
        assignments = {
            row.patient_id: assignment_lookup[row.study_uid] for row in cohort
        }
        if evaluation.get("partitions") != _partition_summary(
            cohort, assignments, targets
        ):
            raise ValueError("Candidate partition counts do not match private rows")
        for partition in ("train", "holdout"):
            partition_counts = _required_mapping(
                _required_mapping(evaluation, "partitions"), partition
            )
            per_target = _required_mapping(partition_counts, "per_target")
            for target in targets:
                counts = _required_mapping(per_target, target)
                if counts.get("positive", 0) < 1 or counts.get("negative", 0) < 1:
                    raise ValueError("Candidate partition lacks both target classes")
        assignment_digest = _required_mapping(evaluation, "assignment_digest")
        seed = _required_int(evaluation, "seed")
        if assignment_digest.get("value") != _assignment_digest(assignment_rows, seed):
            raise ValueError("Candidate assignment digest mismatch")
    elif decision == "Ineligible":
        if assignment_rows or evaluation.get("assignment_digest") is not None:
            raise ValueError("Ineligible audit must not publish an assignment")
        findings = evaluation.get("unsupported_findings")
        if not isinstance(findings, list) or not findings:
            raise ValueError("Ineligible audit must report precise findings")
    else:
        raise ValueError("Unknown evaluation eligibility decision")

    privacy = _required_mapping(summary, "privacy")
    if privacy != {
        "public_safe_summary": True,
        "row_level_membership_committed": False,
        "row_level_labels_committed": False,
        "patient_ids_committed": False,
        "row_level_assignments_committed": False,
        "private_files_remain_in_kaggle_artifact": True,
    }:
        raise ValueError("Gold audit public/private boundary has drifted")
    return summary


def export_public_gold_audit_contract(
    artifact_dir: Path, destination: Path
) -> dict[str, Any]:
    """Export only the verified aggregate/digest summary from a private artifact."""
    summary = verify_gold_audit_artifact(artifact_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _evaluation_assignment(
    cohort: Sequence[_CohortRow],
    targets: Sequence[str],
    seed: int,
    *,
    input_contract_valid: bool,
) -> tuple[dict[str, Partition] | None, list[dict[str, Any]]]:
    findings: list[dict[str, Any]] = []
    if not input_contract_valid:
        findings.append({"code": "input_contract_rejected"})
    missing_patient_count = sum(not row.patient_id for row in cohort)
    if missing_patient_count:
        findings.append(
            {
                "code": "missing_patient_id",
                "affected_study_count": missing_patient_count,
            }
        )

    constraints: list[frozenset[str]] = []
    for target_index, target in enumerate(targets):
        for label, class_name in ((0, "negative"), (1, "positive")):
            patients = frozenset(
                row.patient_id
                for row in cohort
                if row.patient_id and row.labels[target_index] == label
            )
            if len(patients) < 2:
                findings.append(
                    {
                        "code": "insufficient_distinct_patient_groups",
                        "target": target,
                        "class": class_name,
                        "distinct_patient_count": len(patients),
                        "required": 2,
                        "required_partitions": list(_PARTITIONS),
                    }
                )
            constraints.append(patients)
    if findings:
        return None, findings

    patient_ids = sorted({row.patient_id for row in cohort})
    assignment = _solve_two_partition_constraints(patient_ids, constraints, seed)
    if assignment is None:
        return None, [
            {
                "code": "no_patient_disjoint_assignment_satisfies_all_class_constraints",
                "search_complete": True,
                "constraint_count": len(constraints),
                "patient_count": len(patient_ids),
            }
        ]
    return assignment, []


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
                assigned_values = {
                    assignment[patient]
                    for patient in constraint
                    if patient in assignment
                }
                if len(assigned_values) == 2:
                    continue
                unassigned = [
                    patient for patient in constraint if patient not in assignment
                ]
                if not unassigned:
                    return None
                if len(unassigned) == 1 and len(assigned_values) == 1:
                    forced: Partition = (
                        "holdout" if "train" in assigned_values else "train"
                    )
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


def _unsupported_smoke_targets(
    counts: Mapping[str, Mapping[str, int]], targets: Sequence[str]
) -> list[dict[str, Any]]:
    unsupported: list[dict[str, Any]] = []
    for target in targets:
        missing_classes: list[str] = []
        if counts[target]["negative"] < 1:
            missing_classes.append("negative")
        if counts[target]["positive"] < 1:
            missing_classes.append("positive")
        if missing_classes:
            unsupported.append({"target": target, "missing_classes": missing_classes})
    return unsupported


def _target_counts(
    rows: Sequence[_CohortRow], targets: Sequence[str]
) -> dict[str, dict[str, int]]:
    counts = _empty_target_counts(targets)
    for row in rows:
        for target, label in zip(targets, row.labels, strict=True):
            if label is None:
                counts[target]["missing"] += 1
            elif label == 1:
                counts[target]["known"] += 1
                counts[target]["positive"] += 1
            else:
                counts[target]["known"] += 1
                counts[target]["negative"] += 1
    return counts


def _empty_target_counts(targets: Sequence[str]) -> dict[str, dict[str, int]]:
    return {
        target: {"known": 0, "missing": 0, "positive": 0, "negative": 0}
        for target in targets
    }


def _partition_summary(
    cohort: Sequence[_CohortRow],
    assignments: Mapping[str, Partition],
    targets: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for partition in _PARTITIONS:
        rows = [row for row in cohort if assignments[row.patient_id] == partition]
        result[partition] = {
            "study_count": len(rows),
            "patient_count": len({row.patient_id for row in rows}),
            "per_target": _target_counts(rows, targets),
        }
    return result


def _cohort_digest(rows: Sequence[_CohortRow], targets: Sequence[str]) -> str:
    digest = hashlib.sha256()
    header = {"schema": _COHORT_DIGEST_VERSION, "targets": list(targets)}
    digest.update(_canonical_json(header))
    digest.update(b"\n")
    for row in sorted(rows, key=lambda item: item.study_uid.encode("utf-8")):
        digest.update(_canonical_json([row.study_uid, list(row.labels)]))
        digest.update(b"\n")
    return digest.hexdigest()


def _assignment_rows(
    cohort: Sequence[_CohortRow], assignments: Mapping[str, Partition] | None
) -> list[dict[str, str]]:
    if assignments is None:
        return []
    return [
        {
            "StudyInstanceUID": row.study_uid,
            "PatientID": row.patient_id,
            "partition": assignments[row.patient_id],
        }
        for row in sorted(
            cohort,
            key=lambda item: (
                item.patient_id.encode("utf-8"),
                item.study_uid.encode("utf-8"),
            ),
        )
    ]


def _assignment_digest(rows: Sequence[Mapping[str, str]], seed: int) -> str:
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "schema": _ASSIGNMENT_DIGEST_VERSION,
                "allocation_rule": _ALLOCATION_RULE_VERSION,
                "seed": seed,
            }
        )
    )
    digest.update(b"\n")
    for row in sorted(
        rows,
        key=lambda item: (
            item["PatientID"].encode("utf-8"),
            item["StudyInstanceUID"].encode("utf-8"),
        ),
    ):
        digest.update(
            _canonical_json(
                [row["PatientID"], row["StudyInstanceUID"], row["partition"]]
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _parse_label(raw: str) -> tuple[Label, str | None]:
    stripped = raw.strip()
    if not stripped:
        return None, None
    try:
        value = Decimal(stripped)
    except InvalidOperation:
        return None, "not_numeric"
    if not value.is_finite():
        return None, "non_finite"
    if value == 0:
        return 0, None
    if value == 1:
        return 1, None
    return None, "outside_binary_domain"


def _private_label(raw: str) -> Label:
    label, category = _parse_label(raw)
    if category is not None:
        raise ValueError("Private cohort contains an invalid label")
    return label


def _index_row_is_usable(row: Mapping[str, str]) -> bool:
    try:
        usable_series = int(row["usable_series_count"])
        usable_slices = int(row["usable_slice_count"])
    except (KeyError, ValueError) as error:
        raise ValueError(
            "Index studies.csv contains invalid usability counts"
        ) from error
    return usable_series > 0 and usable_slices > 0


def _verify_accepted_index(
    artifact_dir: Path,
    index_contract: Mapping[str, Any],
    gold_contract: VerifiedGoldContract,
) -> dict[str, Any]:
    _validate_data_index_compatibility(index_contract, gold_contract)
    manifest_path = artifact_dir / "index-manifest.json"
    studies_path = artifact_dir / "studies.csv"
    outputs = _required_mapping(
        _required_mapping(index_contract, "run_provenance"), "outputs"
    )
    if _sha256(manifest_path) != _required_text(outputs, "manifest_sha256"):
        raise ValueError("Index manifest does not match the accepted Kaggle output")
    artifact = _required_mapping(index_contract, "artifact")
    accepted_studies = _required_mapping(
        _required_mapping(artifact, "files"), "studies.csv"
    )
    if studies_path.stat().st_size != _required_int(
        accepted_studies, "size_bytes"
    ) or _sha256(studies_path) != _required_text(accepted_studies, "sha256"):
        raise ValueError("studies.csv does not match the accepted Kaggle output")
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text()))
    manifest_files = _required_mapping(manifest, "files")
    manifest_studies = _required_mapping(manifest_files, "studies.csv")
    if (
        manifest_studies.get("columns") != list(_STUDY_COLUMNS)
        or manifest_studies.get("sha256") != _sha256(studies_path)
        or manifest_studies.get("size_bytes") != studies_path.stat().st_size
    ):
        raise ValueError("Index manifest studies.csv contract is invalid")
    if manifest.get("decoded_images_persisted") is not False:
        raise ValueError("Accepted index must not persist decoded images")
    _read_csv(studies_path, _STUDY_COLUMNS)
    return manifest


def _validate_data_index_compatibility(
    index_contract: Mapping[str, Any], gold_contract: VerifiedGoldContract
) -> None:
    current = _required_mapping(index_contract, "current_competition_contract")
    if current.get("sha256") != gold_contract.sha256:
        raise ValueError("Accepted index is not compatible with the gold contract")
    artifact = _required_mapping(index_contract, "artifact")
    if artifact.get("decoded_images_persisted") is not False:
        raise ValueError("Accepted index contract must prohibit decoded images")


def _load_data_index_contract(
    path: Path | None,
) -> tuple[dict[str, Any], str]:
    raw: bytes
    if path is not None:
        raw = path.read_bytes()
    else:
        source_path = (
            Path(__file__).resolve().parents[2] / "contracts" / "data-index.json"
        )
        if source_path.is_file():
            raw = source_path.read_bytes()
        else:
            raw = files("rsna_knee").joinpath("data-index.json").read_bytes()
    contract = cast(dict[str, Any], json.loads(raw))
    if contract.get("schema_version") != 1:
        raise ValueError("Unsupported data-index contract schema version")
    return contract, hashlib.sha256(raw).hexdigest()


def _read_csv(path: Path, expected_columns: Sequence[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(expected_columns):
            raise ValueError(f"Unexpected CSV columns for {path.name}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV row in {path.name}")
    return rows


def _write_cohort(
    path: Path, rows: Iterable[_CohortRow], targets: Sequence[str]
) -> None:
    columns = ["StudyInstanceUID", "PatientID", *targets]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            record: dict[str, object] = {
                "StudyInstanceUID": row.study_uid,
                "PatientID": row.patient_id,
            }
            record.update(
                {
                    target: "" if label is None else label
                    for target, label in zip(targets, row.labels, strict=True)
                }
            )
            writer.writerow(record)


def _write_exclusions(path: Path, rows: Iterable[_ExcludedRow]) -> None:
    columns = ["source_row", "StudyInstanceUID", "reason"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source_row": row.source_row,
                    "StudyInstanceUID": row.study_uid,
                    "reason": row.reason,
                }
            )


def _write_invalid_labels(path: Path, rows: Iterable[_InvalidLabel]) -> None:
    columns = ["source_row", "StudyInstanceUID", "target", "raw_value", "category"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source_row": row.source_row,
                    "StudyInstanceUID": row.study_uid,
                    "target": row.target,
                    "raw_value": row.raw_value,
                    "category": row.category,
                }
            )


def _write_assignments(path: Path, rows: Iterable[Mapping[str, str]]) -> None:
    columns = ["StudyInstanceUID", "PatientID", "partition"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _required_mapping(parent: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Contract field {key} must be an object")
    return cast(dict[str, Any], value)


def _required_text(parent: Mapping[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"Contract field {key} must be a nonempty string")
    return value


def _required_int(parent: Mapping[str, Any], key: str) -> int:
    value = parent.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"Contract field {key} must be an integer")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
