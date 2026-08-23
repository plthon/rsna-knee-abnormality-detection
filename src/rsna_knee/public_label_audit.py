"""Aggregate-only validation for the frozen public report-label dataset."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

type Decision = Literal[
    "usable_directly", "usable_with_restrictions", "insufficiently_documented"
]
type Provenance = Literal["documented", "partial", "opaque"]

_DECISIONS = {
    "usable_directly",
    "usable_with_restrictions",
    "insufficiently_documented",
}
_PROVENANCE_REQUIREMENTS = {
    "model_family",
    "model_version",
    "prompt_method",
    "output_semantics",
    "language_handling",
}
_PRIVATE_KEY_FRAGMENTS = (
    "studyinstanceuid",
    "study_ids",
    "study_uids",
    "raw_report",
    "report_text",
    "normalized_report_sha256",
    "provenance_key",
    "evidence_span",
)


@dataclass(frozen=True, slots=True)
class LabelAudit:
    row_count: int
    unique_study_count: int
    missing_identifier_count: int
    duplicate_identifier_count: int
    target_order_matches: bool
    per_target: dict[str, dict[str, object]]
    errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "layout": "wide",
            "row_count": self.row_count,
            "unique_study_count": self.unique_study_count,
            "missing_identifier_count": self.missing_identifier_count,
            "duplicate_identifier_count": self.duplicate_identifier_count,
            "target_order_matches": self.target_order_matches,
            "per_target": self.per_target,
            "errors": list(self.errors),
        }


def resolve_dataset_root(candidates: Sequence[Path], marker: str) -> Path:
    """Resolve one hash-checkable attachment from Kaggle's known mount layouts."""
    matches = {root.resolve() for root in candidates if (root / marker).is_file()}
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one attached dataset containing {marker}; found {len(matches)}"
        )
    return next(iter(matches))


def validate_wide_labels(
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[str],
    *,
    identifier_column: str,
    target_order: Sequence[str],
    value_kind: str,
    auxiliary_domains: Mapping[str, Mapping[str, object]] | None = None,
) -> LabelAudit:
    """Validate one-row-per-study labels without retaining study identifiers."""
    if value_kind not in {"binary", "probability", "bounded_score"}:
        raise ValueError("value_kind must be binary, probability, or bounded_score")
    auxiliary = auxiliary_domains or {}
    errors: list[str] = []

    def add_error(error: str) -> None:
        if error not in errors:
            errors.append(error)

    expected_columns = [identifier_column]
    for target in target_order:
        expected_columns.append(target)
        expected_columns.extend(f"{target}{suffix}" for suffix in auxiliary)
    if list(columns) != expected_columns:
        add_error("schema_columns")

    target_set = set(target_order)
    present_targets = tuple(column for column in columns if column in target_set)
    if set(present_targets) != target_set:
        add_error("missing_targets")
    target_order_matches = present_targets == tuple(target_order)
    if not target_order_matches:
        add_error("target_order")

    seen_identifiers: set[str] = set()
    missing_identifier_count = 0
    duplicate_identifier_count = 0
    for row in rows:
        identifier = row.get(identifier_column)
        normalized = "" if _is_missing(identifier) else str(identifier).strip()
        if not normalized:
            missing_identifier_count += 1
        elif normalized in seen_identifiers:
            duplicate_identifier_count += 1
        else:
            seen_identifiers.add(normalized)
    if missing_identifier_count:
        add_error("missing_identifiers")
    if duplicate_identifier_count:
        add_error("duplicate_identifiers")

    per_target: dict[str, dict[str, object]] = {}
    any_invalid = False
    any_missing = False
    for target in target_order:
        numeric_values: list[float] = []
        invalid_value_count = 0
        missing_value_count = 0
        invalid_auxiliary_value_count = 0
        verdict_counts: Counter[str] = Counter()
        for row in rows:
            raw_value = row.get(target)
            if _is_missing(raw_value):
                missing_value_count += 1
            else:
                value = _number(raw_value)
                if value is None or not _valid_number(value, value_kind):
                    invalid_value_count += 1
                else:
                    numeric_values.append(value)
            for suffix, domain in auxiliary.items():
                raw_auxiliary = row.get(f"{target}{suffix}")
                if not _valid_domain_value(raw_auxiliary, domain):
                    invalid_auxiliary_value_count += 1
                elif suffix == "__verdict":
                    verdict_counts[str(raw_auxiliary)] += 1

        if invalid_value_count or invalid_auxiliary_value_count:
            any_invalid = True
        if missing_value_count:
            any_missing = True
        denominator = len(rows)
        if verdict_counts:
            positive_rate = verdict_counts["YES"] / denominator if denominator else None
        elif value_kind == "binary":
            positive_rate = sum(numeric_values) / denominator if denominator else None
        else:
            positive_rate = None
        per_target[target] = {
            "nonmissing_value_count": len(numeric_values),
            "missing_value_count": missing_value_count,
            "invalid_value_count": invalid_value_count,
            "invalid_auxiliary_value_count": invalid_auxiliary_value_count,
            "minimum": min(numeric_values) if numeric_values else None,
            "maximum": max(numeric_values) if numeric_values else None,
            "unique_values": sorted(set(numeric_values)),
            "verdict_counts": dict(sorted(verdict_counts.items())),
            "positive_rate": positive_rate,
        }
    if any_missing:
        add_error("missing_target_values")
    if any_invalid:
        add_error("invalid_values")

    return LabelAudit(
        row_count=len(rows),
        unique_study_count=len(seen_identifiers),
        missing_identifier_count=missing_identifier_count,
        duplicate_identifier_count=duplicate_identifier_count,
        target_order_matches=target_order_matches,
        per_target=per_target,
        errors=tuple(errors),
    )


def audit_file_inventory(
    dataset_root: Path, expected: Sequence[Mapping[str, object]]
) -> tuple[tuple[dict[str, object], ...], tuple[str, ...]]:
    """Hash the exact expected files and report presence or identity drift."""
    expected_names = {str(record["name"]) for record in expected}
    actual_names = {path.name for path in dataset_root.iterdir() if path.is_file()}
    errors: list[str] = []
    if actual_names != expected_names:
        errors.append("file_presence")
    inventory: list[dict[str, object]] = []
    identity_error = False
    for expected_record in expected:
        name = str(expected_record["name"])
        path = dataset_root / name
        if not path.is_file():
            continue
        record: dict[str, object] = {
            "name": name,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        inventory.append(record)
        if record["size_bytes"] != expected_record.get("size_bytes") or record[
            "sha256"
        ] != expected_record.get("sha256"):
            identity_error = True
    if identity_error:
        errors.append("file_identity")
    return tuple(inventory), tuple(errors)


def summarize_coverage(
    public_study_ids: Collection[str],
    competition_rows: Sequence[Mapping[str, object]],
    *,
    identifier_column: str,
    target_order: Sequence[str],
) -> dict[str, int]:
    """Compare study membership while observing only target missingness."""
    competition_ids: set[str] = set()
    gold_ids: set[str] = set()
    non_gold_ids: set[str] = set()
    partially_labeled_ids: set[str] = set()
    for row in competition_rows:
        identifier = str(row[identifier_column])
        competition_ids.add(identifier)
        present = [not _is_missing(row.get(target)) for target in target_order]
        if all(present):
            gold_ids.add(identifier)
        elif any(present):
            partially_labeled_ids.add(identifier)
        else:
            non_gold_ids.add(identifier)
    public_ids = set(public_study_ids)
    return {
        "competition_study_count": len(competition_ids),
        "gold_study_count": len(gold_ids),
        "non_gold_study_count": len(non_gold_ids),
        "partially_labeled_study_count": len(partially_labeled_ids),
        "covered_study_count": len(public_ids),
        "competition_overlap_count": len(public_ids & competition_ids),
        "gold_overlap_count": len(public_ids & gold_ids),
        "non_gold_overlap_count": len(public_ids & non_gold_ids),
        "missing_competition_study_count": len(competition_ids - public_ids),
        "extra_public_study_count": len(public_ids - competition_ids),
    }


def compare_label_sources(
    public_rows: Sequence[Mapping[str, object]],
    report_rows: Sequence[Mapping[str, object]],
    *,
    identifier_column: str,
    target_order: Sequence[str],
) -> dict[str, dict[str, object]]:
    """Aggregate agreement without claiming either report-derived source is correct."""
    public_by_study = {
        str(row[identifier_column]): row
        for row in public_rows
        if not _is_missing(row.get(identifier_column))
    }
    counters = {target: Counter[str]() for target in target_order}
    for report_row in report_rows:
        target = str(report_row.get("target", ""))
        if target not in counters:
            continue
        identifier = str(report_row.get(identifier_column, ""))
        public_row = public_by_study.get(identifier)
        if public_row is None:
            continue
        report_state = str(report_row.get("state", ""))
        public_state = _public_state(public_row, target)
        counter = counters[target]
        counter["overlap"] += 1
        counter[f"report_{report_state}"] += 1
        counter[f"public_{public_state}"] += 1
        if report_state == "uncertain":
            counter["excluded_report_uncertain"] += 1
            continue
        if report_state == "unknown":
            counter["excluded_report_unknown"] += 1
            continue
        if report_state not in {"positive", "negative"}:
            counter["invalid_report_state"] += 1
            continue
        counter["decisive_report"] += 1
        if public_state not in {"positive", "negative"}:
            counter["excluded_public_nondecisive"] += 1
            continue
        counter["compared"] += 1
        if public_state == report_state:
            counter["agreements"] += 1
        elif public_state == "positive":
            counter["public_positive_report_negative"] += 1
        else:
            counter["public_negative_report_positive"] += 1

    comparison: dict[str, dict[str, object]] = {}
    for target in target_order:
        counter = counters[target]
        overlap = counter["overlap"]
        compared = counter["compared"]
        comparison[target] = {
            "overlap_row_count": overlap,
            "public_positive_rate": (
                counter["public_positive"] / overlap if overlap else None
            ),
            "report_state_counts": {
                state: counter[f"report_{state}"]
                for state in ("positive", "negative", "uncertain", "unknown")
            },
            "decisive_report_overlap_count": counter["decisive_report"],
            "compared_decisive_row_count": compared,
            "agreement_rate": counter["agreements"] / compared if compared else None,
            "public_positive_report_negative_conflicts": counter[
                "public_positive_report_negative"
            ],
            "public_negative_report_positive_conflicts": counter[
                "public_negative_report_positive"
            ],
            "excluded_report_uncertain": counter["excluded_report_uncertain"],
            "excluded_report_unknown": counter["excluded_report_unknown"],
            "excluded_public_nondecisive": counter["excluded_public_nondecisive"],
            "invalid_report_state_count": counter["invalid_report_state"],
        }
    return comparison


def classify_provenance(evidence: Mapping[str, bool]) -> Provenance:
    documented = {name for name, present in evidence.items() if present}
    if not documented:
        return "opaque"
    if _PROVENANCE_REQUIREMENTS <= documented:
        return "documented"
    return "partial"


def decide_usability(**facts: object) -> Decision:
    """Apply issue #24's three-way decision rule."""
    provenance = facts.get("provenance")
    if provenance not in {"documented", "partial", "opaque"}:
        raise ValueError("Unknown provenance classification")
    blocking = (
        not _required_bool(facts, "identity_frozen")
        or not _required_bool(facts, "structurally_valid")
        or not _required_bool(facts, "contract_compatible")
        or not _required_bool(facts, "semantics_unambiguous")
        or provenance == "opaque"
    )
    if blocking:
        return "insufficiently_documented"
    restrictions = facts.get("restrictions", ())
    if not isinstance(restrictions, (tuple, list)):
        raise TypeError("restrictions must be a sequence")
    if (
        provenance == "partial"
        or not _required_bool(facts, "coverage_complete")
        or bool(restrictions)
    ):
        return "usable_with_restrictions"
    return "usable_directly"


def public_safe_summary(
    *,
    dataset_identity: Mapping[str, object],
    schema_summary: Mapping[str, object],
    coverage_summary: Mapping[str, object],
    provenance: str,
    comparison_summary: Mapping[str, object],
    decision: str,
    justification: Sequence[str],
) -> dict[str, object]:
    """Build the allowlisted aggregate output and reject row-level fields."""
    if decision not in _DECISIONS:
        raise ValueError("Unknown public-label decision")
    summary: dict[str, object] = {
        "schema_version": 1,
        "dataset_identity": dict(dataset_identity),
        "schema_summary": dict(schema_summary),
        "coverage_summary": dict(coverage_summary),
        "provenance_classification": provenance,
        "comparison_summary": dict(comparison_summary),
        "final_decision": decision,
        "justification": list(justification),
    }
    _reject_private_keys(summary)
    return summary


def _public_state(row: Mapping[str, object], target: str) -> str:
    verdict = row.get(f"{target}__verdict")
    if not _is_missing(verdict):
        return {"YES": "positive", "NO": "negative", "UNK": "unknown"}.get(
            str(verdict), "invalid"
        )
    value = _number(row.get(target))
    if value == 1:
        return "positive"
    if value == 0:
        return "negative"
    return "unknown"


def _valid_domain_value(value: object, domain: Mapping[str, object]) -> bool:
    kind = domain.get("kind")
    if kind in {"probability", "binary", "bounded_score"}:
        number = _number(value)
        return number is not None and _valid_number(number, kind)
    if kind == "enum":
        values = domain.get("values")
        return isinstance(values, list) and value in values
    raise ValueError(f"Unsupported value domain: {kind}")


def _number(value: object) -> float | None:
    try:
        return float(cast(str | int | float, value))
    except (TypeError, ValueError):
        return None


def _valid_number(value: float, kind: str) -> bool:
    if not math.isfinite(value):
        return False
    if kind == "binary":
        return value in {0.0, 1.0}
    return 0.0 <= value <= 1.0


def _is_missing(value: object) -> bool:
    return (
        value is None
        or (isinstance(value, str) and not value.strip())
        or (isinstance(value, float) and math.isnan(value))
    )


def _required_bool(facts: Mapping[str, object], name: str) -> bool:
    value = facts.get(name)
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _reject_private_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            folded = str(key).casefold()
            if any(fragment in folded for fragment in _PRIVATE_KEY_FRAGMENTS):
                raise ValueError("Public summary contains a private row-level field")
            _reject_private_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_private_keys(item)


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()
