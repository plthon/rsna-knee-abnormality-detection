"""Deterministic, auditable extraction of candidate labels from reports."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from io import StringIO
from pathlib import Path
from typing import Literal, cast

NORMALIZATION_VERSION = "unicode-nfc-horizontal-whitespace-v1"
AUDIT_IMPLEMENTATION_VERSION = "report-corpus-audit-v1"
AUDIT_MIN_DOCUMENT_FREQUENCY = 5

type LabelState = Literal["positive", "negative", "uncertain", "unknown"]
type AssertionType = Literal["affirmed", "negated", "uncertain", "contextual_exclusion"]
type RuleDirection = Literal["forward", "backward", "bidirectional"]

_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")
_SECTION_HEADING = re.compile(
    r"^(?P<heading>[A-Za-z][A-Za-z0-9 /&'_-]{0,49}):(?P<body>.*)$"
)
_SENTENCE = re.compile(r"\S(?:.*?\S)?(?:[.!?]+(?=\s|$)|$)")
_RULE_ID = re.compile(
    r"^[A-Z][A-Z0-9_]*\.(?:CONCEPT|POS|NEG|UNCERTAIN|EXCLUDE|SECTION|CONFLICT)\.\d{3}$"
)
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = re.compile(r"\w+")

_AUDIT_CUES: dict[str, tuple[str, ...]] = {
    "positive": ("present", "demonstrates", "shows", "seen"),
    "explicit_negative": ("no", "not", "without", "absent", "intact"),
    "uncertainty_hedging": ("possible", "possibly", "may", "might", "question of"),
    "historical": ("history of", "prior", "previous", "remote"),
    "postoperative": ("postoperative", "post-operative", "status post"),
    "comparison": ("compared with", "comparison", "unchanged", "interval"),
    "chronicity": ("acute", "chronic", "subacute"),
    "recommendation": ("recommend", "recommended", "correlate", "follow-up"),
}
_TARGET_AUDIT_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "ACL": ("anterior cruciate ligament",),
    "MCL": ("medial collateral ligament",),
    "Medial OA": ("medial osteoarthritis",),
    "Lateral OA": ("lateral osteoarthritis",),
    "PF OA": ("patellofemoral osteoarthritis", "patellofemoral oa"),
    "Baker's": ("baker cyst", "baker's cyst"),
}


@dataclass(frozen=True, slots=True)
class ReportSegment:
    section: str
    text: str
    span_start: int
    span_end: int


@dataclass(frozen=True, slots=True)
class PatternRule:
    id: str
    pattern: str
    direction: RuleDirection = "bidirectional"
    window_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class SectionRule:
    id: str
    sections: tuple[str, ...]
    assertion_type: AssertionType


@dataclass(frozen=True, slots=True)
class TargetRules:
    canonical_name: str
    target_key: str
    concept_patterns: tuple[PatternRule, ...]
    positive_rules: tuple[PatternRule, ...]
    negative_rules: tuple[PatternRule, ...]
    uncertainty_rules: tuple[PatternRule, ...]
    contextual_exclusions: tuple[PatternRule, ...]
    section_context_rules: tuple[SectionRule, ...]
    conflict_rule_id: str
    conflict_policy: str


@dataclass(frozen=True, slots=True)
class RuleContract:
    schema_version: int
    extractor_version: str
    audit_artifact_sha256: str
    target_order: tuple[str, ...]
    normalization_version: str
    section_rules: tuple[SectionRule, ...]
    window_tokens: int
    scope_terminators: tuple[str, ...]
    negation_rules: tuple[PatternRule, ...]
    uncertainty_rules: tuple[PatternRule, ...]
    contextual_exclusions: tuple[PatternRule, ...]
    targets: tuple[TargetRules, ...]


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    provenance_key: str
    study_uid: str
    target: str
    extractor_version: str
    normalized_report_sha256: str
    rule_ids: tuple[str, ...]
    concept_id: str
    assertion_type: AssertionType
    section: str
    span_start: int
    span_end: int
    conflict_rule: str

    def as_dict(self) -> dict[str, object]:
        return {
            "provenance_key": self.provenance_key,
            "StudyInstanceUID": self.study_uid,
            "target": self.target,
            "extractor_version": self.extractor_version,
            "normalized_report_sha256": self.normalized_report_sha256,
            "rule_ids": list(self.rule_ids),
            "concept_id": self.concept_id,
            "assertion_type": self.assertion_type,
            "section": self.section,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "conflict_rule": self.conflict_rule,
        }


@dataclass(frozen=True, slots=True)
class LabelRecord:
    study_uid: str
    target: str
    state: LabelState
    extractor_version: str
    normalized_report_sha256: str
    evidence_count: int
    provenance_key: str

    def as_dict(self) -> dict[str, object]:
        return {
            "StudyInstanceUID": self.study_uid,
            "target": self.target,
            "state": self.state,
            "extractor_version": self.extractor_version,
            "normalized_report_sha256": self.normalized_report_sha256,
            "evidence_count": self.evidence_count,
            "provenance_key": self.provenance_key,
        }


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    labels: tuple[LabelRecord, ...]
    evidence: tuple[EvidenceRecord, ...]


@dataclass(frozen=True, slots=True)
class ReportRow:
    study_uid: str
    report: str | None


def normalize_report(text: str | None) -> str:
    """Return the one canonical report representation used by this extractor."""
    if text is None:
        return ""
    if not isinstance(text, str):
        raise TypeError("Report text must be a string or None")
    normalized = (
        unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    )
    normalized = "\n".join(
        _HORIZONTAL_WHITESPACE.sub(" ", line).strip() for line in normalized.split("\n")
    )
    return normalized.strip()


def normalized_report_sha256(text: str | None) -> str:
    return hashlib.sha256(normalize_report(text).encode("utf-8")).hexdigest()


def segment_report(normalized_report: str) -> tuple[ReportSegment, ...]:
    """Split a normalized report into section-aware sentences with exact offsets."""
    if normalize_report(normalized_report) != normalized_report:
        raise ValueError("segment_report requires normalized report text")
    segments: list[ReportSegment] = []
    section = "unsectioned"
    line_start = 0
    for line in normalized_report.splitlines(keepends=True):
        content = line.removesuffix("\n")
        body = content
        body_start = line_start
        heading = _SECTION_HEADING.fullmatch(content)
        if heading is not None:
            section = _HORIZONTAL_WHITESPACE.sub(
                " ", heading.group("heading")
            ).casefold()
            raw_body = heading.group("body")
            body = raw_body.strip()
            body_start = (
                line_start
                + heading.start("body")
                + len(raw_body)
                - len(raw_body.lstrip())
            )
        for match in _SENTENCE.finditer(body):
            text = match.group().strip()
            if not text:
                continue
            start = (
                body_start
                + match.start()
                + len(match.group())
                - len(match.group().lstrip())
            )
            segments.append(
                ReportSegment(
                    section=section,
                    text=text,
                    span_start=start,
                    span_end=start + len(text),
                )
            )
        line_start += len(line)
    return tuple(segments)


def load_rule_contract(
    source: Path | Mapping[str, object],
    *,
    expected_target_order: Sequence[str],
) -> RuleContract:
    """Load a frozen rule inventory and reject malformed or drifting rules."""
    if isinstance(source, Path):
        raw = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("Rule contract must be a JSON object")
        data = cast(dict[str, object], raw)
    else:
        data = dict(source)
    schema_version = _required_int(data, "schema_version")
    if schema_version != 1:
        raise ValueError(f"Unsupported rule contract schema version: {schema_version}")
    extractor_version = _required_text(data, "extractor_version")
    audit_sha256 = _required_text(data, "audit_artifact_sha256")
    if _HEX_SHA256.fullmatch(audit_sha256) is None:
        raise ValueError("audit_artifact_sha256 must be a lowercase SHA-256")
    target_order = _text_tuple(data, "target_order")
    if target_order != tuple(expected_target_order):
        raise ValueError("Rule contract target order differs from the verified order")
    normalization_version = _required_text(data, "normalization_version")
    if normalization_version != NORMALIZATION_VERSION:
        raise ValueError("Rule contract normalization version is unsupported")
    section_rules = _section_rules(data.get("section_rules"), "GLOBAL")
    assertion = _required_mapping(data, "assertion_scope_rules")
    window_tokens = _required_int(assertion, "window_tokens")
    if window_tokens < 0:
        raise ValueError("Assertion window_tokens must be nonnegative")
    scope_terminators = _text_tuple(assertion, "scope_terminators")
    for pattern in scope_terminators:
        _compile_pattern(pattern)
    negation_rules = _pattern_rules(assertion.get("negation"), "GLOBAL", "NEG")
    uncertainty_rules = _pattern_rules(
        assertion.get("uncertainty"), "GLOBAL", "UNCERTAIN"
    )
    exclusions = _pattern_rules(
        assertion.get("contextual_exclusions"), "GLOBAL", "EXCLUDE"
    )

    raw_targets = data.get("targets")
    if not isinstance(raw_targets, list):
        raise TypeError("Rule contract targets must be an array")
    targets = tuple(_target_rules(item) for item in raw_targets)
    if tuple(target.canonical_name for target in targets) != target_order:
        raise ValueError("Rule target entries differ from target_order")
    all_ids = [rule_id for target in targets for rule_id in _target_rule_ids(target)]
    all_ids.extend(rule.id for rule in section_rules)
    all_ids.extend(rule.id for rule in negation_rules)
    all_ids.extend(rule.id for rule in uncertainty_rules)
    all_ids.extend(rule.id for rule in exclusions)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Rule IDs must be globally unique")
    return RuleContract(
        schema_version=schema_version,
        extractor_version=extractor_version,
        audit_artifact_sha256=audit_sha256,
        target_order=target_order,
        normalization_version=normalization_version,
        section_rules=section_rules,
        window_tokens=window_tokens,
        scope_terminators=scope_terminators,
        negation_rules=negation_rules,
        uncertainty_rules=uncertainty_rules,
        contextual_exclusions=exclusions,
        targets=targets,
    )


def provenance_key(
    study_uid: str,
    target: str,
    extractor_version: str,
    report_sha256: str,
) -> str:
    """Hash a compact JSON array, which gives each UTF-8 field an unambiguous edge."""
    serialized = json.dumps(
        [study_uid, target, extractor_version, report_sha256],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def extract_report(
    study_uid: str,
    report: str | None,
    contract: RuleContract,
) -> ExtractionResult:
    """Extract all ordered target states and auditable evidence for one report."""
    if not isinstance(study_uid, str) or not study_uid:
        raise ValueError("StudyInstanceUID must be a nonempty string")
    normalized = normalize_report(report)
    report_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    segments = segment_report(normalized)
    labels: list[LabelRecord] = []
    all_evidence: list[EvidenceRecord] = []
    for target in contract.targets:
        key = provenance_key(
            study_uid, target.canonical_name, contract.extractor_version, report_hash
        )
        evidence = _target_evidence(
            study_uid,
            key,
            report_hash,
            segments,
            target,
            contract,
        )
        state, conflict_rule = _aggregate_evidence(evidence, target)
        if conflict_rule:
            evidence = tuple(
                replace(item, conflict_rule=conflict_rule) for item in evidence
            )
        labels.append(
            LabelRecord(
                study_uid=study_uid,
                target=target.canonical_name,
                state=state,
                extractor_version=contract.extractor_version,
                normalized_report_sha256=report_hash,
                evidence_count=sum(
                    item.assertion_type != "contextual_exclusion" for item in evidence
                ),
                provenance_key=key,
            )
        )
        all_evidence.extend(evidence)
    return ExtractionResult(labels=tuple(labels), evidence=tuple(all_evidence))


def extract_corpus(
    rows: Sequence[ReportRow],
    contract: RuleContract,
    *,
    expected_study_count: int | None = None,
) -> ExtractionResult:
    """Extract deterministic long-form rows in source-study and target order."""
    if expected_study_count is not None and len(rows) != expected_study_count:
        raise ValueError(
            f"Expected {expected_study_count} report rows, received {len(rows)}"
        )
    study_uids = [row.study_uid for row in rows]
    if len(study_uids) != len(set(study_uids)):
        raise ValueError("Duplicate StudyInstanceUID in report corpus")
    labels: list[LabelRecord] = []
    evidence: list[EvidenceRecord] = []
    normalized_lengths: dict[str, int] = {}
    for row in rows:
        normalized_lengths[row.study_uid] = len(normalize_report(row.report))
        extraction = extract_report(row.study_uid, row.report, contract)
        labels.extend(extraction.labels)
        evidence.extend(extraction.evidence)
    outcome = ExtractionResult(labels=tuple(labels), evidence=tuple(evidence))
    _validate_corpus_extraction(outcome, rows, contract, normalized_lengths)
    return outcome


def read_report_csv(
    path: Path,
    *,
    requested_columns: Sequence[str],
    target_names: Sequence[str],
) -> tuple[ReportRow, ...]:
    """Read only report identity and text, rejecting organizer-gold requests."""
    requested = tuple(requested_columns)
    if set(requested) & set(target_names):
        raise PermissionError(
            "The report audit must not request organizer-gold target columns"
        )
    if requested != ("StudyInstanceUID", "Report"):
        raise ValueError("Report input columns must be StudyInstanceUID and Report")
    rows: list[ReportRow] = []
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.reader(source)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("Report CSV is empty") from error
        if len(header) != len(set(header)):
            raise ValueError("Report CSV header contains duplicate columns")
        missing = [column for column in requested if column not in header]
        if missing:
            raise ValueError(f"Report CSV is missing required columns: {missing}")
        study_index = header.index("StudyInstanceUID")
        report_index = header.index("Report")
        for source_row, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f"Malformed report CSV row {source_row}")
            rows.append(ReportRow(row[study_index].strip(), row[report_index]))
    return tuple(rows)


def audit_report_corpus(
    rows: Sequence[ReportRow],
    target_names: Sequence[str],
    *,
    expected_study_count: int | None = None,
    source_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Return privacy-thresholded full-corpus structure and lexical statistics."""
    if expected_study_count is not None and len(rows) != expected_study_count:
        raise ValueError(f"Expected {expected_study_count} rows, received {len(rows)}")
    study_uids = [row.study_uid for row in rows]
    if any(not uid for uid in study_uids):
        raise ValueError("Every report row requires a StudyInstanceUID")
    if len(study_uids) != len(set(study_uids)):
        raise ValueError("Duplicate StudyInstanceUID in report corpus")
    normalized = [normalize_report(row.report) for row in rows]
    ngram_frequencies: dict[int, Counter[str]] = {
        1: Counter(),
        2: Counter(),
        3: Counter(),
    }
    heading_frequencies: Counter[str] = Counter()
    heading_count_histogram: Counter[int] = Counter()
    for report in normalized:
        tokens = [token.casefold() for token in _TOKEN.findall(report)]
        for size, frequencies in ngram_frequencies.items():
            frequencies.update(
                {
                    " ".join(tokens[index : index + size])
                    for index in range(len(tokens) - size + 1)
                }
            )
        headings = {
            match.group("heading").casefold()
            for line in report.splitlines()
            if (match := _SECTION_HEADING.fullmatch(line)) is not None
        }
        heading_frequencies.update(headings)
        heading_count_histogram[len(headings)] += 1

    lengths = [len(report) for report in normalized]
    target_terminology: dict[str, object] = {}
    near_misses: dict[str, object] = {}
    for target in target_names:
        seeds = (target.casefold(), *_TARGET_AUDIT_EXPANSIONS.get(target, ()))
        seed_records = _phrase_records(
            Counter({seed: _document_frequency(normalized, seed) for seed in seeds})
        )
        neighbors = _target_neighbor_frequencies(normalized, seeds)
        for seed in seeds:
            neighbors.pop(
                " ".join(token.casefold() for token in _TOKEN.findall(seed)), None
            )
        neighbor_records = _phrase_records(neighbors, limit=100)
        target_terminology[target] = {
            "seed_terms": seed_records,
            "corpus_observed_neighbors": neighbor_records,
        }
        near_misses[target] = [
            {**record, "status": "unresolved"} for record in neighbor_records
        ]

    cue_candidates = {
        category: _phrase_records(
            Counter({cue: _document_frequency(normalized, cue) for cue in candidates})
        )
        for category, candidates in _AUDIT_CUES.items()
    }
    selected_input_sha256 = hashlib.sha256(
        "".join(
            json.dumps(
                [row.study_uid, report], ensure_ascii=False, separators=(",", ":")
            )
            + "\n"
            for row, report in zip(rows, normalized, strict=True)
        ).encode("utf-8")
    ).hexdigest()
    audit: dict[str, object] = {
        "schema_version": 1,
        "audit_implementation_version": AUDIT_IMPLEMENTATION_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "privacy": {
            "minimum_lexical_document_frequency": AUDIT_MIN_DOCUMENT_FREQUENCY,
            "aggregate_only": True,
            "raw_report_text_exported": False,
            "row_identifiers_exported": False,
            "concordance_exported": False,
        },
        "source_input_hashes": {
            **dict(source_hashes or {}),
            "normalized_selected_columns_sha256": selected_input_sha256,
        },
        "corpus": {
            "total_rows": len(rows),
            "unique_study_identifiers": len(set(study_uids)),
            "empty_normalized_reports": sum(not report for report in normalized),
            "nonempty_normalized_reports": sum(bool(report) for report in normalized),
            "normalized_length_characters": _length_distribution(lengths),
        },
        "section_heading_counts": _phrase_records(heading_frequencies),
        "structural_patterns": {
            "reports_without_detected_headings": heading_count_histogram[0],
            "reports_with_detected_headings": len(rows) - heading_count_histogram[0],
            "detected_heading_count_per_report": {
                str(count): frequency
                for count, frequency in sorted(heading_count_histogram.items())
            },
        },
        "script_family_counts": _script_family_counts(normalized),
        "language": "unresolved",
        "document_frequency": {
            "unigrams": _phrase_records(ngram_frequencies[1], limit=500),
            "bigrams": _phrase_records(ngram_frequencies[2], limit=500),
            "trigrams": _phrase_records(ngram_frequencies[3], limit=500),
        },
        "target_oriented_candidate_terminology": target_terminology,
        "cue_candidates": cue_candidates,
        "potentially_contradictory_assertion_patterns": (
            _target_contradiction_statistics(normalized, target_names)
        ),
        "near_miss_ambiguity_candidates": near_misses,
    }
    _validate_audit_privacy(audit)
    return audit


def write_extraction_artifacts(
    rows: Sequence[ReportRow],
    contract: RuleContract,
    output_root: Path,
    *,
    expected_study_count: int = 4407,
    competition_contract_sha256: str,
    rule_contract_sha256: str,
    source_provenance: Mapping[str, object],
    runtime_seconds: float | None = None,
) -> dict[str, Path]:
    """Write the fixed private extraction artifacts after a repeatability check."""
    started = time.monotonic()
    _validate_sha256("competition_contract_sha256", competition_contract_sha256)
    _validate_sha256("rule_contract_sha256", rule_contract_sha256)
    if runtime_seconds is not None and (
        isinstance(runtime_seconds, bool) or runtime_seconds < 0
    ):
        raise ValueError("runtime_seconds must be nonnegative")
    source = _validated_source_provenance(source_provenance)
    paths = {
        "labels": output_root / "report-labels-v1.csv",
        "evidence": output_root / "report-evidence-v1.jsonl",
        "manifest": output_root / "report-extraction-manifest.json",
        "summary": output_root / "report-extraction-summary.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Extraction artifacts already exist: {existing}")

    first = extract_corpus(rows, contract, expected_study_count=expected_study_count)
    second = extract_corpus(rows, contract, expected_study_count=expected_study_count)
    if first != second:
        raise RuntimeError("Repeated in-memory report extraction differs")
    label_bytes = _label_csv_bytes(first.labels)
    evidence_bytes = _evidence_jsonl_bytes(first.evidence)
    if label_bytes != _label_csv_bytes(
        second.labels
    ) or evidence_bytes != _evidence_jsonl_bytes(second.evidence):
        raise RuntimeError("Repeated serialized extraction differs")

    output_root.mkdir(parents=True, exist_ok=True)
    paths["labels"].write_bytes(label_bytes)
    paths["evidence"].write_bytes(evidence_bytes)
    state_counts, coverage_rates = _state_counts(first.labels, contract.target_order)
    rule_use_counts = Counter(
        rule_id
        for item in first.evidence
        for rule_id in (
            *item.rule_ids,
            *([item.conflict_rule] if item.conflict_rule else []),
        )
    )
    conflict_counts = {
        target: len(
            {
                item.provenance_key
                for item in first.evidence
                if item.target == target and item.conflict_rule
            }
        )
        for target in contract.target_order
    }
    exclusion_counts = Counter(
        item.target
        for item in first.evidence
        if item.assertion_type == "contextual_exclusion"
    )
    private_artifacts = {
        paths["labels"].name: _artifact_record(paths["labels"]),
        paths["evidence"].name: _artifact_record(paths["evidence"]),
    }
    summary: dict[str, object] = {
        "schema_version": 1,
        "extractor_version": contract.extractor_version,
        "normalization_version": contract.normalization_version,
        "audit_artifact_sha256": contract.audit_artifact_sha256,
        "rule_contract_sha256": rule_contract_sha256,
        "per_target_state_counts": state_counts,
        "per_target_coverage_rates": coverage_rates,
        "rule_use_counts": dict(sorted(rule_use_counts.items())),
        "conflict_ambiguity_counts": conflict_counts,
        "aggregate_failure_modes": {
            "contextual_exclusion_evidence_counts": {
                target: exclusion_counts[target] for target in contract.target_order
            },
            "malformed_reports": 0,
            "extraction_errors": 0,
        },
        "artifact_identities": private_artifacts,
        "exact_label_row_count_complete": len(first.labels)
        == expected_study_count * len(contract.target_order),
        "validation_readiness": {
            "status": "technically_ready_for_issue_21_manual_validation",
            "label_correctness_established": False,
        },
    }
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    expected_label_count = expected_study_count * len(contract.target_order)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "extractor_version": contract.extractor_version,
        "normalization_version": contract.normalization_version,
        "git": {
            "commit": source["git_commit"],
            "dirty": source["git_dirty"],
        },
        "notebook": {
            "source_sha256": source["notebook_source_sha256"],
            "rendered_sha256": source["rendered_notebook_sha256"],
            "rendered_sha256_canonicalization": source[
                "rendered_notebook_sha256_canonicalization"
            ],
        },
        "package_archive_sha256": source["package_archive_sha256"],
        "competition_contract_sha256": competition_contract_sha256,
        "rule_contract_sha256": rule_contract_sha256,
        "audit_artifact_sha256": contract.audit_artifact_sha256,
        "input_competition_identity": {
            "slug": "rsna-knee-abnormality-detection",
            "mount": "/kaggle/input/competitions/rsna-knee-abnormality-detection",
            "table": "train.csv",
            "columns_read": ["StudyInstanceUID", "Report"],
        },
        "normalized_report_count": len(rows),
        "expected_label_row_count": expected_label_count,
        "actual_label_row_count": len(first.labels),
        "evidence_row_count": len(first.evidence),
        "target_order": list(contract.target_order),
        "per_target_state_counts": state_counts,
        "per_target_coverage_rates": coverage_rates,
        "artifacts": {
            **private_artifacts,
            paths["summary"].name: _artifact_record(paths["summary"]),
        },
        "runtime_seconds": (
            runtime_seconds
            if runtime_seconds is not None
            else time.monotonic() - started
        ),
        "runtime_policy": {
            "cpu_only": True,
            "internet_enabled": False,
            "gpu_enabled": False,
            "tpu_enabled": False,
        },
        "malformed_report_or_extraction_errors": [],
        "deterministic_equality_verified": True,
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def _label_csv_bytes(labels: Sequence[LabelRecord]) -> bytes:
    columns = (
        "StudyInstanceUID",
        "target",
        "state",
        "extractor_version",
        "normalized_report_sha256",
        "evidence_count",
        "provenance_key",
    )
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(label.as_dict() for label in labels)
    return output.getvalue().encode("utf-8")


def _evidence_jsonl_bytes(evidence: Sequence[EvidenceRecord]) -> bytes:
    return "".join(
        json.dumps(
            item.as_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        + "\n"
        for item in evidence
    ).encode("utf-8")


def _state_counts(
    labels: Sequence[LabelRecord], target_order: Sequence[str]
) -> tuple[dict[str, dict[str, int]], dict[str, float]]:
    states = ("positive", "negative", "uncertain", "unknown")
    counts = {
        target: {
            state: sum(
                label.target == target and label.state == state for label in labels
            )
            for state in states
        }
        for target in target_order
    }
    study_count = len(labels) // len(target_order)
    coverage = {
        target: (
            (study_count - target_counts["unknown"]) / study_count
            if study_count
            else 0.0
        )
        for target, target_counts in counts.items()
    }
    return counts, coverage


def _artifact_record(path: Path) -> dict[str, object]:
    return {
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _validated_source_provenance(
    source: Mapping[str, object],
) -> dict[str, object]:
    required_hashes = (
        "notebook_source_sha256",
        "rendered_notebook_sha256",
        "package_archive_sha256",
    )
    for key in required_hashes:
        value = source.get(key)
        if not isinstance(value, str):
            raise TypeError(f"Source provenance {key} must be a string")
        _validate_sha256(key, value)
    git_commit = source.get("git_commit")
    git_dirty = source.get("git_dirty")
    canonicalization = source.get("rendered_notebook_sha256_canonicalization")
    if not isinstance(git_commit, str) or not git_commit:
        raise TypeError("Source provenance git_commit must be nonempty")
    if not isinstance(git_dirty, bool):
        raise TypeError("Source provenance git_dirty must be boolean")
    if not isinstance(canonicalization, str) or not canonicalization:
        raise TypeError("Rendered notebook hash canonicalization must be nonempty")
    return {
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "notebook_source_sha256": source["notebook_source_sha256"],
        "rendered_notebook_sha256": source["rendered_notebook_sha256"],
        "rendered_notebook_sha256_canonicalization": canonicalization,
        "package_archive_sha256": source["package_archive_sha256"],
    }


def _validate_sha256(name: str, value: str) -> None:
    if _HEX_SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _phrase_records(
    frequencies: Mapping[str, int], *, limit: int | None = None
) -> list[dict[str, object]]:
    records = [
        {"phrase": phrase, "document_frequency": frequency}
        for phrase, frequency in frequencies.items()
        if frequency >= AUDIT_MIN_DOCUMENT_FREQUENCY
    ]
    records.sort(
        key=lambda item: (
            -cast(int, item["document_frequency"]),
            cast(str, item["phrase"]),
        )
    )
    return records if limit is None else records[:limit]


def _document_frequency(reports: Sequence[str], phrase: str) -> int:
    return sum(_contains_phrase(report, phrase) for report in reports)


def _contains_phrase(report: str, phrase: str) -> bool:
    pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
    return re.search(pattern, report, flags=re.IGNORECASE | re.UNICODE) is not None


def _target_neighbor_frequencies(
    reports: Sequence[str], seeds: Sequence[str]
) -> Counter[str]:
    frequencies: Counter[str] = Counter()
    seed_tokens = [
        tuple(token.casefold() for token in _TOKEN.findall(seed)) for seed in seeds
    ]
    for report in reports:
        report_candidates: set[str] = set()
        for segment in segment_report(report):
            tokens = [token.casefold() for token in _TOKEN.findall(segment.text)]
            for seed in seed_tokens:
                for index in range(len(tokens) - len(seed) + 1):
                    if tuple(tokens[index : index + len(seed)]) != seed:
                        continue
                    for size in (2, 3):
                        for start in range(index - size + 1, index + len(seed)):
                            end = start + size
                            if start >= 0 and end <= len(tokens):
                                report_candidates.add(" ".join(tokens[start:end]))
        frequencies.update(report_candidates)
    return frequencies


def _target_contradiction_statistics(
    reports: Sequence[str], target_names: Sequence[str]
) -> dict[str, object]:
    result: dict[str, object] = {}
    for target in target_names:
        seeds = (target.casefold(), *_TARGET_AUDIT_EXPANSIONS.get(target, ()))
        pair_frequencies: Counter[tuple[str, str]] = Counter()
        contradiction_reports = 0
        for report in reports:
            positive: set[str] = set()
            negative: set[str] = set()
            for segment in segment_report(report):
                if not any(_contains_phrase(segment.text, seed) for seed in seeds):
                    continue
                positive.update(
                    cue
                    for cue in _AUDIT_CUES["positive"]
                    if _contains_phrase(segment.text, cue)
                )
                negative.update(
                    cue
                    for cue in _AUDIT_CUES["explicit_negative"]
                    if _contains_phrase(segment.text, cue)
                )
            if positive and negative:
                contradiction_reports += 1
                pair_frequencies.update(
                    {
                        (positive_cue, negative_cue)
                        for positive_cue in positive
                        for negative_cue in negative
                    }
                )
        patterns = [
            {
                "positive_cue": positive_cue,
                "negative_cue": negative_cue,
                "document_frequency": frequency,
            }
            for (positive_cue, negative_cue), frequency in pair_frequencies.items()
            if frequency >= AUDIT_MIN_DOCUMENT_FREQUENCY
        ]
        patterns.sort(
            key=lambda item: (
                -cast(int, item["document_frequency"]),
                cast(str, item["positive_cue"]),
                cast(str, item["negative_cue"]),
            )
        )
        result[target] = {
            "reports_with_potential_contradiction": contradiction_reports,
            "patterns": patterns,
        }
    return result


def _length_distribution(lengths: Sequence[int]) -> dict[str, object]:
    if not lengths:
        return {
            "minimum": 0,
            "maximum": 0,
            "mean": 0.0,
            "p25": 0,
            "p50": 0,
            "p75": 0,
            "p95": 0,
        }
    ordered = sorted(lengths)
    return {
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 6),
        "p25": _nearest_rank(ordered, 0.25),
        "p50": _nearest_rank(ordered, 0.50),
        "p75": _nearest_rank(ordered, 0.75),
        "p95": _nearest_rank(ordered, 0.95),
    }


def _nearest_rank(ordered: Sequence[int], quantile: float) -> int:
    return ordered[max(0, math.ceil(len(ordered) * quantile) - 1)]


def _script_family_counts(reports: Sequence[str]) -> dict[str, object]:
    character_counts: Counter[str] = Counter()
    report_counts: Counter[str] = Counter()
    for report in reports:
        families: set[str] = set()
        for character in report:
            if character.isspace():
                continue
            family = _script_family(character)
            character_counts[family] += 1
            families.add(family)
        report_counts.update(families)
    return {
        family: {
            "character_count": character_counts[family],
            "report_count": report_counts[family],
        }
        for family in sorted(character_counts)
    }


def _script_family(character: str) -> str:
    name = unicodedata.name(character, "")
    for family in (
        "LATIN",
        "GREEK",
        "CYRILLIC",
        "ARABIC",
        "HEBREW",
        "DEVANAGARI",
        "HANGUL",
        "HIRAGANA",
        "KATAKANA",
    ):
        if family in name:
            return family.title()
    if "CJK UNIFIED" in name or "IDEOGRAPH" in name:
        return "Han"
    if character.isdigit() or unicodedata.category(character)[0] in {"P", "S"}:
        return "Common"
    return "Other"


def _validate_audit_privacy(audit: Mapping[str, object]) -> None:
    serialized = json.dumps(audit, ensure_ascii=False, sort_keys=True).casefold()
    for forbidden in (
        "studyinstanceuid",
        "patientid",
        "raw report sentence",
        "context window",
        "row number",
    ):
        if forbidden in serialized:
            raise ValueError(
                f"Audit output contains prohibited private field: {forbidden}"
            )
    for record in _walk_lexical_records(audit):
        frequency = record.get("document_frequency")
        if (
            isinstance(frequency, bool)
            or not isinstance(frequency, int)
            or frequency < AUDIT_MIN_DOCUMENT_FREQUENCY
        ):
            raise ValueError("Audit lexical output violates the privacy threshold")


def _walk_lexical_records(value: object) -> list[Mapping[str, object]]:
    if isinstance(value, dict):
        mapping = cast(dict[str, object], value)
        lexical_keys = {"phrase", "positive_cue", "negative_cue"}
        found: list[Mapping[str, object]] = (
            [mapping] if lexical_keys & mapping.keys() else []
        )
        for child in mapping.values():
            found.extend(_walk_lexical_records(child))
        return found
    if isinstance(value, list):
        return [record for child in value for record in _walk_lexical_records(child)]
    return []


def _validate_corpus_extraction(
    extraction: ExtractionResult,
    rows: Sequence[ReportRow],
    contract: RuleContract,
    normalized_lengths: Mapping[str, int],
) -> None:
    expected_pairs = [
        (row.study_uid, target) for row in rows for target in contract.target_order
    ]
    actual_pairs = [(row.study_uid, row.target) for row in extraction.labels]
    if actual_pairs != expected_pairs:
        raise ValueError("Long-form labels do not preserve study and target order")
    if len(actual_pairs) != len(set(actual_pairs)):
        raise ValueError("Duplicate study-target label pair")
    valid_states = {"positive", "negative", "uncertain", "unknown"}
    for label in extraction.labels:
        if label.state not in valid_states:
            raise ValueError(f"Invalid extraction state: {label.state}")
        if label.state == "unknown" and label.evidence_count != 0:
            raise ValueError("Unknown labels must have evidence_count = 0")
        if label.state != "unknown" and label.evidence_count < 1:
            raise ValueError("Non-unknown labels require auditable evidence")
    known_rule_ids = {
        rule_id for target in contract.targets for rule_id in _target_rule_ids(target)
    }
    known_rule_ids.update(rule.id for rule in contract.section_rules)
    known_rule_ids.update(rule.id for rule in contract.negation_rules)
    known_rule_ids.update(rule.id for rule in contract.uncertainty_rules)
    known_rule_ids.update(rule.id for rule in contract.contextual_exclusions)
    for item in extraction.evidence:
        if (
            not 0
            <= item.span_start
            < item.span_end
            <= normalized_lengths[item.study_uid]
        ):
            raise ValueError("Evidence span is outside the normalized report")
        if not set(item.rule_ids).issubset(known_rule_ids):
            raise ValueError("Evidence references an unknown rule ID")


def _target_evidence(
    study_uid: str,
    key: str,
    report_hash: str,
    segments: tuple[ReportSegment, ...],
    target: TargetRules,
    contract: RuleContract,
) -> tuple[EvidenceRecord, ...]:
    evidence: list[EvidenceRecord] = []
    seen_mentions: set[tuple[int, int]] = set()
    for segment in segments:
        for concept in target.concept_patterns:
            for match in _compile_pattern(concept.pattern).finditer(segment.text):
                span = (
                    segment.span_start + match.start(),
                    segment.span_start + match.end(),
                )
                if span in seen_mentions:
                    continue
                seen_mentions.add(span)
                assertion, assertion_rule_ids = _evaluate_assertion(
                    segment,
                    match.start(),
                    match.end(),
                    target,
                    contract,
                )
                evidence.append(
                    EvidenceRecord(
                        provenance_key=key,
                        study_uid=study_uid,
                        target=target.canonical_name,
                        extractor_version=contract.extractor_version,
                        normalized_report_sha256=report_hash,
                        rule_ids=(concept.id, *assertion_rule_ids),
                        concept_id=concept.id,
                        assertion_type=assertion,
                        section=segment.section,
                        span_start=span[0],
                        span_end=span[1],
                        conflict_rule="",
                    )
                )
    return tuple(
        sorted(
            evidence, key=lambda item: (item.span_start, item.span_end, item.rule_ids)
        )
    )


def _evaluate_assertion(
    segment: ReportSegment,
    mention_start: int,
    mention_end: int,
    target: TargetRules,
    contract: RuleContract,
) -> tuple[AssertionType, tuple[str, ...]]:
    section_matches = tuple(
        rule
        for rule in (*target.section_context_rules, *contract.section_rules)
        if segment.section in rule.sections
    )
    if section_matches:
        assertions = {rule.assertion_type for rule in section_matches}
        assertion: AssertionType = (
            next(iter(assertions)) if len(assertions) == 1 else "uncertain"
        )
        return assertion, tuple(sorted(rule.id for rule in section_matches))

    target_groups: tuple[tuple[AssertionType, tuple[PatternRule, ...]], ...] = (
        ("contextual_exclusion", target.contextual_exclusions),
        ("negated", target.negative_rules),
        ("uncertain", target.uncertainty_rules),
        ("affirmed", target.positive_rules),
    )
    target_matches = _matching_assertions(
        segment,
        mention_start,
        mention_end,
        target_groups,
        contract,
    )
    if target_matches:
        return _resolve_mention_assertions(target_matches)

    generic_groups: tuple[tuple[AssertionType, tuple[PatternRule, ...]], ...] = (
        ("contextual_exclusion", contract.contextual_exclusions),
        ("negated", contract.negation_rules),
        ("uncertain", contract.uncertainty_rules),
    )
    generic_matches = _matching_assertions(
        segment,
        mention_start,
        mention_end,
        generic_groups,
        contract,
    )
    if generic_matches:
        return _resolve_mention_assertions(generic_matches)
    return "affirmed", ()


def _matching_assertions(
    segment: ReportSegment,
    mention_start: int,
    mention_end: int,
    groups: tuple[tuple[AssertionType, tuple[PatternRule, ...]], ...],
    contract: RuleContract,
) -> list[tuple[AssertionType, str]]:
    return [
        (assertion, rule.id)
        for assertion, rules in groups
        for rule in rules
        if _rule_is_in_scope(
            rule,
            segment.text,
            mention_start,
            mention_end,
            contract.window_tokens,
            contract.scope_terminators,
        )
    ]


def _resolve_mention_assertions(
    matches: list[tuple[AssertionType, str]],
) -> tuple[AssertionType, tuple[str, ...]]:
    assertions = {assertion for assertion, _ in matches}
    if "contextual_exclusion" in assertions:
        assertion: AssertionType = "contextual_exclusion"
    elif len(assertions) > 1:
        assertion = "uncertain"
    else:
        assertion = next(iter(assertions))
    return assertion, tuple(sorted(rule_id for _, rule_id in matches))


def _rule_is_in_scope(
    rule: PatternRule,
    text: str,
    mention_start: int,
    mention_end: int,
    default_window: int,
    terminators: tuple[str, ...],
) -> bool:
    window = default_window if rule.window_tokens is None else rule.window_tokens
    for match in _compile_pattern(rule.pattern).finditer(text):
        if match.start() <= mention_start and match.end() >= mention_end:
            return True
        if (
            rule.direction in {"forward", "bidirectional"}
            and match.end() <= mention_start
        ):
            between = text[match.end() : mention_start]
            if _token_distance(between) <= window and not _has_terminator(
                between, terminators
            ):
                return True
        if (
            rule.direction in {"backward", "bidirectional"}
            and match.start() >= mention_end
        ):
            between = text[mention_end : match.start()]
            if _token_distance(between) <= window and not _has_terminator(
                between, terminators
            ):
                return True
    return False


def _aggregate_evidence(
    evidence: tuple[EvidenceRecord, ...], target: TargetRules
) -> tuple[LabelState, str]:
    assertions = {
        item.assertion_type
        for item in evidence
        if item.assertion_type != "contextual_exclusion"
    }
    if not assertions:
        return "unknown", ""
    if assertions == {"affirmed"}:
        return "positive", ""
    if assertions == {"negated"}:
        return "negative", ""
    if assertions == {"uncertain"}:
        return "uncertain", ""
    if target.conflict_policy != "unresolved_to_uncertain":
        raise ValueError(f"Unsupported conflict policy: {target.conflict_policy}")
    return "uncertain", target.conflict_rule_id


def _target_rules(value: object) -> TargetRules:
    if not isinstance(value, dict):
        raise TypeError("Each rule target must be an object")
    data = cast(dict[str, object], value)
    name = _required_text(data, "canonical_name")
    target_key = _required_text(data, "target_key")
    conflict = _required_mapping(data, "conflict_policy")
    conflict_id = _required_rule_id(conflict, target_key, "CONFLICT")
    conflict_policy = _required_text(conflict, "policy")
    if conflict_policy != "unresolved_to_uncertain":
        raise ValueError("V1 conflict policy must be unresolved_to_uncertain")
    concepts = _pattern_rules(data.get("concept_patterns"), target_key, "CONCEPT")
    if not concepts:
        raise ValueError(f"Target {name} must define at least one concept pattern")
    return TargetRules(
        canonical_name=name,
        target_key=target_key,
        concept_patterns=concepts,
        positive_rules=_pattern_rules(data.get("positive_rules"), target_key, "POS"),
        negative_rules=_pattern_rules(data.get("negative_rules"), target_key, "NEG"),
        uncertainty_rules=_pattern_rules(
            data.get("uncertainty_rules"), target_key, "UNCERTAIN"
        ),
        contextual_exclusions=_pattern_rules(
            data.get("contextual_exclusions"), target_key, "EXCLUDE"
        ),
        section_context_rules=_section_rules(
            data.get("section_context_rules"), target_key
        ),
        conflict_rule_id=conflict_id,
        conflict_policy=conflict_policy,
    )


def _pattern_rules(
    value: object, expected_prefix: str, expected_class: str
) -> tuple[PatternRule, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{expected_class} rules must be an array")
    rules: list[PatternRule] = []
    for item in value:
        if not isinstance(item, dict):
            raise TypeError("Pattern rule must be an object")
        data = cast(dict[str, object], item)
        rule_id = _required_rule_id(data, expected_prefix, expected_class)
        pattern = _required_text(data, "pattern")
        _compile_pattern(pattern)
        direction_value = data.get("direction", "bidirectional")
        if direction_value == "forward":
            direction: RuleDirection = "forward"
        elif direction_value == "backward":
            direction = "backward"
        elif direction_value == "bidirectional":
            direction = "bidirectional"
        else:
            raise ValueError(f"Invalid rule direction for {rule_id}")
        window_value = data.get("window_tokens")
        if window_value is not None and (
            isinstance(window_value, bool)
            or not isinstance(window_value, int)
            or window_value < 0
        ):
            raise TypeError(f"window_tokens for {rule_id} must be nonnegative")
        rules.append(
            PatternRule(
                id=rule_id,
                pattern=pattern,
                direction=direction,
                window_tokens=window_value,
            )
        )
    return tuple(rules)


def _section_rules(value: object, expected_prefix: str) -> tuple[SectionRule, ...]:
    if not isinstance(value, list):
        raise TypeError("Section rules must be an array")
    rules: list[SectionRule] = []
    for item in value:
        if not isinstance(item, dict):
            raise TypeError("Section rule must be an object")
        data = cast(dict[str, object], item)
        rule_id = _required_rule_id(data, expected_prefix, "SECTION")
        sections = tuple(
            section.casefold() for section in _text_tuple(data, "sections")
        )
        assertion_value = data.get("assertion_type")
        if assertion_value not in {
            "affirmed",
            "negated",
            "uncertain",
            "contextual_exclusion",
        }:
            raise ValueError(f"Invalid section assertion for {rule_id}")
        rules.append(
            SectionRule(
                id=rule_id,
                sections=sections,
                assertion_type=assertion_value,
            )
        )
    return tuple(rules)


def _target_rule_ids(target: TargetRules) -> tuple[str, ...]:
    return (
        *(rule.id for rule in target.concept_patterns),
        *(rule.id for rule in target.positive_rules),
        *(rule.id for rule in target.negative_rules),
        *(rule.id for rule in target.uncertainty_rules),
        *(rule.id for rule in target.contextual_exclusions),
        *(rule.id for rule in target.section_context_rules),
        target.conflict_rule_id,
    )


def _required_mapping(parent: Mapping[str, object], key: str) -> dict[str, object]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Rule contract field {key} must be an object")
    return cast(dict[str, object], value)


def _required_text(parent: Mapping[str, object], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"Rule contract field {key} must be a nonempty string")
    return value


def _required_int(parent: Mapping[str, object], key: str) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Rule contract field {key} must be an integer")
    return value


def _text_tuple(parent: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = parent.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise TypeError(f"Rule contract field {key} must be nonempty strings")
    return tuple(value)


def _required_rule_id(
    parent: Mapping[str, object], expected_prefix: str, expected_class: str
) -> str:
    rule_id = _required_text(parent, "id")
    if _RULE_ID.fullmatch(rule_id) is None or not rule_id.startswith(
        f"{expected_prefix}.{expected_class}."
    ):
        raise ValueError(
            f"Rule ID must match {expected_prefix}.{expected_class}.NNN: {rule_id}"
        )
    return rule_id


def _compile_pattern(pattern: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern, flags=re.IGNORECASE | re.UNICODE)
    except re.error as error:
        raise ValueError(f"Invalid rule regex: {pattern}") from error


def _token_distance(text: str) -> int:
    return len(_TOKEN.findall(text))


def _has_terminator(text: str, terminators: tuple[str, ...]) -> bool:
    return any(
        _compile_pattern(pattern).search(text) is not None for pattern in terminators
    )
