"""Frozen local-LLM extraction of candidate labels from knee MRI reports."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from io import StringIO
from pathlib import Path
from typing import Literal, cast

from rsna_knee.reports import (
    NORMALIZATION_VERSION,
    ReportRow,
    normalize_report,
    normalized_report_sha256,
)

type LLMLabelState = Literal["positive", "negative", "uncertain", "unknown"]

LABEL_STATES: tuple[LLMLabelState, ...] = (
    "positive",
    "negative",
    "uncertain",
    "unknown",
)
EXTRACTOR_VERSION = "report-llm-qwen2.5-3b-v1"
PROMPT_VERSION = "rsna-knee-12target-fourstate-v1"
MODEL_REPOSITORY = "Qwen/Qwen2.5-3B-Instruct"

_TARGET_DEFINITIONS = (
    (
        "ACL",
        (
            "ACL tear, sprain, or rupture, partial or complete; reconstruction or "
            "graft language alone is not a current injury label."
        ),
    ),
    ("MCL", "MCL tear or sprain, partial or complete."),
    (
        "Medial Meniscus",
        (
            "tear of the medial meniscus; degeneration without tear is not "
            "automatically positive."
        ),
    ),
    (
        "Lateral Meniscus",
        (
            "tear of the lateral meniscus; degeneration without tear is not "
            "automatically positive."
        ),
    ),
    (
        "Medial OA",
        (
            "osteoarthritic or degenerative cartilage or joint-space disease localized "
            "to the medial tibiofemoral compartment, including clearly degenerative "
            "compartment-specific cartilage loss, chondrosis, or osteophytes."
        ),
    ),
    (
        "Lateral OA",
        (
            "osteoarthritic or degenerative cartilage or joint-space disease localized "
            "to the lateral tibiofemoral compartment, including clearly degenerative "
            "compartment-specific cartilage loss, chondrosis, or osteophytes."
        ),
    ),
    (
        "PF OA",
        (
            "osteoarthritic or degenerative cartilage disease localized to the "
            "patellofemoral compartment."
        ),
    ),
    (
        "Effusion",
        (
            "joint effusion or fluid, including trace or small amounts when explicitly "
            "present."
        ),
    ),
    (
        "Synovitis",
        (
            "synovitis, synovial inflammation, or synovial thickening; do not broaden "
            "to unrelated impingement or plica findings."
        ),
    ),
    ("Baker's", "Baker or popliteal cyst."),
    (
        "Contusion",
        (
            "acute traumatic bone contusion or bone bruise; do not count ordinary "
            "chronic degenerative subchondral marrow edema."
        ),
    ),
    (
        "Fracture",
        (
            "fracture, including acute, avulsion, insufficiency, or stress fracture "
            "when current report semantics support it."
        ),
    ),
)

_TARGET_TEXT = "\n".join(
    f"{index}. {target} — {definition}"
    for index, (target, definition) in enumerate(_TARGET_DEFINITIONS, 1)
)
_OUTPUT_EXAMPLE = json.dumps(
    {target: "unknown" for target, _ in _TARGET_DEFINITIONS},
    ensure_ascii=False,
    separators=(",", ":"),
)
SYSTEM_PROMPT = f"""You extract structured findings from knee MRI radiology reports.
Read the report in its original language. Return only one JSON object with exactly
the 12 keys below and one of: positive, negative, uncertain, unknown.

State semantics:
- positive: current report affirmatively establishes the target abnormality.
- negative: current report explicitly negates it or clearly establishes a normal or intact target state.
- uncertain: it is possible, equivocal, indeterminate, or current usable statements remain contradictory.
- unknown: the report has no usable current evidence sufficient for another state.

A target not mentioned is unknown, never negative. A historical or prior finding
alone is not current positive. Postoperative or reconstruction language alone is
not current positive. Preserve compartment specificity. Do not provide reasoning,
confidence, explanations, markdown, or keys other than the required schema.

Targets in required order:
{_TARGET_TEXT}

Required JSON shape:
{_OUTPUT_EXAMPLE}"""


@dataclass(frozen=True, slots=True)
class ParsedGeneration:
    states: dict[str, LLMLabelState]
    parse_failed: bool


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    study_uid: str
    normalized_report: str
    normalized_report_sha256: str
    messages: tuple[dict[str, str], ...]


@dataclass(frozen=True, slots=True)
class GenerationResponse:
    text: str
    input_truncated: bool


@dataclass(frozen=True, slots=True)
class LLMStudyResult:
    study_uid: str
    states: dict[str, LLMLabelState]
    normalized_report_sha256: str
    parse_failed: bool
    retry_count: int
    input_truncated: bool
    checkpoint_sequence: int


@dataclass(frozen=True, slots=True)
class LLMLabelRecord:
    study_uid: str
    target: str
    state: LLMLabelState
    extractor_version: str
    model_revision: str
    prompt_version: str
    normalized_report_sha256: str
    parse_failed: bool
    input_truncated: bool
    provenance_key: str

    def as_dict(self) -> dict[str, object]:
        return {
            "StudyInstanceUID": self.study_uid,
            "target": self.target,
            "state": self.state,
            "extractor_version": self.extractor_version,
            "model_revision": self.model_revision,
            "prompt_version": self.prompt_version,
            "normalized_report_sha256": self.normalized_report_sha256,
            "parse_failed": str(self.parse_failed).lower(),
            "input_truncated": str(self.input_truncated).lower(),
            "provenance_key": self.provenance_key,
        }


@dataclass(frozen=True, slots=True)
class RuleLabelRecord:
    study_uid: str
    target: str
    state: LLMLabelState


type BatchGenerator = Callable[
    [tuple[GenerationRequest, ...]], tuple[GenerationResponse, ...]
]


@dataclass(frozen=True, slots=True)
class LLMContract:
    schema_version: int
    extractor_version: str
    model_repository: str
    model_revision: str
    tokenizer_revision: str
    model_license: Mapping[str, object]
    model_asset_identity: Mapping[str, object]
    model_file_hashes: Mapping[str, object]
    target_order: tuple[str, ...]
    prompt_version: str
    system_prompt_sha256: str
    output_schema: Mapping[str, object]
    decoding: Mapping[str, object]
    input_token_policy: Mapping[str, object]
    parse_retry_policy: Mapping[str, object]
    checkpoint_policy: Mapping[str, object]
    normalization_version: str
    runtime: Mapping[str, object]


def parse_generated_output(text: str, target_order: Sequence[str]) -> ParsedGeneration:
    """Strictly parse one bare JSON object or return an explicit unknown failure."""
    unknown: dict[str, LLMLabelState] = {target: "unknown" for target in target_order}
    try:
        raw = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        return ParsedGeneration(unknown, True)
    if not isinstance(raw, dict) or set(raw) != set(target_order):
        return ParsedGeneration(unknown, True)
    if any(
        not isinstance(raw[target], str) or raw[target] not in LABEL_STATES
        for target in target_order
    ):
        return ParsedGeneration(unknown, True)
    states = {target: raw[target] for target in target_order}
    return ParsedGeneration(states, False)


def build_generation_request(row: ReportRow) -> GenerationRequest:
    """Build the one frozen prompt while retaining #20's report identity."""
    if not row.study_uid:
        raise ValueError("StudyInstanceUID must be nonempty")
    normalized = normalize_report(row.report)
    return GenerationRequest(
        study_uid=row.study_uid,
        normalized_report=normalized,
        normalized_report_sha256=normalized_report_sha256(row.report),
        messages=(
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f'Report:\n"""\n{normalized}\n"""\n\n'
                    "Return the required JSON object."
                ),
            },
        ),
    )


def llm_provenance_key(
    study_uid: str,
    target: str,
    extractor_version: str,
    model_revision: str,
    prompt_version: str,
    report_sha256: str,
) -> str:
    """Hash a compact JSON array so all six UTF-8 field boundaries are explicit."""
    serialized = json.dumps(
        [
            study_uid,
            target,
            extractor_version,
            model_revision,
            prompt_version,
            report_sha256,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_label_records(
    rows: Sequence[ReportRow],
    results: Sequence[LLMStudyResult],
    contract: LLMContract,
    *,
    expected_study_count: int = 4407,
) -> tuple[LLMLabelRecord, ...]:
    """Return exact long-form rows in source-study and frozen target order."""
    expected_label_count = expected_study_count * len(contract.target_order)
    if len(rows) != expected_study_count or len(results) != expected_study_count:
        raise ValueError(
            f"Expected {expected_study_count} studies and {expected_label_count} labels"
        )
    results_by_uid = {result.study_uid: result for result in results}
    if len(results_by_uid) != len(results):
        raise ValueError("Duplicate StudyInstanceUID in LLM results")
    if set(results_by_uid) != {row.study_uid for row in rows}:
        raise ValueError("LLM results do not match the requested study corpus")
    labels: list[LLMLabelRecord] = []
    for row in rows:
        result = results_by_uid[row.study_uid]
        if result.normalized_report_sha256 != normalized_report_sha256(row.report):
            raise ValueError("LLM result report identity differs from source")
        for target in contract.target_order:
            labels.append(
                LLMLabelRecord(
                    study_uid=row.study_uid,
                    target=target,
                    state=result.states[target],
                    extractor_version=contract.extractor_version,
                    model_revision=contract.model_revision,
                    prompt_version=contract.prompt_version,
                    normalized_report_sha256=result.normalized_report_sha256,
                    parse_failed=result.parse_failed,
                    input_truncated=result.input_truncated,
                    provenance_key=llm_provenance_key(
                        row.study_uid,
                        target,
                        contract.extractor_version,
                        contract.model_revision,
                        contract.prompt_version,
                        result.normalized_report_sha256,
                    ),
                )
            )
    if len(labels) != expected_label_count:
        raise ValueError(f"Expected exactly {expected_label_count} LLM label rows")
    return tuple(labels)


def serialize_label_csv(labels: Sequence[LLMLabelRecord]) -> str:
    fields = (
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
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(label.as_dict() for label in labels)
    return output.getvalue()


def serialize_study_status_csv(
    rows: Sequence[ReportRow],
    results: Sequence[LLMStudyResult],
    contract: LLMContract,
) -> str:
    fields = (
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
    results_by_uid = {result.study_uid: result for result in results}
    if len(results_by_uid) != len(results) or set(results_by_uid) != {
        row.study_uid for row in rows
    }:
        raise ValueError("LLM study status does not match the requested corpus")
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        result = results_by_uid[row.study_uid]
        writer.writerow(
            {
                "StudyInstanceUID": row.study_uid,
                "extractor_version": contract.extractor_version,
                "model_revision": contract.model_revision,
                "prompt_version": contract.prompt_version,
                "normalized_report_sha256": result.normalized_report_sha256,
                "parse_failed": str(result.parse_failed).lower(),
                "retry_count": result.retry_count,
                "input_truncated": str(result.input_truncated).lower(),
                "checkpoint_sequence": result.checkpoint_sequence,
            }
        )
    return output.getvalue()


def read_non_gold_study_uids(path: Path, target_names: Sequence[str]) -> frozenset[str]:
    """Read only target missingness, never organizer-gold values."""
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.reader(source)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("Report CSV is empty") from error
        if len(header) != len(set(header)):
            raise ValueError("Report CSV header contains duplicate columns")
        required = ("StudyInstanceUID", *target_names)
        missing = [column for column in required if column not in header]
        if missing:
            raise ValueError(f"Report CSV is missing required columns: {missing}")
        uid_index = header.index("StudyInstanceUID")
        target_indices = tuple(header.index(target) for target in target_names)
        non_gold: set[str] = set()
        seen: set[str] = set()
        for source_row, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f"Malformed report CSV row {source_row}")
            uid = row[uid_index].strip()
            if not uid or uid in seen:
                raise ValueError("StudyInstanceUID missing or duplicated in report CSV")
            seen.add(uid)
            if all(not row[index].strip() for index in target_indices):
                non_gold.add(uid)
    return frozenset(non_gold)


def select_hashed_reports(
    rows: Sequence[ReportRow],
    eligible_study_uids: frozenset[str],
    *,
    salt: str,
    count: int,
) -> tuple[ReportRow, ...]:
    """Select a fixed privacy-safe sample by salted StudyInstanceUID hash."""
    by_uid = {row.study_uid: row for row in rows}
    if len(by_uid) != len(rows):
        raise ValueError("Duplicate StudyInstanceUID in report corpus")
    if not eligible_study_uids <= set(by_uid):
        raise ValueError("Eligible study set differs from the report corpus")
    if count < 1 or len(eligible_study_uids) < count:
        raise ValueError("Insufficient eligible reports for deterministic sample")
    ordered = sorted(
        eligible_study_uids,
        key=lambda uid: hashlib.sha256(f"{salt}|{uid}".encode()).hexdigest(),
    )
    return tuple(by_uid[uid] for uid in ordered[:count])


def truncate_token_ids(
    token_ids: Sequence[int], *, max_tokens: int
) -> tuple[tuple[int, ...], bool]:
    """Apply the frozen token-level beginning/end truncation policy."""
    if max_tokens < 2:
        raise ValueError("Token budget must preserve both report boundaries")
    tokens = tuple(token_ids)
    if len(tokens) <= max_tokens:
        return tokens, False
    beginning = (max_tokens + 1) // 2
    end = max_tokens - beginning
    return (*tokens[:beginning], *tokens[-end:]), True


def verify_model_asset(model_root: Path, contract: LLMContract) -> dict[str, object]:
    """Verify every frozen model/tokenizer byte before model construction."""
    verified: dict[str, object] = {}
    for filename, raw_record in contract.model_file_hashes.items():
        if Path(filename).name != filename or not isinstance(raw_record, dict):
            raise ValueError("Frozen model asset inventory is malformed")
        record = cast(dict[str, object], raw_record)
        expected_size = _required_int(record, "size_bytes")
        expected_sha256 = _required_sha(record, "sha256")
        path = model_root / filename
        if (
            not path.is_file()
            or path.stat().st_size != expected_size
            or _file_sha256(path) != expected_sha256
        ):
            raise ValueError(f"Frozen model asset verification failed for {filename}")
        verified[filename] = {
            "size_bytes": expected_size,
            "sha256": expected_sha256,
        }
    return verified


def read_rule_label_csv(
    path: Path,
    *,
    expected_sha256: str,
    target_order: Sequence[str],
    expected_study_count: int = 4407,
) -> tuple[RuleLabelRecord, ...]:
    """Read only #20's exact private state rows after artifact verification."""
    if _file_sha256(path) != expected_sha256:
        raise ValueError("Rule-label SHA-256 differs from frozen #20 evidence")
    expected_fields = (
        "StudyInstanceUID",
        "target",
        "state",
        "extractor_version",
        "normalized_report_sha256",
        "evidence_count",
        "provenance_key",
    )
    labels: list[RuleLabelRecord] = []
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise ValueError("Rule-label columns differ from frozen #20 schema")
        for row in reader:
            state = row["state"]
            if state not in LABEL_STATES:
                raise ValueError("Rule-label state differs from four-state vocabulary")
            labels.append(
                RuleLabelRecord(
                    study_uid=row["StudyInstanceUID"],
                    target=row["target"],
                    state=cast(LLMLabelState, state),
                )
            )
    expected_count = expected_study_count * len(target_order)
    if len(labels) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} frozen rule-label rows")
    expected_targets = tuple(target_order)
    for start in range(0, len(labels), len(expected_targets)):
        if (
            tuple(
                label.target for label in labels[start : start + len(expected_targets)]
            )
            != expected_targets
        ):
            raise ValueError("Rule-label target order differs from frozen #20 schema")
    if len({(label.study_uid, label.target) for label in labels}) != len(labels):
        raise ValueError("Rule-label artifact contains duplicate study-target rows")
    return tuple(labels)


def compare_rule_and_llm(
    llm_labels: Sequence[LLMLabelRecord],
    rule_labels: Sequence[RuleLabelRecord],
    target_order: Sequence[str],
) -> dict[str, dict[str, int | float]]:
    """Aggregate private #20 comparison without treating agreement as correctness."""
    llm_by_key = {(label.study_uid, label.target): label.state for label in llm_labels}
    rule_by_key = {
        (label.study_uid, label.target): label.state for label in rule_labels
    }
    if len(llm_by_key) != len(llm_labels) or len(rule_by_key) != len(rule_labels):
        raise ValueError("Duplicate study-target row in report label comparison")
    if set(llm_by_key) != set(rule_by_key):
        raise ValueError("Rule and LLM label keys differ")
    comparison: dict[str, dict[str, int | float]] = {}
    for target in target_order:
        pairs = [
            (rule_state, llm_state)
            for (study_uid, row_target), llm_state in llm_by_key.items()
            if row_target == target
            for rule_state in (rule_by_key[(study_uid, row_target)],)
        ]
        if not pairs:
            raise ValueError(f"No comparison rows for target {target}")
        exact = sum(rule == llm for rule, llm in pairs)
        rule_coverage = sum(rule != "unknown" for rule, _ in pairs)
        llm_coverage = sum(llm != "unknown" for _, llm in pairs)
        comparison[target] = {
            "study_count": len(pairs),
            "exact_agreement_count": exact,
            "exact_agreement_rate": exact / len(pairs),
            "both_positive_count": sum(
                rule == llm == "positive" for rule, llm in pairs
            ),
            "both_negative_count": sum(
                rule == llm == "negative" for rule, llm in pairs
            ),
            "rule_unknown_llm_decisive_count": sum(
                rule == "unknown" and llm in {"positive", "negative"}
                for rule, llm in pairs
            ),
            "rule_decisive_llm_unknown_count": sum(
                rule in {"positive", "negative"} and llm == "unknown"
                for rule, llm in pairs
            ),
            "positive_negative_disagreement_count": sum(
                {rule, llm} == {"positive", "negative"} for rule, llm in pairs
            ),
            "uncertain_disagreement_count": sum(
                (rule == "uncertain") != (llm == "uncertain") for rule, llm in pairs
            ),
            "llm_coverage_count": llm_coverage,
            "llm_coverage_rate": llm_coverage / len(pairs),
            "rule_coverage_count": rule_coverage,
            "rule_coverage_rate": rule_coverage / len(pairs),
        }
    return comparison


def compute_stability(
    first: Sequence[LLMStudyResult],
    second: Sequence[LLMStudyResult],
    target_order: Sequence[str],
) -> dict[str, object]:
    """Compare two frozen greedy runs over the same ordered stability sample."""
    if [result.study_uid for result in first] != [
        result.study_uid for result in second
    ]:
        raise ValueError("Stability runs must use the identical ordered sample")
    if not first:
        raise ValueError("Stability sample must not be empty")
    exact = sum(
        left.states == right.states for left, right in zip(first, second, strict=True)
    )
    return {
        "study_count": len(first),
        "exact_structured_output_agreement_count": exact,
        "exact_structured_output_agreement_rate": exact / len(first),
        "per_target_disagreement_counts": {
            target: sum(
                left.states[target] != right.states[target]
                for left, right in zip(first, second, strict=True)
            )
            for target in target_order
        },
        "parse_failure_consistency_count": sum(
            left.parse_failed == right.parse_failed
            for left, right in zip(first, second, strict=True)
        ),
        "first_parse_failure_count": sum(result.parse_failed for result in first),
        "second_parse_failure_count": sum(result.parse_failed for result in second),
    }


def build_safe_summary(
    labels: Sequence[LLMLabelRecord],
    results: Sequence[LLMStudyResult],
    contract: LLMContract,
    *,
    comparison: Mapping[str, object],
    execution: Mapping[str, object],
    private_artifacts: Mapping[str, object],
) -> dict[str, object]:
    state_counts = {
        target: {
            state: sum(
                label.target == target and label.state == state for label in labels
            )
            for state in LABEL_STATES
        }
        for target in contract.target_order
    }
    study_count = len(results)
    summary: dict[str, object] = {
        "schema_version": 1,
        "extractor_version": contract.extractor_version,
        "model_repository": contract.model_repository,
        "model_revision": contract.model_revision,
        "model_asset_identity": dict(contract.model_asset_identity),
        "prompt_version": contract.prompt_version,
        "system_prompt_sha256": contract.system_prompt_sha256,
        "target_order": list(contract.target_order),
        "per_target_state_counts": state_counts,
        "per_target_coverage": {
            target: 1 - counts["unknown"] / study_count
            for target, counts in state_counts.items()
        },
        "study_parse_failure_count": sum(result.parse_failed for result in results),
        "study_parse_failure_rate": (
            sum(result.parse_failed for result in results) / study_count
            if study_count
            else 0.0
        ),
        "retry_count": sum(result.retry_count for result in results),
        "input_truncation_count": sum(result.input_truncated for result in results),
        "input_truncation_rate": (
            sum(result.input_truncated for result in results) / study_count
            if study_count
            else 0.0
        ),
        "execution": dict(execution),
        "private_artifacts": dict(private_artifacts),
        "rule_vs_llm_comparison": dict(comparison),
        "agreement_establishes_correctness": False,
    }
    serialized = json.dumps(summary, ensure_ascii=False)
    forbidden = (
        "StudyInstanceUID",
        "PatientID",
        "normalized_report_sha256",
        "report_text",
        "generated_text",
    )
    if any(value in serialized for value in forbidden):
        raise ValueError("Public-safe LLM summary contains a private row-level field")
    return summary


def write_llm_artifacts(
    rows: Sequence[ReportRow],
    results: Sequence[LLMStudyResult],
    contract: LLMContract,
    output_root: Path,
    *,
    expected_study_count: int = 4407,
    competition_contract_sha256: str,
    llm_contract_sha256: str,
    source_provenance: Mapping[str, object],
    execution: Mapping[str, object],
    rule_labels: Sequence[RuleLabelRecord],
    stability: Mapping[str, object],
) -> dict[str, Path]:
    """Write deterministic private rows and aggregate-safe extraction evidence."""
    _validate_sha256("competition_contract_sha256", competition_contract_sha256)
    _validate_sha256("llm_contract_sha256", llm_contract_sha256)
    paths = {
        "labels": output_root / "report-llm-labels-v1.csv",
        "study_status": output_root / "report-llm-study-status-v1.csv",
        "manifest": output_root / "report-llm-extraction-manifest.json",
        "summary": output_root / "report-llm-extraction-summary.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"LLM extraction artifacts already exist: {existing}")
    labels = build_label_records(
        rows, results, contract, expected_study_count=expected_study_count
    )
    label_bytes = serialize_label_csv(labels).encode("utf-8")
    status_bytes = serialize_study_status_csv(rows, results, contract).encode("utf-8")
    output_root.mkdir(parents=True, exist_ok=True)
    paths["labels"].write_bytes(label_bytes)
    paths["study_status"].write_bytes(status_bytes)
    private_artifacts = {
        paths[key].name: _artifact_record(paths[key])
        for key in ("labels", "study_status")
    }
    comparison = compare_rule_and_llm(labels, rule_labels, contract.target_order)
    summary = build_safe_summary(
        labels,
        results,
        contract,
        comparison=comparison,
        execution=execution,
        private_artifacts=private_artifacts,
    )
    summary["stability"] = dict(stability)
    stability_rate = stability.get("exact_structured_output_agreement_rate")
    summary["readiness"] = (
        "ready-for-weak-supervision-selection"
        if isinstance(stability_rate, (int, float)) and stability_rate >= 0.99
        else "not-ready"
    )
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    state_counts = Counter(label.state for label in labels)
    per_target_state_counts = {
        target: {
            state: sum(
                label.target == target and label.state == state for label in labels
            )
            for state in LABEL_STATES
        }
        for target in contract.target_order
    }
    parse_failure_count = sum(result.parse_failed for result in results)
    manifest = {
        "schema_version": 1,
        "extractor_version": contract.extractor_version,
        "prompt_version": contract.prompt_version,
        "git": {
            "commit": _required_text(source_provenance, "git_commit"),
            "dirty": _required_bool(source_provenance, "git_dirty"),
        },
        "notebook": {
            "source_sha256": _required_sha(source_provenance, "notebook_source_sha256"),
            "rendered_sha256": _required_sha(
                source_provenance, "rendered_notebook_sha256"
            ),
        },
        "competition_contract_sha256": competition_contract_sha256,
        "llm_contract_sha256": llm_contract_sha256,
        "model_repository": contract.model_repository,
        "model_revision": contract.model_revision,
        "model_license": dict(contract.model_license),
        "model_asset_identity": dict(contract.model_asset_identity),
        "model_file_hashes": dict(contract.model_file_hashes),
        "runtime": dict(execution),
        "internet_disabled": execution.get("internet_disabled") is True,
        "input_report_count": len(rows),
        "expected_label_row_count": expected_study_count * len(contract.target_order),
        "actual_label_row_count": len(labels),
        "target_order": list(contract.target_order),
        "state_counts": {state: state_counts[state] for state in LABEL_STATES},
        "per_target_state_counts": per_target_state_counts,
        "per_target_coverage": {
            target: 1 - counts["unknown"] / len(results)
            for target, counts in per_target_state_counts.items()
        },
        "study_parse_failure_count": parse_failure_count,
        "study_parse_failure_rate": parse_failure_count / len(results),
        "retry_count": sum(result.retry_count for result in results),
        "input_truncation_count": sum(result.input_truncated for result in results),
        "checkpoint": {
            "interval": contract.checkpoint_policy["completed_study_interval"],
            "maximum_sequence": max(
                (result.checkpoint_sequence for result in results), default=0
            ),
            "resume_exercised": execution.get("checkpoint_resume_exercised"),
            "resumed_study_count": execution.get("checkpoint_resumed_study_count"),
        },
        "stability": dict(stability),
        "private_artifacts": private_artifacts,
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def run_llm_extraction(
    rows: Sequence[ReportRow],
    contract: LLMContract,
    generate: BatchGenerator,
    *,
    checkpoint_path: Path,
) -> tuple[LLMStudyResult, ...]:
    """Run ordered batches with one same-request retry and idempotent resume."""
    requests = tuple(build_generation_request(row) for row in rows)
    study_uids = [request.study_uid for request in requests]
    if len(study_uids) != len(set(study_uids)):
        raise ValueError("Duplicate StudyInstanceUID in LLM report corpus")
    by_uid = {request.study_uid: request for request in requests}
    completed, checkpoint_sequence = _load_checkpoint(checkpoint_path, contract)
    completed_by_uid = {result.study_uid: result for result in completed}
    if not set(completed_by_uid) <= set(by_uid):
        raise ValueError("LLM checkpoint contains a study outside the requested corpus")
    for result in completed:
        if (
            result.normalized_report_sha256
            != by_uid[result.study_uid].normalized_report_sha256
        ):
            raise ValueError("LLM checkpoint report identity differs from input")

    results = list(completed)
    pending = [
        request for request in requests if request.study_uid not in completed_by_uid
    ]
    batch_size = _required_int(contract.decoding, "default_batch_size")
    interval = _required_int(contract.checkpoint_policy, "completed_study_interval")
    while pending:
        until_checkpoint = interval - (len(results) % interval)
        batch = tuple(pending[: min(batch_size, until_checkpoint)])
        del pending[: len(batch)]
        generated = generate(batch)
        if len(generated) != len(batch):
            raise RuntimeError("LLM generator returned the wrong batch size")
        parsed = [
            parse_generated_output(response.text, contract.target_order)
            for response in generated
        ]
        failed_indices = [
            index for index, item in enumerate(parsed) if item.parse_failed
        ]
        if failed_indices:
            retry_requests = tuple(batch[index] for index in failed_indices)
            retried = generate(retry_requests)
            if len(retried) != len(retry_requests):
                raise RuntimeError("LLM retry returned the wrong batch size")
            for retry_index, (original_index, response) in enumerate(
                zip(failed_indices, retried, strict=True)
            ):
                if (
                    response.input_truncated
                    != generated[original_index].input_truncated
                ):
                    raise RuntimeError("LLM retry changed the frozen input truncation")
                parsed[original_index] = parse_generated_output(
                    response.text, contract.target_order
                )
                generated = (
                    *generated[:original_index],
                    response,
                    *generated[original_index + 1 :],
                )
        failed_set = set(failed_indices)
        results.extend(
            LLMStudyResult(
                study_uid=request.study_uid,
                states=item.states,
                normalized_report_sha256=request.normalized_report_sha256,
                parse_failed=item.parse_failed,
                retry_count=int(index in failed_set),
                input_truncated=response.input_truncated,
                checkpoint_sequence=0,
            )
            for index, (request, response, item) in enumerate(
                zip(batch, generated, parsed, strict=True)
            )
        )
        if len(results) % interval == 0:
            checkpoint_sequence += 1
            results = _assign_checkpoint_sequence(results, checkpoint_sequence)
            _write_checkpoint(checkpoint_path, contract, results, checkpoint_sequence)
    if any(result.checkpoint_sequence == 0 for result in results):
        checkpoint_sequence += 1
        results = _assign_checkpoint_sequence(results, checkpoint_sequence)
        _write_checkpoint(checkpoint_path, contract, results, checkpoint_sequence)
    ordered = {result.study_uid: result for result in results}
    return tuple(ordered[request.study_uid] for request in requests)


def _assign_checkpoint_sequence(
    results: Sequence[LLMStudyResult], checkpoint_sequence: int
) -> list[LLMStudyResult]:
    return [
        replace(result, checkpoint_sequence=checkpoint_sequence)
        if result.checkpoint_sequence == 0
        else result
        for result in results
    ]


def _write_checkpoint(
    path: Path,
    contract: LLMContract,
    results: Sequence[LLMStudyResult],
    checkpoint_sequence: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "extractor_version": contract.extractor_version,
        "model_revision": contract.model_revision,
        "prompt_version": contract.prompt_version,
        "target_order": list(contract.target_order),
        "checkpoint_sequence": checkpoint_sequence,
        "completed_studies": [
            {
                "StudyInstanceUID": result.study_uid,
                "normalized_report_sha256": result.normalized_report_sha256,
                "states": result.states,
                "parse_failed": result.parse_failed,
                "retry_count": result.retry_count,
                "input_truncated": result.input_truncated,
                "checkpoint_sequence": result.checkpoint_sequence,
            }
            for result in results
        ],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_checkpoint(
    path: Path, contract: LLMContract
) -> tuple[tuple[LLMStudyResult, ...], int]:
    if not path.exists():
        return (), 0
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError("LLM checkpoint is corrupt") from error
    if not isinstance(raw, dict):
        raise TypeError("LLM checkpoint must be a JSON object")
    data = cast(dict[str, object], raw)
    expected_identity = {
        "schema_version": 1,
        "extractor_version": contract.extractor_version,
        "model_revision": contract.model_revision,
        "prompt_version": contract.prompt_version,
        "target_order": list(contract.target_order),
    }
    if any(data.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("LLM checkpoint contract identity differs")
    checkpoint_sequence = _required_int(data, "checkpoint_sequence")
    raw_results = data.get("completed_studies")
    if not isinstance(raw_results, list):
        raise TypeError("LLM checkpoint completed_studies is incomplete")
    results: list[LLMStudyResult] = []
    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            raise TypeError("LLM checkpoint contains a malformed study")
        item = cast(dict[str, object], raw_result)
        study_uid = _required_text(item, "StudyInstanceUID")
        report_sha = _required_sha(item, "normalized_report_sha256")
        states = item.get("states")
        if not isinstance(states, dict):
            raise TypeError("LLM checkpoint states are incomplete")
        parsed = parse_generated_output(json.dumps(states), contract.target_order)
        if parsed.parse_failed:
            raise ValueError("LLM checkpoint states violate the output schema")
        parse_failed = item.get("parse_failed")
        input_truncated = item.get("input_truncated")
        if not isinstance(parse_failed, bool) or not isinstance(input_truncated, bool):
            raise TypeError("LLM checkpoint status flags are malformed")
        results.append(
            LLMStudyResult(
                study_uid=study_uid,
                states=parsed.states,
                normalized_report_sha256=report_sha,
                parse_failed=parse_failed,
                retry_count=_required_int(item, "retry_count"),
                input_truncated=input_truncated,
                checkpoint_sequence=_required_int(item, "checkpoint_sequence"),
            )
        )
    study_uids = [result.study_uid for result in results]
    if len(study_uids) != len(set(study_uids)):
        raise ValueError("LLM checkpoint contains duplicate StudyInstanceUIDs")
    return tuple(results), checkpoint_sequence


def load_llm_contract(
    source: Path | Mapping[str, object], *, expected_target_order: Sequence[str]
) -> LLMContract:
    """Load the one frozen Qwen report-extraction contract and reject drift."""
    raw = (
        json.loads(source.read_text(encoding="utf-8"))
        if isinstance(source, Path)
        else dict(source)
    )
    if not isinstance(raw, dict):
        raise TypeError("LLM contract must be a JSON object")
    data = cast(dict[str, object], raw)
    if _required_int(data, "schema_version") != 1:
        raise ValueError("Unsupported LLM contract schema version")
    if _required_text(data, "extractor_version") != EXTRACTOR_VERSION:
        raise ValueError("LLM extractor version is not frozen v1")
    if _required_text(data, "model_repository") != MODEL_REPOSITORY:
        raise ValueError("LLM model repository differs from frozen Qwen")
    model_revision = _required_sha(data, "model_revision")
    tokenizer_revision = _required_sha(data, "tokenizer_revision")
    if tokenizer_revision != model_revision:
        raise ValueError("LLM tokenizer revision must match the frozen model revision")
    model_license = _required_mapping(data, "model_license")
    if (
        model_license.get("identifier") != "qwen-research"
        or model_license.get("compatibility")
        != "compatible for non-commercial research and evaluation only"
        or model_license.get("sha256")
        != "ef52482bb785733093dc9a2e8edd8e764c77d12d8e9d8f10a80c9b547d32d0f9"
        or model_revision not in str(model_license.get("text_location"))
    ):
        raise ValueError("LLM license identity or compatibility differs from frozen v1")
    model_asset = _required_mapping(data, "model_asset_identity")
    if (
        model_asset.get("kaggle_model") != "qwen-lm/qwen2.5/transformers/3b-instruct/1"
        or model_asset.get("mount_path")
        != "/kaggle/input/qwen2.5/transformers/3b-instruct/1"
        or model_asset.get("accessibility") != "public Kaggle Model"
        or model_asset.get("mount_omitted_source_files")
        != {
            ".gitattributes": {
                "sha256": (
                    "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361"
                ),
                "size_bytes": 1519,
            },
            "LICENSE": {
                "sha256": (
                    "ef52482bb785733093dc9a2e8edd8e764c77d12d8e9d8f10a80c9b547d32d0f9"
                ),
                "size_bytes": 7388,
            },
            "README.md": {
                "sha256": (
                    "2c74693e6ae36ba878124d465195b4dad0f71035b54ece1c9fd1553cc497ef8c"
                ),
                "size_bytes": 4930,
            },
        }
    ):
        raise ValueError("LLM Kaggle model asset differs from frozen v1")
    model_files = _required_mapping(data, "model_file_hashes")
    expected_model_files = {
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
    if set(model_files) != expected_model_files:
        raise ValueError("LLM model file inventory differs from frozen v1")
    for filename, record in model_files.items():
        if not isinstance(record, dict):
            raise TypeError(f"LLM model file record is malformed: {filename}")
        _required_int(record, "size_bytes")
        _required_sha(record, "sha256")
    target_order = _text_tuple(data, "target_order")
    if target_order != tuple(expected_target_order) or target_order != tuple(
        target for target, _ in _TARGET_DEFINITIONS
    ):
        raise ValueError("LLM contract target order differs from the verified order")
    if _required_text(data, "prompt_version") != PROMPT_VERSION:
        raise ValueError("LLM prompt version differs from frozen v1")
    prompt_sha = _required_sha(data, "system_prompt_sha256")
    if prompt_sha != hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest():
        raise ValueError("LLM system prompt hash differs from the implementation")
    output_schema = _required_mapping(data, "output_schema")
    if output_schema.get("keys") != list(target_order) or output_schema.get(
        "states"
    ) != list(LABEL_STATES):
        raise ValueError("LLM output schema differs from the four-state contract")
    decoding = _required_mapping(data, "decoding")
    if (
        decoding.get("do_sample") is not False
        or decoding.get("temperature") is not None
    ):
        raise ValueError("LLM decoding must use the frozen greedy path")
    if decoding.get("max_new_tokens") != 256 or decoding.get("default_batch_size") != 8:
        raise ValueError("LLM decoding limits differ from the frozen contract")
    if decoding.get("dtype") != "float16" or decoding.get("quantization") != "none":
        raise ValueError("LLM precision differs from frozen unquantized FP16")
    input_policy = _required_mapping(data, "input_token_policy")
    if (
        input_policy.get("truncation") != "preserve_beginning_and_end"
        or input_policy.get("max_input_tokens") != 32512
        or input_policy.get("model_context_tokens") != 32768
        or input_policy.get("preserved_regions")
        != "equal token counts from beginning and end"
    ):
        raise ValueError("LLM input-token policy differs from frozen v1")
    retry_policy = _required_mapping(data, "parse_retry_policy")
    if retry_policy != {"max_retries": 1, "same_contract": True}:
        raise ValueError("LLM retry policy differs from frozen v1")
    checkpoint_policy = _required_mapping(data, "checkpoint_policy")
    if checkpoint_policy != {
        "clean_shutdown": True,
        "completed_study_interval": 50,
        "resume_key": "StudyInstanceUID",
    }:
        raise ValueError("LLM checkpoint policy differs from frozen v1")
    normalization_version = _required_text(data, "normalization_version")
    if normalization_version != NORMALIZATION_VERSION:
        raise ValueError("LLM normalization must reuse report-extractor-v1")
    runtime = _required_mapping(data, "runtime")
    if runtime != {
        "cuda": "12.8",
        "gpu": "Tesla T4",
        "python": "3.12.13",
        "torch": "2.10.0+cu128",
        "transformers": "5.0.0",
    }:
        raise ValueError("LLM Kaggle runtime differs from frozen v1")
    return LLMContract(
        schema_version=1,
        extractor_version=EXTRACTOR_VERSION,
        model_repository=MODEL_REPOSITORY,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        model_license=model_license,
        model_asset_identity=model_asset,
        model_file_hashes=model_files,
        target_order=target_order,
        prompt_version=PROMPT_VERSION,
        system_prompt_sha256=prompt_sha,
        output_schema=output_schema,
        decoding=decoding,
        input_token_policy=input_policy,
        parse_retry_policy=retry_policy,
        checkpoint_policy=checkpoint_policy,
        normalization_version=normalization_version,
        runtime=runtime,
    )


def _required_mapping(parent: Mapping[str, object], key: str) -> dict[str, object]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"LLM contract field {key} must be an object")
    return cast(dict[str, object], value)


def _required_text(parent: Mapping[str, object], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"LLM contract field {key} must be nonempty text")
    return value


def _required_int(parent: Mapping[str, object], key: str) -> int:
    value = parent.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"LLM contract field {key} must be an integer")
    return value


def _required_bool(parent: Mapping[str, object], key: str) -> bool:
    value = parent.get(key)
    if not isinstance(value, bool):
        raise TypeError(f"LLM contract field {key} must be a boolean")
    return value


def _required_sha(parent: Mapping[str, object], key: str) -> str:
    value = _required_text(parent, key)
    if len(value) != 40 and len(value) != 64:
        raise ValueError(f"LLM contract field {key} must be a commit or SHA-256")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"LLM contract field {key} must be lowercase hexadecimal")
    return value


def _text_tuple(parent: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = parent.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise TypeError(f"LLM contract field {key} must be an array of text")
    return tuple(value)


def _validate_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: Path) -> dict[str, object]:
    return {"size_bytes": path.stat().st_size, "sha256": _file_sha256(path)}
