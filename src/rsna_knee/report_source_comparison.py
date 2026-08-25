"""Aggregate three-way comparison of frozen report-label sources."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import cast

type SourceState = str  # one of positive | negative | uncertain | unknown

DECISIVE_STATES: frozenset[str] = frozenset({"positive", "negative"})
STATE_VOCABULARY = frozenset({"positive", "negative", "uncertain", "unknown"})
PUBLIC_VERDICT_STATES = frozenset({"YES", "NO", "UNK"})
PUBLIC_VERDICT_MAPPING: Mapping[str, str] = {
    "YES": "positive",
    "NO": "negative",
    "UNK": "unknown",
}

COMPARISON_VERSION = "report-source-comparison-v1"
CONTRACT_RESOURCE = "report-source-comparison-v1.json"
EXPECTED_STUDY_COUNT = 4_407
GOLD_STUDY_COUNT = 58
NON_GOLD_STUDY_COUNT = 4_349
STRATUM_ORDER: tuple[str, ...] = (
    "triple-decisive-agree",
    "pairwise-agree-third-opposes",
    "pairwise-agree-third-abstains",
    "two-decisive-opposed",
    "single-decisive-only",
    "zero-decisive",
)
SOURCE_NAMES: tuple[str, ...] = ("rules", "llm", "public")
PAIR_NAMES: tuple[tuple[str, str], ...] = (
    ("rules", "llm"),
    ("rules", "public"),
    ("llm", "public"),
)
UNDER_CALL_FRACTION_THRESHOLD = 0.25
_PRIVATE_KEY_FRAGMENTS = (
    "studyinstanceuid",
    "patientid",
    "study_ids",
    "study_uids",
    "raw_report",
    "report_text",
    "generated_text",
    "normalized_report_sha256",
    "provenance_key",
    "evidence_span",
)


@dataclass(frozen=True, slots=True)
class ComparisonInputs:
    rules: Mapping[tuple[str, str], str]
    llm: Mapping[tuple[str, str], str]
    llm_parse_failed: Mapping[tuple[str, str], bool]
    public: Mapping[tuple[str, str], str]

    def source(self, name: str) -> Mapping[tuple[str, str], str]:
        if name == "rules":
            return self.rules
        if name == "llm":
            return self.llm
        return self.public


@dataclass(frozen=True, slots=True)
class SourceTriple:
    study_uid: str
    target: str
    rules: str | None
    llm: str | None
    public: str | None

    def state(self, source: str) -> str | None:
        if source == "rules":
            return self.rules
        if source == "llm":
            return self.llm
        return self.public

    def key(self) -> tuple[str, str]:
        return (self.study_uid, self.target)


@dataclass(frozen=True, slots=True)
class JoinedSources:
    """Every analysis-cohort study-target row; None marks an absent source."""

    triples: tuple[SourceTriple, ...]
    complete_triple_count: int
    public_missing_count: int

    def for_target(self, target: str) -> tuple[SourceTriple, ...]:
        return tuple(triple for triple in self.triples if triple.target == target)


@dataclass(frozen=True, slots=True)
class Cohorts:
    non_gold: frozenset[str]
    gold: frozenset[str]
    study_count: int

    @property
    def non_gold_study_count(self) -> int:
        return len(self.non_gold)

    @property
    def gold_study_count(self) -> int:
        return len(self.gold)

    @property
    def partially_labeled_study_count(self) -> int:
        return self.study_count - len(self.non_gold) - len(self.gold)


def _load_contract_bytes(path: Path | None, resource_name: str) -> bytes:
    if path is not None:
        return path.read_bytes()
    source_contract = Path(__file__).parents[2] / "contracts" / resource_name
    if source_contract.is_file():
        return source_contract.read_bytes()
    resource = files("rsna_knee").joinpath(resource_name)
    return resource.read_bytes()


def load_comparison_contract(
    path: Path | None,
    *,
    expected_target_order: Sequence[str],
) -> dict[str, object]:
    """Load the frozen comparison contract and reject any drift."""
    raw_bytes = _load_contract_bytes(path, CONTRACT_RESOURCE)
    contract = cast(dict[str, object], json.loads(raw_bytes))
    if not isinstance(contract, dict):
        raise TypeError("Comparison contract must be a JSON object")
    if contract.get("schema_version") != 1:
        raise ValueError("Unsupported comparison contract schema version")
    if contract.get("comparison_version") != COMPARISON_VERSION:
        raise ValueError("Comparison contract version differs from the implementation")
    target_order = contract.get("target_order")
    expected = tuple(expected_target_order)
    if target_order is None:
        # Target order is delegated to the competition contract.
        contract["target_order"] = list(expected)
    elif not isinstance(target_order, list) or tuple(target_order) != expected:
        raise ValueError("Comparison contract target order differs from competition")
    policy = contract.get("policy")
    if not isinstance(policy, dict):
        raise TypeError("Comparison contract policy must be an object")
    required_policy = {
        "gpu": False,
        "internet_enabled": False,
        "hosted_api_allowed": False,
        "organizer_gold_values_read": False,
        "merges_sources_into_training_labels": False,
        "modifies_any_extractor": False,
        "claims_correctness": False,
    }
    for key, value in required_policy.items():
        if policy.get(key) is not value:
            raise ValueError(f"Comparison contract policy violates {key}")
    inputs = contract.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise TypeError("Comparison contract must enumerate its frozen inputs")
    seen_roles: set[str] = set()
    for record in inputs:
        if not isinstance(record, dict):
            raise TypeError("Comparison contract input records must be objects")
        role = record.get("role")
        if not isinstance(role, str) or not role or role in seen_roles:
            raise ValueError(f"Comparison contract input role is invalid: {role!r}")
        seen_roles.add(role)
        sha256 = record.get("sha256")
        if not isinstance(sha256, str):
            raise TypeError(f"Input {role} must pin a SHA-256 identity")
        _validate_sha256(sha256)
        size = record.get("size_bytes")
        row_count = record.get("row_count")
        if size is not None and (not isinstance(size, int) or isinstance(size, bool)):
            raise TypeError(f"Input {role} pins a malformed size")
        if row_count is not None and (
            not isinstance(row_count, int) or isinstance(row_count, bool)
        ):
            raise TypeError(f"Input {role} pins a malformed row count")
    expected_roles = {
        "competition_contract",
        "rule_extractor_contract",
        "rule_labels",
        "llm_extractor_contract",
        "llm_labels",
        "llm_study_status",
        "public_dataset_contract",
        "public_labels",
    }
    missing_roles = expected_roles - seen_roles
    if missing_roles:
        raise ValueError(
            f"Comparison contract lacks pinned inputs: {sorted(missing_roles)}"
        )
    return contract


def pinned_input(
    contract: Mapping[str, object],
    *,
    role: str,
) -> dict[str, object]:
    """Return one frozen input record by role."""
    raw_inputs = contract["inputs"]
    assert isinstance(raw_inputs, list)
    for record in raw_inputs:
        assert isinstance(record, dict)
        if record.get("role") == role:
            return cast(dict[str, object], record)
    raise KeyError(f"Comparison contract has no pinned input {role!r}")


def verify_artifact_identity(
    role: str,
    path: Path,
    *,
    expected_size_bytes: int,
    expected_sha256: str,
) -> dict[str, object]:
    """Verify one frozen input's byte identity before any computation."""
    _validate_sha256(expected_sha256)
    actual_size = path.stat().st_size
    actual_sha256 = _file_sha256(path)
    if actual_size != expected_size_bytes or actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Frozen artifact identity mismatch for {role}: "
            f"{actual_size} bytes / {actual_sha256[:12]}…"
        )
    return {
        "role": role,
        "filename": path.name,
        "size_bytes": actual_size,
        "sha256": actual_sha256,
    }


def normalize_public_state(verdict: object) -> str:
    """Map one public verdict into the shared four-symbol vocabulary."""
    key = "" if verdict is None else str(verdict)
    if key not in PUBLIC_VERDICT_MAPPING:
        raise ValueError(
            f"Public verdict {key!r} lies outside the frozen state vocabulary"
        )
    return PUBLIC_VERDICT_MAPPING[key]


def normalize_report_state(value: object) -> str:
    """Validate one rules-or-LLM emitted state against the shared vocabulary."""
    key = "" if value is None else str(value)
    if key not in STATE_VOCABULARY:
        raise ValueError(
            f"Report-label state {key!r} lies outside the frozen state vocabulary"
        )
    return key


def read_competition_cohorts(
    path: Path,
    *,
    target_names: Sequence[str],
    identifier_column: str = "StudyInstanceUID",
) -> Cohorts:
    """Read only organizer-target missingness; never a target value."""
    import csv

    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.reader(source)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("Competition CSV is empty") from error
        if len(header) != len(set(header)):
            raise ValueError("Competition CSV header contains duplicate columns")
        required = (identifier_column, *target_names)
        missing = [column for column in required if column not in header]
        if missing:
            raise ValueError(f"Competition CSV is missing required columns: {missing}")
        uid_index = header.index(identifier_column)
        target_indices = tuple(header.index(target) for target in target_names)
        non_gold: set[str] = set()
        gold: set[str] = set()
        seen: set[str] = set()
        for source_row, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f"Malformed competition CSV row {source_row}")
            uid = row[uid_index].strip()
            if not uid or uid in seen:
                raise ValueError("StudyInstanceUID missing or duplicated in cohort CSV")
            seen.add(uid)
            present = [bool(row[index].strip()) for index in target_indices]
            if all(present):
                gold.add(uid)
            elif any(present):
                raise ValueError(
                    f"Partially labeled study at CSV row {source_row} violates the "
                    "all-or-nothing gold membership mask"
                )
            else:
                non_gold.add(uid)
    return Cohorts(
        non_gold=frozenset(non_gold), gold=frozenset(gold), study_count=len(seen)
    )


def ingest_report_labels(
    rows: Iterable[Mapping[str, object]],
    *,
    identifier_column: str = "StudyInstanceUID",
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], bool]]:
    """Ingest long-format rule or LLM label rows, rejecting duplicates."""
    states: dict[tuple[str, str], str] = {}
    parse_failed: dict[tuple[str, str], bool] = {}
    for row in rows:
        study_uid = str(row[identifier_column]).strip()
        target = str(row["target"]).strip()
        if not study_uid:
            raise ValueError("Report label rows require a nonblank study identifier")
        key = (study_uid, target)
        if key in states:
            raise ValueError(f"Duplicate study-target label row for target {target!r}")
        states[key] = normalize_report_state(row["state"])
        parse_failed[key] = str(row.get("parse_failed", "false")).strip().lower() == (
            "true"
        )
    return states, parse_failed


def ingest_public_labels(
    rows: Iterable[Mapping[str, object]],
    *,
    target_order: Sequence[str],
    identifier_column: str = "StudyInstanceUID",
) -> dict[tuple[str, str], str]:
    """Map public wide-format verdicts into states, rejecting drift."""
    allowed_targets = set(target_order)
    states: dict[tuple[str, str], str] = {}
    seen_studies: set[str] = set()
    for row in rows:
        study_uid = str(row[identifier_column]).strip()
        if not study_uid:
            raise ValueError("Public label rows require a nonblank study identifier")
        if study_uid in seen_studies:
            raise ValueError("Duplicate study identifier in public labels")
        seen_studies.add(study_uid)
        for target in allowed_targets:
            verdict = row[f"{target}__verdict"]
            key = (study_uid, target)
            if key in states:
                raise ValueError(
                    f"Duplicate study-target label row for target {target!r}"
                )
            states[key] = normalize_public_state(verdict)
    return states


def join_three_sources(
    inputs: ComparisonInputs,
    *,
    target_order: Sequence[str],
    cohort_uids: Collection[str],
) -> JoinedSources:
    """Join rule/LLM/public states per study-target over the given cohort."""
    cohort = set(cohort_uids)
    public_missing_count = 0
    triples: list[SourceTriple] = []
    complete = 0
    for study_uid in sorted(cohort):
        for target in target_order:
            key = (study_uid, target)
            rules_state = inputs.rules.get(key)
            llm_state = inputs.llm.get(key)
            public_state = inputs.public.get(key)
            if public_state is None:
                public_missing_count += 1
            if (
                rules_state is not None
                and llm_state is not None
                and public_state is not None
            ):
                complete += 1
            triples.append(
                SourceTriple(study_uid, target, rules_state, llm_state, public_state)
            )
    return JoinedSources(
        triples=tuple(triples),
        complete_triple_count=complete,
        public_missing_count=public_missing_count,
    )


def _is_decisive(state: str | None) -> bool:
    return state is not None and state in DECISIVE_STATES


def _single_triple_stratum(triple: SourceTriple) -> tuple[str, str | None]:
    """Classify one triple and detail how its non-unanimous shape arises."""
    states = (triple.rules, triple.llm, triple.public)
    decisive_indices = [
        index for index, state in enumerate(states) if _is_decisive(state)
    ]
    votes = [states[index] for index in decisive_indices]
    positives = sum(state == "positive" for state in votes)
    if len(votes) == 3:
        if positives in {0, 3}:
            return ("triple-decisive-agree", None)
        return ("pairwise-agree-third-opposes", None)
    if len(votes) == 2:
        abstaining = SOURCE_NAMES[
            next(index for index in range(3) if index not in decisive_indices)
        ]
        stratum = (
            "pairwise-agree-third-abstains"
            if positives in {0, 2}
            else "two-decisive-opposed"
        )
        return (stratum, f"abstaining-source={abstaining}")
    if len(votes) == 1:
        source = SOURCE_NAMES[decisive_indices[0]]
        return (
            "single-decisive-only",
            f"decisive-source={source};polarity={votes[0]}",
        )
    return ("zero-decisive", None)


def count_triple_combinations(
    triples: Sequence[SourceTriple],
) -> dict[str, int]:
    """Count observed (rules, LLM, public) triple combinations."""
    counts: Counter[str] = Counter()
    for triple in triples:
        key = (
            f"{triple.rules or 'missing'}|"
            f"{triple.llm or 'missing'}|"
            f"{triple.public or 'missing'}"
        )
        counts[key] += 1
    return dict(sorted(counts.items()))


def partition_strata(triples: Sequence[SourceTriple]) -> dict[str, int]:
    """Count every triple into exactly one rollup stratum."""
    counts: dict[str, int] = dict.fromkeys(STRATUM_ORDER, 0)
    for triple in triples:
        counts[_single_triple_stratum(triple)[0]] += 1
    return counts


def partition_strata_by_detail(
    triples: Sequence[SourceTriple],
) -> dict[str, dict[str, int]]:
    """Split non-unanimous strata by abstaining or decisive source detail."""
    splits: dict[str, dict[str, int]] = {
        "pairwise-agree-third-abstains": {},
        "two-decisive-opposed": {},
        "single-decisive-only": {},
    }
    for triple in triples:
        stratum, detail = _single_triple_stratum(triple)
        if detail is not None and stratum in splits:
            bucket = splits[stratum]
            bucket[detail] = bucket.get(detail, 0) + 1
    return splits


def count_any_uncertain(triples: Sequence[SourceTriple]) -> int:
    """Crosscut count of triples holding a rules-or-LLM uncertain state."""
    return sum(
        triple.rules == "uncertain" or triple.llm == "uncertain" for triple in triples
    )


def compare_pairwise(
    triples: Sequence[SourceTriple],
    *,
    pairs: Sequence[tuple[str, str]] = PAIR_NAMES,
) -> dict[tuple[str, str], dict[str, object]]:
    """Joint-decisive agreement computed identically across report generations."""
    results: dict[tuple[str, str], dict[str, object]] = {}
    for first_name, second_name in pairs:
        compared = 0
        agreements = 0
        conflicts = 0
        first_positive_second_negative = 0
        first_negative_second_positive = 0
        exact_agreements = 0
        overlap = 0
        for triple in triples:
            first = triple.state(first_name)
            second = triple.state(second_name)
            if first is None or second is None:
                continue
            overlap += 1
            if first == second:
                exact_agreements += 1
            if _is_decisive(first) and _is_decisive(second):
                compared += 1
                if first == second:
                    agreements += 1
                else:
                    conflicts += 1
                    if first == "positive":
                        first_positive_second_negative += 1
                    else:
                        first_negative_second_positive += 1
        results[(first_name, second_name)] = {
            "overlap_row_count": overlap,
            "exact_agreement_rate": exact_agreements / overlap if overlap else None,
            "compared_row_count": compared,
            "agreement_rate": agreements / compared if compared else None,
            "agreement_count": agreements,
            "polarity_conflict_count": conflicts,
            "first_positive_second_negative_count": first_positive_second_negative,
            "first_negative_second_positive_count": first_negative_second_positive,
        }
    return results


def compute_deep_dives(
    joined: JoinedSources,
    *,
    target_order: Sequence[str],
    parse_failed_keys: Collection[tuple[str, str]],
) -> dict[str, object]:
    """MCL disposition, OA pools, under-calling flags, and parse loss."""
    oa_targets = [target for target in target_order if "OA" in target]
    return {
        "mcl_conflict_disposition": _mcl_conflict_disposition(joined.for_target("MCL")),
        "oa_coverage_expansion": {
            target: _coverage_expansion(joined.for_target(target))
            for target in oa_targets
        },
        "under_calling_profile": _under_calling_profile(
            joined, target_order=target_order
        ),
        "parse_loss_counterfactual": compute_parse_loss(
            joined, parse_failed_keys=parse_failed_keys
        ),
    }


def _mcl_conflict_disposition(
    triples: tuple[SourceTriple, ...],
) -> dict[str, object]:
    directions: dict[str, object] = {}
    opposite = {"positive": "negative", "negative": "positive"}
    for direction, rules_polarity in (
        ("public-negative/rule-positive", "positive"),
        ("public-positive/rule-negative", "negative"),
    ):
        llm_votes: Counter[str] = Counter()
        count = 0
        for triple in triples:
            if (
                triple.rules == rules_polarity
                and triple.public == opposite[rules_polarity]
            ):
                count += 1
                llm_votes[triple.llm if triple.llm is not None else "missing"] += 1
        directions[direction] = {
            "conflict_row_count": count,
            "llm_vote_distribution": dict(sorted(llm_votes.items())),
        }
    return directions


def _decisive_indices(triples: tuple[SourceTriple, ...], source: str) -> set[int]:
    return {
        index
        for index, triple in enumerate(triples)
        if _is_decisive(triple.state(source))
    }


def _coverage_expansion(triples: tuple[SourceTriple, ...]) -> dict[str, object]:
    def pool(source: str) -> Counter[str]:
        states: list[str] = []
        for triple in triples:
            state = triple.state(source)
            if state is not None and _is_decisive(state):
                states.append(state)
        return Counter(states)

    pools = {name: pool(name) for name in SOURCE_NAMES}
    llm_keys = _decisive_indices(triples, "llm")
    public_keys = _decisive_indices(triples, "public")

    def positive_count(source: str) -> int:
        return pools[source]["positive"]

    return {
        "decisive_counts": {name: sum(pools[name].values()) for name in SOURCE_NAMES},
        "by_state": {
            state: {name: pools[name][state] for name in SOURCE_NAMES}
            for state in sorted(DECISIVE_STATES)
        },
        "llm_public_decisive_overlap": len(llm_keys & public_keys),
        "llm_only_decisive_pool": len(llm_keys - public_keys),
        "public_only_decisive_pool": len(public_keys - llm_keys),
        "rules_decisive_outside_llm_and_public": len(
            _decisive_indices(triples, "rules") - llm_keys - public_keys
        ),
        "llm_positive_public_positive_overlap": positive_count("llm")
        and sum(
            1
            for triple in triples
            if triple.llm == "positive" and triple.public == "positive"
        ),
    }


def _under_calling_profile(
    joined: JoinedSources,
    *,
    target_order: Sequence[str],
) -> dict[str, object]:
    profile: dict[str, object] = {}
    flagged: list[str] = []
    for target in target_order:
        target_triples = joined.for_target(target)

        def positives(
            source: str, triples: tuple[SourceTriple, ...] = target_triples
        ) -> int:
            return sum(triple.state(source) == "positive" for triple in triples)

        llm_positives = positives("llm")
        comparisons = {
            other: {
                "other_positive_count": (count := positives(other)),
                "ratio_llm_over_other": llm_positives / count if count else None,
                "under_called": bool(count)
                and llm_positives / count < UNDER_CALL_FRACTION_THRESHOLD,
            }
            for other in ("rules", "public")
        }
        if any(entry["under_called"] for entry in comparisons.values()):
            flagged.append(target)
        profile[target] = {"llm_positive_count": llm_positives, **comparisons}
    return {
        "flag_threshold_fraction": UNDER_CALL_FRACTION_THRESHOLD,
        "per_target": profile,
        "flagged_targets": flagged,
    }


def compute_parse_loss(
    joined: JoinedSources,
    *,
    parse_failed_keys: Collection[tuple[str, str]],
) -> dict[str, object]:
    """Decisive supervision the other two sources supply on parse-failed rows."""
    failed = set(parse_failed_keys)
    buckets: dict[str, Counter[str]] = {}
    total_rows = 0
    for triple in joined.triples:
        if triple.key() not in failed:
            continue
        total_rows += 1
        bucket = buckets.setdefault(triple.target, Counter())
        bucket["row_count"] += 1
        either_decisive = False
        for source, prefix in (("rules", "rules"), ("public", "public")):
            state = triple.state(source)
            if _is_decisive(state):
                assert state is not None
                bucket[f"{prefix}_decisive_count"] += 1
                bucket[f"{prefix}_{state}_count"] += 1
                either_decisive = True
        if either_decisive:
            bucket["either_other_source_decisive_count"] += 1
    ordered = {target: dict(buckets[target]) for target in sorted(buckets)}
    return {
        "parse_failed_row_count": total_rows,
        "per_target": ordered,
        "totals": {
            name: sum(bucket.get(name, 0) for bucket in buckets.values())
            for name in (
                "rules_decisive_count",
                "public_decisive_count",
                "either_other_source_decisive_count",
            )
        },
    }


def summarize_public_descriptive_domains(
    rows: Iterable[Mapping[str, object]],
    *,
    target_order: Sequence[str],
) -> dict[str, dict[str, dict[str, int]]]:
    """Record descriptive domain frequencies for public continuous score and conf columns."""
    scores: dict[str, Counter[str]] = {target: Counter() for target in target_order}
    confs: dict[str, Counter[str]] = {target: Counter() for target in target_order}
    for row in rows:
        for target in target_order:
            score_val = row.get(target)
            if score_val is not None and str(score_val).strip():
                scores[target][str(score_val).strip()] += 1
            conf_val = row.get(f"{target}__conf")
            if conf_val is not None and str(conf_val).strip():
                confs[target][str(conf_val).strip()] += 1
    return {
        "score_frequencies": {
            target: dict(sorted(scores[target].items())) for target in target_order
        },
        "confidence_frequencies": {
            target: dict(sorted(confs[target].items())) for target in target_order
        },
    }


def build_quota_feasibility(
    triples: Sequence[SourceTriple],
    *,
    target_order: Sequence[str],
) -> dict[str, dict[str, object]]:
    """Per target × source × decisive-state candidate evidence for #21."""
    contested_strata = {"pairwise-agree-third-opposes", "two-decisive-opposed"}
    table: dict[str, dict[str, object]] = {}
    for target in target_order:
        target_triples = [triple for triple in triples if triple.target == target]
        for source in SOURCE_NAMES:
            for state in sorted(DECISIVE_STATES):
                candidates = [
                    triple for triple in target_triples if triple.state(source) == state
                ]
                candidate_strata = partition_strata(candidates)
                table[f"{target}|{source}|{state}"] = {
                    "candidate_count": len(candidates),
                    "stratum_counts": {
                        name: candidate_strata[name] for name in STRATUM_ORDER
                    },
                    "triple_decisive_agree_count": candidate_strata[
                        "triple-decisive-agree"
                    ],
                    "contested_count": sum(
                        candidate_strata[name] for name in contested_strata
                    ),
                }
    return table


def build_public_safe_summary(
    *,
    comparison_version: str,
    input_identities: Sequence[Mapping[str, object]],
    completeness: Mapping[str, object],
    coverage: Mapping[str, object],
    per_target_strata: Mapping[str, object],
    pairwise: Mapping[str, object],
    deep_dives: Mapping[str, object],
    quota_feasibility: Mapping[str, object],
) -> dict[str, object]:
    """Build the aggregate-only summary and reject private row-level fields."""
    summary: dict[str, object] = {
        "schema_version": 1,
        "comparison_version": comparison_version,
        "input_identities": [dict(record) for record in input_identities],
        "completeness": dict(completeness),
        "cohort_coverage": dict(coverage),
        "per_target_rollup_strata": per_target_strata,
        "pairwise_joint_decisive": pairwise,
        "deep_dive_aggregates": deep_dives,
        "quota_feasibility": quota_feasibility,
        "makes_correctness_claim": False,
        "organizer_gold_values_read": False,
        "sources_merged_into_training_labels": False,
        "selects_validation_design_for_issue_21": False,
    }
    _reject_private_keys(summary)
    return summary


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


def _validate_sha256(value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("Pinned identities must be lowercase SHA-256 strings")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
