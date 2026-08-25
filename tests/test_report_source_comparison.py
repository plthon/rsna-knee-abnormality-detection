import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from rsna_knee.competition_contract import load_verified_submission_contract
from rsna_knee.report_source_comparison import (
    COMPARISON_VERSION,
    NON_GOLD_STUDY_COUNT,
    PUBLIC_VERDICT_STATES,
    STATE_VOCABULARY,
    ComparisonInputs,
    SourceTriple,
    build_public_safe_summary,
    build_quota_feasibility,
    compare_pairwise,
    compute_deep_dives,
    compute_parse_loss,
    ingest_public_labels,
    ingest_report_labels,
    join_three_sources,
    load_comparison_contract,
    normalize_public_state,
    normalize_report_state,
    partition_strata,
    read_competition_cohorts,
)

PROJECT_ROOT = Path(__file__).parents[1]
TARGET_ORDER = load_verified_submission_contract(
    PROJECT_ROOT / "contracts" / "competition.json"
).target_names


def _contract() -> dict[str, object]:
    return load_comparison_contract(
        PROJECT_ROOT / "contracts" / "report-source-comparison-v1.json",
        expected_target_order=TARGET_ORDER,
    )


def _inputs() -> ComparisonInputs:
    return ComparisonInputs(
        rules={
            ("study-1", "ACL"): "positive",
            ("study-1", "MCL"): "negative",
            ("study-2", "ACL"): "unknown",
            ("study-2", "MCL"): "positive",
        },
        llm={
            ("study-1", "ACL"): "positive",
            ("study-1", "MCL"): "uncertain",
            ("study-2", "ACL"): "negative",
            ("study-2", "MCL"): "positive",
        },
        llm_parse_failed={},
        public={
            ("study-1", "ACL"): "positive",
            ("study-1", "MCL"): "negative",
            ("study-2", "ACL"): "negative",
            ("study-2", "MCL"): "positive",
        },
    )


# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------


def test_frozen_contract_matches_pinned_input_identities() -> None:
    contract = _contract()

    by_role = {
        record["role"]: record for record in json.loads(json.dumps(contract))["inputs"]
    }
    assert (
        by_role["rule_labels"]["sha256"]
        == "6651f60b00e5bec292e8c98e7fa83c75e8aa5af5931414a9a68c8ce10a52d527"
    )
    assert (
        by_role["llm_labels"]["sha256"]
        == "d78c7f1d27437f6d2870e48fbd730a57c60d34380720cd4e0e07811a699db52a"
    )
    assert (
        by_role["llm_study_status"]["sha256"]
        == "151d0554cbd7622028df1bd58a4e2c37d18796c5860502cf3eb0917ec6baf923"
    )
    assert (
        by_role["public_labels"]["sha256"]
        == "6f704a7bdb2f894cc49445b19ba7c4378c3f548d3449e00361e10044bee40920"
    )
    assert by_role["rule_labels"]["size_bytes"] == 12_409_688
    assert by_role["llm_labels"]["size_bytes"] == 17_063_200
    assert by_role["llm_study_status"]["row_count"] == 4_407
    assert by_role["public_labels"]["row_count"] == 4_406
    assert contract["comparison_version"] == COMPARISON_VERSION


def test_contract_rejects_target_order_drift(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "contracts" / "report-source-comparison-v1.json"
    raw = json.loads(source.read_text())
    raw["target_order"] = ["ACL"]
    drifted = tmp_path / "drifted.json"
    drifted.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="target order"):
        load_comparison_contract(drifted, expected_target_order=TARGET_ORDER)


# ---------------------------------------------------------------------------
# State normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [("YES", "positive"), ("NO", "negative"), ("UNK", "unknown")],
)
def test_public_verdicts_normalize_to_shared_states(
    verdict: str, expected: str
) -> None:
    assert normalize_public_state(verdict) == expected


@pytest.mark.parametrize("verdict", ["yes", "MAYBE", "", "POSITIVE"])
def test_out_of_domain_public_verdicts_are_rejected(verdict: str) -> None:
    with pytest.raises(ValueError, match="state vocabulary"):
        normalize_public_state(verdict)


def test_state_vocabulary_holds_the_four_state_domain() -> None:
    assert STATE_VOCABULARY == frozenset(
        {"positive", "negative", "uncertain", "unknown"}
    )
    assert PUBLIC_VERDICT_STATES == frozenset({"YES", "NO", "UNK"})


# ---------------------------------------------------------------------------
# Three-way join
# ---------------------------------------------------------------------------


def test_join_keeps_absent_public_source_as_a_coverage_fact() -> None:
    inputs = ComparisonInputs(
        rules={("s1", "ACL"): "positive"},
        llm={("s1", "ACL"): "negative"},
        llm_parse_failed={},
        public={},
    )

    joined = join_three_sources(inputs, target_order=("ACL",), cohort_uids={"s1"})

    # The row is retained with an absent source; nothing silently drops.
    assert joined.triples == (
        (SourceTriple("s1", "ACL", "positive", "negative", None)),
    )
    assert joined.public_missing_count == 1
    assert joined.complete_triple_count == 0


def test_ingest_rejects_duplicate_study_target_rows() -> None:
    rows = [
        {"StudyInstanceUID": "s1", "target": "ACL", "state": "positive"},
        {"StudyInstanceUID": "s1", "target": "ACL", "state": "negative"},
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        ingest_report_labels(rows)


def test_ingest_normalizes_states_and_records_parse_flags() -> None:
    rows = [
        {
            "StudyInstanceUID": f"s{index}",
            "target": "ACL",
            "state": state,
            "parse_failed": flag,
        }
        for index, (state, flag) in enumerate(
            [("positive", "false"), ("unknown", "true")]
        )
    ]

    states, parse_failed = ingest_report_labels(rows)

    assert states == {("s0", "ACL"): "positive", ("s1", "ACL"): "unknown"}
    assert parse_failed == {("s0", "ACL"): False, ("s1", "ACL"): True}


def test_ingest_rejects_out_of_domain_report_state() -> None:
    with pytest.raises(ValueError, match="state vocabulary"):
        normalize_report_state("maybe")


def test_public_ingestion_maps_verdicts_and_rejects_duplicates() -> None:
    rows = [
        {"StudyInstanceUID": "s1", "ACL__verdict": "YES", "MCL__verdict": "NO"},
        {"StudyInstanceUID": "s2", "ACL__verdict": "UNK", "MCL__verdict": "YES"},
    ]

    states = ingest_public_labels(rows, target_order=("ACL", "MCL"))

    assert states == {
        ("s1", "ACL"): "positive",
        ("s1", "MCL"): "negative",
        ("s2", "ACL"): "unknown",
        ("s2", "MCL"): "positive",
    }
    with pytest.raises(ValueError, match="Duplicate study identifier"):
        ingest_public_labels(
            [
                {"StudyInstanceUID": "s1", "ACL__verdict": "YES"},
                {"StudyInstanceUID": "s1", "ACL__verdict": "NO"},
            ],
            target_order=("ACL",),
        )


def test_join_subsets_to_cohort_and_target_order() -> None:
    inputs = ComparisonInputs(
        rules={
            ("gold-study", "ACL"): "positive",
            ("s1", "ACL"): "positive",
            ("s1", "MCL"): "negative",
        },
        llm={
            ("gold-study", "ACL"): "negative",
            ("s1", "ACL"): "positive",
            ("s1", "MCL"): "negative",
        },
        llm_parse_failed={},
        public={
            ("gold-study", "ACL"): "positive",
            ("s1", "ACL"): "positive",
            ("s1", "MCL"): "negative",
        },
    )
    joined = join_three_sources(
        inputs, target_order=("ACL",), cohort_uids=frozenset({"s1"})
    )
    assert len(joined.triples) == 1
    assert joined.triples[0].study_uid == "s1"
    assert joined.triples[0].target == "ACL"
    assert joined.triples[0].rules == "positive"
    assert joined.complete_triple_count == 1


# ---------------------------------------------------------------------------
# Strata taxonomy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rules_state", "llm_state", "public_state", "stratum"),
    [
        ("positive", "positive", "positive", "triple-decisive-agree"),
        ("negative", "negative", "negative", "triple-decisive-agree"),
        ("positive", "positive", "negative", "pairwise-agree-third-opposes"),
        ("positive", "negative", "positive", "pairwise-agree-third-opposes"),
        ("positive", "positive", "unknown", "pairwise-agree-third-abstains"),
        ("positive", "positive", None, "pairwise-agree-third-abstains"),
        ("positive", "negative", "unknown", "two-decisive-opposed"),
        ("negative", "positive", None, "two-decisive-opposed"),
        ("positive", "unknown", "unknown", "single-decisive-only"),
        (None, "negative", None, "single-decisive-only"),
        ("uncertain", "unknown", None, "zero-decisive"),
        ("unknown", "uncertain", "unknown", "zero-decisive"),
    ],
)
def test_boundary_triples_land_in_exactly_one_stratum(
    rules_state: str | None,
    llm_state: str | None,
    public_state: str | None,
    stratum: str,
) -> None:
    triple = SourceTriple("s1", "ACL", rules_state, llm_state, public_state)

    counts = partition_strata((triple,))

    expected = dict.fromkeys(counts, 0)
    expected[stratum] = 1
    assert counts == expected
    assert sum(counts.values()) == 1


def test_count_triple_combinations() -> None:
    from rsna_knee.report_source_comparison import count_triple_combinations

    triples = (
        SourceTriple("s1", "ACL", "positive", "positive", "positive"),
        SourceTriple("s2", "ACL", "positive", "positive", "positive"),
        SourceTriple("s3", "ACL", "positive", "negative", "negative"),
        SourceTriple("s4", "ACL", "unknown", "unknown", None),
    )
    counts = count_triple_combinations(triples)
    assert counts == {
        "positive|negative|negative": 1,
        "positive|positive|positive": 2,
        "unknown|unknown|missing": 1,
    }


def test_every_synthetic_triple_partition_is_exact_and_total() -> None:
    states = ("positive", "negative", "uncertain", "unknown")
    triples = tuple(
        SourceTriple(f"s{index}", "ACL", *combination)
        for index, combination in enumerate(
            (r, l, p) for r in states for l in states for p in (*states, None)
        )
    )

    strata = partition_strata(triples)

    assert sum(strata.values()) == len(triples)
    assert set(strata) <= {
        "triple-decisive-agree",
        "pairwise-agree-third-opposes",
        "pairwise-agree-third-abstains",
        "two-decisive-opposed",
        "single-decisive-only",
        "zero-decisive",
    }


# ---------------------------------------------------------------------------
# Pairwise joint-decisive agreement
# ---------------------------------------------------------------------------


def test_pairwise_metrics_exclude_non_decisive_and_reconcile_by_hand() -> None:
    triples = (
        SourceTriple("s1", "ACL", "positive", "positive", "positive"),
        SourceTriple("s2", "ACL", "positive", "negative", "negative"),
        SourceTriple("s3", "ACL", "negative", "negative", "unknown"),
        SourceTriple("s4", "ACL", "positive", "unknown", "negative"),
        SourceTriple("s5", "ACL", "uncertain", "positive", "positive"),
    )

    pairwise = compare_pairwise(triples, pairs=(("rules", "llm"), ("llm", "public")))

    # rules∩llm compared rows: s1 (+/+ agree), s2 (+/- conflict),
    # s3 (-/- agree); s4 and s5 excluded because one member is non-decisive.
    rules_llm = pairwise[("rules", "llm")]
    assert rules_llm["compared_row_count"] == 3
    assert rules_llm["agreement_rate"] == pytest.approx(2 / 3)
    assert rules_llm["polarity_conflict_count"] == 1
    assert rules_llm["first_positive_second_negative_count"] == 1
    assert rules_llm["first_negative_second_positive_count"] == 0

    # llm∩public compared rows: s1 agree, s2 agree, s5 agree; s3/s4 exclude.
    llm_public = pairwise[("llm", "public")]
    assert llm_public["compared_row_count"] == 3
    assert llm_public["agreement_rate"] == 1.0
    assert llm_public["polarity_conflict_count"] == 0


# ---------------------------------------------------------------------------
# Cohort firewall
# ---------------------------------------------------------------------------


def test_cohort_split_reads_membership_only_and_fails_on_values_path(
    tmp_path: Path,
) -> None:
    frame = "StudyInstanceUID,Report,ACL,MCL\ns1,text,,\ngold-1,text,1,0\n"
    path = tmp_path / "train.csv"
    path.write_text(frame)

    cohorts = read_competition_cohorts(path, target_names=("ACL", "MCL"))

    assert cohorts.non_gold_study_count == 1
    assert cohorts.gold_study_count == 1
    assert cohorts.partially_labeled_study_count == 0
    assert cohorts.non_gold == frozenset({"s1"})
    assert cohorts.gold == frozenset({"gold-1"})


def test_partially_labeled_cohort_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    path.write_text("StudyInstanceUID,Report,ACL,MCL\ns1,text,1,\n")

    with pytest.raises(ValueError, match="[Pp]artially labeled"):
        read_competition_cohorts(path, target_names=("ACL", "MCL"))


# ---------------------------------------------------------------------------
# Quota feasibility
# ---------------------------------------------------------------------------


def _inputs_from_triples(
    triples: tuple[SourceTriple, ...],
    *,
    parse_failed_keys: frozenset[tuple[str, str]] = frozenset(),
) -> ComparisonInputs:
    return ComparisonInputs(
        rules={t.key(): t.rules for t in triples if t.rules is not None},
        llm={t.key(): t.llm for t in triples if t.llm is not None},
        llm_parse_failed={
            t.key(): t.key() in parse_failed_keys for t in triples if t.llm is not None
        },
        public={t.key(): t.public for t in triples if t.public is not None},
    )


def test_quota_feasibility_arithmetic() -> None:
    triples = (
        SourceTriple("s1", "ACL", "positive", "positive", "positive"),
        SourceTriple("s2", "ACL", "positive", "positive", "negative"),
        SourceTriple("s3", "ACL", "positive", "unknown", "positive"),
        SourceTriple("s4", "ACL", "negative", "unknown", None),
        SourceTriple("s5", "ACL", "unknown", "unknown", "unknown"),
    )

    table = build_quota_feasibility(triples, target_order=("ACL",))

    row = table["ACL|rules|positive"]
    strata = cast(dict[str, int], row["stratum_counts"])
    assert row["candidate_count"] == 3
    assert strata["triple-decisive-agree"] == 1
    assert strata["pairwise-agree-third-opposes"] == 1
    assert strata["pairwise-agree-third-abstains"] == 1
    assert row["contested_count"] == 1
    assert row["triple_decisive_agree_count"] == 1
    empty = table["ACL|llm|positive"]
    # s1 agrees unanimously and s2 is contested by the public negative vote.
    assert empty["candidate_count"] == 2
    assert empty["contested_count"] == 1
    assert table["ACL|llm|negative"]["candidate_count"] == 0
    negative = table["ACL|rules|negative"]
    negative_strata = cast(dict[str, int], negative["stratum_counts"])
    assert negative["candidate_count"] == 1
    assert negative_strata["single-decisive-only"] == 1


# ---------------------------------------------------------------------------
# Deep dives
# ---------------------------------------------------------------------------


def test_mcl_conflict_disposition_breaks_out_llm_votes() -> None:
    triples = (
        # Two public-negative/rule-positive conflicts with different LLM votes.
        SourceTriple("s1", "MCL", "positive", "unknown", "negative"),
        SourceTriple("s2", "MCL", "positive", "negative", "negative"),
        # One reverse-direction conflict.
        SourceTriple("s3", "MCL", "negative", "positive", "positive"),
        # Agreement rows must not appear in the disposition.
        SourceTriple("s4", "MCL", "positive", "positive", "positive"),
    )
    joined = join_three_sources(
        _inputs_from_triples(triples),
        target_order=("MCL",),
        cohort_uids={t.study_uid for t in triples},
    )

    disposition = cast(
        dict[str, dict[str, object]],
        compute_deep_dives(joined, target_order=("MCL",), parse_failed_keys=())[
            "mcl_conflict_disposition"
        ],
    )
    forward = disposition["public-negative/rule-positive"]
    reverse = disposition["public-positive/rule-negative"]

    assert forward["conflict_row_count"] == 2
    assert forward["llm_vote_distribution"] == {"negative": 1, "unknown": 1}
    assert reverse["conflict_row_count"] == 1
    assert reverse["llm_vote_distribution"] == {"positive": 1}


def test_parse_loss_counterfactual_counts_other_source_decisive_rows() -> None:
    triples = (
        SourceTriple("s1", "MCL", "positive", "unknown", "negative"),
        SourceTriple("s2", "MCL", "unknown", "unknown", None),
        SourceTriple("s3", "MCL", "negative", "negative", "negative"),
    )
    joined = join_three_sources(
        _inputs_from_triples(triples, parse_failed_keys=frozenset({("s1", "MCL")})),
        target_order=("MCL",),
        cohort_uids={"s1", "s2", "s3"},
    )

    parse_loss = compute_parse_loss(
        joined, parse_failed_keys={("s1", "MCL"), ("s2", "MCL")}
    )

    assert parse_loss["parse_failed_row_count"] == 2
    per_target = cast(dict[str, object], parse_loss["per_target"])
    mcl = cast(dict[str, int], per_target["MCL"])
    totals = cast(dict[str, int], parse_loss["totals"])
    assert mcl["rules_decisive_count"] == 1
    assert mcl["rules_positive_count"] == 1
    assert mcl["public_decisive_count"] == 1
    assert mcl["public_negative_count"] == 1
    assert mcl["either_other_source_decisive_count"] == 1
    assert totals["either_other_source_decisive_count"] == 1


def test_summarize_public_descriptive_domains() -> None:
    from rsna_knee.report_source_comparison import summarize_public_descriptive_domains

    rows = [
        {"StudyInstanceUID": "s1", "ACL": "0.94", "ACL__conf": "0.95"},
        {"StudyInstanceUID": "s2", "ACL": "0.08", "ACL__conf": "0.85"},
        {"StudyInstanceUID": "s3", "ACL": "0.94", "ACL__conf": "0.95"},
    ]
    summary = summarize_public_descriptive_domains(rows, target_order=("ACL",))
    assert summary["score_frequencies"]["ACL"] == {"0.08": 1, "0.94": 2}
    assert summary["confidence_frequencies"]["ACL"] == {"0.85": 1, "0.95": 2}


# ---------------------------------------------------------------------------
# Public-safe summary generation
# ---------------------------------------------------------------------------


def test_public_safe_summary_carries_aggregates_only() -> None:
    summary = build_public_safe_summary(
        comparison_version=COMPARISON_VERSION,
        input_identities=[{"role": "rule_labels", "sha256": "ab" * 32}],
        completeness={"rule_label_row_count": 52_884},
        coverage={"non_gold_study_count": NON_GOLD_STUDY_COUNT},
        per_target_strata={"ACL": {"triple-decisive-agree": 10}},
        pairwise={"rules|llm": {"agreement_rate": 0.9}},
        deep_dives={"under_calling_profile": {"flagged_targets": []}},
        quota_feasibility={"ACL|rules|positive": {"candidate_count": 40}},
    )

    serialized = json.dumps(summary)
    assert summary["schema_version"] == 1
    assert summary["makes_correctness_claim"] is False
    assert summary["organizer_gold_values_read"] is False
    assert summary["sources_merged_into_training_labels"] is False
    for forbidden in (
        "StudyInstanceUID",
        "PatientID",
        "normalized_report_sha256",
        "report_text",
        "provenance_key",
        "study-",
    ):
        assert forbidden not in serialized


def test_public_safe_summary_rejects_private_payload_keys() -> None:
    with pytest.raises(ValueError, match="private row-level field"):
        build_public_safe_summary(
            comparison_version=COMPARISON_VERSION,
            input_identities=[{"StudyInstanceUID": "leak"}],
            completeness={},
            coverage={},
            per_target_strata={},
            pairwise={},
            deep_dives={},
            quota_feasibility={},
        )


def test_package_archive_contains_the_comparison_contract() -> None:
    from rsna_knee.kaggle import _package_archive

    archive = _package_archive(
        PROJECT_ROOT / "src" / "rsna_knee",
        additional_contracts=("report-source-comparison-v1.json",),
    )
    digest = hashlib.sha256(archive).hexdigest()
    assert digest


def test_count_any_uncertain() -> None:
    from rsna_knee.report_source_comparison import count_any_uncertain

    triples = (
        SourceTriple("s1", "ACL", "uncertain", "positive", "positive"),
        SourceTriple("s2", "ACL", "positive", "uncertain", "positive"),
        SourceTriple("s3", "ACL", "positive", "positive", "unknown"),
        SourceTriple("s4", "ACL", "negative", "negative", "negative"),
    )
    assert count_any_uncertain(triples) == 2


def test_verify_artifact_identity(tmp_path: Path) -> None:
    from rsna_knee.report_source_comparison import verify_artifact_identity

    test_file = tmp_path / "test.csv"
    test_file.write_bytes(b"hello world")
    expected_hash = hashlib.sha256(b"hello world").hexdigest()

    result = verify_artifact_identity(
        "test_role",
        test_file,
        expected_size_bytes=11,
        expected_sha256=expected_hash,
    )
    assert result["role"] == "test_role"
    assert result["size_bytes"] == 11
    assert result["sha256"] == expected_hash

    with pytest.raises(RuntimeError, match="Frozen artifact identity mismatch"):
        verify_artifact_identity(
            "test_role",
            test_file,
            expected_size_bytes=10,
            expected_sha256=expected_hash,
        )

    with pytest.raises(RuntimeError, match="Frozen artifact identity mismatch"):
        verify_artifact_identity(
            "test_role",
            test_file,
            expected_size_bytes=11,
            expected_sha256="0" * 64,
        )


def test_oa_coverage_expansion_and_undercalling_profile() -> None:
    triples = (
        SourceTriple("s1", "Medial OA", "unknown", "positive", "positive"),
        SourceTriple("s2", "Medial OA", "unknown", "positive", "negative"),
        SourceTriple("s3", "Medial OA", "unknown", "negative", "positive"),
        SourceTriple("s4", "Medial OA", "unknown", "unknown", "positive"),
        SourceTriple("s5", "ACL", "positive", "positive", "positive"),
        SourceTriple("s6", "ACL", "positive", "unknown", "positive"),
        SourceTriple("s7", "ACL", "positive", "unknown", "positive"),
        SourceTriple("s8", "ACL", "positive", "unknown", "positive"),
        SourceTriple("s9", "ACL", "positive", "unknown", "positive"),
    )
    joined = join_three_sources(
        _inputs_from_triples(triples),
        target_order=("ACL", "Medial OA"),
        cohort_uids={t.study_uid for t in triples},
    )

    dives = compute_deep_dives(
        joined,
        target_order=("ACL", "Medial OA"),
        parse_failed_keys=(),
    )

    oa_expansion = cast(dict[str, dict[str, object]], dives["oa_coverage_expansion"])
    medial_oa = oa_expansion["Medial OA"]
    decisive_counts = cast(dict[str, int], medial_oa["decisive_counts"])
    assert decisive_counts["rules"] == 0
    assert decisive_counts["llm"] == 3
    assert decisive_counts["public"] == 4
    assert medial_oa["llm_public_decisive_overlap"] == 3
    assert medial_oa["llm_only_decisive_pool"] == 0
    assert medial_oa["public_only_decisive_pool"] == 1
    assert medial_oa["llm_positive_public_positive_overlap"] == 1

    undercalling = cast(dict[str, object], dives["under_calling_profile"])
    flagged = cast(list[str], undercalling["flagged_targets"])
    # For ACL: rules has 5 positives, public has 5 positives, LLM has 1 positive (1/5 = 0.20 < 0.25).
    assert "ACL" in flagged


def test_expected_non_gold_population_constant_matches_issue_evidence() -> None:
    assert NON_GOLD_STUDY_COUNT == 4_349
