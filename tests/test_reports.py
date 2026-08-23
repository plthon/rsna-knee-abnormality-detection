import csv
import hashlib
import json
from pathlib import Path

import pytest

from rsna_knee.competition_contract import load_verified_submission_contract
from rsna_knee.reports import (
    ReportRow,
    RuleContract,
    audit_report_corpus,
    extract_corpus,
    extract_report,
    load_rule_contract,
    normalize_report,
    normalized_report_sha256,
    provenance_key,
    read_report_csv,
    segment_report,
    write_extraction_artifacts,
)


def _rule_contract() -> dict[str, object]:
    return {
        "schema_version": 1,
        "extractor_version": "synthetic-v1",
        "audit_artifact_sha256": "a" * 64,
        "target_order": ["ACL", "Effusion"],
        "normalization_version": "unicode-nfc-horizontal-whitespace-v1",
        "section_rules": [],
        "assertion_scope_rules": {
            "window_tokens": 5,
            "scope_terminators": [r"\bbut\b", r"\bhowever\b", r"\byet\b"],
            "negation": [
                {
                    "id": "GLOBAL.NEG.001",
                    "pattern": r"\b(?:no|without)\b",
                    "direction": "forward",
                }
            ],
            "uncertainty": [
                {
                    "id": "GLOBAL.UNCERTAIN.001",
                    "pattern": r"\b(?:possible|question of)\b",
                    "direction": "forward",
                }
            ],
            "contextual_exclusions": [
                {
                    "id": "GLOBAL.EXCLUDE.001",
                    "pattern": r"\bstatus post\b",
                    "direction": "forward",
                }
            ],
        },
        "targets": [
            {
                "canonical_name": "ACL",
                "target_key": "ACL",
                "concept_patterns": [
                    {
                        "id": "ACL.CONCEPT.001",
                        "pattern": r"\bACL(?: tear)?\b",
                    },
                    {
                        "id": "ACL.CONCEPT.002",
                        "pattern": r"\banterior cruciate ligament(?: tear)?\b",
                    },
                ],
                "positive_rules": [],
                "negative_rules": [
                    {
                        "id": "ACL.NEG.001",
                        "pattern": r"\bintact\b",
                        "direction": "backward",
                    }
                ],
                "uncertainty_rules": [],
                "contextual_exclusions": [
                    {
                        "id": "ACL.EXCLUDE.001",
                        "pattern": r"\bACL reconstruction\b",
                        "direction": "bidirectional",
                    }
                ],
                "section_context_rules": [
                    {
                        "id": "ACL.SECTION.001",
                        "sections": ["history"],
                        "assertion_type": "contextual_exclusion",
                    }
                ],
                "conflict_policy": {
                    "id": "ACL.CONFLICT.001",
                    "policy": "unresolved_to_uncertain",
                },
            },
            {
                "canonical_name": "Effusion",
                "target_key": "EFFUSION",
                "concept_patterns": [
                    {
                        "id": "EFFUSION.CONCEPT.001",
                        "pattern": r"\beffusion\b",
                    }
                ],
                "positive_rules": [],
                "negative_rules": [],
                "uncertainty_rules": [],
                "contextual_exclusions": [],
                "section_context_rules": [],
                "conflict_policy": {
                    "id": "EFFUSION.CONFLICT.001",
                    "policy": "unresolved_to_uncertain",
                },
            },
        ],
    }


def _frozen_contract() -> RuleContract:
    project = Path(__file__).parents[1]
    target_order = load_verified_submission_contract(
        project / "contracts" / "competition.json"
    ).target_names
    return load_rule_contract(
        project / "contracts" / "report-extractor-v1.json",
        expected_target_order=target_order,
    )


def test_report_normalization_and_hash_preserve_clinical_structure() -> None:
    report = "  FINDINGS:\r\nNo\t ACL tear.  3 mm effusion.\rIMPRESSION: Caf\u00e9  "
    normalized = "FINDINGS:\nNo ACL tear. 3 mm effusion.\nIMPRESSION: Caf\u00e9"

    assert normalize_report(report) == normalized
    assert (
        normalized_report_sha256(report)
        == hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    )
    assert normalize_report(None) == ""


def test_report_segmentation_keeps_sections_sentences_and_offsets() -> None:
    report = normalize_report(
        "HISTORY: Remote injury.\nFINDINGS:\nNo ACL tear. Effusion is present."
    )

    segments = segment_report(report)

    assert [(item.section, item.text) for item in segments] == [
        ("history", "Remote injury."),
        ("findings", "No ACL tear."),
        ("findings", "Effusion is present."),
    ]
    assert all(
        report[item.span_start : item.span_end] == item.text for item in segments
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ACL tear is present.", "positive"),
        ("No ACL tear.", "negative"),
        ("Possible ACL tear.", "uncertain"),
        ("No acute osseous abnormality.", "unknown"),
        ("Anterior cruciate ligament tear is present.", "positive"),
        ("The ACL is intact.", "negative"),
    ],
)
def test_report_extraction_resolves_basic_assertion_states(
    text: str, expected: str
) -> None:
    contract = load_rule_contract(
        _rule_contract(), expected_target_order=("ACL", "Effusion")
    )

    extraction = extract_report("study-1", text, contract)

    assert extraction.labels[0].state == expected
    assert extraction.labels[1].state == "unknown"


def test_assertion_scopes_stop_at_unrelated_findings() -> None:
    contract = load_rule_contract(
        _rule_contract(), expected_target_order=("ACL", "Effusion")
    )

    negated = extract_report(
        "study-1", "No ACL tear but effusion is present.", contract
    )
    uncertain = extract_report(
        "study-2", "Possible ACL tear but effusion is present.", contract
    )

    assert [label.state for label in negated.labels] == ["negative", "positive"]
    assert [label.state for label in uncertain.labels] == ["uncertain", "positive"]


def test_conflicts_context_and_multiple_mentions_are_conservative() -> None:
    contract = load_rule_contract(
        _rule_contract(), expected_target_order=("ACL", "Effusion")
    )

    contradiction = extract_report(
        "study-1", "No ACL tear. ACL tear is present.", contract
    )
    historical = extract_report(
        "study-2", "HISTORY: ACL tear.\nFINDINGS: No effusion.", contract
    )
    postoperative = extract_report(
        "study-3", "Status post ACL reconstruction.", contract
    )
    repeated = extract_report(
        "study-4", "ACL tear is present. ACL tear is again seen.", contract
    )

    assert contradiction.labels[0].state == "uncertain"
    assert {item.conflict_rule for item in contradiction.evidence} == {
        "ACL.CONFLICT.001"
    }
    assert [label.state for label in historical.labels] == ["unknown", "negative"]
    assert historical.labels[0].evidence_count == 0
    assert historical.evidence[0].assertion_type == "contextual_exclusion"
    assert postoperative.labels[0].state == "unknown"
    assert postoperative.evidence[0].assertion_type == "contextual_exclusion"
    assert repeated.labels[0].state == "positive"
    assert repeated.labels[0].evidence_count == 2


def test_provenance_contract_and_evidence_spans_are_stable() -> None:
    contract = load_rule_contract(
        _rule_contract(), expected_target_order=("ACL", "Effusion")
    )
    extraction = extract_report("study-1", "No ACL tear.", contract)
    normalized = normalize_report("No ACL tear.")

    assert (
        provenance_key(
            "study-1",
            "ACL",
            "synthetic-v1",
            normalized_report_sha256(normalized),
        )
        == "a2bafde14be350793a7ddc54e313b688eaba565b40a9e0aad73ceb8bdf2ce57b"
    )
    assert extraction.labels[0].provenance_key == provenance_key(
        "study-1",
        "ACL",
        "synthetic-v1",
        normalized_report_sha256(normalized),
    )
    assert all(
        normalized[item.span_start : item.span_end].casefold().startswith("acl")
        for item in extraction.evidence
    )
    assert extraction.evidence[0].as_dict().keys() == {
        "provenance_key",
        "StudyInstanceUID",
        "target",
        "extractor_version",
        "normalized_report_sha256",
        "rule_ids",
        "concept_id",
        "assertion_type",
        "section",
        "span_start",
        "span_end",
        "conflict_rule",
    }


def test_rule_contract_rejects_drift_and_unstable_ids() -> None:
    wrong_version = _rule_contract()
    wrong_version["schema_version"] = 2
    with pytest.raises(ValueError, match="schema version"):
        load_rule_contract(wrong_version, expected_target_order=("ACL", "Effusion"))

    wrong_order = _rule_contract()
    wrong_order["target_order"] = ["Effusion", "ACL"]
    with pytest.raises(ValueError, match="verified order"):
        load_rule_contract(wrong_order, expected_target_order=("ACL", "Effusion"))

    bad_id = _rule_contract()
    targets = bad_id["targets"]
    assert isinstance(targets, list)
    first_target = targets[0]
    assert isinstance(first_target, dict)
    concepts = first_target["concept_patterns"]
    assert isinstance(concepts, list)
    first_concept = concepts[0]
    assert isinstance(first_concept, dict)
    first_concept["id"] = "ACL-1"
    with pytest.raises(ValueError, match="ACL.CONCEPT.NNN"):
        load_rule_contract(bad_id, expected_target_order=("ACL", "Effusion"))


def test_full_corpus_extraction_is_ordered_complete_and_repeatable() -> None:
    contract = load_rule_contract(
        _rule_contract(), expected_target_order=("ACL", "Effusion")
    )
    rows = (
        ReportRow("study-2", "Effusion is present."),
        ReportRow("study-1", "No ACL tear."),
    )

    first = extract_corpus(rows, contract, expected_study_count=2)
    second = extract_corpus(rows, contract, expected_study_count=2)

    assert first == second
    assert [(row.study_uid, row.target) for row in first.labels] == [
        ("study-2", "ACL"),
        ("study-2", "Effusion"),
        ("study-1", "ACL"),
        ("study-1", "Effusion"),
    ]
    assert len(first.labels) == 4
    assert [item.span_start for item in first.evidence] == [0, 3]

    with pytest.raises(ValueError, match="Duplicate StudyInstanceUID"):
        extract_corpus((rows[0], rows[0]), contract)


def test_audit_is_aggregate_only_and_enforces_document_frequency_privacy() -> None:
    rows = tuple(
        ReportRow(
            f"private-study-{index}",
            "FINDINGS: ACL tear present. Effusion absent.",
        )
        for index in range(5)
    ) + (ReportRow("private-study-5", "FINDINGS: rareword."),)

    audit = audit_report_corpus(
        rows,
        ("ACL", "Effusion"),
        expected_study_count=6,
        source_hashes={"competition_contract_sha256": "b" * 64},
    )
    serialized = json.dumps(audit, sort_keys=True)

    assert audit["corpus"] == {
        "total_rows": 6,
        "unique_study_identifiers": 6,
        "empty_normalized_reports": 0,
        "nonempty_normalized_reports": 6,
        "normalized_length_characters": {
            "minimum": 19,
            "maximum": 44,
            "mean": 39.833333,
            "p25": 44,
            "p50": 44,
            "p75": 44,
            "p95": 44,
        },
    }
    assert audit["section_heading_counts"] == [
        {"document_frequency": 6, "phrase": "findings"}
    ]
    assert "rareword" not in serialized
    assert "private-study" not in serialized
    assert audit["language"] == "unresolved"

    def lexical_records(value: object) -> list[dict[str, object]]:
        if isinstance(value, dict):
            records = [value] if "phrase" in value else []
            return records + [
                item for child in value.values() for item in lexical_records(child)
            ]
        if isinstance(value, list):
            return [item for child in value for item in lexical_records(child)]
        return []

    assert all(
        isinstance(item["document_frequency"], int) and item["document_frequency"] >= 5
        for item in lexical_records(audit)
    )
    contradictions = audit["potentially_contradictory_assertion_patterns"]
    assert isinstance(contradictions, dict)
    assert contradictions["ACL"] == {
        "reports_with_potential_contradiction": 0,
        "patterns": [],
    }
    assert contradictions["Effusion"] == {
        "reports_with_potential_contradiction": 0,
        "patterns": [],
    }


def test_audit_neighbors_and_contradictions_are_anchored_to_each_target() -> None:
    rows = tuple(
        ReportRow(
            f"study-{index}",
            (
                "FINDINGS: Medial meniscus tear. Medial compartment narrowing. "
                "No ACL tear. ACL tear is present."
            ),
        )
        for index in range(5)
    )

    audit = audit_report_corpus(rows, ("Medial Meniscus", "ACL"))
    terminology = audit["target_oriented_candidate_terminology"]
    assert isinstance(terminology, dict)
    meniscus = terminology["Medial Meniscus"]
    assert isinstance(meniscus, dict)
    neighbors = meniscus["corpus_observed_neighbors"]
    assert isinstance(neighbors, list)

    assert {item["phrase"] for item in neighbors} >= {"meniscus tear"}
    assert "medial compartment" not in {item["phrase"] for item in neighbors}
    contradictions = audit["potentially_contradictory_assertion_patterns"]
    assert isinstance(contradictions, dict)
    acl = contradictions["ACL"]
    assert isinstance(acl, dict)
    assert acl == {
        "reports_with_potential_contradiction": 5,
        "patterns": [
            {
                "positive_cue": "present",
                "negative_cue": "no",
                "document_frequency": 5,
            }
        ],
    }


def test_audit_csv_reader_never_requests_organizer_gold(tmp_path: Path) -> None:
    source = tmp_path / "train.csv"
    source.write_text(
        'StudyInstanceUID,Report,ACL,Effusion\nstudy-1,"No ACL tear.",0,1\n',
        encoding="utf-8",
    )

    assert read_report_csv(
        source,
        requested_columns=("StudyInstanceUID", "Report"),
        target_names=("ACL", "Effusion"),
    ) == (ReportRow("study-1", "No ACL tear."),)
    with pytest.raises(PermissionError, match="organizer-gold"):
        read_report_csv(
            source,
            requested_columns=("StudyInstanceUID", "Report", "ACL"),
            target_names=("ACL", "Effusion"),
        )


@pytest.mark.parametrize(
    ("target", "text"),
    [
        ("ACL", "ACL tear."),
        ("MCL", "MCL sprain."),
        ("Medial Meniscus", "Medial meniscus tear."),
        ("Lateral Meniscus", "Lateral meniscus tear."),
        ("Medial OA", "Medial osteoarthritis."),
        ("Lateral OA", "Lateral osteoarthritis."),
        ("PF OA", "Patellofemoral osteoarthritis."),
        ("Effusion", "Joint effusion is present."),
        ("Synovitis", "Synovitis is present."),
        ("Baker's", "Baker's cyst is present."),
        ("Contusion", "Osseous contusion is present."),
        ("Fracture", "Fracture is present."),
    ],
)
def test_frozen_contract_recognizes_each_target(target: str, text: str) -> None:
    extraction = extract_report("study-1", text, _frozen_contract())

    assert {row.target: row.state for row in extraction.labels}[target] == "positive"


@pytest.mark.parametrize(
    ("target", "text"),
    [
        ("ACL", "ACL normal."),
        ("MCL", "The MCL is intact."),
        ("Medial Meniscus", "Medial meniscus normal."),
        ("Lateral Meniscus", "The lateral meniscus is intact."),
        ("Effusion", "Effusion no intra-articular bodies."),
        ("Synovitis", "No synovitis."),
        ("Baker's", "No Baker's cyst."),
        ("Contusion", "No bone contusion."),
        ("Fracture", "No acute fracture."),
    ],
)
def test_frozen_contract_recognizes_explicit_negative_evidence(
    target: str, text: str
) -> None:
    extraction = extract_report("study-1", text, _frozen_contract())

    assert {row.target: row.state for row in extraction.labels}[target] == "negative"


def test_frozen_contract_handles_uncertainty_context_and_conflicts() -> None:
    contract = _frozen_contract()

    uncertain = extract_report("study-1", "Possible ACL tear.", contract)
    history = extract_report("study-2", "HISTORY: ACL tear.", contract)
    remote = extract_report("study-3", "Remote bone contusion.", contract)
    conflict = extract_report("study-4", "No ACL tear. ACL tear is present.", contract)

    assert uncertain.labels[0].state == "uncertain"
    assert history.labels[0].state == "unknown"
    assert remote.labels[10].state == "unknown"
    assert conflict.labels[0].state == "uncertain"
    assert {item.conflict_rule for item in conflict.evidence} == {"ACL.CONFLICT.001"}


def test_private_extraction_artifacts_are_complete_deterministic_and_public_safe(
    tmp_path: Path,
) -> None:
    contract = load_rule_contract(
        _rule_contract(), expected_target_order=("ACL", "Effusion")
    )
    rows = (
        ReportRow("study-2", "Effusion is present."),
        ReportRow("study-1", "No ACL tear."),
    )
    provenance = {
        "git_commit": "test-commit",
        "git_dirty": False,
        "notebook_source_sha256": "b" * 64,
        "rendered_notebook_sha256": "c" * 64,
        "rendered_notebook_sha256_canonicalization": (
            "recorded field replaced by 64 zeroes"
        ),
        "package_archive_sha256": "d" * 64,
    }

    first = write_extraction_artifacts(
        rows,
        contract,
        tmp_path / "first",
        expected_study_count=2,
        competition_contract_sha256="e" * 64,
        rule_contract_sha256="f" * 64,
        source_provenance=provenance,
        runtime_seconds=1.25,
    )
    second = write_extraction_artifacts(
        rows,
        contract,
        tmp_path / "second",
        expected_study_count=2,
        competition_contract_sha256="e" * 64,
        rule_contract_sha256="f" * 64,
        source_provenance=provenance,
        runtime_seconds=1.25,
    )

    with first["labels"].open(newline="", encoding="utf-8") as source:
        label_reader = csv.DictReader(source)
        label_rows = list(label_reader)
    evidence_rows = [
        json.loads(line)
        for line in first["evidence"].read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads(first["manifest"].read_text(encoding="utf-8"))
    summary = json.loads(first["summary"].read_text(encoding="utf-8"))

    assert label_reader.fieldnames == [
        "StudyInstanceUID",
        "target",
        "state",
        "extractor_version",
        "normalized_report_sha256",
        "evidence_count",
        "provenance_key",
    ]
    assert [(row["StudyInstanceUID"], row["target"]) for row in label_rows] == [
        ("study-2", "ACL"),
        ("study-2", "Effusion"),
        ("study-1", "ACL"),
        ("study-1", "Effusion"),
    ]
    assert len(evidence_rows) == 2
    assert "Effusion is present" not in first["evidence"].read_text(encoding="utf-8")
    assert manifest["expected_label_row_count"] == 4
    assert manifest["actual_label_row_count"] == 4
    assert manifest["deterministic_equality_verified"] is True
    assert summary["validation_readiness"] == {
        "label_correctness_established": False,
        "status": "technically_ready_for_issue_21_manual_validation",
    }
    assert "study-1" not in json.dumps(summary)
    assert "study-2" not in json.dumps(manifest)
    assert {
        name: first[name].read_bytes() for name in ("labels", "evidence", "summary")
    } == {name: second[name].read_bytes() for name in ("labels", "evidence", "summary")}
