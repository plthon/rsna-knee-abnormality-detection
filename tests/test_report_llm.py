import copy
import csv
import hashlib
import json
from dataclasses import replace
from io import StringIO
from pathlib import Path
from typing import cast

import pytest

from rsna_knee.competition_contract import load_verified_submission_contract
from rsna_knee.report_llm import (
    SYSTEM_PROMPT,
    GenerationRequest,
    GenerationResponse,
    LLMContract,
    LLMLabelState,
    LLMStudyResult,
    RuleLabelRecord,
    build_generation_request,
    build_label_records,
    build_safe_summary,
    compare_rule_and_llm,
    compute_stability,
    llm_provenance_key,
    load_llm_contract,
    parse_generated_output,
    read_non_gold_study_uids,
    read_rule_label_csv,
    run_llm_extraction,
    select_hashed_reports,
    serialize_label_csv,
    serialize_study_status_csv,
    truncate_token_ids,
    verify_model_asset,
    write_llm_artifacts,
)
from rsna_knee.reports import ReportRow, normalized_report_sha256

PROJECT_ROOT = Path(__file__).parents[1]
TARGET_ORDER = load_verified_submission_contract(
    PROJECT_ROOT / "contracts" / "competition.json"
).target_names


def _contract() -> LLMContract:
    return load_llm_contract(
        PROJECT_ROOT / "contracts" / "report-llm-extractor-v1.json",
        expected_target_order=TARGET_ORDER,
    )


def _valid_output() -> dict[str, str]:
    return {
        target: ("positive", "negative", "uncertain", "unknown")[index % 4]
        for index, target in enumerate(TARGET_ORDER)
    }


def _study_result(
    study_uid: str,
    *,
    report: str | None = None,
    states: dict[str, str] | None = None,
    parse_failed: bool = False,
    retry_count: int = 0,
    input_truncated: bool = False,
) -> LLMStudyResult:
    return LLMStudyResult(
        study_uid=study_uid,
        states=states or _valid_output(),  # type: ignore[arg-type]
        normalized_report_sha256=normalized_report_sha256(
            report if report is not None else f"Report {study_uid}."
        ),
        parse_failed=parse_failed,
        retry_count=retry_count,
        input_truncated=input_truncated,
        checkpoint_sequence=1,
    )


def test_parser_accepts_exact_four_state_schema() -> None:
    expected = _valid_output()

    parsed = parse_generated_output(json.dumps(expected), TARGET_ORDER)

    assert parsed.states == expected
    assert parsed.parse_failed is False


@pytest.mark.parametrize("mutation", ["missing", "extra", "invalid"])
def test_parser_rejects_schema_drift_without_defaulting_negative(
    mutation: str,
) -> None:
    generated = _valid_output()
    if mutation == "missing":
        generated.pop("ACL")
    elif mutation == "extra":
        generated["Other"] = "unknown"
    else:
        generated["ACL"] = "absent"

    parsed = parse_generated_output(json.dumps(generated), TARGET_ORDER)

    assert parsed.states == {target: "unknown" for target in TARGET_ORDER}
    assert parsed.parse_failed is True


def test_parser_requires_one_bare_json_object() -> None:
    parsed = parse_generated_output(
        f"Here is the answer: {_valid_output()!r}", TARGET_ORDER
    )

    assert parsed.parse_failed is True
    assert set(parsed.states.values()) == {"unknown"}


def test_frozen_prompt_and_contract_encode_required_semantics() -> None:
    raw = json.loads(
        (PROJECT_ROOT / "contracts" / "report-llm-extractor-v1.json").read_text()
    )
    contract = load_llm_contract(raw, expected_target_order=TARGET_ORDER)

    assert contract.target_order == TARGET_ORDER
    assert (
        contract.system_prompt_sha256
        == hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    )
    assert "not mentioned" in SYSTEM_PROMPT
    assert "unknown" in SYSTEM_PROMPT
    assert "never negative" in SYSTEM_PROMPT
    assert [
        SYSTEM_PROMPT.index(f"{index}. {target}")
        for index, target in enumerate(TARGET_ORDER, 1)
    ] == sorted(
        SYSTEM_PROMPT.index(f"{index}. {target}")
        for index, target in enumerate(TARGET_ORDER, 1)
    )

    drifted = copy.deepcopy(raw)
    drifted["decoding"]["do_sample"] = True
    with pytest.raises(ValueError, match="greedy"):
        load_llm_contract(drifted, expected_target_order=TARGET_ORDER)


def test_generation_request_reuses_report_extractor_normalization() -> None:
    request = build_generation_request(ReportRow("study-1", "  No  tear.\r\n"))

    assert request.study_uid == "study-1"
    assert request.normalized_report == "No tear."
    assert request.normalized_report_sha256 == normalized_report_sha256(
        "  No  tear.\r\n"
    )
    assert request.messages == (
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": 'Report:\n"""\nNo tear.\n"""\n\nReturn the required JSON object.',
        },
    )


def test_same_request_retry_records_recovery(tmp_path: Path) -> None:
    requests: list[tuple[GenerationRequest, ...]] = []

    def generate(
        batch: tuple[GenerationRequest, ...],
    ) -> tuple[GenerationResponse, ...]:
        requests.append(batch)
        if len(requests) == 1:
            return (GenerationResponse("not json", False),)
        return (GenerationResponse(json.dumps(_valid_output()), False),)

    results = run_llm_extraction(
        (ReportRow("study-1", "Report text."),),
        _contract(),
        generate,
        checkpoint_path=tmp_path / "checkpoint.json",
    )

    assert requests[0] == requests[1]
    assert results[0].states == _valid_output()
    assert results[0].parse_failed is False
    assert results[0].retry_count == 1


def test_checkpoint_resume_skips_completed_studies_and_prevents_duplicates(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ...]] = []

    def generate(
        batch: tuple[GenerationRequest, ...],
    ) -> tuple[GenerationResponse, ...]:
        calls.append(tuple(request.study_uid for request in batch))
        return tuple(
            GenerationResponse(json.dumps(_valid_output()), False) for _ in batch
        )

    checkpoint = tmp_path / "checkpoint.json"
    first_rows = (
        ReportRow("study-1", "First."),
        ReportRow("study-2", "Second."),
    )
    run_llm_extraction(first_rows, _contract(), generate, checkpoint_path=checkpoint)
    calls.clear()

    results = run_llm_extraction(
        (*first_rows, ReportRow("study-3", "Third.")),
        _contract(),
        generate,
        checkpoint_path=checkpoint,
    )

    assert calls == [("study-3",)]
    assert [result.study_uid for result in results] == [
        "study-1",
        "study-2",
        "study-3",
    ]
    assert len({result.study_uid for result in results}) == 3

    with pytest.raises(ValueError, match="Duplicate StudyInstanceUID"):
        run_llm_extraction(
            (first_rows[0], first_rows[0]),
            _contract(),
            generate,
            checkpoint_path=tmp_path / "duplicate.json",
        )


def test_checkpoint_is_written_at_50_studies_and_clean_shutdown(
    tmp_path: Path,
) -> None:
    batch_sizes: list[int] = []

    def generate(
        batch: tuple[GenerationRequest, ...],
    ) -> tuple[GenerationResponse, ...]:
        batch_sizes.append(len(batch))
        return tuple(
            GenerationResponse(json.dumps(_valid_output()), False) for _ in batch
        )

    rows = tuple(ReportRow(f"study-{index:02}", "Normal.") for index in range(51))
    results = run_llm_extraction(
        rows,
        _contract(),
        generate,
        checkpoint_path=tmp_path / "checkpoint.json",
    )

    assert batch_sizes == [8, 8, 8, 8, 8, 8, 2, 1]
    assert {result.checkpoint_sequence for result in results[:50]} == {1}
    assert results[50].checkpoint_sequence == 2
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    assert checkpoint["checkpoint_sequence"] == 2
    assert len(checkpoint["completed_studies"]) == 51
    assert "Normal." not in json.dumps(checkpoint)


def test_corrupt_or_incompatible_checkpoint_fails_closed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{broken")

    with pytest.raises(ValueError, match="checkpoint"):
        run_llm_extraction(
            (ReportRow("study-1", "Report."),),
            _contract(),
            lambda _: (),
            checkpoint_path=checkpoint,
        )


def test_private_rows_use_original_study_and_frozen_target_order() -> None:
    rows = (
        ReportRow("study-2", "Report study-2."),
        ReportRow("study-1", "Report study-1."),
    )
    results = (_study_result("study-1"), _study_result("study-2"))

    labels = build_label_records(rows, results, _contract(), expected_study_count=2)

    assert [(label.study_uid, label.target) for label in labels] == [
        (study_uid, target)
        for study_uid in ("study-2", "study-1")
        for target in TARGET_ORDER
    ]
    expected_key = hashlib.sha256(
        json.dumps(
            [
                "study-2",
                "ACL",
                _contract().extractor_version,
                _contract().model_revision,
                _contract().prompt_version,
                normalized_report_sha256("Report study-2."),
            ],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert labels[0].provenance_key == expected_key
    assert (
        llm_provenance_key(
            "study-2",
            "ACL",
            _contract().extractor_version,
            _contract().model_revision,
            _contract().prompt_version,
            normalized_report_sha256("Report study-2."),
        )
        == expected_key
    )

    label_csv = serialize_label_csv(labels)
    status_csv = serialize_study_status_csv(rows, results, _contract())
    assert label_csv == serialize_label_csv(labels)
    assert next(csv.reader(StringIO(label_csv))) == [
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
    ]
    assert next(csv.reader(StringIO(status_csv))) == [
        "StudyInstanceUID",
        "extractor_version",
        "model_revision",
        "prompt_version",
        "normalized_report_sha256",
        "parse_failed",
        "retry_count",
        "input_truncated",
        "checkpoint_sequence",
    ]


def test_full_label_validation_requires_exact_52884_rows() -> None:
    with pytest.raises(ValueError, match="52884"):
        build_label_records(
            (ReportRow("study-1", "Report study-1."),),
            (_study_result("study-1"),),
            _contract(),
            expected_study_count=4407,
        )


def test_missingness_reader_never_returns_organizer_gold_values(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "StudyInstanceUID,Report,"
        + ",".join(TARGET_ORDER)
        + "\n"
        + "non-gold,Report,"
        + "," * 11
        + "\n"
        + "gold,Report,"
        + ",".join("1" if index % 2 else "0" for index in range(12))
        + "\n"
    )

    non_gold = read_non_gold_study_uids(train_csv, TARGET_ORDER)

    assert non_gold == frozenset({"non-gold"})
    assert all(not isinstance(value, (int, float)) for value in non_gold)


def test_smoke_and_stability_selection_use_fixed_hash_order() -> None:
    rows = tuple(ReportRow(f"study-{index}", "Report.") for index in range(100))
    eligible = frozenset(row.study_uid for row in rows if row.study_uid != "study-3")

    smoke = select_hashed_reports(rows, eligible, salt="report-llm-smoke-v1", count=32)
    stability = select_hashed_reports(
        rows, eligible, salt="report-llm-stability-v1", count=64
    )

    assert [row.study_uid for row in smoke] == sorted(
        eligible,
        key=lambda uid: hashlib.sha256(
            f"report-llm-smoke-v1|{uid}".encode()
        ).hexdigest(),
    )[:32]
    assert [row.study_uid for row in stability] == sorted(
        eligible,
        key=lambda uid: hashlib.sha256(
            f"report-llm-stability-v1|{uid}".encode()
        ).hexdigest(),
    )[:64]


def test_rule_comparison_reports_aggregate_four_state_relationships() -> None:
    llm_states = {target: "unknown" for target in TARGET_ORDER}
    llm_states["ACL"] = "positive"
    llm_states["MCL"] = "negative"
    llm_states["Effusion"] = "uncertain"
    labels = build_label_records(
        (ReportRow("study-1", "Report study-1."),),
        (_study_result("study-1", states=llm_states),),
        _contract(),
        expected_study_count=1,
    )
    rules = tuple(
        RuleLabelRecord(
            study_uid="study-1",
            target=target,
            state=cast(
                LLMLabelState,
                {"ACL": "negative", "MCL": "unknown", "Effusion": "positive"}.get(
                    target, "unknown"
                ),
            ),
        )
        for target in TARGET_ORDER
    )

    comparison = compare_rule_and_llm(labels, rules, TARGET_ORDER)

    assert comparison["ACL"]["positive_negative_disagreement_count"] == 1
    assert comparison["MCL"]["rule_unknown_llm_decisive_count"] == 1
    assert comparison["Effusion"]["uncertain_disagreement_count"] == 1
    assert comparison["Medial OA"]["exact_agreement_rate"] == 1.0
    assert comparison["ACL"]["llm_coverage_count"] == 1
    assert comparison["MCL"]["rule_coverage_count"] == 0


def test_safe_summary_excludes_row_identity_and_text() -> None:
    rows = (ReportRow("private-study", "sensitive report text"),)
    results = (
        _study_result(
            "private-study",
            report="sensitive report text",
            parse_failed=True,
            retry_count=1,
            input_truncated=True,
        ),
    )
    labels = build_label_records(rows, results, _contract(), expected_study_count=1)
    comparison = compare_rule_and_llm(
        labels,
        tuple(
            RuleLabelRecord("private-study", target, "unknown")
            for target in TARGET_ORDER
        ),
        TARGET_ORDER,
    )

    summary = build_safe_summary(
        labels,
        results,
        _contract(),
        comparison=comparison,
        execution={"runtime_seconds": 12.5, "gpu": "Tesla T4"},
        private_artifacts={"report-llm-labels-v1.csv": {"sha256": "a" * 64}},
    )
    serialized = json.dumps(summary)

    assert "private-study" not in serialized
    assert "sensitive report text" not in serialized
    assert normalized_report_sha256("Report private-study.") not in serialized
    assert summary["study_parse_failure_count"] == 1
    assert summary["input_truncation_count"] == 1


def test_model_asset_verification_checks_every_frozen_file(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    model_file = model_root / "weights.safetensors"
    model_file.write_bytes(b"frozen-weights")
    contract = replace(
        _contract(),
        model_file_hashes={
            model_file.name: {
                "size_bytes": model_file.stat().st_size,
                "sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(),
            }
        },
    )

    verified = verify_model_asset(model_root, contract)

    assert verified == contract.model_file_hashes
    model_file.write_bytes(b"drift")
    with pytest.raises(ValueError, match="model asset"):
        verify_model_asset(model_root, contract)


def test_token_truncation_preserves_fixed_beginning_and_end() -> None:
    token_ids = tuple(range(11))

    truncated, changed = truncate_token_ids(token_ids, max_tokens=6)

    assert truncated == (0, 1, 2, 8, 9, 10)
    assert changed is True
    assert truncate_token_ids(token_ids, max_tokens=11) == (token_ids, False)


def test_rule_label_reader_verifies_private_v1_artifact(tmp_path: Path) -> None:
    path = tmp_path / "report-labels-v1.csv"
    rows = [
        {
            "StudyInstanceUID": "study-1",
            "target": target,
            "state": "unknown",
            "extractor_version": "report-extractor-v1",
            "normalized_report_sha256": "a" * 64,
            "evidence_count": "0",
            "provenance_key": "b" * 64,
        }
        for target in TARGET_ORDER
    ]
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=rows[0], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(output.getvalue())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    labels = read_rule_label_csv(
        path,
        expected_sha256=digest,
        target_order=TARGET_ORDER,
        expected_study_count=1,
    )

    assert labels == tuple(
        RuleLabelRecord("study-1", target, "unknown") for target in TARGET_ORDER
    )
    with pytest.raises(ValueError, match="SHA-256"):
        read_rule_label_csv(
            path,
            expected_sha256="0" * 64,
            target_order=TARGET_ORDER,
            expected_study_count=1,
        )


def test_stability_requires_identical_sample_and_reports_disagreements() -> None:
    first_states = {target: "unknown" for target in TARGET_ORDER}
    second_states = dict(first_states)
    second_states["ACL"] = "positive"
    first = (
        _study_result("study-1", states=first_states),
        _study_result("study-2", states=first_states, parse_failed=True),
    )
    second = (
        _study_result("study-1", states=second_states),
        _study_result("study-2", states=first_states, parse_failed=True),
    )

    stability = compute_stability(first, second, TARGET_ORDER)

    assert stability["study_count"] == 2
    assert stability["exact_structured_output_agreement_rate"] == 0.5
    assert cast(dict[str, int], stability["per_target_disagreement_counts"])["ACL"] == 1
    assert stability["parse_failure_consistency_count"] == 2
    with pytest.raises(ValueError, match="sample"):
        compute_stability(first, second[:1], TARGET_ORDER)


def test_artifact_writer_emits_exact_private_schema_and_safe_aggregates(
    tmp_path: Path,
) -> None:
    rows = (
        ReportRow("study-1", "Report study-1."),
        ReportRow("study-2", "Report study-2."),
    )
    results = (_study_result("study-1"), _study_result("study-2"))
    rule_labels = tuple(
        RuleLabelRecord(row.study_uid, target, "unknown")
        for row in rows
        for target in TARGET_ORDER
    )

    paths = write_llm_artifacts(
        rows,
        results,
        _contract(),
        tmp_path,
        expected_study_count=2,
        competition_contract_sha256="a" * 64,
        llm_contract_sha256="b" * 64,
        source_provenance={
            "git_commit": "c" * 40,
            "git_dirty": False,
            "notebook_source_sha256": "d" * 64,
            "rendered_notebook_sha256": "e" * 64,
        },
        execution={
            "internet_disabled": True,
            "gpu": "Tesla T4",
            "runtime_seconds": 10.0,
            "throughput_studies_per_second": 0.2,
            "peak_gpu_memory_bytes": 123,
            "torch": "2.10.0+cu128",
            "transformers": "5.0.0",
        },
        rule_labels=rule_labels,
        stability={"exact_structured_output_agreement_rate": 1.0},
    )

    assert set(paths) == {"labels", "study_status", "manifest", "summary"}
    assert len(list(csv.DictReader(paths["labels"].open()))) == 24
    assert len(list(csv.DictReader(paths["study_status"].open()))) == 2
    manifest = json.loads(paths["manifest"].read_text())
    summary = json.loads(paths["summary"].read_text())
    assert manifest["actual_label_row_count"] == 24
    assert manifest["expected_label_row_count"] == 24
    assert manifest["internet_disabled"] is True
    assert (
        cast(dict[str, object], summary["stability"])[
            "exact_structured_output_agreement_rate"
        ]
        == 1.0
    )
    assert "study-1" not in json.dumps(summary)

    with pytest.raises(FileExistsError):
        write_llm_artifacts(
            rows,
            results,
            _contract(),
            tmp_path,
            expected_study_count=2,
            competition_contract_sha256="a" * 64,
            llm_contract_sha256="b" * 64,
            source_provenance={},
            execution={},
            rule_labels=rule_labels,
            stability={},
        )
