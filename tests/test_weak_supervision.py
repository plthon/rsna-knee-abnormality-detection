import hashlib
from pathlib import Path

import pytest

from rsna_knee.competition_contract import load_verified_submission_contract
from rsna_knee.weak_supervision import (
    PUBLIC_VERDICT_TO_LABEL,
    QWEN_STATE_TO_LABEL,
    classify_gold_membership,
    load_weak_supervision_policy,
    resolve_weak_labels,
    verify_source_identity,
)

PROJECT_ROOT = Path(__file__).parents[1]
TARGETS = load_verified_submission_contract(
    PROJECT_ROOT / "contracts" / "competition.json"
).target_names


def test_policy_freezes_direct_candidates_and_shared_public_mapping() -> None:
    policy = load_weak_supervision_policy(
        PROJECT_ROOT / "contracts" / "report-weak-supervision-v1.json"
    )

    assert policy.policy_version == "report-weak-supervision-v1"
    assert policy.selection_owner == "issue-10"
    assert set(policy.candidates) == {
        "gold_only",
        "public_direct_v1",
        "qwen_direct_v1",
    }
    assert PUBLIC_VERDICT_TO_LABEL == {"YES": 1, "NO": 0, "UNK": None}
    assert QWEN_STATE_TO_LABEL == {
        "positive": 1,
        "negative": 0,
        "uncertain": None,
        "unknown": None,
    }
    assert policy.sources["public"]["provenance_status"] == (
        "external_benchmark_with_possible_gold_influence"
    )
    assert policy.sources["rules"]["artifact_identities"] == {
        "contracts/report-extractor-v1.json": {
            "sha256": "901e8a53173fef0d85e12ed93cbea05dd5b8cfc900ef111e0125cb751634156a",
            "size_bytes": 15896,
        },
        "report-labels-v1.csv": {
            "sha256": "6651f60b00e5bec292e8c98e7fa83c75e8aa5af5931414a9a68c8ce10a52d527",
            "size_bytes": 12409688,
        },
    }

    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from rsna_knee.combined_supervision import (
        PUBLIC_VERDICT_TO_LABEL as combined_public_map,
    )

    assert combined_public_map is PUBLIC_VERDICT_TO_LABEL


def test_public_mapping_gold_exclusion_and_provenance() -> None:
    labels = resolve_weak_labels(
        "public_direct_v1",
        gold_rows=(_gold_row("gold", [1] * len(TARGETS)), _gold_row("weak", [])),
        source_rows=(
            _public_row("gold", ["NO"] * len(TARGETS)),
            _public_row("weak", ["YES", "NO", "UNK"] + ["NO"] * 9),
        ),
    )

    by_key = {(label.study_uid, label.target): label for label in labels}
    assert len(labels) == len(TARGETS) * 2
    assert all(
        label.provenance == "gold" for label in labels if label.study_uid == "gold"
    )
    assert by_key[("gold", TARGETS[0])].value == 1
    assert by_key[("weak", TARGETS[0])].value == 1
    assert by_key[("weak", TARGETS[1])].value == 0
    assert by_key[("weak", TARGETS[2])].value is None
    assert by_key[("weak", TARGETS[2])].observed is False
    assert by_key[("weak", TARGETS[2])].source_state == "UNK"
    assert by_key[("weak", TARGETS[0])].provenance == "public_direct_v1"


def test_qwen_four_state_mapping_and_parse_failure_masking() -> None:
    states = ["positive", "negative", "uncertain", "unknown"] + ["positive"] * 8
    labels = resolve_weak_labels(
        "qwen_direct_v1",
        gold_rows=(_gold_row("weak", []),),
        source_rows=_qwen_rows("weak", states, parse_failed=True),
        qwen_status_rows=(_qwen_status("weak", True),),
    )

    by_target = {label.target: label for label in labels}
    assert by_target[TARGETS[0]].value is None
    assert by_target[TARGETS[0]].source_state == "positive"
    assert by_target[TARGETS[0]].observed is False
    assert by_target[TARGETS[1]].value is None
    assert by_target[TARGETS[2]].value is None
    assert by_target[TARGETS[3]].value is None

    non_failed = resolve_weak_labels(
        "qwen_direct_v1",
        source_rows=_qwen_rows("weak", states),
        qwen_status_rows=(_qwen_status("weak", False),),
    )
    assert [label.value for label in non_failed[:4]] == [1, 0, None, None]


def test_gold_only_has_no_weak_observations() -> None:
    labels = resolve_weak_labels(
        "gold_only",
        gold_rows=(_gold_row("gold", [1] * len(TARGETS)), _gold_row("weak", [])),
    )

    assert len(labels) == len(TARGETS)
    assert {label.provenance for label in labels} == {"gold"}
    assert all(label.observed for label in labels)


def test_direct_candidate_rejects_incomplete_non_gold_coverage() -> None:
    with pytest.raises(ValueError, match="coverage"):
        resolve_weak_labels(
            "public_direct_v1",
            gold_rows=(_gold_row("gold", [1] * len(TARGETS)), _gold_row("weak", [])),
            source_rows=(_public_row("gold", ["NO"] * len(TARGETS)),),
        )


def test_source_validation_rejects_duplicates_invalid_values_and_target_drift() -> None:
    public = _public_row("weak", ["YES"] * len(TARGETS))
    with pytest.raises(ValueError, match="(?i)duplicate study-target"):
        resolve_weak_labels("public_direct_v1", source_rows=(public, public))

    invalid = _public_row("weak", ["YES"] * len(TARGETS))
    invalid[f"{TARGETS[0]}__verdict"] = "MAYBE"
    with pytest.raises(ValueError, match="Invalid public verdict"):
        resolve_weak_labels("public_direct_v1", source_rows=(invalid,))

    qwen = list(_qwen_rows("weak", ["unknown"] * len(TARGETS)))
    qwen[0]["state"] = "maybe"
    with pytest.raises(ValueError, match="Invalid Qwen state"):
        resolve_weak_labels(
            "qwen_direct_v1",
            source_rows=tuple(qwen),
            qwen_status_rows=(_qwen_status("weak", False),),
        )

    drifted = list(_qwen_rows("weak", ["unknown"] * len(TARGETS)))
    drifted[0], drifted[1] = drifted[1], drifted[0]
    with pytest.raises(ValueError, match="target order"):
        resolve_weak_labels(
            "qwen_direct_v1",
            source_rows=tuple(drifted),
            qwen_status_rows=(_qwen_status("weak", False),),
        )


def test_qwen_parse_failure_identity_must_match_study_status() -> None:
    with pytest.raises(ValueError, match="parse_failed"):
        resolve_weak_labels(
            "qwen_direct_v1",
            source_rows=_qwen_rows("weak", ["positive"] * len(TARGETS)),
            qwen_status_rows=(_qwen_status("weak", True),),
        )


def test_partial_gold_target_pattern_is_rejected() -> None:
    with pytest.raises(ValueError, match="(?i)partially labeled"):
        classify_gold_membership(["1", "", *([""] * (len(TARGETS) - 2))])


def test_rules_and_unregistered_hybrids_are_not_candidates() -> None:
    for candidate in ("rules_direct_v1", "public_plus_qwen", ""):
        with pytest.raises(ValueError, match="candidate"):
            resolve_weak_labels(candidate, source_rows=())


def test_source_identity_verification_fails_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "labels.csv"
    artifact.write_text("synthetic\n", encoding="utf-8")
    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()

    assert verify_source_identity(artifact, expected) == expected
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_source_identity(artifact, "0" * 64)


def _gold_row(study_uid: str, labels: list[int]) -> dict[str, object]:
    row: dict[str, object] = {"StudyInstanceUID": study_uid, "Report": "synthetic"}
    for index, target in enumerate(TARGETS):
        row[target] = str(labels[index]) if labels else ""
    return row


def _public_row(study_uid: str, verdicts: list[str]) -> dict[str, object]:
    row: dict[str, object] = {"StudyInstanceUID": study_uid}
    for index, target in enumerate(TARGETS):
        row[target] = "0.5"
        row[f"{target}__conf"] = "0.9"
        row[f"{target}__verdict"] = verdicts[index]
    return row


def _qwen_rows(
    study_uid: str, states: list[str], *, parse_failed: bool = False
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for target, state in zip(TARGETS, states, strict=True):
        rows.append(
            {
                "StudyInstanceUID": study_uid,
                "target": target,
                "state": state,
                "extractor_version": "report-llm-qwen2.5-3b-v1",
                "model_revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
                "prompt_version": "rsna-knee-12target-fourstate-v1",
                "normalized_report_sha256": "0" * 64,
                "parse_failed": str(parse_failed).lower(),
                "input_truncated": "false",
                "provenance_key": "0" * 64,
            }
        )
    return tuple(rows)


def _qwen_status(study_uid: str, parse_failed: bool) -> dict[str, object]:
    return {
        "StudyInstanceUID": study_uid,
        "extractor_version": "report-llm-qwen2.5-3b-v1",
        "model_revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "prompt_version": "rsna-knee-12target-fourstate-v1",
        "normalized_report_sha256": "0" * 64,
        "parse_failed": str(parse_failed).lower(),
        "retry_count": "0",
        "input_truncated": "false",
        "checkpoint_sequence": "1",
    }
