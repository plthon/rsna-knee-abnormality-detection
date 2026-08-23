import base64
import hashlib
import json
import re
import zipfile
from io import BytesIO
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from rsna_knee.kaggle import (
    build_public_label_audit_bundle,
    pull_kaggle_output,
    push_kaggle_bundle,
)
from rsna_knee.public_label_audit import (
    audit_file_inventory,
    classify_provenance,
    compare_label_sources,
    decide_usability,
    public_safe_summary,
    resolve_dataset_root,
    summarize_coverage,
    validate_wide_labels,
)

TARGETS = ("ACL", "MCL")


def test_dataset_root_falls_back_to_one_attached_marker(tmp_path: Path) -> None:
    mounted = tmp_path / "datasets" / "pilkwang" / "rsna-knee-llm-labels"
    mounted.mkdir(parents=True)
    (mounted / "report_labels_v2.csv").touch()

    assert (
        resolve_dataset_root(
            (tmp_path / "rsna-knee-llm-labels", mounted),
            "report_labels_v2.csv",
        )
        == mounted
    )


def test_valid_wide_binary_labels_are_accepted() -> None:
    rows = [
        {"StudyInstanceUID": "study-1", "ACL": "1", "MCL": "0"},
        {"StudyInstanceUID": "study-2", "ACL": "0", "MCL": "1"},
    ]

    audit = validate_wide_labels(
        rows,
        ("StudyInstanceUID", *TARGETS),
        identifier_column="StudyInstanceUID",
        target_order=TARGETS,
        value_kind="binary",
    )

    assert audit.valid
    assert audit.row_count == audit.unique_study_count == 2
    assert audit.per_target["ACL"]["positive_rate"] == 0.5


@pytest.mark.parametrize(
    ("rows", "columns", "error"),
    [
        (
            [{"StudyInstanceUID": "study-1", "target": "ACL", "value": "1"}],
            ("StudyInstanceUID", "target", "value"),
            "schema_columns",
        ),
        (
            [{"StudyInstanceUID": "", "ACL": "1", "MCL": "0"}],
            ("StudyInstanceUID", *TARGETS),
            "missing_identifiers",
        ),
        (
            [
                {"StudyInstanceUID": "study-1", "ACL": "1", "MCL": "0"},
                {"StudyInstanceUID": "study-1", "ACL": "0", "MCL": "1"},
            ],
            ("StudyInstanceUID", *TARGETS),
            "duplicate_identifiers",
        ),
        (
            [{"StudyInstanceUID": "study-1", "ACL": "1", "MCL": "0"}],
            ("StudyInstanceUID", "MCL", "ACL"),
            "target_order",
        ),
        (
            [{"StudyInstanceUID": "study-1", "ACL": "1"}],
            ("StudyInstanceUID", "ACL"),
            "missing_targets",
        ),
    ],
)
def test_schema_and_identifier_failures_are_reported_without_study_ids(
    rows: list[dict[str, str]], columns: tuple[str, ...], error: str
) -> None:
    audit = validate_wide_labels(
        rows,
        columns,
        identifier_column="StudyInstanceUID",
        target_order=TARGETS,
        value_kind="binary",
    )

    assert not audit.valid
    assert error in audit.errors
    assert "study-1" not in json.dumps(audit.as_dict())


@pytest.mark.parametrize(
    ("value_kind", "invalid_value"),
    [("binary", "0.5"), ("probability", "1.01"), ("probability", "nan")],
)
def test_invalid_binary_and_probability_values_are_rejected(
    value_kind: str, invalid_value: str
) -> None:
    audit = validate_wide_labels(
        [{"StudyInstanceUID": "study-1", "ACL": invalid_value, "MCL": "0"}],
        ("StudyInstanceUID", *TARGETS),
        identifier_column="StudyInstanceUID",
        target_order=TARGETS,
        value_kind=value_kind,
    )

    assert not audit.valid
    assert audit.per_target["ACL"]["invalid_value_count"] == 1
    assert "invalid_values" in audit.errors


def test_auxiliary_probability_and_verdict_domains_are_validated() -> None:
    columns = (
        "StudyInstanceUID",
        "ACL",
        "ACL__conf",
        "ACL__verdict",
        "MCL",
        "MCL__conf",
        "MCL__verdict",
    )
    audit = validate_wide_labels(
        [
            {
                "StudyInstanceUID": "study-1",
                "ACL": "0.94",
                "ACL__conf": "0.95",
                "ACL__verdict": "YES",
                "MCL": "0.08",
                "MCL__conf": "2",
                "MCL__verdict": "MAYBE",
            }
        ],
        columns,
        identifier_column="StudyInstanceUID",
        target_order=TARGETS,
        value_kind="probability",
        auxiliary_domains={
            "__conf": {"kind": "probability"},
            "__verdict": {"kind": "enum", "values": ["YES", "NO", "UNK"]},
        },
    )

    assert not audit.valid
    assert audit.per_target["MCL"]["invalid_auxiliary_value_count"] == 2


@pytest.mark.parametrize(
    ("facts", "expected"),
    [
        (
            {
                "identity_frozen": True,
                "structurally_valid": True,
                "contract_compatible": True,
                "semantics_unambiguous": True,
                "provenance": "documented",
                "coverage_complete": True,
                "restrictions": (),
            },
            "usable_directly",
        ),
        (
            {
                "identity_frozen": True,
                "structurally_valid": True,
                "contract_compatible": True,
                "semantics_unambiguous": True,
                "provenance": "partial",
                "coverage_complete": False,
                "restrictions": ("one study is missing",),
            },
            "usable_with_restrictions",
        ),
        (
            {
                "identity_frozen": False,
                "structurally_valid": True,
                "contract_compatible": True,
                "semantics_unambiguous": True,
                "provenance": "documented",
                "coverage_complete": True,
                "restrictions": (),
            },
            "insufficiently_documented",
        ),
    ],
)
def test_each_final_decision_state(facts: dict[str, object], expected: str) -> None:
    assert decide_usability(**facts) == expected


def test_decisive_overlap_comparison_excludes_unknown_and_uncertain() -> None:
    public_rows = [
        {"StudyInstanceUID": "s1", "ACL": "1", "MCL": "0"},
        {"StudyInstanceUID": "s2", "ACL": "0", "MCL": "1"},
        {"StudyInstanceUID": "s3", "ACL": "1", "MCL": "0"},
        {"StudyInstanceUID": "s4", "ACL": "0", "MCL": "1"},
    ]
    report_rows = [
        {"StudyInstanceUID": "s1", "target": "ACL", "state": "positive"},
        {"StudyInstanceUID": "s2", "target": "ACL", "state": "positive"},
        {"StudyInstanceUID": "s3", "target": "ACL", "state": "negative"},
        {"StudyInstanceUID": "s4", "target": "ACL", "state": "uncertain"},
        {"StudyInstanceUID": "s1", "target": "MCL", "state": "unknown"},
    ]

    comparison = compare_label_sources(
        public_rows,
        report_rows,
        identifier_column="StudyInstanceUID",
        target_order=TARGETS,
    )

    acl = comparison["ACL"]
    assert acl["overlap_row_count"] == 4
    assert acl["agreement_rate"] == pytest.approx(1 / 3)
    assert acl["public_positive_report_negative_conflicts"] == 1
    assert acl["public_negative_report_positive_conflicts"] == 1
    assert acl["excluded_report_uncertain"] == 1
    assert comparison["MCL"]["excluded_report_unknown"] == 1


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        ({}, "opaque"),
        ({"model_family": True}, "partial"),
        (
            {
                "model_family": True,
                "model_version": True,
                "prompt_method": True,
                "output_semantics": True,
                "language_handling": True,
            },
            "documented",
        ),
    ],
)
def test_provenance_classification(evidence: dict[str, bool], expected: str) -> None:
    assert classify_provenance(evidence) == expected


def test_coverage_uses_only_label_missingness_and_emits_counts() -> None:
    competition_rows: list[dict[str, object]] = [
        {"StudyInstanceUID": "s1", "ACL": "secret", "MCL": "secret"},
        {"StudyInstanceUID": "s2", "ACL": "", "MCL": ""},
        {"StudyInstanceUID": "s3", "ACL": float("nan"), "MCL": float("nan")},
    ]

    coverage = summarize_coverage(
        {"s1", "s2", "outside"},
        competition_rows,
        identifier_column="StudyInstanceUID",
        target_order=TARGETS,
    )

    assert coverage == {
        "competition_study_count": 3,
        "gold_study_count": 1,
        "non_gold_study_count": 2,
        "partially_labeled_study_count": 0,
        "covered_study_count": 3,
        "competition_overlap_count": 2,
        "gold_overlap_count": 1,
        "non_gold_overlap_count": 1,
        "missing_competition_study_count": 1,
        "extra_public_study_count": 1,
    }
    assert "secret" not in json.dumps(coverage)
    assert "s1" not in json.dumps(coverage)


def test_pandas_nan_identifier_is_missing() -> None:
    audit = validate_wide_labels(
        [{"StudyInstanceUID": float("nan"), "ACL": "1", "MCL": "0"}],
        ("StudyInstanceUID", *TARGETS),
        identifier_column="StudyInstanceUID",
        target_order=TARGETS,
        value_kind="binary",
    )

    assert audit.missing_identifier_count == 1
    assert "missing_identifiers" in audit.errors


def test_file_inventory_requires_exact_names_sizes_and_hashes(tmp_path: Path) -> None:
    first = tmp_path / "labels.csv"
    second = tmp_path / "README.txt"
    first.write_bytes(b"labels\n")
    second.write_bytes(b"documentation\n")
    expected = [
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in (first, second)
    ]

    inventory, errors = audit_file_inventory(tmp_path, expected)

    assert errors == ()
    assert inventory == tuple(expected)

    second.write_text("drifted\n")
    _, errors = audit_file_inventory(tmp_path, expected)
    assert errors == ("file_identity",)


def test_public_safe_summary_rejects_raw_reports_and_identifiers() -> None:
    summary = public_safe_summary(
        dataset_identity={"owner": "pilkwang", "slug": "labels", "version": 1},
        schema_summary={"row_count": 4406},
        coverage_summary={"competition_overlap_count": 4406},
        provenance="partial",
        comparison_summary={"ACL": {"agreement_rate": 0.5}},
        decision="usable_with_restrictions",
        justification=["One study is missing."],
    )
    assert "StudyInstanceUID" not in json.dumps(summary)
    assert "Report" not in json.dumps(summary)

    with pytest.raises(ValueError, match="private row-level"):
        public_safe_summary(
            dataset_identity={"owner": "pilkwang"},
            schema_summary={"raw_report_text": "private text"},
            coverage_summary={},
            provenance="partial",
            comparison_summary={},
            decision="usable_with_restrictions",
            justification=[],
        )


def test_frozen_contract_matches_competition_targets_and_public_files() -> None:
    root = Path(__file__).parents[1]
    competition = json.loads((root / "contracts" / "competition.json").read_text())
    contract = json.loads(
        (root / "contracts" / "public-report-label-dataset-v1.json").read_text()
    )

    assert contract["audit_version"] == "public-report-label-dataset-audit-v1"
    assert contract["dataset_owner"] == "pilkwang"
    assert contract["dataset_slug"] == "rsna-knee-llm-labels"
    assert contract["dataset_version"] == 1
    assert contract["target_order"] == competition["targets"]["order"]
    assert contract["dataset_documentation"]["description_sha256"] == (
        "867ad502b0460033f4f18dc40a3ee83fa85ffade34e8d53e2b34f5e700ddaca5"
    )
    assert contract["file_hashes"] == {
        "api_labeler.py": (
            "045f0f858b20ee477869c999c4bc9b484154cb5ddfd507b79e7c732632abc52e"
        ),
        "report_labels_v2.csv": (
            "6f704a7bdb2f894cc49445b19ba7c4378c3f548d3449e00361e10044bee40920"
        ),
    }


def test_public_label_audit_bundle_is_private_offline_cpu_and_frozen(
    tmp_path: Path,
) -> None:
    project = Path(__file__).parents[1]

    bundle = build_public_label_audit_bundle(
        project, tmp_path / "public-label-audit", stage="audit"
    )
    metadata = json.loads(bundle.kernel_metadata_path.read_text())
    notebook = json.loads(bundle.notebook_path.read_text())
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert metadata["competition_sources"] == ["rsna-knee-abnormality-detection"]
    assert metadata["dataset_sources"] == ["pilkwang/rsna-knee-llm-labels"]
    assert metadata["kernel_sources"] == ["plthon/rsna-knee-report-labels"]
    assert metadata["enable_gpu"] == "false"
    assert metadata["enable_internet"] == "false"
    assert metadata["is_private"] == "true"
    assert 'STAGE = "audit"' in source
    assert 'usecols=["StudyInstanceUID", *target_order]' in source
    assert "__RSNA_KNEE_" not in source

    archive_marker = "__RSNA_KNEE_PACKAGE_ARCHIVE__"
    assert archive_marker not in source
    archive_match = re.search(r'base64\.b64decode\("([A-Za-z0-9+/=]+)"\)', source)
    assert archive_match is not None
    archive_bytes = base64.b64decode(archive_match.group(1))
    with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
        assert "rsna_knee/public-report-label-dataset-v1.json" in archive.namelist()

    calls: list[list[str]] = []

    def runner(command: list[str]) -> CompletedProcess[str]:
        calls.append(command)
        return CompletedProcess(command, 0)

    with pytest.raises(PermissionError, match="explicit remote approval"):
        push_kaggle_bundle(bundle, runner=runner)
    assert calls == []


def test_public_label_audit_boundary_rejects_wrong_stage_and_private_pull(
    tmp_path: Path,
) -> None:
    project = Path(__file__).parents[1]
    with pytest.raises(ValueError, match="stage must be audit"):
        build_public_label_audit_bundle(
            project, tmp_path / "wrong-stage", stage="extract"
        )
    assert not (tmp_path / "wrong-stage").exists()

    bundle = build_public_label_audit_bundle(
        project, tmp_path / "public-label-audit", stage="audit"
    )
    with pytest.raises(PermissionError, match="row-level"):
        pull_kaggle_output(
            bundle,
            tmp_path / ".artifacts" / "audit",
            artifact_root=tmp_path / ".artifacts",
            filenames=("public-label-row-validation.csv",),
            runner=lambda command: CompletedProcess(command, 0),
        )
