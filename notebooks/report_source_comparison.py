# %% [markdown]
# # RSNA Knee Report Source Comparison
#
# Private, CPU-only, internet-disabled three-way comparison of the frozen #20
# rule labels, the frozen #23 local-LLM labels, and the #24-audited public
# label dataset over the 4,349-study non-gold cohort. Organizer-gold target
# columns are observed only through missingness. Outputs are aggregate
# evidence only; no correctness claim is made and no source is selected.

# %%
import base64
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from importlib.resources import files
from pathlib import Path
from typing import cast

import pandas as pd  # type: ignore[import-untyped]

os.environ["CUDA_VISIBLE_DEVICES"] = ""
STAGE = "__RSNA_KNEE_REPORT_SOURCE_STAGE__"
if STAGE != "compare":
    raise ValueError(f"Unsupported report source comparison stage: {STAGE}")

COMPETITION_ROOT = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
PUBLIC_DATASET_ROOTS = (
    Path("/kaggle/input/datasets/pilkwang/rsna-knee-llm-labels"),
    Path("/kaggle/input/rsna-knee-llm-labels"),
)
PRIVATE_REPORT_LABEL_ROOTS = (
    Path("/kaggle/input/kernels/plthon/rsna-knee-report-labels"),
    Path("/kaggle/input/plthon/rsna-knee-report-labels"),
    Path("/kaggle/input/rsna-knee-report-labels"),
)
PRIVATE_LLM_LABEL_ROOTS = (
    Path("/kaggle/input/kernels/plthon/rsna-knee-report-llm-labels"),
    Path("/kaggle/input/plthon/rsna-knee-report-llm-labels"),
    Path("/kaggle/input/rsna-knee-report-llm-labels"),
)
WORK_ROOT = Path("/kaggle/working")
PACKAGE_ARCHIVE = WORK_ROOT / "rsna_knee_source.zip"
PACKAGE_ARCHIVE.write_bytes(base64.b64decode("__RSNA_KNEE_PACKAGE_ARCHIVE__"))
sys.path.insert(0, str(PACKAGE_ARCHIVE))
started = time.monotonic()

# %%
from rsna_knee.competition_contract import load_verified_submission_contract
from rsna_knee.public_label_audit import (
    resolve_dataset_root,
    resolve_private_kernel_root,
)
from rsna_knee.report_source_comparison import (
    EXPECTED_STUDY_COUNT,
    GOLD_STUDY_COUNT,
    NON_GOLD_STUDY_COUNT,
    ComparisonInputs,
    build_public_safe_summary,
    build_quota_feasibility,
    compare_pairwise,
    compute_deep_dives,
    count_any_uncertain,
    count_triple_combinations,
    ingest_public_labels,
    ingest_report_labels,
    join_three_sources,
    load_comparison_contract,
    partition_strata,
    partition_strata_by_detail,
    pinned_input,
    read_competition_cohorts,
    summarize_public_descriptive_domains,
    verify_artifact_identity,
)

competition_contract = load_verified_submission_contract()
target_order = competition_contract.target_names
contract_bytes = (
    files("rsna_knee").joinpath("report-source-comparison-v1.json").read_bytes()
)
contract = load_comparison_contract(
    None,
    expected_target_order=target_order,
)
comp_pinned = pinned_input(contract, role="competition_contract")
if competition_contract.sha256 != str(comp_pinned["sha256"]):
    raise RuntimeError(
        f"Competition contract SHA mismatch against pinned comparison contract: "
        f"{competition_contract.sha256} != {comp_pinned['sha256']}"
    )

# %%
# Verify every frozen input identity before any computation. Any mismatch
# stops the kernel; a substituted input is never acceptable.
rule_record = pinned_input(contract, role="rule_labels")
RULE_LABEL_PATH = resolve_private_kernel_root(
    PRIVATE_REPORT_LABEL_ROOTS, str(rule_record["filename"])
) / str(rule_record["filename"])
llm_label_record = pinned_input(contract, role="llm_labels")
LLM_LABEL_PATH = resolve_private_kernel_root(
    PRIVATE_LLM_LABEL_ROOTS, str(llm_label_record["filename"])
) / str(llm_label_record["filename"])
llm_status_record = pinned_input(contract, role="llm_study_status")
LLM_STATUS_PATH = resolve_private_kernel_root(
    PRIVATE_LLM_LABEL_ROOTS, str(llm_status_record["filename"])
) / str(llm_status_record["filename"])
public_record = pinned_input(contract, role="public_labels")
PUBLIC_DATASET_ROOT = resolve_dataset_root(
    PUBLIC_DATASET_ROOTS, str(public_record["filename"])
)
PUBLIC_LABEL_PATH = PUBLIC_DATASET_ROOT / str(public_record["filename"])

verified_inputs = [
    verify_artifact_identity(
        "rule_labels",
        RULE_LABEL_PATH,
        expected_size_bytes=int(str(rule_record["size_bytes"])),
        expected_sha256=str(rule_record["sha256"]),
    ),
    verify_artifact_identity(
        "llm_labels",
        LLM_LABEL_PATH,
        expected_size_bytes=int(str(llm_label_record["size_bytes"])),
        expected_sha256=str(llm_label_record["sha256"]),
    ),
    verify_artifact_identity(
        "llm_study_status",
        LLM_STATUS_PATH,
        expected_size_bytes=int(str(llm_status_record["size_bytes"])),
        expected_sha256=str(llm_status_record["sha256"]),
    ),
    verify_artifact_identity(
        "public_labels",
        PUBLIC_LABEL_PATH,
        expected_size_bytes=int(str(public_record["size_bytes"])),
        expected_sha256=str(public_record["sha256"]),
    ),
]
raw_contract_inputs = contract.get("inputs")
assert isinstance(raw_contract_inputs, list)
for record in raw_contract_inputs:
    assert isinstance(record, dict)
    if record.get("kind") != "contract":
        continue
    name = str(record["path"]).split("/")[-1]
    contract_file = files("rsna_knee").joinpath(name)
    file_bytes = contract_file.read_bytes()
    file_sha256 = hashlib.sha256(file_bytes).hexdigest()
    if file_sha256 != str(record["sha256"]):
        raise RuntimeError(
            f"Frozen contract identity mismatch for {record['role']}: "
            f"{file_sha256} != {record['sha256']}"
        )
    verified_inputs.append(
        {
            "role": str(record["role"]),
            "filename": name,
            "size_bytes": len(file_bytes),
            "sha256": file_sha256,
        }
    )

# %%
# Cohort membership reads organizer target-column missingness only; the
# Report column and every target value are never parsed.
cohorts = read_competition_cohorts(
    COMPETITION_ROOT / "train.csv", target_names=target_order
)
if cohorts.non_gold_study_count != NON_GOLD_STUDY_COUNT:
    raise RuntimeError(
        f"Non-gold cohort drifted: {cohorts.non_gold_study_count} "
        f"!= {NON_GOLD_STUDY_COUNT}"
    )
if cohorts.gold_study_count != GOLD_STUDY_COUNT:
    raise RuntimeError(f"Gold cohort drifted: {cohorts.gold_study_count}")

# %%
rule_frame = pd.read_csv(
    RULE_LABEL_PATH,
    usecols=["StudyInstanceUID", "target", "state"],
    dtype={"StudyInstanceUID": str, "target": str, "state": str},
)
rule_states, _ = ingest_report_labels(
    cast(list[dict[str, object]], rule_frame.to_dict(orient="records"))
)
llm_frame = pd.read_csv(
    LLM_LABEL_PATH,
    usecols=["StudyInstanceUID", "target", "state", "parse_failed"],
    dtype={
        "StudyInstanceUID": str,
        "target": str,
        "state": str,
        "parse_failed": str,
    },
)
llm_states, llm_parse_failed = ingest_report_labels(
    cast(list[dict[str, object]], llm_frame.to_dict(orient="records"))
)
status_frame = pd.read_csv(
    LLM_STATUS_PATH,
    usecols=["StudyInstanceUID", "parse_failed"],
    dtype={"StudyInstanceUID": str, "parse_failed": str},
)
parse_failed_studies = frozenset(
    str(row["StudyInstanceUID"]).strip()
    for row in cast(list[dict[str, object]], status_frame.to_dict(orient="records"))
    if str(row["parse_failed"]).strip().lower() == "true"
)
for (study_uid, target), failed in llm_parse_failed.items():
    expected_failed = study_uid in parse_failed_studies
    if failed != expected_failed:
        raise RuntimeError(
            f"Parse-failure flag mismatch for study {study_uid}, target {target}: "
            f"row-level={failed}, study-status={expected_failed}"
        )
    if failed and llm_states[(study_uid, target)] != "unknown":
        raise RuntimeError(
            f"Parse-failed study {study_uid}, target {target} must emit unknown state, "
            f"got {llm_states[(study_uid, target)]!r}"
        )
public_frame = pd.read_csv(
    PUBLIC_LABEL_PATH,
    dtype={"StudyInstanceUID": str},
)
public_rows = cast(list[dict[str, object]], public_frame.to_dict(orient="records"))
public_states = ingest_public_labels(
    public_rows,
    target_order=target_order,
)
public_descriptive_frequencies = summarize_public_descriptive_domains(
    public_rows,
    target_order=target_order,
)

# %%
inputs = ComparisonInputs(
    rules=rule_states,
    llm=llm_states,
    llm_parse_failed=llm_parse_failed,
    public=public_states,
)
joined = join_three_sources(
    inputs, target_order=target_order, cohort_uids=cohorts.non_gold
)
expected_rows = NON_GOLD_STUDY_COUNT * len(target_order)
if len(joined.triples) != expected_rows:
    raise RuntimeError(
        f"Cohort join produced {len(joined.triples)} rows, expected {expected_rows}"
    )
if joined.complete_triple_count != expected_rows:
    raise RuntimeError(
        f"Cohort join completeness violated: {joined.complete_triple_count} "
        f"complete triples != expected {expected_rows}"
    )
if joined.public_missing_count != 0:
    raise RuntimeError(
        f"Unexpected missing public non-gold studies: {joined.public_missing_count}"
    )
complete_rate = joined.complete_triple_count / expected_rows

# %%
per_target_strata: dict[str, object] = {}
pairwise_flat: dict[str, object] = {}
for target in target_order:
    target_triples = joined.for_target(target)
    per_target_strata[target] = {
        "row_count": len(target_triples),
        "full_triple_table": count_triple_combinations(target_triples),
        "rollup_strata": partition_strata(target_triples),
        "stratum_details": partition_strata_by_detail(target_triples),
        "any_uncertain_count": count_any_uncertain(target_triples),
    }
pairwise = compare_pairwise(joined.triples)
for (first, second), metrics in pairwise.items():
    pairwise_flat[f"{first}|{second}"] = metrics
deep_dives = compute_deep_dives(
    joined,
    target_order=target_order,
    parse_failed_keys={
        (study_uid, target)
        for study_uid in parse_failed_studies
        for target in target_order
    },
)
deep_dives["public_descriptive_frequencies"] = public_descriptive_frequencies
quota_feasibility = build_quota_feasibility(joined.triples, target_order=target_order)

# %%
completeness = {
    "rule_label_row_count": int(rule_frame.shape[0]),
    "llm_label_row_count": int(llm_frame.shape[0]),
    "llm_study_status_row_count": int(status_frame.shape[0]),
    "public_label_row_count": int(public_frame.shape[0]),
    "analysis_row_count": len(joined.triples),
    "complete_triple_row_count": joined.complete_triple_count,
    "complete_triple_rate": complete_rate,
}
coverage = {
    "total_study_count": EXPECTED_STUDY_COUNT,
    "gold_study_count": cohorts.gold_study_count,
    "non_gold_study_count": cohorts.non_gold_study_count,
    "gold_cohort_line_membership_only": (
        "the 58-study gold cohort contributes no values to any table"
    ),
    "public_absent_non_gold_study_count": sum(
        1
        for triple in joined.triples
        if triple.public is None and triple.target == target_order[0]
    ),
}
summary = build_public_safe_summary(
    comparison_version=str(contract["comparison_version"]),
    input_identities=verified_inputs,
    completeness=completeness,
    coverage=coverage,
    per_target_strata=per_target_strata,
    pairwise=pairwise_flat,
    deep_dives=deep_dives,
    quota_feasibility=quota_feasibility,
)

# %%
manifest = {
    "schema_version": 1,
    "comparison_version": str(contract["comparison_version"]),
    "issue": contract["issue"],
    "policy": contract["policy"],
    "git": {
        "commit": "__RSNA_KNEE_GIT_COMMIT__",
        "dirty": json.loads("__RSNA_KNEE_GIT_DIRTY__"),
    },
    "notebook": {
        "source_sha256": "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__",
        "rendered_notebook_sha256": "__RSNA_KNEE_RENDERED_NOTEBOOK_SHA256__",
    },
    "package_archive_sha256": hashlib.sha256(PACKAGE_ARCHIVE.read_bytes()).hexdigest(),
    "comparison_contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
    "competition_contract_sha256": competition_contract.sha256,
    "input_identities": verified_inputs,
    "completeness": completeness,
    "cohort_coverage": {
        key: value for key, value in coverage.items() if not isinstance(value, str)
    },
    "runtime": {
        "seconds": time.monotonic() - started,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": importlib.metadata.version("pandas"),
        "cpu_only": True,
        "internet_enabled": False,
        "gpu_used": False,
    },
    "row_count_checks": {
        "rule_labels_expected": int(str(rule_record["row_count"])),
        "llm_labels_expected": int(str(llm_label_record["row_count"])),
        "public_labels_expected": int(str(public_record["row_count"])),
    },
    "errors": [],
}

# %%
outputs = {
    "source-comparison-manifest.json": manifest,
    "source-comparison-tables.json": {
        "schema_version": 1,
        "comparison_version": str(contract["comparison_version"]),
        "per_target_strata": per_target_strata,
        "pairwise_joint_decisive": pairwise_flat,
        "deep_dives": deep_dives,
        "quota_feasibility": quota_feasibility,
        "completeness": completeness,
        "parse_failed_study_count": len(parse_failed_studies),
    },
    "source-comparison-summary.json": summary,
}
for filename, payload in outputs.items():
    path = WORK_ROOT / filename
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(path)
print(f"Completed {STAGE}: {len(joined.triples)} non-gold study-target rows")
