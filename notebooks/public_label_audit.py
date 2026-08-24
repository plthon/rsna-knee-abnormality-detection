# %% [markdown]
# # RSNA Knee Public Label Dataset Audit
#
# Private, CPU-only, internet-disabled audit of the frozen public label dataset.
# Competition labels are observed only through missingness to identify the 58
# organizer-gold studies. Outputs contain aggregate evidence only.

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
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]

os.environ["CUDA_VISIBLE_DEVICES"] = ""
STAGE = "__RSNA_KNEE_PUBLIC_LABEL_STAGE__"
if STAGE != "audit":
    raise ValueError(f"Unsupported public-label stage: {STAGE}")

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
WORK_ROOT = Path("/kaggle/working")
PACKAGE_ARCHIVE = WORK_ROOT / "rsna_knee_source.zip"
PACKAGE_ARCHIVE.write_bytes(base64.b64decode("__RSNA_KNEE_PACKAGE_ARCHIVE__"))
sys.path.insert(0, str(PACKAGE_ARCHIVE))

# %%
from rsna_knee.public_label_audit import (
    audit_file_inventory,
    classify_provenance,
    compare_label_sources,
    decide_usability,
    public_safe_summary,
    resolve_dataset_root,
    resolve_private_kernel_root,
    summarize_coverage,
    validate_wide_labels,
)

started = time.monotonic()
PUBLIC_DATASET_ROOT = resolve_dataset_root(PUBLIC_DATASET_ROOTS, "report_labels_v2.csv")
PRIVATE_REPORT_LABEL_PATH = (
    resolve_private_kernel_root(
        PRIVATE_REPORT_LABEL_ROOTS,
        "report-labels-v1.csv",
    )
    / "report-labels-v1.csv"
)
contract_bytes = (
    files("rsna_knee").joinpath("public-report-label-dataset-v1.json").read_bytes()
)
contract = cast(dict[str, Any], json.loads(contract_bytes))
identifier_column = cast(str, contract["expected_identifier_column"])
target_order = cast(list[str], contract["target_order"])
value_contract = cast(dict[str, Any], contract["value_contract"])
primary_contract = cast(dict[str, object], value_contract["primary"])
auxiliary_domains = cast(
    dict[str, dict[str, object]], value_contract["auxiliary_domains"]
)

expected_inventory = cast(list[dict[str, object]], contract["file_inventory"])
file_inventory, file_errors = audit_file_inventory(
    PUBLIC_DATASET_ROOT, expected_inventory
)
if file_errors:
    raise RuntimeError(f"Frozen public dataset identity mismatch: {file_errors}")

# %%
public_label_path = PUBLIC_DATASET_ROOT / "report_labels_v2.csv"
public_frame = pd.read_csv(public_label_path, dtype={identifier_column: str})
public_rows = cast(list[dict[str, object]], public_frame.to_dict(orient="records"))
label_audit = validate_wide_labels(
    public_rows,
    list(public_frame.columns),
    identifier_column=identifier_column,
    target_order=target_order,
    value_kind=cast(str, primary_contract["kind"]),
    auxiliary_domains=auxiliary_domains,
)
if not label_audit.valid:
    raise RuntimeError(f"Public label schema is invalid: {label_audit.errors}")

# Read no Report column and inspect organizer target columns only for missingness.
competition_frame = pd.read_csv(
    COMPETITION_ROOT / "train.csv",
    usecols=["StudyInstanceUID", *target_order],
    dtype={"StudyInstanceUID": str},
)
competition_rows = cast(
    list[dict[str, object]], competition_frame.to_dict(orient="records")
)
public_study_ids = set(public_frame[identifier_column].astype(str))
coverage = summarize_coverage(
    public_study_ids,
    competition_rows,
    identifier_column=identifier_column,
    target_order=target_order,
)

# %%
comparison_input = cast(
    dict[str, object],
    cast(dict[str, Any], contract["comparison_inputs"])["private_report_labels"],
)
if not PRIVATE_REPORT_LABEL_PATH.is_file():
    raise FileNotFoundError("Frozen private #20 report-label artifact is missing")
private_size = PRIVATE_REPORT_LABEL_PATH.stat().st_size
private_sha256 = hashlib.sha256(PRIVATE_REPORT_LABEL_PATH.read_bytes()).hexdigest()
if (
    private_size != comparison_input["size_bytes"]
    or private_sha256 != comparison_input["sha256"]
):
    raise RuntimeError("Frozen private #20 report-label artifact identity mismatch")
report_frame = pd.read_csv(
    PRIVATE_REPORT_LABEL_PATH,
    usecols=["StudyInstanceUID", "target", "state"],
    dtype={"StudyInstanceUID": str, "target": str, "state": str},
)
report_rows = cast(list[dict[str, object]], report_frame.to_dict(orient="records"))
comparison = compare_label_sources(
    public_rows,
    report_rows,
    identifier_column=identifier_column,
    target_order=target_order,
)

# %%
provenance_record = cast(dict[str, object], contract["provenance_documentation"])
provenance_evidence = {
    name: bool(provenance_record[name])
    for name in (
        "model_family",
        "model_version",
        "prompt_method",
        "output_semantics",
        "language_handling",
    )
}
provenance = classify_provenance(provenance_evidence)
if provenance != provenance_record["classification"]:
    raise RuntimeError("Frozen provenance classification is inconsistent")

coverage_complete = (
    coverage["competition_overlap_count"] == coverage["competition_study_count"]
    and coverage["extra_public_study_count"] == 0
)
restrictions = []
if not coverage_complete:
    restrictions.append("Public labels do not cover the exact competition study set.")
if provenance != "documented":
    restrictions.append("The exact generator model/API version is not frozen.")
if bool(provenance_record["gold_label_evaluation_or_tuning_disclosed"]):
    restrictions.append(
        "Organizer annotations influenced reader evaluation or selection; later "
        "evaluation must isolate that influence."
    )
restrictions.append(
    "Primary values are rank-usable scores, not calibrated probabilities; retain "
    "the explicit YES/NO/UNK semantics."
)
decision = decide_usability(
    identity_frozen=not file_errors,
    structurally_valid=label_audit.valid,
    contract_compatible=label_audit.target_order_matches,
    semantics_unambiguous=True,
    provenance=provenance,
    coverage_complete=coverage_complete,
    restrictions=tuple(restrictions),
)

# %%
dataset_identity = {
    "title": contract["dataset_title"],
    "owner": contract["dataset_owner"],
    "slug": contract["dataset_slug"],
    "version": contract["dataset_version"],
    "url": contract["dataset_url"],
    "license": contract["license"],
    "files": list(file_inventory),
}
schema_summary = label_audit.as_dict()
comparison_summary = {
    "schema_version": 1,
    "audit_version": contract["audit_version"],
    "per_target": comparison,
}
summary = public_safe_summary(
    dataset_identity=dataset_identity,
    schema_summary=schema_summary,
    coverage_summary=coverage,
    provenance=provenance,
    comparison_summary=comparison_summary,
    decision=decision,
    justification=restrictions,
)
manifest = {
    "schema_version": 1,
    "audit_version": contract["audit_version"],
    "dataset_identity": dataset_identity,
    "schema_summary": schema_summary,
    "coverage_summary": coverage,
    "provenance_classification": provenance,
    "comparison_inputs": {
        "private_report_labels": {
            "filename": PRIVATE_REPORT_LABEL_PATH.name,
            "size_bytes": private_size,
            "sha256": private_sha256,
            "row_count": len(report_frame),
        },
        "competition_contract_sha256": cast(
            dict[str, Any], contract["comparison_inputs"]
        )["competition_contract"]["sha256"],
    },
    "runtime": {
        "seconds": time.monotonic() - started,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": importlib.metadata.version("pandas"),
        "cpu_only": True,
        "internet_enabled": False,
    },
    "source": {
        "git_commit": "__RSNA_KNEE_GIT_COMMIT__",
        "git_dirty": json.loads("__RSNA_KNEE_GIT_DIRTY__"),
        "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "notebook_source_sha256": "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__",
        "rendered_notebook_sha256": "__RSNA_KNEE_RENDERED_NOTEBOOK_SHA256__",
        "package_archive_sha256": hashlib.sha256(
            PACKAGE_ARCHIVE.read_bytes()
        ).hexdigest(),
    },
    "audit_errors": [*file_errors, *label_audit.errors],
}

outputs = {
    "public-label-audit-manifest.json": manifest,
    "public-label-comparison-summary.json": comparison_summary,
    "public-label-audit-summary.json": summary,
}
for filename, payload in outputs.items():
    path = WORK_ROOT / filename
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(path)
