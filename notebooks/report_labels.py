# %% [markdown]
# # RSNA Knee Report Labels
#
# Private, CPU-only, internet-disabled deterministic report audit and extraction.
# The audit exports aggregate counts only; row-level extraction artifacts remain
# private Kaggle outputs.

# %%
import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
STAGE = "__RSNA_KNEE_REPORT_STAGE__"
COMPETITION_ROOT = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
WORK_ROOT = Path("/kaggle/working")
PACKAGE_ARCHIVE = WORK_ROOT / "rsna_knee_source.zip"
PACKAGE_ARCHIVE.write_bytes(base64.b64decode("__RSNA_KNEE_PACKAGE_ARCHIVE__"))
sys.path.insert(0, str(PACKAGE_ARCHIVE))

# %%
from rsna_knee.competition_contract import load_verified_submission_contract
from rsna_knee.reports import audit_report_corpus, read_report_csv

started = time.monotonic()
competition_contract = load_verified_submission_contract()
report_rows = read_report_csv(
    COMPETITION_ROOT / "train.csv",
    requested_columns=("StudyInstanceUID", "Report"),
    target_names=competition_contract.target_names,
)

# %%
if STAGE == "audit":
    audit = audit_report_corpus(
        report_rows,
        competition_contract.target_names,
        expected_study_count=4407,
        source_hashes={
            "competition_contract_sha256": competition_contract.sha256,
            "git_commit": "__RSNA_KNEE_GIT_COMMIT__",
            "git_dirty": json.loads("__RSNA_KNEE_GIT_DIRTY__"),
            "jupytext_source_sha256": "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__",
            "package_archive_sha256": hashlib.sha256(
                PACKAGE_ARCHIVE.read_bytes()
            ).hexdigest(),
        },
    )
    audit_path = WORK_ROOT / "report-audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "stage": "audit",
        "runtime_seconds": time.monotonic() - started,
        "artifacts": {
            audit_path.name: {
                "size_bytes": audit_path.stat().st_size,
                "sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            }
        },
    }
    manifest_path = WORK_ROOT / "report-audit-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(audit_path)
    print(manifest_path)
elif STAGE == "extract":
    raise RuntimeError(
        "The extract stage requires the rule contract frozen from report-audit.json"
    )
else:
    raise ValueError(f"Unsupported report stage: {STAGE}")
