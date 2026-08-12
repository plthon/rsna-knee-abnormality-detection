# %% [markdown]
# # RSNA Knee Gold Supervision Audit
#
# Private, CPU-only, internet-disabled organizer-gold audit for issue #7. It
# consumes the accepted private study index, does not process or export report
# text, reads no DICOM pixels, and prints only the aggregate/digest summary. Row-level labels,
# PatientIDs, cohort membership, exclusions, and assignments remain in the
# private Kaggle output artifact.

# %%
import base64
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
package_archive = Path("/kaggle/working/rsna_knee_source.zip")
package_archive.write_bytes(base64.b64decode("__RSNA_KNEE_PACKAGE_ARCHIVE__"))
sys.path.insert(0, str(package_archive))

# %%
from rsna_knee import (
    audit_gold_supervision,
    discover_gold_audit_inputs,
    verify_gold_audit_artifact,
)

inputs = discover_gold_audit_inputs(Path("/kaggle/input"))
outcome = audit_gold_supervision(
    competition_root=inputs.competition_root,
    index_artifact_dir=inputs.index_artifact_dir,
    work_root=Path("/kaggle/working"),
    source_provenance={
        "git_commit": "__RSNA_KNEE_GIT_COMMIT__",
        "git_dirty": json.loads("__RSNA_KNEE_GIT_DIRTY__"),
        "jupytext_source_sha256": "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__",
        "package_archive_sha256": hashlib.sha256(
            package_archive.read_bytes()
        ).hexdigest(),
    },
)
verified_summary = verify_gold_audit_artifact(outcome.artifact_dir)

# %%
print(outcome.manifest_path)
print(json.dumps(verified_summary, indent=2, sort_keys=True))
