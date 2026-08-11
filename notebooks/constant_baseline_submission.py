# %% [markdown]
# # RSNA Knee Constant Baseline Submission
#
# Private, CPU-only, internet-disabled verification of the complete competition
# submission path. The package discovers the current test studies, consumes the
# verified target contract, writes a constant prediction for every study, and
# validates the exact CSV before recording its compact artifacts and manifest.

# %%
import base64
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
package_archive = Path("/kaggle/working/rsna_knee_source.zip")
package_archive_bytes = base64.b64decode("__RSNA_KNEE_PACKAGE_ARCHIVE__")
package_archive.write_bytes(package_archive_bytes)
sys.path.insert(0, str(package_archive))

# %%
from rsna_knee import run_constant_baseline
from rsna_knee.provenance import create_run_id

kernel_run_id = os.environ.get("KAGGLE_KERNEL_RUN_ID")
outcome = run_constant_baseline(
    input_root=Path("/kaggle/input"),
    work_root=Path("/kaggle/working"),
    probability=0.5,
    run_id=create_run_id("constant-baseline", external_id=kernel_run_id),
    notebook={
        "enable_gpu": False,
        "enable_internet": False,
        "kaggle_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        "kernel_id": "plthon/rsna-knee-constant-baseline",
        "kernel_run_id": kernel_run_id,
        "jupytext_source_sha256": "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__",
    },
    source_provenance={
        "git_commit": "__RSNA_KNEE_GIT_COMMIT__",
        "git_dirty": json.loads("__RSNA_KNEE_GIT_DIRTY__"),
        "package_archive": {
            "sha256": hashlib.sha256(package_archive_bytes).hexdigest(),
        },
    },
)

# %%
print(outcome.submission_path)
print(outcome.manifest_path)
print(
    json.dumps(
        json.loads(outcome.validation_report_path.read_text()),
        indent=2,
        sort_keys=True,
    )
)
