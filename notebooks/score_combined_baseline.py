# %% [markdown]
# # RSNA Knee Gold + Public Baseline: Inference-only Competition Scoring
#
# This notebook reuses the completed combined-supervision checkpoint. It reads
# the scoring-time `test.csv` and `test_series.csv`, rebuilds a compact index for
# the live test DICOMs, runs the shared MRI inference path, and validates the
# exact competition submission schema.
#
# The mounted example test set and the frozen full-data index are not used for
# scoring membership or test-series discovery. This notebook has no training,
# validation, optimizer, or epoch path.

# %%
import base64
import json
import shutil
import sys
from pathlib import Path

GPU_APPROVED = json.loads("__RSNA_KNEE_APPROVED_GPU__")
WORK_ROOT = Path("/kaggle/working")
PACKAGE_ARCHIVE = WORK_ROOT / "rsna_knee_source.zip"
PACKAGE_ARCHIVE.write_bytes(base64.b64decode("__RSNA_KNEE_PACKAGE_ARCHIVE__"))
sys.path.insert(0, str(PACKAGE_ARCHIVE))

from rsna_knee.combined_supervision import (
    COMBINED_BASELINE_CHECKPOINT_SHA256,
    discover_combined_checkpoint,
    load_verified_combined_checkpoint,
    run_combined_scoring,
)
from rsna_knee.indexing import build_competition_index, verify_index_artifact
from rsna_knee.submission import (
    discover_submission_inputs,
    validate_submission,
    write_submission,
)
from rsna_knee.training import load_indexed_studies

# These references make the scoring contract inspectable in the rendered bundle;
# run_combined_scoring invokes them in this order without any training branch.
# The live index is explicitly test-only: build_competition_index(splits=("test",)).
SCORING_COMPONENTS = (
    load_verified_combined_checkpoint,
    discover_submission_inputs,
    build_competition_index,
    verify_index_artifact,
    load_indexed_studies,
    write_submission,
    validate_submission,
)

# %% [markdown]
# ## Load the required frozen checkpoint and score the live test mount

# %%
print("Required checkpoint SHA-256:", COMBINED_BASELINE_CHECKPOINT_SHA256)
checkpoint_path = discover_combined_checkpoint(Path("/kaggle/input"))
outcome = run_combined_scoring(
    input_root=Path("/kaggle/input"),
    work_root=WORK_ROOT,
    approved_gpu=GPU_APPROVED,
    checkpoint_path=checkpoint_path,
    source_provenance={
        "git_commit": "__RSNA_KNEE_GIT_COMMIT__",
        "git_dirty": json.loads("__RSNA_KNEE_GIT_DIRTY__"),
        "jupytext_source_sha256": "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__",
        "rendered_notebook_sha256": "__RSNA_KNEE_RENDERED_NOTEBOOK_SHA256__",
    },
    notebook={
        "kind": "combined_inference_only_scoring",
        "kernel_id": "plthon/rsna-knee-gold-public-scoring",
    },
)

# %% [markdown]
# ## Publish only the validated scoring artifacts to the notebook output

# %%
for artifact in (
    outcome.submission_path,
    outcome.fallbacks_path,
    outcome.validation_report_path,
    outcome.scoring_report_path,
    outcome.manifest_path,
):
    shutil.copyfile(artifact, WORK_ROOT / artifact.name)

report = json.loads(outcome.scoring_report_path.read_text())
print("Mode:", report["mode"])
print("Live test studies:", report["live_test"]["study_count"])
print("Indexed test studies:", report["live_test"]["indexed_study_count"])
print("Predictions:", report["inference"]["prediction_count"])
print("Fallbacks:", report["inference"]["fallback_count"])
print("Submission:", WORK_ROOT / "submission.csv")
