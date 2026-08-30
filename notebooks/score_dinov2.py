# %% [markdown]
# # RSNA Knee DINOv2 ViT-S/14 Inference-Only Scoring
#
# Built only after the controlled validation keep rule passes. It reloads the
# exact retained checkpoint, rebuilds the live test-only index, runs inference,
# and validates submission.csv. It never trains or submits.

# %%
import base64
import json
import shutil
import sys
from pathlib import Path

GPU_APPROVED = json.loads("__RSNA_KNEE_APPROVED_GPU__")
EXPECTED_CHECKPOINT_SHA256 = "__RSNA_KNEE_CHECKPOINT_SHA256__"
WORK_ROOT = Path("/kaggle/working")
PACKAGE_ARCHIVE = WORK_ROOT / "rsna_knee_source.zip"
PACKAGE_ARCHIVE.write_bytes(base64.b64decode("__RSNA_KNEE_PACKAGE_ARCHIVE__"))
sys.path.insert(0, str(PACKAGE_ARCHIVE))

from rsna_knee.combined_supervision import (
    discover_combined_checkpoint,
    load_verified_dinov2_checkpoint,
    run_combined_scoring,
)
from rsna_knee.indexing import build_competition_index, verify_index_artifact
from rsna_knee.submission import (
    discover_submission_inputs,
    validate_submission,
    write_submission,
)
from rsna_knee.training import load_indexed_studies

SCORING_COMPONENTS = (
    load_verified_dinov2_checkpoint,
    discover_submission_inputs,
    build_competition_index,
    verify_index_artifact,
    load_indexed_studies,
    write_submission,
    validate_submission,
)

# %% [markdown]
# ## Score the live test mount with the retained DINOv2 checkpoint

# %%
input_root = Path("/kaggle/input")
checkpoint_path = discover_combined_checkpoint(
    input_root, expected_sha256=EXPECTED_CHECKPOINT_SHA256
)
outcome = run_combined_scoring(
    input_root=input_root,
    work_root=WORK_ROOT,
    approved_gpu=GPU_APPROVED,
    checkpoint_path=checkpoint_path,
    backbone="dinov2_vits14",
    expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
    source_provenance={
        "git_commit": "__RSNA_KNEE_GIT_COMMIT__",
        "git_dirty": json.loads("__RSNA_KNEE_GIT_DIRTY__"),
        "jupytext_source_sha256": "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__",
        "rendered_notebook_sha256": "__RSNA_KNEE_RENDERED_NOTEBOOK_SHA256__",
    },
    notebook={
        "kind": "retained_dinov2_vits14_inference_only_scoring",
        "kernel_id": "__RSNA_KNEE_KERNEL_ID__",
        "competition_submission": False,
    },
)

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
print("Backbone:", report["checkpoint"]["backbone"])
print("Predictions:", report["inference"]["prediction_count"])
print("Fallbacks:", report["inference"]["fallback_count"])
print("Validated submission artifact:", WORK_ROOT / "submission.csv")
print("Competition submission: not run")
