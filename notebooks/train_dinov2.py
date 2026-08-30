# %% [markdown]
# # RSNA Knee DINOv2 ViT-S/14 Controlled Backbone Experiment
#
# Trains exactly one offline DINOv2 ViT-S/14 candidate and compares it with
# the frozen #27 ResNet-18 control on the identical validation partition.
# This notebook does not run hidden-test inference or submit.

# %%
import base64
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

GPU_APPROVED = json.loads("__RSNA_KNEE_APPROVED_GPU__")
GIT_DIRTY = json.loads("__RSNA_KNEE_GIT_DIRTY__")
if GIT_DIRTY:
    raise RuntimeError("DINOv2 training requires a clean committed Git revision")

WORK_ROOT = Path("/kaggle/working")
PACKAGE_ARCHIVE = WORK_ROOT / "rsna_knee_source.zip"
PACKAGE_ARCHIVE.write_bytes(base64.b64decode("__RSNA_KNEE_PACKAGE_ARCHIVE__"))
sys.path.insert(0, str(PACKAGE_ARCHIVE))

CONFIG_BYTES = base64.b64decode("__RSNA_KNEE_CONFIG__")
CONFIG_PATH = WORK_ROOT / "dinov2.toml"
CONFIG_PATH.write_bytes(CONFIG_BYTES)

from rsna_knee.combined_supervision import (
    discover_combined_checkpoint,
    load_combined_baseline_config,
    run_combined_baseline,
    verify_dinov2_artifact,
)

# %% [markdown]
# ## Train the one predeclared candidate and stop after paired validation

# %%
config = load_combined_baseline_config(CONFIG_PATH)
if config.architecture != "dinov2_vits14":
    raise RuntimeError("Issue #29 requires exactly dinov2_vits14")

input_root = Path("/kaggle/input")
control_checkpoint = discover_combined_checkpoint(input_root)
source_provenance = {
    "git_commit": "__RSNA_KNEE_GIT_COMMIT__",
    "git_dirty": GIT_DIRTY,
    "jupytext_source_sha256": "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__",
    "rendered_notebook_sha256": "__RSNA_KNEE_RENDERED_NOTEBOOK_SHA256__",
    "config_sha256": hashlib.sha256(CONFIG_BYTES).hexdigest(),
    "pytorch_allocator_configuration": os.environ["PYTORCH_ALLOC_CONF"],
}
outcome = run_combined_baseline(
    input_root=input_root,
    work_root=WORK_ROOT,
    config=config,
    approved_gpu=GPU_APPROVED,
    control_checkpoint_path=control_checkpoint,
    run_test_inference=False,
    source_provenance=source_provenance,
    notebook={
        "kind": "approved_kaggle_gpu_dinov2_vits14_training",
        "kernel_id": "plthon/rsna-knee-dinov2-vit-s-14",
        "test_inference": False,
        "competition_submission": False,
    },
)

# %% [markdown]
# ## Verify and publish the checkpoint plus public-safe validation summary

# %%
verification = verify_dinov2_artifact(outcome.artifact_dir)
for artifact in (
    outcome.report_path,
    outcome.manifest_path,
    outcome.checkpoint_path,
    outcome.artifact_dir / "validation-predictions.csv",
):
    shutil.copyfile(artifact, WORK_ROOT / artifact.name)

report = json.loads(outcome.report_path.read_text())
comparison = report["controlled_comparison"]
print("Verification:", json.dumps(verification, indent=2))
print("ResNet-18 macro AUC:", comparison["metrics"]["baseline"]["official_macro_auc"])
print("DINOv2-S macro AUC:", comparison["metrics"]["candidate"]["official_macro_auc"])
print("Paired macro AUC delta:", comparison["metrics"]["delta"]["official_macro_auc"])
print("Decision:", comparison["decision"]["result"])
print("Hidden-test inference: not run")
print("Competition submission: not run")
