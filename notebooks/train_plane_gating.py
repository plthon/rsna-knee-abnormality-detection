# %% [markdown]
# # RSNA Knee Target-Specific Anatomical-Plane Gating
#
# Controlled issue #28 training and paired validation only. The frozen #27
# checkpoint supplies control predictions for the identical PatientID-disjoint
# validation partition. This notebook does not run test inference or submit.

# %%
import base64
import hashlib
import json
import shutil
import sys
from pathlib import Path

GPU_APPROVED = json.loads("__RSNA_KNEE_APPROVED_GPU__")
WORK_ROOT = Path("/kaggle/working")
PACKAGE_ARCHIVE = WORK_ROOT / "rsna_knee_source.zip"
PACKAGE_ARCHIVE.write_bytes(base64.b64decode("__RSNA_KNEE_PACKAGE_ARCHIVE__"))
sys.path.insert(0, str(PACKAGE_ARCHIVE))

CONFIG_BYTES = base64.b64decode("__RSNA_KNEE_CONFIG__")
CONFIG_PATH = WORK_ROOT / "plane-gating.toml"
CONFIG_PATH.write_bytes(CONFIG_BYTES)

from rsna_knee.combined_supervision import (
    discover_combined_checkpoint,
    load_combined_baseline_config,
    run_combined_baseline,
    verify_plane_gating_artifact,
)

# %% [markdown]
# ## Train one controlled candidate and compare it with the frozen #27 control

# %%
config = load_combined_baseline_config(CONFIG_PATH)
if config.aggregation != "target_plane_gate":
    raise RuntimeError("Issue #28 requires target_plane_gate aggregation")

input_root = Path("/kaggle/input")
control_checkpoint = discover_combined_checkpoint(input_root)
source_provenance = {
    "git_commit": "__RSNA_KNEE_GIT_COMMIT__",
    "git_dirty": json.loads("__RSNA_KNEE_GIT_DIRTY__"),
    "jupytext_source_sha256": "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__",
    "rendered_notebook_sha256": "__RSNA_KNEE_RENDERED_NOTEBOOK_SHA256__",
    "config_sha256": hashlib.sha256(CONFIG_BYTES).hexdigest(),
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
        "kind": "approved_kaggle_gpu_plane_gating_training",
        "kernel_id": "plthon/rsna-knee-plane-gating",
        "test_inference": False,
        "competition_submission": False,
    },
)

# %% [markdown]
# ## Verify and publish only public-safe summary artifacts at the output root

# %%
verification = verify_plane_gating_artifact(outcome.artifact_dir)
for artifact in (outcome.report_path, outcome.manifest_path, outcome.checkpoint_path):
    shutil.copyfile(artifact, WORK_ROOT / artifact.name)

report = json.loads(outcome.report_path.read_text())
comparison = report["controlled_comparison"]
print("Verification:", json.dumps(verification, indent=2))
print("Baseline macro AUC:", comparison["metrics"]["baseline"]["official_macro_auc"])
print(
    "Plane-gated macro AUC:", comparison["metrics"]["candidate"]["official_macro_auc"]
)
print("Paired macro AUC delta:", comparison["metrics"]["delta"]["official_macro_auc"])
print("Decision:", comparison["decision"]["result"])
print("Hidden-test inference: not run")
print("Competition submission: not run")
