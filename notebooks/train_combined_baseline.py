# %% [markdown]
# # RSNA Knee Combined Gold and Public Label Baseline
#
# Training-only reproducibility notebook. Competition scoring uses the separate
# inference-only `score_combined_baseline.py` notebook.
#
# Trains a ResNet-18 baseline on magnetic resonance imaging (MRI) studies
# using combined organizer-gold supervision for the 58 gold studies and #24
# frozen public report-derived labels (YES -> 1, NO -> 0, UNK -> masked) for
# the 4,349 non-gold studies.
#
# Validates checkpoint reload reproducibility on validation partition before
# test inference. Its three-study test result is only a mounted example check,
# not hidden-scoring evidence. Produces and strictly validates submission.csv.

# %%
import base64
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

started = time.monotonic()
GPU_APPROVED = json.loads("__RSNA_KNEE_APPROVED_GPU__")

WORK_ROOT = Path("/kaggle/working")
PACKAGE_ARCHIVE = WORK_ROOT / "rsna_knee_source.zip"
PACKAGE_ARCHIVE.write_bytes(base64.b64decode("__RSNA_KNEE_PACKAGE_ARCHIVE__"))
sys.path.insert(0, str(PACKAGE_ARCHIVE))

CONFIG_BYTES = base64.b64decode("__RSNA_KNEE_CONFIG__")
CONFIG_PATH = WORK_ROOT / "combined-baseline.toml"
CONFIG_PATH.write_bytes(CONFIG_BYTES)

from rsna_knee.combined_supervision import (
    load_combined_baseline_config,
    run_combined_baseline,
    verify_combined_baseline_artifact,
)
from rsna_knee.training import require_approved_cuda

# %% [markdown]
# ## 1. Verify CUDA and runtime readiness

# %%
cuda_readiness = require_approved_cuda(approved=GPU_APPROVED)
print("CUDA Readiness:", json.dumps(cuda_readiness, indent=2))

# %% [markdown]
# ## 2. Execute Combined Baseline Training and Test Inference

# %%
config = load_combined_baseline_config(CONFIG_PATH)
print("Training Config:", config)

input_root = Path("/kaggle/input")
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
    source_provenance=source_provenance,
    notebook={
        "kind": "approved_kaggle_gpu_combined_baseline",
        "kernel_id": "plthon/rsna-knee-gold-public-baseline",
    },
)

print(f"Outcome completed in {outcome.artifact_dir}")
if outcome.submission_path is None or outcome.validation_report_path is None:
    raise RuntimeError(
        "Baseline training must produce example-test inference artifacts"
    )

# %% [markdown]
# ## 3. Copy Key Artifacts to Working Root for Kaggle Capture

# %%
shutil.copyfile(outcome.submission_path, WORK_ROOT / "submission.csv")
shutil.copyfile(
    outcome.validation_report_path, WORK_ROOT / "submission-validation.json"
)
shutil.copyfile(outcome.report_path, WORK_ROOT / "training-report.json")
shutil.copyfile(outcome.manifest_path, WORK_ROOT / "run-manifest.json")
shutil.copyfile(outcome.checkpoint_path, WORK_ROOT / "checkpoint.pt")

# %% [markdown]
# ## 4. Verify Completed Baseline Artifact

# %%
verification = verify_combined_baseline_artifact(outcome.artifact_dir)
print("Artifact Verification:", json.dumps(verification, indent=2))

report = json.loads((WORK_ROOT / "training-report.json").read_text())
print("\n--- Training Execution Summary ---")
print("Total Training Seconds:", report["training_execution"]["total_training_seconds"])
print("Peak GPU Memory Bytes:", report["training_execution"]["peak_cuda_memory_bytes"])
print("Max Checkpoint Reload Diff:", report["checkpoint"]["max_reload_absolute_diff"])
print("Epoch Logs:")
for ep in report["training_execution"]["epoch_logs"]:
    print(
        f"  Epoch {ep['epoch']}: Train Loss = {ep['train_mean_loss']:.4f}, Val BCE = {ep['val_mean_bce']:.4f}, Val Acc = {ep['val_accuracy_at_0_5']:.4f}, Val AUC = {ep['val_mean_roc_auc']}"
    )

print("\n--- Submission Summary ---")
print("Submission SHA-256:", report["submission"]["sha256"])
print("Prediction Count:", report["submission"]["prediction_count"])
print("Fallback Count:", report["submission"]["fallback_count"])
print(f"\nAll steps finished successfully in {time.monotonic() - started:.2f}s.")
