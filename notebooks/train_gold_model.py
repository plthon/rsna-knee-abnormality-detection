# %% [markdown]
# # RSNA Knee Gold-Only ResNet-18 Smoke Training
#
# Private, internet-disabled, explicitly approved GPU smoke run for issue #8.
# It independently re-derives the canonical organizer-gold cohort, validates
# the exact offline ImageNet initialization before model construction, trains
# one mixed-precision epoch, reloads complete state, and validates predictions.
# Report text and report-derived labels are never read.

# %%
import base64
import hashlib
import json
import sys
from pathlib import Path

PACKAGE_ARCHIVE = Path("/kaggle/working/rsna_knee_training_source.zip")
PACKAGE_ARCHIVE.write_bytes(base64.b64decode("__RSNA_KNEE_PACKAGE_ARCHIVE__"))
sys.path.insert(0, str(PACKAGE_ARCHIVE))
CONFIG_PATH = Path("/kaggle/working/kaggle-training.toml")
CONFIG_PATH.write_bytes(base64.b64decode("__RSNA_KNEE_CONFIG__"))
if hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest() != (
    "__RSNA_KNEE_CONFIG_SHA256__"
):
    raise RuntimeError("Embedded training configuration hash mismatch")
GPU_APPROVED = json.loads("__RSNA_KNEE_GPU_APPROVED__")

# %%
from rsna_knee.training import (
    load_smoke_training_config,
    run_gold_smoke_training,
    verify_training_artifact,
)

configuration = load_smoke_training_config(CONFIG_PATH)
outcome = run_gold_smoke_training(
    input_root=Path("/kaggle/input"),
    work_root=Path("/kaggle/working"),
    config=configuration,
    approved_gpu=GPU_APPROVED,
    source_provenance={
        "git_commit": "__RSNA_KNEE_GIT_COMMIT__",
        "git_dirty": json.loads("__RSNA_KNEE_GIT_DIRTY__"),
        "jupytext_source_sha256": "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__",
        "package_archive_sha256": hashlib.sha256(
            PACKAGE_ARCHIVE.read_bytes()
        ).hexdigest(),
        "training_config_sha256": "__RSNA_KNEE_CONFIG_SHA256__",
        "weight_dataset_id": "__RSNA_KNEE_DATASET_ID__",
    },
    notebook={
        "kernel_id": "__RSNA_KNEE_KERNEL_ID__",
        "kind": "private_approved_kaggle_gpu_gold_smoke",
        "internet_enabled": False,
    },
)
verified = verify_training_artifact(outcome.artifact_dir)

# %%
print(outcome.manifest_path)
print(json.dumps(verified, indent=2, sort_keys=True))
