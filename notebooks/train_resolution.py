# %% [markdown]
# # RSNA Knee 336 Resolution
# One resolution-only candidate against cached, verified #31 predictions.
# Smoke and full training are separately built and approved stages.
# No hidden-test inference or submission runs here.

# %%
import base64
import hashlib
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

STAGE = "__RSNA_KNEE_RESOLUTION_STAGE__"
GPU_APPROVED = json.loads("__RSNA_KNEE_APPROVED_GPU__")
WORK_ROOT = Path("/kaggle/working")
PACKAGE_ARCHIVE = WORK_ROOT / "rsna_knee_source.zip"
PACKAGE_ARCHIVE.write_bytes(base64.b64decode("__RSNA_KNEE_PACKAGE_ARCHIVE__"))
sys.path.insert(0, str(PACKAGE_ARCHIVE))
CONFIG_BYTES = base64.b64decode("__RSNA_KNEE_CONFIG__")
CONFIG_PATH = WORK_ROOT / "resolution-336.toml"
CONFIG_PATH.write_bytes(CONFIG_BYTES)

from rsna_knee.combined_supervision import (
    load_combined_baseline_config,
    run_combined_baseline,
    verify_physical_crop_artifact,
)
from rsna_knee.resolution import run_resolution_smoke

source_provenance = {
    "git_commit": "__RSNA_KNEE_GIT_COMMIT__",
    "git_dirty": json.loads("__RSNA_KNEE_GIT_DIRTY__"),
    "jupytext_source_sha256": "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__",
    "rendered_notebook_sha256": "__RSNA_KNEE_RENDERED_NOTEBOOK_SHA256__",
    "config_sha256": hashlib.sha256(CONFIG_BYTES).hexdigest(),
}
config = load_combined_baseline_config(CONFIG_PATH)
if STAGE == "smoke":
    report_path = run_resolution_smoke(
        input_root=Path("/kaggle/input"),
        work_root=WORK_ROOT,
        config=config,
        approved_gpu=GPU_APPROVED,
        source_provenance=source_provenance,
    )
    print(report_path.read_text())
    print(
        "STOP: full training requires separate approval after reviewing smoke evidence."
    )
elif STAGE == "full":
    SMOKE_PATH = WORK_ROOT / "resolution-resource-smoke.json"
    SMOKE_PATH.write_bytes(base64.b64decode("__RSNA_KNEE_RESOURCE_SMOKE__"))
    batch_size = json.loads(SMOKE_PATH.read_text())["configuration"]["training"][
        "batch_size"
    ]
    config = replace(
        config, batch_size=batch_size, gradient_accumulation_steps=16 // batch_size
    )
    outcome = run_combined_baseline(
        input_root=Path("/kaggle/input"),
        work_root=WORK_ROOT,
        config=config,
        approved_gpu=GPU_APPROVED,
        resource_smoke_path=SMOKE_PATH,
        run_test_inference=False,
        source_provenance=source_provenance,
        notebook={
            "kind": "resolution_336_training",
            "test_inference": False,
            "competition_submission": False,
        },
    )
    verification = verify_physical_crop_artifact(
        outcome.artifact_dir, resolution_336=True
    )
    for artifact in (
        outcome.report_path,
        outcome.manifest_path,
        outcome.checkpoint_path,
    ):
        shutil.copyfile(artifact, WORK_ROOT / artifact.name)
    print(json.dumps(verification, indent=2))
    print(
        "STOP: any hidden inference and competition submission require separate approval."
    )
else:
    raise ValueError("Build an explicit smoke or full stage")
