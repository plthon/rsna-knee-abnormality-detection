# %% [markdown]
# # RSNA Knee 336 Resource Smoke
# Bounded resource measurement only. This does not authorize full training,
# hidden-test inference, or competition submission.

# %%
import base64
import hashlib
import json
import sys
from pathlib import Path

GPU_APPROVED = json.loads("__RSNA_KNEE_APPROVED_GPU__")
WORK_ROOT = Path("/kaggle/working")
PACKAGE_ARCHIVE = WORK_ROOT / "rsna_knee_source.zip"
PACKAGE_ARCHIVE.write_bytes(base64.b64decode("__RSNA_KNEE_PACKAGE_ARCHIVE__"))
sys.path.insert(0, str(PACKAGE_ARCHIVE))
CONFIG_BYTES = base64.b64decode("__RSNA_KNEE_CONFIG__")
CONFIG_PATH = WORK_ROOT / "resolution-336.toml"
CONFIG_PATH.write_bytes(CONFIG_BYTES)

from rsna_knee.combined_supervision import load_combined_baseline_config
from rsna_knee.resolution import run_resolution_smoke

source_provenance = {
    "git_commit": "__RSNA_KNEE_GIT_COMMIT__",
    "git_dirty": json.loads("__RSNA_KNEE_GIT_DIRTY__"),
    "jupytext_source_sha256": "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__",
    "rendered_notebook_sha256": "__RSNA_KNEE_RENDERED_NOTEBOOK_SHA256__",
    "config_sha256": hashlib.sha256(CONFIG_BYTES).hexdigest(),
}
report_path = run_resolution_smoke(
    input_root=Path("/kaggle/input"),
    work_root=WORK_ROOT,
    config=load_combined_baseline_config(CONFIG_PATH),
    approved_gpu=GPU_APPROVED,
    source_provenance=source_provenance,
)
print(report_path.read_text())
print("STOP: full training requires separate approval after reviewing smoke evidence.")
