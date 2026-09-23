# %% [markdown]
# # RSNA Knee Raptor reference hidden inference
#
# Approved issue #36 Stop point 2 only. This notebook runs the exact published
# checkpoint over the hidden test mount and writes a strictly validated local
# submission artifact. It does not submit to the competition or train a model.

# %%
import base64
import hashlib
import json
import sys
from pathlib import Path

import torch

GPU_APPROVED = json.loads("__RSNA_KNEE_APPROVED_GPU__")
if not GPU_APPROVED:
    raise PermissionError("Raptor hidden inference requires explicit GPU approval")
if not torch.cuda.is_available():
    raise RuntimeError("Raptor hidden inference requires a CUDA device")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working")
PACKAGE_ARCHIVE = Path("/tmp/rsna_knee_source.zip")
PACKAGE_ARCHIVE.write_bytes(base64.b64decode("__RSNA_KNEE_PACKAGE_ARCHIVE__"))
sys.path.insert(0, str(PACKAGE_ARCHIVE))

from rsna_knee.raptor_reference import (
    CHECKPOINT_FILENAME,
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE_BYTES,
    run_raptor_reference,
)

# %% [markdown]
# ## Resolve the exact published checkpoint and run hidden inference once

# %%
checkpoint_candidates = [
    path
    for path in INPUT_ROOT.rglob(CHECKPOINT_FILENAME)
    if path.is_file()
    and path.stat().st_size == CHECKPOINT_SIZE_BYTES
    and hashlib.sha256(path.read_bytes()).hexdigest() == CHECKPOINT_SHA256
]
if len(checkpoint_candidates) != 1:
    raise FileNotFoundError(
        f"Expected one exact {CHECKPOINT_FILENAME}, found {len(checkpoint_candidates)}"
    )

outcome = run_raptor_reference(
    input_root=INPUT_ROOT,
    work_root=WORK_ROOT,
    checkpoint_path=checkpoint_candidates[0],
    device=torch.device("cuda"),
    run_id="raptor-reference-hidden-scoring-v1",
    source_provenance={
        "git_commit": "__RSNA_KNEE_GIT_COMMIT__",
        "git_dirty": json.loads("__RSNA_KNEE_GIT_DIRTY__"),
        "jupytext_source_sha256": "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__",
        "kernel_id": "__RSNA_KNEE_KERNEL_ID__",
        "rendered_notebook_sha256": "__RSNA_KNEE_RENDERED_NOTEBOOK_SHA256__",
    },
    notebook={
        "competition_submission": False,
        "hidden_test_inference": True,
        "kernel_id": "__RSNA_KNEE_KERNEL_ID__",
        "kind": "approved_kaggle_raptor_reference_hidden_scoring",
        "training": False,
    },
)

expected_outputs = {
    "raptor-reference-run.json",
    "run-manifest.json",
    "submission-fallbacks.csv",
    "submission-validation.json",
    "submission.csv",
}
actual_outputs = {path.name for path in WORK_ROOT.iterdir() if path.is_file()}
if actual_outputs != expected_outputs:
    raise ValueError(
        f"Raptor hidden inference emitted unexpected outputs: "
        f"{sorted(actual_outputs ^ expected_outputs)}"
    )

report = json.loads(outcome.report_path.read_text())
validation = json.loads(outcome.submission.validation_report_path.read_text())
print("Hidden studies:", report["inference"]["study_count"])
print("Raw study failures:", report["inference"]["raw_failure_count"])
print("Inference seconds:", report["inference"]["seconds"])
print("Peak GPU memory bytes:", report["inference"]["peak_gpu_memory_bytes"])
print("Submission rows:", validation["row_count"])
print("Validated submission artifact:", outcome.submission.submission_path)
print("Hidden-test inference: completed")
print("Competition submission: not run")
print("Training: not run")
