# %% [markdown]
# # Issue #38: Raptor T4 derivative
#
# Batch 4, accumulation 2. Training and organizer-gold validation only.

# %%
import base64
import json
import os
import sys
from pathlib import Path
from typing import Literal, cast

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

GPU_APPROVED = json.loads("__RSNA_KNEE_APPROVED_GPU__")
GIT_DIRTY = json.loads("__RSNA_KNEE_GIT_DIRTY__")
STAGE_LITERAL = "__RSNA_KNEE_DERIVATIVE_STAGE__"
STAGE = cast(Literal["smoke", "full"], STAGE_LITERAL)
RESUME = json.loads("__RSNA_KNEE_DERIVATIVE_RESUME__")
KERNEL_ID = "__RSNA_KNEE_KERNEL_ID__"
KERNEL_VERSION = os.environ.get("KAGGLE_KERNEL_VERSION")
if GIT_DIRTY:
    raise RuntimeError("Raptor T4 derivative requires a clean committed revision")
if STAGE not in {"smoke", "full"}:
    raise RuntimeError("Raptor T4 stage must be smoke or full")

WORK_ROOT = Path("/kaggle/working")
INPUT_ROOT = Path("/kaggle/input")
PACKAGE_ARCHIVE = WORK_ROOT / "rsna_knee_source.zip"
PACKAGE_ARCHIVE.write_bytes(base64.b64decode("__RSNA_KNEE_PACKAGE_ARCHIVE__"))
sys.path.insert(0, str(PACKAGE_ARCHIVE))

import torch

from rsna_knee.raptor_t4_derivative import (
    derivative_config,
    run_raptor_t4_derivative,
    verify_raptor_t4_artifact,
)

resume_path = None
if RESUME:
    matches = list(INPUT_ROOT.rglob("raptor-t4-resume.pt"))
    if len(matches) != 1:
        raise FileNotFoundError("Expected exactly one Raptor T4 resume checkpoint")
    resume_path = matches[0]

source_provenance = {
    "git_commit": "__RSNA_KNEE_GIT_COMMIT__",
    "git_dirty": GIT_DIRTY,
    "jupytext_source_sha256": "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__",
    "rendered_notebook_sha256": "__RSNA_KNEE_RENDERED_NOTEBOOK_SHA256__",
}
notebook_provenance = {
    "kind": "issue_38_raptor_t4_derivative",
    "kernel_id": KERNEL_ID,
    "kernel_version": KERNEL_VERSION,
    "stage": STAGE,
    "training": True,
    "hidden_test_inference": False,
    "competition_submission": False,
}
try:
    report_path = run_raptor_t4_derivative(
        input_root=INPUT_ROOT,
        work_root=WORK_ROOT,
        stage=STAGE,
        approved_gpu=GPU_APPROVED,
        source_provenance=source_provenance,
        notebook=notebook_provenance,
        resume_path=resume_path,
    )
except torch.OutOfMemoryError:
    failure_path = WORK_ROOT / "raptor-t4-report.json"
    if not failure_path.exists():
        config, _, derivative_sha = derivative_config()
        failure_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "stage": STAGE,
                    "status": "resource_infeasible",
                    "source": source_provenance,
                    "notebook": notebook_provenance,
                    "derivative_sha256": derivative_sha,
                    "configuration": {
                        **config.record(),
                        "gradient_accumulation_steps": 2,
                        "effective_batch_size": 8,
                    },
                    "optimizer_update_count": None,
                    "scheduler_step_count": None,
                    "execution": {
                        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_gpu_bytes": torch.cuda.max_memory_reserved(),
                        "oom_or_retry": True,
                        "hidden_test_inference": False,
                        "competition_submission": False,
                    },
                    "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    PACKAGE_ARCHIVE.unlink(missing_ok=True)
    raise
PACKAGE_ARCHIVE.unlink()
verification = verify_raptor_t4_artifact(WORK_ROOT)


def validate_raptor_t4_outputs(work_root: Path, stage: str) -> None:
    expected = {
        "raptor-t4-best.pt",
        "raptor-t4-swa.pt",
        "raptor-t4-report.json",
        "run-manifest.json",
    }
    if stage == "full":
        expected.add("raptor-t4-resume.pt")
    actual = {
        path.name
        for path in work_root.iterdir()
        if path.is_file() and path.name != "__notebook__.ipynb"
    }
    if actual != expected:
        raise ValueError(
            f"Raptor T4 emitted unexpected outputs: {sorted(actual ^ expected)}"
        )


validate_raptor_t4_outputs(WORK_ROOT, STAGE)
report = json.loads(report_path.read_text())
print("Verification:", json.dumps(verification, indent=2, sort_keys=True))
print("Stage:", report["stage"])
print("Optimizer updates:", report["optimizer_update_count"])
print("Scheduler steps:", report["scheduler_step_count"])
print("Best gold macro AUC:", report["checkpoint_selection"]["best_gold_macro_auc"])
print("Training stage: completed")
print("Hidden-test inference: not run")
print("Competition submission: not run")
