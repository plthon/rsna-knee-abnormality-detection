# %% [markdown]
# # Raptor CoAtNet Phase 2 best-evidence reconstruction
#
# Runs the one issue #36 reconstruction recipe. It does not read hidden-test
# data, produce competition predictions, or submit.

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
STAGE_LITERAL = "__RSNA_KNEE_RECONSTRUCTION_STAGE__"
STAGE = cast(Literal["smoke", "full"], STAGE_LITERAL)
KERNEL_ID = "__RSNA_KNEE_KERNEL_ID__"
if GIT_DIRTY:
    raise RuntimeError("Raptor reconstruction requires a clean committed revision")
if STAGE not in {"smoke", "full"}:
    raise RuntimeError("Raptor reconstruction stage must be smoke or full")

WORK_ROOT = Path("/kaggle/working")
PACKAGE_ARCHIVE = WORK_ROOT / "rsna_knee_source.zip"
PACKAGE_ARCHIVE.write_bytes(base64.b64decode("__RSNA_KNEE_PACKAGE_ARCHIVE__"))
sys.path.insert(0, str(PACKAGE_ARCHIVE))

from rsna_knee.raptor_reconstruction import (
    run_raptor_reconstruction,
    verify_raptor_reconstruction_artifact,
)

# %% [markdown]
# ## Train only the selected reconstruction stage

# %%
source_provenance = {
    "git_commit": "__RSNA_KNEE_GIT_COMMIT__",
    "git_dirty": GIT_DIRTY,
    "jupytext_source_sha256": "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__",
    "rendered_notebook_sha256": "__RSNA_KNEE_RENDERED_NOTEBOOK_SHA256__",
}
outcome = run_raptor_reconstruction(
    input_root=Path("/kaggle/input"),
    work_root=WORK_ROOT,
    stage=STAGE,
    approved_gpu=GPU_APPROVED,
    source_provenance=source_provenance,
    notebook={
        "kind": "issue_36_best_evidence_raptor_reconstruction",
        "kernel_id": KERNEL_ID,
        "stage": STAGE,
        "training": True,
        "hidden_test_inference": False,
        "competition_submission": False,
    },
)
PACKAGE_ARCHIVE.unlink()
verification = verify_raptor_reconstruction_artifact(WORK_ROOT)


def validate_raptor_reconstruction_outputs(work_root: Path) -> None:
    expected_outputs = {
        "raptor-reconstruction-best.pt",
        "raptor-reconstruction-swa.pt",
        "raptor-reconstruction-report.json",
        "run-manifest.json",
    }
    kaggle_managed_outputs = {"__notebook__.ipynb"}
    actual_outputs = {
        path.name
        for path in work_root.iterdir()
        if path.is_file() and path.name not in kaggle_managed_outputs
    }
    if actual_outputs != expected_outputs:
        raise ValueError(
            "Raptor reconstruction emitted unexpected outputs: "
            f"{sorted(actual_outputs ^ expected_outputs)}"
        )


validate_raptor_reconstruction_outputs(WORK_ROOT)
report = json.loads(outcome.report_path.read_text())
print("Verification:", json.dumps(verification, indent=2, sort_keys=True))
print("Stage:", report["stage"])
print("Best epoch:", report["checkpoint_selection"]["best_epoch"])
print(
    "Best gold macro AUC:",
    report["checkpoint_selection"]["best_gold_macro_auc"],
)
print(
    "Averaged gold macro AUC:",
    report["checkpoint_selection"]["averaged_gold_macro_auc"],
)
print("Training stage: completed")
print("Hidden-test inference: not run")
print("Competition submission: not run")
