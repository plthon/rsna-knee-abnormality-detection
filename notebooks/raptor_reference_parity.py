# %% [markdown]
# # RSNA Knee Raptor reference parity
#
# Approved issue #36 checkpoint parity and organizer-gold sanity diagnostics only.
# This notebook reads organizer-labeled training studies. It does not access test
# images, create a competition submission, or train a model.

# %%
import base64
import csv
import gc
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

GPU_APPROVED = json.loads("__RSNA_KNEE_APPROVED_GPU__")
if not GPU_APPROVED:
    raise PermissionError("Raptor parity requires explicit GPU approval")
if not torch.cuda.is_available():
    raise RuntimeError("Raptor parity requires a CUDA device")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

WORK_ROOT = Path("/kaggle/working")
INPUT_ROOT = Path("/kaggle/input")
COMPETITION_ROOT = INPUT_ROOT / "competitions" / "rsna-knee-abnormality-detection"
PACKAGE_ARCHIVE = WORK_ROOT / "rsna_knee_source.zip"
PACKAGE_ARCHIVE.write_bytes(base64.b64decode("__RSNA_KNEE_PACKAGE_ARCHIVE__"))
sys.path.insert(0, str(PACKAGE_ARCHIVE))

from rsna_knee.holdout import evaluate_point_predictions
from rsna_knee.provenance import write_run_manifest
from rsna_knee.raptor_reference import (
    CHECKPOINT_FILENAME,
    CHECKPOINT_SHA256,
    TARGET_NAMES,
    WINDOW_COUNT,
    build_study_stack,
    eval_windows,
    infer_probabilities,
    load_raptor_model,
    load_reference_contract,
    make_reader,
)
from rsna_knee.training import derive_gold_training_cohort

# %% [markdown]
# ## Resolve only frozen public and organizer-gold inputs

# %%
required_training_inputs = (
    COMPETITION_ROOT / "train.csv",
    COMPETITION_ROOT / "train_series.csv",
    COMPETITION_ROOT / "train_series",
)
if not all(path.exists() for path in required_training_inputs):
    raise FileNotFoundError("Organizer training inputs are incomplete")

checkpoint_candidates = [
    path
    for path in INPUT_ROOT.rglob(CHECKPOINT_FILENAME)
    if path.is_file()
    and path.stat().st_size == 292829402
    and hashlib.sha256(path.read_bytes()).hexdigest() == CHECKPOINT_SHA256
]
if len(checkpoint_candidates) != 1:
    raise FileNotFoundError(
        f"Expected one exact {CHECKPOINT_FILENAME}, found {len(checkpoint_candidates)}"
    )
checkpoint_path = checkpoint_candidates[0]

index_candidates = [
    path.parent
    for path in INPUT_ROOT.rglob("studies.csv")
    if (path.parent / "index-manifest.json").is_file()
]
if len(index_candidates) != 1:
    raise FileNotFoundError(
        f"Expected one attached data-index artifact, found {len(index_candidates)}"
    )
index_root = index_candidates[0]
cohort = derive_gold_training_cohort(COMPETITION_ROOT, index_root)
if len(cohort.studies) != 58 or cohort.target_names != TARGET_NAMES:
    raise ValueError("Organizer-gold cohort identity does not match the reference")

series_records: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
with (COMPETITION_ROOT / "train_series.csv").open(
    newline="", encoding="utf-8-sig"
) as file:
    for row in csv.DictReader(file):
        series_records[str(row["StudyInstanceUID"])].append(dict(row))

# %% [markdown]
# ## Compare CPU and public T4 execution on one deterministic gold study

# %%
reader = make_reader()
parity_study = cohort.studies[0]
parity_stack = build_study_stack(
    parity_study.study_uid,
    series_records,
    COMPETITION_ROOT / "train_series",
    reader=reader,
)
parity_windows = eval_windows(parity_stack.volume, parity_stack.mask)

cpu_model, cpu_metadata = load_raptor_model(checkpoint_path, device=torch.device("cpu"))
cpu_probability = infer_probabilities(cpu_model, parity_windows, torch.device("cpu"))
del cpu_model
gc.collect()

gpu_device = torch.device("cuda")
torch.cuda.reset_peak_memory_stats(gpu_device)
gpu_model, gpu_metadata = load_raptor_model(checkpoint_path, device=gpu_device)
gpu_probability = infer_probabilities(gpu_model, parity_windows, gpu_device)
if cpu_metadata != gpu_metadata:
    raise ValueError("CPU and GPU checkpoint metadata differ")
parity_absolute_difference = np.abs(cpu_probability - gpu_probability)

# %% [markdown]
# ## Evaluate all 58 organizer-gold studies on the public checkpoint

# %%
started = time.perf_counter()
predictions = np.empty((len(cohort.studies), len(TARGET_NAMES)), dtype=np.float32)
predictions[0] = gpu_probability
for index, study in enumerate(cohort.studies[1:], start=1):
    stack = build_study_stack(
        study.study_uid,
        series_records,
        COMPETITION_ROOT / "train_series",
        reader=reader,
    )
    windows = eval_windows(stack.volume, stack.mask)
    predictions[index] = infer_probabilities(gpu_model, windows, gpu_device)
    del stack, windows
    if (index + 1) % 10 == 0 or index + 1 == len(cohort.studies):
        print(f"Organizer-gold progress: {index + 1}/{len(cohort.studies)}", flush=True)

inference_seconds = time.perf_counter() - started
peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated(gpu_device))
if not np.isfinite(predictions).all() or bool(
    ((predictions < 0) | (predictions > 1)).any()
):
    raise ValueError("Raptor gold predictions are not finite probabilities")

point_metrics = evaluate_point_predictions(
    labels=[study.labels for study in cohort.studies],
    predictions=predictions.tolist(),
    target_names=TARGET_NAMES,
)
reference_contract, reference_contract_sha256 = load_reference_contract()
expected_metadata = reference_contract["checkpoint"]["metadata"]
expected_aucs = expected_metadata["aucs"]
observed_aucs = {
    target: point_metrics["per_target"][target]["roc_auc"] for target in TARGET_NAMES
}
auc_absolute_differences = {
    target: abs(float(observed_aucs[target]) - float(expected_aucs[target]))
    for target in TARGET_NAMES
}
observed_macro_auc = float(point_metrics["official_macro_auc"])
expected_macro_auc = float(expected_metadata["gold_auc"])
macro_absolute_difference = abs(observed_macro_auc - expected_macro_auc)
metadata_parity = max(auc_absolute_differences.values()) <= 1e-12

# %% [markdown]
# ## Write aggregate-only parity evidence

# %%
report_path = WORK_ROOT / "raptor-parity-report.json"
max_probability_difference = float(parity_absolute_difference.max())
report = {
    "boundaries": {
        "competition_submission": False,
        "hidden_test_inference": False,
        "training": False,
    },
    "checkpoint": {
        "architecture": gpu_metadata.architecture,
        "epoch": gpu_metadata.epoch,
        "filename": checkpoint_path.name,
        "resolution": gpu_metadata.resolution,
        "sha256": gpu_metadata.sha256,
        "source": gpu_metadata.source,
        "targets": list(gpu_metadata.targets),
    },
    "cpu_gpu_parity": {
        "comparison_scope": "first StudyInstanceUID in canonical UTF-8 cohort order",
        "mean_absolute_probability_difference": float(
            parity_absolute_difference.mean()
        ),
        "max_absolute_probability_difference": max_probability_difference,
        "output_count": int(parity_absolute_difference.size),
    },
    "execution": {
        "cuda_backend": {
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
        "cuda_device": torch.cuda.get_device_name(gpu_device),
        "inference_seconds": inference_seconds,
        "opencv_version": cv2.__version__,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "study_count": len(cohort.studies),
        "timm_version": importlib.metadata.version("timm"),
        "torch_version": torch.__version__,
        "window_count_per_study": WINDOW_COUNT,
    },
    "gold_cohort": {
        "digest": cohort.digest,
        "gold_contract_sha256": cohort.gold_contract_sha256,
        "gold_supervision_contract_sha256": (cohort.gold_supervision_contract_sha256),
        "selection_rule_version": "organizer-gold-all-eligible-v1",
        "study_count": len(cohort.studies),
    },
    "gold_metric": {
        "auc_absolute_differences": auc_absolute_differences,
        "expected_macro_roc_auc": expected_macro_auc,
        "macro_absolute_difference": macro_absolute_difference,
        "matches_checkpoint_metadata": metadata_parity,
        "observed": point_metrics,
        "observed_macro_roc_auc": observed_macro_auc,
    },
    "kernel": {
        "git_commit": "__RSNA_KNEE_GIT_COMMIT__",
        "git_dirty": json.loads("__RSNA_KNEE_GIT_DIRTY__"),
        "id": "__RSNA_KNEE_KERNEL_ID__",
        "jupytext_source_sha256": "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__",
        "rendered_notebook_sha256": "__RSNA_KNEE_RENDERED_NOTEBOOK_SHA256__",
    },
    "reference_contract_sha256": reference_contract_sha256,
    "schema_version": 1,
}
report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
manifest_path = write_run_manifest(
    WORK_ROOT / "run-manifest.json",
    run_id="raptor-reference-parity-v1",
    source={
        "checkpoint_sha256": gpu_metadata.sha256,
        "cohort_digest": cohort.digest,
        "raptor_reference_contract_sha256": reference_contract_sha256,
    },
    configuration={
        "device": "cpu-and-cuda",
        "gold_study_count": len(cohort.studies),
        "target_order": list(TARGET_NAMES),
        "window_count": WINDOW_COUNT,
    },
    notebook={
        "competition_submission": False,
        "hidden_test_inference": False,
        "kernel_id": "__RSNA_KNEE_KERNEL_ID__",
        "kind": "approved_kaggle_raptor_reference_parity",
        "training": False,
    },
    artifacts=[report_path],
    artifact_root=WORK_ROOT,
)

print(
    json.dumps(
        {
            "cpu_gpu_max_abs": max_probability_difference,
            "gold_macro_auc": observed_macro_auc,
            "matches_checkpoint_metadata": metadata_parity,
            "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        },
        indent=2,
        sort_keys=True,
    )
)
print(f"Aggregate report: {report_path}")
print(f"Run manifest: {manifest_path}")
print("Hidden-test inference: not run")
print("Competition submission: not run")
print("Training: not run")
