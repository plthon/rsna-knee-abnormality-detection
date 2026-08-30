# RSNA Knee Abnormality Detection

This repository is a small, CPU-first research platform for the RSNA Knee
Abnormalities Detection competition. It reads knee MRI DICOM series and emits
one probability for each of the 12 verified abnormality targets.

## Verified project facts

The canonical competition schema is [`contracts/competition.json`](contracts/competition.json).
The verified training data contains 4,407 studies, 24,371 series, and 819,078
images. The test mount is an example replacement with three studies, 15 series,
and 557 images. `PatientID` is the image-model grouping boundary; the observed
training data has one study per patient and no train/test study overlap.

The pipeline preserves the correctness boundaries established by the committed
evidence: modality rescale and MONOCHROME1 handling, geometry-first ordering,
deterministic sampling and padding masks, finite shared preprocessing, target
order and submission validation, masked loss and masked pooling, exact offline
ResNet-18 initialization, checkpoint reload, holdout metrics, and unusable-study
fallbacks. The relevant reports and contracts remain under `docs/` and
`contracts/`.

## Current image baseline

The baseline uses a shared torchvision ResNet-18 initialized from the exact
offline `IMAGENET1K_V1` artifact. It repeats grayscale slices to RGB, averages
valid slice features within each series, then averages valid series features
within each study. Unknown organizer labels are masked from BCE loss.

Issue #28 tested an opt-in `target_plane_gate` aggregation mode with exactly 36
initialized-neutral target-by-plane scalars. The paired macro-AUC delta was only
`+0.00107`, so the precommitted rule rejected it and the default remains `mean`.
See the [plane-gating experiment report](docs/plane-gating-report.md).

Issue #29 adds one controlled candidate, `dinov2_vits14`, while retaining the
same 224-pixel grayscale-to-RGB input, eight slices per series, masked loss, and
slice-to-series-to-study means. Its official LVD-142M artifact is hash-verified
before a local, network-free ViT-S/14 is constructed. An approved Tesla T4 run
proved that batch 16 does not fit, so the candidate uses a micro-batch of four
with four-step gradient accumulation to preserve the effective batch size of
16. A later variable-size batch fragmented 2.51 GiB of reserved CUDA memory, so
the training notebook enables expandable allocator segments before importing
PyTorch. No completed validation or hidden-test result has been produced yet.

The recorded gold-only run reports macro AUC 0.6111 on a frozen holdout of four
patients. Its PatientID bootstrap has 9,065 undefined macro-AUC replicates out
of 10,000, so this is pipeline evidence, not a reliable model-comparison
benchmark.

## Repository layout

```text
contracts/                 verified schemas and committed evidence
docs/                      immutable inspection and experiment reports
notebooks/                 public, training-only, and scoring notebook sources
src/rsna_knee/             DICOM, indexing, preprocessing, training, evaluation,
                           submission, provenance, and Kaggle boundaries
scripts/kaggle.py          one guarded public-notebook build/push command
scripts/run_local_training_check.py
scripts/verify_training_artifact.py
tests/                     focused behavioral and contract checks
```

Completed private-stage notebook sources and their one-purpose wrappers were
removed after their evidence was recorded. The published sources remain active:
`notebooks/public_dicom_ordering.py` and `notebooks/public_gold_baseline.py`.

## Local checks

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

The optional training extra is needed for model and checkpoint tests:

```bash
uv sync --extra training --locked
uv run --extra training pytest -m training
```

For an exact local CPU forward/backward/checkpoint/reload check, provide the
verified weight file outside the repository:

```bash
uv run --extra training python scripts/run_local_training_check.py \
  --weights .artifacts/weights/resnet18-f37072fd.pth
```

No local validation command contacts Kaggle. Remote execution is available only
through the explicit approval gates in the consolidated command:

```bash
uv run python scripts/kaggle.py build ordering
uv run --extra training python scripts/kaggle.py build baseline --approve-gpu
uv run --extra training python scripts/kaggle.py build combined-scoring --approve-gpu
uv run --extra training python scripts/kaggle.py build plane-gating --approve-gpu
uv run --extra training python scripts/kaggle.py build dinov2-training --weight-path /path/to/dinov2_vits14_pretrain.pth --approve-gpu
uv run python scripts/kaggle.py push build/kaggle/public-dicom-ordering \
  --approve-remote
```

The GPU bundle builds additionally require `--approve-gpu`. The combined
scoring bundle is inference-only and requires the attached frozen checkpoint;
its live test index is rebuilt at scoring time. Builds are local and private by
default; the command never changes visibility or publishes a public resource.
The plane-gating bundle stops after paired validation. Its separate scoring
bundle additionally requires a recorded keep decision, exact checkpoint hash,
and checkpoint Dataset ID.
The DINOv2 training bundle packages the hash-verified weights as a managed
private Dataset; its guarded push waits for that Dataset before starting the
kernel. It has the same stop point and additionally refuses to
build from a dirty Git revision. Its scoring bundle is available only after the
precommitted validation keep rule passes.

## Project status

The DICOM, indexing, preprocessing, training, evaluation, submission, and
Kaggle-safety foundations are complete, and the repository architecture cleanup
is finished. The post-baseline evaluation and supervision roadmap is now being
realigned before further model development. Model improvements, pseudo-labels,
new folds, leaderboard decisions, and new remote runs are intentionally deferred
until that roadmap is defined.
