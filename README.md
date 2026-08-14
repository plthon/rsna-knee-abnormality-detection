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

The recorded gold-only run reports macro AUC 0.6111 on a frozen holdout of four
patients. Its PatientID bootstrap has 9,065 undefined macro-AUC replicates out
of 10,000, so this is pipeline evidence, not a reliable model-comparison
benchmark.

## Repository layout

```text
contracts/                 verified schemas and committed evidence
docs/                      immutable inspection and experiment reports
notebooks/                 two current public notebook sources
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
uv run python scripts/kaggle.py push build/kaggle/public-dicom-ordering \
  --approve-remote
```

The baseline push additionally requires `--approve-gpu`. Builds are local and
private by default; the command never changes visibility or publishes a public
resource.

## Project status

The DICOM, indexing, preprocessing, training, evaluation, submission, and
Kaggle-safety foundations are complete, and the repository architecture cleanup
is finished. The post-baseline evaluation and supervision roadmap is now being
realigned before further model development. Model improvements, pseudo-labels,
new folds, leaderboard decisions, and new remote runs are intentionally deferred
until that roadmap is defined.
