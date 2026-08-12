# RSNA Knee Abnormality Detection

CPU-only local foundations for the competition pipeline.

## Canonical competition contract

Later data, validation, metric, model, and submission code must consume
[`contracts/competition.json`](contracts/competition.json). Its concise evidence
summary and unresolved facts are documented in
[`docs/competition-inspection-report.md`](docs/competition-inspection-report.md).
The completed full-corpus metadata index and representative preprocessing
evidence are recorded in [`contracts/data-index.json`](contracts/data-index.json)
and [`docs/data-index-report.md`](docs/data-index-report.md).
The organizer-gold cohort, smoke-sufficiency decision, and frozen evaluation
witness are recorded in
[`contracts/gold-supervision.json`](contracts/gold-supervision.json) and
[`docs/gold-supervision-report.md`](docs/gold-supervision-report.md).
The exact ResNet-18 initialization identity, upstream/access and license
evidence, byte hash, preprocessing adapter, and offline-only construction policy
are frozen in
[`contracts/resnet18-imagenet1k-v1.json`](contracts/resnet18-imagenet1k-v1.json).
The verified one-epoch GPU smoke result, complete checkpoint evidence, and
submission reload proof are recorded in
[`contracts/gold-training.json`](contracts/gold-training.json) and
[`docs/gold-training-report.md`](docs/gold-training-report.md).
The first leakage-safe organizer-gold holdout baseline, PatientID bootstrap
uncertainty, and evidence-adequacy decision for issue #11 are recorded in
[`contracts/gold-baseline.json`](contracts/gold-baseline.json) and
[`docs/gold-baseline-report.md`](docs/gold-baseline-report.md).
The issues #2–#5 review, corrections, and proceed recommendation are recorded in
[`docs/foundation-hardening-report.md`](docs/foundation-hardening-report.md).

## Local checks

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

The issue #8 model checks use the optional training extra:

```bash
uv sync --extra training --locked
uv run --extra training pytest -m training
```

## Real-data inspection notebook

Build the private Kaggle inspection bundle locally:

```bash
uv run python scripts/build_inspection_notebook.py
```

The ignored `build/kaggle/inspect-real-data/` directory contains the generated
notebook and the exact `kernel-metadata.json` that would be pushed. The bundle
embeds the current `rsna_knee` package source so this evidence-gathering run does
not depend on the permanent wheel Dataset workflow.

The push entry point refuses to contact Kaggle unless the user has separately
approved the private CPU run. After that explicit approval, the guarded command
is:

```bash
uv run python scripts/push_inspection_notebook.py --approve-remote
```

The metadata validator requires a private notebook with CPU only, TPU off,
internet off, and only the `rsna-knee-abnormality-detection` competition input.
Pushing a Kaggle kernel also executes it, so do not run the approved command as
a build or test step. The notebook writes compact findings and ordered-slice
PNGs below `/kaggle/working/rsna-knee-inspection/`; it never decodes the full
image corpus.

## Packaged synthetic Kaggle run

Build a pure Python wheel, private Dataset metadata, and a Jupytext-authored
CPU notebook with resolved Kaggle IDs:

```bash
uv run python scripts/build_kaggle_run.py \
  --dataset-id plthon/rsna-knee-offline-wheel \
  --kernel-id plthon/rsna-knee-synthetic-pipeline
```

The ignored `build/kaggle/synthetic-run/` directory contains the wheel and its
Git commit/SHA-256 provenance, `dataset-metadata.json`, the rendered notebook,
and private CPU-only, internet-off `kernel-metadata.json`. The notebook installs
the wheel with `pip --no-index --no-deps`, generates tiny DICOMs, and calls the
same `preprocess_series` package interface as the local tests.

Building does not contact Kaggle. Dataset creation and notebook execution are
both refused unless the following explicit approval is supplied:

```bash
uv run python scripts/push_kaggle_run.py --approve-remote
```

If a prior kernel attempt failed after the Dataset became ready, a separately
approved retry can preserve and rerender against the exact uploaded wheel with
`build_kaggle_run.py --reuse-wheel-dataset`, then push with
`push_kaggle_run.py --approve-remote --reuse-ready-dataset`.

The Dataset command omits Kaggle's `--public` option, so creation remains
private. CPU metadata is the default. Creating or pushing GPU metadata also
requires the separate `--approve-gpu` flag.

After Kaggle reports a successful run, download kernel outputs (not competition
data), verify the selected output against `run-manifest.json`, and add both to
the ignored local experiment registry:

```bash
uv run python scripts/register_kaggle_run.py \
  --kernel-id plthon/rsna-knee-synthetic-pipeline \
  --artifact synthetic-output.npz
```

The compact run manifest records source, configuration, runtime, notebook, and
artifact provenance. Registered runs live under `.artifacts/experiments/runs/`
with an append-only `registry.jsonl` index.

## Public DICOM ordering notebook

Build the standalone public-facing notebook and its review metadata locally:

```bash
uv run python scripts/build_public_notebook.py
```

The ignored `build/kaggle/public-dicom-ordering/` directory contains the
rendered notebook and `kernel-metadata.json`. The notebook embeds no repository
package or private Dataset. It discovers the attached competition data,
inspects only three representative series, and remains CPU-only with internet
disabled.

The review metadata intentionally keeps the notebook private. After explicit
approval to run it on Kaggle, execute:

```bash
uv run python scripts/push_public_notebook.py --approve-remote
```

Pushing starts a Kaggle execution. Review the completed title, introduction,
statistics, code, plots, ordering claims, takeaways, and runtime before making
any visibility change. Publication is a separate human approval gate and is not
performed by either script.

## Real-data index and preprocessing check

Build the private, CPU-only, internet-off indexing notebook without contacting
Kaggle:

```bash
uv run python scripts/build_index_notebook.py
```

The generated bundle in `build/kaggle/index-real-data/` reads the verified
`train_series.csv` and `test_series.csv` fields, completes a header-only index,
and only then decodes eight deterministic slices from representative series
through the shared training/inference transform. Its versioned output contains
normalized study, series, slice, selection, error, and statistics tables. It
does not contain DICOMs, decoded arrays, or image files.

Pushing the notebook starts a remote Kaggle CPU job and therefore remains behind
an explicit approval gate:

```bash
uv run python scripts/push_index_notebook.py --approve-remote
```

After downloading a successful kernel output, verify its hashes, no-pixel
invariant, and representative preprocessing evidence locally:

```bash
uv run python scripts/verify_index_artifact.py \
  path/to/rsna-knee-index-v1
```

## Constant baseline submission notebook

Build the private, CPU-only submission notebook locally without contacting
Kaggle:

```bash
uv run python scripts/build_submission_notebook.py
```

The ignored `build/kaggle/constant-baseline-submission/` bundle embeds the
current package and verified competition contract. With internet disabled, the
notebook dynamically discovers the current test studies, writes a `0.5`
prediction for every required target, validates `submission.csv`, and records
`submission-validation.json`, `submission-fallbacks.csv`, and
`run-manifest.json`.

Pushing starts a Kaggle CPU execution and remains behind an explicit approval
gate:

```bash
uv run python scripts/push_submission_notebook.py --approve-remote
```

## Organizer-gold supervision audit

Build the private, CPU-only issue #7 audit without contacting Kaggle:

```bash
uv run python scripts/build_gold_audit_notebook.py
```

The ignored `build/kaggle/gold-supervision-audit/` bundle attaches the accepted
private `plthon/rsna-knee-data-index` kernel output and the competition mount.
It verifies the accepted study-index hashes, audits `train.csv` without
processing or exporting report text and without reading DICOM pixels, freezes
the canonical all-eligible organizer-gold cohort, and performs a complete
deterministic search for one PatientID-disjoint train/holdout eligibility
witness.

Row-level membership, labels, PatientIDs, exclusions, and assignments remain in
the private Kaggle output. The summary contains only aggregate counts, metadata,
structured findings, and cryptographic digests. Pushing starts the remote CPU
run and remains behind explicit approval:

```bash
uv run python scripts/push_gold_audit_notebook.py --approve-remote
```

After downloading the completed private artifact, verify it and export only its
public-safe aggregate contract:

```bash
uv run python scripts/verify_gold_audit_artifact.py \
  path/to/rsna-knee-gold-audit-v1
uv run python scripts/export_gold_audit_contract.py \
  path/to/rsna-knee-gold-audit-v1
```

## Gold-only ResNet-18 smoke training

The training path consumes the accepted private index and competition mount,
independently re-derives the frozen all-eligible organizer-gold cohort, and
rejects any membership/label digest mismatch before model construction. It uses
one shared torchvision ResNet-18, excludes padded slices and series from both
masked means, masks unknown targets in BCE, saves full resumable state, reloads
that state, and sends predictions through the issue #6 submission validator.
Report text is not interpreted or used, and report-derived labels do not enter
training or evaluation.

Download the one named upstream artifact outside Kaggle and run the full local
CPU forward/backward/optimizer/checkpoint/reload/prediction check:

```bash
curl --fail --location \
  --output .artifacts/weights/resnet18-f37072fd.pth \
  https://download.pytorch.org/models/resnet18-f37072fd.pth
uv run --extra training python scripts/run_local_training_check.py \
  --weights .artifacts/weights/resnet18-f37072fd.pth
```

Both bundle rendering and remote execution require a separate explicit GPU
approval. Rendering copies the hash-validated bytes into a private Kaggle
Dataset bundle; runtime internet remains disabled and the notebook validates
the mounted file again before constructing the model:

```bash
uv run --extra training python scripts/build_training_notebook.py \
  --weights .artifacts/weights/resnet18-f37072fd.pth \
  --approve-gpu
uv run --extra training python scripts/push_training_notebook.py \
  --approve-remote \
  --approve-gpu
```

Pushing creates the private versioned weight Dataset and starts a GPU notebook,
so neither command is part of local tests. If the exact Dataset is already
ready, use `--reuse-ready-dataset` on the push command. After completion,
download, register, and verify all manifest hashes, complete checkpoint fields,
approved CUDA and mixed-precision evidence, gold-only invariants, reload status,
and submission validation with:

```bash
uv run --extra training python scripts/register_training_run.py
```

`scripts/verify_training_artifact.py` performs the same verification against an
already-downloaded `rsna-knee-gold-training-v1` directory.

The generated metadata pins `NvidiaTeslaT4`: the current Kaggle PyTorch 2.10
image does not contain CUDA kernels for the older default Tesla P100. The
completed private evidence run is kernel version 2; see the training report for
the preserved version-1 preflight failure and successful T4 verification.
