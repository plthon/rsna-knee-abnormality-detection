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
The issues #2–#5 review, corrections, and proceed recommendation are recorded in
[`docs/foundation-hardening-report.md`](docs/foundation-hardening-report.md).

## Local checks

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest
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
