# RSNA Knee Abnormality Detection

CPU-only local foundations for the competition pipeline.

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
