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
