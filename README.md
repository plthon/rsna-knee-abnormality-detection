# RSNA Knee Abnormality Detection

Code for the RSNA Knee Abnormalities Detection Kaggle competition. The pipeline
reads knee MRI DICOM series and outputs one probability for each of the 12
targets in [`contracts/competition.json`](contracts/competition.json). The
project is finished and is no longer maintained.

## Results

| Model | Issue | Evidence | Report |
|-------|-------|----------|--------|
| Public Raptor CoAtNet reproduction | #36 | Public LB **0.924**, exact parity with the published notebook | [raptor-reference-report.md](docs/raptor-reference-report.md) |
| Raptor derivative trained on Kaggle T4 (batch 4, accumulation 2, 16 epochs) | #38 | Best organizer-gold macro ROC-AUC 0.9076 (epoch 11); not submitted | [raptor-t4-derivative-report.md](docs/raptor-t4-derivative-report.md) |
| ResNet-18 with 130 mm physical crop at 224 px (best internal model) | #31 | Public LB 0.830 | [physical-crop-report.md](docs/physical-crop-report.md) |

All of these experiments on top of #31 were rejected by their precommitted
keep rules: plane gating (#28), DINOv2 (#29, #30), adjacent-slice (#32) and
clipped-window (#33) 2.5D inputs, slot attention (#34), and 336 px resolution
(#35). The report-derived weak-supervision line (#20–#26) was not used for the
final models. Its code and contracts are still here. Its reports are only in
git history.

## Layout

```text
contracts/      verified schemas, weight hashes and frozen experiment contracts
configs/        TOML configs for each experiment
docs/           reports for the kept models
notebooks/      Jupytext sources for the Kaggle notebooks
src/rsna_knee/  DICOM, indexing, preprocessing, training, scoring, Kaggle bundling
scripts/        kaggle.py (guarded build/push/pull), local checks, Raptor runner
tests/          CPU tests on synthetic DICOMs
```

## Local checks

```bash
uv sync --locked
uv run ruff check .
uv run pytest
```

Model tests need the training extra:

```bash
uv sync --extra training --locked
uv run --extra training pytest -m training
```

## Kaggle

Bundles are built locally and are private by default. GPU bundles need
`--approve-gpu` and pushes need `--approve-remote`:

```bash
uv run --extra training python scripts/kaggle.py build raptor-scoring --approve-gpu
uv run --extra training python scripts/kaggle.py build physical-crop-scoring --approve-gpu
uv run python scripts/kaggle.py push build/kaggle/<bundle> --approve-remote --approve-gpu
```

Run `uv run python scripts/kaggle.py build --help` to see every target.
