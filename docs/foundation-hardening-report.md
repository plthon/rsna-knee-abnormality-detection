# Foundation hardening report

Issue #17 reviewed the completed work from issues #2 through #5 against their
original acceptance criteria, the canonical contracts, retained Kaggle evidence,
current source, tests, repository policy, and relevant Git history. No remote job,
GPU action, public-resource mutation, or private-data download was performed.

## Fixed findings

| Severity | Issue | Files | Demonstrated problem | Correction | Regression evidence |
| --- | --- | --- | --- | --- | --- |
| High | #2, #5 | `src/rsna_knee/dicom.py`, `src/rsna_knee/preprocessing.py` | Pydicom applies rescale only when both tags exist, while the index recorded independent slope/intercept defaults. A one-tag DICOM could therefore be indexed with one transform and decoded with another. Non-finite rescale values could also reach percentile normalization and produce invalid tensors. | Apply finite slope/intercept values with independent DICOM defaults, preserve modality-LUT precedence, reject non-finite decoded values, and retain structured slice failures. | `test_rescale_fallbacks_and_nonfinite_values_cannot_silently_change_output` covers slope-only, intercept-only, constant-intensity, and NaN rescale cases. |
| High | #3 | `notebooks/run_synthetic_pipeline.py`, `src/rsna_knee/kaggle_run.py` | A configuration changed after notebook rendering passed the push validator; the notebook would consume the changed file while its manifest reported the old SHA-256. | Verify the mounted configuration at notebook runtime and reject a local bundle whose current configuration digest is absent from the rendered notebook. | `test_remote_push_rejects_configuration_changed_after_render` proves rejection occurs before any Kaggle command. |
| Medium | #3 | `src/rsna_knee/kaggle_run.py` | Wheel dirty-state provenance used `--untracked-files=no`; an untracked Python module could be packaged while provenance claimed a clean tree. Capturing status after building could also count generated output rather than source state. | Capture commit and source status before building and include all non-ignored untracked files. | `test_wheel_provenance_includes_untracked_package_sources` builds a wheel containing an untracked module and requires `git_dirty: true`. |
| High | #3 | `src/rsna_knee/kaggle_run.py` | Registering a second downloaded output with the same run ID copied over the existing manifest and artifacts before collision checking; an identical registry entry could therefore hide changed bytes. | Verify every selected artifact before mutation, compare existing registered bytes for idempotent repeats, and reject run-directory or registry collisions before copying. Duplicate selected or manifest artifact paths are also rejected. | `test_existing_registered_run_cannot_be_overwritten_by_colliding_output` proves the original manifest and artifact survive a colliding download unchanged. |
| High | #5 | `src/rsna_knee/indexing.py` | The artifact verifier accepted an incomplete manifest inventory and considered a representative series successful when only one selected slice decoded. Shape, mask, range, percentile, and selected/decode-count consistency were not enforced. | Require the exact six-file inventory and exact table columns. Representative checks now pass only when all selected slices decode into the configured finite, bounded shape with a consistent mask and finite percentile bounds. | `test_competition_index_is_ordered_compact_and_header_first` now demonstrates rejection of both an omitted slice table and internally inconsistent representative evidence. |
| Medium | #4 | `contracts/competition.json`, `contracts/data-index.json`, `docs/competition-inspection-report.md`, `docs/data-index-report.md` | The issue #4 contract still classified mounted transfer-syntax frequencies as unresolved after issue #5 counted Explicit VR Little Endian in all 819,635 mounted headers. An unqualified all-competition claim would still overreach because Kaggle replaces the example test set and the official documentation describes a mix. | Verify the mounted snapshot frequency, keep hidden scoring-set frequencies unresolved, cite the full-corpus index, and retain both the historical input-contract hash and the current hardened-contract hash. | Contract tests assert the scope-specific verified and unresolved states and the two-hash provenance chain. |
| Low | #4 | `notebooks/inspect_real_data.py` | The repository-wide strict mypy invocation failed on the Kaggle-provided optional `IPython.display` import. | Add a precise import-only mypy suppression rather than a speculative local IPython dependency. | Strict mypy passes across source, tests, scripts, and notebook sources. |

## Reviewed but unchanged

- The Python 3.12 `uv.lock` resolves without changes, the package imports in the
  test environment, and two independent wheel builds are byte-identical pure
  Python `py3-none-any` wheels.
- Synthetic DICOM generation, positive/negative rescale behavior, MONOCHROME1
  inversion, deterministic long/short sampling, mask padding, bounded
  normalization, CPU forcing, local/Kaggle path modes, Git exclusions, and the
  shared `AGENTS.md`/`CLAUDE.md` policy link remain appropriate.
- Jupytext rendering, private CPU/internet-off Kaggle metadata, placeholder
  rejection, explicit remote approval, independent GPU approval, offline wheel
  installation, wheel/output SHA-256 checks, and retained successful CPU-run
  evidence remain appropriate.
- Target names/order, identifiers, table hierarchy, report availability and
  duplicates, label missingness, patient-level split grouping, metric,
  submission contract, runtime/package provenance, and rule classifications are
  supported by the retained evidence at their recorded scopes.
- Header-first indexing, verified table joins, geometry projection, deterministic
  instance/filename fallback, tie handling, transfer-syntax counting, corrupt and
  empty-series errors, inconsistent-dimension reporting, eight-slice selection,
  short-series masks, MONOCHROME1, aspect-preserving resize, center padding,
  shared train/inference preprocessing, and the no-decoded-pixel artifact policy
  remain appropriate.
- The retained Kaggle artifact hashes match the hashes cited by the competition
  and data-index contracts. The successful real run decoded eight representative
  series and did not claim full-corpus pixel decoding.

## Observations not implemented

- Percentile normalization remains a provisional modelling choice. The current
  smoke evidence proves execution and finite output, not comparative suitability.
- Report languages, joint row-wise target missingness, global SOP Instance UID
  uniqueness, the official disk quota, and hidden scoring-set transfer syntaxes
  remain explicitly unresolved. No new remote inspection was justified for this
  hardening pass.
- No optional decoder, framework, cache, model, pseudo-label extractor, or
  architecture refactor was added because the mounted evidence does not
  demonstrate a need.

## Validation

- `uv sync --locked --dry-run`
- `uv run pytest -q` — 39 passed
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run mypy src tests scripts notebooks`
- two independent `uv build --wheel` outputs compared by SHA-256
- `git diff --check`
- retained Kaggle manifest, statistics, log, inspection, and identifier-evidence
  hashes checked against their contracts

## Proceed recommendation

No Critical or High foundation defect remains open after these corrections.

- Early public Kaggle notebook: proceed.
- Issue #6, schema-valid constant baseline submission: proceed.
- Issue #7, label-source decision and conditional pseudo-label work: proceed.
- Model training: wait until issues #6 and #7 establish the end-to-end submission
  path and training-label source. The foundation itself is no longer the blocker.
