# Test-suite maintenance (#39)

## Measured baseline (before test edits)

Source: `50207600591f419f8cc64353bf0d001fc69dd56e` on
`issue-35-resolution-336`. The pre-existing deleted `CLAUDE.md` and untracked
`.claude/` directory are outside this change.

Measured locally on macOS, Python 3.12.13, pytest 9.1.1, with the locked training
extra installed:

- 496 collected, 496 passed, 0 skipped; two existing pydicom deprecation warnings.
- 32 test modules, 13,029 lines; 33 Python files / 13,032 lines including conftest.
- Full pytest wall time: 76.76 seconds (pytest session measurement).
- Largest modules: Kaggle 1,341 lines; combined supervision 1,000; reports 753;
  DINOv2 739; Raptor reference 676; report LLM 664; report source comparison 661;
  resolution 641; training 635; public label audit 470.
- Slowest calls: Raptor geometry reader 9.31s; published Raptor checkpoint reload
  8.56s; soft-label parquet identity 7.47s; DINOv2 offline build/gradients 7.12s;
  live 336 checkpoint inference 1.90s. These protect different integration seams,
  not interchangeable speed-test candidates.
- Baseline gates passed: both locked sync commands, Ruff check, Ruff formatting
  (611 files), training-extra mypy (85 source files), training-extra full pytest,
  wheel/sdist build, and `git diff --check`.

Commands: `uv sync --locked`, `uv sync --extra training --locked`,
`uv run ruff check .`, `uv run ruff format --check .`,
`uv run --extra training pytest --durations=15`,
`uv run --extra training mypy src tests scripts notebooks`, `uv build`,
`git diff --check`. Counts use `tests/test_*.py`; LOC counts physical lines,
including blank lines. Supporting Python files are reported separately to avoid
hiding extraction cost.

## Audit and consolidation plan (recorded before test edits)

1. **Safe helper extraction, highest value first:** move the cross-imported
   synthetic DICOM writer and ordered-series builder out of test modules into
   focused DICOM support. Consolidate equivalent secondary-capture writers while
   preserving pixel arrays, optional tags, and identity/geometry differences.
   Extract the identical training/combined-cohort CSV schemas, organizer rows,
   indexed-study rows, CSV writer and file hash helper. Keep the two top-level
   cohort/artifact fixtures local: they encode different gold/public governance
   and provenance requirements. Do not introduce a fixture framework.
2. **Safe parametrization:** combine the three public-label CSV rejection cases
   (missing study, invalid verdict, duplicate study), retaining independent case
   IDs, original mutations and exact error checks. Keep the stale-hash rejection
   separate from semantically invalid data with an updated, matching hash.
3. **Candidate redundant tests:** none selected for deletion. Shared setup is
   redundant; the scientific and safety assertions are not. Preserve every
   existing case, including rejected-experiment behavior still supported in code.
4. **Apparently repetitive tests retained:** geometry/rescale/MONOCHROME1,
   crop/fallback, 224-to-336 invariance, adjacent-slice substitution, clipped-window
   failure policy, checkpoint structure/reload/hash-before-load, gold-only versus
   combined cohort, remote/GPU approval, containment, discovery, and Raptor/CoAtNet
   compatibility exercise distinct boundaries. Share setup, not these assertions.

Other audited duplication includes CSV read/write variants in holdout, indexing,
submission and scoring; canonical JSON in training and holdout; wheel/weight
artifact scaffolding in Kaggle; checkpoint payloads across experiment modules;
control configuration and target/provenance setup. Keep specialized encodings,
checkpoint schemas, literal contract assertions and tiny one-use helpers local
rather than centralizing mechanically. Kaggle artifact construction and deeper
index-manifest construction are follow-up candidates, not prerequisites for this
focused pass. Many crop, spacing, sampling and report-label cases are already
parametrized.

Existing public-behavior tests are the refactoring safety net; no new production
seam or red/green feature cycle is needed. Run the relevant files and mypy after
each extraction, then all required gates and the complete suite at the end.

## Completed changes and coverage accounting

- `tests/dicom_support.py` owns `write_dicom` and `write_ordered_series`. Four
  local DICOM writers (synthetic pipeline, combined scoring, combined supervision,
  submission) now share the same header construction. The original arrays and
  dimensions remain explicit at callers; missing rescale tags remain absent via
  `slope=None, intercept=None`. The ordered-series helper moved unchanged from
  adjacent-slice tests. All seven cross-test-module helper imports are gone.
- `tests/cohort_support.py` owns the shared organizer/index table columns,
  contract-derived targets, `train_row`, `study_row`, `write_csv`, and
  `sha256_file`. Training and combined supervision no longer each define these
  four helpers and schemas. Patient IDs, splits, unknown-label empty cells and
  the private-report sentinel remain unchanged. No pytest fixture was introduced.
- Three public-label rejection functions became three independently reported
  cases of `test_public_label_semantics_rejected_with_matching_hash`. Each still
  changes the same CSV content, updates its recorded hash and checks the same
  error message. There are no dropped test scenarios or assertions.

Exact old-to-new test mapping (all in `test_combined_supervision.py`):

| Old function | Surviving parameterized case |
| --- | --- |
| `test_missing_non_gold_study_in_public_dataset_raises` | `test_public_label_semantics_rejected_with_matching_hash[missing-non-gold-study]` |
| `test_invalid_verdict_raises` | `test_public_label_semantics_rejected_with_matching_hash[invalid-verdict]` |
| `test_duplicate_study_rejected_in_public_csv` | `test_public_label_semantics_rejected_with_matching_hash[duplicate-study]` |

A before/after pytest node-ID comparison confirmed exactly these three mappings;
all other 493 identities are unchanged. No test was deleted without replacement.

Deliberately separate: gold-only and combined cohort fixtures; stale public-file
hash rejection; organizer CSV duplicates; target-order mismatch; all distinct
preprocessing, checkpoint, provenance, approval and governance tests listed in
the plan. Indexing's richer DICOM metadata writer and public-notebook construction
remain local rather than adding identity/geometry options to a generic factory.
Canonical JSON remains local because this change does not need a second consumer.
No production files, model configuration, dependency declarations or lockfiles
changed. Production behavior did not intentionally change.

## Before/after evidence

Exact source revisions:

- Before: `50207600591f419f8cc64353bf0d001fc69dd56e`.
- After test refactor: `af32c2f29b1374c7d74dba995fe4d9867d25c984`.
- This report is committed separately after the source commit so it can record
  that immutable after SHA; the report commit does not change executable source.

| Measurement | Before | After |
| --- | ---: | ---: |
| Collected / passed / skipped | 496 / 496 / 0 | 496 / 496 / 0 |
| Test modules (`test_*.py`) | 32 | 32 |
| Test-module physical LOC | 13,029 | 12,766 |
| Python files including test support | 33 | 35 |
| Physical LOC including test support | 13,032 | 12,915 |
| Full pytest wall time | 76.76 s | 29.04 s |
| Existing pydicom deprecation warnings | 2 | 2 |

The net reduction including support is 117 lines. Test count was not a target.
The first complete post-refactor run took 56.99s; the final run after the review's
naming cleanup took 29.04s. These are not controlled performance benchmarks:
unchanged Raptor/DINOv2 integration calls account for much of the variation.
The final slowest calls were Raptor checkpoint reload (3.30s), DINOv2 offline
build/gradients (3.04s), parquet identity (1.89s), and local CPU checkpoint/reload
(1.89s). No speedup is attributed to this maintenance change.

## Verification

All required baseline and final gates passed:

| Command | Baseline | Final |
| --- | --- | --- |
| `uv sync --locked` | Pass | Pass |
| `uv sync --extra training --locked` | Pass | Pass |
| `uv run ruff check .` | Pass | Pass |
| `uv run ruff format --check .` | Pass | Pass |
| `uv run --extra training pytest --durations=15` | 496 passed | 496 passed |
| `uv run --extra training mypy src tests scripts notebooks` | Pass, 85 files | Pass, 87 files |
| `uv build` | Wheel and sdist pass | Wheel and sdist pass |
| `git diff --check` | Pass | Pass |

Incremental verification: 144 extraction-focused tests passed in 17.12s; all 17
combined-supervision cases passed after parametrization in 4.95s. The extracted
index schema tuples were also compared against both originals and matched.

During implementation Ruff caught import ordering and unused imports; these were
fixed. An attempted `mypy tests` invocation omitted the repository's source roots
and produced untyped-import errors. The next source-aware incremental run reported
five module-attribute errors; a source-aware `--no-incremental` run passed, as did
subsequent runs of the exact required command. No production or typing-policy
change was made to accommodate that invocation/cache issue.

## Review

### Standards

No hard documented-standard violations. One low-priority naming observation was
addressed: adjacent-slice and clipped-window tests now import and call
`write_ordered_series` directly instead of aliasing it to `_series`.

### Spec

No findings: the helper extractions preserve DICOM tags, pixels, schema columns,
cohort rows and provenance setup; all three consolidated rejection scenarios
retain their mutations and exact errors. There is no production change or test
framework expansion. Both reviews were read-only and compared shared-checkout
files with baseline content; their isolated harness did not permit running Git
against the shared checkout. The main session performed the actual Git diff and
node-ID comparisons.

Review total: one standards readability observation (fixed), zero spec findings;
no outstanding findings on either axis.

## Remaining maintenance opportunities

Kaggle wheel/weight dataset-artifact scaffolding and the repeated series/slice
rows plus index manifests in the two cohort fixtures are the highest-cost
remaining duplication. A future focused change could share their common records
while keeping gold/public provenance differences explicit. CSV variants in
indexing, holdout and scoring and specialized checkpoint payloads remain local;
no large fixture hierarchy, broad checkpoint factory or module split was added.
