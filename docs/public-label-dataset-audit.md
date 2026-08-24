# Public label dataset audit

## Status

Complete. Private Kaggle kernel `plthon/rsna-knee-public-label-audit`, version 4,
ran successfully on CPU with internet disabled. The in-notebook audit took 1.09
seconds and recorded source commit
`675b01eaf441ff15a7b7dfbc0c672bb9d4319737` with a clean worktree.

Only these aggregate JSON outputs were retained locally:

| Output | Bytes | SHA-256 |
| --- | ---: | --- |
| `public-label-audit-manifest.json` | 8,451 | `8a03181bbb29038f420d09941c1d08e3af3f31a259e7355aa8e9adc19551f367` |
| `public-label-comparison-summary.json` | 7,747 | `63f93499dc8a93aaa23a5a27f5cb6685886537fba276b380d2c7d0d26a21d8e1` |
| `public-label-audit-summary.json` | 15,998 | `2b3600568e95604718ac22f061b234a65e0dfd1d54f15de378f61f073be24348` |

No row-level validation output, raw report, organizer target value, or study
identifier was retrieved.

## Frozen dataset identity

- Title: `RSNA Knee LLM-read report labels`
- Owner/slug: `pilkwang/rsna-knee-llm-labels`
- Dataset ID: `11529209`
- Version: `1`
- URL: <https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels/versions/1>
- License: `CC0-1.0`

The exact-title Kaggle search returned one match. The cited public notebook
`pilkwang/rsna-knee-baseline-v1` attaches that owner/slug, and Kaggle reports no
version 2. Version 1 was publicly attachable during the audit. Its complete
file inventory matched the frozen sizes and SHA-256 hashes:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `api_labeler.py` | 12,976 | `045f0f858b20ee477869c999c4bc9b484154cb5ddfd507b79e7c732632abc52e` |
| `report_labels_v2.csv` | 1,004,839 | `6f704a7bdb2f894cc49445b19ba7c4378c3f548d3449e00361e10044bee40920` |

The owner, slug, version, URL, file inventory, metadata text, and hashes are
frozen in `contracts/public-report-label-dataset-v1.json`. Referencing version
1 and verifying every file fails closed if the public asset changes.

Kaggle metadata describes twelve per-study scores with confidence and verdict
fields derived from reports in their original language. The exact subtitle and
description text and their digests are frozen in the contract. The metadata
names no update frequency and contains no separate README.

## Schema and value semantics

`report_labels_v2.csv` is wide format with one row per study. It has 4,406
rows, 4,406 unique nonblank identifiers, and 37 columns. `StudyInstanceUID` is
followed in exact competition target order by three columns per target:

- the target score;
- `{target}__conf`; and
- `{target}__verdict`.

All twelve target scores use the observed domain
`{0.08, 0.28, 0.68, 0.82, 0.94}`. Confidence uses
`{0.05, 0.85, 0.95}`, and verdict uses `{YES, NO, UNK}`. Every target has 4,406
nonmissing values and no invalid score, confidence, or verdict. There are no
duplicate or missing identifiers, malformed rows, missing targets, extra
targets, or ordering errors.

Dataset documentation says the scores are rank-usable, not calibrated
probabilities. A responsible training mapping is `YES -> positive`,
`NO -> negative`, and `UNK -> unknown/masked`. This preserves the documented
verdict semantics but cannot recreate #20's separate `uncertain` state. The
score may be retained as ordinal confidence, not interpreted as probability.

## Study coverage

The public identifiers form an exact 4,406-study subset of the 4,407
competition studies:

| Cohort | Competition | Public overlap | Missing | Extra |
| --- | ---: | ---: | ---: | ---: |
| All studies | 4,407 | 4,406 | 1 | 0 |
| Organizer-gold | 58 | 57 | 1 | 0 |
| Non-gold | 4,349 | 4,349 | 0 | 0 |

The absent study is in the organizer-gold cohort. The audit established this
using only study IDs and target-column missingness/nonmissingness; it never read
or used an organizer target value. The dataset therefore fully covers the
intended non-gold weak-supervision cohort but must not be treated as a complete
4,407-study label source.

## Provenance

Provenance is `partial`.

Dataset metadata documents constrained twelve-finding verdict/severity output,
multilingual handling, and score semantics, but calls the generator only a
language model. It does not name the family, full prompt, immutable model/API
version, dependency version, or invocation that produced the CSV. The included
source suggests an Anthropic client and a default model string of
`claude-opus-5`; source defaults are not proof of the run that produced the
labels and do not upgrade the metadata-only classification.

The included source and cited notebook disclose comparison with annotated
studies when choosing or tuning report readers. They do not freeze exactly how
the 58 organizer annotations affected the emitted public labels. Any later
evaluation must isolate this influence.

## Structural comparison with #20

The private comparison input was verified before use:
`report-labels-v1.csv`, 52,884 rows, 12,409,688 bytes, SHA-256
`6651f60b00e5bec292e8c98e7fa83c75e8aa5af5931414a9a68c8ce10a52d527`.
It remained on Kaggle. All targets had 4,406 study overlaps. Agreement was
computed only where #20 was `positive` or `negative` and the public verdict was
decisive. #20 `uncertain` and `unknown` rows and public `UNK` verdicts were
excluded.

| Target | Public YES | #20 + / - / uncertain / unknown | Compared | Agreement | Public+ / #20- | Public- / #20+ | Public UNK excluded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACL | 27.8% | 634 / 710 / 24 / 3,038 | 1,343 | 85.5% | 5 | 190 | 1 |
| MCL | 17.4% | 853 / 481 / 27 / 3,045 | 1,333 | 60.9% | 3 | 518 | 1 |
| Medial Meniscus | 53.3% | 727 / 347 / 19 / 3,313 | 1,073 | 92.5% | 6 | 75 | 1 |
| Lateral Meniscus | 25.4% | 506 / 632 / 13 / 3,255 | 1,133 | 83.7% | 4 | 181 | 5 |
| Medial OA | 36.7% | 0 / 0 / 0 / 4,406 | 0 | n/a | 0 | 0 | 0 |
| Lateral OA | 24.7% | 0 / 0 / 0 / 4,406 | 0 | n/a | 0 | 0 | 0 |
| PF OA | 45.2% | 21 / 0 / 0 / 4,385 | 21 | 100.0% | 0 | 0 | 0 |
| Effusion | 60.5% | 957 / 405 / 8 / 3,036 | 1,362 | 99.0% | 2 | 11 | 0 |
| Synovitis | 13.2% | 259 / 12 / 3 / 4,132 | 269 | 99.6% | 0 | 1 | 2 |
| Baker's | 25.3% | 347 / 245 / 3 / 3,811 | 592 | 99.2% | 1 | 4 | 0 |
| Contusion | 29.4% | 150 / 153 / 19 / 4,084 | 302 | 96.7% | 1 | 9 | 1 |
| Fracture | 7.1% | 91 / 496 / 11 / 3,808 | 586 | 99.0% | 0 | 6 | 1 |

The comparison is structural, not clinical validation. MCL has the lowest
decisive agreement and 518 public-negative/#20-positive conflicts. Medial OA and
Lateral OA cannot be compared because #20 is unknown for every overlapping
study. PF OA's 100% is based on only 21 decisive positive rows and no decisive
negative rows. No result establishes that either report-derived source is
correct.

## Caveats and risks

- One organizer-gold study is absent; the dataset is not a complete
  competition-study label source.
- Scores are ordinal/rank-oriented, not calibrated probabilities.
- `UNK` is absence of a definite report verdict, not a negative label.
- Public labels do not express #20's separate `uncertain` state.
- Exact generator runtime and model/API version are not reproducible from the
  dataset alone.
- Organizer annotations influenced reader evaluation or selection in an
  incompletely documented way.
- Cross-source agreement is uneven and is unavailable for both OA targets.

## Recommended next action

Treat the public dataset as an additional candidate weak-supervision source for
the 4,349 non-gold studies when rewriting and freezing #21. Exclude all 58 gold
studies from weak-label training and source selection, map verdicts explicitly,
mask `UNK`, retain source identity, and do not treat the scores as calibrated
probabilities. Compare it with #23 under a leakage-aware evaluation design
rather than silently merging or replacing sources.

## Final decision

The identity and schema are frozen and valid, all non-gold studies are covered,
and verdict semantics permit responsible experimental use. Direct unrestricted
use is not justified because coverage is incomplete, provenance is partial,
gold-label influence is not fully documented, score calibration is absent, and
agreement with #20 is materially uneven for some targets.

`usable_with_restrictions`
