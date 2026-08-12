# Organizer-gold supervision audit

Verified 2026-08-12 for `rsna-knee-abnormality-detection`. The canonical
machine-readable aggregate contract is
[`contracts/gold-supervision.json`](../contracts/gold-supervision.json).
Row-level cohort membership, labels, PatientIDs, exclusions, and assignments
remain in the private Kaggle artifact and are not committed.

## Platform run

Private Kaggle kernel
[`plthon/rsna-knee-gold-supervision-audit`](https://www.kaggle.com/code/plthon/rsna-knee-gold-supervision-audit),
version 1, completed successfully on CPU with internet, GPU, and TPU disabled.
The aggregate result was written after 587.10 seconds; Kaggle completed notebook
conversion after 596.54 seconds.

The audit consumed:

- `contracts/competition.json`, SHA-256
  `3c6d42d4a4883a9626294c79af02f2ff687ba9413da8f316db9b877ebb1e2292`;
- `contracts/data-index.json`, SHA-256
  `3e9c66134a11e9f6ebcb79027a81deae92d43618f28984c740a5fe79e9a324d3`;
- the mounted `train.csv`; and
- the accepted private index manifest and `studies.csv`, whose SHA-256 values
  are `85970899...80caa` and `a8331f8e...62c2`.

The run used source commit `5bc4b7d` with a dirty worktree. Its exact Jupytext
source and embedded package archive have SHA-256 values
`b6aac2d8...8501b` and `62e2ed9e...438e1`. The verified public-safe summary has
SHA-256 `80747838...cc1d9`; the downloaded kernel log has SHA-256
`d155b73f...b87b`.

## Gold-label contract

All 4,407 `train.csv` rows were audited against the verified ordered 12-target
schema. Values outside `{0, 1, missing}` numbered zero. Joint row-wise
completeness is now verified:

| Known target count in a row | Rows |
| ---: | ---: |
| 0 | 4,349 |
| 12 | 58 |

No partially labeled row exists in the mounted snapshot. The 58 organizer-gold
rows are therefore complete across all targets, although the implementation and
digest format continue to preserve missing values as masked `null` values.

| Target | Known | Missing in source | Negative | Positive |
| --- | ---: | ---: | ---: | ---: |
| ACL | 58 | 4,349 | 34 | 24 |
| MCL | 58 | 4,349 | 49 | 9 |
| Medial Meniscus | 58 | 4,349 | 32 | 26 |
| Lateral Meniscus | 58 | 4,349 | 35 | 23 |
| Medial OA | 58 | 4,349 | 43 | 15 |
| Lateral OA | 58 | 4,349 | 47 | 11 |
| PF OA | 58 | 4,349 | 37 | 21 |
| Effusion | 58 | 4,349 | 23 | 35 |
| Synovitis | 58 | 4,349 | 31 | 27 |
| Baker's | 58 | 4,349 | 46 | 12 |
| Contusion | 58 | 4,349 | 39 | 19 |
| Fracture | 58 | 4,349 | 40 | 18 |

## Canonical smoke cohort

Selection rule `organizer-gold-all-eligible-v1` includes every organizer-labeled
study with at least one nonmissing verified target that resolves through the
authoritative index to exactly one usable study. It uses no sampling, set cover,
hand selection, or class-balancing selection.

All 58 labeled studies resolved to one usable indexed training study. The other
4,349 studies were excluded only because they have no organizer gold label.
The canonical membership/label SHA-256 is
`ea739a412709b0ac65429ca47f29161614ba384e2e92643301f817712cd6b7bc`.
Its complete canonical representation is versioned in the machine-readable
contract so issue #8 can independently rederive and compare it.

Every target has at least one gold positive and one gold negative. The decision
is therefore **`smoke-sufficient`**. There are no unsupported targets, no
substitute cohort, no weakened target contract, and no pseudo-label activation.
The 58-study cohort does not conflict with issue #8's declared one-epoch bounded
smoke objective.

## Evaluation eligibility witness

Allocation rule `patient-disjoint-exact-backtracking-v1`, seed `20260812`, froze
one exact PatientID-disjoint assignment before any model predictions existed.
It uses neither predictions nor leaderboard results and satisfies only the
declared leakage and both-class constraints.

The witness contains 54 patients/studies in training and 4 patients/studies in
holdout. No PatientID crosses partitions. Both classes occur for every target
in both partitions:

| Target | Train negative | Train positive | Holdout negative | Holdout positive |
| --- | ---: | ---: | ---: | ---: |
| ACL | 33 | 21 | 1 | 3 |
| MCL | 46 | 8 | 3 | 1 |
| Medial Meniscus | 30 | 24 | 2 | 2 |
| Lateral Meniscus | 32 | 22 | 3 | 1 |
| Medial OA | 41 | 13 | 2 | 2 |
| Lateral OA | 44 | 10 | 3 | 1 |
| PF OA | 36 | 18 | 1 | 3 |
| Effusion | 21 | 33 | 2 | 2 |
| Synovitis | 29 | 25 | 2 | 2 |
| Baker's | 43 | 11 | 3 | 1 |
| Contusion | 37 | 17 | 2 | 2 |
| Fracture | 37 | 17 | 3 | 1 |

The assignment SHA-256 is
`8dccc31852a3e26ec677062a289301004fdce29eb64a556bea8922dfd2bb263d`.
The evaluation decision is **`Candidate`**. This establishes only that a valid
gold-only attempt is possible; it makes no claim of statistical adequacy. Issue
#9 must independently rederive this exact assignment and reject a digest
mismatch.

## Report-derived-supervision recommendation

Organizer gold covers 58 of 4,407 usable indexed training studies. Additional
report-derived supervision could plausibly address that training-coverage
limitation, but this is informational only. It did not enter either decision,
was not treated as organizer gold, activated no report processing or pseudo-label
work, and created no downstream ticket.

## Privacy and artifact verification

The private Kaggle artifact contains hashed row-level files for the cohort,
exclusions, invalid-label ledger, and exact assignment. The local verifier
recomputed aggregate counts, cohort and assignment digests, PatientID
disjointness, full assignment coverage, and both-class support in each
partition. It also verified every private file's manifest hash and size.

Only the byte-identical aggregate summary was copied to
`contracts/gold-supervision.json`. No StudyInstanceUID, PatientID, row-level
label, or row-level assignment from the competition data is committed.

## Acceptance-criterion map

| Criterion | Evidence | Outcome |
| --- | --- | --- |
| Joint completeness and class counts | 58 fully known rows, 4,349 fully missing rows, per-target table | Satisfied |
| Reject values outside `{0,1,missing}` | Strict parser and private invalid-value ledger; observed count zero | Satisfied |
| Every selected label resolves once to a usable study | Accepted index hashes plus 58/58 unique usable links | Satisfied |
| Freeze all-eligible cohort and digest | Rule `organizer-gold-all-eligible-v1`; cohort SHA-256 | Satisfied |
| Decide smoke support for all targets | Both classes present for all 12; `smoke-sufficient` | Satisfied |
| Unsupported targets explicit; no pseudo-label activation | Empty unsupported list; activation false | Satisfied |
| Freeze exact patient-disjoint witness or explain ineligibility | 54/4 exact witness and assignment SHA-256 | Satisfied |
| Both classes in both partitions | Per-target partition table and verifier | Satisfied |
| Keep private row data out of Git | Private artifact inventory; aggregate-only committed contract | Satisfied |
| Report-derived supervision is non-activating | Informational recommendation with all activation flags false | Satisfied |
| Require approval for remote execution | Guard test plus explicit approval before version 1 push | Satisfied |

Issues #8 and #9 may proceed under their existing preconditions. Issue #8 must
match the cohort digest; issue #9 must additionally match the assignment digest.
