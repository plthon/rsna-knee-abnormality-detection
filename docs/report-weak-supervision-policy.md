# Report weak-supervision policy v1

Verified 2026-08-27 for issue #26. The machine-readable policy is
[`contracts/report-weak-supervision-v1.json`](../contracts/report-weak-supervision-v1.json).
It freezes candidates for the first organizer-gold OOF image-model comparison;
it does not select a winning source.

## Why direct candidates replace manual adjudication

The previous #21 manual-adjudication design was retired because agreement
between report-derived sources is not clinical ground truth. The completed #25
comparison provides coverage and disagreement evidence only; it cannot establish
which source is correct. Issue #26 therefore precommits simple direct candidates
and leaves empirical selection to #10 through #22.

## Frozen source identities

| Source | Artifact | Size | SHA-256 | v1 role |
| --- | --- | ---: | --- | --- |
| #20 rules | `contracts/report-extractor-v1.json` | 15,896 | `901e8a53173fef0d85e12ed93cbea05dd5b8cfc900ef111e0125cb751634156a` | diagnostic reference only |
| #20 rules | `report-labels-v1.csv` | 12,409,688 | `6651f60b00e5bec292e8c98e7fa83c75e8aa5af5931414a9a68c8ce10a52d527` | diagnostic reference only |
| #23 Qwen | `contracts/report-llm-extractor-v1.json` | 4,364 | `c428d1b44bb6c87f84e7423feeaefa6c635a5ba1cb5dd277468128d7df0bbe51` | direct candidate |
| #23 Qwen | `report-llm-labels-v1.csv` | 17,063,200 | `d78c7f1d27437f6d2870e48fbd730a57c60d34380720cd4e0e07811a699db52a` | direct candidate |
| #23 Qwen | `report-llm-study-status-v1.csv` | 1,079,236 | `151d0554cbd7622028df1bd58a4e2c37d18796c5860502cf3eb0917ec6baf923` | parse-failure status |
| #24 public | `contracts/public-report-label-dataset-v1.json` | 5,901 | `3b607558a5bea66d2d10dd3884bd6fa9e2ac81b4617c6ea372d7feb0c00068e9` | direct candidate |
| #24 public | `report_labels_v2.csv` | 1,004,839 | `6f704a7bdb2f894cc49445b19ba7c4378c3f548d3449e00361e10044bee40920` | direct candidate |

The 4,349-study non-gold comparison and its aggregate-only evidence are frozen
by #25's `contracts/report-source-comparison-v1.json` and
`docs/report-source-comparison-report.md`. No raw report, private row-level
label, or private study identifier is committed.

## Candidate semantics

All candidates use the competition target order and the common representation
`StudyInstanceUID`, target, value, observed, provenance, and source state.
Observed values are 0 or 1; masked values are `None` and have `observed=false`.

| Candidate | Source mapping | Gold policy | Provenance caveat |
| --- | --- | --- | --- |
| `gold_only` | organizer gold 1/0; missing masked | no weak source contributes to loss | control |
| `public_direct_v1` | `YES -> 1`, `NO -> 0`, `UNK -> masked` | gold labels authoritative; public rows for gold studies excluded | external benchmark may have been influenced by organizer-gold studies during its own development/evaluation |
| `qwen_direct_v1` | `positive -> 1`, `negative -> 0`, `uncertain -> masked`, `unknown -> masked` | gold labels authoritative; Qwen rows for gold studies excluded | locally reproducible frozen Qwen extraction, distinct from clinical correctness evidence |

Public numeric scores and confidence columns are not calibrated targets and have
no v1 training effect. Qwen `parse_failed=true` masks every target in that study,
regardless of an emitted state; row-level and study-status flags and report
identities must agree.

## Gold firewall

The boundary is shared with the completed #27 public path:

- gold means all twelve organizer target fields are present;
- non-gold means all twelve are absent;
- a partially labeled study is rejected;
- only the 4,349 non-gold studies may receive report-derived supervision;
- organizer-gold labels remain authoritative and are never replaced by a report
  source.

The loader fails closed on identity drift, duplicate study-target rows, unknown
targets, target-order drift, invalid states, parse-failure mismatches, incomplete
source coverage, and weak rows that would cross the gold firewall; gold-source
rows are never admitted as weak observations.

## Public-safe aggregate inventory

Counts below describe the full 4,407-study candidate representation. The public
and Qwen positive/negative/masked counts are the non-gold #25 aggregate counts
plus the verified 58-study gold counts. For Qwen, masked combines `uncertain`,
`unknown`, and the 176 parse-failed studies (2,112 rows per target); usable weak
labels are positive plus negative. `gold_only` has 4,349 masked non-gold studies
per target and zero usable weak labels.

| Target | `gold_only` + / − / masked | `public_direct_v1` + / − / masked | Public usable | `qwen_direct_v1` + / − / masked | Qwen usable | Qwen parse-failed excluded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ACL | 24 / 34 / 4,349 | 1,218 / 2,838 / 351 | 3,998 | 589 / 439 / 3,379 | 970 | 2,112 |
| MCL | 9 / 49 / 4,349 | 758 / 3,225 / 424 | 3,925 | 89 / 441 / 3,877 | 472 | 2,112 |
| Medial Meniscus | 26 / 32 / 4,349 | 2,341 / 1,818 / 248 | 4,101 | 1,197 / 226 / 2,984 | 1,365 | 2,112 |
| Lateral Meniscus | 23 / 35 / 4,349 | 1,114 / 2,858 / 435 | 3,914 | 723 / 409 / 3,275 | 1,074 | 2,112 |
| Medial OA | 15 / 43 / 4,349 | 1,612 / 1,684 / 1,111 | 3,238 | 702 / 88 / 3,617 | 732 | 2,112 |
| Lateral OA | 11 / 47 / 4,349 | 1,082 / 1,890 / 1,435 | 2,914 | 292 / 88 / 4,027 | 322 | 2,112 |
| PF OA | 21 / 37 / 4,349 | 1,988 / 1,618 / 801 | 3,548 | 535 / 169 / 3,703 | 646 | 2,112 |
| Effusion | 35 / 23 / 4,349 | 2,651 / 1,324 / 432 | 3,917 | 1,715 / 369 / 2,323 | 2,026 | 2,112 |
| Synovitis | 27 / 31 / 4,349 | 591 / 145 / 3,671 | 678 | 413 / 110 / 3,884 | 465 | 2,112 |
| Baker's | 12 / 46 / 4,349 | 1,110 / 1,303 / 1,994 | 2,355 | 926 / 275 / 3,206 | 1,143 | 2,112 |
| Contusion | 19 / 39 / 4,349 | 1,280 / 2,221 / 906 | 3,443 | 256 / 99 / 4,052 | 297 | 2,112 |
| Fracture | 18 / 40 / 4,349 | 313 / 1,634 / 2,460 | 1,889 | 133 / 82 / 4,192 | 157 | 2,112 |

## Non-selection boundaries

#20 remains diagnostic-only; no `rules_direct_v1` candidate is registered. The
following are forbidden before #10 compares the simple candidates: consensus or
majority labels, public/Qwen/rules hybrids, target-specific source selection,
disagreement filtering, confidence or numeric-score thresholds, target-specific
loss weights, special-case target removal, and post-hoc source selection under an
existing candidate name.

Issue #27's public-path implementation and leaderboard score `0.819` are recorded
as operational evidence only. That score is not ground truth and is not
controlled candidate-selection evidence. Likewise, source agreement is not
correctness evidence.

The frozen policy is the handoff to #22 for deterministic masked-BCE inputs.
Issue #10 owns the controlled organizer-gold OOF comparison and source selection.
