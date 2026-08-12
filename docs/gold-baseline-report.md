# Leakage-safe organizer-gold holdout baseline

Verified 2026-08-12 for `rsna-knee-abnormality-detection`. The canonical
public-safe machine-readable evidence is
[`contracts/gold-baseline.json`](../contracts/gold-baseline.json). Private
PatientIDs, study assignments, labels, predictions, checkpoints, and the full
run artifact remain under the ignored local artifact registry and private
Kaggle output; none are committed.

## Result and scope

Private, internet-disabled Kaggle kernel
[`plthon/rsna-knee-gold-holdout-baseline`](https://www.kaggle.com/code/plthon/rsna-knee-gold-holdout-baseline),
version 1, completed on a Tesla T4. The best reloaded checkpoint produced an
official unweighted 12-target macro ROC AUC of **0.611111** on the exact frozen
four-PatientID organizer-gold holdout.

This is evidence only for deciding whether issue #11 can motivate and evaluate
one model change. It is not a claim of general statistical sufficiency,
generalization, clinical utility, or competitive performance. No leaderboard
evidence or report-derived label was used.

## Frozen split reproduction

Before model construction or prediction review, the run independently rebuilt
issue #7's `patient-disjoint-exact-backtracking-v1` allocation using seed
`20260812`. The derived assignment SHA-256 was
`8dccc31852a3e26ec677062a289301004fdce29eb64a556bea8922dfd2bb263d`,
exactly matching the frozen witness. The split contains 54 train PatientIDs and
four holdout PatientIDs, one study per PatientID, with zero PatientID overlap.
Report-hash grouping was not applied because this is a gold-only path. Every
target has both organizer-gold classes in both partitions; missing labels would
remain masked, although this 58-study cohort has none.

| Target | Train + / − (prevalence) | Holdout + / − (prevalence) |
| --- | ---: | ---: |
| ACL | 21 / 33 (38.9%) | 3 / 1 (75%) |
| MCL | 8 / 46 (14.8%) | 1 / 3 (25%) |
| Medial Meniscus | 24 / 30 (44.4%) | 2 / 2 (50%) |
| Lateral Meniscus | 22 / 32 (40.7%) | 1 / 3 (25%) |
| Medial OA | 13 / 41 (24.1%) | 2 / 2 (50%) |
| Lateral OA | 10 / 44 (18.5%) | 1 / 3 (25%) |
| PF OA | 18 / 36 (33.3%) | 3 / 1 (75%) |
| Effusion | 33 / 21 (61.1%) | 2 / 2 (50%) |
| Synovitis | 25 / 29 (46.3%) | 2 / 2 (50%) |
| Baker's | 11 / 43 (20.4%) | 1 / 3 (25%) |
| Contusion | 17 / 37 (31.5%) | 2 / 2 (50%) |
| Fracture | 17 / 37 (31.5%) | 1 / 3 (25%) |

For every train target, known/missing counts are 54/0. For every holdout target,
they are 4/0. Exact counts and prevalence are retained in the machine-readable
contract.

## Metric and uncertainty

The uncertainty method was frozen before predictions as
`patient-id-nonparametric-bootstrap-v1`, seed `20260812`, 10,000 replicates,
95% two-sided percentile intervals, and linear quantiles. Each replicate samples
four PatientIDs with replacement and includes every study belonging to every
drawn PatientID.

A target AUC is undefined when a replicate lacks either class. Macro AUC is
undefined when any required target AUC is undefined. No default score is
assigned, and macro AUC is never averaged over only the defined targets.

| Target | AUC | 95% interval over valid replicates | Valid | Undefined (rate) |
| --- | ---: | ---: | ---: | ---: |
| ACL | 0.6667 | [0, 1] | 6,774 | 3,226 (32.26%) |
| MCL | 0 | [0, 0] | 6,811 | 3,189 (31.89%) |
| Medial Meniscus | 0 | [0, 0] | 8,767 | 1,233 (12.33%) |
| Lateral Meniscus | 1 | [1, 1] | 6,763 | 3,237 (32.37%) |
| Medial OA | 0.5 | [0, 1] | 8,767 | 1,233 (12.33%) |
| Lateral OA | 0.6667 | [0, 1] | 6,763 | 3,237 (32.37%) |
| PF OA | 1 | [1, 1] | 6,832 | 3,168 (31.68%) |
| Effusion | 0.75 | [0, 1] | 8,732 | 1,268 (12.68%) |
| Synovitis | 1 | [1, 1] | 8,732 | 1,268 (12.68%) |
| Baker's | 1 | [1, 1] | 6,763 | 3,237 (32.37%) |
| Contusion | 0.75 | [0, 1] | 8,767 | 1,233 (12.33%) |
| Fracture | 0 | [0, 0] | 6,811 | 3,189 (31.89%) |
| **Official macro AUC** | **0.6111** | **[0.6111, 0.6111]** | **935** | **9,065 (90.65%)** |

The degenerate macro interval is not evidence of precision. With this joint
label pattern, only 9.35% of resamples preserve both classes for all 12 targets;
those valid draws reproduce the same rank information. The 90.65% undefined
rate is therefore essential context and is not replaced by a default score.

## Training and resource evidence

The run reused issue #8's exact target order, 224×224 preprocessing, eight
selected slices per series, torchvision ResNet-18, masked mean slice-to-series
and series-to-study pooling, masked binary cross entropy, AdamW, mixed
precision, and offline `ResNet18_Weights.IMAGENET1K_V1` bytes. The initialization
SHA-256 was
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`;
it was validated before model construction with no network or random-weight
fallback.

Issue #8's already-approved comparable T4 epoch measured 58.53 seconds across
58 studies and gave a pre-run estimate of 54.50 seconds per 54-study baseline
epoch. The complete 10-epoch run measured:

- 306.48 seconds for the complete training and per-epoch validation loop;
- 289.67 seconds across training epochs;
- 25,280 valid training slices, or 87.27 slices/second;
- 241.90 seconds decoding and 45.72 seconds in forward/backward work;
- 1,101,347,328 bytes peak CUDA memory;
- zero structured train or best-reload validation decode failures.

Decoding represented 83.51% of measured epoch training time. A cache or
embedding build is nevertheless **not recommended yet**: this single run does
not measure reuse benefit or cache construction/storage cost, so it does not
materially justify creating a potentially stale cache.

## Checkpoints, reload, and registration

Only the best and last complete resumable checkpoints were retained. The best
was epoch 9, SHA-256
`4bac5d264cd7900fc06b6dc354ea89114ce5d9665836f4b164632f007163b2a0`;
the last was epoch 10, SHA-256
`20890ff267fab7b026582999b0231efc85f9a62b2c79566f2c1460fa8e3342fb`.
Each is 134,350,530 bytes and contains model, optimizer, scaler, target,
preprocessing, configuration, runtime, initialization, input, code, cohort,
split, and RNG state. Reloaded best-checkpoint predictions matched the original
best validation predictions exactly (maximum absolute difference 0, tolerance
`1e-6`).

The verified local run ID is
`gold-holdout-baseline-339561a2212740aca14420513c30953d`. Its registered
artifact size is 268,732,153 bytes. Immutable configuration, Jupytext source,
packaged source, notebook identity, runtime, input, initialization, split, and
artifact hashes are recorded in the public contract; private artifacts remain
under `.artifacts/`, which Git ignores.

## Evidence adequacy for issue #11

**Decision: `inadequate`.**

The frozen holdout has only four PatientIDs. Several targets have only one
example of one class, and 90.65% of macro-AUC bootstrap replicates are undefined.
This evidence is too fragile to attribute a measured failure to the baseline or
to evaluate one model change. This decision is limited to issue #11, uses no
leaderboard evidence, claims no general statistical sufficiency, and invents no
minimum-support or interval-width threshold. Per issue #11's precondition, that
ticket must stop for explicit roadmap realignment.
