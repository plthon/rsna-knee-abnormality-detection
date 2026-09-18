# Clipped-window three-anchor 2.5D — issue #33

Status: complete. The user-executed GPU run finished successfully, but #33 was
**rejected** by the frozen keep rule. Retain #31. #32 remains a completed negative
isolated-adjacency experiment. No hidden inference or competition submission was
run for #33.

The retained #31 control is commit
`e85e24ebf0b78c6c73e63465d08c6b3dfa098141`, checkpoint SHA-256
`55e9da7854d284a91c2569e25c799281d09301bae61dc5adcecebbd17209f013`,
validation macro ROC-AUC `0.847212`, public score `0.830`.

The [public discussion cited by #33](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735154)
motivates clipping stack ends and using three adjacent triplets. The exact
competitor clip fraction was not disclosed. Our frozen 10%–90% window is an
operationalization, not an exact reproduction of a private configuration.
Any gain belongs to the entire slice-layout intervention, not adjacency alone.

## Frozen contract

For an already physically/DICOM ordered stack of length `n >= 3`:

```text
left  = max(1, floor((n - 1) / 10))
right = min(n - 2, floor(9 * (n - 1) / 10))
centers = deduplicate_in_order([left, floor((left + right) / 2), right])
triplet(i) = [max(i - 1, 0), i, min(i + 1, n - 1)]
```

Integer arithmetic implements floor exactly. The window is inclusive. The
inward endpoint constraint reserves real previous/next neighbors. For example,
`n=20` selects `[1,9,17]`; `n=21` selects `[2,10,18]`. No interpolation,
random selection, target-dependent positioning, or farther-neighbor search occurs.

Series with `n>=5` have exactly three centers. For `n=3`, use `[1]`; for `n=4`,
use `[1,2]`. For `n=2`, use centers `[0,1]` and triplets `[0,0,1]`, `[0,1,1]`;
for `n=1`, use `[0,0,0]` only where the existing index admits the series. Empty
series still fail. Reduced series retain three tensor slots with unused slots
masked, so padding never reaches the encoder. The short-series counter means
`n<5`; duplicate counts describe removed linspace centers for `n>=3`.

Percentiles (1st/99th) use only usable selected center anchors, then freeze for
all channel sources. This intentionally changes the percentile population from
#31's eight centers to #33's three centers; it never uses all nine sources.
A failed center is masked under the retained pipeline semantics. A failed
neighbor uses the processed center for that side; side and decode failure reason
are counted. Each decoded source uses its own PixelSpacing for the unchanged
centered 130 mm crop, deterministic padding/fallback, and 224×224 resize.

Preprocessing emits exact center indices, triplets, and window bounds. Reports
retain a public-safe geometry frequency histogram keyed by stack length, window,
centers, and triplets, plus short-series, duplicate, clamp, substitution, unique
decode, encoder-input, and crop/fallback counts. No study identifiers appear in
that histogram. The checkpoint schema is `rsna-knee-clipped-window-2-5d-v1`;
metadata freezes layout, window, anchor/triplet/failure policies, normalization
population, crop/FOV, representation identity, and target order. #31/#32 schemas
and configurations remain available and cannot be silently substituted.

## Fixed training and evaluation

ResNet-18 and exact offline `ResNet18_Weights.IMAGENET1K_V1`, ImageNet channel
normalization, ordinary anchor → series → study means, masked BCE, gold precedence,
the 4,407-study cohort, and all optimizer/seed/checkpoint semantics remain fixed.
Training uses three AdamW epochs, LR/weight decay `0.0001`, batch size 16, and
the retained mixed precision policy.

The split stays 3,957 train / 450 validation, PatientID-disjoint, with validation
membership digest
`8b1240fc2456779ba64fa115b23956adb131fb387f01c0f0424a933a2096ee88`.
The runner verifies #31's checkpoint provenance and regenerates its predictions
with eight full-span same-slice anchors before identifier-aligned paired evaluation.
#32 is historical context, never the comparison control.

The existing paired evaluation emits all 12 AUCs/deltas, macro AUC/delta, masked
BCE, non-decreasing target count, 10,000-replicate seeded bootstrap interval,
positive fraction and valid/undefined counts. It also emits training runtime,
validation decode/forward time, throughput, peak GPU memory, checkpoint size,
and reload agreement (absolute tolerance `1e-5`). Measured results are recorded below.

Retain only at macro delta `>=0.005`, bootstrap-positive fraction `>=0.95`, and
no target regression worse than `-0.02`. Eight non-decreasing targets is a
diagnostic preference. On success, a separate experiment may test 336 px with
this layout fixed. On failure, retain #31 and consider plane/sequence-aware
aggregation; do not tune clip percentages or anchor count.

Expected normal per-series work is three encoder inputs (37.5% of #31/#32's
eight) and nine requested source positions, with overlapping positions decoded
once. #31 normally decodes eight sources; #32 can request up to 24. Actual
runtime and memory must be measured, not inferred from these counts.

## Completed validation result

The run used clean source `6cb50b24a434964ac820b0f044aec3af700c8ab3` and
completed all three epochs on a Tesla T4. The paired validation cohort was the
same 450 studies and membership digest specified above.

| Metric | #31 control | #33 candidate | Delta |
| --- | ---: | ---: | ---: |
| Macro ROC-AUC | 0.847212 | 0.831454 | -0.015758 |
| Masked BCE | 0.462097 | 0.490426 | +0.028329 |

| Target | #31 AUC | #33 AUC | Delta |
| --- | ---: | ---: | ---: |
| ACL | 0.844484 | 0.844197 | -0.000287 |
| Baker's | 0.859292 | 0.824607 | -0.034685 |
| Contusion | 0.806030 | 0.769779 | -0.036251 |
| Effusion | 0.858636 | 0.838546 | -0.020090 |
| Fracture | 0.898986 | 0.881843 | -0.017143 |
| Lateral Meniscus | 0.791687 | 0.780172 | -0.011515 |
| Lateral OA | 0.869330 | 0.841903 | -0.027427 |
| MCL | 0.782509 | 0.778317 | -0.004192 |
| Medial Meniscus | 0.837201 | 0.806294 | -0.030907 |
| Medial OA | 0.892097 | 0.875264 | -0.016834 |
| PF OA | 0.857407 | 0.842644 | -0.014762 |
| Synovitis | 0.868889 | 0.893889 | +0.025000 |

Only 1/12 targets was non-decreasing (Synovitis). Baker's, Contusion, Effusion,
Lateral OA, and Medial Meniscus regressed by more than 0.02 AUC.

The paired PatientID bootstrap used seed `20260825`, 10,000 valid replicates,
and zero undefined replicates. The 95% macro-delta interval was
`[-0.026326, -0.005384]`; only **0.19%** of replicates favored #33.

### Sampling and resources

All three training epochs had the same 21,895 series, 65,685 encoder inputs,
and 197,055 unique requested/attempted/decoded source slices per epoch.
Validation had 2,476 series, 7,428 encoder inputs, and 22,284 unique source
slices. Every series produced exactly three anchors and nine distinct sources.
The public-safe report retains the full stack-length/window/center/triplet
frequency histogram; its frequencies sum to the corresponding series counts.

There were no short series, rounded duplicate anchors, boundary clamps,
failed-neighbor substitutions, or failed centers in the selected #33 inputs.
All decoded source slices used physical crops; PixelSpacing fallback usage
was zero in training and validation. The frozen 10%–90% window was not tuned.

Training took **9,818.47 seconds (2.73 hours)**, `0.868×` #31's 11,312.74
seconds, or 13.2% less time. Throughput was 22.27 valid encoder inputs/second;
the three epochs processed 197,055 encoder inputs in total. Validation decoding
took 139.63 seconds versus the freshly evaluated control's 301.50 seconds;
forward time was 5.72 versus 13.73 seconds. Peak GPU allocation was
4,807,094,784 bytes (`0.412×` #31). The checkpoint was 134,349,959 bytes.
The reduction in compute/memory did not compensate for the loss of accuracy.

### Provenance and verification

- Config SHA-256: `d56a6941a1a69c7c928ec00ecee0262d56c729e6fbb4bf271a60f02598b47037`
- Notebook source SHA-256: `5ede7d0ce44913410efaf9d162eb788e3bfffbd3f798b1479d7e24bc1b84c19e`
- Rendered notebook SHA-256: `e058ac2d31e4a4dc2b7536b77bb0305c32c721ca7d2bfa52cc7df5ee5444b2c5`
- Checkpoint SHA-256: `9ebc083c1d57d793554c76842a7ce518a182c730758b8a55ec543ef156811f93`
- Public-safe report SHA-256: `d12e5aa0462e2cc4007964637a6aa0a93c625e7835ade3c752556e95a1cac8dc`
- Private validation predictions SHA-256: `b4f1525dee25ff462e73dffd17617bf7ce11a97eac001274befaa6bc987ef334`

The remote full-artifact verifier passed before the notebook published its
summary files. Local checks verified the downloaded report/checkpoint hashes
and sizes, clean source and config/notebook provenance, strict #33 checkpoint
schema/configuration/target order/initialization, validation membership, and
recomputed keep decision. The reported reload maximum absolute difference was
`0.0`, within `1e-5`. Private study-level predictions remain on Kaggle, so their
row-level verification was performed remotely rather than repeated locally.

### Decision

**Reject #33 and retain #31.** Both mandatory improvement/bootstrap thresholds
failed, with five severe target regressions. This is a negative result for the
entire clipped-window/three-anchor/triplet bundle; it does not isolate the effect
of clipping, anchor count, adjacency, or the changed center-percentile population.

Do not tune neighboring clip fractions or anchor counts after this result.
The next controlled experiment should consider plane/sequence-aware aggregation.
The conditional 336 px follow-up is not selected because this layout was rejected.
Hidden inference and submission approval boundaries are not reached.

## Local verification and execution boundary

Local tests cover sampling/short series, center-only normalization, failed
neighbors/centers, per-source crop/fallback, physical ordering, three-input
encoding, unchanged recipe, schema/target protection, prediction reload, live
synthetic hidden inference, and local bundle/approval guards. Run the required
quality gates with the optional training extra installed so model tests execute:

```bash
uv sync --locked
uv sync --extra training --locked
uv run ruff check .
uv run ruff format --check .
uv run --extra training pytest
uv run --extra training mypy src tests scripts notebooks
uv build
git diff --check
```

Implementation gates passed before execution: 381 tests including the training
extra, locked dependency sync, Ruff lint/format, mypy, package build, and whitespace
checks. Independent standards and spec reviews found no actionable issues.

The user executed the approved training/paired-validation bundle. After completion,
Codex performed read-only status/output retrieval and local verification only.
No additional remote execution or submission is warranted by this rejected result.
