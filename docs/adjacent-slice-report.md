# Adjacent-slice 2.5D context — issue #32

Status: complete. The adjacent-slice candidate was **rejected** by the
precommitted validation rule. Retain issue #31's same-slice RGB ResNet-18 with
the 130 mm physical crop at 224×224. No hidden-test inference or competition
submission was run.

## Frozen experiment

The retained control is issue #31 at commit
`e85e24ebf0b78c6c73e63465d08c6b3dfa098141`, checkpoint SHA-256
`55e9da7854d284a91c2569e25c799281d09301bae61dc5adcecebbd17209f013`.
It uses ResNet-18, eight deterministic anchors per series, same-slice grayscale
RGB repetition, a centered 130 mm physical crop, 224×224 inputs, and ordinary
slice → series → study means.

The candidate's only intended representation change was channel assignment.
For anchor position `i` in the established physical/DICOM series order, the
input channels were `[max(i-1,0), i, min(i+1,n-1)]` in fixed
previous/center/next order. The eight center indices remained identical to #31.
An undecodable neighbor would use the processed center channel without searching
farther; center failures retained #31 semantics.

Series 1st/99th percentiles were computed from the unchanged center-anchor
population only, then frozen and applied to neighbors. Each channel used its
source DICOM's own `PixelSpacing`, the unchanged 130 mm crop/fallback contract,
and 224×224 output. The model, initialization, cohort, target order, loss,
split, optimizer, seeds, aggregation, anchor count, and encoder call count were
otherwise fixed.

The definitive run used clean source
`7a8b3ccea481305c33b40e6b7d854332915a98e2`. Key artifact identities:

- config SHA-256: `4b75b35f3362fcba82c5d71bcb3fdb5819cf8360c7d05644559d3717511815dd`
- notebook source SHA-256: `093c8916dfa05d517106e0f6e7a5ca0f44d0e1559ff8780c3a9a2e98039e2afc`
- rendered notebook SHA-256: `4bba527ad4e3b9faaceb3be2631305d747d0ccf7b3a94b7f7035923e235196d9`
- checkpoint SHA-256: `d6f8291403a7b6585a1f362925e6bcaae39c82b2e3ed74b2dabe672cfe05bee2`
- public-safe report SHA-256: `274901c2fc5feb50f6352a37315ec62646051e8979c846f8ca65961810b6f02f`
- private validation predictions SHA-256: `4879e154bb25ce3004703552db1cdb2e60ed2711628a9b11a2605e06f80e1b37`

The remote artifact verifier passed before publication. The downloaded
checkpoint also passed the strict local #32 schema, configuration, target-order,
initialization, and digest checks. Reloaded predictions matched exactly
(maximum absolute difference `0.0`).

## Validation result

Validation used the same 450 studies and PatientID-disjoint membership as #31,
with digest
`8b1240fc2456779ba64fa115b23956adb131fb387f01c0f0424a933a2096ee88`.

| Metric | #31 control | Adjacent-slice | Delta |
| --- | ---: | ---: | ---: |
| Macro ROC-AUC | 0.847212 | 0.847923 | +0.000710 |
| Masked BCE | 0.462097 | 0.566762 | +0.104666 |

| Target | #31 AUC | Adjacent-slice AUC | Delta |
| --- | ---: | ---: | ---: |
| ACL | 0.844484 | 0.840939 | -0.003545 |
| Baker's | 0.859292 | 0.878390 | +0.019098 |
| Contusion | 0.806030 | 0.808358 | +0.002328 |
| Effusion | 0.858636 | 0.863578 | +0.004942 |
| Fracture | 0.898986 | 0.899816 | +0.000829 |
| Lateral Meniscus | 0.791687 | 0.800717 | +0.009030 |
| Lateral OA | 0.869330 | 0.855988 | -0.013342 |
| MCL | 0.782509 | 0.785720 | +0.003210 |
| Medial Meniscus | 0.837201 | 0.827588 | -0.009613 |
| Medial OA | 0.892097 | 0.878538 | -0.013560 |
| PF OA | 0.857407 | 0.856551 | -0.000856 |
| Synovitis | 0.868889 | 0.878889 | +0.010000 |

Seven of twelve targets were non-decreasing, below the diagnostic preference
of eight. No target crossed the precommitted severe-regression threshold of
`-0.02` AUC.

The paired PatientID bootstrap used seed `20260825`, 10,000 valid replicates,
and no undefined replicates. The macro-AUC delta 95% percentile interval was
`[-0.006672, +0.007925]`; 58.21% of replicates were positive.

## Preprocessing evidence

Each training epoch requested 471,433 unique source slices, attempted 471,431,
decoded 471,429, and produced 175,158 encoder inputs. It contained 175,160
anchor triplets; 21,895 triplets clamped on each side (12.5% per side). The two
unusable centers followed #31's existing semantics. No previous- or
next-neighbor center substitutions were needed.

Validation requested, attempted, and decoded 53,370 unique source slices and
produced 19,808 encoder inputs. Its 19,808 triplets included 2,476 left clamps
and 2,476 right clamps (12.5% per side), with no neighbor substitutions.

All 471,429 decoded training source slices per epoch and all 53,370 validation
source slices used the physical crop. Crop usage was 100% at slice, series, and
study level; there were no `PixelSpacing` fallbacks. This evidence covers
decoded center and neighbor channel sources separately from neighbor failures.

## Resources and decision

Three-epoch training took 24,826.77 seconds (6.90 hours), `2.195×` issue #31's
11,312.74 seconds. Throughput was 23.44 valid encoder inputs/second over 525,474
training inputs. Candidate validation decoding took 747.43 seconds versus
306.76 seconds for the same-slice control, while forward time was 12.85 versus
13.82 seconds (`0.930×`). The added cost was preprocessing/I/O, not encoder
work.

Peak CUDA allocation was 12,281,359,872 bytes (`1.053×` #31 and below the
10% material-difference threshold). The 11,182,668-parameter checkpoint was
134,349,575 bytes.

The candidate fails both mandatory keep conditions: `+0.000710` is below the
required `+0.005` macro-AUC gain, and 58.21% bootstrap positivity is below
95%. The result is **reject**. Keep #31 unchanged and test a different single
MRI-specific intervention, such as sequence-aware series handling or improved
study aggregation, in a separate controlled issue. Do not tune another neighbor
window under issue #32.

Because validation failed the keep rule, the hidden-inference and submission
approval boundaries are not reached. No hidden artifact or submission was
created.
