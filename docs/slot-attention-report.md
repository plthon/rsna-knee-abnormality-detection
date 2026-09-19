# Issue #34: four-slot target-specific masked attention

Status: completed GPU validation; reject #34 and retain #31. The user ran the
training experiment. No hidden inference or competition submission was performed.

## Frozen experiment

Control: #31, retained revision `e85e24ebf0b78c6c73e63465d08c6b3dfa098141`,
training source `5fe133ed01d418b45643f3fe0458be5f0d1f558d`, checkpoint SHA-256
`55e9da7854d284a91c2569e25c799281d09301bae61dc5adcecebbd17209f013`.
Validation macro ROC-AUC: 0.847212; public competition score: 0.830.
The implementation source revision is recorded by the local bundle and its
notebook provenance. The final implementation commit accompanies delivery.

Frozen slot order (`rsna-knee-slot-attention-v1`):

1. `axial_fluid_sensitive`: axial and Fluid_Sensitive true.
2. `sagittal_fluid_sensitive`: sagittal and Fluid_Sensitive true.
3. `coronal_fluid_sensitive`: coronal and Fluid_Sensitive true.
4. `sagittal_non_fluid_sensitive`: sagittal and Fluid_Sensitive false.

One helper reads the indexed Anatomical_Plane and Fluid_Sensitive fields, strips
whitespace and case-folds; yes/true/1 and no/false/0 are recognized. Missing or
unrecognized values are unmatched. Fat_Suppression is unused. Membership is
exclusive and independent of file names. Unmatched series are counted and
excluded from candidate aggregation; there is no other slot.

The unchanged ResNet-18 penultimate features are averaged over valid anchors,
then equally over series sharing a slot. For target t and available slot s:

```
a[s,t] = Linear(512,12)(h[s])[t]
l[s,t] = existing_shared_classifier(h[s])[t]
w[s,t] = exp(a[s,t]) / sum_available_slots exp(a[:,t])
study_logit[t] = sum_available_slots w[s,t] * l[s,t]
```

Unavailable slots are masked before softmax and receive exactly zero weight.
Each target has its own normalization; one available slot receives weight one.
Zero available slots use the existing training-label prevalence priors, with
no all-series mean substitution. Such training studies do not contribute an
optimizer update; validation retains them, and hidden submission records an
explicit fallback reason. All cases are counted in public-safe aggregate evidence.
The checkpoint stores the prior logits, validates its slot identity and uses a
separate schema from #31. The #31 mean path and shared head remain selectable.

## Fixed representation and evaluation

The candidate retains eight full-span deterministic anchors, same-slice RGB,
series 1st/99th percentile normalization, physical/DICOM ordering, centered
130 mm crop, 224 px output, ImageNet normalization and exact offline
ResNet18_Weights.IMAGENET1K_V1 initialization. No 2.5D or clipped-window path is
active. The same three-epoch AdamW recipe, batch size 16, mixed precision,
learning rate/weight decay 0.0001, seed policy, label masks, gold precedence,
4,407-study cohort and 3,957/450 PatientID-disjoint partition are enforced.
Validation membership digest:
`8b1240fc2456779ba64fa115b23956adb131fb387f01c0f0424a933a2096ee88`.

Verified #31 validation predictions are reused when attached with matching
report provenance, prediction digest, target order, study/PatientID membership
and labels. Otherwise the exact retained checkpoint is evaluated; #31 is never
retrained. Paired bootstrap retains the existing seed and 10,000 replicates.
Keep requires macro delta >= 0.005 and bootstrap positive fraction >= 0.95,
with severe target regressions checked using the existing decision policy.

The training report records macro/per-target AUC and deltas, masked BCE,
bootstrap interval/positive fraction/valid and undefined counts, slot coverage,
co-occurrence patterns, multiplicities, unmatched series, zero-slot studies,
per-target/slot mean and median attention, reload tolerance, runtime, throughput,
GPU memory and checkpoint size. Attention summaries include all validation
studies with unavailable slots contributing zero, and expose no private IDs.
Observed cohort statistics and performance results are recorded below.

The only added learned component is a 512-to-12 linear attention projection
(6,156 parameters). Encoder calls and input sizes remain the same for usable
training studies; zero-slot training studies are skipped. Actual runtime and
memory must be measured remotely rather than assumed equal.

## Execution boundary

The user completed `plthon/rsna-knee-slot-attention` from clean source
`b9ec1dd5431c04b8a6338095898d4816670b3730`. Kaggle reports COMPLETE.
The experiment fails the frozen keep rule; no hidden inference, submission,
slot tuning, or repeat GPU execution follows from this result.

## Local verification and review

All required gates passed: locked base and training dependency sync, Ruff lint
and formatting, 388 tests with the training extra, strict mypy over
`src tests scripts notebooks`, source/wheel build, and `git diff --check`.
The local bundle was built and loaded with GPU disabled. Tests exercise raw
metadata, file-order independence, exact anchor preprocessing, equal feature
pooling, masking and gradients, single/zero-slot cases, unchanged mean control,
checkpoint rejection/reload, cached-control provenance, and validation/hidden
inference agreement. These implementation gates preceded the user-run GPU experiment.

### Standards

No actionable findings or documented-standard violations. The experiment
selection branches follow the existing control-path structure; restructuring
them is outside this narrowly scoped change.

### Spec

One provenance defect was found and resolved: cached #31 reports must match
training source `5fe133ed...`, rather than the later retention-report commit.
The fix and prior/fallback consistency validation were independently reviewed.
No remaining actionable implementation findings.

## Completed validation result

The downloaded report matches the run manifest's SHA-256 and size. The same
450-study PatientID-disjoint validation membership and frozen cohort/configuration
were verified. Exact #31 predictions were reused with verified provenance.
The image representation and encoder contract remain unchanged.

Macro ROC-AUC: **0.847212 → 0.836031**
(delta **-0.011182**). Paired bootstrap 95% delta interval:
[-0.020082, -0.002241]; positive fraction **0.66%**; 10,000 valid / 0 undefined
replicates. Masked BCE: **0.462096 → 0.535793** (+0.073697).
Only 2/12 targets were non-decreasing; Effusion and Fracture regressed by more
than 0.02. Both mandatory thresholds fail. **Reject #34; retain #31.**

| Target | #31 AUC | #34 AUC | Delta |
| --- | ---: | ---: | ---: |
| ACL | 0.844484 | 0.834191 | -0.010293 |
| MCL | 0.782509 | 0.783133 | +0.000623 |
| Medial Meniscus | 0.837201 | 0.835907 | -0.001294 |
| Lateral Meniscus | 0.791687 | 0.779838 | -0.011849 |
| Medial OA | 0.892097 | 0.874838 | -0.017259 |
| Lateral OA | 0.869330 | 0.850421 | -0.018909 |
| PF OA | 0.857407 | 0.841623 | -0.015784 |
| Effusion | 0.858636 | 0.836525 | -0.022111 |
| Synovitis | 0.868889 | 0.855556 | -0.013333 |
| Baker's | 0.859292 | 0.871792 | +0.012500 |
| Contusion | 0.806030 | 0.793154 | -0.012876 |
| Fracture | 0.898986 | 0.875392 | -0.023594 |

Epoch validation AUCs were 0.807464, 0.844993, and 0.836031. The frozen recipe
retains the third epoch. Even epoch two is below #31; no post-hoc checkpoint
selection or architecture rescue is warranted.

## Slot coverage and attention

Training counts below are per epoch and identical across all three epochs.
There were 4,628 unmatched training series and 536 unmatched validation series.
Zero-slot studies: **0/3,957 train and 0/450 validation**; no prior fallback was
needed. Thus missing all slots does not explain the degradation.

| Slot | Train available | Validation available |
| --- | ---: | ---: |
| axial_fluid_sensitive | 3957/3957 (100.00%) | 450/450 (100.00%) |
| sagittal_fluid_sensitive | 3716/3957 (93.91%) | 434/450 (96.44%) |
| coronal_fluid_sensitive | 3810/3957 (96.29%) | 438/450 (97.33%) |
| sagittal_non_fluid_sensitive | 3834/3957 (96.89%) | 432/450 (96.00%) |

Co-occurrence bits follow the frozen four-slot order (1 = available).

| Pattern | Train | Validation |
| --- | ---: | ---: |
| 1111 | 3446 | 404 |
| 1011 | 241 | 16 |
| 1101 | 147 | 12 |
| 1110 | 123 | 18 |

Multiplicity distributions (`series in slot: study count`):

| Slot | Train | Validation |
| --- | --- | --- |
| axial_fluid_sensitive | 1: 3688, 2: 257, 3: 12 | 1: 421, 2: 27, 3: 2 |
| sagittal_fluid_sensitive | 0: 241, 1: 3250, 2: 461, 3: 5 | 0: 16, 1: 389, 2: 44, 3: 1 |
| coronal_fluid_sensitive | 0: 147, 1: 3484, 2: 312, 3: 13, 4: 1 | 0: 12, 1: 407, 2: 27, 3: 4 |
| sagittal_non_fluid_sensitive | 0: 123, 1: 3036, 2: 744, 3: 49, 4: 5 | 0: 18, 1: 365, 2: 60, 3: 7 |

Validation attention **mean / median**, including unavailable slots as zero:

| Target | Axial fluid | Sagittal fluid | Coronal fluid | Sagittal non-fluid |
| --- | ---: | ---: | ---: | ---: |
| ACL | 0.0253 / 0.0228 | 0.3635 / 0.3499 | 0.2160 / 0.2042 | 0.3953 / 0.4109 |
| MCL | 0.0415 / 0.0366 | 0.2951 / 0.2883 | 0.5024 / 0.5024 | 0.1610 / 0.1522 |
| Medial Meniscus | 0.0974 / 0.0916 | 0.2681 / 0.2674 | 0.3343 / 0.3434 | 0.3001 / 0.2879 |
| Lateral Meniscus | 0.1744 / 0.1657 | 0.2975 / 0.3008 | 0.2539 / 0.2529 | 0.2743 / 0.2697 |
| Medial OA | 0.1681 / 0.1639 | 0.2154 / 0.2076 | 0.1746 / 0.1680 | 0.4419 / 0.4296 |
| Lateral OA | 0.0659 / 0.0586 | 0.2440 / 0.2310 | 0.1809 / 0.1689 | 0.5092 / 0.5309 |
| PF OA | 0.2391 / 0.2227 | 0.2160 / 0.2100 | 0.1269 / 0.1196 | 0.4180 / 0.4082 |
| Effusion | 0.1611 / 0.1375 | 0.4011 / 0.3864 | 0.2975 / 0.2670 | 0.1403 / 0.1228 |
| Synovitis | 0.1886 / 0.1787 | 0.3459 / 0.3416 | 0.3436 / 0.3365 | 0.1219 / 0.1089 |
| Baker's | 0.6427 / 0.6374 | 0.1479 / 0.1411 | 0.1196 / 0.1041 | 0.0897 / 0.0850 |
| Contusion | 0.0518 / 0.0451 | 0.4220 / 0.4220 | 0.4272 / 0.4268 | 0.0990 / 0.0900 |
| Fracture | 0.3654 / 0.3472 | 0.2519 / 0.2416 | 0.3178 / 0.3185 | 0.0650 / 0.0595 |

The learned weights differ by target (for example, Baker's favors axial fluid,
MCL favors coronal fluid, and lateral OA favors sagittal non-fluid), but this
specialization did not improve the primary validation metric. This experiment
combines slot exclusion/equal within-slot pooling and learned cross-slot weights;
it does not isolate which component caused the loss.

## Resources and checkpoint evidence

- GPU: Tesla T4. Reported total training loop: 12,642.02 seconds (3.51 hours),
  11.75% above the recorded #31 reference; includes epoch validation.
- Training throughput: 46.27 valid slices/sec, versus recorded #31 51.34.
- Final validation decode: 188.64 seconds; forward: 13.23 seconds; total 201.88.
  Control predictions were cached, so control runtime fields are zero and no
  same-run control timing comparison is available.
- Peak GPU memory: 11,665,598,976 bytes, versus #31 11,664,797,184 (+0.0069%).
- Parameters: 11,188,824; checkpoint: 134,426,795 bytes.
- The report counts 175,160 valid training anchors per epoch versus historical
  #31 175,158 (six additional counted anchors across three epochs). This small
  observed count difference is recorded, not represented as exact decode-count
  equality; the frozen preprocessing and sampling contracts did not change.
- Kaggle checkpoint reload maximum absolute prediction difference: **0.0**.
- Candidate checkpoint SHA-256:
  `db67b6b8bc439880495d52360e57fad7da84aa3708bdc57eac175f5d75d27d30`.

Recommended next controlled question: a different encoder while keeping the
retained #31 representation and equal series mean. This is a future issue,
not authorization to run or implement another experiment. No leaderboard score
exists for rejected #34; #31's public score remains 0.830.

Local post-run verification confirmed report/checkpoint hashes and sizes against
the downloaded manifest, clean source provenance, frozen configuration/cohort/
partition, target and slot identity, and fallback-prior consistency. The keep
rule was recomputed from the summary metrics and returned reject. No local or
remote hidden inference was executed; private validation rows stayed on Kaggle.
