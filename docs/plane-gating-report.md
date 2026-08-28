# Target-specific anatomical-plane gating

Verified 2026-08-28 for issue #28. The intervention was **rejected**. It did
not meet the precommitted validation rule, so the repository retains the #27
equal-series mean as the default and no hidden-test inference, competition
submission, or issue #29 was run or opened.

## Controlled experiment

The control was the #27 ResNet-18 checkpoint from implementation commit
`e69f79af9c3de06e17a77e43bb41cb971879aed0`, SHA-256
`1285f9890eb01e6a2383552e2ef1efcccdbc956ce14de049042bc268281240c4`.
Both arms used the same ImageNet initialization, 4,407-study gold-plus-public
cohort, 3,957-study training partition, 450-study PatientID-disjoint validation
partition, three epochs, seed `20260825`, 8 sampled slices per series, 224-pixel
inputs, masked BCE, AdamW settings, preprocessing, and slice/series selection.
The validation membership digest was
`8b1240fc2456779ba64fa115b23956adb131fb387f01c0f0424a933a2096ee88`.
The seven organizer-gold validation studies were excluded from training.
The exact training settings were batch size 16, learning rate `1e-4`, weight
decay `1e-4`, mixed precision enabled, 1st/99th percentile normalization, and
reload tolerance `1e-5`. The paired bootstrap used seed `20260825`, 10,000
replicates, and a 95% interval.

The only model change was the final series-to-study aggregation. The candidate
used 36 positive target-by-plane scalars (`exp(log_weight)`), initialized at
zero log-weight, with unresolved planes fixed at weight 1. A fixed probe with
duplicate sagittal, unresolved, and padded series reproduced equal averaging
exactly (maximum absolute difference `0.0`, tolerance `1e-6`). Encoder inputs,
forward/backward pass counts, and the shared ResNet-18 were unchanged.

Execution source identity:

- Git commit: `022e50fdffde3b0fd167644718eef7e4e90a9316` with embedded dirty worktree
- submitted notebook file SHA-256: `076b0990bed7842eeaf8eb9092d6845a2b3b6b64791399e2cd4843e81455a5d2`
- embedded rendered-notebook provenance value: `94ea5b9c7f907e26799b66a2b98b7ad428302ae3b26466518739aa843337ea0a`
- notebook source SHA-256: `5aaffceab544a8f69ca34b42a80d0f596c4cf6dd4e3082730c8318226c4d37b5`
- config SHA-256: `8f24f1d1dba1a3b50528e71283917ef5a88171fb62680ed5d38dfc3d632b5364`
- candidate checkpoint SHA-256: `e7e1cabb4914c2f3d1f4f0835e10956897545de4446d2b4c2d2547694f9205f6`
- public-safe training report SHA-256: `46c4a2bad2ba1c5f025ef752857cd26c29356bdb57557ccad4adfc7b6826e55b`
- private paired-prediction SHA-256: `ca253425be2a06e675e1df91184a04ba616b3a82fa290dfa9be53559cc51cfff`

The remote artifact verifier passed before publication. The downloaded
checkpoint also passed the strict local schema, configuration, target-order,
initialization, and hash checks; reload prediction difference was `0.0`.

## Validation result

| Metric | #27 mean | Plane gate | Delta |
| --- | ---: | ---: | ---: |
| Macro ROC-AUC | 0.838277 | 0.839343 | +0.001067 |
| Masked BCE | 0.520846 | 0.472933 | -0.047912 |

The paired PatientID bootstrap used 10,000 replicates. Its macro-AUC delta had
95% percentile interval `[-0.005032, 0.006885]`; only 63.92% of replicates were
positive. Seven of twelve targets were non-decreasing, and no target regressed
by more than the precommitted `0.02` inspection threshold.

| Target | Known (+/−) | #27 AUC | Gate AUC | Delta |
| --- | ---: | ---: | ---: | ---: |
| ACL | 411 (130/281) | 0.827635 | 0.834164 | +0.006529 |
| Baker's | 258 (118/140) | 0.837621 | 0.837893 | +0.000272 |
| Contusion | 365 (136/229) | 0.784774 | 0.782398 | -0.002376 |
| Effusion | 409 (259/150) | 0.857027 | 0.864891 | +0.007864 |
| Fracture | 206 (31/175) | 0.889032 | 0.880184 | -0.008848 |
| Lateral Meniscus | 404 (105/299) | 0.780045 | 0.796751 | +0.016706 |
| Lateral OA | 299 (96/203) | 0.868740 | 0.854731 | -0.014009 |
| MCL | 402 (83/319) | 0.748820 | 0.756128 | +0.007308 |
| Medial Meniscus | 426 (226/200) | 0.833164 | 0.845686 | +0.012522 |
| Medial OA | 330 (151/179) | 0.899922 | 0.893855 | -0.006068 |
| PF OA | 365 (186/179) | 0.863098 | 0.855439 | -0.007659 |
| Synovitis | 65 (45/20) | 0.869444 | 0.870000 | +0.000556 |

## Learned weights and plane coverage

| Target | Sagittal | Coronal | Axial |
| --- | ---: | ---: | ---: |
| ACL | 1.014965 | 0.997646 | 0.991246 |
| Baker's | 0.993301 | 0.988420 | 1.015291 |
| Contusion | 1.004337 | 1.004688 | 0.993433 |
| Effusion | 0.988014 | 0.987199 | 1.019097 |
| Fracture | 1.001941 | 0.995163 | 1.003188 |
| Lateral Meniscus | 1.006689 | 0.998459 | 0.994270 |
| Lateral OA | 1.003723 | 1.005697 | 0.990426 |
| MCL | 0.998098 | 0.998762 | 1.005488 |
| Medial Meniscus | 1.009825 | 0.996678 | 0.989118 |
| Medial OA | 1.006264 | 1.005474 | 0.986981 |
| PF OA | 1.016061 | 0.987507 | 0.990164 |
| Synovitis | 1.006239 | 0.999546 | 0.995229 |

All indexed planes resolved. Training contained 21,895 series (8,878 sagittal,
7,716 coronal, 5,301 axial); validation contained 2,476 (986 sagittal, 893
coronal, 597 axial). The mounted three-study example test contained 15 series
(7 sagittal, 4 coronal, 4 axial). This is not hidden-test coverage evidence.

The weights stayed close to one (`0.986981` to `1.019097`). The clearest gains
were for lateral and medial meniscus, but they were offset by regressions led
by lateral OA. Plausible explanations remain weak-label/protocol confounding,
plane identity being too coarse, and the backbone already inferring plane.
There were no unresolved-plane studies with which to test differential behavior.

## Resources and decision

The run used a Tesla T4 and completed three epochs in 14,635.04 seconds
(4.07 hours), 11.7% longer than #27's recorded 13,102.42 seconds. Epoch
throughput was 40.38, 38.62, and 40.39 valid slices/second. Candidate paired
validation forward time was 13.08 seconds versus 15.44 seconds for the control
(`0.847x`), consistent with unchanged encoder inference work. Candidate peak
CUDA allocation was 11,667,191,296 bytes.

The published #27 peak-memory value of 741,710,848 bytes came from a later
checkpoint-reuse report rather than the original training process. The apparent
`15.73x` ratio in the raw #28 report is therefore not a valid training-to-training
comparison and cannot establish a gate-caused memory regression. The candidate
run still fit the same 15.6 GB T4, and the architecture added only 36 scalars,
but a comparable #27 training peak was not captured. This evidence limitation
is recorded rather than repaired with another run after the rejection decision.

The intervention fails both mandatory keep conditions: `+0.001067` is below
the `+0.005` macro-AUC threshold, and 63.92% bootstrap positivity is below 95%.
The result is **reject**. Equal series averaging remains active; no hidden-test
artifact or Kaggle submission was produced. Sequence-aware issue #29 should
not be opened from this result.
