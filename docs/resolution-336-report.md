# Resolution-only experiment — issue #35

Status: **completed; reject #35 and retain #31 at 224 px**. The approved full run completed three epochs and failed both mandatory keep thresholds. No hidden-test inference or competition submission was performed for #35.

## Run and control identities

- Candidate: [full training version 1](https://www.kaggle.com/code/plthon/rsna-knee-resolution-336-full); kernel `plthon/rsna-knee-resolution-336-full`, version 1. Source commit: `bca35fd04edb59d3ae3e06587204ea73ac28f761`; clean source.
- Resource gate: [smoke version 5](https://www.kaggle.com/code/plthon/rsna-knee-resolution-336-smoke), same source commit; full training was separately approved after the smoke passed.
- Candidate checkpoint schema: `rsna-knee-resolution-336-v1`; SHA-256 `370829035442b708007b12bf6a6ba8cc7d09976fe2ac6c10031ca71544bacd1a`; 134,349,255 bytes.
- Control: #31, retained revision `e85e24ebf0b78c6c73e63465d08c6b3dfa098141`, training source `5fe133ed01d418b45643f3fe0458be5f0d1f558d`; checkpoint SHA-256 `55e9da7854d284a91c2569e25c799281d09301bae61dc5adcecebbd17209f013`.
- Control validation macro ROC-AUC: **0.847212**; retained public competition score: **0.830**. Exact cached control predictions were reused after provenance and row-alignment checks; #31 was not retrained.
- Validation membership: the same 450 PatientID-disjoint studies, digest `8b1240fc2456779ba64fa115b23956adb131fb387f01c0f0424a933a2096ee88`; 3,957 training studies from the unchanged 4,407-study gold/public cohort.
- Training-report SHA-256: `240a1751ccaa43a5810497585f96b5224e617a6a5f1d21f6899d7af71601e692` (verified against the downloaded manifest). The report's keep decision was independently recomputed from its metrics. Checkpoint/reload evidence below is from the completed run, not a separate local checkpoint replay.

## Executed contract and interpretation

ResNet-18 with exact offline `ResNet18_Weights.IMAGENET1K_V1` initialization; 130 mm centered physical crop resized to 336×336; eight unchanged deterministic anchors; same-slice grayscale RGB; ordinary anchor → series → study means. The architecture and parameter count (11,182,668), pre-resize crop, PixelSpacing/fallback behavior, DICOM handling, percentile population, target order, supervision masks, split, seed policy, and paired PatientID bootstrap were unchanged.

Training used three epochs of AdamW, learning rate and weight decay 0.0001, with **physical batch 2 and eight-step gradient accumulation (effective batch 16)**. The smoke justified the minimal safe batch reduction: batches 16 and 8 ran out of memory; batch 4 exceeded the 90% reserved-memory safety threshold; batch 2 passed.

This is a negative result for **336 px with that necessary physical-batch adaptation**. Gradient accumulation preserves effective masked-loss weighting but does not reproduce physical-batch-16 BatchNorm statistics. The decline cannot be attributed solely to resolution. No alternative resolution, normalization, optimizer, or FOV rescue was attempted under #35.

## Paired validation result

| Metric | #31 control | #35 candidate / change |
| --- | ---: | ---: |
| Macro ROC-AUC | 0.847212 | 0.816869 |
| Paired macro delta | — | -0.030344 |
| Masked BCE | 0.462096 | 0.481576 (+0.019480) |
| Non-decreasing targets | — | 0 / 12 |

Paired bootstrap: 10,000 PatientID replicates, seed 20260825; **95% macro-delta interval [−0.040786, −0.019906]**, **positive fraction 0%**, 10,000 valid and 0 undefined replicates.

| Target | #31 control AUC | #35 candidate AUC | Delta |
| --- | ---: | ---: | ---: |
| ACL | 0.844484 | 0.763592 | -0.080892 |
| Baker's | 0.859292 | 0.775847 | -0.083444 |
| Contusion | 0.806030 | 0.786829 | -0.019201 |
| Effusion | 0.858636 | 0.849254 | -0.009382 |
| Fracture | 0.898986 | 0.843502 | -0.055484 |
| Lateral Meniscus | 0.791687 | 0.786877 | -0.004810 |
| Lateral OA | 0.869330 | 0.859349 | -0.009981 |
| MCL | 0.782509 | 0.765740 | -0.016769 |
| Medial Meniscus | 0.837201 | 0.804558 | -0.032644 |
| Medial OA | 0.892097 | 0.890710 | -0.001387 |
| PF OA | 0.857407 | 0.812834 | -0.044573 |
| Synovitis | 0.868889 | 0.863333 | -0.005556 |

The precommitted keep rule required macro delta ≥ +0.005, bootstrap positive fraction ≥ 0.95, and no severe target regression. Both mandatory thresholds failed; five targets regressed more than 0.02 AUC: ACL, Baker's, Fracture, Medial Meniscus, and PF OA. **Reject #35; retain #31.**

## Runtime, crop, and checkpoint evidence

- GPU: Tesla T4, CUDA 12.8. Full-run peak allocated memory: 5,227,067,904 bytes (4.87 GiB).
- Notebook log ends at 15,651.03 seconds (about 4 h 21 min), below the nine-hour boundary. Smoke projected 16,673.28 seconds (4 h 38 min), including reporting allowance and 25% margin.
- Training plus epoch validation loop: 14,219.89 seconds. Training-only epoch timers sum to 12,960.52 seconds; 525,480 valid anchor inputs, 40.54 anchors/s over those timers.
- Final reloaded validation: decode 199.91 s + forward 25.98 s = 225.89 s; 19,808 valid anchors, 87.69 anchors/s.
- Per training epoch: 3,957 studies, 21,895 series, 175,160 cropped anchors; zero crop-fallback studies. Final validation: 450 studies, 2,476 series, 19,808 cropped anchors; zero crop-fallback studies.
- Epoch validation macro AUCs: 0.776072, 0.797170, 0.816869. All three epochs completed, with 248 optimizer steps per epoch.
- Checkpoint reload reproduced predictions exactly: maximum absolute difference **0.0**, within tolerance 1e-5.
- Local implementation gates before execution: 449 tests passed, locked base/training syncs, Ruff lint/format, mypy, package build, and `git diff --check` passed. Final result recording changes documentation only.

## Smoke execution history

Versions 1–4 exposed setup/launch problems rather than training feasibility: duplicate identical retained checkpoints, a bundle built without GPU approval, and setup exceeding the one-hour smoke deadline. Version 4 measured 4,070.10 s of setup, including 1,449.60 s input discovery and 2,071.25 s index loading.

The shared setup fix skips DICOM trees during metadata/artifact discovery and resolves each series directory once while retaining slice-symlink containment checks. Version 5 reduced input discovery to **1.22 s** and total setup to **1,740.53 s**; index loading remained 1,725.90 s. The complete smoke took 2,016.20 s and passed at batch 2: peak allocated 5.50 GiB, reserved 8.71 GiB. Its measurements and source/hardware identity passed the full-run gate before approval.

## Disposition and next experiment

Issue #35 is complete as a negative experiment. Keep the #31 checkpoint and its 224 px inference configuration; do not run hidden inference or submit the rejected #35 checkpoint. The existing 0.830 public score remains the retained result.

A reasonable next independently scoped experiment is an encoder comparison at the retained 130 mm / 224 px representation, fixing sampling, supervision, split, aggregation, and evaluation. Precommit its resource and keep rules separately. This is a recommendation, not authorization to start another experiment; #35 provides no basis for a post-hoc resolution sweep.
