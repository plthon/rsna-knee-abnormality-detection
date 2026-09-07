# DINOv2 ViT-S/14 fine-tuning experiment (#30)

Reviewed 2026-09-07. The saved experiment completed 13 epochs and selected
epoch 7. The candidate is **rejected** under the precommitted keep rule;
retain the #27 ResNet-18 baseline and stop further DINOv2 tuning under this
2D mean-pooling representation. No hidden inference or submission was run.
The historical [#29 report](dinov2-report.md) remains unchanged.

## Recipe and validation

The existing pinned DINOv2 ViT-S/14 LVD-142M model used frozen input/token
parameters and blocks 0–5; blocks 6–11, final norm and classification head
were trainable. Two constant AdamW groups used backbone LR 8e-6 and head LR
1e-3, both with weight decay 1e-4. Micro-batch 4 × accumulation 4 preserved
effective batch 16 with mixed precision.

The same 4,407-study cohort, 3,957 training studies and 450 PatientID-disjoint
validation studies were retained, with 224×224 inputs, eight deterministic
slices, masked BCE and ordinary slice/series/study means. Validation membership
digest: `8b1240fc2456779ba64fa115b23956adb131fb387f01c0f0424a933a2096ee88`.

Early stopping used min 10/max 20 epochs, patience 4, min_delta 0.002.
It stopped at epoch 13 with four eligible non-improving epochs
(`patience_exhausted`). Epoch 7 had the highest observed macro AUC;
the early-stopping reference best remained 0.846693 because the later
improvement was smaller than 0.002. Reported checkpoint reload difference: 0.0.

## Results

| Metric | #27 ResNet-18 | #30 DINOv2 | Delta |
| --- | ---: | ---: | ---: |
| Macro AUC | 0.838277 | 0.847875 | +0.009598 |
| Masked BCE | 0.520846 | 0.493540 | -0.027306 |

The improvement over #29's 0.634180 macro AUC is +0.213695.
The paired PatientID bootstrap used 10,000 replicates and seed 20260825.
Its 95% delta interval is [-0.001906, 0.021668], with 94.73% positive
replicates. The +0.005 macro improvement criterion passes, but the required
95% bootstrap positivity does not. Only 6/12 targets are non-decreasing;
Baker's and PF OA regress by more than 0.02. The recorded decision is reject,
without post-hoc changes to the rule.

| Target | #27 AUC | #30 AUC | Delta |
| --- | ---: | ---: | ---: |
| ACL | 0.827635 | 0.833698 | 0.006064 |
| Baker's | 0.837621 | 0.794431 | -0.043190 |
| Contusion | 0.784774 | 0.836983 | 0.052209 |
| Effusion | 0.857027 | 0.852728 | -0.004299 |
| Fracture | 0.889032 | 0.953641 | 0.064608 |
| Lateral Meniscus | 0.780045 | 0.778165 | -0.001879 |
| Lateral OA | 0.868740 | 0.875821 | 0.007081 |
| MCL | 0.748820 | 0.790006 | 0.041187 |
| Medial Meniscus | 0.833164 | 0.849159 | 0.015996 |
| Medial OA | 0.899922 | 0.899552 | -0.000370 |
| PF OA | 0.863098 | 0.841428 | -0.021671 |
| Synovitis | 0.869444 | 0.868889 | -0.000556 |

## Epoch trajectory

| Epoch | Training loss | Validation macro AUC | Validation masked BCE |
| --- | ---: | ---: | ---: |
| 1 | 0.573986 | 0.798060 | 0.591798 |
| 2 | 0.511792 | 0.821258 | 0.495451 |
| 3 | 0.475489 | 0.833944 | 0.459933 |
| 4 | 0.452548 | 0.839767 | 0.458623 |
| 5 | 0.431317 | 0.846693 | 0.465790 |
| 6 | 0.408783 | 0.842533 | 0.465280 |
| 7 | 0.376875 | 0.847875 | 0.493540 |
| 8 | 0.345702 | 0.846384 | 0.468761 |
| 9 | 0.298953 | 0.846732 | 0.493626 |
| 10 | 0.243785 | 0.842453 | 0.511587 |
| 11 | 0.192215 | 0.837272 | 0.558100 |
| 12 | 0.135650 | 0.837836 | 0.559407 |
| 13 | 0.101171 | 0.834958 | 0.635221 |

## Resources

The report records 54,182.09 seconds (15.05 hours) total training time across
the retained history and 46.77 valid training slices/second. Candidate
validation decoding took 184.83 seconds and forward inference 31.73 seconds,
versus 14.15 seconds control forward inference (2.24×). The model has
22,061,196 parameters: 10,656,780 trainable and 11,404,416 frozen.
The checkpoint is 261,981,159 bytes.

Reported peak CUDA allocation is 1,177,867,264 bytes in the final reporting
process on Tesla T4. Because this execution reused completed training state,
this value does not establish the peak across the original training runs;
do not interpret it as the full training memory requirement.

## Execution evidence and notebook failure

Artifacts were retrieved with `kaggle kernels output
plthon/rsna-knee-dinov2-vit-s-14/9`. The returned manifest was captured at
2026-09-06T23:25:53Z and identifies clean source commit
`4a08936315bb08d83073848697c8e55e49ed1746`.
The returned log fails at about 1,798 seconds, whereas the reported Version 9
failure duration was 18,640.4 seconds. Exact version attribution remains
unresolved; these results are identified by manifest and hashes, not by an
assumed version number.

The log reports artifact verification successful, then prints macro AUC and
BCE before failing with `KeyError: 'percentile_interval'`. The display code
used that key, but the report schema exposes `interval`. The checkpoint,
report and validation predictions were saved before the display exception.
The notebook now reads the existing schema key; a focused test executes the
complete summary block and reproduces the old failure before the fix.
No remote rerun is needed to recover the saved metrics.

Artifact identity:

- Notebook source SHA-256: `19c86f562bc438b1162ca08a1877a2732f1181f55de3ab4c755699097d7dedfe`
- Rendered notebook SHA-256: `d821547c159754e49075fefb1b5d5fc5ec1631ccbcd660ad0622030fa915aeab`
- Configuration SHA-256: `910031e31534f1104c3afc4b002d51dfbaac5ce33b8afb76eb13594a9279c5cb`
- Checkpoint SHA-256: `58895e1cc935c80a315a71518661b7d1abb678fb6dae404018b475744c39e2c6`
- Report SHA-256 (locally checked against manifest): `f7b4e61b67de3a34c08fc5d13a010dcc3b0e423bcd389b806df37fcbbaa1586a`
- Validation predictions SHA-256 (manifest): `c2c573da1472ec9779ac9d346523385fe4e7b587ed2dc4195437d67d2fbc7e0d`

This review checked the report hash and replayed its display locally; it did
not rerun GPU training or independently reverify the full checkpoint bundle.
The notebook's successful verifier output and report provide the reload
evidence above. Remote notebook completion remains failed despite the saved
experiment results.
