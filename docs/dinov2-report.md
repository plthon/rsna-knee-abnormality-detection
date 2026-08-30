# DINOv2 ViT-S/14 backbone experiment

Verified 2026-08-31 for issue #29. The candidate was **rejected**. It failed
the precommitted paired-validation rule, so the repository retains the #27
ResNet-18 equal-series-mean baseline. No hidden-test inference or competition
submission was run.

## Controlled experiment

The control was the #27 ResNet-18 checkpoint from commit
`e69f79af9c3de06e17a77e43bb41cb971879aed0`, SHA-256
`1285f9890eb01e6a2383552e2ef1efcccdbc956ce14de049042bc268281240c4`.
The candidate changed only the encoder to end-to-end fine-tuned DINOv2
ViT-S/14 with its final normalized 384-dimensional CLS token and one linear
12-target head. Ordinary slice-to-series and series-to-study means remained
active; no plane gate or learned aggregation was present.

Both arms used the same 4,407-study gold-plus-public cohort, 3,957-study
training partition, 450-study PatientID-disjoint validation partition, three
epochs, seed `20260825`, 8 sampled slices per series, 224-pixel inputs, masked
BCE, AdamW settings, preprocessing, target order, and slice/series selection.
The validation membership digest was
`8b1240fc2456779ba64fa115b23956adb131fb387f01c0f0424a933a2096ee88`.

The official LVD-142M weight artifact was 88,283,115 bytes with SHA-256
`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.
The pinned upstream implementation commit was
`7764ea0f912e53c92e82eb78a2a1631e92725fc8`. The private Kaggle run was
offline and verified the artifact before strict model construction.

Batch 16 exhausted the Tesla T4, so the candidate used micro-batch 4 with four
gradient-accumulation steps, preserving effective batch 16. A retry exposed
2.51 GiB of reserved-but-unusable CUDA memory; the definitive run therefore
set `PYTORCH_ALLOC_CONF=expandable_segments:True` before importing PyTorch.
No resolution, slice-count, epoch, cohort, or optimizer change was made.

Execution source identity:

- clean Git commit: `c79f706d5e5d05d638d5a5fee7485a320e112d7d`
- notebook source SHA-256: `dd786ff678603d1b446f0216ecaf22501deaabfc66c908403e457416b3a11314`
- rendered-notebook SHA-256: `78c900e90c43575580c603ac438859d3a994e29ec8a793211b335871a9661c4c`
- config SHA-256: `aaea8b3c8334394fd013e4512b0c696194a1b44517ae6482ee039e0d870194b1`
- checkpoint SHA-256: `ea53d1dfceee2fd869027798b59926c61618721400fc1c3e0c9a440a478970bb`
- public-safe report SHA-256: `4142c6350861b34872b0ba758253bbea70fb5833db238bf48188ad9ae2e8154d`
- private paired-prediction SHA-256: `8f4110a8775b54c458c80e1b818917c8748660f9f8875b5f25dd9d55a0625db3`

The remote artifact verifier passed. Checkpoint reload reproduced predictions
with maximum absolute difference `0.0`. The downloaded report and paired
predictions matched their run-manifest hashes.

## Validation result

| Metric | #27 ResNet-18 | DINOv2-S | Delta |
| --- | ---: | ---: | ---: |
| Macro ROC-AUC | 0.838277 | 0.634180 | -0.204097 |
| Masked BCE | 0.520846 | 0.614756 | +0.093910 |

The paired PatientID bootstrap used seed `20260825` and 10,000 replicates. The
macro-AUC delta 95% percentile interval was `[-0.235594, -0.174270]`; 0% of
replicates were positive. All 12 targets regressed by more than the
precommitted `0.02` inspection threshold.

| Target | Known (+/−) | #27 AUC | DINOv2 AUC | Delta |
| --- | ---: | ---: | ---: | ---: |
| ACL | 411 (130/281) | 0.827635 | 0.619737 | -0.207898 |
| Baker's | 258 (118/140) | 0.837621 | 0.652331 | -0.185291 |
| Contusion | 365 (136/229) | 0.784774 | 0.670161 | -0.114613 |
| Effusion | 409 (259/150) | 0.857027 | 0.683449 | -0.173578 |
| Fracture | 206 (31/175) | 0.889032 | 0.650876 | -0.238157 |
| Lateral Meniscus | 404 (105/299) | 0.780045 | 0.572129 | -0.207915 |
| Lateral OA | 299 (96/203) | 0.868740 | 0.552571 | -0.316169 |
| MCL | 402 (83/319) | 0.748820 | 0.674019 | -0.074801 |
| Medial Meniscus | 426 (226/200) | 0.833164 | 0.636305 | -0.196858 |
| Medial OA | 330 (151/179) | 0.899922 | 0.623478 | -0.276444 |
| PF OA | 365 (186/179) | 0.863098 | 0.583994 | -0.279104 |
| Synovitis | 65 (45/20) | 0.869444 | 0.691111 | -0.178333 |

Candidate macro AUC rose from `0.607689` after epoch 1 to `0.621121` after
epoch 2 and `0.634180` after epoch 3, but remained far below the paired control.

## Resources and decision

The Tesla T4 run completed three epochs in 13,385.59 seconds (3.72 hours),
2.16% longer than #27's recorded 13,102.42 seconds. Candidate throughput was
43.14 valid slices/second. Paired validation forward time was 31.57 seconds
versus 14.10 seconds for the control (`2.24x`). Peak CUDA allocation was
14,707,285,504 bytes. The candidate had 22,061,196 parameters and a
264,970,363-byte checkpoint, versus 11,182,668 parameters and a
134,348,487-byte control checkpoint.

The published #27 peak-memory value came from checkpoint reuse rather than its
original training process, so the raw `19.83x` ratio is not a valid
training-to-training comparison. The DINOv2 run nevertheless nearly filled the
15,636,037,632-byte T4 and required the documented allocator compatibility
changes.

The candidate fails every keep signal: macro AUC is `0.204097` lower rather
than at least `0.005` higher, bootstrap positivity is 0% rather than at least
95%, and no target is non-decreasing. The result is **reject**. Generic
backbone replacement should stop under this pipeline; the retained ResNet-18
baseline remains the scoring model. Post-hoc optimization, hidden-test
inference, and competition submission were not run.
