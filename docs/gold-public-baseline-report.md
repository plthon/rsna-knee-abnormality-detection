# Gold + public report-label combined baseline

Verified 2026-08-25 for `rsna-knee-abnormality-detection`. The canonical
public-safe machine-readable contract is
[`contracts/gold-public-baseline.json`](../contracts/gold-public-baseline.json).
Private study identifiers, predictions, labels, and checkpoints remain under the
ignored local `.artifacts/` directory and private Kaggle outputs; none are committed.

## Result and summary

A full 3-epoch ResNet-18 baseline training run was executed on Kaggle GPU from
a 4,407-study unified supervision cohort. The optimizer saw 3,957 studies in
the deterministic training partition; 450 studies were held out for diagnostic
validation:
- **58 organizer-gold studies** supervised directly by competition ground truth (`train.csv`).
- **4,349 non-gold studies** supervised by frozen issue #24 public report-derived labels (`report_labels_v2.csv`).
- **Verdict mapping**: `YES -> 1`, `NO -> 0`, `UNK -> masked in loss`.
- **Precedence**: Gold labels strictly take precedence over public report labels.

On the diagnostic 10% PatientID-disjoint validation split (450 studies, 450 patients),
the trained model achieved:
- **Macro ROC-AUC**: **0.8383**
- **Mean Binary Cross-Entropy**: **0.5208**
- **Mean Accuracy @ 0.5**: **0.7720**

Checkpoint reload reproducibility verified exact numerical consistency (**max diff: 0.0**, tolerance `1e-5`). The local run inferred the three mounted example-test studies with 0 fallbacks and produced a strictly validated `submission.csv`; this is not evidence that hidden competition scoring works.

## Supervision cohort inventory

The full 4,407-study supervision cohort (`digest: b066a7e6cd7119f4bef330a006a86a0e08e664d7b5347a111ec781e72abe1fdc`) contains:

| Target | Gold + / − | Public + / − / masked | Total Known | Total Prevalence |
| --- | ---: | ---: | ---: | ---: |
| ACL | 24 / 34 | 1,194 / 2,804 / 351 | 4,056 | 30.0% |
| Baker's | 12 / 46 | 1,098 / 1,257 / 1,994 | 2,413 | 46.0% |
| Contusion | 19 / 39 | 1,261 / 2,182 / 906 | 3,501 | 36.6% |
| Effusion | 35 / 23 | 2,616 / 1,301 / 432 | 3,975 | 66.7% |
| Fracture | 18 / 40 | 295 / 1,594 / 2,460 | 1,947 | 16.1% |
| Lateral Meniscus | 23 / 35 | 1,091 / 2,823 / 435 | 3,972 | 28.0% |
| Lateral OA | 11 / 47 | 1,071 / 1,843 / 1,435 | 2,972 | 36.4% |
| MCL | 9 / 49 | 749 / 3,176 / 424 | 3,983 | 19.0% |
| Medial Meniscus | 26 / 32 | 2,315 / 1,786 / 248 | 4,159 | 56.3% |
| Medial OA | 15 / 43 | 1,597 / 1,641 / 1,111 | 3,296 | 48.9% |
| PF OA | 21 / 37 | 1,967 / 1,581 / 801 | 3,606 | 55.1% |
| Synovitis | 27 / 31 | 564 / 114 / 3,671 | 736 | 80.3% |

Every study belongs to exactly one PatientID in this competition cohort.

## Diagnostic validation performance

Evaluated on the 10% validation partition (450 studies):

| Target | Known / Missing | Accuracy @ 0.5 | Binary Cross Entropy | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| ACL | 411 / 39 | 78.8% | 0.5086 | 0.8276 |
| Baker's | 258 / 192 | 77.1% | 0.4882 | 0.8376 |
| Contusion | 365 / 85 | 73.7% | 0.5654 | 0.7848 |
| Effusion | 409 / 41 | 77.0% | 0.4625 | 0.8570 |
| Fracture | 206 / 244 | 86.4% | 0.2978 | 0.8890 |
| Lateral Meniscus | 404 / 46 | 79.2% | 0.5301 | 0.7800 |
| Lateral OA | 299 / 151 | 78.3% | 0.5061 | 0.8687 |
| MCL | 402 / 48 | 79.4% | 0.5144 | 0.7488 |
| Medial Meniscus | 426 / 24 | 64.8% | 0.8730 | 0.8332 |
| Medial OA | 330 / 120 | 73.9% | 0.5878 | 0.8999 |
| PF OA | 365 / 85 | 77.8% | 0.4877 | 0.8631 |
| Synovitis | 65 / 385 | 80.0% | 0.4286 | 0.8694 |
| **Macro Average** | **-** | **77.2%** | **0.5208** | **0.8383** |

## Training execution and resources

- **Environment**: Kaggle GPU kernel `plthon/rsna-knee-gold-public-baseline` on Tesla T4.
- **Model Architecture**: ResNet-18 backbone initialized from offline `ResNet18_Weights.IMAGENET1K_V1` with linear 12-target multi-label classification head.
- **Preprocessing**: 8 sampled slices per series, 224×224 resolution, provisional percentile normalization.
- **Training Duration**: 13,102.42 seconds (~3.64 hours) for 3 full epochs across 3,957 training-partition studies; 450 studies were diagnostic validation only.
- **Peak CUDA Memory**: 741,710,848 bytes (~741.7 MB, well within 15.6 GB T4 limit).
- **DICOM / Decode Failures**: 0 failures across all studies.

## Checkpoint and reload verification

- **Path**: `checkpoint.pt`
- **SHA-256**: `1285f9890eb01e6a2383552e2ef1efcccdbc956ce14de049042bc268281240c4`
- **Size**: 134,348,487 bytes
- **Reload diff**: Maximum absolute difference between original and reloaded validation predictions is `0.0` (passes tolerance `1e-5`).

## Test inference and submission artifact

- **Inference Time**: 2.99 seconds on Tesla T4.
- **Prediction Count**: 3 competition test studies, 12 targets each.
- **Output Range**: `[0.01374, 0.96729]` (finite sigmoid outputs; not calibrated probabilities).
- **Fallback Count**: 0 studies required fallback priors.
- **Submission SHA-256**: `2d36fcc40f046da23dc7b6f90d0130aad0458f063c65ea4e2686497155bee3e9`
- **Submission Schema**: Verified against competition contract.

## Competition scoring boundary

The three-study result above is only a local/example execution. The final
scoring bundle is inference-only: it verifies checkpoint
`1285f9890eb01e6a2383552e2ef1efcccdbc956ce14de049042bc268281240c4`, reads the
live scoring-time `test.csv` and `test_series.csv`, builds a test-only index
with the shared geometry ordering, uniform slice selection, and preprocessing
path, then runs inference and strict submission validation. It does not use the
frozen example-test index, construct the training cohort, split data, train
epochs, or run validation inference.
