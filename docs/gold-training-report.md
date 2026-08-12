# Gold-only ResNet-18 smoke training

Verified 2026-08-12 for `rsna-knee-abnormality-detection`. The public-safe
machine-readable evidence is
[`contracts/gold-training.json`](../contracts/gold-training.json), and the
reusable initialization contract is
[`contracts/resnet18-imagenet1k-v1.json`](../contracts/resnet18-imagenet1k-v1.json).
The checkpoint, predictions, row identifiers, and private logs remain under
ignored local storage and in the private Kaggle artifact.

## Platform run

Private Kaggle kernel
[`plthon/rsna-knee-gold-model-smoke`](https://www.kaggle.com/code/plthon/rsna-knee-gold-model-smoke),
version 2, completed successfully with internet and TPU disabled on one Tesla
T4. The server metadata pins `NvidiaTeslaT4`, and the server-returned code-cell
source exactly matches the rendered bundle.

Version 1 was deliberately not accepted as a training run. Kaggle assigned a
Tesla P100 (`sm_60`), but its PyTorch 2.10 image contained CUDA kernels only for
`sm_70` and newer. The required real CUDA operation failed before model or data
construction. Pinning `NvidiaTeslaT4` fixed the scheduling mismatch without
changing model code, data, weights, or training configuration.

The successful CUDA gate executed
`(torch.ones(1, device='cuda') * 2).item()` and observed `2.0`. Runtime versions
were Python 3.12.13, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, and CUDA
12.8. The run used source base commit `1ab01a4` with a dirty worktree; exact
source, package archive, configuration, server-source, artifact, and log hashes
are retained in the machine-readable contract and checkpoint.

## Initialization and model contract

The exact `ResNet18_Weights.IMAGENET1K_V1` bytes were placed in private Kaggle
Dataset `plthon/torchvision-resnet18-imagenet1k-v1`, version 1. Kaggle reports
the Dataset ready and lists `resnet18-f37072fd.pth` at 46,830,571 bytes. The
notebook independently revalidated SHA-256
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`
before constructing `torchvision.models.resnet18`. Runtime internet, alternate
weight fallback, and random-initialization fallback are prohibited.

One shared ResNet-18 encodes valid 224×224 slices. Padded slice pixels are not
passed to the encoder. Slice-to-series and series-to-study reductions are masked
means, so padded slices and empty padded series cannot influence the output.
The verified 12-target order drives the output head and checkpoint.

## Canonical supervision

The notebook re-derived `organizer-gold-all-eligible-v1` from the authoritative
competition `train.csv` and accepted index. It rejected any mismatch against
the #7 contract before model construction. The observed 58-study digest was
`ea739a412709b0ac65429ca47f29161614ba384e2e92643301f817712cd6b7bc`,
exactly matching #7. All 12 targets had both positive and negative organizer-gold
support. Unknown labels remain representable and are masked in binary
cross-entropy. Report text was not interpreted or used, and no report-derived
label entered training or evaluation.

## One-epoch diagnostics

The approved mixed-precision epoch attempted and optimized all 58 canonical
studies in 58 optimizer steps. It processed 2,688 valid slices in 58.53 seconds
(45.92 slices/s), reached 1,106,189,312 bytes peak CUDA memory, and recorded zero
training or inference decode errors. Mean online batch loss was 0.67299; test
inference after checkpoint reload took 3.17 seconds.

The following values are online smoke diagnostics on the training cohort. They
are recorded only to prove finite metric execution and are not estimates of
model quality, generalization, statistical adequacy, or competition performance.

| Target | BCE | Accuracy at 0.5 | ROC AUC |
| --- | ---: | ---: | ---: |
| ACL | 0.70754 | 0.46552 | 0.41422 |
| MCL | 0.54584 | 0.77586 | 0.31746 |
| Medial Meniscus | 0.70821 | 0.50000 | 0.32091 |
| Lateral Meniscus | 0.70867 | 0.43103 | 0.37764 |
| Medial OA | 0.62169 | 0.70690 | 0.50078 |
| Lateral OA | 0.69613 | 0.58621 | 0.52805 |
| PF OA | 0.67501 | 0.63793 | 0.25032 |
| Effusion | 0.70494 | 0.53448 | 0.28323 |
| Synovitis | 0.70554 | 0.44828 | 0.40382 |
| Baker's | 0.64549 | 0.63793 | 0.61957 |
| Contusion | 0.64469 | 0.60345 | 0.57220 |
| Fracture | 0.71212 | 0.50000 | 0.45833 |

## Checkpoint, reload, and submission

Run `gold-training-smoke-47fa76c2c31e4c91812d5c8b791f0440` produced a
134,348,295-byte checkpoint with SHA-256
`ff285942dd75a165a5079cee7558aac1afafb1e75269d00d11d2d1cdd04ace82`.
It contains model, optimizer, AMP scaler, Python/NumPy/CPU/CUDA RNG state,
targets, priors, preprocessing, cohort and label contracts, initialization
provenance, training configuration, code provenance, runtime, and artifact
hashes. Model, optimizer, and scaler state were reloaded before inference.

The established submission writer and validator accepted three example-test
rows by 12 targets, all produced from the reloaded checkpoint. All values were
finite and within `[0.17346, 0.65918]`; no fallback was needed. The submission
SHA-256 is
`cb93aad6f20c7e026e3af75cc4a461c830863ce695069bbc1737a5f7acf31e1e`.
The private downloaded run was added to the ignored append-only experiment
registry only after manifest and artifact verification.

The independent local CPU check also completed forward, backward, optimizer
step, checkpoint save/reload, and finite prediction for a synthetic study. Its
output shape was `1×12`, loss was 0.59763, and checkpoint SHA-256 was
`594f99dce16c500f1672e9cf1cc4c126a7537da4f99fc3be8740da33226328c5`.

## Acceptance-criterion map

| Criterion | Evidence | Outcome |
| --- | --- | --- |
| Consume #7 and independently match canonical digest | Frozen rule, 58-study re-derivation, exact digest | Satisfied |
| Verified targets and both-class support | Ordered 12-output head; all targets positive and negative | Satisfied |
| Mask unknown gold and exclude padding | Masked BCE; valid-only encoder path and two masked means | Satisfied |
| Exact offline initialization before construction | Private Dataset v1; size/SHA check; no network or fallback | Satisfied |
| Complete local CPU lifecycle | Forward/backward/step/save/reload/predict evidence | Satisfied |
| Approval and real CUDA gate | Guard tests; explicit approval; real T4 operation returned `2.0` | Satisfied |
| One canonical mixed-precision epoch | 58 attempted and optimized studies; one completed epoch | Satisfied |
| Engineering diagnostics without quality claims | Time, memory, throughput, decode, loss, and scoped metrics | Satisfied |
| Complete resumable checkpoint | Strict inventory and field/value verification; optimizer/scaler reload | Satisfied |
| Reloaded predictions pass validator | 3×12 valid output, zero fallbacks, verified SHA-256 | Satisfied |
| No report-derived labels | Organizer-gold-only source and explicit false provenance flag | Satisfied |
