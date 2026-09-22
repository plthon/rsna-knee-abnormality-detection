# Public Raptor/CoAtNet reproduction report

## Status

The repository now contains a local reproduction of the public Dread Development Raptor inference pipeline. It is intentionally separate from the retained issue #31 ResNet-18 baseline.

Only public asset retrieval and local synthetic verification have occurred. No Kaggle CPU or GPU execution, organizer-gold inference, hidden-test inference, competition submission, training, or later leaderboard experiment has been run. Those actions remain behind the approval boundaries in issue #36.

No final A/B/C reproduction outcome is assigned because public-leaderboard parity has not been attempted.

## Frozen public identities

### Inference notebook

- URL: <https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model>
- Kaggle notebook ID: `131625748`
- Version: `7`
- Source SHA-256: `859aae12ecea593b45ab0a5f6463fa573cd3083f340c161e920f10fbba29e81b`
- Kernel metadata SHA-256: `41bd29eaa2b8079dacf886b103793c4b849cb96aceef6751f77ec838fa3cee7f`
- Last run: `2026-09-18T23:06:07.922Z`
- Machine shape: `NvidiaTeslaT4`
- Internet: disabled
- Docker image: `gcr.io/kaggle-images/python@sha256:dafd4ce5668bbf1ad422e4c109e0f18c9623c3a7c7f48b0235f13142755c40b9`

### Model dataset and checkpoint

- Dataset URL: <https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense>
- Dataset ID/version: `11742423` / `2`
- Dataset metadata SHA-256: `eea3f4c07fad7c2490320be3a5297ec031b3897368aef99ea318e8f95f948b3a`
- License: CC0-1.0
- Checkpoint: `raptor_ft_coatnet_v4_full.pt`
- Checkpoint size: `292829402` bytes
- Checkpoint SHA-256: `89606f05849838529e1b4658d28fb049623205d9853371548403bca460361ded`
- Architecture: `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`
- Resolution: `384`
- Epoch: `12`
- Source metadata: `timm-pretrained`
- State-dict entries: `486`
- Gold macro ROC AUC stored in the checkpoint: `0.9167492483170476`
- Public leaderboard score reported by the author: `0.924`

The loader verifies the exact file hash before deserialization, uses `torch.load(..., weights_only=True)`, requires the exact top-level schema and target order, constructs the named CoAtNet without pretrained weight retrieval, and loads the state dictionary strictly.

## Executable preprocessing contract

Each study becomes a 64-slice `uint8` stack with five fixed slots in this order:

1. Sagittal, fluid-sensitive preferred: 18 slices.
2. Sagittal, non-fluid-sensitive preferred: 14 slices.
3. Coronal, fluid-sensitive preferred: 12 slices.
4. Coronal, non-fluid-sensitive preferred: 8 slices.
5. Axial, no fluid preference: 12 slices.

Series selection preserves `test_series.csv` row order and excludes series already used by an earlier slot. Slices are geometry-sorted using the dot product of `ImagePositionPatient` and the normal derived from `ImageOrientationPatient`; `InstanceNumber` is the fallback. Sampling uses rounded linear spacing from `int(n * 0.06)` through `max(int(n * 0.94) - 1, lo)`. Repeated source indices are retained.

Pixel data is modality-LUT transformed and converted to `float32`. `MONOCHROME1` uses `pixels.max() - pixels`. The 2nd and 98th percentiles are computed over all successfully decoded sampled pixels in a selected series. Each slice is normalized and clipped to `[0, 1]`, center-cropped to a square corresponding to 140 mm using only `PixelSpacing[0]`, clamped to the source dimensions without padding, resized to 336×336 with OpenCV `INTER_AREA`, multiplied by 255, and truncated to `uint8`. Missing slots and failed decodes remain zero-filled.

Inference constructs exactly 42 previous/center/next three-channel windows. Only the first and last nonzero stack slices bound candidate centers; internal zero-filled gaps and slot boundaries are not masked. Rounded linear sampling may omit or duplicate candidate centers. Windows are resized from 336 to 384 with PyTorch bilinear interpolation and `align_corners=False`, then ImageNet-normalized.

The model uses a 1024-feature CoAtNet backbone followed by layer normalization and a 1024→256→12 attention projection. Softmax attention is calculated across windows independently for each target. Each target has its own classification weight and bias. No attention mask is supplied.

Logits receive a sigmoid. Study-level failures retain an initial raw probability of 0.5. The one model arm is still transformed per target with `argsort(argsort(x)) / max(1, study_count - 1)`, so output values are percentile ranks rather than direct sigmoid probabilities. The existing strict submission writer enforces entity coverage, target order, finite values, and `[0, 1]` bounds.

## Executable source discrepancies

The executable source controls where comments and prose disagree:

- Prose says missing slots are skipped, but code uses the mask only to bound center selection. Internal zero-filled gaps can still produce attention windows.
- Prose describes a 140 mm pixel-spacing crop generally, while code uses `PixelSpacing[0]` for both square dimensions, clamps to the source image, and never pads.
- The `K_EVAL` comment says every window position, while code samples exactly 42 rounded positions and can omit or duplicate centers.
- Stale comments mention three arms, 24 windows, and older scores. Executable configuration contains one arm and 42 windows.
- Submission code applies percentile ranking even though prose commonly refers to model probabilities.

## Local evidence

The local verification suite uses independently authored synthetic fixtures; it does not import or execute the downloaded notebook source. It checks:

- the frozen public identities, target order, slot order, and counts;
- row-order series preference and series reuse prevention;
- sampling endpoints and duplicate indices for short series;
- fixed stack shape and missing-slot zero filling;
- square clamped crop behavior and exact OpenCV resize;
- 42-window center selection across internal gaps and previous/center/next channel order;
- target-specific, unmasked attention pooling;
- exact percentile-rank behavior;
- strict submission ordering and end-to-end local artifact generation; and
- strict loading of the downloaded public checkpoint, two clean reloads, bit-identical synthetic probabilities, and finite sigmoid bounds.

The strict checkpoint test constructs a 73,136,096-parameter model. The locked local environment used for this verification contains:

- Python 3.12
- PyTorch `2.10.0`
- timm `1.0.29`
- OpenCV `4.14.0`

The public notebook does not state its exact timm or OpenCV package versions. Its Docker image digest is frozen instead; no version was guessed.

## Local invocation

The local entry point is:

```bash
uv run --extra training python scripts/run_raptor_reference.py \
  --input-root /path/to/input \
  --work-root /path/to/output \
  --checkpoint /path/to/raptor_ft_coatnet_v4_full.pt \
  --device cpu
```

It writes a validated prediction CSV, fallback record, run report, and provenance manifest. It contains no remote execution, upload, competition submission, or training operation.

## Pending approval-gated evidence

The following fields remain deliberately unfilled:

- Kaggle CPU parity result: not run.
- Kaggle T4 parity result: not run.
- Organizer-gold cohort result: not run.
- Hidden-test inference result: not run.
- Public leaderboard result: not submitted.
- Training result: not run.
- Final issue #36 outcome classification: pending.

The next boundary is Stop point 1 in issue #36, before any remote Kaggle parity or organizer-gold execution.
