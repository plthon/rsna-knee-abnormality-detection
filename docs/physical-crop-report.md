# Fixed 130 mm physical crop — issue #31

Status: GPU training completed; the physical crop passed the precommitted keep
rule. Hidden inference and competition submission have not run. The frozen #27
whole-image model remains the paired control.

## Frozen experiment

Control: #27, commit `e69f79af9c3de06e17a77e43bb41cb971879aed0` (the full identity and checkpoint digest are recorded
by the existing combined-baseline runner). Expected control macro ROC-AUC:
`0.838277`; hidden score: `0.819`. The candidate uses the same offline ImageNet
ResNet-18 initialization, 4,407-study supervision cohort, 3,957/450 patient-disjoint
split, 12 targets, masked BCE, three-epoch AdamW recipe, and seeds.

Validation membership digest must be
`8b1240fc2456779ba64fa115b23956adb131fb387f01c0f0424a933a2096ee88`.
The runner evaluates the verified #27 checkpoint on this same partition with
whole-image preprocessing. It resets the training seed after control evaluation.

Configuration: `configs/physical-crop.toml`. Full decoded sampled slices supply
the unchanged cross-slice 1st/99th percentiles. Normalize and clip first. Then use
DICOM `PixelSpacing=[row_mm,column_mm]` to request
`(round(130/row_mm), round(130/column_mm))` pixels. Python ties-to-even rounding is
fixed. Center crop or zero-pad; where parity prevents exact integer symmetry,
the extra pixel is bottom/right. Resize that physical square directly to
224×224, including anisotropic acquisitions. Keep eight deterministic slice
positions, grayscale RGB repetition, ImageNet normalization, and ordinary
slice → series → study means. No plane gate or attention is enabled.

Missing, malformed, nonfinite, or nonpositive spacing uses the original
whole-image resize/pad and records the reason. No spacing is guessed. Positive
spacing producing a zero-pixel or numerically unrepresentable crop fails
explicitly instead of inventing a different FOV.

Each preprocessing result identifies every decodable sampled slice's crop or
fallback and requested dimensions. Epoch and validation reports aggregate
slice/series/study counts and fractions, fallback reasons and crop-dimension
histograms. Mixed series/studies count in both rates; denominators include only
units with decoded slices. Training summaries are per epoch, avoiding inflated
unique-cohort counts. Any fallback is flagged for inspection, without changing
the experiment in response. Hidden inference reports the same evidence.

Candidate checkpoints use a distinct schema, preserve the FOV, fallback and
rounding policy, and cannot be loaded as the frozen whole-image control. The
shared inference loader restores spatial settings from the checkpoint.

## Local artifact and execution boundary

Build a private, execution-disabled review bundle:

```sh
uv run python scripts/kaggle.py build physical-crop
```

The existing builder's GPU-metadata approval requirement remains enforced.
Without approval the #31 draft has GPU disabled and `GPU_APPROVED=false`, and
cannot train. After explicit GPU-training approval, rebuild with
`--approve-gpu` from the clean committed revision. The bundle records exact
source SHA, dirty status, source/config/notebook and package hashes. No remote
push or run is part of local construction. The training notebook explicitly
uses `run_test_inference=False`.

Expected encoder compute is approximately 1.00× #27: unchanged eight 224×224
slices per series. #27 training took approximately 13,102 seconds (3.64 hours);
additional CPU crop/pad and paired control evaluation must be measured rather
than assumed free. The report already captures training/inference time,
valid slices/sec, peak CUDA memory, and checkpoint bytes.

## Completed GPU validation

The report hash matches the downloaded run manifest. Training source:
`5fe133ed01d418b45643f3fe0458be5f0d1f558d` (clean).

- Control macro AUC: 0.838277; candidate: 0.847212; delta: +0.008935.
- Paired 95% delta interval: [+0.002261, +0.015622]; positive fraction: 99.57%.
- 10,000 valid bootstrap replicates; zero undefined. Nine targets improved;
  no target regressed by more than 0.02. Decision: **keep**.
- Masked BCE: 0.520846 → 0.462096.
- The same 450-study validation membership digest was verified.
- Every decoded sampled slice used the crop: 175,158 training slices per epoch
  across 21,895 series/3,957 studies; 19,808 validation slices across 2,476
  series/450 studies. All slice/series/study crop rates were 100%; no fallbacks.
- Training: 11,312.74 seconds (3.14 hours); reported throughput: 51.34 valid
  slices/sec. Validation decode + forward: 204.73 seconds.
- Peak CUDA memory: 11,664,797,184 bytes. This is materially above the recorded
  #27 reference of 741,710,848 bytes; the report does not establish the cause.
- Checkpoint: 134,349,191 bytes; reload maximum absolute difference: 0.
  SHA-256: `55e9da7854d284a91c2569e25c799281d09301bae61dc5adcecebbd17209f013`.

| Target | Control AUC | Crop AUC | Delta |
| --- | ---: | ---: | ---: |
| ACL | 0.827635 | 0.844484 | +0.016849 |
| Baker's | 0.837621 | 0.859292 | +0.021671 |
| Contusion | 0.784774 | 0.806030 | +0.021256 |
| Effusion | 0.857027 | 0.858636 | +0.001609 |
| Fracture | 0.889032 | 0.898986 | +0.009954 |
| Lateral Meniscus | 0.780045 | 0.791687 | +0.011642 |
| Lateral OA | 0.868740 | 0.869330 | +0.000590 |
| MCL | 0.748820 | 0.782509 | +0.033690 |
| Medial Meniscus | 0.833164 | 0.837201 | +0.004038 |
| Medial OA | 0.899922 | 0.892097 | -0.007825 |
| PF OA | 0.863098 | 0.857407 | -0.005692 |
| Synovitis | 0.869444 | 0.868889 | -0.000556 |

Next: score the retained 130 mm/224×224 checkpoint. A later controlled experiment
may test higher resolution with the same FOV, without changing this checkpoint.

## Inference notebook

`notebooks/score_physical_crop.py` attaches the completed training kernel output,
selects the exact retained checkpoint by hash (root/nested identical copies are
allowed), and calls the shared scoring path with physical crop and ordinary
mean aggregation. It rebuilds the live test-only index and strictly validates
`submission.csv`. It never trains or submits. Local builds disable execution;
a clean revision and `--approve-gpu` enable the user-launched inference run.

```sh
uv run python scripts/kaggle.py build physical-crop-scoring --approve-gpu
uv run python scripts/kaggle.py push build/kaggle/physical-crop-scoring --approve-remote --approve-gpu
```

Running the notebook normally checks the mounted example test studies. Kaggle
provides hidden test studies during competition scoring; the same notebook then
indexes those live inputs. Competition submission remains a separate step.

## Local verification

Completed: locked dependency sync (including the existing training extra),
343 tests, strict mypy across source/tests/scripts/notebooks, Ruff lint and
format checks, source and wheel builds, and `git diff --check`. Synthetic checks
cover crop geometry, invalid spacing, unchanged normalization/sampling, actual
ResNet checkpoint reload, live hidden-index inference, and notebook safeguards.

The separate standards and specification reviews have no remaining findings.
Their fixes exclude entirely undecodable studies from prevalence denominators
and validate the crop notebook's actual verifier and mean-aggregation guard.
The definitive source SHA and artifact hashes are in the locally built bundle's
provenance; the prepared inference bundle remains execution-disabled until the
user rebuilds it with GPU approval.
