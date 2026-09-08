# Fixed 130 mm physical crop — issue #31

Status: implemented locally; GPU training, hidden inference and submission have
not been run. The retained baseline remains #27 whole-image ResNet-18.
No validation gain, fallback prevalence, or keep decision is claimed yet.

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

## Pending GPU results

The generated public-safe training report will contain control/candidate macro
AUC, all 12 target AUCs and deltas, non-decreasing target count, secondary masked
BCE, the established 10,000-resample PatientID bootstrap interval, positive
fraction and valid/undefined counts, crop/fallback prevalence and dimension
histograms, resource measurements, and checkpoint reload error (tolerance
`1e-5`). Actual values require the approved GPU run.

Keep only with macro delta ≥0.005, positive bootstrap fraction ≥0.95, and no
regression worse than −0.02 AUC; report ≥8/12 non-decreasing as a diagnostic
preference. This conservatively treats the issue's approximate regression
inspection threshold as a veto. A miss retains #27 without crop-size tuning.
If kept, the next controlled question is resolution at the same 130 mm FOV;
otherwise investigate slice context/position. Hidden inference and submission
remain separate approval steps after a keep result.

## Local verification

Completed: locked dependency sync (including the existing training extra),
341 tests, strict mypy across source/tests/scripts/notebooks, Ruff lint and
format checks, source and wheel builds, and `git diff --check`. Synthetic checks
cover crop geometry, invalid spacing, unchanged normalization/sampling, actual
ResNet checkpoint reload, live hidden-index inference, and notebook safeguards.

The separate standards and specification reviews have no remaining findings.
Their fixes exclude entirely undecodable studies from prevalence denominators
and validate the crop notebook's actual verifier and mean-aggregation guard.
The definitive source SHA and artifact hashes are in the locally built bundle's
provenance; GPU execution remains disabled pending approval.
