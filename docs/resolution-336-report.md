# Resolution-only experiment — issue #35

Status: resource-smoke version 1 failed during control-checkpoint discovery, before resource measurements. **#31 at 224 px remains retained.** No candidate validation, GPU feasibility, hidden-test inference, or competition score is claimed.

Version 1 ended after 818.3 seconds with `Expected exactly one attached checkpoint.pt`: multiple attached copies matched the exact retained #31 SHA-256. The #35 smoke and full-training paths now use the existing hash-verified identical-copy option; other experiments retain their strict default. The corrected smoke has not been launched.

## Frozen identities

- Control: #31, ResNet-18, 130 mm centered crop, 224×224, eight deterministic full-span anchors, same-slice RGB, ordinary anchor/series/study means.
- Control training source: `5fe133ed01d418b45643f3fe0458be5f0d1f558d`.
- Control checkpoint SHA-256: `55e9da7854d284a91c2569e25c799281d09301bae61dc5adcecebbd17209f013`.
- Control validation macro ROC-AUC: **0.847212**; public score: **0.830**.
- Candidate config: `configs/resolution-336.toml`. The only scientific configuration difference from `configs/physical-crop.toml` is `image_size = 336`.
- Candidate checkpoint schema: `rsna-knee-resolution-336-v1`. The frozen #31 loader rejects it; the 336 loader rejects 224 metadata or the wrong schema.
- Validation membership: the identical 450 PatientID-disjoint studies, digest `8b1240fc2456779ba64fa115b23956adb131fb387f01c0f0424a933a2096ee88`; training membership remains 3,957 studies from the unchanged 4,407-study supervision cohort.

The crop and DICOM handling code are unchanged. Series percentiles still use the same eight source slices before cropping/resizing. PixelSpacing, deterministic center/padding, invalid-spacing fallback, slice ordering, anchor indices, channel repetition, ImageNet normalization, series inclusion, and equal means remain shared with #31. The ResNet-18 architecture, conv1, adaptive pooling, parameter count (11,182,668), and exact offline ImageNet initialization are unchanged. The optimizer remains three epochs of AdamW, LR/weight decay 0.0001, with the retained seeds and masking/target order.

The candidate requires cached #31 predictions, checked against the retained checkpoint, source/configuration, CSV digest, exact study/PatientID/label alignment, and target order. It never retrains #31 or silently substitutes fresh control predictions.

## Resource smoke and batch policy

The smoke stage of `notebooks/train_resolution.py` calls the existing `train_epoch` and `predict_partition` interfaces. It measures:

- a fixed 16-study stress set with the largest indexed anchor counts;
- 64 deterministically shuffled training studies;
- 16 deterministically shuffled validation studies;
- peak CUDA allocated/reserved bytes, training step time, valid anchor inputs/sec, validation decode/forward time, and crop/fallback counts.

Default physical/effective batch is **16/16**, accumulation **1**. Only CUDA out-of-memory or peak reserved memory at least 90% of device memory permits the next smaller physical batch: 8, 4, 2, then 1. Each retry initializes a fresh model/optimizer and keeps effective batch 16 with deterministic accumulation. LR, weights, aggregation, and scientific settings are not tuned. Accumulation preserves masked-loss weighting, including incomplete groups; it does not make BatchNorm statistics mathematically identical to a physical batch of 16. Any such resource adaptation is explicitly recorded.

The smoke has finite workloads and a one-hour deadline checked between training batches and validation studies. An in-progress decode/kernel is not forcibly interrupted. It emits `resolution-resource-smoke.json`, never a trained candidate checkpoint. A timeout or memory-infeasible result cannot unlock full training.

The projection takes the slower of study-scaled and anchor-scaled measurements, covers three training epochs **and all five validation passes** (three epoch evaluations plus before/after reload), includes measured setup time and a 30-minute checkpoint/bootstrap/reporting allowance, then adds 25% safety margin. It must be strictly below **32,400 seconds (9 hours)**. Hidden inference is separate and excluded. This is an estimate, not a guarantee.

A safe memory configuration that misses the runtime limit is reported as resource-infeasible; it does not trigger further batch or resolution tuning. Full training checks smoke provenance, configuration, minimal batch-reduction history, recomputed runtime, and the actual accelerator identity/memory/CUDA version. Smoke and full training must use the same committed Git SHA.

## Local commands and approval boundaries

The existing controlled-experiment builder renders one shared notebook with an explicit `smoke` or `full` stage. The stages retain separate bundle paths, kernel identities, and approvals; smoke cannot fall through into full training. No separate resolution builder or smoke bootstrap is maintained.

Build an execution-disabled smoke bundle locally:

```sh
uv run --extra training python scripts/kaggle.py build resolution-336 --stage smoke
```

After explicit smoke approval, from the **clean committed source revision**:

```sh
uv run --extra training python scripts/kaggle.py build resolution-336 --stage smoke --approve-gpu
uv run --extra training python scripts/kaggle.py push build/kaggle/resolution-336-smoke --approve-remote --approve-gpu
```

These commands launch **only the resource smoke**. They are not full-training approval. The notebook is private, offline, and attaches the retained index, supervision/initialization inputs, and #31 output.

After the smoke finishes, retrieve its JSON and inspect the measured batch/memory/throughput/projection. Only after a separate full-training approval may the same clean revision build and push:

```sh
uv run --extra training python scripts/kaggle.py build resolution-336 --stage full --resource-smoke /absolute/path/resolution-resource-smoke.json --approve-gpu
uv run --extra training python scripts/kaggle.py push build/kaggle/resolution-336-full --approve-remote --approve-gpu
```

Full bundles cannot be built without successful, matching smoke evidence. Full training requires `run_test_inference=False`, including direct API calls; it cannot cross the hidden-inference approval boundary. Hidden scoring requires a verified validation-kept artifact, the exact candidate checkpoint hash, `physical_crop=True`, and `resolution_336=True`; it restores resolution from checkpoint metadata, rebuilds the live hidden index, and uses existing strict submission validation. No separate inference bundle or competition submission is prepared before a keep result.

## Precommitted decision and pending results

Keep only with paired macro delta **≥ +0.005**, bootstrap positive fraction **≥ 0.95**, and no target regression worse than −0.02. Report the diagnostic count of non-decreasing targets (preference ≥8/12). The existing 10,000-replicate paired PatientID bootstrap and seeds are unchanged. Otherwise reject 336 and retain #31 at 224. No intermediate resolution/FOV/model rescue is allowed.

The shared full-run report records all 12 control/candidate AUCs and deltas, macro delta, bootstrap interval/positive fraction/valid and undefined counts, masked BCE, crop/fallback evidence, training and validation timing, throughput, peak GPU memory, checkpoint bytes, and reload tolerance. Candidate fields below remain **not measured** until the separately approved remote stages finish.

| Target | #31 control AUC | #35 candidate AUC / delta |
| --- | ---: | --- |
| ACL | 0.844484 | Not measured |
| Baker's | 0.859292 | Not measured |
| Contusion | 0.806030 | Not measured |
| Effusion | 0.858636 | Not measured |
| Fracture | 0.898986 | Not measured |
| Lateral Meniscus | 0.791687 | Not measured |
| Lateral OA | 0.869330 | Not measured |
| MCL | 0.782509 | Not measured |
| Medial Meniscus | 0.837201 | Not measured |
| Medial OA | 0.892097 | Not measured |
| PF OA | 0.857407 | Not measured |
| Synovitis | 0.868889 | Not measured |

Source for control results: `docs/physical-crop-report.md`. Local tests cover paired crop/resize behavior, spacing fallback, percentiles, anchors, RGB and mean pooling, exact offline weights, real 336 checkpoint reload, live synthetic hidden inference, configuration drift, keep thresholds, and smoke/full approval gates. Final gate results and the implementation SHA are supplied with the handoff.

No retain/reject/resource-infeasible scientific decision has been made for #35. A subsequent experiment recommendation depends on its result; the implementation records a separately scoped encoder comparison as a possible follow-up, not authorization to execute one.
