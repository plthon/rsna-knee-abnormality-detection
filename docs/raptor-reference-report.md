# Public Raptor/CoAtNet reproduction report

## Status

The repository contains a local reproduction of the public Dread Development Raptor inference pipeline. It is intentionally separate from the retained issue #31 ResNet-18 baseline.

Approved private, offline Kaggle T4 runs completed CPU/T4 prediction comparison, organizer-gold inference over all 58 eligible studies, and hidden-test inference. Checkpoint, cohort, notebook, submission, and artifact identities were verified after retrieval. The observed organizer-gold macro ROC AUC was `0.9126943301855063`, an absolute difference of `0.004054918131541352` from the checkpoint value `0.9167492483170476`. This is close as a sanity result but is not exact metric parity. The hidden submission artifact passed strict local verification with no raw study failures or fallback rows.

The one approved reproduction submission scored `0.924` on the public leaderboard, exactly matching the published score at Kaggle's displayed precision. Phase 1 therefore succeeded, and the Raptor inference implementation is a verified competitive external reference. Issue #31 remains unchanged as the retained internally trained model. No training or later leaderboard experiment has occurred, so the final A/B/C outcome remains pending Phase 2.

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
- Notebook prose and checkpoint metadata report approximately `0.9167` macro ROC AUC on 58 gold studies, while executable-source comments report `0.9054` for this model on a 58-study gate. The public source does not reconcile these values.

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

## Approved remote Phase 1A evidence

The authoritative evidence is private Kaggle kernel version 2 (`plthon/rsna-knee-raptor-reference-parity`) from clean commit `8ff2dd77066d1204ba304c527845bcc271f10783`. It was offline on a Tesla T4 and emitted only aggregate evidence. Version 2 restored the public notebook's `cudnn.benchmark = True` and `matmul.allow_tf32 = True` settings, both recorded as enabled in the output. Version 1 omitted those settings and is superseded; its per-target and macro ROC AUC values were nevertheless identical to version 2, so the omission did not explain the gold-metric difference.

Retrieved artifact integrity, the clean source commit, rendered notebook digest, exact checkpoint identity, exact target order, canonical cohort digest, and CUDA backend settings were verified locally. The version 2 `raptor-parity-report.json` SHA-256 is `f31eca95f4080a8ae0983668db4aaafbacfc3322997868f6f6d35ae675bbba53`; the version 2 `run-manifest.json` SHA-256 is `dfb21f53946247f828e163cc9f3614edc991238a7f01bda31c7bbc05406dd02b`.

Boundary flags in both retrieved artifacts record:

- hidden-test inference: `false`;
- competition submission: `false`; and
- training: `false`.

The CPU float32 and T4 public-autocast paths were compared on the first study in canonical cohort order. Across 12 sigmoid outputs:

- maximum absolute probability difference: `0.000521928071975708`;
- mean absolute probability difference: `0.00018805514264386147`.

The organizer-gold diagnostic used all 58 studies under `organizer-gold-all-eligible-v1`, with cohort digest `ea739a412709b0ac65429ca47f29161614ba384e2e92643301f817712cd6b7bc`. Every target had 58 known labels and both classes represented.

| Target | Observed ROC AUC | Checkpoint ROC AUC | Absolute difference |
|---|---:|---:|---:|
| ACL | 0.979166666667 | 0.966911764706 | 0.012254901961 |
| MCL | 0.972789115646 | 0.970521541950 | 0.002267573696 |
| Medial Meniscus | 0.960336538462 | 0.948317307692 | 0.012019230769 |
| Lateral Meniscus | 0.838509316770 | 0.880745341615 | 0.042236024845 |
| Medial OA | 0.981395348837 | 0.986046511628 | 0.004651162791 |
| Lateral OA | 0.843326885880 | 0.868471953578 | 0.025145067698 |
| PF OA | 0.819819819820 | 0.828828828829 | 0.009009009009 |
| Effusion | 0.983850931677 | 0.978881987578 | 0.004968944099 |
| Synovitis | 0.776583034648 | 0.757467144564 | 0.019115890084 |
| Baker's | 0.971014492754 | 0.987318840580 | 0.016304347826 |
| Contusion | 0.933873144399 | 0.914979757085 | 0.018893387314 |
| Fracture | 0.891666666667 | 0.912500000000 | 0.020833333333 |
| **Macro** | **0.912694330186** | **0.916749248317** | **0.004054918132** |

The largest target-level difference was `0.042236024845` for Lateral Meniscus. Therefore the run does **not** reproduce the checkpoint's stored per-target or macro metrics exactly. The 58-study number remains a leakage-prone sanity diagnostic, not an unbiased model-selection estimate, and this difference is recorded rather than treated as a pass or used for tuning.

The recorded gold-loop elapsed time was `174.7550272410001` seconds and peak allocated GPU memory was `8,173,242,368` bytes. The remote environment used Python `3.12.13`, PyTorch `2.10.0+cu128`, timm `1.0.26`, and OpenCV `4.13.0`.

## Approved remote Stop point 2 hidden-inference evidence

Private Kaggle kernel version 2 (`plthon/rsna-knee-raptor-reference-scoring`) completed from clean commit `93987f5b103ca71fa9009018ab25bc3fcde6b9dd`. It was offline on a Tesla T4, used the exact frozen public checkpoint, and called the shared Raptor reference runner once. Version 1 completed inference but failed its final output-set assertion because Kaggle creates `__notebook__.ipynb` in `/kaggle/working`; version 2 explicitly allowed only that Kaggle-managed file while retaining rejection of all other unexpected outputs.

The corrected bundle recorded:

- Jupytext source SHA-256: `3c20ec79f79e9d682de711abc4a778a709329c1257ed6ed9c90e1b59f97d4176`;
- rendered notebook canonical SHA-256: `2e71672f8a39c7f59182a32ca0576416b1775503c892a486eb7af8ec8d9f1a51`;
- rendered notebook file SHA-256: `5d07bf7ac1577936d300c03a19ff1c7949ab699746b8be703d67c298c75e77bf`; and
- kernel metadata SHA-256: `c4d37d52a9e6a185b6322180b599dbc9842ad67c786cad0f2833b5faa15fc0df`.

Only the five approved evidence artifacts were retrieved. Local aggregate verification checked their manifest-recorded byte sizes and SHA-256 digests, exact checkpoint and public-source identities, clean source provenance, strict submission validation, CUDA execution and backend flags, frozen configuration, and boundary flags. Verification returned:

- submission rows: `3`;
- minimum/maximum ranked probability: `0.0` / `1.0`;
- raw study failures: `0`;
- fallback rows: `0`;
- inference time: `33.87474749399996` seconds;
- peak allocated GPU memory: `8,173,242,368` bytes; and
- strict submission validation: `true`.

Artifact SHA-256 identities are:

| Artifact | SHA-256 |
|---|---|
| `submission.csv` | `4d6789a4449d7327382a55e1b2b06efb4adbd991b0f5aa01e4f00e57c8a894aa` |
| `submission-fallbacks.csv` | `536bf73e9e4aeb34109a55856e1ce649e57546ef2997f1fa2c36fe4e6861043a` |
| `submission-validation.json` | `b249e66a2c4f8b7d61f7dae33ecb0d0cf9c069fc7e4d7c7ee0f1540cd5715bae` |
| `raptor-reference-run.json` | `a10a70af8c6ee9df4ca69f99f8419f9f2ca5db7179cf1a2780c72377721794d6` |
| `run-manifest.json` | `06fd09d04a49fd9836257f53562099d286d7710f8a9515eaa48848d28401e6eb` |

The submission remained only in ignored local artifact storage. No hidden study identifiers or prediction rows were printed or added to the repository. Retrieved boundary evidence records:

- hidden-test inference: `true`;
- competition submission: `false`; and
- training: `false`.

## Approved Phase 1 reproduction submission

After separate Stop point 3 approval, the exact locally verified artifact was submitted once through private scoring kernel version 2. Kaggle submission `56485634` completed on 2026-09-23 with public score `0.924` and no reported private score. The difference from the published `0.924` at displayed precision is `0.000`.

This establishes Phase 1 success without tuning, alternate checkpoints, geometry changes, TTA, ensembles, or variant submissions. The Raptor checkpoint inference path is therefore a verified competitive external reference; it does not replace issue #31 as the retained internally trained model.

## Phase 2 best-evidence training reconstruction

The public training source and attached-input metadata remain frozen in `contracts/raptor-training-reference-v1.json`. The 2026-09-24 issue decision narrows the objective from unavailable byte-identical historical forensics to one best-evidence reconstruction. The contract now keeps three evidence semantics separate:

- `published_authoritative`: behavior and identities directly observed in public source, metadata, attached corpora, or verified Phase 1 evidence;
- `best_evidence_inferred`: the selected reconstruction recipe and supervision source, supported by public evidence but not proven to be the author's historical run;
- `reconstruction_specific_pinned_identity`: bytes, environment, manifests, and checkpoints measured from this reconstruction, without a historical-provenance claim.

All earlier source capture remains unchanged. The notebook is current metadata version 8, `dreaddevelopment/knee-mri-training-the-twelve-finding-model`, kernel ID `131724100`, with captured source SHA-256 `6ea209b03c33bf21d50d89e9e9696e2aecf90c4bcccdd1037c0b3711ab3d3699`. Its attached corpora remain, in order, `dreaddevelopment/knee-raptor-corpus` ID `11703975` v2 and `dreaddevelopment/knee-raptor-corpus-ext` ID `11735963` v1, describing 4,407 studies in `(4407, 44, 336, 336)` uint8 arrays. Exact observed file sizes and ETag-MD5 storage identities remain frozen; the approximately 21.89 GB payload has not been downloaded locally.

The selected reconstruction configuration is fixed without search or tuning:

- `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, timm pretrained initialization, resolution 384;
- 16 epochs, training/validation batch sizes 8/4, 12/24 training/evaluation windows;
- native gradient checkpointing, no frozen backbone blocks;
- AdamW backbone/head learning rates `3e-5` / `1e-3`, weight decay `0.02`;
- OneCycleLR with `pct_start=0.15`, stepped after every optimizer step;
- CUDA BF16 autocast, no GradScaler, no accumulation, gradient clipping at `3.0`;
- seed 42 using the source's Python/NumPy/`torch.manual_seed` calls only;
- uint8 scaling by 255, bilinear resize, normalization disabled, ROI disabled;
- only multiplicative intensity gain `[0.9, 1.1]`, with no flips or added augmentation;
- continuous soft-label `BCEWithLogitsLoss` with source prevalence-derived `pos_weight`;
- the 58 all-known organizer studies excluded from training and used only for 12-target macro ROC-AUC validation;
- strict macro-AUC improvement for the best checkpoint and equal tensor averaging of the top three epoch states.

`dreaddevelopment/rsna-knee-labels` ID `11640568` v3 / `labels_llm_soft.parquet` is selected as **best-evidence inferred** supervision because it has the required basename, 4,349-study coverage, common public author/source, and released-model context. It is still not represented as a proven historical notebook attachment. After separate Stop point 0 approval, only that 176,422-byte parquet was retrieved. Its SHA-256 is `15f7339514ff9af961da64c4b7eb40cb3c231aa6aaac93c99cde42201a2f9373`; its MD5 matches the frozen Kaggle ETag identity `40e6ca84dd07f329f7833f071c1a5db0`. It contains 4,349 unique studies, the exact UID-plus-12-target column order, finite probabilities in `[0.05, 0.95]`, and no out-of-range values.

The repository now contains a Raptor-specific reconstruction path that reuses the verified `RaptorClassifier` state layout, validates corpus/cohort separation, implements the captured 44-slice window behavior and exact selected optimizer/scheduler/checkpoint policy, and emits source-style best and top-three-averaged checkpoints. The private Kaggle bundle is training-only: it has no hidden-test inference or competition-submission path. It captures the actual Python/PyTorch/timm/CUDA environment and the resolved Hugging Face pretrained revision/file hash in run provenance. The frozen Phase 1 public checkpoint loader remains hash-pinned and unchanged in authority.

Historical uncertainties remain explicit: the author's exact label attachment, executed CLI, package environment, pretrained bytes, and completed training output are unknown. Reconstruction outputs will be compared with released checkpoint evidence (best epoch 12, stored gold macro ROC-AUC `0.9167492483170476`, public leaderboard `0.924`) but are not required to produce identical weights and will not trigger post-hoc tuning.

### Phase 2 resource result

After explicit Stop point 4 smoke approval, private kernel `plthon/rsna-knee-raptor-reconstruction-smoke` version 4 reached the first exact-recipe training backward pass using source commit `b30e909`. The smoke preserved CoAtNet/384, batch size 8, 12 windows per study, CUDA BF16 autocast, and native gradient checkpointing. It failed during checkpoint recomputation with `torch.OutOfMemoryError` before completing one optimizer step.

The T4 reported 14.56 GiB total GPU capacity, 12.02 GiB process memory in use, 11.64 GiB allocated by PyTorch, 241.66 MiB reserved but unallocated, and 2.54 GiB free when a further 3.38 GiB allocation was requested. `PYTORCH_ALLOC_CONF=expandable_segments:True` was already active, so reserved-memory fragmentation cannot account for the missing capacity. No full training, hidden inference, submission, variant, or tuning was started.

This is the issue-defined resource boundary: fitting the recipe would require materially changing batch size, window count, resolution, model, precision behavior, or accumulation policy. None is authorized as faithful reconstruction. The final issue #36 outcome is therefore **B. Phase 1 reproduced `0.924`; Phase 2 is resource-infeasible under current Kaggle limits.** The verified public checkpoint inference remains the competitive external reference, and #31 remains the retained internally trained model.
