# Issue #34: four-slot target-specific masked attention

Status: locally implemented; no Kaggle training, hidden inference, or competition
submission executed. #31 remains retained. Validation results and retain/reject
decision are pending the separately authorized experiment.

## Frozen experiment

Control: #31, source `e85e24ebf0b78c6c73e63465d08c6b3dfa098141`, checkpoint SHA-256
`55e9da7854d284a91c2569e25c799281d09301bae61dc5adcecebbd17209f013`.
Validation macro ROC-AUC: 0.847212; public competition score: 0.830.
The implementation source revision is recorded by the local bundle and its
notebook provenance. The final implementation commit accompanies delivery.

Frozen slot order (`rsna-knee-slot-attention-v1`):

1. `axial_fluid_sensitive`: axial and Fluid_Sensitive true.
2. `sagittal_fluid_sensitive`: sagittal and Fluid_Sensitive true.
3. `coronal_fluid_sensitive`: coronal and Fluid_Sensitive true.
4. `sagittal_non_fluid_sensitive`: sagittal and Fluid_Sensitive false.

One helper reads the indexed Anatomical_Plane and Fluid_Sensitive fields, strips
whitespace and case-folds; yes/true/1 and no/false/0 are recognized. Missing or
unrecognized values are unmatched. Fat_Suppression is unused. Membership is
exclusive and independent of file names. Unmatched series are counted and
excluded from candidate aggregation; there is no other slot.

The unchanged ResNet-18 penultimate features are averaged over valid anchors,
then equally over series sharing a slot. For target t and available slot s:

```
a[s,t] = Linear(512,12)(h[s])[t]
l[s,t] = existing_shared_classifier(h[s])[t]
w[s,t] = exp(a[s,t]) / sum_available_slots exp(a[:,t])
study_logit[t] = sum_available_slots w[s,t] * l[s,t]
```

Unavailable slots are masked before softmax and receive exactly zero weight.
Each target has its own normalization; one available slot receives weight one.
Zero available slots use the existing training-label prevalence priors, with
no all-series mean substitution. Such training studies do not contribute an
optimizer update; validation retains them, and hidden submission records an
explicit fallback reason. All cases are counted in public-safe aggregate evidence.
The checkpoint stores the prior logits, validates its slot identity and uses a
separate schema from #31. The #31 mean path and shared head remain selectable.

## Fixed representation and evaluation

The candidate retains eight full-span deterministic anchors, same-slice RGB,
series 1st/99th percentile normalization, physical/DICOM ordering, centered
130 mm crop, 224 px output, ImageNet normalization and exact offline
ResNet18_Weights.IMAGENET1K_V1 initialization. No 2.5D or clipped-window path is
active. The same three-epoch AdamW recipe, batch size 16, mixed precision,
learning rate/weight decay 0.0001, seed policy, label masks, gold precedence,
4,407-study cohort and 3,957/450 PatientID-disjoint partition are enforced.
Validation membership digest:
`8b1240fc2456779ba64fa115b23956adb131fb387f01c0f0424a933a2096ee88`.

Verified #31 validation predictions are reused when attached with matching
report provenance, prediction digest, target order, study/PatientID membership
and labels. Otherwise the exact retained checkpoint is evaluated; #31 is never
retrained. Paired bootstrap retains the existing seed and 10,000 replicates.
Keep requires macro delta >= 0.005 and bootstrap positive fraction >= 0.95,
with severe target regressions checked using the existing decision policy.

The training report records macro/per-target AUC and deltas, masked BCE,
bootstrap interval/positive fraction/valid and undefined counts, slot coverage,
co-occurrence patterns, multiplicities, unmatched series, zero-slot studies,
per-target/slot mean and median attention, reload tolerance, runtime, throughput,
GPU memory and checkpoint size. Attention summaries include all validation
studies with unavailable slots contributing zero, and expose no private IDs.
All actual cohort statistics and performance results remain pending remote training.

The only added learned component is a 512-to-12 linear attention projection
(6,156 parameters). Encoder calls and input sizes remain the same for usable
training studies; zero-slot training studies are skipped. Actual runtime and
memory must be measured remotely rather than assumed equal.

## Execution boundary

Local preview bundle (GPU disabled):

```bash
uv run --extra training python scripts/kaggle.py build slot-attention
```

After reviewing the committed source, the user can build the GPU-enabled bundle
and explicitly start the one training experiment:

```bash
uv run --extra training python scripts/kaggle.py build slot-attention --approve-gpu
uv run --extra training python scripts/kaggle.py push build/kaggle/slot-attention --approve-remote --approve-gpu
```

These commands are provided for the user; Codex did not execute the push.
The notebook performs training and paired validation only. Hidden inference
and competition submission require separate authorization after a keep result.
