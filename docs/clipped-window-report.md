# Clipped-window three-anchor 2.5D — issue #33

Status: implemented locally; awaiting user-run GPU training. No Kaggle remote
execution, hidden inference, or submission has occurred for #33. Validation
results and the retain/reject decision remain pending. #31 remains the retained
control; #32 remains a completed negative isolated-adjacency experiment.

The retained #31 control is commit
`e85e24ebf0b78c6c73e63465d08c6b3dfa098141`, checkpoint SHA-256
`55e9da7854d284a91c2569e25c799281d09301bae61dc5adcecebbd17209f013`,
validation macro ROC-AUC `0.847212`, public score `0.830`.

The [public discussion cited by #33](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735154)
motivates clipping stack ends and using three adjacent triplets. The exact
competitor clip fraction was not disclosed. Our frozen 10%–90% window is an
operationalization, not an exact reproduction of a private configuration.
Any gain belongs to the entire slice-layout intervention, not adjacency alone.

## Frozen contract

For an already physically/DICOM ordered stack of length `n >= 3`:

```text
left  = max(1, floor((n - 1) / 10))
right = min(n - 2, floor(9 * (n - 1) / 10))
centers = deduplicate_in_order([left, floor((left + right) / 2), right])
triplet(i) = [max(i - 1, 0), i, min(i + 1, n - 1)]
```

Integer arithmetic implements floor exactly. The window is inclusive. The
inward endpoint constraint reserves real previous/next neighbors. For example,
`n=20` selects `[1,9,17]`; `n=21` selects `[2,10,18]`. No interpolation,
random selection, target-dependent positioning, or farther-neighbor search occurs.

Series with `n>=5` have exactly three centers. For `n=3`, use `[1]`; for `n=4`,
use `[1,2]`. For `n=2`, use centers `[0,1]` and triplets `[0,0,1]`, `[0,1,1]`;
for `n=1`, use `[0,0,0]` only where the existing index admits the series. Empty
series still fail. Reduced series retain three tensor slots with unused slots
masked, so padding never reaches the encoder. The short-series counter means
`n<5`; duplicate counts describe removed linspace centers for `n>=3`.

Percentiles (1st/99th) use only usable selected center anchors, then freeze for
all channel sources. This intentionally changes the percentile population from
#31's eight centers to #33's three centers; it never uses all nine sources.
A failed center is masked under the retained pipeline semantics. A failed
neighbor uses the processed center for that side; side and decode failure reason
are counted. Each decoded source uses its own PixelSpacing for the unchanged
centered 130 mm crop, deterministic padding/fallback, and 224×224 resize.

Preprocessing emits exact center indices, triplets, and window bounds. Reports
retain a public-safe geometry frequency histogram keyed by stack length, window,
centers, and triplets, plus short-series, duplicate, clamp, substitution, unique
decode, encoder-input, and crop/fallback counts. No study identifiers appear in
that histogram. The checkpoint schema is `rsna-knee-clipped-window-2-5d-v1`;
metadata freezes layout, window, anchor/triplet/failure policies, normalization
population, crop/FOV, representation identity, and target order. #31/#32 schemas
and configurations remain available and cannot be silently substituted.

## Fixed training and evaluation

ResNet-18 and exact offline `ResNet18_Weights.IMAGENET1K_V1`, ImageNet channel
normalization, ordinary anchor → series → study means, masked BCE, gold precedence,
the 4,407-study cohort, and all optimizer/seed/checkpoint semantics remain fixed.
Training uses three AdamW epochs, LR/weight decay `0.0001`, batch size 16, and
the retained mixed precision policy.

The split stays 3,957 train / 450 validation, PatientID-disjoint, with validation
membership digest
`8b1240fc2456779ba64fa115b23956adb131fb387f01c0f0424a933a2096ee88`.
The runner verifies #31's checkpoint provenance and regenerates its predictions
with eight full-span same-slice anchors before identifier-aligned paired evaluation.
#32 is historical context, never the comparison control.

The existing paired evaluation emits all 12 AUCs/deltas, macro AUC/delta, masked
BCE, non-decreasing target count, 10,000-replicate seeded bootstrap interval,
positive fraction and valid/undefined counts. It also emits training runtime,
validation decode/forward time, throughput, peak GPU memory, checkpoint size,
and reload agreement (absolute tolerance `1e-5`). These measured results are
**pending remote training**, not estimated validation evidence.

Retain only at macro delta `>=0.005`, bootstrap-positive fraction `>=0.95`, and
no target regression worse than `-0.02`. Eight non-decreasing targets is a
diagnostic preference. On success, a separate experiment may test 336 px with
this layout fixed. On failure, retain #31 and consider plane/sequence-aware
aggregation; do not tune clip percentages or anchor count.

Expected normal per-series work is three encoder inputs (37.5% of #31/#32's
eight) and nine requested source positions, with overlapping positions decoded
once. #31 normally decodes eight sources; #32 can request up to 24. Actual
runtime and memory must be measured, not inferred from these counts.

## Local verification and execution boundary

Local tests cover sampling/short series, center-only normalization, failed
neighbors/centers, per-source crop/fallback, physical ordering, three-input
encoding, unchanged recipe, schema/target protection, prediction reload, live
synthetic hidden inference, and local bundle/approval guards. Run the required
quality gates with the optional training extra installed so model tests execute:

```bash
uv sync --locked
uv sync --extra training --locked
uv run ruff check .
uv run ruff format --check .
uv run --extra training pytest
uv run --extra training mypy src tests scripts notebooks
uv build
git diff --check
```

The local unapproved bundle uses GPU disabled. After reviewing the exact committed
SHA and passing gates, the user can execute the following from the repository:

```bash
uv run python scripts/kaggle.py build clipped-window --approve-gpu
uv run python scripts/kaggle.py push build/kaggle/clipped-window --approve-remote --approve-gpu
```

The approved build requires a clean committed revision. The notebook trains and
validates only, using the retained #31 kernel output. It verifies and publishes
public-safe summary artifacts; paired study-level predictions stay inside the
artifact directory. Hidden inference requires a passing keep rule and separate
approval. Competition submission requires another separate approval. Neither is
included in these commands.
