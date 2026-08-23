# Public label dataset audit

## Status

Phase A is complete. The private CPU audit bundle is built locally, and this
report stops at the mandatory boundary before Kaggle push. Coverage alignment
against competition study IDs and structural comparison with the private #20
artifact require the approved remote audit and are not inferred here.

## Frozen dataset identity

- Title: `RSNA Knee LLM-read report labels`
- Owner/slug: `pilkwang/rsna-knee-llm-labels`
- Dataset ID: `11529209`
- Version: `1`
- URL: <https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels/versions/1>
- License: `CC0-1.0`

The exact-title Kaggle search returned one match. The cited public notebook
`pilkwang/rsna-knee-baseline-v1` attaches that owner/slug, and Kaggle reports no
version 2. The contract freezes version 1 by file size and SHA-256 so a later
latest-version change will fail closed.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `api_labeler.py` | 12,976 | `045f0f858b20ee477869c999c4bc9b484154cb5ddfd507b79e7c732632abc52e` |
| `report_labels_v2.csv` | 1,004,839 | `6f704a7bdb2f894cc49445b19ba7c4378c3f548d3449e00361e10044bee40920` |

Kaggle metadata describes the dataset as twelve per-study scores with
per-target confidence and verdict fields derived from reports in their original
language. The exact subtitle and description text, their SHA-256 digests, and
the metadata-snapshot digest are frozen in the contract. The metadata names no
update frequency and contains no separate README.

## Local schema and semantics evidence

The label file has 4,406 rows, 4,406 unique nonblank study identifiers, and 37
columns. It is wide format: `StudyInstanceUID`, followed in exact competition
target order by three columns per target:

- the target score;
- `{target}__conf`; and
- `{target}__verdict`.

All twelve target scores use the observed domain
`{0.08, 0.28, 0.68, 0.82, 0.94}`. Confidence uses
`{0.05, 0.85, 0.95}`, and verdict uses `{YES, NO, UNK}`. There are no blank
label, confidence, or verdict cells. Dataset documentation says the scores are
rank-usable and are not calibrated probabilities. The verdict is therefore the
responsible field for mapping to `positive`, `negative`, or `unknown`.

The file cannot cover all 4,407 competition studies because it contains only
4,406 unique studies. Which competition study is absent, whether it belongs to
the 58-gold or 4,349 non-gold cohort, and whether any public identifier is extra
remain remote-audit questions. No organizer target value was read locally.

## Provenance

Provenance is classified `partial`.

Dataset metadata documents constrained twelve-finding verdict/severity output,
multilingual handling, and score semantics, but calls the generator only a
language model. It does not name the family, full prompt, immutable model/API
version, dependency version, or invocation that produced the CSV. The included
source suggests an Anthropic client and a default model string of
`claude-opus-5`; source defaults are not proof of the run that produced the
labels and do not upgrade the metadata-only provenance classification.

The included source and cited notebook disclose comparison with annotated
studies when choosing or tuning report readers. They do not freeze exactly how
the 58 organizer annotations affected the emitted public labels. Any later
evaluation must isolate this influence.

## Caveats and risks

- One study is absent by cardinality, with exact cohort alignment pending.
- Scores are ordinal/rank-oriented, not calibrated probabilities.
- `UNK` is absence of a definite report verdict, not a negative label.
- Exact generator runtime and model version are not reproducible from the
  dataset alone.
- Organizer annotations influenced reader evaluation or selection in an
  incompletely documented way.
- Agreement with #20 is not evidence that either report-derived source is
  clinically correct.

## Comparison with #20

The local scaffold freezes `report-labels-v1.csv` at 12,409,688 bytes and
SHA-256
`6651f60b00e5bec292e8c98e7fa83c75e8aa5af5931414a9a68c8ce10a52d527`.
The file remains private on Kaggle. The audit notebook will compare aggregate
per-target counts only, exclude #20 `unknown` and `uncertain` rows from direct
agreement, and report both conflict directions without claiming either source
is correct.

No comparison rate is reported before that notebook runs.

## Recommended next action

After explicit approval for one private CPU Kaggle execution, push the locally
built `public-label-audit` bundle. Retrieve only
`public-label-audit-manifest.json`,
`public-label-comparison-summary.json`, and
`public-label-audit-summary.json`; then replace this stop-point report with the
aggregate results.

## Final decision

Pending the approved remote audit. No final category is claimed at the
mandatory stop point.
