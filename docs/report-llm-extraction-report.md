# Frozen local-LLM report extraction

This public-safe report records the candidate extractor required by issue #23.
It contains no report text, generated text, study or patient identifier,
row-level label, or normalized report hash.

## Model and prompt contract

The frozen model is
[`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
at model and tokenizer revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`. It uses the Qwen Research
License, whose committed identity is `qwen-research` and whose use here is
limited to non-commercial research and evaluation. The exact license text,
runtime file inventory, sizes, and SHA-256 values are frozen in
[`contracts/report-llm-extractor-v1.json`](../contracts/report-llm-extractor-v1.json).

The exact revision is already available as public Kaggle Model
`qwen-lm/qwen2.5/transformers/3b-instruct/1`. Its complete file names and sizes
match the pinned upstream revision, including two safetensors files with
SHA-256 values `67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2`
and `a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d`.
No acquisition job, model download, or private model publication is required.
The offline notebook verifies every attached model and tokenizer file before
constructing the model.

Smoke versions 1 through 3 failed sequentially on `.gitattributes`, `LICENSE`,
and `config.json`, establishing that the legacy model root was absent rather
than individual files being filtered. The contract now resolves Kaggle's
structured model root first and retains the legacy root as a compatibility
fallback. A ranged probe of the public model archive independently verified
the pinned source metadata without downloading model weights. All 12 attached
model, tokenizer, configuration, license, and documentation files remain
runtime hash-verified before model loading.

Prompt version `rsna-knee-12target-fourstate-v1` has SHA-256
`172f2d83d05f15deb928b879d2401cd5fd76f414431b6e43af9c7c5afd9b2826`.
It defines all 12 verified targets and exactly four states: `positive`,
`negative`, `uncertain`, and `unknown`. No mention is `unknown`, never an
implicit negative. Historical or postoperative language alone is not a current
positive; hedging and unresolved contradictions are `uncertain`. The prompt
preserves compartment-specific OA, excludes ordinary degenerative marrow edema
from traumatic contusion, and counts explicit trace or small effusions.

Generation is greedy (`do_sample=false`, no temperature), FP16 without
quantization, batch size 8, and at most 256 new tokens. The frozen 32,768-token
model context reserves 32,512 input tokens. An over-budget report is truncated
only at token level, retaining fixed beginning and end regions and recording
`input_truncated=true`. Checkpoints are written every 50 completed studies and
at clean shutdown; resume is keyed by `StudyInstanceUID` and rejects duplicates
or changed report hashes. One retry may reuse the identical prompt and decoding
contract, and is recorded separately.

The public MIT implementation
[`kernels/extract_labels_v1/extract.py`](https://github.com/liamd-schneider/RSNA-Knee-Abnormality-Detection/blob/main/kernels/extract_labels_v1/extract.py)
was consulted as the requested feasibility reference. No source code or prompt
was copied. This implementation replaces its binary absent semantics,
all-negative parse fallback, arbitrary character truncation, organizer-gold
prompt tuning, and non-resumable loop with the issue #23 contract.

## Execution

Smoke versions 1 through 3 failed after 43.9, 44.3, and 47.2 seconds during
pre-inference asset validation because they used the absent legacy model root.
All three failed before `train.csv` or any report was read. Version 4 passed
model verification, read reports only inside the private Kaggle notebook, and
selected the frozen 32-report smoke set before failing after 106.4 seconds: its
rule-label lookup did not search Kaggle's nested kernel-output mount. It failed
before reading rule labels, constructing the model, running inference, or
writing artifacts. The corrected bundle reused the shared, tested nested
kernel-output resolver.

Smoke version 5 completed all 32 deterministic non-gold reports and wrote the
expected 384 study-target rows. It was private and internet-disabled, verified
all 12 frozen model files, and used a Tesla T4 with Python 3.12.13, PyTorch
2.10.0+cu128, Transformers 5.0.0, CUDA 12.8, FP16, and batch size 8. Runtime was
119.56 seconds, throughput was 0.2676 studies/second, and peak GPU memory was
7,584,029,696 bytes. Report processing requested exactly `StudyInstanceUID`
and `Report`; organizer-gold values were never parsed.

Checkpoint/resume was exercised from a 16-study partial checkpoint. The smoke
recorded one same-contract retry, one explicit parse failure (3.125%), and zero
input truncations. The private label and study-status artifacts remain on
Kaggle with SHA-256 values
`5f6d2aa1ae0ce35834d44e47bc7186967443446261c7259275a4585d50b4a30f` and
`ac2c478fb52fa02fb8e7a1886c8b4b729f1ba9a2b220c4c7fe81b73e755c33fa`.

Full-run completeness, runtime, throughput, peak memory, checkpoint evidence,
parse failures, and truncation counts remain pending the approved full extraction.

## LLM label distribution

Across the 32-report smoke, the aggregate state counts were 46 positive, 17
negative, one uncertain, and 320 unknown. Medial, lateral, and patellofemoral OA
coverage was 15.625%, 6.25%, and 18.75%, respectively. This small execution
sample validates the pipeline only; no label-distribution or correctness claim
is made before the full extraction.

## Rule-vs-LLM comparison

The private smoke join used only `StudyInstanceUID + target` against #20's
hash-verified private `report-labels-v1.csv`; organizer-gold values were
excluded. For the three OA targets, the rules had zero coverage in the smoke
sample while the LLM had 5 medial, 2 lateral, and 6 patellofemoral decisive
outputs. MCL had the sample's only two positive-vs-negative direct
disagreements. These are execution-scale comparison observations, not evidence
of correctness. Full-corpus comparison remains pending.

The fixed 64-report non-gold stability rerun is also pending. A structured
output agreement rate below 99% will stop the roadmap as `not-ready`.

## Readiness decision

`not-ready` — local implementation, contract evidence, and the mandatory offline
GPU smoke are complete, but the full extraction, full aggregate comparison, and
64-report stability evidence have not yet run. No weak label is activated for
training.
