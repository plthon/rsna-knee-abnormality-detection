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

Smoke version 1 showed that Kaggle's mounted model filesystem omits the hidden
repository metadata file `.gitattributes`, even though the model archive lists
the 1,519-byte file and its SHA-256 matches the pinned upstream revision. The
contract records that source-only file under `mount_omitted_source_files` and
excludes it from the runtime-mounted inventory. All inference, tokenizer,
configuration, license, and visible documentation files remain hash-verified.

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

Smoke version 1 failed after 43.9 seconds during pre-inference asset validation
because it required the unmounted `.gitattributes` file. It failed before
`train.csv` or any report was read. A corrected local bundle is pending renewed
GPU-smoke approval. The rendered smoke bundle is
private, internet-disabled, attaches the frozen public model and immutable #20
kernel output, and uses a Tesla T4 with Python 3.12.13, PyTorch 2.10.0+cu128,
Transformers 5.0.0, and CUDA 12.8. Report processing requests exactly
`StudyInstanceUID` and `Report`; organizer-gold values are never parsed. The
missingness-only boundary selects 32 non-gold smoke reports by the frozen salted
SHA-256 order.

Full-run completeness, runtime, throughput, peak memory, checkpoint evidence,
parse failures, and truncation counts remain pending the approved executions.

## LLM label distribution

Pending the approved full extraction over all 4,407 reports. No coverage or
correctness claim is made before aggregate evidence exists.

## Rule-vs-LLM comparison

Pending the approved full extraction. The private join will use only
`StudyInstanceUID + target` against #20's hash-verified private
`report-labels-v1.csv`; organizer-gold values are excluded. Agreement will be
reported as comparison evidence, not correctness.

The fixed 64-report non-gold stability rerun is also pending. A structured
output agreement rate below 99% will stop the roadmap as `not-ready`.

## Readiness decision

`not-ready` — local implementation and contract evidence are complete, but the
mandatory offline GPU smoke, full extraction, aggregate comparison, and 64-report
stability evidence have not yet run. No weak label is activated for training.
