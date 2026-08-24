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

The approved full extraction from clean Git commit
`3a2cbfa1a988722c759c491e4da93894941f0f23` completed all 4,407 reports and
exactly 52,884 ordered study-target rows. The Jupytext source and rendered
notebook SHA-256 values were
`92314652e5c7298e2aacf7873ed65e5227c1e8f3cbb7d2788ce1b656f141c051` and
`b8ffd96034fa3d8c3d6c28c49389df8c9ca10455bf5c12851698e7f239257608`.
The competition and LLM contract SHA-256 values were
`3c6d42d4a4883a9626294c79af02f2ff687ba9413da8f316db9b877ebb1e2292`
and `c428d1b44bb6c87f84e7423feeaefa6c635a5ba1cb5dd277468128d7df0bbe51`.

The full run used the same private, offline T4 runtime and verified the same 12
model files. Runtime was 6,354.81 seconds, throughput was 0.6935
studies/second, and peak GPU memory was 9,592,860,160 bytes. It wrote
checkpoints every 50 studies and at clean shutdown, reaching checkpoint
sequence 89 without needing a resume; the smoke had already exercised the
duplicate-safe resume path. The run recorded 188 same-contract retries, 180
final explicit parse failures (4.0844%), and zero input truncations.

The full private label and study-status artifacts remain on Kaggle. Their sizes
and SHA-256 values are 17,063,200 bytes and
`d78c7f1d27437f6d2870e48fbd730a57c60d34380720cd4e0e07811a699db52a`,
and 1,079,236 bytes and
`151d0554cbd7622028df1bd58a4e2c37d18796c5860502cf3eb0917ec6baf923`,
respectively.

## LLM label distribution

The full aggregate contains 7,500 positive, 2,375 negative, 124 uncertain, and
42,885 unknown states. Coverage is the fraction that is not `unknown`.

| Target | Positive | Negative | Uncertain | Unknown | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACL | 584 | 412 | 0 | 3,411 | 22.60% |
| MCL | 82 | 401 | 61 | 3,863 | 12.34% |
| Medial Meniscus | 1,189 | 199 | 44 | 2,975 | 32.49% |
| Lateral Meniscus | 718 | 381 | 13 | 3,295 | 25.23% |
| Medial OA | 697 | 46 | 0 | 3,664 | 16.86% |
| Lateral OA | 288 | 41 | 0 | 4,078 | 7.47% |
| PF OA | 525 | 133 | 0 | 3,749 | 14.93% |
| Effusion | 1,717 | 348 | 0 | 2,342 | 46.86% |
| Synovitis | 399 | 81 | 1 | 3,926 | 10.91% |
| Baker's | 928 | 231 | 1 | 3,247 | 26.32% |
| Contusion | 251 | 60 | 0 | 4,096 | 7.06% |
| Fracture | 122 | 42 | 4 | 4,239 | 3.81% |

The OA coverage values describe candidate-label availability, not correctness.

## Rule-vs-LLM comparison

The private full-corpus join used only `StudyInstanceUID + target` against
#20's hash-verified `report-labels-v1.csv`; organizer-gold values were excluded.

| Target | Exact agreement | Both + | Both - | Rule unknown / LLM decisive | Rule decisive / LLM unknown | +/- disagreement | Uncertain disagreement | LLM coverage | Rule coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACL | 2,850 (64.67%) | 333 | 25 | 546 | 911 | 76 | 24 | 22.60% | 31.06% |
| MCL | 2,786 (63.22%) | 50 | 14 | 287 | 1,121 | 129 | 84 | 12.34% | 30.88% |
| Medial Meniscus | 2,906 (65.94%) | 473 | 13 | 856 | 545 | 39 | 61 | 32.49% | 24.82% |
| Lateral Meniscus | 2,673 (60.65%) | 200 | 27 | 804 | 843 | 61 | 26 | 25.23% | 26.12% |
| Medial OA | 3,664 (83.14%) | 0 | 0 | 743 | 0 | 0 | 0 | 16.86% | 0.00% |
| Lateral OA | 4,078 (92.53%) | 0 | 0 | 329 | 0 | 0 | 0 | 7.47% | 0.00% |
| PF OA | 3,762 (85.36%) | 16 | 0 | 640 | 3 | 2 | 0 | 14.93% | 0.48% |
| Effusion | 2,679 (60.79%) | 713 | 85 | 1,155 | 458 | 107 | 8 | 46.86% | 31.11% |
| Synovitis | 4,099 (93.01%) | 218 | 2 | 252 | 46 | 6 | 4 | 10.91% | 6.24% |
| Baker's | 3,217 (73.00%) | 251 | 12 | 857 | 292 | 37 | 4 | 26.32% | 13.50% |
| Contusion | 3,998 (90.72%) | 97 | 5 | 188 | 193 | 9 | 19 | 7.06% | 7.33% |
| Fracture | 3,745 (84.98%) | 46 | 2 | 110 | 534 | 3 | 15 | 3.81% | 13.59% |

The LLM materially expands candidate coverage for medial, lateral, and
patellofemoral OA relative to the rule extractor's 0.00%, 0.00%, and 0.48%, but
agreement and coverage do not establish correctness. The comparison preserves
both sources for #21 rather than selecting between them here.

The fixed 64-report non-gold stability sample produced 64/64 exact structured
matches (100%). Every target had zero disagreements; both passes had two parse
failures, and parse-failure status was consistent for all 64 reports. This
passes the required 99% stability threshold.

## Readiness decision

`ready-for-weak-supervision-selection` — model identity, privacy boundaries,
offline GPU execution, full artifact completeness, aggregate comparison, and
the stability threshold all pass. This decision makes the frozen candidate
available to #21; it does not activate a weak label for training or claim label
correctness.
