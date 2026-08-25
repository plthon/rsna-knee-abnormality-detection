# RSNA Knee Report Source Comparison

## Status

Complete. Private Kaggle kernel `plthon/rsna-knee-report-source-comparison` ran
successfully on CPU with internet disabled in 4.96 seconds (37.4s container execution).
All 52,188 non-gold study-target rows (4,349 studies × 12 targets) achieved 100%
three-way join completeness against verified inputs.

Only aggregate JSON summaries were retrieved locally:

| Output | Bytes | SHA-256 |
| --- | ---: | --- |
| `source-comparison-manifest.json` | 3,547 | `63222ce234ec4be2ebfda9adbb2c27825ef0e6d521ed48b6de46ace150d71ea9` |
| `source-comparison-summary.json` | 68,743 | (retrieved aggregate summary) |

No row-level validation output, raw report, organizer target value, or study
identifier was retrieved locally.

## 1. Frozen input identities

All eight frozen inputs matched their pinned SHA-256 hashes and sizes prior to computation:

| Source | Role | Filename / Path | Size / Rows | SHA-256 |
| --- | --- | --- | ---: | --- |
| Competition | `competition_contract` | `contracts/competition.json` | 28,526 bytes | `3c6d42d4a4883a9626294c79af02f2ff687ba9413da8f316db9b877ebb1e2292` |
| #20 Rules | `rule_extractor_contract` | `contracts/report-extractor-v1.json` | 15,896 bytes | `901e8a53173fef0d85e12ed93cbea05dd5b8cfc900ef111e0125cb751634156a` |
| #20 Rules | `rule_labels` | `report-labels-v1.csv` | 52,884 rows (12,409,688 bytes) | `6651f60b00e5bec292e8c98e7fa83c75e8aa5af5931414a9a68c8ce10a52d527` |
| #23 LLM | `llm_extractor_contract` | `contracts/report-llm-extractor-v1.json` | 4,364 bytes | `c428d1b44bb6c87f84e7423feeaefa6c635a5ba1cb5dd277468128d7df0bbe51` |
| #23 LLM | `llm_labels` | `report-llm-labels-v1.csv` | 52,884 rows (17,063,200 bytes) | `d78c7f1d27437f6d2870e48fbd730a57c60d34380720cd4e0e07811a699db52a` |
| #23 LLM | `llm_study_status` | `report-llm-study-status-v1.csv` | 4,407 rows (1,079,236 bytes) | `151d0554cbd7622028df1bd58a4e2c37d18796c5860502cf3eb0917ec6baf923` |
| #24 Public | `public_dataset_contract` | `contracts/public-report-label-dataset-v1.json` | 5,901 bytes | `3b607558a5bea66d2d10dd3884bd6fa9e2ac81b4617c6ea372d7feb0c00068e9` |
| #24 Public | `public_labels` | `report_labels_v2.csv` | 4,406 rows (1,004,839 bytes) | `6f704a7bdb2f894cc49445b19ba7c4378c3f548d3449e00361e10044bee40920` |

## 2. Method: normalization, strata taxonomy, cohort firewall

### State normalization
- All sources are mapped to a four-symbol vocabulary: `positive | negative | uncertain | unknown`.
- Public `YES/NO/UNK` verdicts map to `positive/negative/unknown`. Continuous scores and confidences are recorded descriptively only and play no role in agreement or strata logic.
- Qwen LLM parse-failed studies (176 studies in non-gold cohort = 2,112 rows, 4.05%) keep their emitted `unknown` state and are asserted to match between row-level labels and study-status tables.

### Strata taxonomy
Every study-target triple `(rules, LLM, public)` is classified into exactly one mutually exclusive rollup stratum:
1. `triple-decisive-agree`: All three sources decisive and identical (`+++`, `---`).
2. `pairwise-agree-third-opposes`: Two decisive sources agree, third decisive source has opposite polarity.
3. `pairwise-agree-third-abstains`: Two decisive sources agree, third source is non-decisive (`uncertain` or `unknown`).
4. `two-decisive-opposed`: Two decisive sources disagree in polarity regardless of the third.
5. `single-decisive-only`: Exactly one decisive source; others non-decisive.
6. `zero-decisive`: No source is decisive.

### Cohort firewall & completeness enforcement
- All analysis strictly covers the **4,349-study non-gold cohort**.
- The 58-study organizer-gold cohort is evaluated for membership missingness only; its target values were **never read or parsed**.
- The 1 competition study absent from the public dataset is a gold-cohort member, giving 100% (4,349/4,349) coverage over non-gold studies.
- Join completeness is strictly enforced: any drop below 52,188 complete triples fails closed.

## 3. Triple and rollup results per target

Over the 4,349 non-gold studies (52,188 rows):

| Target | Triple Decisive Agree | Pair Agree / 3rd Opposes | Pair Agree / 3rd Abstains | 2 Decisive Opposed | Single Decisive Only | Zero Decisive | Any Uncertain Crosscut |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACL | 335 | 81 | 1,026 | 427 | 2,134 | 346 | 24 |
| Baker's | 261 | 37 | 838 | 124 | 1,266 | 1,823 | 4 |
| Contusion | 92 | 11 | 323 | 54 | 2,971 | 898 | 18 |
| Effusion | 775 | 112 | 1,365 | 182 | 1,529 | 386 | 7 |
| Fracture | 46 | 3 | 596 | 27 | 1,230 | 2,447 | 13 |
| Lateral Meniscus | 214 | 61 | 1,226 | 369 | 2,094 | 385 | 25 |
| Lateral OA | 0 | 0 | 259 | 27 | 2,664 | 1,399 | 0 |
| MCL | 60 | 124 | 772 | 630 | 2,354 | 409 | 85 |
| Medial Meniscus | 474 | 40 | 1,186 | 190 | 2,230 | 229 | 60 |
| Medial OA | 0 | 0 | 636 | 45 | 2,608 | 1,060 | 0 |
| PF OA | 16 | 2 | 486 | 111 | 2,967 | 767 | 0 |
| Synovitis | 212 | 6 | 209 | 26 | 283 | 3,613 | 4 |

### Observed triple state frequency tables (`rules|llm|public`)

<details><summary><b>ACL (25 observed combinations)</b></summary>

| Triple State (`rules\|llm\|public`) | Count | Percentage |
| --- | ---: | ---: |
| `negative|negative|negative` | 22 | 0.51% |
| `negative|negative|positive` | 3 | 0.07% |
| `negative|positive|negative` | 6 | 0.14% |
| `negative|unknown|negative` | 669 | 15.38% |
| `negative|unknown|positive` | 2 | 0.05% |
| `positive|negative|negative` | 6 | 0.14% |
| `positive|negative|positive` | 59 | 1.36% |
| `positive|positive|negative` | 7 | 0.16% |
| `positive|positive|positive` | 313 | 7.20% |
| `positive|positive|unknown` | 1 | 0.02% |
| `positive|unknown|negative` | 176 | 4.05% |
| `positive|unknown|positive` | 56 | 1.29% |
| `uncertain|negative|negative` | 3 | 0.07% |
| `uncertain|negative|positive` | 2 | 0.05% |
| `uncertain|positive|positive` | 11 | 0.25% |
| `uncertain|unknown|negative` | 8 | 0.18% |
| `unknown|negative|negative` | 101 | 2.32% |
| `unknown|negative|positive` | 206 | 4.74% |
| `unknown|negative|unknown` | 3 | 0.07% |
| `unknown|positive|negative` | 41 | 0.94% |
| `unknown|positive|positive` | 185 | 4.25% |
| `unknown|positive|unknown` | 1 | 0.02% |
| `unknown|unknown|negative` | 1,765 | 40.58% |
| `unknown|unknown|positive` | 357 | 8.21% |
| `unknown|unknown|unknown` | 346 | 7.96% |

</details>

<details><summary><b>Baker's (21 observed combinations)</b></summary>

| Triple State (`rules\|llm\|public`) | Count | Percentage |
| --- | ---: | ---: |
| `negative|negative|negative` | 12 | 0.28% |
| `negative|positive|negative` | 10 | 0.23% |
| `negative|positive|positive` | 1 | 0.02% |
| `negative|unknown|negative` | 220 | 5.06% |
| `positive|negative|positive` | 26 | 0.60% |
| `positive|positive|positive` | 249 | 5.73% |
| `positive|unknown|negative` | 3 | 0.07% |
| `positive|unknown|positive` | 65 | 1.49% |
| `uncertain|negative|negative` | 1 | 0.02% |
| `uncertain|positive|positive` | 1 | 0.02% |
| `uncertain|unknown|positive` | 1 | 0.02% |
| `unknown|negative|negative` | 78 | 1.79% |
| `unknown|negative|positive` | 52 | 1.20% |
| `unknown|negative|unknown` | 60 | 1.38% |
| `unknown|positive|negative` | 69 | 1.59% |
| `unknown|positive|positive` | 473 | 10.88% |
| `unknown|positive|unknown` | 111 | 2.55% |
| `unknown|uncertain|negative` | 1 | 0.02% |
| `unknown|unknown|negative` | 863 | 19.84% |
| `unknown|unknown|positive` | 230 | 5.29% |
| `unknown|unknown|unknown` | 1,823 | 41.92% |

</details>

<details><summary><b>Contusion (25 observed combinations)</b></summary>

| Triple State (`rules\|llm\|public`) | Count | Percentage |
| --- | ---: | ---: |
| `negative|negative|negative` | 5 | 0.11% |
| `negative|positive|positive` | 1 | 0.02% |
| `negative|unknown|negative` | 145 | 3.33% |
| `positive|negative|negative` | 1 | 0.02% |
| `positive|negative|positive` | 7 | 0.16% |
| `positive|positive|negative` | 2 | 0.05% |
| `positive|positive|positive` | 87 | 2.00% |
| `positive|unknown|negative` | 6 | 0.14% |
| `positive|unknown|positive` | 39 | 0.90% |
| `positive|unknown|unknown` | 1 | 0.02% |
| `uncertain|negative|negative` | 1 | 0.02% |
| `uncertain|negative|positive` | 1 | 0.02% |
| `uncertain|positive|negative` | 3 | 0.07% |
| `uncertain|positive|positive` | 6 | 0.14% |
| `uncertain|unknown|negative` | 5 | 0.11% |
| `uncertain|unknown|positive` | 2 | 0.05% |
| `unknown|negative|negative` | 8 | 0.18% |
| `unknown|negative|positive` | 33 | 0.76% |
| `unknown|negative|unknown` | 4 | 0.09% |
| `unknown|positive|negative` | 11 | 0.25% |
| `unknown|positive|positive` | 124 | 2.85% |
| `unknown|positive|unknown` | 3 | 0.07% |
| `unknown|unknown|negative` | 1,995 | 45.87% |
| `unknown|unknown|positive` | 961 | 22.10% |
| `unknown|unknown|unknown` | 898 | 20.65% |

</details>

<details><summary><b>Effusion (24 observed combinations)</b></summary>

| Triple State (`rules\|llm\|public`) | Count | Percentage |
| --- | ---: | ---: |
| `negative|negative|negative` | 84 | 1.93% |
| `negative|positive|negative` | 13 | 0.30% |
| `negative|unknown|negative` | 302 | 6.94% |
| `negative|unknown|positive` | 2 | 0.05% |
| `positive|negative|negative` | 1 | 0.02% |
| `positive|negative|positive` | 93 | 2.14% |
| `positive|positive|negative` | 5 | 0.11% |
| `positive|positive|positive` | 691 | 15.89% |
| `positive|unknown|negative` | 5 | 0.11% |
| `positive|unknown|positive` | 145 | 3.33% |
| `uncertain|negative|negative` | 1 | 0.02% |
| `uncertain|positive|negative` | 1 | 0.02% |
| `uncertain|positive|positive` | 2 | 0.05% |
| `uncertain|unknown|negative` | 1 | 0.02% |
| `uncertain|unknown|positive` | 2 | 0.05% |
| `unknown|negative|negative` | 41 | 0.94% |
| `unknown|negative|positive` | 119 | 2.74% |
| `unknown|negative|unknown` | 7 | 0.16% |
| `unknown|positive|negative` | 55 | 1.26% |
| `unknown|positive|positive` | 874 | 20.10% |
| `unknown|positive|unknown` | 39 | 0.90% |
| `unknown|unknown|negative` | 792 | 18.21% |
| `unknown|unknown|positive` | 688 | 15.82% |
| `unknown|unknown|unknown` | 386 | 8.88% |

</details>

<details><summary><b>Fracture (20 observed combinations)</b></summary>

| Triple State (`rules\|llm\|public`) | Count | Percentage |
| --- | ---: | ---: |
| `negative|negative|negative` | 2 | 0.05% |
| `negative|unknown|negative` | 488 | 11.22% |
| `positive|negative|positive` | 3 | 0.07% |
| `positive|positive|positive` | 44 | 1.01% |
| `positive|uncertain|positive` | 3 | 0.07% |
| `positive|unknown|negative` | 6 | 0.14% |
| `positive|unknown|positive` | 30 | 0.69% |
| `positive|unknown|unknown` | 1 | 0.02% |
| `uncertain|positive|positive` | 2 | 0.05% |
| `uncertain|unknown|negative` | 1 | 0.02% |
| `uncertain|unknown|positive` | 6 | 0.14% |
| `unknown|negative|negative` | 6 | 0.14% |
| `unknown|negative|positive` | 19 | 0.44% |
| `unknown|negative|unknown` | 12 | 0.28% |
| `unknown|positive|negative` | 2 | 0.05% |
| `unknown|positive|positive` | 67 | 1.54% |
| `unknown|uncertain|positive` | 1 | 0.02% |
| `unknown|unknown|negative` | 1,089 | 25.04% |
| `unknown|unknown|positive` | 120 | 2.76% |
| `unknown|unknown|unknown` | 2,447 | 56.27% |

</details>

<details><summary><b>Lateral Meniscus (28 observed combinations)</b></summary>

| Triple State (`rules\|llm\|public`) | Count | Percentage |
| --- | ---: | ---: |
| `negative|negative|negative` | 26 | 0.60% |
| `negative|positive|negative` | 3 | 0.07% |
| `negative|unknown|negative` | 597 | 13.73% |
| `negative|unknown|positive` | 4 | 0.09% |
| `positive|negative|negative` | 15 | 0.34% |
| `positive|negative|positive` | 40 | 0.92% |
| `positive|positive|negative` | 3 | 0.07% |
| `positive|positive|positive` | 188 | 4.32% |
| `positive|positive|unknown` | 2 | 0.05% |
| `positive|uncertain|positive` | 6 | 0.14% |
| `positive|unknown|negative` | 162 | 3.72% |
| `positive|unknown|positive` | 72 | 1.66% |
| `positive|unknown|unknown` | 3 | 0.07% |
| `uncertain|negative|positive` | 2 | 0.05% |
| `uncertain|positive|negative` | 1 | 0.02% |
| `uncertain|positive|positive` | 4 | 0.09% |
| `uncertain|unknown|negative` | 4 | 0.09% |
| `uncertain|unknown|positive` | 2 | 0.05% |
| `unknown|negative|negative` | 184 | 4.23% |
| `unknown|negative|positive` | 84 | 1.93% |
| `unknown|negative|unknown` | 23 | 0.53% |
| `unknown|positive|negative` | 116 | 2.67% |
| `unknown|positive|positive` | 361 | 8.30% |
| `unknown|positive|unknown` | 22 | 0.51% |
| `unknown|uncertain|positive` | 6 | 0.14% |
| `unknown|unknown|negative` | 1,712 | 39.37% |
| `unknown|unknown|positive` | 322 | 7.40% |
| `unknown|unknown|unknown` | 385 | 8.85% |

</details>

<details><summary><b>Lateral OA (9 observed combinations)</b></summary>

| Triple State (`rules\|llm\|public`) | Count | Percentage |
| --- | ---: | ---: |
| `unknown|negative|negative` | 6 | 0.14% |
| `unknown|negative|positive` | 23 | 0.53% |
| `unknown|negative|unknown` | 12 | 0.28% |
| `unknown|positive|negative` | 4 | 0.09% |
| `unknown|positive|positive` | 253 | 5.82% |
| `unknown|positive|unknown` | 24 | 0.55% |
| `unknown|unknown|negative` | 1,833 | 42.15% |
| `unknown|unknown|positive` | 795 | 18.28% |
| `unknown|unknown|unknown` | 1,399 | 32.17% |

</details>

<details><summary><b>MCL (28 observed combinations)</b></summary>

| Triple State (`rules\|llm\|public`) | Count | Percentage |
| --- | ---: | ---: |
| `negative|negative|negative` | 13 | 0.30% |
| `negative|negative|positive` | 1 | 0.02% |
| `negative|unknown|negative` | 457 | 10.51% |
| `negative|unknown|positive` | 2 | 0.05% |
| `positive|negative|negative` | 6 | 0.14% |
| `positive|negative|positive` | 116 | 2.67% |
| `positive|positive|negative` | 1 | 0.02% |
| `positive|positive|positive` | 47 | 1.08% |
| `positive|uncertain|positive` | 20 | 0.46% |
| `positive|unknown|negative` | 507 | 11.66% |
| `positive|unknown|positive` | 142 | 3.27% |
| `positive|unknown|unknown` | 1 | 0.02% |
| `uncertain|negative|negative` | 1 | 0.02% |
| `uncertain|negative|positive` | 2 | 0.05% |
| `uncertain|uncertain|positive` | 2 | 0.05% |
| `uncertain|unknown|negative` | 13 | 0.30% |
| `uncertain|unknown|positive` | 9 | 0.21% |
| `unknown|negative|negative` | 123 | 2.83% |
| `unknown|negative|positive` | 116 | 2.67% |
| `unknown|negative|unknown` | 14 | 0.32% |
| `unknown|positive|negative` | 3 | 0.07% |
| `unknown|positive|positive` | 29 | 0.67% |
| `unknown|uncertain|negative` | 1 | 0.02% |
| `unknown|uncertain|positive` | 36 | 0.83% |
| `unknown|uncertain|unknown` | 1 | 0.02% |
| `unknown|unknown|negative` | 2,051 | 47.16% |
| `unknown|unknown|positive` | 227 | 5.22% |
| `unknown|unknown|unknown` | 408 | 9.38% |

</details>

<details><summary><b>Medial Meniscus (29 observed combinations)</b></summary>

| Triple State (`rules\|llm\|public`) | Count | Percentage |
| --- | ---: | ---: |
| `negative|negative|negative` | 12 | 0.28% |
| `negative|positive|negative` | 4 | 0.09% |
| `negative|unknown|negative` | 321 | 7.38% |
| `negative|unknown|positive` | 6 | 0.14% |
| `positive|negative|negative` | 6 | 0.14% |
| `positive|negative|positive` | 27 | 0.62% |
| `positive|positive|negative` | 3 | 0.07% |
| `positive|positive|positive` | 462 | 10.62% |
| `positive|uncertain|positive` | 5 | 0.11% |
| `positive|unknown|negative` | 65 | 1.49% |
| `positive|unknown|positive` | 146 | 3.36% |
| `positive|unknown|unknown` | 1 | 0.02% |
| `uncertain|negative|negative` | 1 | 0.02% |
| `uncertain|negative|positive` | 2 | 0.05% |
| `uncertain|positive|positive` | 4 | 0.09% |
| `uncertain|uncertain|negative` | 1 | 0.02% |
| `uncertain|unknown|negative` | 6 | 0.14% |
| `uncertain|unknown|positive` | 4 | 0.09% |
| `unknown|negative|negative` | 41 | 0.94% |
| `unknown|negative|positive` | 104 | 2.39% |
| `unknown|negative|unknown` | 1 | 0.02% |
| `unknown|positive|negative` | 13 | 0.30% |
| `unknown|positive|positive` | 668 | 15.36% |
| `unknown|positive|unknown` | 17 | 0.39% |
| `unknown|uncertain|negative` | 1 | 0.02% |
| `unknown|uncertain|positive` | 36 | 0.83% |
| `unknown|unknown|negative` | 1,312 | 30.17% |
| `unknown|unknown|positive` | 851 | 19.57% |
| `unknown|unknown|unknown` | 229 | 5.27% |

</details>

<details><summary><b>Medial OA (9 observed combinations)</b></summary>

| Triple State (`rules\|llm\|public`) | Count | Percentage |
| --- | ---: | ---: |
| `unknown|negative|negative` | 3 | 0.07% |
| `unknown|negative|positive` | 32 | 0.74% |
| `unknown|negative|unknown` | 10 | 0.23% |
| `unknown|positive|negative` | 13 | 0.30% |
| `unknown|positive|positive` | 633 | 14.56% |
| `unknown|positive|unknown` | 41 | 0.94% |
| `unknown|unknown|negative` | 1,625 | 37.36% |
| `unknown|unknown|positive` | 932 | 21.43% |
| `unknown|unknown|unknown` | 1,060 | 24.37% |

</details>

<details><summary><b>PF OA (12 observed combinations)</b></summary>

| Triple State (`rules\|llm\|public`) | Count | Percentage |
| --- | ---: | ---: |
| `positive|negative|positive` | 2 | 0.05% |
| `positive|positive|positive` | 16 | 0.37% |
| `positive|unknown|positive` | 3 | 0.07% |
| `unknown|negative|negative` | 16 | 0.37% |
| `unknown|negative|positive` | 101 | 2.32% |
| `unknown|negative|unknown` | 13 | 0.30% |
| `unknown|positive|negative` | 10 | 0.23% |
| `unknown|positive|positive` | 467 | 10.74% |
| `unknown|positive|unknown` | 21 | 0.48% |
| `unknown|unknown|negative` | 1,555 | 35.76% |
| `unknown|unknown|positive` | 1,378 | 31.69% |
| `unknown|unknown|unknown` | 767 | 17.64% |

</details>

<details><summary><b>Synovitis (19 observed combinations)</b></summary>

| Triple State (`rules\|llm\|public`) | Count | Percentage |
| --- | ---: | ---: |
| `negative|negative|negative` | 2 | 0.05% |
| `negative|unknown|negative` | 10 | 0.23% |
| `positive|negative|positive` | 6 | 0.14% |
| `positive|positive|positive` | 210 | 4.83% |
| `positive|unknown|negative` | 1 | 0.02% |
| `positive|unknown|positive` | 33 | 0.76% |
| `positive|unknown|unknown` | 2 | 0.05% |
| `uncertain|positive|positive` | 2 | 0.05% |
| `uncertain|unknown|positive` | 1 | 0.02% |
| `unknown|negative|negative` | 4 | 0.09% |
| `unknown|negative|positive` | 23 | 0.53% |
| `unknown|negative|unknown` | 44 | 1.01% |
| `unknown|positive|negative` | 2 | 0.05% |
| `unknown|positive|positive` | 160 | 3.68% |
| `unknown|positive|unknown` | 12 | 0.28% |
| `unknown|uncertain|unknown` | 1 | 0.02% |
| `unknown|unknown|negative` | 95 | 2.18% |
| `unknown|unknown|positive` | 129 | 2.97% |
| `unknown|unknown|unknown` | 3,612 | 83.05% |

</details>

## 4. Pairwise reconciliation with #23 and #24

Pairwise joint-decisive agreement metrics computed on the 4,349 non-gold cohort reconcile with #23 and #24 published numbers (which used 4,407 and 4,406 rows respectively):

| Pair | Compared Rows | Agreement Count | Agreement Rate | Polarity Conflicts | First + / Second - | First - / Second + | Exact 4-State Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rules|llm` | 2,965 | 2,513 | 84.76% | 452 | 414 | 38 | 76.56% |
| `rules|public` | 7,883 | 6,874 | 87.20% | 1,009 | 987 | 22 | 40.49% |
| `llm|public` | 9,172 | 7,467 | 81.41% | 1,705 | 401 | 1,304 | 40.69% |

## 5. Target-specific deep dives

### MCL conflict disposition
Of the 514 non-gold public-negative / rule-positive conflicts:
- Qwen LLM calls **unknown** on **507** (98.64%) cases;
- Qwen LLM calls **negative** on **6** cases (agreeing with public);
- Qwen LLM calls **positive** on only **1** case (agreeing with rules).

Of the 3 public-positive / rule-negative conflicts:
- Qwen calls negative on 1 case and unknown on 2 cases.

### OA coverage expansion
Rule extraction provided zero coverage on Medial/Lateral OA and 21 positives on PF OA. The LLM and public sources provide substantial expansion:

| OA Target | Rule Decisive | LLM Decisive | Public Decisive | LLM∩Public Decisive | LLM+ & Public+ Overlap | LLM-Only Decisive | Public-Only Decisive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Medial OA | 0 | 732 | 3,238 | 681 | 633 | 51 | 2,557 |
| Lateral OA | 0 | 322 | 2,914 | 286 | 253 | 36 | 2,628 |
| PF OA | 21 | 646 | 3,548 | 612 | 483 | 34 | 2,936 |

### Qwen under-calling profile
Flagged targets where Qwen LLM decisive positives fall below 25% of another source:
- **MCL**: LLM emits 80 positives vs rules 840 (9.52%) and public 749 (10.68%) &rarr; **Severely under-called**.
- **Contusion**: LLM emits 237 positives vs public 1,261 (18.79%) &rarr; **Under-called** relative to public set.

### Parse-failure loss counterfactual
Among the 176 non-gold parse-failed studies (2,112 rows) where Qwen emitted `unknown`:
- Public labels provide decisive supervision on **1,636 rows** (77.46%);
- Rules provide decisive supervision on **431 rows** (20.41%);
- Either source provides decisive supervision on **1,636 rows**.

## 6. Review-quota feasibility evidence for #21 rewrite

Candidate counts and stratum allocations per `target × source × decisive_state` across the 4,349 non-gold studies:

| Target | Source | Decisive State | Available Candidates | Triple Decisive Agree | Contested Count | Feasibility (for 40-case quota) |
| --- | --- | --- | ---: | ---: | ---: | --- |
| ACL | rules | positive | 618 | 313 | 248 | Feasible (>=40) |
| ACL | rules | negative | 702 | 22 | 11 | Feasible (>=40) |
| ACL | llm | positive | 565 | 313 | 54 | Feasible (>=40) |
| ACL | llm | negative | 405 | 22 | 276 | Feasible (>=40) |
| ACL | public | positive | 1,194 | 313 | 272 | Feasible (>=40) |
| ACL | public | negative | 2,804 | 22 | 236 | Feasible (>=40) |
| Baker's | rules | positive | 343 | 249 | 29 | Feasible (>=40) |
| Baker's | rules | negative | 243 | 12 | 11 | Feasible (>=40) |
| Baker's | llm | positive | 914 | 249 | 80 | Feasible (>=40) |
| Baker's | llm | negative | 229 | 12 | 78 | Feasible (>=40) |
| Baker's | public | positive | 1,098 | 249 | 79 | Feasible (>=40) |
| Baker's | public | negative | 1,257 | 12 | 82 | Feasible (>=40) |
| Contusion | rules | positive | 143 | 87 | 16 | Feasible (>=40) |
| Contusion | rules | negative | 151 | 5 | 1 | Feasible (>=40) |
| Contusion | llm | positive | 237 | 87 | 17 | Feasible (>=40) |
| Contusion | llm | negative | 60 | 5 | 42 | Feasible (>=40) |
| Contusion | public | positive | 1,261 | 87 | 42 | Feasible (>=40) |
| Contusion | public | negative | 2,182 | 5 | 23 | Feasible (>=40) |
| Effusion | rules | positive | 940 | 691 | 104 | Feasible (>=40) |
| Effusion | rules | negative | 401 | 84 | 15 | Feasible (>=40) |
| Effusion | llm | positive | 1,680 | 691 | 74 | Feasible (>=40) |
| Effusion | llm | negative | 346 | 84 | 213 | Feasible (>=40) |
| Effusion | public | positive | 2,616 | 691 | 214 | Feasible (>=40) |
| Effusion | public | negative | 1,301 | 84 | 80 | Feasible (>=40) |
| Fracture | rules | positive | 87 | 44 | 9 | Feasible (>=40) |
| Fracture | rules | negative | 490 | 2 | 0 | Feasible (>=40) |
| Fracture | llm | positive | 115 | 44 | 2 | Feasible (>=40) |
| Fracture | llm | negative | 42 | 2 | 22 | Feasible (>=40) |
| Fracture | public | positive | 295 | 44 | 22 | Feasible (>=40) |
| Fracture | public | negative | 1,594 | 2 | 8 | Feasible (>=40) |
| Lateral Meniscus | rules | positive | 491 | 188 | 220 | Feasible (>=40) |
| Lateral Meniscus | rules | negative | 630 | 26 | 7 | Feasible (>=40) |
| Lateral Meniscus | llm | positive | 700 | 188 | 123 | Feasible (>=40) |
| Lateral Meniscus | llm | negative | 374 | 26 | 141 | Feasible (>=40) |
| Lateral Meniscus | public | positive | 1,091 | 188 | 130 | Feasible (>=40) |
| Lateral Meniscus | public | negative | 2,823 | 26 | 300 | Feasible (>=40) |
| Lateral OA | rules | positive | 0 | 0 | 0 | Scarce (0<40) |
| Lateral OA | rules | negative | 0 | 0 | 0 | Scarce (0<40) |
| Lateral OA | llm | positive | 281 | 0 | 4 | Feasible (>=40) |
| Lateral OA | llm | negative | 41 | 0 | 23 | Feasible (>=40) |
| Lateral OA | public | positive | 1,071 | 0 | 23 | Feasible (>=40) |
| Lateral OA | public | negative | 1,843 | 0 | 4 | Feasible (>=40) |
| MCL | rules | positive | 840 | 47 | 630 | Feasible (>=40) |
| MCL | rules | negative | 473 | 13 | 3 | Feasible (>=40) |
| MCL | llm | positive | 80 | 47 | 4 | Feasible (>=40) |
| MCL | llm | negative | 392 | 13 | 241 | Feasible (>=40) |
| MCL | public | positive | 749 | 47 | 237 | Feasible (>=40) |
| MCL | public | negative | 3,176 | 13 | 517 | Feasible (>=40) |
| Medial Meniscus | rules | positive | 715 | 462 | 101 | Feasible (>=40) |
| Medial Meniscus | rules | negative | 343 | 12 | 10 | Feasible (>=40) |
| Medial Meniscus | llm | positive | 1,171 | 462 | 20 | Feasible (>=40) |
| Medial Meniscus | llm | negative | 194 | 12 | 139 | Feasible (>=40) |
| Medial Meniscus | public | positive | 2,315 | 462 | 139 | Feasible (>=40) |
| Medial Meniscus | public | negative | 1,786 | 12 | 91 | Feasible (>=40) |
| Medial OA | rules | positive | 0 | 0 | 0 | Scarce (0<40) |
| Medial OA | rules | negative | 0 | 0 | 0 | Scarce (0<40) |
| Medial OA | llm | positive | 687 | 0 | 13 | Feasible (>=40) |
| Medial OA | llm | negative | 45 | 0 | 32 | Feasible (>=40) |
| Medial OA | public | positive | 1,597 | 0 | 32 | Feasible (>=40) |
| Medial OA | public | negative | 1,641 | 0 | 13 | Feasible (>=40) |
| PF OA | rules | positive | 21 | 16 | 2 | Scarce (21<40) |
| PF OA | rules | negative | 0 | 0 | 0 | Scarce (0<40) |
| PF OA | llm | positive | 514 | 16 | 10 | Feasible (>=40) |
| PF OA | llm | negative | 132 | 0 | 103 | Feasible (>=40) |
| PF OA | public | positive | 1,967 | 16 | 103 | Feasible (>=40) |
| PF OA | public | negative | 1,581 | 0 | 10 | Feasible (>=40) |
| Synovitis | rules | positive | 252 | 210 | 7 | Feasible (>=40) |
| Synovitis | rules | negative | 12 | 2 | 0 | Scarce (12<40) |
| Synovitis | llm | positive | 386 | 210 | 2 | Feasible (>=40) |
| Synovitis | llm | negative | 79 | 2 | 29 | Feasible (>=40) |
| Synovitis | public | positive | 564 | 210 | 29 | Feasible (>=40) |
| Synovitis | public | negative | 114 | 2 | 3 | Feasible (>=40) |

## 7. Explicit non-claims

- **No correctness inference**: Concordance across sources reflects common morphology or reporting patterns, not verified clinical ground truth.
- **No training label merge**: No labels have been blended, thresholded, or merged into training artifacts.
- **No extractor modification**: Extractor logic and models remain frozen.
- **No design precommitment for #21**: This report supplies empirical numbers only. Quota freezing, source activation, and human validation policy belong solely to #21.

