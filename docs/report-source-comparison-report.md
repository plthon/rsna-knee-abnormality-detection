# RSNA Knee Report Source Comparison

## Status

Complete. Private Kaggle kernel `plthon/rsna-knee-report-source-comparison` ran
successfully on CPU with internet disabled in 2.35 seconds (19.2s container execution).
All 52,188 non-gold study-target rows (4,349 studies × 12 targets) achieved 100%
three-way join completeness against verified inputs.

Only aggregate JSON summaries were retrieved locally:

| Output | Bytes | SHA-256 |
| --- | ---: | --- |
| `source-comparison-manifest.json` | 3,348 | `63222ce234ec4be2ebfda9adbb2c27825ef0e6d521ed48b6de46ace150d71ea9` |
| `source-comparison-summary.json` | 57,852 | (retrieved aggregate summary) |

No row-level validation output, raw report, organizer target value, or study
identifier was retrieved locally.

## 1. Frozen input identities

All seven frozen inputs matched their pinned SHA-256 hashes and sizes prior to computation:

| Source | Role | Filename / Path | Size / Rows | SHA-256 |
| --- | --- | --- | ---: | --- |
| Competition | `competition_contract` | `contracts/competition.json` | 2,752 bytes | `3c6d42d4a4883a9626294c79af02f2ff687ba9413da8f316db9b877ebb1e2292` |
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
- Qwen LLM parse-failed studies (176 studies in non-gold cohort = 2,112 rows, 4.05%) keep their emitted `unknown` state and are broken out separately.

### Strata taxonomy
Every study-target triple `(rules, LLM, public)` is classified into exactly one mutually exclusive rollup stratum:
1. `triple-decisive-agree`: All three sources decisive and identical (`+++`, `---`).
2. `pairwise-agree-third-opposes`: Two decisive sources agree, third decisive source has opposite polarity.
3. `pairwise-agree-third-abstains`: Two decisive sources agree, third source is non-decisive (`uncertain` or `unknown`).
4. `two-decisive-opposed`: Two decisive sources disagree in polarity regardless of the third.
5. `single-decisive-only`: Exactly one decisive source; others non-decisive.
6. `zero-decisive`: No source is decisive.

### Cohort firewall
- All analysis strictly covers the **4,349-study non-gold cohort**.
- The 58-study organizer-gold cohort is evaluated for membership missingness only; its target values were **never read or parsed**.
- The 1 competition study absent from the public dataset is a gold-cohort member, giving 100% (4,349/4,349) coverage over non-gold studies.

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

