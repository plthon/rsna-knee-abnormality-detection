# Deterministic report-label extraction

Verified 2026-08-23 for `rsna-knee-abnormality-detection`. The frozen
machine-readable rule inventory is
[`contracts/report-extractor-v1.json`](../contracts/report-extractor-v1.json),
SHA-256
`901e8a53173fef0d85e12ed93cbea05dd5b8cfc900ef111e0125cb751634156a`.
This report contains aggregate evidence only. The row-level candidate labels,
evidence spans, study identifiers, normalized report hashes, and report text
remain private Kaggle artifacts.

## Corpus findings

The audit read exactly `StudyInstanceUID` and `Report` from the competition
mount's `train.csv`. All 4,407 rows had unique study identifiers and nonempty
normalized reports. Normalized length ranged from 52 to 4,630 characters, with
mean 1,091.51 and p25/p50/p75/p95 values of 585/972/1,447/2,446.

The deterministic parser found at least one heading in 1,963 reports and no
heading in 2,444. Frequent headings included `findings` (791 reports),
`impression` (649), `technique` (602), `comparison` (454), `lateral meniscus`
(439), `medial meniscus` (438), `indication` (406), and `conclusion` (378).
The two meniscus headings are explicit target context in the frozen contract.
History, indication, technique, comparison, prior-imaging, and exam-type
sections are excluded from current-finding assertions.

Script-family counts are nonexclusive because a report can contain more than
one script: Latin appeared in 4,301 reports, Greek in 321, Cyrillic in 220,
Other in 309, and Common characters in all 4,407. The audit did not have a
deterministic language classifier, so language remains **unresolved** rather
than being inferred from script.

### Target terminology

Only aggregate terms occurring in at least five reports were exported. The
following table records the audit evidence that supported the deliberately
narrow concept inventory.

| Target | Audit-supported terminology and document frequency | Frozen interpretation |
| --- | --- | --- |
| ACL | `ACL` 1,348; `anterior cruciate ligament` 648 | Both forms are concepts; nearby `normal` and `intact` are explicit negative descriptors. |
| MCL | `MCL` 1,240; `medial collateral ligament` 691 | Both forms are concepts; nearby `normal` and `intact` are explicit negative descriptors. |
| Medial Meniscus | `medial meniscus` 1,278; heading 438 | Canonical phrase is the concept; `normal`, `intact`, `maybe`, and the target heading have explicit rules. |
| Lateral Meniscus | `lateral meniscus` 1,340; heading 439 | Canonical phrase is the concept; `normal`, `intact`, and the target heading have explicit rules. |
| Medial OA | No seeded phrase met the five-report privacy threshold | Only the canonical name and direct `medial osteoarthritis` expansion were retained; the run found no matches. |
| Lateral OA | No seeded phrase met the five-report privacy threshold | Only the canonical name and direct `lateral osteoarthritis` expansion were retained; the run found no matches. |
| PF OA | `patellofemoral osteoarthritis` 20 | Canonical `PF OA` and the direct patellofemoral expansion are concepts. |
| Effusion | `effusion` 1,571; `joint effusion` 835; `knee effusion` 502; `suprapatellar effusion` 16 | Optional anatomical modifiers are part of one concept rule; nearby `normal` is target-specific negative evidence. |
| Synovitis | `synovitis` 297 | Canonical term only. |
| Baker's | `Baker's` 757; `Baker's cyst` 754; `Baker cyst` 42 | Apostrophe and non-apostrophe cyst forms are one concept rule. |
| Contusion | `contusion` 337; `bone contusion` 264; `osseous` 23; `marrow` 6; `soft tissue` 7 | Unmodified and the four audit-supported modifiers are one concept rule. |
| Fracture | `fracture` 613; `acute fracture` 296 | Canonical term only; chronicity is preserved for assertion context rather than made a separate target. |

Corpus neighbors that did not clearly denote the verified target were left out.
This is especially consequential for medial and lateral OA: aggregate evidence
was insufficient, so every study remains `unknown` for those two targets.

### Assertion and context findings

The main explicit-negative cues were `no` (1,901 reports), `intact` (1,060),
`without` (623), `not` (586), and `absent` (7). Affirmative cues were `present`
(256), `seen` (234), `shows` (60), and `demonstrates` (29). Hedging included
`may` (68), `possible` (39), and `possibly` (14). Historical cues included
`prior` (344), `history of` (55), `previous` (44), and `remote` (16). No
postoperative candidate passed the five-report audit threshold.

The audit's intentionally broad contradiction screen flagged 63 effusion, 42
fracture, 24 Baker's cyst, 14 contusion, 12 medial-meniscus, 8
lateral-meniscus, 2 ACL, 2 MCL, and 1 synovitis reports; it flagged none for
the three OA targets. Common exported patterns included `no` with `present` for
effusion and `no` with `seen` for fracture, Baker's cyst, and contusion. These
are screening counts, not adjudicated contradictions. The extractor's exact
conflict counts are reported separately below.

The audit exported no raw report, excerpt, concordance, row number, patient
identifier, or study identifier. Its minimum lexical document frequency was
five. This privacy boundary prevents reviewing rare terms or their local
context, so excluded terminology and the language field remain intentionally
unresolved.

## Extractor design

`report-extractor-v1` uses normalization version
`unicode-nfc-horizontal-whitespace-v1` and only Python standard-library text
processing plus existing project machinery. It introduced no NLP framework,
model, embedding service, LLM, translation service, external report API, or new
dependency.

The deterministic processing path is:

1. normalize Unicode to NFC, line endings to LF, and horizontal whitespace
   while preserving line boundaries and clinically meaningful text;
2. segment inline and standalone headings plus sentences with offsets into the
   normalized report;
3. match target concepts from the frozen regular-expression inventory;
4. evaluate target and global affirmation, negation, uncertainty, historical,
   and section context in a five-token scope;
5. stop scopes at hard clause boundaries and distinguish shared coordinated
   assertions from a later finding with its own affirmative cue;
6. aggregate all usable mentions in report order and resolve unresolved
   positive/negative or other incompatible evidence to `uncertain`; and
7. generate the provenance key as SHA-256 of a compact UTF-8 JSON array
   containing study identifier, target, extractor version, and the one
   normalized-report SHA-256.

Historical or excluded-section evidence is recorded as
`contextual_exclusion` but does not make a current label positive. Absence of a
usable mention is `unknown`, never implicit negative. Uncertainty has precedence
over negative descriptors at the same mention, and report-level conflicts use
the target's explicit `unresolved_to_uncertain` rule. A generic assertion rule
cannot silently override target-specific evidence.

### Frozen rule inventory

The global inventory contains one section rule, one affirmative rule, five
negative rules, three uncertainty rules, and four contextual-exclusion rules.
Every target has one explicit conflict rule.

| Target | Concept | Target assertion/context rules |
| --- | ---: | --- |
| ACL | 2 | 2 negative |
| MCL | 2 | 2 negative |
| Medial Meniscus | 1 | 2 negative, 1 uncertainty, 1 section |
| Lateral Meniscus | 1 | 2 negative, 1 section |
| Medial OA | 2 | none |
| Lateral OA | 2 | none |
| PF OA | 2 | none |
| Effusion | 1 | 1 negative |
| Synovitis | 1 | none |
| Baker's | 1 | none |
| Contusion | 1 | none |
| Fracture | 1 | none |

## Extraction results

Private kernel
[`plthon/rsna-knee-report-labels`](https://www.kaggle.com/code/plthon/rsna-knee-report-labels)
completed the extraction stage on CPU with internet, GPU, and TPU disabled. It
used clean Git commit
`77006928878dbda1810a9c4ea15b8cc201c0215f`, the exact competition mount and
two allowed columns, and ran extraction in 19.50 seconds. The notebook source,
canonical rendered notebook, embedded package archive, competition contract,
rule contract, and audit artifact matched their recorded SHA-256 values.

The run normalized all 4,407 reports, produced exactly 52,884 ordered
study-target label rows (`4,407 × 12`), and produced 13,544 evidence rows. Its
repeat extraction/equality check passed. There were zero duplicate-order or
state validation failures, zero malformed reports, and zero extraction errors.
Across all targets there were 4,551 positive, 3,481 negative, 127 uncertain,
and 44,725 unknown rows; overall non-unknown coverage was 15.43%.

| Target | Positive | Negative | Uncertain | Unknown | Coverage | Conflict labels | Context-exclusion evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACL | 635 | 710 | 24 | 3,038 | 31.06% | 24 | 253 |
| MCL | 853 | 481 | 27 | 3,046 | 30.88% | 24 | 216 |
| Medial Meniscus | 728 | 347 | 19 | 3,313 | 24.82% | 19 | 317 |
| Lateral Meniscus | 506 | 632 | 13 | 3,256 | 26.12% | 13 | 230 |
| Medial OA | 0 | 0 | 0 | 4,407 | 0.00% | 0 | 0 |
| Lateral OA | 0 | 0 | 0 | 4,407 | 0.00% | 0 | 0 |
| PF OA | 21 | 0 | 0 | 4,386 | 0.48% | 0 | 0 |
| Effusion | 958 | 405 | 8 | 3,036 | 31.11% | 8 | 244 |
| Synovitis | 260 | 12 | 3 | 4,132 | 6.24% | 0 | 43 |
| Baker's | 347 | 245 | 3 | 3,812 | 13.50% | 3 | 232 |
| Contusion | 151 | 153 | 19 | 4,084 | 7.33% | 13 | 19 |
| Fracture | 92 | 496 | 11 | 3,808 | 13.59% | 9 | 38 |

Coverage is the fraction of positive, negative, or uncertain rows. It is an
engineering completeness measure only, not a correctness or usefulness score.

### Rule use

Global rule-use counts were:

| Rule | Uses | Rule | Uses |
| --- | ---: | --- | ---: |
| `GLOBAL.SECTION.001` | 1,585 | `GLOBAL.POS.001` | 368 |
| `GLOBAL.NEG.001` | 1,373 | `GLOBAL.NEG.002` | 21 |
| `GLOBAL.NEG.003` | 3 | `GLOBAL.NEG.004` | 3 |
| `GLOBAL.NEG.005` | 165 | `GLOBAL.UNCERTAIN.001` | 17 |
| `GLOBAL.UNCERTAIN.002` | 0 | `GLOBAL.UNCERTAIN.003` | 16 |
| `GLOBAL.EXCLUDE.001` | 1 | `GLOBAL.EXCLUDE.002` | 3 |
| `GLOBAL.EXCLUDE.003` | 1 | `GLOBAL.EXCLUDE.004` | 2 |

Target-specific rule-use counts were:

| Target | Concept-rule uses | Assertion/section-rule uses | Conflict-rule evidence uses |
| --- | --- | --- | ---: |
| ACL | `.CONCEPT.001` 1,731; `.CONCEPT.002` 801 | `.NEG.001` 841; `.NEG.002` 166 | 71 |
| MCL | `.CONCEPT.001` 1,449; `.CONCEPT.002` 810 | `.NEG.001` 601; `.NEG.002` 203 | 74 |
| Medial Meniscus | `.CONCEPT.001` 2,155 | `.NEG.001` 236; `.NEG.002` 84; `.UNCERTAIN.001` 0; `.SECTION.001` 524 | 47 |
| Lateral Meniscus | `.CONCEPT.001` 1,755 | `.NEG.001` 415; `.NEG.002` 151; `.SECTION.001` 463 | 36 |
| Medial OA | both concept rules 0 | none | 0 |
| Lateral OA | both concept rules 0 | none | 0 |
| PF OA | `.CONCEPT.001` 0; `.CONCEPT.002` 22 | none | 0 |
| Effusion | `.CONCEPT.001` 1,999 | `.NEG.001` 2 | 16 |
| Synovitis | `.CONCEPT.001` 511 | none | 0 |
| Baker's | `.CONCEPT.001` 1,007 | none | 6 |
| Contusion | `.CONCEPT.001` 495 | none | 34 |
| Fracture | `.CONCEPT.001` 809 | none | 31 |

Conflict-rule use counts above count every evidence row annotated with that
rule; the earlier conflict-label column counts unique affected study-target
labels. Unlisted frozen rules had zero uses.

### Artifact identity and privacy

| Artifact | Bytes | SHA-256 | Handling |
| --- | ---: | --- | --- |
| `report-labels-v1.csv` | 12,409,688 | `6651f60b00e5bec292e8c98e7fa83c75e8aa5af5931414a9a68c8ce10a52d527` | Private; not downloaded |
| `report-evidence-v1.jsonl` | 7,025,777 | `63e821f146cb76b20b10297f94c79b9a65cb46ed9f6bfaf85d7ef43b81600257` | Private; not downloaded |
| `report-extraction-summary.json` | 4,811 | `d631a658047dbd50124b75dc160e240e6a53146d04ee9826d913a207ef4e8f72` | Aggregate-only; downloaded to ignored storage |
| `report-extraction-manifest.json` | 4,154 | `acbf3b10a9698228e80ec6b2be535f47beea5143c3ccaa1ca4791dd747e3cebd` | Public-safe manifest; downloaded to ignored storage |
| execution log | 2,507 | `b497845fd69541b94e0dce360e7873167f5c40d5d74b5e5b791a6380f23e8856` | No report content; downloaded to ignored storage |

The summary's private-artifact sizes and hashes exactly match the manifest.
Only the aggregate summary, public-safe manifest, and log were retrieved after
the run. The pull boundary rejects the two row-level filenames. No report text,
row-level label, evidence span, study identifier, normalized report hash, or
organizer-gold value was downloaded or committed.

## Validation readiness

**The frozen artifact is technically ready for issue #21 manual validation.**

This means its schema, completeness, ordering, provenance, determinism, privacy,
and aggregate execution evidence have been verified. It does **not** establish
label correctness. Coverage, rule frequency, plausible class counts, and a
successful deterministic run are not substitutes for blinded comparison with
organizer gold and manual review. In particular, zero coverage for medial and
lateral OA, very low PF OA coverage, unresolved language, rare-term privacy
limits, and nonzero conflict counts must be treated as validation risks.

No report-derived label has been represented as organizer gold or used for
image-model training or validation. Activation remains exclusively a decision
for issue #21.

## Acceptance-criterion map

| Criterion | Evidence | Outcome |
| --- | --- | --- |
| Exact report input; no DICOM or organizer-gold tuning | Manifest records `train.csv` and only `StudyInstanceUID`/`Report`; notebook boundary tests | Satisfied |
| Aggregate audit before rule freeze | Audit SHA-256 pinned in contract; five-report lexical threshold | Satisfied |
| Deterministic lexical, section-aware scoped extractor | Frozen rule contract, synthetic regressions, repeat extraction check | Satisfied |
| Exact target order and four-state vocabulary | Contract loader and 52,884-row manifest validation | Satisfied |
| Auditable non-unknown evidence and stable provenance | 13,544 ordered evidence rows; stable rule IDs and provenance tests | Satisfied |
| Historical, uncertainty, coordination, and conflict handling | Frozen global/target rules and synthetic interaction tests | Satisfied |
| Private artifact contract and no raw matched text | Exact artifact names/columns/fields, hashes, private storage | Satisfied |
| CPU, internet-off guarded Kaggle workflow | Completed private kernel and explicit two-stage approval gates | Satisfied |
| No new NLP dependency or learned/hosted method | Standard-library implementation and unchanged dependency set | Satisfied |
| Public-safe final evidence and #21 readiness statement | This aggregate-only report and explicit non-correctness decision | Satisfied |
