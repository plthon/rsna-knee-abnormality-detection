# Canonical competition inspection report

Verified 2026-08-09 and hardened 2026-08-11 for `rsna-knee-abnormality-detection`. The machine-readable source of truth is [`contracts/competition.json`](../contracts/competition.json). This report combines the completed real-data inspection, its narrow follow-up, the official competition pages, and the later full-corpus header index. It is not a second EDA.

## Acceptance-criterion map

| Criterion | Initial classification against prior outputs | Missing evidence collected | Final outcome |
| --- | --- | --- | --- |
| AC1: files, sizes, schemas, IDs, reports, languages, targets, missing fields, duplicates | Requires additional lightweight inspection | Five-CSV quality pass; official file API | Satisfied; observed languages are explicitly unresolved |
| AC2: representative DICOM contract | Requires additional lightweight inspection | One non-pixel header per training study for `Modality` | Satisfied |
| AC3: leakage and grouping | Requires additional lightweight inspection | Aggregate patient-tag and cross-table checks | Satisfied |
| AC4: metric, submission, runtime, disk, internet, packages | Requires additional lightweight inspection | Official Evaluation and Code Requirements pages | Satisfied; official disk quota is explicitly unresolved |
| AC5: external data, weights, generative AI, report processing | Requires additional lightweight inspection | Official Rules and Code Requirements pages | Satisfied |
| AC6: runtime and image provenance | Already satisfied | Reconfirmed in the narrow run | Satisfied |
| AC7: no private-data download or commit | Already satisfied | Repository and artifact review | Satisfied |
| AC8: every provisional assumption classified | Requires additional lightweight inspection | Reconciled data and documentation evidence | Satisfied |

## Canonical facts

The competition mount contains exactly 819,640 files and 569,763,893,144 bytes: 819,635 DICOMs (569,754,741,408 bytes) and five CSVs (9,151,736 bytes). Training contains 4,407 studies, 24,371 series, and 819,078 images. The mounted test example contains three studies, 15 series, and 557 images; Kaggle replaces it with the hidden scoring set.

| Table | Bytes | Rows | Missing fields | Duplicate identifier rows |
| --- | ---: | ---: | --- | ---: |
| `train.csv` | 5,690,007 | 4,407 | 4,349 in each of the 12 target columns; none in `StudyInstanceUID` or `Report` | 0 |
| `train_series.csv` | 3,458,834 | 24,371 | None | 0 |
| `test.csv` | 212 | 3 | None | 0 |
| `test_series.csv` | 2,213 | 15 | None | 0 |
| `sample_submission.csv` | 470 | 3 | None | 0 |

The primary table keys and every `SeriesInstanceUID` are unique. Repeated `StudyInstanceUID` values in the series tables are expected foreign-key repetition: 4,407 studies own 24,371 training series, with at most 14 series per study. Global `SOPInstanceUID` uniqueness remains unresolved because proving it would repeat the full 819,635-path scan.

The mounted `train.csv` does not contain the `PatientSex` column listed by the official Data page. Code must follow the mounted schema. All 4,407 reports are nonempty; 134 rows duplicate another report text. No text was exported. The official page says reports may use several languages, but the internet-off image had no language detector, so observed languages remain unresolved.

The 12 ordered binary targets are: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, and Fracture. Each target has 58 populated rows and 4,349 missing rows. The compact evidence did not retain whether the same 58 rows are populated for every target, so joint row-wise missingness is unresolved.

Representative DICOMs ranged from 512×512 to 960×960 pixels and from 11 to 320 slices per series (median 30). Orientation metadata and axial, sagittal, and coronal planes were verified. All 4,407 one-per-study headers reported `MR`. The later full-corpus index counted Explicit VR Little Endian for every one of the 819,635 DICOMs in the mounted train and example-test snapshot; eight representative series decoded without failure. The official Data page nevertheless describes a mixed corpus, and Kaggle replaces the example test data during scoring, so hidden-set transfer syntaxes remain unresolved.

## Leakage and split grouping

Study, series, and image relationships are complete, with no train/test study or series overlap and no orphan table relationships. `PatientID` was present in all 4,407 training-study headers, with 4,407 unique values and no value spanning multiple studies. The highest safe available image-model grouping entity is therefore `PatientID`. It currently produces the same groups as `StudyInstanceUID`, but retaining the patient boundary is the safer contract.

The 134 duplicate report rows necessarily cross patient groups because PatientID and study are one-to-one. For any report model or report-label extractor, deduplicate or group validation by normalized report hash in addition to the patient split. Report hash is a content-leakage control, not an identity claim.

## Metric, submission, and platform

The official metric is the unweighted mean of the 12 per-target ROC AUC values (macro-averaged ROC AUC), maximized. Submission is notebook-only and must write `submission.csv` with one row per `StudyInstanceUID`, followed by the 12 targets in the order above. No inference API is required or specified.

The official CPU and GPU runtime limit is 9 hours, and internet must be disabled. No separate official disk quota was found. The observed `/kaggle/working` filesystem was 20,957,446,144 bytes with 20,940,529,664 bytes free; treat this as mutable observed capacity, not a guaranteed quota.

| Runtime fact | Observed value |
| --- | --- |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cpu |
| torchvision | 0.25.0+cpu |
| pydicom | 3.0.2 |
| NumPy | 2.0.2 |
| Accelerator / CUDA | CPU / unavailable (`null`) |
| Kaggle image commit | `fc61d5cda7da39530055bae9bd0e92865f995cd9` |
| Memory | 33,659,383,808 bytes |

## Competition rules

- External data is permitted when publicly and equally accessible at no cost, or otherwise reasonable under the rules, with compatible licensing.
- Freely and publicly available pretrained models are permitted and must be available offline to the submission notebook.
- Generative-AI/LLM tools are permitted subject to accessibility, reasonable cost, licensing, and data-security requirements.
- Hosted external report processing is prohibited when it transmits competition reports to an unauthorized third party. Offline processing with otherwise permitted tools remains allowed.

## Provisional-assumption closure

| Assumption | Status | Basis |
| --- | --- | --- |
| Approximately 600 GB dataset | Verified | Exact size is 569,763,893,144 bytes |
| Reports are available | Verified | 4,407/4,407 nonempty |
| Reports are multilingual | Unresolved | Documentation says possible; observed values were not reliably classified |
| Study → series → image hierarchy | Verified | Paths and tables agree |
| Patient grouping is available | Verified | `PatientID` present and unique for every training study |
| 12-target schema and order | Verified | Sample submission and Evaluation page agree |
| Macro mean ROC AUC metric | Verified | Official Evaluation page |
| CSV submission contract | Verified | Official Evaluation and Code Requirements pages |
| External data allowed | Verified | Permitted with conditions |
| Pretrained weights allowed | Verified | Permitted with conditions |
| Generative AI allowed | Verified | Permitted with conditions |
| Hosted external report APIs prohibited | Verified | Competition-data security rule |
| Internet-disabled inference | Verified | Official Code Requirements page |
| Runtime environment captured | Verified | Versions, image, CPU, memory, and disk recorded |
| Runtime limits fully known | Unresolved | Nine-hour limit verified; official disk quota unavailable |
| Observed-sample decoding works | Verified | Four representative series, no failures |
| Geometry-based ordering works | Verified | Selected on all representative series |
| Deterministic eight-slice sampling executes | Verified | Exercised on real sample |
| MONOCHROME1 path verified on competition data | Unresolved | Not observed in decoded sample |
| Non-default rescale path verified on competition data | Unresolved | Not observed in decoded sample |
| Percentile normalization is suitable | Unresolved | Smoke execution is not modelling evidence |
| Report pseudo-labels are needed | Unresolved | Missing labels are verified per target, but pseudo-labeling is a modelling choice |
| `PatientSex` exists in mounted `train.csv` | Rejected | Column absent despite documentation |
| All mounted-snapshot DICOMs use Explicit VR syntax | Verified | Full-corpus index counted 819,635/819,635 headers |
| All hidden scoring DICOMs use Explicit VR syntax | Unresolved | Official page documents a mix; hidden data is unavailable before scoring |
| Three example test rows equal scoring-set size | Rejected | Kaggle replaces the example at scoring time |
| Global `SOPInstanceUID` uniqueness | Unresolved | Would require repeating the excluded full-corpus path scan |

## Evidence and safeguards

Primary data evidence is private Kaggle kernel [`plthon/rsna-knee-real-data-inspection`](https://www.kaggle.com/code/plthon/rsna-knee-real-data-inspection), version 2. The gap-only follow-up is private kernel [`plthon/rsna-knee-identifier-finalization`](https://www.kaggle.com/code/plthon/rsna-knee-identifier-finalization): version 1 read five CSVs and one header per training study, and version 2 read only CSV identifier columns. Neither used pixels, GPU, internet, model training, pseudo-labeling, nor report/identifier export.

Policy evidence comes from the official [Evaluation](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation), [Code Requirements](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/code-requirements), [Rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules), and [Data](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data) pages, retrieved 2026-08-09.

Issue #4 downloaded only compact aggregate JSON and logs; it downloaded no competition CSV, report text, raw identifier values, or DICOM. The ignored workspace already contained issue #15's derived slice PNGs and JSON paths with UIDs. Those pre-existing private inspection artifacts were reused and were not committed.
