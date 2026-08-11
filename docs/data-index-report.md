# Canonical data index report

Verified 2026-08-11 for `rsna-knee-abnormality-detection`. The compact
machine-readable evidence is [`contracts/data-index.json`](../contracts/data-index.json).
The canonical input schema remains
[`contracts/competition.json`](../contracts/competition.json).

## Real-data result

Private Kaggle kernel
[`plthon/rsna-knee-data-index`](https://www.kaggle.com/code/plthon/rsna-knee-data-index),
version 1, completed successfully on CPU with internet, GPU, and TPU disabled.
Its verified output appeared after 10,063.59 seconds. The notebook completed all
header indexing before its representative pixel-decoding phase and ran the
artifact verifier as its final package operation.

The source is the Kaggle competition mount
`/kaggle/input/competitions/rsna-knee-abnormality-detection`. Kaggle exposes no
immutable platform version ID for a competition mount, so the source version is
recorded as a verified snapshot identified by the canonical competition-contract
SHA-256, its 2026-08-09 verification date, 819,640-file inventory, and exact
569,763,893,144-byte size.

The run provenance records base Git commit `adf8cd9`, the fact that the source
worktree was dirty, and SHA-256 hashes for the Jupytext source, exact rendered
notebook, safe kernel metadata, artifact manifest, statistics, and execution log.
The indexing configuration records the two source tables, verified join fields,
path template, requested header tags, ordering priority, eight-slice uniform
selection, short-series mask policy, representative-series limit, and complete
preprocessing parameters.

The index covers the complete mounted layout:

| Split | Studies | Series | Slices | Usable slices | Selected slices |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 4,407 | 24,371 | 819,078 | 819,078 | 194,968 |
| Example test | 3 | 15 | 557 | 557 | 120 |
| Total | 4,410 | 24,386 | 819,635 | 819,635 | 195,088 |

Every series used geometry ordering. All 819,635 headers reported Explicit VR
Little Endian (`1.2.840.10008.1.2.1`), so pydicom remains the only decoder
dependency and no optional plugin is justified. The run found no corrupt
headers, inconsistent dimensions within a series, metadata fallback, identifier
mismatch, or representative pixel-decode failure. The structured error table is
therefore header-only (94 bytes) and contains zero data rows.

The result does not weaken the fallback contract: synthetic integration tests
exercise corrupt headers and pixels, empty and unusable series, inconsistent
dimensions, instance-number ordering, deterministic filename ordering, and
short-series masks.

## Representative preprocessing

Eight real series spanning seven studies, both splits, and axial, coronal, and
sagittal planes were transformed through
`shared_train_inference_preprocess_series_result`. Every series decoded all eight
selected slices into finite `8 × 224 × 224` arrays with normalized range
`[0, 1]`. The run exercised observed nonzero and negative lower percentiles;
percentile normalization remains explicitly provisional.

The shared transform applies pydicom modality rescale, MONOCHROME1 inversion,
series-level percentile normalization, bilinear aspect-preserving resize, and
center padding. Short series retain fewer selected rows and a false-padded
eight-slot mask; slices are never duplicated.

## Versioned artifacts and safeguards

Kernel version 1 contains `studies.csv`, `series.csv`, `slices.csv`,
`selected_slices.csv`, `statistics.json`, `errors.csv`, and
`index-manifest.json`. The manifest records schema version 1, exact columns,
sizes, SHA-256 hashes, the canonical competition-contract hash, and
`decoded_images_persisted: false`.

Only the manifest, statistics, empty error table, and execution log were
downloaded for verification. The 175.5 MB slice index, normalized tables, source
DICOMs, and decoded images remain outside Git. No raw DICOM, decoded tensor, or
image file was downloaded or committed.

## Acceptance-criterion map

| Criterion | Evidence | Outcome |
| --- | --- | --- |
| Headers precede pixels; verified joins only | Canonical-contract loader, two-phase index test, successful real run | Satisfied |
| Geometry → instance → deterministic fallback | Synthetic ordering cases; all 24,386 real series used geometry | Satisfied |
| Count transfer syntaxes; evidence-driven decoder | 819,635 Explicit VR headers; zero sampled failures; pydicom only | Satisfied |
| Structured imperfect-data errors | Synthetic corrupt/empty/dimension/fallback cases; versioned `errors.csv` | Satisfied |
| Eight deterministic slices; short masks | Synthetic long/short selection tests; 195,088 real selections | Satisfied |
| Shared complete preprocessing contract | Unit and synthetic seams; eight successful real series | Satisfied |
| Version compact metadata, never pixels | Versioned manifest and six hashed artifacts; no decoded-image output | Satisfied |
| Synthetic and real integration checks | 35 local tests and completed Kaggle kernel version 1 | Satisfied |
