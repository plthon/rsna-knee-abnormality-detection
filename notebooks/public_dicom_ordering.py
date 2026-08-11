# %% [markdown]
# # RSNA Knee MRI | Correct DICOM Ordering & Sampling
#
# Knee MRI studies contain multiple imaging series, and each series contains many
# DICOM slices. The filenames are identifiers, not anatomical coordinates, so
# lexicographic filename order can scramble a volume. This notebook builds a
# compact, reusable loader that prefers DICOM geometry, samples slices
# deterministically, and applies a lightweight display-oriented normalization.
#
# The notebook reads headers for only three representative series (axial,
# sagittal, and coronal) and decodes only the displayed slices. It does **not**
# rebuild the full metadata index, train a model, require a GPU, use the internet,
# or write persistent artifacts.

# %%
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

import matplotlib.pyplot as plt  # type: ignore[import-not-found]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from IPython.display import display  # type: ignore[import-not-found]
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.pixels.processing import apply_modality_lut

NOTEBOOK_STARTED_AT = perf_counter()
PLANES = ("Axial", "Sagittal", "Coronal")
DICOM_SUFFIXES = {".dcm", ".dicom"}

# %% [markdown]
# ## 1. Competition dataset at a glance
#
# Kaggle can mount competition inputs under different parent directories. Instead
# of assuming one absolute path, the discovery function checks the shallow input
# layout for the expected CSV files and image directories, then requires exactly
# one matching root.


# %%
def discover_competition_root(input_root: Path = Path("/kaggle/input")) -> Path:
    required_files = {
        "train.csv",
        "train_series.csv",
        "test.csv",
        "test_series.csv",
        "sample_submission.csv",
    }
    required_directories = {"train_series", "test_series"}
    csv_candidates = {
        *input_root.glob("train_series.csv"),
        *input_root.glob("*/train_series.csv"),
        *input_root.glob("*/*/train_series.csv"),
    }
    roots = sorted(
        {
            path.parent
            for path in csv_candidates
            if all((path.parent / name).is_file() for name in required_files)
            and all((path.parent / name).is_dir() for name in required_directories)
        },
        key=lambda path: path.as_posix(),
    )
    if len(roots) != 1:
        raise FileNotFoundError(
            f"Expected one competition root below {input_root}, found {len(roots)}"
        )
    return roots[0]


competition_root = discover_competition_root()
print(f"Competition root: {competition_root}")

# %%
metadata_files = sorted(competition_root.glob("*.csv"))
metadata_inventory = pd.DataFrame(
    [
        {
            "file": path.name,
            "columns": ", ".join(pd.read_csv(path, nrows=0).columns),
        }
        for path in metadata_files
    ]
)
display(metadata_inventory)

sample_submission_columns = tuple(
    pd.read_csv(competition_root / "sample_submission.csv", nrows=0).columns
)
if not sample_submission_columns or sample_submission_columns[0] != "StudyInstanceUID":
    raise ValueError("Unexpected sample submission identifier column")
target_names = sample_submission_columns[1:]
if not target_names:
    raise ValueError("Sample submission contains no target columns")

train_studies = pd.read_csv(
    competition_root / "train.csv", usecols=["StudyInstanceUID"]
)
test_studies = pd.read_csv(competition_root / "test.csv", usecols=["StudyInstanceUID"])
train_series = pd.read_csv(competition_root / "train_series.csv").assign(
    split="train", image_root="train_series"
)
test_series = pd.read_csv(competition_root / "test_series.csv").assign(
    split="test", image_root="test_series"
)
series_table = pd.concat([train_series, test_series], ignore_index=True)

current_mount = pd.DataFrame(
    [
        {
            "split": "train",
            "studies in CSV": len(train_studies),
            "series in CSV": len(train_series),
        },
        {
            "split": "example test",
            "studies in CSV": len(test_studies),
            "series in CSV": len(test_series),
        },
    ]
)
display(current_mount)

# %% [markdown]
# The target names and order come directly from `sample_submission.csv`:

# %%
display(
    pd.DataFrame(
        {
            "submission position": range(1, len(target_names) + 1),
            "target": target_names,
        }
    )
)

# %% [markdown]
# ### Verified snapshot scale
#
# The following figures come from a completed header-only index of the mounted
# competition snapshot (verified 11 August 2026). They are not recomputed here:
# doing so required about 2 hours 48 minutes on Kaggle CPU and would defeat this
# notebook's lightweight purpose.

# %%
verified_snapshot = pd.DataFrame(
    [
        {
            "scope": "train",
            "studies": 4_407,
            "series": 24_371,
            "DICOM headers": 819_078,
            "unique sampled positions": 194_968,
        },
        {
            "scope": "example test",
            "studies": 3,
            "series": 15,
            "DICOM headers": 557,
            "unique sampled positions": 120,
        },
        {
            "scope": "total",
            "studies": 4_410,
            "series": 24_386,
            "DICOM headers": 819_635,
            "unique sampled positions": 195_088,
        },
    ]
)
display(verified_snapshot)

# %% [markdown]
# All 819,635 headers in that snapshot were indexed without a structured error,
# and geometry-based ordering was available and selected for all 24,386 series.
# The eight-position policy selected 195,088 unique positions without duplicating
# slices from short series.
# This is a metadata-wide result. It does **not** mean every pixel array was
# decoded or every ordered volume was visually inspected. Pixel decoding and
# visual checks were performed only on representative series.
#
# The identifier hierarchy is:
#
# ```text
# StudyInstanceUID               one MRI study
# └── SeriesInstanceUID          one acquisition within that study
#     └── SOPInstanceUID.dcm     one image/slice in that series
# ```

# %% [markdown]
# ## 2. Choose representative axial, sagittal, and coronal series
#
# Selection is deterministic: example-test rows are considered before training
# rows, then identifiers are sorted. Only the first existing series for each
# plane is used.


# %%
def dicom_paths(series_directory: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (
                path
                for path in series_directory.iterdir()
                if path.is_file() and path.suffix.casefold() in DICOM_SUFFIXES
            ),
            key=lambda path: path.name,
        )
    )


def select_representative_series(
    table: pd.DataFrame, root: Path
) -> list[dict[str, Any]]:
    ordered = table.assign(
        split_priority=table["split"].map({"test": 0, "train": 1})
    ).sort_values(
        ["split_priority", "StudyInstanceUID", "SeriesInstanceUID"],
        kind="stable",
    )
    selected: list[dict[str, Any]] = []
    for plane in PLANES:
        candidates = ordered[
            ordered["Anatomical_Plane"].astype(str).str.casefold() == plane.casefold()
        ]
        for row in candidates.to_dict(orient="records"):
            directory = (
                root
                / str(row["image_root"])
                / str(row["StudyInstanceUID"])
                / str(row["SeriesInstanceUID"])
            )
            paths = dicom_paths(directory)
            if paths:
                selected.append({**row, "directory": directory, "paths": paths})
                break
        else:
            raise FileNotFoundError(f"No readable {plane} series was found")
    return selected


representatives = select_representative_series(series_table, competition_root)
representative_summary = pd.DataFrame(
    [
        {
            "plane": item["Anatomical_Plane"],
            "split": item["split"],
            "StudyInstanceUID": item["StudyInstanceUID"],
            "SeriesInstanceUID": item["SeriesInstanceUID"],
            "slices": len(item["paths"]),
        }
        for item in representatives
    ]
)
display(representative_summary)

# %%
example_study_uid = str(representatives[0]["StudyInstanceUID"])
study_hierarchy = series_table[
    series_table["StudyInstanceUID"].astype(str) == example_study_uid
][
    [
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "Anatomical_Plane",
        "Fluid_Sensitive",
        "Fat_Suppression",
    ]
].sort_values("SeriesInstanceUID")
display(study_hierarchy)

# %% [markdown]
# ## 3. Inspect DICOM headers without decoding pixels
#
# For spatial ordering we need `ImageOrientationPatient` and
# `ImagePositionPatient`. `InstanceNumber` is useful as a comparison or fallback,
# while the filename is only a deterministic last resort.


# %%
@dataclass(frozen=True, slots=True)
class SliceRecord:
    path: Path
    instance_number: int | None
    orientation: tuple[float, ...] | None
    position: tuple[float, ...] | None


@dataclass(frozen=True, slots=True)
class OrderedSeries:
    records: tuple[SliceRecord, ...]
    method: Literal["geometry", "instance_number", "filename"]
    geometry_coordinates: dict[Path, float]


def float_sequence(
    dataset: Dataset, attribute: str, expected_length: int
) -> tuple[float, ...] | None:
    value: object = getattr(dataset, attribute, None)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return result if len(result) == expected_length else None


def integer_attribute(dataset: Dataset, attribute: str) -> int | None:
    value: object = getattr(dataset, attribute, None)
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return None


def read_series_headers(paths: Sequence[Path]) -> tuple[SliceRecord, ...]:
    records: list[SliceRecord] = []
    for path in paths:
        dataset = dcmread(
            path,
            stop_before_pixels=True,
            specific_tags=[
                "InstanceNumber",
                "ImageOrientationPatient",
                "ImagePositionPatient",
            ],
        )
        records.append(
            SliceRecord(
                path=path,
                instance_number=integer_attribute(dataset, "InstanceNumber"),
                orientation=float_sequence(dataset, "ImageOrientationPatient", 6),
                position=float_sequence(dataset, "ImagePositionPatient", 3),
            )
        )
    return tuple(records)


representative_records = [
    read_series_headers(item["paths"]) for item in representatives
]

# %%
header_rows: list[dict[str, Any]] = []
for item, records in zip(representatives, representative_records, strict=True):
    dataset = dcmread(records[0].path, stop_before_pixels=True)
    header_rows.append(
        {
            "plane": item["Anatomical_Plane"],
            "Modality": getattr(dataset, "Modality", None),
            "Rows × Columns": f"{getattr(dataset, 'Rows', '?')} × {getattr(dataset, 'Columns', '?')}",
            "InstanceNumber": records[0].instance_number,
            "ImageOrientationPatient": records[0].orientation,
            "ImagePositionPatient": records[0].position,
            "PhotometricInterpretation": getattr(
                dataset, "PhotometricInterpretation", None
            ),
            "RescaleSlope": float(getattr(dataset, "RescaleSlope", 1.0)),
            "RescaleIntercept": float(getattr(dataset, "RescaleIntercept", 0.0)),
            "TransferSyntaxUID": str(
                getattr(dataset.file_meta, "TransferSyntaxUID", "")
            ),
        }
    )
display(pd.DataFrame(header_rows))

# %% [markdown]
# ## 4. Order slices by geometry
#
# A slice plane is defined by two direction vectors in
# `ImageOrientationPatient`. Their cross product is the plane normal. Projecting
# `ImagePositionPatient` onto a consistent unit normal gives one scalar coordinate
# per slice; sorting that coordinate produces a spatially consistent sequence.
# The sign of the normal depends on the stored direction cosines, so ascending
# projected coordinate can run either way through the anatomy.
# It does not define a universal clinical anatomical direction.
#
# If complete, consistent, unique geometry is unavailable, the reusable function
# below tries unique `InstanceNumber` values, then filename order. It never treats
# a partially populated or duplicated `InstanceNumber` sequence as authoritative.


# %%
def geometry_coordinates(
    records: Sequence[SliceRecord],
) -> dict[Path, float] | None:
    normals: list[np.ndarray] = []
    for record in records:
        if record.orientation is None or record.position is None:
            return None
        row = np.asarray(record.orientation[:3], dtype=np.float64)
        column = np.asarray(record.orientation[3:], dtype=np.float64)
        normal = np.cross(row, column)
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(normal).all() or norm < 1e-6:
            return None
        normals.append(normal / norm)
        if not np.isfinite(np.asarray(record.position, dtype=np.float64)).all():
            return None

    reference = normals[0]
    if any(float(np.dot(reference, normal)) < 0.999 for normal in normals[1:]):
        return None
    coordinates = {
        record.path: float(np.dot(reference, np.asarray(record.position)))
        for record in records
    }
    rounded = [round(value, 6) for value in coordinates.values()]
    return coordinates if len(set(rounded)) == len(rounded) else None


def order_series(records: Sequence[SliceRecord]) -> OrderedSeries:
    if not records:
        raise ValueError("Cannot order an empty series")

    coordinates = geometry_coordinates(records)
    if coordinates is not None:
        return OrderedSeries(
            records=tuple(sorted(records, key=lambda item: coordinates[item.path])),
            method="geometry",
            geometry_coordinates=coordinates,
        )

    instance_numbers = [record.instance_number for record in records]
    if all(value is not None for value in instance_numbers) and len(
        set(instance_numbers)
    ) == len(records):
        return OrderedSeries(
            records=tuple(
                sorted(records, key=lambda item: cast(int, item.instance_number))
            ),
            method="instance_number",
            geometry_coordinates={},
        )

    return OrderedSeries(
        records=tuple(sorted(records, key=lambda item: item.path.name)),
        method="filename",
        geometry_coordinates={},
    )


def instance_order(
    records: Sequence[SliceRecord],
) -> tuple[SliceRecord, ...] | None:
    values = [record.instance_number for record in records]
    if any(value is None for value in values) or len(set(values)) != len(records):
        return None
    return tuple(sorted(records, key=lambda item: cast(int, item.instance_number)))


ordered_representatives = [order_series(records) for records in representative_records]

# %%
comparison_rows = []
for item, records, ordered in zip(
    representatives,
    representative_records,
    ordered_representatives,
    strict=True,
):
    filename_records = tuple(sorted(records, key=lambda record: record.path.name))
    by_instance = instance_order(records)
    comparison_rows.append(
        {
            "plane": item["Anatomical_Plane"],
            "slices": len(records),
            "selected method": ordered.method,
            "filename = geometry": filename_records == ordered.records,
            "InstanceNumber = geometry": (
                by_instance == ordered.records if by_instance is not None else None
            ),
        }
    )
ordering_comparison = pd.DataFrame(comparison_rows)
display(ordering_comparison)

# %% [markdown]
# In the three representative series shown here, filename ordering disagrees
# with geometry in all three, while `InstanceNumber` disagrees in two of three.
# Separately, the 11 August 2026 header-only index found valid geometry for all
# 24,386 series in the mounted snapshot, so no fallback ordering was required
# during indexing.

# %% [markdown]
# ## 5. Visual comparison: filename order versus geometry
#
# Each row samples the same relative positions from a different ordering. The
# title `g=` gives that image's rank in the geometry-ordered series. A smoothly
# increasing geometry rank is what filename order cannot guarantee. The rank
# describes sequence consistency, not a universal clinical anatomical direction.


# %%
def uniform_indices(length: int, count: int) -> tuple[int, ...]:
    if length < 0 or count <= 0:
        raise ValueError("length must be non-negative and count must be positive")
    if length <= count:
        return tuple(range(length))
    return tuple(np.linspace(0, length - 1, count, dtype=int).tolist())


def label_figure_rows(figure: Any, axes: np.ndarray, labels: Sequence[str]) -> None:
    if len(axes) != len(labels):
        raise ValueError("Each plot row needs exactly one label")
    for row_axes, label in zip(axes, labels, strict=True):
        position = row_axes[0].get_position()
        figure.text(
            0.015,
            (position.y0 + position.y1) / 2,
            label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=12,
        )


def decode_pixels(path: Path) -> np.ndarray:
    dataset = dcmread(path)
    stored_pixels = dataset.pixel_array
    if "ModalityLUTSequence" in dataset:
        transformed = np.asarray(
            apply_modality_lut(stored_pixels, dataset), dtype=np.float64
        )
    else:
        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        if not np.isfinite([slope, intercept]).all():
            raise ValueError(f"Non-finite rescale metadata in {path.name}")
        transformed = np.asarray(stored_pixels, dtype=np.float64) * slope + intercept

    if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
        transformed = transformed.max() + transformed.min() - transformed
    output = np.asarray(transformed, dtype=np.float32)
    if not np.isfinite(output).all():
        raise ValueError(f"Decoded pixels contain non-finite values: {path.name}")
    return output


def normalize_images(
    images: Sequence[np.ndarray],
    lower_percentile: float = 1,
    upper_percentile: float = 99,
) -> tuple[np.ndarray, ...]:
    if not images:
        raise ValueError("Cannot normalize an empty image sequence")
    values = np.concatenate([image.reshape(-1) for image in images])
    lower, upper = np.percentile(values, (lower_percentile, upper_percentile))
    if not np.isfinite([lower, upper]).all():
        raise ValueError("Normalization percentiles must be finite")
    if lower == upper:
        return tuple(np.zeros_like(image, dtype=np.float32) for image in images)
    normalized = tuple(
        np.clip((image - lower) / (upper - lower), 0, 1).astype(np.float32)
        for image in images
    )
    if not all(np.isfinite(image).all() for image in normalized):
        raise ValueError("Normalized pixels contain non-finite values")
    return normalized


contrast_index = next(
    (
        index
        for index, (records, ordered) in enumerate(
            zip(representative_records, ordered_representatives, strict=True)
        )
        if tuple(sorted(records, key=lambda item: item.path.name)) != ordered.records
    ),
    0,
)
contrast_records = representative_records[contrast_index]
contrast_geometry = ordered_representatives[contrast_index].records
contrast_filename = tuple(sorted(contrast_records, key=lambda item: item.path.name))
comparison_count = 6
filename_positions = uniform_indices(len(contrast_filename), comparison_count)
geometry_positions = uniform_indices(len(contrast_geometry), comparison_count)
filename_sample = tuple(contrast_filename[index] for index in filename_positions)
geometry_sample = tuple(contrast_geometry[index] for index in geometry_positions)
unique_paths = tuple(
    dict.fromkeys(record.path for record in (*filename_sample, *geometry_sample))
)
normalized_by_path = dict(
    zip(
        unique_paths,
        normalize_images([decode_pixels(path) for path in unique_paths]),
        strict=True,
    )
)
geometry_rank = {record.path: index for index, record in enumerate(contrast_geometry)}

figure, axes = plt.subplots(2, comparison_count, figsize=(18, 7))
for row_index, sample in enumerate((filename_sample, geometry_sample)):
    for column_index, record in enumerate(sample):
        axes[row_index, column_index].imshow(
            normalized_by_path[record.path], cmap="gray", vmin=0, vmax=1
        )
        title = f"g={geometry_rank[record.path]}"
        if row_index == 0:
            title = f"sample {column_index + 1}\n{title}"
        axes[row_index, column_index].set_title(title, fontsize=11, pad=3)
        axes[row_index, column_index].axis("off")
figure.suptitle(
    f"{representatives[contrast_index]['Anatomical_Plane']} series: filename vs geometry order",
    fontsize=16,
)
figure.tight_layout(rect=(0.045, 0, 1, 0.94), h_pad=2.5)
label_figure_rows(figure, axes, ("Filename order", "Geometry order"))
plt.show()

# %% [markdown]
# ## 6. Deterministic slice sampling
#
# `uniform_indices` spans the first through last slice for a long series. When a
# series is shorter than the requested count, it returns every position once and
# never duplicates a slice.

# %%
long_length = len(ordered_representatives[0].records)
sampling_examples = pd.DataFrame(
    [
        {
            "case": "representative long series",
            "series length": long_length,
            "requested": 8,
            "selected positions": uniform_indices(long_length, 8),
        },
        {
            "case": "short series example",
            "series length": 5,
            "requested": 8,
            "selected positions": uniform_indices(5, 8),
        },
    ]
)
display(sampling_examples)

# %% [markdown]
# ## 7. Lightweight preprocessing and three-plane visualization
#
# The baseline below applies DICOM modality rescale information, inverts
# `MONOCHROME1` images when needed, and clips to the 1st and 99th percentile
# values computed jointly across the deterministically selected slices in each
# series. The loader preserves each slice's native row and column resolution; it
# does not resize or pad the decoded images.


# %%
def load_series(
    paths: Sequence[Path], count: int = 8
) -> tuple[tuple[np.ndarray, ...], tuple[Path, ...], tuple[int, ...], str]:
    ordered = order_series(read_series_headers(paths))
    selected_positions = uniform_indices(len(ordered.records), count)
    selected_paths = tuple(ordered.records[index].path for index in selected_positions)
    normalized = normalize_images([decode_pixels(path) for path in selected_paths])
    if not all(
        float(image.min()) >= 0 and float(image.max()) <= 1 for image in normalized
    ):
        raise ValueError("Expected normalized finite images in [0, 1]")
    return normalized, selected_paths, selected_positions, ordered.method


loaded_representatives = [load_series(item["paths"]) for item in representatives]
preprocessing_checks = pd.DataFrame(
    [
        {
            "plane": item["Anatomical_Plane"],
            "ordering": method,
            "selected positions": positions,
            "decoded slices": len(images),
            "native shapes": tuple(sorted({image.shape for image in images})),
            "all finite": all(np.isfinite(image).all() for image in images),
            "minimum": min(float(image.min()) for image in images),
            "maximum": max(float(image.max()) for image in images),
        }
        for item, (images, _paths, positions, method) in zip(
            representatives, loaded_representatives, strict=True
        )
    ]
)
display(preprocessing_checks)

# %%
figure, axes = plt.subplots(len(PLANES), 8, figsize=(20, 8))
plane_row_labels = []
for row_index, (item, loaded) in enumerate(
    zip(representatives, loaded_representatives, strict=True)
):
    images, _paths, positions, method = loaded
    plane_row_labels.append(f"{item['Anatomical_Plane']}\n({method})")
    for column_index, (image, position) in enumerate(
        zip(images, positions, strict=True)
    ):
        axes[row_index, column_index].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[row_index, column_index].set_title(f"position {position}")
        axes[row_index, column_index].axis("off")
figure.suptitle(
    "Geometry-ordered, uniformly sampled representative knee MRI series", fontsize=16
)
figure.tight_layout(rect=(0.045, 0, 1, 0.94), h_pad=1.5)
label_figure_rows(figure, axes, plane_row_labels)
plt.show()

# %% [markdown]
# In this notebook run, the table and figure above directly demonstrate eight
# deterministically selected slices from each representative axial, sagittal, and
# coronal series. Every displayed slice is finite and normalized into `[0, 1]`.
# The `native shapes` column reports the actual rows and columns returned for each
# series; no resizing or padding is performed. This representative demonstration
# does not show that all 819,635 pixel arrays can be decoded.

# %% [markdown]
# ## 8. Reusable loader
#
# The small public interface is `load_series(paths, count=8)`. It returns the
# normalized images, their source paths, their positions in the geometry-ordered
# series, and the ordering method that was actually used. Each returned array
# retains its native DICOM dimensions. DICOM read or decode failures are
# deliberately allowed to raise with their path context rather than being silently
# swallowed.

# %%
example_images, example_paths, example_positions, example_method = load_series(
    representatives[0]["paths"], count=8
)
print(
    {
        "ordering_method": example_method,
        "selected_positions": example_positions,
        "selected_file_count": len(example_paths),
        "decoded_slices": len(example_images),
        "native_shapes": tuple(sorted({image.shape for image in example_images})),
    }
)

# %% [markdown]
# ## 9. Practical takeaways
#
# - Prefer geometry from `ImageOrientationPatient` and `ImagePositionPatient` for
#   a spatially consistent sequence, without assigning a universal anatomical
#   direction to ascending coordinates.
# - Do not assume DICOM filenames encode anatomical order.
# - Use a complete, unique `InstanceNumber` sequence only as a fallback or
#   comparison signal.
# - Keep filename sorting as a deterministic final fallback.
# - Inspect headers first; avoid decoding an entire corpus for basic structure.
# - Sample deterministically, and never duplicate slices from a short series.
# - Compute display percentiles jointly across the selected slices, while keeping
#   their native image dimensions.
#
# This notebook uses NumPy, pandas, Matplotlib, and pydicom. DICOM geometry and
# modality transforms follow the corresponding DICOM metadata fields exposed by
# pydicom.

# %%
runtime_seconds = perf_counter() - NOTEBOOK_STARTED_AT
print(f"Notebook runtime: {runtime_seconds:.1f} seconds on this session")
