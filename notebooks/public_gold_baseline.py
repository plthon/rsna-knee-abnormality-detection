# %% [markdown]
# # A small reproducible MRI baseline for RSNA knee abnormalities
#
# A baseline is a simple reference model. This notebook uses magnetic resonance
# imaging (MRI) studies of the knee. One MRI study is one example for the model.
#
# A study can contain multiple image series. An acquisition is one MRI scan with
# a specified setup. A series is a group of related images from one acquisition.
# Each image is a two-dimensional slice. The files use the Digital Imaging and
# Communications in Medicine (DICOM) standard. ResNet-18 uses these slices to
# make probabilities for 12 abnormalities.
#
# This notebook explains each important part of the data path.
# Patient-level validation keeps each patient in only one data split. A fixed
# method selects the slices. The model averages valid features across slices and
# series.
#
# Inference is the use of a trained model to make predictions. The notebook
# reloads a saved checkpoint before test inference. A checkpoint is a file that
# contains saved model weights. The last cell makes and validates a submission
# file.
#
# Training uses only labels supplied by the competition organizers. The model
# does not use radiology reports or external labels. A pseudo-label is a label
# that another model makes. This model does not use pseudo-labels or leaderboard
# feedback.
#
# This reference run is small.
# It trains on 54 patients and evaluates on 4 different patients. The run shows
# that the pipeline can complete all steps.
# A score from four patients cannot give a stable estimate of model quality.

# %% [markdown]
# ## Load the tested code
#
# The notebook uses short examples to explain the main steps. The `rsna_knee`
# package does the complete run. A Python wheel is a file that contains a Python
# package. An attached Kaggle Dataset contains this wheel.
#
# The notebook does not need internet access. A hash is a value that identifies
# the exact contents of a file. The reproducibility section shows the wheel hash
# and other source information.

# %%
import base64
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

notebook_started = time.perf_counter()

WHEEL_DATASET_ID = "__RSNA_KNEE_WHEEL_DATASET_ID__"
WHEEL_FILENAME = "__RSNA_KNEE_WHEEL_FILENAME__"
WHEEL_SHA256 = "__RSNA_KNEE_WHEEL_SHA256__"
WEIGHT_DATASET_ID = "__RSNA_KNEE_WEIGHT_DATASET_ID__"
KERNEL_ID = "__RSNA_KNEE_KERNEL_ID__"
GPU_APPROVED = json.loads("__RSNA_KNEE_APPROVED_GPU__")

wheel_roots = (
    Path("/kaggle/input/datasets") / WHEEL_DATASET_ID,
    Path("/kaggle/input") / "__RSNA_KNEE_WHEEL_DATASET_SLUG__",
)
wheel_candidates = {
    path.resolve()
    for root in wheel_roots
    if root.exists()
    for path in root.rglob(WHEEL_FILENAME)
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == WHEEL_SHA256
}
if len(wheel_candidates) != 1:
    raise FileNotFoundError(
        f"Expected one hash-verified {WHEEL_FILENAME}, found {len(wheel_candidates)}"
    )
wheel_path = next(iter(wheel_candidates))
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        str(wheel_path),
    ],
    check=True,
)

import rsna_knee

for contract_name in (
    "data-index.json",
    "gold-baseline.json",
    "gold-supervision.json",
    "gold-training.json",
    "resnet18-imagenet1k-v1.json",
):
    contract_source = wheel_path.parent / contract_name
    if not contract_source.is_file():
        raise FileNotFoundError(f"Missing attached runtime contract: {contract_name}")
    shutil.copy2(contract_source, Path(rsna_knee.__file__).with_name(contract_name))

# %%
import matplotlib.pyplot as plt  # type: ignore[import-not-found]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
import torch.nn.functional as F
from IPython.display import display  # type: ignore[import-not-found]
from pydicom import dcmread
from pydicom.dataset import Dataset

from rsna_knee import (
    PreprocessingConfig,
    decode_pixels,
    preprocess_series_result,
    uniform_indices,
)
from rsna_knee.publication import (
    load_reference_baseline_summary,
    reproduce_reference_baseline,
)
from rsna_knee.training import (
    build_initialized_model,
    masked_binary_cross_entropy,
)

reference = load_reference_baseline_summary()
targets = reference["targets"]
print(f"Loaded the reference run for {len(targets)} prediction targets")

# %% [markdown]
# ## What the model predicts
#
# The competition requires one probability for each target in each study. The
# sample submission defines the output order. The code reads this order. It does
# not use a manual list.

# %%
display(
    pd.DataFrame(
        {
            "Output position": range(1, len(targets) + 1),
            "Target": targets,
        }
    )
)

# %% [markdown]
# ## What one MRI study contains
#
# `StudyInstanceUID` identifies each study. Each series has one directory of
# DICOM slices. The series table gives the anatomical plane. It also shows if an
# acquisition is fluid sensitive or uses fat suppression.
#
# This cell uses a deterministic method to select one study with all 12 labels.
# A deterministic method gives the same result for the same input. The method
# first selects studies that have axial, coronal, and sagittal series. It then
# uses the first study ID in the sorted list. The code does not read report text.

# %%
competition_root = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
required_training_files = ("train.csv", "train_series.csv", "train_series")
if not all((competition_root / name).exists() for name in required_training_files):
    raise FileNotFoundError(
        f"Competition training data not found at {competition_root}"
    )
train_labels = pd.read_csv(
    competition_root / "train.csv",
    usecols=["StudyInstanceUID", *targets],
    dtype={"StudyInstanceUID": str},
)
series_table = pd.read_csv(
    competition_root / "train_series.csv",
    dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
)

fully_labeled = set(train_labels.dropna(subset=targets)["StudyInstanceUID"].astype(str))
series_summary = (
    series_table[series_table["StudyInstanceUID"].isin(fully_labeled)]
    .groupby("StudyInstanceUID")
    .agg(
        series_count=("SeriesInstanceUID", "size"),
        plane_count=("Anatomical_Plane", "nunique"),
    )
)
three_plane_studies = series_summary[series_summary["plane_count"] >= 3]
two_plane_studies = series_summary[series_summary["plane_count"] >= 2]
study_candidates = (
    three_plane_studies
    if not three_plane_studies.empty
    else two_plane_studies
    if not two_plane_studies.empty
    else series_summary
)
study_uid = min(study_candidates.index)
study_rows = (
    series_table[series_table["StudyInstanceUID"] == study_uid]
    .sort_values(["Anatomical_Plane", "SeriesInstanceUID"])
    .reset_index(drop=True)
)

series_files: dict[str, list[Path]] = {}
series_inventory: list[dict[str, object]] = []
patient_ids: set[str] = set()
for row in study_rows.itertuples(index=False):
    series_uid = str(row.SeriesInstanceUID)
    series_dir = competition_root / "train_series" / study_uid / series_uid
    paths = sorted(
        path
        for path in series_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in {".dcm", ".dicom"}
    )
    if not paths:
        raise FileNotFoundError(f"No DICOM slices found in {series_dir}")
    header = dcmread(paths[0], stop_before_pixels=True)
    patient_id = str(getattr(header, "PatientID", ""))
    if patient_id:
        patient_ids.add(patient_id)
    series_files[series_uid] = paths
    series_inventory.append(
        {
            "SeriesInstanceUID": series_uid,
            "Plane": str(row.Anatomical_Plane),
            "Fluid sensitive": str(row.Fluid_Sensitive),
            "Fat suppression": str(row.Fat_Suppression),
            "Slices": len(paths),
            "Matrix": f"{getattr(header, 'Rows', '?')} x {getattr(header, 'Columns', '?')}",
            "Photometric": str(getattr(header, "PhotometricInterpretation", "")),
        }
    )

if len(patient_ids) != 1:
    raise ValueError("Expected one PatientID across the representative study")
patient_id = next(iter(patient_ids))
series_inventory_frame = pd.DataFrame(series_inventory)

print(f"PatientID: {patient_id}")
print(f"StudyInstanceUID: {study_uid}")
display(series_inventory_frame)

# %% [markdown]
# The next output shows a short directory tree. Each file name is a DICOM SOP
# Instance UID. A unique identifier (UID) is a value that identifies one item.
# The output shows only the first and last file name in each series.

# %%
print(f"train_series/{study_uid}/")
for record in series_inventory:
    series_uid = str(record["SeriesInstanceUID"])
    paths = series_files[series_uid]
    print(f"  {series_uid}/")
    print(f"    {paths[0].name} ... {paths[-1].name}  ({len(paths)} DICOM files)")

# %% [markdown]
# ## Order the slices by anatomical position
#
# File names identify slices. They do not give anatomical positions. The code
# calculates each slice coordinate from `ImagePositionPatient` and
# `ImageOrientationPatient`.
#
# A normal vector gives the direction perpendicular to the image plane. The
# package verifies that all normal vectors have the same direction. It also
# verifies that each coordinate is unique.
#
# If the geometry is not valid, the package uses unique `InstanceNumber` values.
# If these values are not valid, the package uses file names. The code below is
# a short example of this logic. `rsna_knee.indexing` contains the complete
# checks and error records.


# %%
def slice_geometry(dataset: Dataset) -> tuple[np.ndarray, float] | None:
    try:
        orientation = np.asarray(dataset.ImageOrientationPatient, dtype=np.float64)
        position = np.asarray(dataset.ImagePositionPatient, dtype=np.float64)
    except AttributeError:
        return None
    if orientation.shape != (6,) or position.shape != (3,):
        return None
    normal = np.cross(orientation[:3], orientation[3:])
    norm = float(np.linalg.norm(normal))
    if norm < 1e-6 or not np.isfinite([*normal, *position]).all():
        return None
    unit_normal = normal / norm
    return unit_normal, float(np.dot(unit_normal, position))


def order_series(paths: list[Path]) -> tuple[list[Path], str]:
    headers = [
        (
            path,
            dcmread(
                path,
                stop_before_pixels=True,
                specific_tags=[
                    "ImageOrientationPatient",
                    "ImagePositionPatient",
                    "InstanceNumber",
                ],
            ),
        )
        for path in paths
    ]
    geometry = [slice_geometry(dataset) for _, dataset in headers]
    if all(item is not None for item in geometry):
        normals = [item[0] for item in geometry if item is not None]
        coordinates = [item[1] for item in geometry if item is not None]
        geometry_is_consistent = all(
            float(np.dot(normals[0], normal)) >= 0.999 for normal in normals[1:]
        ) and len({round(value, 6) for value in coordinates}) == len(coordinates)
        if geometry_is_consistent:
            return (
                [
                    path
                    for _, path in sorted(
                        zip(coordinates, paths, strict=True), key=lambda item: item[0]
                    )
                ],
                "ImagePositionPatient geometry",
            )

    instances: list[object | None] = [
        getattr(dataset, "InstanceNumber", None) for _, dataset in headers
    ]
    if all(value is not None for value in instances) and len(set(instances)) == len(
        instances
    ):
        numeric_instances = [
            int(cast(str | int, value)) for value in instances if value is not None
        ]
        return (
            [
                path
                for _, path in sorted(
                    zip(numeric_instances, paths, strict=True),
                    key=lambda item: item[0],
                )
            ],
            "InstanceNumber fallback",
        )
    return sorted(paths, key=lambda path: path.name), "filename fallback"


sagittal = series_inventory_frame[
    series_inventory_frame["Plane"].str.casefold() == "sagittal"
]
visual_record = (
    (sagittal if not sagittal.empty else series_inventory_frame)
    .sort_values("Slices", ascending=False)
    .iloc[0]
)
visual_series_uid = str(visual_record["SeriesInstanceUID"])
ordered_paths, ordering_method = order_series(series_files[visual_series_uid])
print(f"Series: {visual_series_uid}")
print(f"Plane: {visual_record['Plane']}")
print(f"Ordering used: {ordering_method}")
print(f"Ordered slices: {len(ordered_paths)}")

# %% [markdown]
# `InstanceNumber` can increase while the spatial coordinate decreases. This
# does not indicate an error. DICOM geometry locates slices in the patient
# coordinate system. Instance numbers and file names do not always follow the
# anatomical direction. Thus, the package uses valid geometry as the primary
# order. It uses `InstanceNumber` only when the geometry is not valid.

# %% [markdown]
# ## Select eight slices from the volume
#
# The baseline selects integer indices at equal intervals. It includes the first
# and last slice. NumPy uses truncation to make the intermediate positions into
# integers. Truncation removes the decimal part. It does not round the value.
#
# If a series has eight or fewer usable slices, the code selects each slice one
# time. Unused model slots contain zero. A mask records which slots contain valid
# slices. A false mask value marks an unused slot or a decode error. The model
# does not include these slots in the mean.


# %%
# This example is simplified. It matches rsna_knee.preprocessing.uniform_indices.
def teaching_slice_indices(n_slices: int, n_selected: int = 8) -> tuple[int, ...]:
    if n_slices < 0 or n_selected <= 0:
        raise ValueError("n_slices must be non-negative and n_selected positive")
    if n_slices <= n_selected:
        return tuple(range(n_slices))
    return tuple(np.linspace(0, n_slices - 1, n_selected, dtype=int).tolist())


for example_length in (0, 5, 8, 9, 31):
    assert teaching_slice_indices(example_length) == uniform_indices(example_length, 8)

selected_indices = teaching_slice_indices(len(ordered_paths))
display(
    pd.DataFrame(
        {
            "Model slot": range(1, len(selected_indices) + 1),
            "Ordered slice index": selected_indices,
            "Position in series": [
                f"{index + 1}/{len(ordered_paths)}" for index in selected_indices
            ],
        }
    )
)

# %% [markdown]
# ## View the selected MRI slices
#
# The preprocessing step uses these slices from the selected series. The slices
# cover the ordered volume. They are not eight adjacent images.

# %%
figure, axes = plt.subplots(2, 4, figsize=(16, 8))
for slot, (axis, source_index) in enumerate(zip(axes.flat, selected_indices), start=1):
    path = ordered_paths[source_index]
    header = dcmread(path, stop_before_pixels=True, specific_tags=["InstanceNumber"])
    axis.imshow(decode_pixels(path), cmap="gray")
    axis.set_title(
        f"slot {slot}: {source_index + 1}/{len(ordered_paths)}\n"
        f"InstanceNumber {getattr(header, 'InstanceNumber', '?')}"
    )
    axis.axis("off")
for axis in list(axes.flat)[len(selected_indices) :]:
    axis.axis("off")
figure.suptitle(
    f"{visual_record['Plane']} series: eight slices across the volume", fontsize=16
)
figure.tight_layout()
plt.show()

# %% [markdown]
# ### Compare the anatomical planes
#
# The code shows one middle slice from each available plane. It uses the same
# study and does not scan another study.

# %%
orientation_examples = []
for plane in ("Axial", "Coronal", "Sagittal"):
    matching_series = series_inventory_frame[
        series_inventory_frame["Plane"].str.casefold() == plane.casefold()
    ]
    if matching_series.empty:
        continue
    record = matching_series.sort_values("Slices", ascending=False).iloc[0]
    plane_paths, _ = order_series(series_files[str(record["SeriesInstanceUID"])])
    middle_path = plane_paths[(len(plane_paths) - 1) // 2]
    orientation_examples.append((plane, middle_path))

figure, axes = plt.subplots(
    1,
    len(orientation_examples),
    figsize=(4 * len(orientation_examples), 4),
    squeeze=False,
)
for axis, (plane, path) in zip(axes.flat, orientation_examples, strict=True):
    axis.imshow(decode_pixels(path), cmap="gray")
    axis.set_title(plane)
    axis.axis("off")
figure.tight_layout()
plt.show()

# %% [markdown]
# ## Make a model input from one DICOM slice
#
# Pixel decoding converts stored DICOM values to image intensities. A lookup
# table (LUT) defines how to convert one value to another. The code applies the
# DICOM modality LUT when the file contains one.
#
# If there is no LUT, the code uses `RescaleSlope` and `RescaleIntercept`. The
# code inverts a `MONOCHROME1` image. Thus, larger values stay visually brighter.
#
# The code calculates the 1st and 99th percentiles from all selected slices that
# it can decode. Each slice uses the same limits for its series. The code clips
# each value to the range `[0, 1]`.
#
# Aspect ratio is the relation between image width and image height. The code
# keeps this ratio when it changes the image size. It adds zero padding to make a
# `224 x 224` image. ResNet-18 requires three image channels. Thus, the code
# makes three identical channels from the grayscale image.
#
# ImageNet is an image dataset used to initialize ResNet-18. ImageNet
# normalization uses fixed channel means and standard deviations. The code
# applies this normalization before ResNet-18 processes the image.

# %%
preprocessing_config = PreprocessingConfig(
    slice_count=int(reference["preprocessing"]["slice_count"]),
    lower_percentile=float(reference["preprocessing"]["lower_percentile"]),
    upper_percentile=float(reference["preprocessing"]["upper_percentile"]),
    image_size=int(reference["preprocessing"]["image_size"]),
)
processed_series = preprocess_series_result(ordered_paths, preprocessing_config)
volume_middle = (len(ordered_paths) - 1) / 2
valid_slot = min(
    (slot for slot, is_valid in enumerate(processed_series.mask) if is_valid),
    key=lambda slot: abs(processed_series.source_indices[slot] - volume_middle),
)
source_index = processed_series.source_indices[valid_slot]
example_path = ordered_paths[source_index]
example_dataset = dcmread(example_path)
stored_pixels = np.asarray(example_dataset.pixel_array)
decoded_pixels = decode_pixels(example_path)
lower = processed_series.lower_value
upper = processed_series.upper_value
if lower is None or upper is None:
    raise ValueError("Preprocessing did not produce percentile bounds")
scaled_pixels = (
    np.zeros_like(decoded_pixels, dtype=np.float32)
    if lower == upper
    else np.clip((decoded_pixels - lower) / (upper - lower), 0, 1).astype(np.float32)
)
resized_pixels = processed_series.pixels[valid_slot]

figure, axes = plt.subplots(1, 4, figsize=(18, 4))
stages = (
    (stored_pixels, f"Stored DICOM pixels\n{stored_pixels.dtype}"),
    (decoded_pixels, "Corrected pixel values"),
    (scaled_pixels, f"Scaled to percentile limits\n{lower:.1f} to {upper:.1f}"),
    (
        resized_pixels,
        f"Model input\n{resized_pixels.shape[0]} x {resized_pixels.shape[1]}",
    ),
)
for axis, (image, title) in zip(axes, stages, strict=True):
    axis.imshow(image, cmap="gray")
    axis.set_title(title)
    axis.axis("off")
figure.tight_layout()
plt.show()

valid_planes = torch.from_numpy(processed_series.pixels[processed_series.mask])
three_channel = valid_planes.unsqueeze(1).repeat(1, 3, 1, 1)
imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
normalized_planes = (three_channel - imagenet_mean) / imagenet_std
print(f"grayscale batch: {tuple(valid_planes.shape)}")
print(f"ResNet-18 input: {tuple(normalized_planes.shape)}")

# %% [markdown]
# ## Keep each patient in one data split
#
# Studies from the same patient are related. Data leakage occurs when related
# patient data occurs in both data splits. Leakage can make a validation score
# too high.
#
# The split uses `PatientID`. Each patient occurs in only one data split. All 58
# selected studies have known competition labels for all 12 targets. A holdout
# is the data split used for validation. The loss can ignore an unknown label.
# This subset does not contain unknown labels.

# %%
split = reference["split"]
partitions = split["partitions"]
display(
    pd.DataFrame(
        [
            {
                "Split": name.capitalize(),
                "Studies": partitions[name]["study_count"],
                "Patients": partitions[name]["patient_count"],
            }
            for name in ("train", "holdout")
        ]
    )
)
print(f"Patients shared by train and holdout: {split['patient_overlap']}")

# %%
class_balance = []
for target in targets:
    train_counts = partitions["train"]["per_target"][target]
    holdout_counts = partitions["holdout"]["per_target"][target]
    class_balance.append(
        {
            "Target": target,
            "Train positive": train_counts["positive"],
            "Train negative": train_counts["negative"],
            "Holdout positive": holdout_counts["positive"],
            "Holdout negative": holdout_counts["negative"],
        }
    )
display(pd.DataFrame(class_balance))

# %% [markdown]
# ## Make one study prediction from DICOM slices
#
# An encoder converts an input image to a feature vector. ResNet-18 processes
# each valid slice with the same encoder and weights. A feature vector is also
# called an embedding. It is a numeric description that the model learns.
#
# The model averages only valid embeddings. This operation is a masked mean.
# Empty padding slots do not change the result. Thus, a short series does not get
# extra weight from empty slots.
#
# ```text
# MRI study
#   -> one or more image series
#      -> 8 slices selected by a fixed method in each series
#         -> decode DICOM pixels, scale values, and resize to 224 x 224
#         -> process each slice with the shared ResNet-18 encoder
#      -> calculate the mean of valid slice embeddings
#   -> calculate the mean of valid series embeddings
#   -> make 12 logits and sigmoid probabilities
# ```
#
# The fixed selection method covers the full ordered series. It also makes memory
# use and run time more predictable.

# %%
model_record = reference["model"]
preprocessing = reference["preprocessing"]
training = reference["training"]
display(
    pd.DataFrame(
        [
            {"Setting": "Backbone", "Value": "ResNet-18"},
            {
                "Setting": "Initialization",
                "Value": model_record["initialization_identity"],
            },
            {
                "Setting": "Image size",
                "Value": f"{model_record['image_size']} x {model_record['image_size']}",
            },
            {"Setting": "Slices per series", "Value": preprocessing["slice_count"]},
            {
                "Setting": "Slice pooling",
                "Value": model_record["slice_to_series_pooling"],
            },
            {
                "Setting": "Series pooling",
                "Value": model_record["series_to_study_pooling"],
            },
            {"Setting": "Optimizer", "Value": "AdamW"},
            {"Setting": "Epochs", "Value": training["epochs"]},
            {"Setting": "Learning rate", "Value": training["learning_rate"]},
            {"Setting": "Weight decay", "Value": training["weight_decay"]},
            {"Setting": "Mixed precision", "Value": training["mixed_precision"]},
        ]
    )
)

# %% [markdown]
# ## Track the tensors through the model
#
# A tensor is a multidimensional array. The code uses tensors for images, masks,
# features, and outputs. The same ResNet-18 encoder processes every valid slice.
# The example removes the classification layer. Thus, the encoder makes one
# feature vector with 512 values for each slice.
#
# The slice mask has shape `[study, series, slice]`. A false value marks padding
# or a decode error. The series mask has shape `[study, series]`. Its value is
# false when a series has no valid slices.
#
# The code applies each mask before it calculates a mean. Thus, padded zeros do
# not change the study feature vector.
#
# The next cell is a short CPU example. It uses a maximum of three series from
# the selected study. The complete run uses every indexed series. The example
# loads the attached ImageNet weights. The package verifies the exact weight
# file. The example does not train the 12-output head. It shows only the tensor
# path and the tensor shapes.

# %%
# This example is simplified. The tested forward pass is in rsna_knee.training.
teaching_series_uids = [visual_series_uid] + [
    uid
    for uid in series_inventory_frame["SeriesInstanceUID"].astype(str)
    if uid != visual_series_uid
]
teaching_series_uids = teaching_series_uids[:3]
teaching_results = [
    preprocess_series_result(
        order_series(series_files[series_uid])[0], preprocessing_config
    )
    for series_uid in teaching_series_uids
]
study_pixels = torch.from_numpy(
    np.stack([result.pixels for result in teaching_results])
).unsqueeze(0)
slice_mask = torch.from_numpy(
    np.stack([result.mask for result in teaching_results])
).unsqueeze(0)
series_mask = slice_mask.any(dim=2)

weight_roots = (
    Path("/kaggle/input/datasets") / WEIGHT_DATASET_ID,
    Path("/kaggle/input") / WEIGHT_DATASET_ID.partition("/")[2],
)
weight_candidates = {
    path.resolve()
    for root in weight_roots
    if root.exists()
    for path in root.rglob("resnet18-f37072fd.pth")
    if path.is_file()
}
if len(weight_candidates) != 1:
    raise FileNotFoundError(
        f"Expected one offline ResNet-18 weight file, found {len(weight_candidates)}"
    )

torch.manual_seed(int(training["seed"]))
teaching_model, teaching_initialization = build_initialized_model(
    next(iter(weight_candidates)), targets
)
teaching_model.eval()

valid = slice_mask & series_mask.unsqueeze(-1)
flat_valid = valid.reshape(-1)
flat_pixels = study_pixels.reshape(-1, 224, 224)
valid_images = flat_pixels[flat_valid].unsqueeze(1).repeat(1, 3, 1, 1)
valid_images = (
    valid_images - teaching_model.imagenet_mean
) / teaching_model.imagenet_std

with torch.inference_mode():
    valid_slice_features = teaching_model.backbone(valid_images)
    flat_features = valid_slice_features.new_zeros(
        (flat_pixels.shape[0], teaching_model.feature_count)
    )
    flat_features[flat_valid] = valid_slice_features
    slice_features = flat_features.view(
        *study_pixels.shape[:3], teaching_model.feature_count
    )

    slice_weights = valid.unsqueeze(-1).to(slice_features.dtype)
    series_features = (slice_features * slice_weights).sum(dim=2) / (
        slice_weights.sum(dim=2).clamp_min(1)
    )

    effective_series = valid.any(dim=2) & series_mask
    series_weights = effective_series.unsqueeze(-1).to(series_features.dtype)
    study_features = (series_features * series_weights).sum(dim=1) / (
        series_weights.sum(dim=1).clamp_min(1)
    )
    logits = teaching_model.head(study_features)

display(
    pd.DataFrame(
        [
            {"Tensor": "study_pixels", "Shape": tuple(study_pixels.shape)},
            {"Tensor": "slice_mask", "Shape": tuple(slice_mask.shape)},
            {"Tensor": "valid_images", "Shape": tuple(valid_images.shape)},
            {
                "Tensor": "slice_features",
                "Shape": tuple(slice_features.shape),
            },
            {
                "Tensor": "series_features",
                "Shape": tuple(series_features.shape),
            },
            {
                "Tensor": "study_features",
                "Shape": tuple(study_features.shape),
            },
            {"Tensor": "logits", "Shape": tuple(logits.shape)},
        ]
    )
)
print(f"Verified weight SHA-256: {teaching_initialization.sha256}")

# %% [markdown]
# A logit is the model output before conversion to a probability. The output
# head makes one logit for each target. A sigmoid function converts each logit to
# a probability.
#
# This is a multi-label problem. Multi-label means that one study can have more
# than one positive target.

# %% [markdown]
# ## Calculate the multi-label loss
#
# Binary cross-entropy (BCE) is a loss for one binary target. Each study has 12
# binary targets. Thus, the loss calculation starts with 12 separate BCE values.
#
# The known-label mask selects the values for the mean. The reference group has
# all 12 labels for every selected study. The training code can also process a
# row that has unknown labels.

# %%
# This example uses synthetic labels. They make the mask operation easy to see.
example_logits = torch.linspace(-1.5, 1.5, len(targets)).view(1, -1)
example_logits.requires_grad_()
example_targets = torch.tensor(
    [[float(index % 3 == 0) for index in range(len(targets))]]
)
known_mask = torch.ones_like(example_targets, dtype=torch.bool)
known_mask[:, -2:] = False

elementwise_bce = F.binary_cross_entropy_with_logits(
    example_logits, example_targets, reduction="none"
)
teaching_loss = elementwise_bce[known_mask].mean()
production_loss = masked_binary_cross_entropy(
    example_logits, example_targets, known_mask
)
assert torch.allclose(teaching_loss, production_loss)
teaching_loss.backward()  # type: ignore[no-untyped-call]

display(
    pd.DataFrame(
        {
            "Target": targets,
            "Logit": example_logits.detach().squeeze(0),
            "Binary label": example_targets.squeeze(0),
            "Known": known_mask.squeeze(0),
            "Element BCE": elementwise_bce.detach().squeeze(0),
        }
    )
)
print(f"Mean BCE over known targets: {teaching_loss.item():.4f}")
gradient = example_logits.grad
assert gradient is not None
print(
    "Gradient for the two unknown targets:",
    gradient[0, -2:].tolist(),
)

# %% [markdown]
# ## Reference run results
#
# The receiver operating characteristic area under the curve (ROC AUC) measures
# how the model ranks positive and negative examples. The competition metric is
# the mean of the ROC AUC for the 12 targets.
#
# The table gives the holdout class counts. Each target has data from only four
# patients. With four patients, only a few positive and negative pairs determine
# the AUC. An AUC of 0 does not prove that the model always fails. An AUC of 1
# does not prove that the model is perfect.

# %%
results = reference["results"]
result_rows = []
for target in targets:
    target_result = results["per_target"][target]
    result_rows.append(
        {
            "Target": target,
            "Holdout positive": target_result["positive"],
            "Holdout negative": target_result["negative"],
            "ROC AUC": target_result["roc_auc"],
        }
    )
display(pd.DataFrame(result_rows))
print(f"Observed 12-target macro ROC AUC: {results['macro_auc']:.4f}")

# %% [markdown]
# ## Why four validation patients are too few
#
# The run uses bootstrap resampling to examine uncertainty. For each resample,
# the method selects four patients with replacement. Thus, it can select the
# same patient more than one time. The run makes 10,000 bootstrap resamples.
#
# ROC AUC is not defined when a resample has only one class for a target. The
# macro AUC needs a valid AUC for all 12 targets. Thus, one undefined target
# makes the macro AUC undefined for that resample.

# %%
limitations = reference["limitations"]
defined = limitations["defined_macro_replicate_count"]
undefined = limitations["undefined_macro_replicate_count"]
total = limitations["bootstrap_replicate_count"]
display(
    pd.DataFrame(
        [
            {
                "Quantity": "Holdout patients",
                "Value": limitations["holdout_patient_count"],
            },
            {"Quantity": "Observed macro ROC AUC", "Value": results["macro_auc"]},
            {"Quantity": "Bootstrap resamples", "Value": total},
            {
                "Quantity": "Resamples with a defined macro AUC",
                "Value": f"{defined} ({defined / total:.2%})",
            },
            {
                "Quantity": "Resamples with an undefined macro AUC",
                "Value": f"{undefined} ({undefined / total:.2%})",
            },
            {"Quantity": "Evidence for model selection", "Value": "Inadequate"},
        ]
    )
)

# %% [markdown]
# In this run, 9,065 of 10,000 resamples (90.65%) had no macro AUC. The other 935
# resamples gave the same value. This does not show that the score is precise.
# The small holdout caused this result. The calculation also excludes all
# resamples that have an undefined macro AUC.
#
# The pipeline completed training, evaluation, checkpoint reload, and submission
# validation. The value 0.6111 describes only these four holdout patients. It
# does not estimate expected leaderboard or clinical performance. Thus, the
# observed macro AUC is not reliable enough for model selection.

# %% [markdown]
# ## Run time
#
# The recorded 5.1 minutes includes ten training epochs and their holdout
# evaluations on a Kaggle Tesla T4. It does not include the complete notebook
# run. The previous complete notebook run took about 147 minutes.
#
# A DICOM header index records file paths and DICOM attributes. During the
# complete run, the package rebuilt and verified this index. It also reloaded
# the checkpoint and made test predictions. Then, it validated the submission.
# Index building and DICOM decoding used most of the complete run time.

# %%
runtime = reference["runtime"]
checkpoint = reference["checkpoint"]
display(
    pd.DataFrame(
        [
            {"Measurement": "GPU", "Value": runtime["gpu"]},
            {
                "Measurement": "Training and holdout evaluation only",
                "Value": f"{runtime['complete_run_seconds'] / 60:.1f} minutes",
            },
            {
                "Measurement": "Previous complete notebook run",
                "Value": "about 147 minutes",
            },
            {
                "Measurement": "Training DICOM decode time",
                "Value": f"{runtime['training_decode_seconds']:.1f} seconds",
            },
            {
                "Measurement": "Training forward and backward time",
                "Value": f"{runtime['training_forward_backward_seconds']:.1f} seconds",
            },
            {
                "Measurement": "Peak CUDA memory",
                "Value": f"{runtime['peak_cuda_memory_bytes'] / (1024**3):.2f} GiB",
            },
            {
                "Measurement": "Valid training slices each second",
                "Value": f"{runtime['training_slices_per_second']:.1f}",
            },
            {
                "Measurement": "Training decode errors",
                "Value": len(runtime["training_decode_errors"]),
            },
            {
                "Measurement": "Best checkpoint epoch",
                "Value": checkpoint["best"]["epoch"],
            },
        ]
    )
)

# %% [markdown]
# ## Reproducibility details
#
# Reproducibility means that another run can use the same source and settings.
# The teaching cells use short examples. The package does the complete run with
# the exact source and configuration bytes.
#
# The notebook does not use internet access. The package verifies the hashes for
# the wheel, ImageNet weights, and configuration. A split fingerprint identifies
# the exact patient split. The package also verifies the saved checkpoint after
# it reloads the checkpoint.
#
# A dirty working tree has uncommitted source changes. The table reports if the
# working tree was dirty when the build process made the notebook.

# %%
config_bytes = base64.b64decode("__RSNA_KNEE_BASELINE_CONFIG__")
config_sha256 = hashlib.sha256(config_bytes).hexdigest()
if config_sha256 != "__RSNA_KNEE_CONFIG_SHA256__":
    raise RuntimeError("Baseline configuration hash mismatch")

display(
    pd.DataFrame(
        [
            {"Item": "Git commit", "Value": "__RSNA_KNEE_GIT_COMMIT__"},
            {
                "Item": "Working tree was dirty at build",
                "Value": json.loads("__RSNA_KNEE_GIT_DIRTY__"),
            },
            {"Item": "Project wheel SHA-256", "Value": WHEEL_SHA256},
            {"Item": "Configuration SHA-256", "Value": config_sha256},
            {
                "Item": "Jupytext source SHA-256",
                "Value": "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__",
            },
            {
                "Item": "ImageNet weight SHA-256",
                "Value": model_record["initialization_artifact_sha256"],
            },
            {"Item": "Patient split fingerprint", "Value": split["fingerprint"]},
            {
                "Item": "Recorded best checkpoint SHA-256",
                "Value": checkpoint["best"]["sha256"],
            },
            {
                "Item": "Checkpoint reload verification passed",
                "Value": checkpoint["reload_predictions_within_tolerance"],
            },
            {"Item": "Kaggle kernel", "Value": KERNEL_ID},
            {"Item": "Internet enabled", "Value": False},
        ]
    )
)

# %% [markdown]
# ## Run the complete pipeline and validate test predictions
#
# The final cell repeats the complete run with the attached competition data.
# The notebook contains the fixed configuration. The package checks the
# configuration before use.
#
# The package builds the index and preprocesses the DICOM slices. It trains the
# model and evaluates the holdout data. Then, it reloads the checkpoint. It makes
# test predictions and validates the submission.
#
# The temporary index contains row identifiers. A run artifact is a file that a
# run makes. Checkpoints, predictions, and the submission are run artifacts. The
# cell writes these files under `/kaggle/working`. The package removes the files
# before the notebook shows the summary.

# %%
reproduction = reproduce_reference_baseline(
    input_root=Path("/kaggle/input"),
    work_parent=Path("/kaggle/working"),
    config_bytes=config_bytes,
    approved_gpu=GPU_APPROVED,
    source_provenance={
        "git_commit": "__RSNA_KNEE_GIT_COMMIT__",
        "git_dirty": json.loads("__RSNA_KNEE_GIT_DIRTY__"),
        "jupytext_source_sha256": "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__",
        "package_archive_sha256": WHEEL_SHA256,
        "baseline_config_sha256": "__RSNA_KNEE_CONFIG_SHA256__",
        "weight_dataset_id": WEIGHT_DATASET_ID,
    },
    kernel_id=KERNEL_ID,
)
reproduction["notebook_elapsed_minutes"] = round(
    (time.perf_counter() - notebook_started) / 60, 1
)

labels = {
    "training_outputs_verified": "Training output verification passed",
    "patient_split_reproduced": "Patient split matched the reference",
    "best_checkpoint_used_for_test_predictions": "Test predictions used the best checkpoint",
    "submission_valid": "Submission schema valid",
    "submission_rows": "Submission rows",
    "test_predictions": "Test predictions",
    "fallback_predictions": "Fallback predictions",
    "decode_failures": "Inference decode failures",
    "temporary_files_removed": "Temporary file removal passed",
    "notebook_elapsed_minutes": "Complete notebook run time (minutes)",
}
display(
    pd.DataFrame(
        [{"Check": labels[key], "Result": value} for key, value in reproduction.items()]
    )
)

# %% [markdown]
# ## Limits of the reference run
#
# The notebook shows that the study-level MRI pipeline can run from input to
# submission. The data split keeps patients separate. The pipeline can process
# different numbers of slices and series. It reloads the saved checkpoint for
# inference. It also compares the submission rows and columns with the
# competition data.
#
# This run does not show that ResNet-18 is the best architecture. It also does
# not show how the model will perform on new data. More independent validation
# patients are necessary for these conclusions. This run does not change the
# split, tune the model, or add supervision.
