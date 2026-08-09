# %% [markdown]
# # RSNA Knee Synthetic Pipeline
#
# Private, offline, CPU-only verification of the same synthetic DICOM pipeline
# exercised by the local package tests.

# %%
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
dataset_root = Path("/kaggle/input/datasets") / "__RSNA_KNEE_DATASET_ID__"
wheel_path = dataset_root / "__RSNA_KNEE_WHEEL_FILENAME__"
wheel_digest = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
if wheel_digest != "__RSNA_KNEE_WHEEL_SHA256__":
    raise ValueError("Mounted wheel SHA-256 does not match build provenance")
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

# %%
import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    SecondaryCaptureImageStorage,
    generate_uid,
)

from rsna_knee import load_config, preprocess_series
from rsna_knee.provenance import create_run_id, write_run_manifest


def write_dicom(path: Path, pixels: np.ndarray) -> None:
    sop_instance_uid = generate_uid()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dataset = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = sop_instance_uid
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.RescaleSlope = 1
    dataset.RescaleIntercept = 0
    dataset.PixelData = pixels.astype(np.uint16).tobytes()
    dataset.save_as(path, enforce_file_format=True)


# %%
config = load_config(dataset_root / "kaggle-synthetic.toml")
config.paths.input_root.mkdir(parents=True, exist_ok=True)
config.paths.work_root.mkdir(parents=True, exist_ok=True)

base_pixels = np.array([[0, 1], [2, 3]], dtype=np.uint16)
for index in range(10):
    write_dicom(
        config.paths.input_root / f"{index:02}.dcm",
        base_pixels + index,
    )

normalized, mask = preprocess_series(
    sorted(config.paths.input_root.glob("*.dcm")),
    config.preprocessing,
)
assert normalized.shape == (4, 2, 2)
assert np.isfinite(normalized).all()
assert normalized.min() == 0
assert normalized.max() == 1
assert mask.all()

output_path = config.paths.work_root / "synthetic-output.npz"
np.savez_compressed(output_path, normalized=normalized, mask=mask)
print(output_path)

# %%
kernel_run_id = os.environ.get("KAGGLE_KERNEL_RUN_ID")
kaggle_image_commit_path = Path("/etc/git_commit")
kaggle_image_git_commit = (
    kaggle_image_commit_path.read_text().strip()
    if kaggle_image_commit_path.is_file()
    else None
)
manifest_path = write_run_manifest(
    config.paths.work_root / "run-manifest.json",
    run_id=create_run_id("synthetic", external_id=kernel_run_id),
    source={
        "git_commit": "__RSNA_KNEE_GIT_COMMIT__",
        "wheel": {
            "filename": "__RSNA_KNEE_WHEEL_FILENAME__",
            "sha256": "__RSNA_KNEE_WHEEL_SHA256__",
        },
    },
    configuration={
        "config_sha256": "__RSNA_KNEE_CONFIG_SHA256__",
        "paths": {
            "mode": config.paths.mode,
            "input_root": str(config.paths.input_root),
            "work_root": str(config.paths.work_root),
        },
        "preprocessing": {
            "slice_count": config.preprocessing.slice_count,
            "lower_percentile": config.preprocessing.lower_percentile,
            "upper_percentile": config.preprocessing.upper_percentile,
        },
    },
    notebook={
        "dataset_id": "__RSNA_KNEE_DATASET_ID__",
        "enable_gpu": json.loads("__RSNA_KNEE_GPU_ENABLED__"),
        "enable_internet": False,
        "jupytext_source_sha256": "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__",
        "kaggle_image_git_commit": kaggle_image_git_commit,
        "kaggle_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        "kernel_id": "__RSNA_KNEE_KERNEL_ID__",
        "kernel_run_id": kernel_run_id,
    },
    artifacts=[output_path],
    artifact_root=config.paths.work_root,
)
print(manifest_path)
