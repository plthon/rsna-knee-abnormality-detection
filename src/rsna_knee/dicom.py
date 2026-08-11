from pathlib import Path

import numpy as np
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.pixels.processing import apply_modality_lut


def decode_pixels(path: str | Path) -> np.ndarray:
    dataset = dcmread(path)
    pixels = _apply_modality_transform(dataset)
    if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = pixels.max() + pixels.min() - pixels
    output = np.asarray(pixels, dtype=np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Decoded pixels contain non-finite values")
    return output


def _apply_modality_transform(dataset: Dataset) -> np.ndarray:
    stored_pixels = dataset.pixel_array
    if "ModalityLUTSequence" in dataset:
        return np.asarray(apply_modality_lut(stored_pixels, dataset), dtype=np.float64)

    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    if not np.isfinite([slope, intercept]).all():
        raise ValueError("DICOM rescale metadata must be finite")
    return np.asarray(stored_pixels, dtype=np.float64) * slope + intercept
