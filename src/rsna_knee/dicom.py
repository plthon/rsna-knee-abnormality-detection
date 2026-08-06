from pathlib import Path

import numpy as np
from pydicom import dcmread
from pydicom.pixels.processing import apply_modality_lut


def decode_pixels(path: str | Path) -> np.ndarray:
    dataset = dcmread(path)
    pixels = np.asarray(
        apply_modality_lut(dataset.pixel_array, dataset), dtype=np.float32
    )
    if dataset.PhotometricInterpretation == "MONOCHROME1":
        pixels = pixels.max() + pixels.min() - pixels
    return pixels
