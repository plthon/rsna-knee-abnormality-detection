from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .config import PreprocessingConfig
from .dicom import decode_pixels


def uniform_indices(length: int, count: int) -> tuple[int, ...]:
    if length < 0 or count <= 0:
        raise ValueError("length must be non-negative and count must be positive")
    if length <= count:
        return tuple(range(length))
    return tuple(np.linspace(0, length - 1, count, dtype=int).tolist())


def preprocess_series(
    paths: Sequence[str | Path], config: PreprocessingConfig
) -> tuple[np.ndarray, np.ndarray]:
    indices = uniform_indices(len(paths), config.slice_count)
    if not indices:
        raise ValueError("Cannot preprocess an empty series")

    volume = np.stack([decode_pixels(paths[index]) for index in indices])
    lower, upper = np.percentile(
        volume, (config.lower_percentile, config.upper_percentile)
    )
    if lower == upper:
        normalized = np.zeros_like(volume, dtype=np.float32)
    else:
        normalized = np.clip((volume - lower) / (upper - lower), 0, 1).astype(
            np.float32
        )

    padded = np.zeros((config.slice_count, *normalized.shape[1:]), dtype=np.float32)
    padded[: len(indices)] = normalized
    mask = np.arange(config.slice_count) < len(indices)
    return padded, mask
