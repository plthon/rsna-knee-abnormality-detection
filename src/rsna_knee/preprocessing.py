from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from .config import PreprocessingConfig
from .dicom import decode_pixels_with_spacing


@dataclass(frozen=True, slots=True)
class PreprocessingFailure:
    path: Path
    code: str
    exception_type: str


@dataclass(frozen=True, slots=True)
class SpatialEvidence:
    source_index: int
    mode: str
    fallback_reason: str | None
    crop_shape: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class PreprocessedSeries:
    pixels: np.ndarray
    mask: np.ndarray
    source_indices: tuple[int, ...]
    failures: tuple[PreprocessingFailure, ...]
    lower_value: float | None
    upper_value: float | None
    spatial_evidence: tuple[SpatialEvidence, ...] = ()


def uniform_indices(length: int, count: int) -> tuple[int, ...]:
    if length < 0 or count <= 0:
        raise ValueError("length must be non-negative and count must be positive")
    if length <= count:
        return tuple(range(length))
    return tuple(np.linspace(0, length - 1, count, dtype=int).tolist())


def preprocess_series(
    paths: Sequence[str | Path], config: PreprocessingConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a series for either training or inference.

    Isolated unreadable slices are represented by false mask entries. Callers that
    need the corresponding structured failures can use ``preprocess_series_result``.
    """
    result = preprocess_series_result(paths, config)
    if not result.mask.any():
        raise ValueError("Cannot preprocess a series with no decodable slices")
    return result.pixels, result.mask


def preprocess_series_result(
    paths: Sequence[str | Path], config: PreprocessingConfig
) -> PreprocessedSeries:
    """Apply the shared DICOM-to-tensor contract and retain compact failures."""
    indices = uniform_indices(len(paths), config.slice_count)
    if not indices:
        raise ValueError("Cannot preprocess an empty series")

    decoded: list[tuple[int, np.ndarray]] = []
    failures: list[PreprocessingFailure] = []
    spacings: dict[int, object] = {}
    for slot, source_index in enumerate(indices):
        path = Path(paths[source_index])
        try:
            pixels, spacing = decode_pixels_with_spacing(path)
            spacings[slot] = spacing
            if pixels.ndim != 2 or not pixels.size:
                raise ValueError(
                    "Only nonempty single-frame grayscale images are supported"
                )
            if not np.isfinite(pixels).all():
                raise ValueError("Decoded pixels contain non-finite values")
            decoded.append((slot, pixels))
        except Exception as error:  # noqa: BLE001 - DICOM failures are data records
            failures.append(
                PreprocessingFailure(
                    path=path,
                    code="pixel_decode_failed",
                    exception_type=type(error).__name__,
                )
            )

    output = np.zeros(
        (config.slice_count, config.image_size, config.image_size), dtype=np.float32
    )
    mask = np.zeros(config.slice_count, dtype=np.bool_)
    if not decoded:
        return PreprocessedSeries(
            pixels=output,
            mask=mask,
            source_indices=indices,
            failures=tuple(failures),
            lower_value=None,
            upper_value=None,
        )

    values = np.concatenate([pixels.reshape(-1) for _, pixels in decoded])
    lower, upper = np.percentile(
        values, (config.lower_percentile, config.upper_percentile)
    )
    if lower == upper:
        normalized = [np.zeros_like(pixels, dtype=np.float32) for _, pixels in decoded]
    else:
        normalized = [
            np.clip((pixels - lower) / (upper - lower), 0, 1).astype(np.float32)
            for _, pixels in decoded
        ]

    evidence: list[SpatialEvidence] = []
    for (slot, _), image in zip(decoded, normalized, strict=True):
        shape, reason = (None, None)
        if config.spatial_mode == "physical_crop":
            shape, reason = physical_crop_shape(spacings[slot], config.physical_fov_mm)
        if shape is not None:
            output[slot] = _bilinear_resize(
                centered_crop_pad(image, shape), config.image_size, config.image_size
            )
        else:
            output[slot] = _resize_and_pad(image, config.image_size)
        evidence.append(
            SpatialEvidence(
                indices[slot],
                "physical_crop" if shape else "whole_image",
                reason,
                shape,
            )
        )
        mask[slot] = True
    return PreprocessedSeries(
        pixels=output,
        mask=mask,
        source_indices=indices,
        failures=tuple(failures),
        lower_value=float(lower),
        upper_value=float(upper),
        spatial_evidence=tuple(evidence),
    )


def _resize_and_pad(image: np.ndarray, image_size: int) -> np.ndarray:
    height, width = image.shape
    scale = min(image_size / height, image_size / width)
    resized_height = max(1, min(image_size, round(height * scale)))
    resized_width = max(1, min(image_size, round(width * scale)))
    resized = _bilinear_resize(image, resized_height, resized_width)

    padded = np.zeros((image_size, image_size), dtype=np.float32)
    top = (image_size - resized_height) // 2
    left = (image_size - resized_width) // 2
    padded[top : top + resized_height, left : left + resized_width] = resized
    return padded


def _bilinear_resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    source_height, source_width = image.shape
    if (source_height, source_width) == (height, width):
        return image.astype(np.float32, copy=False)

    y = np.linspace(0, source_height - 1, height)
    x = np.linspace(0, source_width - 1, width)
    y0 = np.floor(y).astype(np.intp)
    x0 = np.floor(x).astype(np.intp)
    y1 = np.minimum(y0 + 1, source_height - 1)
    x1 = np.minimum(x0 + 1, source_width - 1)
    y_weight = (y - y0)[:, None]
    x_weight = (x - x0)[None, :]

    top = image[y0[:, None], x0[None, :]] * (1 - x_weight)
    top += image[y0[:, None], x1[None, :]] * x_weight
    bottom = image[y1[:, None], x0[None, :]] * (1 - x_weight)
    bottom += image[y1[:, None], x1[None, :]] * x_weight
    return cast(
        np.ndarray, (top * (1 - y_weight) + bottom * y_weight).astype(np.float32)
    )


def physical_crop_shape(
    spacing: object, fov_mm: float = 130.0
) -> tuple[tuple[int, int] | None, str | None]:
    """DICOM row/column spacing; Python round uses deterministic ties-to-even."""
    if spacing is None:
        return None, "missing_pixel_spacing"
    try:
        if isinstance(spacing, (str, bytes)):
            raise TypeError("Expected two spacing components")
        components: tuple[object, ...] = tuple(spacing)  # type: ignore[arg-type]
        if len(components) != 2:
            raise ValueError("Expected two spacing components")
        row, column = (float(str(value)) for value in components)
    except (TypeError, ValueError, OverflowError):
        return None, "malformed_pixel_spacing"
    if not np.isfinite([row, column]).all():
        return None, "nonfinite_pixel_spacing"
    if row <= 0 or column <= 0:
        return None, "nonpositive_pixel_spacing"
    dimensions = [fov_mm / row, fov_mm / column]
    if not np.isfinite(dimensions).all():
        raise ValueError("Physical crop dimensions exceed numeric limits")
    shape = (round(dimensions[0]), round(dimensions[1]))
    if min(shape) < 1:
        raise ValueError("Physical FOV rounds to an empty pixel crop")
    return shape, None


def centered_crop_pad(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Center crop/pad, assigning an unavoidable odd extra pixel bottom/right."""
    if min(shape) <= 0:
        raise ValueError("Crop dimensions must be positive")
    height, width = image.shape
    crop_h, crop_w = shape
    top, left = max(0, (height - crop_h) // 2), max(0, (width - crop_w) // 2)
    cropped = image[top : top + min(height, crop_h), left : left + min(width, crop_w)]
    pad_h, pad_w = max(0, crop_h - height), max(0, crop_w - width)
    return np.pad(
        cropped,
        ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
        constant_values=0,
    )
