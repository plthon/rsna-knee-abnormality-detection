import struct
import time
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, cast

import numpy as np
from pydicom import dcmread
from pydicom.dataset import Dataset

from .config import PreprocessingConfig
from .dicom import decode_pixels
from .preprocessing import preprocess_series, uniform_indices

_DICOM_SUFFIXES = {".dcm", ".dicom"}
_REPRESENTATIVE_SERIES_LIMIT = 4
_REPRESENTATIVE_CANDIDATE_LIMIT = 24
_MAX_SLICES_PER_REPRESENTATIVE_SERIES = 256


@dataclass(frozen=True, slots=True)
class SeriesInventory:
    path: Path
    slice_count: int
    image_root: str | None
    study_id: str | None
    series_id: str | None
    sample_image_id: str | None

    @property
    def relationships_valid(self) -> bool:
        return all(
            value not in {None, ""}
            for value in (
                self.image_root,
                self.study_id,
                self.series_id,
                self.sample_image_id,
            )
        )


@dataclass(frozen=True, slots=True)
class TransformEvidence:
    observed_in_headers: int
    tested_files: int
    successful_files: int

    @property
    def verified(self) -> bool:
        return self.tested_files > 0 and self.successful_files == self.tested_files

    def as_json(self) -> dict[str, int]:
        return {
            "observed_in_headers": self.observed_in_headers,
            "successful_files": self.successful_files,
            "tested_files": self.tested_files,
        }


@dataclass(frozen=True, slots=True)
class PreprocessingSummary:
    all_finite: bool
    decode_failures: list[str]
    decoded_series: int
    deterministic_sampling: bool
    monochrome1: TransformEvidence
    normalized_range: list[float]
    rescale: TransformEvidence

    def as_json(self) -> dict[str, Any]:
        return {
            "all_finite": self.all_finite,
            "decode_failures": self.decode_failures,
            "decoded_series": self.decoded_series,
            "deterministic_sampling": self.deterministic_sampling,
            "monochrome1_tested": self.monochrome1.verified,
            "normalized_range": self.normalized_range,
            "rescale_tested": self.rescale.verified,
            "transform_evidence": {
                "monochrome1": self.monochrome1.as_json(),
                "rescale": self.rescale.as_json(),
            },
        }


@dataclass(frozen=True, slots=True)
class ImagingInspection:
    dicom: dict[str, Any]
    preprocessing: PreprocessingSummary
    visualization_paths: list[str]
    visualized_planes: list[str]
    visualized_study_ids: list[str]
    geometry_available: bool
    ordered_series_count: int
    header_seconds: float
    preprocessing_seconds: float


@dataclass(frozen=True, slots=True)
class _Header:
    path: Path
    dataset: Dataset


@dataclass(frozen=True, slots=True)
class _SeriesProbe:
    inventory: SeriesInventory
    plane: str


@dataclass(frozen=True, slots=True)
class _OrderedSeries:
    inventory: SeriesInventory
    headers: list[_Header]
    ordered_paths: list[Path]
    selected_method: str
    plane: str
    comparison: dict[str, Any]


def inspect_imaging_sample(
    series: list[SeriesInventory], root: Path, artifact_dir: Path
) -> ImagingInspection:
    header_started = time.perf_counter()
    selected_series, probe_failures, discovered_planes = _select_representative_series(
        series, root
    )
    ordered_candidates, header_failures = _inspect_series_ordering(
        selected_series, root
    )
    header_seconds = time.perf_counter() - header_started

    preprocessing_started = time.perf_counter()
    (
        preprocessing,
        visualization_paths,
        visualized_series,
        attempted_series,
    ) = _exercise_preprocessing(ordered_candidates, discovered_planes, artifact_dir)
    preprocessing_seconds = time.perf_counter() - preprocessing_started

    visualized_planes = _unique(
        series_item.plane
        for series_item in visualized_series
        if series_item.plane != "unknown"
    )
    visualized_study_ids = _unique(
        series_item.inventory.study_id
        for series_item in visualized_series
        if series_item.inventory.study_id is not None
    )
    return ImagingInspection(
        dicom=_summarize_dicom(
            series, attempted_series, [*probe_failures, *header_failures], root
        ),
        preprocessing=preprocessing,
        visualization_paths=visualization_paths,
        visualized_planes=visualized_planes,
        visualized_study_ids=visualized_study_ids,
        geometry_available=any(
            series_item.selected_method == "geometry"
            for series_item in attempted_series
        ),
        ordered_series_count=len(attempted_series),
        header_seconds=header_seconds,
        preprocessing_seconds=preprocessing_seconds,
    )


def _select_representative_series(
    series: list[SeriesInventory], root: Path
) -> tuple[list[SeriesInventory], list[str], set[str]]:
    if not series:
        return [], [], set()
    ordered = sorted(series, key=lambda item: item.path.as_posix())

    probes: list[_SeriesProbe] = []
    failures: list[str] = []
    for candidate in ordered:
        first_path = _first_dicom_path(candidate.path)
        if first_path is None:
            continue
        try:
            dataset = dcmread(first_path, stop_before_pixels=True)
        except Exception as error:  # noqa: BLE001 - record heterogeneous DICOM failures
            failures.append(
                f"{first_path.relative_to(root).as_posix()}: {type(error).__name__}"
            )
            continue
        probes.append(_SeriesProbe(inventory=candidate, plane=_plane(dataset)))

    ranked_probes = _round_robin_by_plane(probes)
    discovered_planes = {probe.plane for probe in probes if probe.plane != "unknown"}
    return (
        [probe.inventory for probe in ranked_probes[:_REPRESENTATIVE_CANDIDATE_LIMIT]],
        failures,
        discovered_planes,
    )


def _round_robin_by_plane(probes: list[_SeriesProbe]) -> list[_SeriesProbe]:
    planes = _unique(probe.plane for probe in probes)
    return _round_robin(
        [
            _round_robin_by_study([probe for probe in probes if probe.plane == plane])
            for plane in planes
        ]
    )


def _round_robin_by_study(probes: list[_SeriesProbe]) -> list[_SeriesProbe]:
    studies = _unique(
        probe.inventory.study_id or probe.inventory.path.parent.name for probe in probes
    )
    return _round_robin(
        [
            [
                probe
                for probe in probes
                if (probe.inventory.study_id or probe.inventory.path.parent.name)
                == study
            ]
            for study in studies
        ]
    )


def _round_robin(groups: list[list[_SeriesProbe]]) -> list[_SeriesProbe]:
    ranked: list[_SeriesProbe] = []
    for index in range(max((len(group) for group in groups), default=0)):
        ranked.extend(group[index] for group in groups if index < len(group))
    return ranked


def _first_dicom_path(series_path: Path) -> Path | None:
    return next(
        (
            path
            for path in sorted(series_path.iterdir())
            if path.is_file() and path.suffix.casefold() in _DICOM_SUFFIXES
        ),
        None,
    )


def _inspect_series_ordering(
    series_items: list[SeriesInventory], root: Path
) -> tuple[list[_OrderedSeries], list[str]]:
    ordered_series: list[_OrderedSeries] = []
    failures: list[str] = []
    for series in series_items:
        paths = sorted(
            path
            for path in series.path.iterdir()
            if path.is_file() and path.suffix.casefold() in _DICOM_SUFFIXES
        )[:_MAX_SLICES_PER_REPRESENTATIVE_SERIES]
        headers: list[_Header] = []
        for path in paths:
            try:
                headers.append(
                    _Header(path=path, dataset=dcmread(path, stop_before_pixels=True))
                )
            except Exception as error:  # noqa: BLE001 - record heterogeneous DICOM failures
                failures.append(
                    f"{path.relative_to(root).as_posix()}: {type(error).__name__}"
                )
        if headers:
            ordered_series.append(_order_series(series, headers, root))
    return ordered_series, failures


def _order_series(
    series: SeriesInventory, headers: list[_Header], root: Path
) -> _OrderedSeries:
    filename_order = sorted(headers, key=lambda header: header.path.name)
    instance_order = _instance_order(headers)
    geometry_order = _geometry_order(headers)

    if geometry_order is not None:
        selected_method = "geometry"
        selected_order = geometry_order
    elif instance_order is not None:
        selected_method = "instance_number"
        selected_order = instance_order
    else:
        selected_method = "filename"
        selected_order = filename_order

    plane = _plane(headers[0].dataset)
    comparison: dict[str, Any] = {
        "filename_agrees_with_geometry": (
            _same_order(filename_order, geometry_order)
            if geometry_order is not None
            else None
        ),
        "geometry_agrees_with_instance": (
            _same_order(geometry_order, instance_order)
            if geometry_order is not None and instance_order is not None
            else None
        ),
        "orders": {
            "filename": [header.path.name for header in filename_order],
            "geometry": (
                [header.path.name for header in geometry_order]
                if geometry_order is not None
                else None
            ),
            "instance_number": (
                [header.path.name for header in instance_order]
                if instance_order is not None
                else None
            ),
        },
        "plane": plane,
        "selected_method": selected_method,
        "series": series.path.relative_to(root).as_posix(),
        "slice_count": len(headers),
    }
    return _OrderedSeries(
        inventory=series,
        headers=headers,
        ordered_paths=[header.path for header in selected_order],
        selected_method=selected_method,
        plane=plane,
        comparison=comparison,
    )


def _instance_order(headers: list[_Header]) -> list[_Header] | None:
    values = [
        _integer_attribute(header.dataset, "InstanceNumber") for header in headers
    ]
    if any(value is None for value in values) or len(set(values)) != len(values):
        return None
    return [
        header
        for _, header in sorted(
            zip(cast(list[int], values), headers, strict=True), key=lambda pair: pair[0]
        )
    ]


def _geometry_order(headers: list[_Header]) -> list[_Header] | None:
    coordinates: list[float] = []
    for header in headers:
        orientation = _float_sequence(header.dataset, "ImageOrientationPatient", 6)
        position = _float_sequence(header.dataset, "ImagePositionPatient", 3)
        if orientation is None or position is None:
            return None
        row = np.asarray(orientation[:3], dtype=np.float64)
        column = np.asarray(orientation[3:], dtype=np.float64)
        normal = np.cross(row, column)
        if not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-6:
            return None
        coordinates.append(float(np.dot(normal, np.asarray(position))))
    rounded_coordinates = [round(value, 6) for value in coordinates]
    if len(set(rounded_coordinates)) != len(rounded_coordinates):
        return None
    return [
        header
        for _, header in sorted(
            zip(coordinates, headers, strict=True), key=lambda pair: pair[0]
        )
    ]


def _same_order(first: Sequence[_Header], second: Sequence[_Header]) -> bool:
    return [header.path for header in first] == [header.path for header in second]


def _plane(dataset: Dataset) -> str:
    orientation = _float_sequence(dataset, "ImageOrientationPatient", 6)
    if orientation is None:
        return "unknown"
    normal = np.abs(
        np.cross(
            np.asarray(orientation[:3], dtype=np.float64),
            np.asarray(orientation[3:], dtype=np.float64),
        )
    )
    axis = int(np.argmax(normal))
    return ("sagittal", "coronal", "axial")[axis]


def _float_sequence(
    dataset: Dataset, attribute: str, expected_length: int
) -> tuple[float, ...] | None:
    value: object = getattr(dataset, attribute, None)
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return result if len(result) == expected_length else None


def _integer_attribute(dataset: Dataset, attribute: str) -> int | None:
    value: object = getattr(dataset, attribute, None)
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return None


def _float_attribute(dataset: Dataset, attribute: str, default: float) -> float:
    value: object = getattr(dataset, attribute, default)
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return default


def _exercise_preprocessing(
    ordered_series: list[_OrderedSeries],
    required_planes: set[str],
    artifact_dir: Path,
) -> tuple[
    PreprocessingSummary,
    list[str],
    list[_OrderedSeries],
    list[_OrderedSeries],
]:
    config = PreprocessingConfig(
        slice_count=8,
        lower_percentile=1,
        upper_percentile=99,
    )
    finite_results: list[bool] = []
    minima: list[float] = []
    maxima: list[float] = []
    visualization_paths: list[str] = []
    visualized_series: list[_OrderedSeries] = []
    attempted_series: list[_OrderedSeries] = []
    decode_failures: list[str] = []
    decoded_series = 0
    deterministic_sampling = True
    monochrome_observed = 0
    monochrome_tested = 0
    monochrome_successful = 0
    rescale_observed = 0
    rescale_tested = 0
    rescale_successful = 0

    for index, series in enumerate(ordered_series, start=1):
        visualized_planes = {item.plane for item in visualized_series}
        missing_planes = required_planes - visualized_planes
        remaining_slots = _REPRESENTATIVE_SERIES_LIMIT - len(visualized_series)
        if not missing_planes and remaining_slots <= 0:
            break
        if series.plane in visualized_planes and len(missing_planes) >= remaining_slots:
            continue
        attempted_series.append(series)
        monochrome_headers = [
            header
            for header in series.headers
            if str(getattr(header.dataset, "PhotometricInterpretation", ""))
            == "MONOCHROME1"
        ]
        rescale_headers = [
            header
            for header in series.headers
            if _float_attribute(header.dataset, "RescaleSlope", 1.0) != 1.0
            or _float_attribute(header.dataset, "RescaleIntercept", 0.0) != 0.0
        ]
        monochrome_observed += len(monochrome_headers)
        rescale_observed += len(rescale_headers)

        selected_indices = uniform_indices(
            len(series.ordered_paths), config.slice_count
        )
        deterministic_sampling = (
            deterministic_sampling
            and selected_indices
            == uniform_indices(len(series.ordered_paths), config.slice_count)
        )
        try:
            normalized, mask = preprocess_series(series.ordered_paths, config)
        except Exception as error:  # noqa: BLE001 - record heterogeneous decoder failures
            decode_failures.append(
                f"{series.inventory.path.name}: {type(error).__name__}"
            )
            continue

        decoded_series += 1
        if monochrome_headers and monochrome_tested == 0:
            monochrome_tested = 1
            monochrome_successful = int(
                _verify_decode_transform(monochrome_headers[0].path)
            )
        if rescale_headers and rescale_tested == 0:
            rescale_tested = 1
            rescale_successful = int(_verify_decode_transform(rescale_headers[0].path))

        visible = normalized[mask]
        finite_results.append(bool(np.isfinite(visible).all()))
        minima.append(float(visible.min()))
        maxima.append(float(visible.max()))
        visualization_name = f"ordered_slices_{index:02}_{series.plane}.png"
        _write_png_montage(artifact_dir / visualization_name, visible)
        visualization_paths.append(visualization_name)
        visualized_series.append(series)

    normalized_range = [min(minima), max(maxima)] if minima else []
    return (
        PreprocessingSummary(
            all_finite=bool(finite_results) and all(finite_results),
            decode_failures=decode_failures,
            decoded_series=decoded_series,
            deterministic_sampling=deterministic_sampling,
            monochrome1=TransformEvidence(
                observed_in_headers=monochrome_observed,
                tested_files=monochrome_tested,
                successful_files=monochrome_successful,
            ),
            normalized_range=normalized_range,
            rescale=TransformEvidence(
                observed_in_headers=rescale_observed,
                tested_files=rescale_tested,
                successful_files=rescale_successful,
            ),
        ),
        visualization_paths,
        visualized_series,
        attempted_series,
    )


def _verify_decode_transform(path: Path) -> bool:
    try:
        dataset = dcmread(path)
        raw = np.asarray(dataset.pixel_array, dtype=np.float32)
        expected = raw * _float_attribute(
            dataset, "RescaleSlope", 1.0
        ) + _float_attribute(dataset, "RescaleIntercept", 0.0)
        if str(getattr(dataset, "PhotometricInterpretation", "")) == "MONOCHROME1":
            expected = expected.max() + expected.min() - expected
        return bool(np.allclose(decode_pixels(path), expected))
    except Exception:  # noqa: BLE001 - a failed verification is evidence, not fatal
        return False


def _write_png_montage(path: Path, volume: np.ndarray) -> None:
    pixels = np.rint(np.clip(volume, 0, 1) * 255).astype(np.uint8)
    spacer = np.zeros((pixels.shape[1], 1), dtype=np.uint8)
    panels: list[np.ndarray] = []
    for index, image in enumerate(pixels):
        if index:
            panels.append(spacer)
        panels.append(image)
    montage = np.hstack(panels)
    height, width = montage.shape
    raw = b"".join(b"\x00" + montage[row].tobytes() for row in range(height))
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    png += _png_chunk(b"IDAT", zlib.compress(raw, level=9))
    png += _png_chunk(b"IEND", b"")
    path.write_bytes(png)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(data, checksum)
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def _summarize_dicom(
    all_series: list[SeriesInventory],
    ordered_series: list[_OrderedSeries],
    header_failures: list[str],
    root: Path,
) -> dict[str, Any]:
    headers = [header for series in ordered_series for header in series.headers]
    dimensions = sorted(
        {
            (rows, columns)
            for header in headers
            if (rows := _integer_attribute(header.dataset, "Rows")) is not None
            and (columns := _integer_attribute(header.dataset, "Columns")) is not None
        }
    )
    transfer_syntaxes = sorted(
        {
            str(header.dataset.file_meta.TransferSyntaxUID)
            for header in headers
            if "TransferSyntaxUID" in header.dataset.file_meta
        }
    )
    photometric_interpretations = sorted(
        {
            str(value)
            for header in headers
            if (value := getattr(header.dataset, "PhotometricInterpretation", None))
            is not None
        }
    )
    orientations = sorted(
        {
            orientation
            for header in headers
            if (
                orientation := _float_sequence(
                    header.dataset, "ImageOrientationPatient", 6
                )
            )
            is not None
        }
    )
    slice_counts = [series.slice_count for series in all_series]
    return {
        "dimensions": [list(item) for item in dimensions],
        "header_failures": header_failures,
        "header_sample_count": len(headers),
        "header_samples": [
            {
                "columns": _integer_attribute(header.dataset, "Columns"),
                "instance_number": _integer_attribute(header.dataset, "InstanceNumber"),
                "orientation": list(
                    _float_sequence(header.dataset, "ImageOrientationPatient", 6) or ()
                ),
                "path": header.path.relative_to(root).as_posix(),
                "photometric_interpretation": str(
                    getattr(header.dataset, "PhotometricInterpretation", "")
                ),
                "position": list(
                    _float_sequence(header.dataset, "ImagePositionPatient", 3) or ()
                ),
                "rescale_intercept": _float_attribute(
                    header.dataset, "RescaleIntercept", 0.0
                ),
                "rescale_slope": _float_attribute(header.dataset, "RescaleSlope", 1.0),
                "rows": _integer_attribute(header.dataset, "Rows"),
                "transfer_syntax": str(
                    getattr(header.dataset.file_meta, "TransferSyntaxUID", "")
                ),
            }
            for header in headers[:24]
        ],
        "orientation_values": [list(item) for item in orientations],
        "ordering_comparisons": [series.comparison for series in ordered_series],
        "photometric_interpretations": photometric_interpretations,
        "slice_counts": {
            "maximum": max(slice_counts) if slice_counts else None,
            "median": median(slice_counts) if slice_counts else None,
            "minimum": min(slice_counts) if slice_counts else None,
        },
        "transfer_syntaxes": transfer_syntaxes,
    }


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))
