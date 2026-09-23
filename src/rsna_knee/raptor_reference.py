from __future__ import annotations

import csv
import gc
import glob
import hashlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .competition_contract import load_verified_submission_contract
from .provenance import create_run_id, write_run_manifest
from .submission import (
    SubmissionOutcome,
    discover_submission_inputs,
    write_submission,
)

INTERMEDIATE_SIZE = 336
PHYSICAL_CROP_MM = 140.0
MODEL_INPUT_SIZE = 384
WINDOW_COUNT = 42
ARCHITECTURE = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
CHECKPOINT_FILENAME = "raptor_ft_coatnet_v4_full.pt"
CHECKPOINT_SIZE_BYTES = 292829402
CHECKPOINT_SHA256 = "89606f05849838529e1b4658d28fb049623205d9853371548403bca460361ded"
TARGET_NAMES = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)
SLOTS = (
    ("Sagittal", 1, 18),
    ("Sagittal", 0, 14),
    ("Coronal", 1, 12),
    ("Coronal", 0, 8),
    ("Axial", -1, 12),
)
STACK_SIZE = sum(slot[2] for slot in SLOTS)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
_EXPECTED_CHECKPOINT_KEYS = {
    "model",
    "gold_auc",
    "aucs",
    "src",
    "res",
    "arch",
    "lab",
    "epoch",
}

OrderedSeries = tuple[list[tuple[Path, float]], float]
SeriesOrderer = Callable[[Path], OrderedSeries]
PixelReader = Callable[[Path], np.ndarray]
CropResizer = Callable[[np.ndarray, float], np.ndarray]
Reader = tuple[SeriesOrderer, PixelReader, CropResizer]


@dataclass(frozen=True, slots=True)
class RaptorStudyStack:
    volume: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True, slots=True)
class RaptorCheckpointMetadata:
    sha256: str
    architecture: str
    resolution: int
    targets: tuple[str, ...]
    epoch: int
    source: str
    gold_auc: float


@dataclass(frozen=True, slots=True)
class RaptorReferenceOutcome:
    submission: SubmissionOutcome
    report_path: Path
    raw_failure_count: int


class RaptorClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feature_count: int = 768,
        target_count: int = 12,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(feature_count)
        self.att = nn.Sequential(
            nn.Linear(feature_count, 256),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(256, target_count),
        )
        self.clsW = nn.Parameter(torch.zeros(target_count, feature_count))
        self.clsb = nn.Parameter(torch.zeros(target_count))
        nn.init.trunc_normal_(self.clsW, std=0.02)

    def encode(self, inputs: Tensor) -> Tensor:
        batch_size, window_count = inputs.shape[:2]
        features = cast(Tensor, self.backbone(inputs.flatten(0, 1)))
        return features.view(batch_size, window_count, -1)

    def head(self, features: Tensor) -> Tensor:
        normalized = self.norm(features)
        attention = torch.softmax(self.att(normalized), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", attention, normalized)
        return (pooled * self.clsW).sum(-1) + self.clsb

    def forward(self, inputs: Tensor) -> Tensor:
        return self.head(self.encode(inputs))


def load_reference_contract(path: Path | None = None) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes() if path is not None else _reference_contract_bytes()
    contract = cast(dict[str, Any], json.loads(raw))
    if contract.get("schema_version") != 1:
        raise ValueError("Unsupported Raptor reference contract schema")
    checkpoint = contract.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise TypeError("Raptor reference contract has no checkpoint object")
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError("Raptor reference contract has no checkpoint metadata")
    if (
        checkpoint.get("filename") != CHECKPOINT_FILENAME
        or checkpoint.get("sha256") != CHECKPOINT_SHA256
        or metadata.get("arch") != ARCHITECTURE
        or metadata.get("res") != MODEL_INPUT_SIZE
        or tuple(metadata.get("lab", ())) != TARGET_NAMES
    ):
        raise ValueError("Raptor reference contract identity has drifted")
    return contract, hashlib.sha256(raw).hexdigest()


def make_reader() -> Reader:
    import cv2
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut

    def order_and_meta(series_dir: Path) -> OrderedSeries:
        records: list[tuple[float, Path, float]] = []
        spacings: list[float] = []
        for raw_path in glob.glob(str(series_dir / "*.dcm")):
            path = Path(raw_path)
            try:
                header = pydicom.dcmread(path, stop_before_pixels=True)
                orientation = getattr(header, "ImageOrientationPatient", None)
                position = getattr(header, "ImagePositionPatient", None)
                if (
                    orientation is not None
                    and position is not None
                    and len(orientation) == 6
                ):
                    row = np.array(orientation[:3], dtype=float)
                    column = np.array(orientation[3:], dtype=float)
                    coordinate = float(
                        np.dot(np.array(position, dtype=float), np.cross(row, column))
                    )
                else:
                    coordinate = float(getattr(header, "InstanceNumber", 0) or 0)
                spacing_value = getattr(header, "PixelSpacing", None)
                spacing = float(spacing_value[0]) if spacing_value is not None else 0.5
                spacings.append(spacing)
                records.append((coordinate, path, spacing))
            except Exception:  # noqa: BLE001 - public reference records header failures
                records.append((0.0, path, 0.5))
        records.sort(key=lambda record: record[0])
        median_spacing = float(np.median(spacings)) if spacings else 0.5
        return [(path, spacing) for _, path, spacing in records], median_spacing

    def read_pixels(path: Path) -> np.ndarray:
        dataset = pydicom.dcmread(path)
        pixels = apply_modality_lut(dataset.pixel_array, dataset).astype(np.float32)
        if str(getattr(dataset, "PhotometricInterpretation", "")) == "MONOCHROME1":
            pixels = pixels.max() - pixels
        return cast(np.ndarray, pixels)

    def crop_resize(image: np.ndarray, spacing: float) -> np.ndarray:
        height, width = image.shape
        side = round(PHYSICAL_CROP_MM / max(spacing, 1e-3))
        side = min(side, min(height, width))
        top = (height - side) // 2
        left = (width - side) // 2
        cropped = image[top : top + side, left : left + side]
        return cast(
            np.ndarray,
            cv2.resize(
                cropped,
                (INTERMEDIATE_SIZE, INTERMEDIATE_SIZE),
                interpolation=cv2.INTER_AREA,
            ),
        )

    return order_and_meta, read_pixels, crop_resize


def select_series_for_slot(
    rows: Sequence[Mapping[str, object]],
    plane: str,
    fluid_sensitive: int,
    used_series: set[str],
) -> Mapping[str, object] | None:
    candidates = [
        row
        for row in rows
        if row["Anatomical_Plane"] == plane
        and str(row["SeriesInstanceUID"]) not in used_series
    ]
    if fluid_sensitive in (0, 1):
        preferred = [
            row
            for row in candidates
            if int(str(row.get("Fluid_Sensitive", 0) or 0)) == fluid_sensitive
        ]
        if preferred:
            return preferred[0]
    return candidates[0] if candidates else None


def sampled_slice_indices(length: int, count: int) -> tuple[int, ...]:
    if length <= 0 or count <= 0:
        raise ValueError("Slice sampling requires positive length and count")
    if length == 1:
        return (0,) * count
    lower = int(length * 0.06)
    upper = max(int(length * 0.94) - 1, lower)
    return tuple(np.linspace(lower, upper, count).round().astype(int).tolist())


def build_study_stack(
    study_id: str,
    series_records: Mapping[str, Sequence[Mapping[str, object]]],
    series_root: Path,
    *,
    reader: Reader | None = None,
) -> RaptorStudyStack:
    order_and_meta, read_pixels, crop_resize = reader or make_reader()
    rows = series_records.get(study_id, ())
    volume = np.zeros(
        (STACK_SIZE, INTERMEDIATE_SIZE, INTERMEDIATE_SIZE), dtype=np.uint8
    )
    output_index = 0
    used_series: set[str] = set()
    for plane, fluid_sensitive, count in SLOTS:
        row = select_series_for_slot(rows, plane, fluid_sensitive, used_series)
        if row is None:
            output_index += count
            continue
        series_id = str(row["SeriesInstanceUID"])
        used_series.add(series_id)
        ordered, median_spacing = order_and_meta(series_root / study_id / series_id)
        if not ordered:
            output_index += count
            continue

        picks = sampled_slice_indices(len(ordered), count)
        decoded: list[np.ndarray | None] = []
        spacings: list[float] = []
        for position in picks:
            path, spacing = ordered[min(position, len(ordered) - 1)]
            try:
                decoded.append(read_pixels(path))
                spacings.append(spacing)
            except Exception:  # noqa: BLE001 - public reference zero-fills failed slices
                decoded.append(None)
                spacings.append(median_spacing)

        valid = [pixels for pixels in decoded if pixels is not None]
        if valid:
            all_pixels = np.concatenate([pixels.ravel() for pixels in valid])
            lower, upper = np.percentile(all_pixels, [2.0, 98.0])
        else:
            lower, upper = 0.0, 1.0

        for pixels, spacing in zip(decoded, spacings, strict=True):
            if output_index >= STACK_SIZE:
                break
            if pixels is None:
                output_index += 1
                continue
            windowed = np.clip((pixels - lower) / (upper - lower + 1e-6), 0, 1)
            resized = crop_resize(windowed, spacing if spacing > 0 else median_spacing)
            volume[output_index] = (resized * 255).astype(np.uint8)
            output_index += 1
        if output_index >= STACK_SIZE:
            break

    mask = (volume.reshape(STACK_SIZE, -1).sum(1) > 0).astype(np.uint8)
    return RaptorStudyStack(volume=volume, mask=mask)


def eval_centers(mask: np.ndarray, depth: int, count: int) -> tuple[int, ...]:
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, depth))
    lower, upper = int(valid.min()), int(valid.max())
    candidates = [
        center
        for center in range(lower + 1, upper)
        if center - 1 >= lower and center + 1 <= upper
    ]
    if not candidates:
        candidates = [max(1, min((lower + upper) // 2, depth - 2))]
    indices = np.linspace(0, len(candidates) - 1, count).round().astype(int)
    return tuple(candidates[index] for index in indices)


def eval_windows(
    volume: np.ndarray,
    mask: np.ndarray,
    count: int = WINDOW_COUNT,
    resolution: int = MODEL_INPUT_SIZE,
) -> Tensor:
    depth = volume.shape[0]
    centers = eval_centers(mask, depth, count)
    windows = np.empty((len(centers), 3, resolution, resolution), dtype=np.float32)
    for index, center in enumerate(centers):
        center = max(1, min(center, depth - 2))
        triplet = (
            np.stack(
                [volume[center - 1], volume[center], volume[center + 1]], axis=0
            ).astype(np.float32)
            / 255.0
        )
        tensor = torch.from_numpy(triplet)
        if tensor.shape[-1] != resolution:
            tensor = F.interpolate(
                tensor[None],
                size=(resolution, resolution),
                mode="bilinear",
                align_corners=False,
            )[0]
        windows[index] = tensor.numpy()
    output = torch.from_numpy(windows)
    return (output - IMAGENET_MEAN) / IMAGENET_STD


def rank_percentiles(probabilities: np.ndarray) -> np.ndarray:
    order = probabilities.argsort(0).argsort(0).astype(np.float64)
    return cast(np.ndarray, order / max(1, probabilities.shape[0] - 1))


def load_raptor_model(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[RaptorClassifier, RaptorCheckpointMetadata]:
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise ValueError("Raptor checkpoint hash does not match the frozen reference")
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or set(raw) != _EXPECTED_CHECKPOINT_KEYS:
        raise ValueError(
            "Raptor checkpoint top-level schema does not match the reference"
        )
    architecture = raw.get("arch")
    resolution = raw.get("res")
    labels = raw.get("lab")
    state_dict = raw.get("model")
    if architecture != ARCHITECTURE or resolution != MODEL_INPUT_SIZE:
        raise ValueError("Raptor checkpoint architecture identity does not match")
    if not isinstance(labels, list) or tuple(labels) != TARGET_NAMES:
        raise ValueError("Raptor checkpoint target order does not match")
    if not isinstance(state_dict, dict) or len(state_dict) != 486:
        raise ValueError("Raptor checkpoint model state does not match")
    competition_targets = load_verified_submission_contract().target_names
    if tuple(labels) != competition_targets:
        raise ValueError("Raptor and competition target order do not match")

    import timm

    backbone = timm.create_model(
        architecture,
        pretrained=False,
        num_classes=0,
        in_chans=3,
        global_pool="avg",
    )
    feature_count = getattr(backbone, "num_features", None)
    if not isinstance(feature_count, int) or feature_count != 1024:
        raise ValueError("Raptor backbone feature count does not match the checkpoint")
    model = RaptorClassifier(
        backbone,
        feature_count=feature_count,
        target_count=len(TARGET_NAMES),
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    metadata = RaptorCheckpointMetadata(
        sha256=digest,
        architecture=architecture,
        resolution=resolution,
        targets=tuple(labels),
        epoch=int(raw["epoch"]),
        source=str(raw["src"]),
        gold_auc=float(raw["gold_auc"]),
    )
    del raw
    gc.collect()
    return model, metadata


@torch.no_grad()
def infer_probabilities(
    model: RaptorClassifier,
    windows: Tensor,
    device: torch.device,
) -> np.ndarray:
    inputs = windows.unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                output = torch.sigmoid(model(inputs).float())
            return output[0].cpu().numpy()
        except RuntimeError:
            torch.cuda.empty_cache()
            output = torch.sigmoid(model(inputs).float())
            return output[0].cpu().numpy()
    output = torch.sigmoid(model(inputs).float())
    return output[0].cpu().numpy()


def run_raptor_reference(
    *,
    input_root: Path,
    work_root: Path,
    checkpoint_path: Path,
    device: torch.device | None = None,
    contract_path: Path | None = None,
    run_id: str | None = None,
    notebook: Mapping[str, object] | None = None,
    source_provenance: Mapping[str, object] | None = None,
) -> RaptorReferenceOutcome:
    started = time.perf_counter()
    reference_contract, reference_contract_sha256 = load_reference_contract()
    inputs = discover_submission_inputs(input_root, contract_path=contract_path)
    if inputs.contract.target_names != TARGET_NAMES:
        raise ValueError("Submission target order differs from the Raptor reference")
    records = _read_series_records(inputs.competition_root / "test_series.csv")
    resolved_device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
    model, checkpoint = load_raptor_model(checkpoint_path, device=resolved_device)
    reader = make_reader()
    raw_probabilities = np.full(
        (len(inputs.entity_ids), len(TARGET_NAMES)), 0.5, dtype=np.float32
    )
    failures: Counter[str] = Counter()
    for index, study_id in enumerate(inputs.entity_ids):
        try:
            stack = build_study_stack(
                study_id,
                records,
                inputs.competition_root / inputs.contract.test_series_image_root,
                reader=reader,
            )
            windows = eval_windows(stack.volume, stack.mask)
            raw_probabilities[index] = infer_probabilities(
                model, windows, resolved_device
            )
        except Exception as error:  # noqa: BLE001 - exact public per-study fallback
            failures[type(error).__name__] += 1

    ranked = rank_percentiles(np.clip(raw_probabilities, 0, 1)).astype(np.float32)
    predictions = {
        study_id: tuple(float(value) for value in ranked[index])
        for index, study_id in enumerate(inputs.entity_ids)
    }
    artifacts = write_submission(
        inputs=inputs,
        work_root=work_root,
        predictions=predictions,
        failures={},
        fallback_probability=0.5,
    )
    inference_seconds = time.perf_counter() - started
    peak_gpu_bytes = (
        int(torch.cuda.max_memory_allocated(resolved_device))
        if resolved_device.type == "cuda"
        else 0
    )
    report_path = work_root / "raptor-reference-run.json"
    report = {
        "checkpoint": {
            "architecture": checkpoint.architecture,
            "epoch": checkpoint.epoch,
            "gold_auc": checkpoint.gold_auc,
            "resolution": checkpoint.resolution,
            "sha256": checkpoint.sha256,
            "source": checkpoint.source,
            "targets": list(checkpoint.targets),
        },
        "inference": {
            "cuda_backend": {
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            "device": str(resolved_device),
            "peak_gpu_memory_bytes": peak_gpu_bytes,
            "raw_failure_count": sum(failures.values()),
            "raw_failure_types": dict(sorted(failures.items())),
            "seconds": inference_seconds,
            "study_count": len(inputs.entity_ids),
            "window_count_per_study": WINDOW_COUNT,
        },
        "output": {
            "activation": "sigmoid",
            "submission_transform": "per-target percentile rank",
            "validation_report": artifacts.validation_report_path.name,
        },
        "reference_contract_sha256": reference_contract_sha256,
        "reference_notebook": reference_contract["public_notebook"],
        "schema_version": 1,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_source = dict(source_provenance or {})
    manifest_source.update(
        {
            "checkpoint": {
                "filename": checkpoint_path.name,
                "sha256": checkpoint.sha256,
            },
            "competition_contract_sha256": inputs.contract.sha256,
            "raptor_reference": {
                "checkpoint": {
                    "filename": reference_contract["checkpoint"]["filename"],
                    "sha256": reference_contract["checkpoint"]["sha256"],
                    "size_bytes": reference_contract["checkpoint"]["size_bytes"],
                },
                "model_dataset": reference_contract["model_dataset"],
                "public_notebook": reference_contract["public_notebook"],
            },
            "raptor_reference_contract_sha256": reference_contract_sha256,
        }
    )
    manifest_path = write_run_manifest(
        work_root / "run-manifest.json",
        run_id=run_id or create_run_id("raptor-reference"),
        source=manifest_source,
        configuration={
            "architecture": ARCHITECTURE,
            "checkpoint_filename": CHECKPOINT_FILENAME,
            "intermediate_size": INTERMEDIATE_SIZE,
            "model_input_size": MODEL_INPUT_SIZE,
            "physical_crop_mm": PHYSICAL_CROP_MM,
            "slot_layout": [list(slot) for slot in SLOTS],
            "target_order": list(TARGET_NAMES),
            "window_count": WINDOW_COUNT,
        },
        notebook=notebook or {"kind": "local_raptor_reference_inference"},
        artifacts=[
            artifacts.submission_path,
            artifacts.fallbacks_path,
            artifacts.validation_report_path,
            report_path,
        ],
        artifact_root=work_root,
    )
    submission = SubmissionOutcome(
        submission_path=artifacts.submission_path,
        fallbacks_path=artifacts.fallbacks_path,
        validation_report_path=artifacts.validation_report_path,
        manifest_path=manifest_path,
    )
    return RaptorReferenceOutcome(
        submission=submission,
        report_path=report_path,
        raw_failure_count=sum(failures.values()),
    )


_SCORING_EVIDENCE_FILENAMES = (
    "submission.csv",
    "submission-fallbacks.csv",
    "submission-validation.json",
    "raptor-reference-run.json",
    "run-manifest.json",
)


def verify_raptor_reference_scoring_artifact(
    artifact_dir: Path,
) -> dict[str, object]:
    """Verify hidden-inference evidence without exposing prediction rows."""
    evidence = {name: artifact_dir / name for name in _SCORING_EVIDENCE_FILENAMES}
    for path in evidence.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing Raptor scoring artifact: {path}")

    manifest = _json_object(evidence["run-manifest.json"], "run manifest")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported Raptor scoring manifest schema")
    records = manifest.get("artifacts")
    if not isinstance(records, list):
        raise TypeError("Raptor scoring artifact inventory is malformed")
    expected_inventory = set(_SCORING_EVIDENCE_FILENAMES[:-1])
    actual_inventory: set[str] = set()
    for raw_record in records:
        if not isinstance(raw_record, dict):
            raise TypeError("Raptor scoring artifact record is malformed")
        relative = raw_record.get("path")
        if not isinstance(relative, str) or relative in actual_inventory:
            raise ValueError("Raptor scoring artifact paths are malformed or duplicate")
        actual_inventory.add(relative)
        if relative not in expected_inventory or Path(relative).name != relative:
            raise ValueError("Raptor scoring manifest contains an unexpected artifact")
        path = evidence[relative]
        if raw_record.get("size_bytes") != path.stat().st_size or raw_record.get(
            "sha256"
        ) != _file_sha256(path):
            raise ValueError(f"Raptor scoring artifact hash differs: {relative}")
    if actual_inventory != expected_inventory:
        raise ValueError("Raptor scoring manifest artifact inventory is incomplete")

    reference, reference_sha256 = load_reference_contract()
    competition = load_verified_submission_contract()
    report = _json_object(evidence["raptor-reference-run.json"], "run report")
    validation = _json_object(
        evidence["submission-validation.json"], "submission validation"
    )
    if report.get("schema_version") != 1:
        raise ValueError("Unsupported Raptor scoring report schema")
    if validation.get("schema_version") != 1 or validation.get("valid") is not True:
        raise ValueError("Raptor submission validation evidence is invalid")

    submission_sha256 = _file_sha256(evidence["submission.csv"])
    if (
        validation.get("competition_contract_sha256") != competition.sha256
        or validation.get("submission_sha256") != submission_sha256
        or validation.get("columns") != list(competition.ordered_columns)
        or validation.get("target_count") != len(TARGET_NAMES)
    ):
        raise ValueError("Raptor submission validation identity differs")
    row_count = validation.get("row_count")
    probabilities = validation.get("probabilities")
    fallbacks = validation.get("fallbacks")
    if (
        type(row_count) is not int
        or row_count <= 0
        or not isinstance(probabilities, dict)
        or not isinstance(fallbacks, dict)
        or fallbacks.get("count") != 0
        or fallbacks.get("path") != "submission-fallbacks.csv"
    ):
        raise ValueError("Raptor submission validation summary is malformed")
    minimum = probabilities.get("minimum")
    maximum = probabilities.get("maximum")
    if (
        not isinstance(minimum, (int, float))
        or not isinstance(maximum, (int, float))
        or not 0.0 <= float(minimum) <= float(maximum) <= 1.0
    ):
        raise ValueError("Raptor submission probability summary is invalid")

    metadata = reference["checkpoint"]["metadata"]
    expected_checkpoint = {
        "architecture": metadata["arch"],
        "epoch": metadata["epoch"],
        "gold_auc": metadata["gold_auc"],
        "resolution": metadata["res"],
        "sha256": CHECKPOINT_SHA256,
        "source": metadata["src"],
        "targets": list(TARGET_NAMES),
    }
    if (
        report.get("checkpoint") != expected_checkpoint
        or report.get("reference_contract_sha256") != reference_sha256
        or report.get("reference_notebook") != reference["public_notebook"]
        or report.get("output")
        != {
            "activation": "sigmoid",
            "submission_transform": "per-target percentile rank",
            "validation_report": "submission-validation.json",
        }
    ):
        raise ValueError("Raptor scoring report differs from the frozen reference")

    inference = report.get("inference")
    if not isinstance(inference, dict):
        raise TypeError("Raptor scoring inference summary is malformed")
    failure_types = inference.get("raw_failure_types")
    raw_failure_count = inference.get("raw_failure_count")
    if (
        inference.get("device") != "cuda"
        or inference.get("cuda_backend")
        != {"cudnn_benchmark": True, "matmul_allow_tf32": True}
        or inference.get("study_count") != row_count
        or inference.get("window_count_per_study") != WINDOW_COUNT
        or type(raw_failure_count) is not int
        or raw_failure_count < 0
        or not isinstance(failure_types, dict)
        or any(type(count) is not int or count < 0 for count in failure_types.values())
        or sum(failure_types.values()) != raw_failure_count
        or not isinstance(inference.get("seconds"), (int, float))
        or float(inference["seconds"]) <= 0
        or type(inference.get("peak_gpu_memory_bytes")) is not int
        or inference["peak_gpu_memory_bytes"] <= 0
    ):
        raise ValueError("Raptor scoring execution evidence is invalid")

    source = manifest.get("source")
    notebook = manifest.get("notebook")
    configuration = manifest.get("configuration")
    if not isinstance(source, dict) or not isinstance(notebook, dict):
        raise TypeError("Raptor scoring provenance is malformed")
    expected_reference_source = {
        "checkpoint": {
            "filename": reference["checkpoint"]["filename"],
            "sha256": reference["checkpoint"]["sha256"],
            "size_bytes": reference["checkpoint"]["size_bytes"],
        },
        "model_dataset": reference["model_dataset"],
        "public_notebook": reference["public_notebook"],
    }
    if (
        source.get("checkpoint")
        != {"filename": CHECKPOINT_FILENAME, "sha256": CHECKPOINT_SHA256}
        or source.get("competition_contract_sha256") != competition.sha256
        or source.get("raptor_reference") != expected_reference_source
        or source.get("raptor_reference_contract_sha256") != reference_sha256
        or source.get("git_dirty") is not False
        or not _is_hex_digest(source.get("git_commit"), 40)
        or not _is_hex_digest(source.get("jupytext_source_sha256"), 64)
        or not _is_hex_digest(source.get("rendered_notebook_sha256"), 64)
        or source.get("kernel_id") != notebook.get("kernel_id")
    ):
        raise ValueError("Raptor scoring source provenance differs")
    if notebook != {
        "competition_submission": False,
        "hidden_test_inference": True,
        "kernel_id": notebook.get("kernel_id"),
        "kind": "approved_kaggle_raptor_reference_hidden_scoring",
        "training": False,
    } or not isinstance(notebook.get("kernel_id"), str):
        raise ValueError("Raptor scoring boundary flags are invalid")
    if configuration != {
        "architecture": ARCHITECTURE,
        "checkpoint_filename": CHECKPOINT_FILENAME,
        "intermediate_size": INTERMEDIATE_SIZE,
        "model_input_size": MODEL_INPUT_SIZE,
        "physical_crop_mm": PHYSICAL_CROP_MM,
        "slot_layout": [list(slot) for slot in SLOTS],
        "target_order": list(TARGET_NAMES),
        "window_count": WINDOW_COUNT,
    }:
        raise ValueError("Raptor scoring configuration differs")

    return {
        "artifact_sha256": {
            name: _file_sha256(path) for name, path in evidence.items()
        },
        "git_commit": source["git_commit"],
        "kernel_id": notebook["kernel_id"],
        "maximum_probability": float(maximum),
        "minimum_probability": float(minimum),
        "peak_gpu_memory_bytes": inference["peak_gpu_memory_bytes"],
        "raw_failure_count": raw_failure_count,
        "row_count": row_count,
        "seconds": float(inference["seconds"]),
        "submission_sha256": submission_sha256,
        "valid": True,
    }


def _json_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Raptor {label} is not a JSON object")
    return cast(dict[str, Any], value)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_hex_digest(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_series_records(
    path: Path,
) -> dict[str, tuple[Mapping[str, object], ...]]:
    grouped: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            grouped[str(row["StudyInstanceUID"])].append(row)
    return {study_id: tuple(rows) for study_id, rows in grouped.items()}


def _reference_contract_bytes() -> bytes:
    source = (
        Path(__file__).resolve().parents[2] / "contracts" / "raptor-reference-v1.json"
    )
    if source.is_file():
        return source.read_bytes()
    return files("rsna_knee").joinpath("raptor-reference-v1.json").read_bytes()
