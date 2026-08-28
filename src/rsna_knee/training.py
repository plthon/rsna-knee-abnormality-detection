"""Gold-only ResNet-18 smoke training with explicit offline initialization."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import re
import time
import tomllib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import (  # type: ignore[import-untyped]
    ResNet18_Weights,
    resnet18,
)

from .competition_contract import (
    VerifiedGoldContract,
    load_verified_gold_contract,
    load_verified_submission_contract,
)
from .config import PreprocessingConfig
from .indexing import verify_index_artifact
from .preprocessing import preprocess_series_result
from .provenance import create_run_id, write_run_manifest
from .submission import (
    SubmissionArtifacts,
    SubmissionInputs,
    discover_submission_inputs,
    write_submission,
)

type Label = Literal[0, 1] | None
type AnatomicalPlane = Literal["sagittal", "coronal", "axial"]
type StudyAggregation = Literal["mean", "target_plane_gate"]

ANATOMICAL_PLANES: tuple[AnatomicalPlane, ...] = ("sagittal", "coronal", "axial")
UNKNOWN_PLANE_INDEX = -1

_CHECKPOINT_SCHEMA_VERSION = 1
_REPORT_SCHEMA_VERSION = 1
_COHORT_RULE_VERSION = "organizer-gold-all-eligible-v1"
_COHORT_DIGEST_VERSION = "study-label-jsonl-v1"
_ARTIFACT_DIRECTORY = "rsna-knee-gold-training-v1"
_STUDY_COLUMNS = (
    "study_index",
    "split",
    "StudyInstanceUID",
    "PatientID",
    "series_count",
    "usable_series_count",
    "slice_count",
    "usable_slice_count",
)
_SERIES_COLUMNS = (
    "series_index",
    "study_index",
    "SeriesInstanceUID",
    "relative_path",
    "Fluid_Sensitive",
    "Fat_Suppression",
    "Anatomical_Plane",
    "slice_count",
    "usable_slice_count",
    "ordering_method",
    "selection_mask",
)
_SLICE_COLUMNS = (
    "slice_index",
    "series_index",
    "SOPInstanceUID",
    "filename",
    "ordered_position",
    "rows",
    "columns",
    "instance_number",
    "geometry_coordinate",
    "transfer_syntax",
    "photometric_interpretation",
    "rescale_slope",
    "rescale_intercept",
)
_SELECTED_COLUMNS = ("series_index", "slot", "slice_index", "ordered_position")


@dataclass(frozen=True, slots=True)
class SmokeTrainingConfig:
    preprocessing: PreprocessingConfig
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int
    mixed_precision: bool
    architecture: str
    weight_identity: str


@dataclass(frozen=True, slots=True)
class InitializationArtifact:
    path: Path
    sha256: str
    size_bytes: int
    contract_sha256: str
    contract: dict[str, Any]


@dataclass(frozen=True, slots=True)
class GoldStudy:
    study_uid: str
    labels: tuple[Label, ...]


@dataclass(frozen=True, slots=True)
class GoldTrainingCohort:
    studies: tuple[GoldStudy, ...]
    target_names: tuple[str, ...]
    digest: str
    per_target: dict[str, dict[str, int]]
    gold_contract_sha256: str
    gold_supervision_contract_sha256: str


@dataclass(frozen=True, slots=True)
class IndexedSeries:
    series_index: int
    paths: tuple[Path, ...]
    plane: AnatomicalPlane | None = None


@dataclass(frozen=True, slots=True)
class IndexedStudy:
    study_uid: str
    series: tuple[IndexedSeries, ...]


class LabeledStudy(Protocol):
    @property
    def study_uid(self) -> str: ...

    @property
    def labels(self) -> Sequence[Label]: ...


class PartitionStudy(LabeledStudy, Protocol):
    @property
    def patient_id(self) -> str: ...


@dataclass(frozen=True, slots=True)
class TrainingInputs:
    competition_root: Path
    index_artifact_dir: Path
    weight_path: Path
    index_verification: Literal["accepted_frozen_artifact", "rebuilt_verified_artifact"]


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    artifact_dir: Path
    checkpoint_path: Path
    report_path: Path
    submission_path: Path
    validation_report_path: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class CheckpointPredictionOutcome:
    submission_path: Path
    fallbacks_path: Path
    validation_report_path: Path
    inference_seconds: float
    prediction_count: int
    fallback_count: int
    decode_errors: dict[str, int]
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class _SubmissionPrediction:
    artifacts: SubmissionArtifacts
    inference_seconds: float
    prediction_count: int
    fallback_count: int
    decode_errors: dict[str, int]


@dataclass(frozen=True, slots=True)
class DecodedStudy:
    pixels: Tensor
    slice_mask: Tensor
    series_mask: Tensor
    labels: Tensor | None
    label_mask: Tensor | None
    valid_slice_count: int
    decode_error_counts: dict[str, int]
    plane_indices: Tensor | None = None


@dataclass(frozen=True, slots=True)
class PartitionPrediction:
    studies: tuple[PartitionStudy, ...]
    probabilities: tuple[tuple[float, ...], ...]
    decode_seconds: float
    forward_seconds: float
    valid_slice_count: int
    decode_errors: dict[str, int]


class TargetSpecificPlaneGate(nn.Module):
    """Positive target-by-plane weights with a fixed neutral unknown weight."""

    def __init__(self, target_count: int):
        super().__init__()
        self.log_weights = nn.Parameter(
            torch.zeros(target_count, len(ANATOMICAL_PLANES))
        )

    def forward(
        self,
        series_logits: Tensor,
        series_mask: Tensor,
        plane_indices: Tensor,
    ) -> Tensor:
        if series_logits.ndim != 3:
            raise ValueError("series logits must have shape [study, series, target]")
        if series_mask.shape != series_logits.shape[:2]:
            raise ValueError("series mask does not match study/series axes")
        if plane_indices.shape != series_logits.shape[:2]:
            raise ValueError("plane indices do not match study/series axes")
        if bool(
            (
                (plane_indices < UNKNOWN_PLANE_INDEX)
                | (plane_indices >= len(ANATOMICAL_PLANES))
            ).any()
        ):
            raise ValueError("plane indices must be -1 or a known plane index")

        safe_indices = plane_indices.clamp_min(0)
        learned = self.log_weights.exp()[:, safe_indices].permute(1, 2, 0)
        weights = torch.where(
            (plane_indices >= 0).unsqueeze(-1), learned, torch.ones_like(learned)
        )
        weights = weights * series_mask.to(dtype=series_logits.dtype).unsqueeze(-1)
        denominator = weights.sum(dim=1).clamp_min(1)
        return (series_logits * weights).sum(dim=1) / denominator

    def effective_weights(self) -> Tensor:
        return self.log_weights.detach().exp()


class GoldResNet18(nn.Module):
    """One shared slice encoder with masked slice and series mean pooling."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_count: int,
        target_count: int,
        *,
        aggregation: StudyAggregation = "mean",
    ):
        super().__init__()
        if aggregation not in {"mean", "target_plane_gate"}:
            raise ValueError(f"Unsupported study aggregation: {aggregation}")
        self.imagenet_mean: Tensor
        self.imagenet_std: Tensor
        self.backbone = backbone
        self.head = nn.Linear(feature_count, target_count)
        self.plane_gate = (
            TargetSpecificPlaneGate(target_count)
            if aggregation == "target_plane_gate"
            else None
        )
        self.register_buffer(
            "imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.feature_count = feature_count
        self.aggregation = aggregation

    def forward(
        self,
        pixels: Tensor,
        slice_mask: Tensor,
        series_mask: Tensor,
        plane_indices: Tensor | None = None,
    ) -> Tensor:
        if pixels.ndim != 5:
            raise ValueError("pixels must have shape [study, series, slice, H, W]")
        if slice_mask.shape != pixels.shape[:3]:
            raise ValueError("slice mask does not match study/series/slice axes")
        if series_mask.shape != pixels.shape[:2]:
            raise ValueError("series mask does not match study/series axes")
        valid = slice_mask.to(dtype=torch.bool) & series_mask.to(
            dtype=torch.bool
        ).unsqueeze(-1)
        flat_valid = valid.reshape(-1)
        if not bool(flat_valid.any()):
            raise ValueError("At least one unpadded slice is required per batch")

        flat_pixels = pixels.reshape(-1, pixels.shape[-2], pixels.shape[-1])
        selected = flat_pixels[flat_valid].unsqueeze(1).repeat(1, 3, 1, 1)
        selected = (selected - self.imagenet_mean) / self.imagenet_std
        selected_features = cast(Tensor, self.backbone(selected))

        flat_features = selected_features.new_zeros(
            (flat_pixels.shape[0], self.feature_count)
        )
        flat_features[flat_valid] = selected_features
        features = flat_features.view(*pixels.shape[:3], self.feature_count)

        slice_weights = valid.unsqueeze(-1).to(dtype=features.dtype)
        slice_counts = slice_weights.sum(dim=2).clamp_min(1)
        series_features = (features * slice_weights).sum(dim=2) / slice_counts

        effective_series = valid.any(dim=2) & series_mask.to(dtype=torch.bool)
        if self.plane_gate is not None:
            if plane_indices is None:
                raise ValueError("plane-gated aggregation requires plane indices")
            return cast(
                Tensor,
                self.plane_gate(
                    self.head(series_features), effective_series, plane_indices
                ),
            )
        series_weights = effective_series.unsqueeze(-1).to(dtype=features.dtype)
        series_counts = series_weights.sum(dim=1).clamp_min(1)
        study_features = (series_features * series_weights).sum(dim=1) / series_counts
        return cast(Tensor, self.head(study_features))


def load_smoke_training_config(path: Path) -> SmokeTrainingConfig:
    """Load and strictly constrain the one-epoch issue #8 smoke configuration."""
    with path.open("rb") as file:
        raw = tomllib.load(file)
    preprocessing_raw = _mapping(raw, "preprocessing")
    training_raw = _mapping(raw, "training")
    preprocessing = PreprocessingConfig(
        slice_count=_positive_int(preprocessing_raw, "slice_count"),
        lower_percentile=_number(preprocessing_raw, "lower_percentile"),
        upper_percentile=_number(preprocessing_raw, "upper_percentile"),
        image_size=_positive_int(preprocessing_raw, "image_size"),
    )
    config = SmokeTrainingConfig(
        preprocessing=preprocessing,
        epochs=_positive_int(training_raw, "epochs"),
        batch_size=_positive_int(training_raw, "batch_size"),
        learning_rate=_positive_number(training_raw, "learning_rate"),
        weight_decay=_nonnegative_number(training_raw, "weight_decay"),
        seed=_positive_int(training_raw, "seed"),
        mixed_precision=_boolean(training_raw, "mixed_precision"),
        architecture=_text(training_raw, "architecture"),
        weight_identity=_text(training_raw, "weight_identity"),
    )
    if config.epochs != 1:
        raise ValueError("Issue #8 smoke training must run exactly one epoch")
    if not (
        0 <= preprocessing.lower_percentile < preprocessing.upper_percentile <= 100
    ):
        raise ValueError("Preprocessing percentiles are invalid")
    if preprocessing.image_size != 224:
        raise ValueError("Issue #8 requires 224 by 224 pixel inputs")
    if config.architecture != "torchvision.models.resnet18":
        raise ValueError("Issue #8 requires exactly torchvision ResNet-18")
    if config.weight_identity != "ResNet18_Weights.IMAGENET1K_V1":
        raise ValueError("Unexpected initialization weight identity")
    return config


def discover_training_inputs(
    input_root: Path,
    *,
    index_artifact_dir: Path | None = None,
) -> TrainingInputs:
    """Discover accepted index, competition data, and one exact offline weight."""
    competition_root = discover_submission_inputs(input_root).competition_root
    if index_artifact_dir is None:
        index_contract, _ = _load_json_contract("data-index.json", None)
        artifact = _mapping(index_contract, "artifact")
        studies_record = _mapping(_mapping(artifact, "files"), "studies.csv")
        outputs = _mapping(_mapping(index_contract, "run_provenance"), "outputs")
        expected_size = _positive_int(studies_record, "size_bytes")
        expected_hash = _text(studies_record, "sha256")
        expected_manifest_hash = _text(outputs, "manifest_sha256")
        candidates = [
            path.parent
            for path in sorted(input_root.rglob("studies.csv"))
            if path.stat().st_size == expected_size
            and _sha256(path) == expected_hash
            and (path.parent / "index-manifest.json").is_file()
            and _sha256(path.parent / "index-manifest.json") == expected_manifest_hash
        ]
        if len(candidates) != 1:
            raise ValueError(
                "Expected exactly one attached index matching the accepted evidence"
            )
        resolved_index_artifact_dir = candidates[0]
        index_verification: Literal[
            "accepted_frozen_artifact", "rebuilt_verified_artifact"
        ] = "accepted_frozen_artifact"
    else:
        verify_index_artifact(index_artifact_dir)
        resolved_index_artifact_dir = index_artifact_dir
        index_verification = "rebuilt_verified_artifact"
    contract, _ = _load_json_contract("resnet18-imagenet1k-v1.json", None)
    artifact = _mapping(contract, "artifact")
    expected_name = _text(artifact, "filename")
    expected_size = _positive_int(artifact, "size_bytes")
    expected_hash = _text(artifact, "sha256")
    candidates = [
        path
        for path in sorted(input_root.rglob(expected_name))
        if path.is_file()
        and path.stat().st_size == expected_size
        and _sha256(path) == expected_hash
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Expected exactly one offline ResNet-18 artifact with the approved hash"
        )
    return TrainingInputs(
        competition_root=competition_root,
        index_artifact_dir=resolved_index_artifact_dir,
        weight_path=candidates[0],
        index_verification=index_verification,
    )


def verify_initialization_artifact(
    path: Path, *, contract_path: Path | None = None
) -> InitializationArtifact:
    """Validate exact bytes and runtime identity before any model is constructed."""
    contract, contract_sha256 = _load_json_contract(
        "resnet18-imagenet1k-v1.json", contract_path
    )
    if contract.get("schema_version") != 1:
        raise ValueError("Unsupported initialization contract schema")
    artifact = _mapping(contract, "artifact")
    if path.name != _text(artifact, "filename"):
        raise ValueError("Initialization artifact filename does not match contract")
    if not path.is_file():
        raise FileNotFoundError(f"Missing offline initialization artifact: {path}")
    size = path.stat().st_size
    digest = _sha256(path)
    if size != _positive_int(artifact, "size_bytes"):
        raise ValueError("Initialization artifact size mismatch")
    if digest != _text(artifact, "sha256"):
        raise ValueError("Initialization artifact SHA-256 mismatch")

    identity = _mapping(contract, "identity")
    expected_torch = _text(identity, "torch_version_base")
    expected_torchvision = _text(identity, "torchvision_version_base")
    if _base_version(torch.__version__) != expected_torch:
        raise RuntimeError("PyTorch runtime does not match initialization contract")
    if _base_version(importlib.metadata.version("torchvision")) != expected_torchvision:
        raise RuntimeError("torchvision runtime does not match initialization contract")
    if identity.get("weight_enum") != "ResNet18_Weights.IMAGENET1K_V1":
        raise ValueError("Initialization contract names an unsupported weight enum")
    if ResNet18_Weights.IMAGENET1K_V1.url != _text(artifact, "source_url"):
        raise RuntimeError("torchvision weight URL does not match frozen contract")
    policy = _mapping(contract, "construction_policy")
    required_policy = {
        "validate_before_model_construction": True,
        "network_download_allowed_at_runtime": False,
        "silent_weight_fallback_allowed": False,
        "random_initialization_fallback_allowed": False,
        "strict_state_dict_load": True,
    }
    if policy != required_policy:
        raise ValueError("Initialization construction policy has drifted")
    return InitializationArtifact(
        path=path,
        sha256=digest,
        size_bytes=size,
        contract_sha256=contract_sha256,
        contract=contract,
    )


def build_initialized_model(
    weight_path: Path,
    target_names: Sequence[str],
    *,
    contract_path: Path | None = None,
    aggregation: StudyAggregation = "mean",
) -> tuple[GoldResNet18, InitializationArtifact]:
    """Build only after exact offline bytes pass the frozen initialization contract."""
    initialization = verify_initialization_artifact(
        weight_path, contract_path=contract_path
    )
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Initialization artifact is not a nonempty state dict")
    backbone = resnet18(weights=None)
    backbone.load_state_dict(state_dict, strict=True)
    feature_count = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return (
        GoldResNet18(
            backbone,
            feature_count,
            len(target_names),
            aggregation=aggregation,
        ),
        initialization,
    )


def derive_gold_training_cohort(
    competition_root: Path,
    index_artifact_dir: Path,
    *,
    competition_contract_path: Path | None = None,
    gold_supervision_contract_path: Path | None = None,
    data_index_contract_path: Path | None = None,
    allow_verified_rebuilt_index: bool = False,
) -> GoldTrainingCohort:
    """Independently rebuild #7's all-eligible cohort and reject any digest drift."""
    gold_contract = load_verified_gold_contract(competition_contract_path)
    supervision, supervision_sha256 = _load_json_contract(
        "gold-supervision.json", gold_supervision_contract_path
    )
    _validate_frozen_supervision(supervision, gold_contract)
    if allow_verified_rebuilt_index:
        verify_index_artifact(index_artifact_dir)
    else:
        _verify_accepted_training_index(
            index_artifact_dir,
            gold_contract,
            data_index_contract_path=data_index_contract_path,
        )

    index_rows = _read_csv(index_artifact_dir / "studies.csv", _STUDY_COLUMNS)
    index_by_uid: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in index_rows:
        index_by_uid[row["StudyInstanceUID"]].append(row)
    train_rows = _read_csv(
        competition_root / gold_contract.train_filename, gold_contract.train_columns
    )
    identifier_counts = Counter(
        row[gold_contract.identifier_column].strip() for row in train_rows
    )

    studies: list[GoldStudy] = []
    for row in train_rows:
        study_uid = row[gold_contract.identifier_column].strip()
        labels = tuple(
            _parse_label(row[target]) for target in gold_contract.target_names
        )
        if (
            identifier_counts[study_uid] != 1
            or not study_uid
            or not any(label is not None for label in labels)
        ):
            continue
        linked = index_by_uid.get(study_uid, [])
        if (
            len(linked) == 1
            and linked[0]["split"] == "train"
            and _index_row_is_usable(linked[0])
        ):
            studies.append(GoldStudy(study_uid=study_uid, labels=labels))
    studies.sort(key=lambda item: item.study_uid.encode("utf-8"))
    digest = _cohort_digest(studies, gold_contract.target_names)
    canonical = _mapping(supervision, "canonical_cohort")
    expected_digest = _text(_mapping(canonical, "digest"), "value")
    if digest != expected_digest:
        raise ValueError("Independently derived membership/label digest mismatch")
    if len(studies) != _positive_int(canonical, "cohort_size"):
        raise ValueError("Independently derived cohort size mismatch")

    per_target = _target_counts(studies, gold_contract.target_names)
    if per_target != _mapping(canonical, "per_target"):
        raise ValueError("Independently derived target aggregates mismatch")
    unsupported = [
        target
        for target in gold_contract.target_names
        if per_target[target]["positive"] < 1 or per_target[target]["negative"] < 1
    ]
    if unsupported:
        raise ValueError(
            "Canonical cohort lacks both organizer-gold classes for: "
            + ", ".join(unsupported)
        )
    return GoldTrainingCohort(
        studies=tuple(studies),
        target_names=gold_contract.target_names,
        digest=digest,
        per_target=per_target,
        gold_contract_sha256=gold_contract.sha256,
        gold_supervision_contract_sha256=supervision_sha256,
    )


def load_indexed_studies(
    competition_root: Path,
    index_artifact_dir: Path,
    study_uids: Sequence[str],
    *,
    split: Literal["train", "test"],
) -> dict[str, IndexedStudy]:
    """Resolve selected index rows to the shared DICOM preprocessing inputs."""
    requested = set(study_uids)
    study_rows = _read_csv(index_artifact_dir / "studies.csv", _STUDY_COLUMNS)
    study_indices: dict[int, str] = {}
    seen_uids: set[str] = set()
    for row in study_rows:
        uid = row["StudyInstanceUID"]
        if row["split"] == split and uid in requested:
            if uid in seen_uids:
                raise ValueError("Index contains a duplicate requested study")
            seen_uids.add(uid)
            study_indices[int(row["study_index"])] = uid

    series_rows = _read_csv(index_artifact_dir / "series.csv", _SERIES_COLUMNS)
    series_metadata: dict[int, tuple[str, Path, int, AnatomicalPlane | None]] = {}
    for row in series_rows:
        study_index = int(row["study_index"])
        if study_index not in study_indices or int(row["usable_slice_count"]) <= 0:
            continue
        series_index = int(row["series_index"])
        relative_path = Path(row["relative_path"])
        expected_selected = row["selection_mask"].count("1")
        series_metadata[series_index] = (
            study_indices[study_index],
            relative_path,
            expected_selected,
            resolve_anatomical_plane(row["Anatomical_Plane"]),
        )

    selected_by_slice: dict[int, tuple[int, int]] = {}
    for row in _read_csv(index_artifact_dir / "selected_slices.csv", _SELECTED_COLUMNS):
        series_index = int(row["series_index"])
        if series_index in series_metadata:
            slice_index = int(row["slice_index"])
            if slice_index in selected_by_slice:
                raise ValueError("Selected index contains a duplicate slice")
            selected_by_slice[slice_index] = (series_index, int(row["slot"]))

    paths_by_series: dict[int, dict[int, Path]] = defaultdict(dict)
    root = competition_root.resolve()
    with (index_artifact_dir / "slices.csv").open(
        newline="", encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != _SLICE_COLUMNS:
            raise ValueError("Unexpected CSV columns for slices.csv")
        for row in reader:
            selected = selected_by_slice.get(int(row["slice_index"]))
            if selected is None:
                continue
            series_index, slot = selected
            filename = row["filename"]
            if Path(filename).name != filename:
                raise ValueError("Index slice filename must be a basename")
            relative_series = series_metadata[series_index][1]
            path = (competition_root / relative_series / filename).resolve()
            path.relative_to(root)
            if slot in paths_by_series[series_index]:
                raise ValueError("Selected index contains a duplicate series slot")
            paths_by_series[series_index][slot] = path

    series_by_study: dict[str, list[IndexedSeries]] = defaultdict(list)
    for series_index, (
        study_uid,
        _,
        expected_selected,
        plane,
    ) in series_metadata.items():
        slots = paths_by_series.get(series_index, {})
        if len(slots) != expected_selected or not slots:
            raise ValueError(
                "Selected slice rows do not match the series selection mask"
            )
        ordered_slots = sorted(slots)
        if ordered_slots != list(range(len(slots))):
            raise ValueError("Selected series slots must be contiguous from zero")
        series_by_study[study_uid].append(
            IndexedSeries(
                series_index=series_index,
                paths=tuple(slots[slot] for slot in ordered_slots),
                plane=plane,
            )
        )
    return {
        uid: IndexedStudy(
            study_uid=uid,
            series=tuple(
                sorted(series_by_study.get(uid, []), key=lambda x: x.series_index)
            ),
        )
        for uid in study_uids
    }


def resolve_anatomical_plane(value: str | None) -> AnatomicalPlane | None:
    """Resolve the competition's three anatomical-plane values."""
    normalized = (value or "").strip().casefold()
    return normalized if normalized in ANATOMICAL_PLANES else None


def masked_binary_cross_entropy(
    logits: Tensor, labels: Tensor, known_mask: Tensor
) -> Tensor:
    """Mean BCE over organizer-known elements only."""
    if logits.shape != labels.shape or logits.shape != known_mask.shape:
        raise ValueError("Logits, labels, and known mask must have identical shapes")
    known = known_mask.to(dtype=torch.bool)
    if not bool(known.any()):
        raise ValueError("Masked BCE requires at least one known organizer-gold label")
    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return losses[known].mean()


def train_epoch(
    *,
    model: GoldResNet18,
    optimizer: torch.optim.Optimizer,
    scaler: torch.GradScaler,
    indexed: Mapping[str, IndexedStudy],
    studies: Sequence[LabeledStudy],
    preprocessing: PreprocessingConfig,
    batch_size: int,
    seed: int,
    mixed_precision: bool,
    device: torch.device,
    epoch: int,
    collect_diagnostics: bool = False,
) -> dict[str, Any]:
    """Train one partition epoch with the shared decode, mask, and pooling path."""
    if batch_size <= 0:
        raise ValueError("Training batch size must be positive")
    ordered = list(studies)
    random.Random(seed + epoch).shuffle(ordered)
    losses: list[float] = []
    decode_errors: Counter[str] = Counter()
    decode_seconds = 0.0
    forward_backward_seconds = 0.0
    valid_slices = 0
    attempted = 0
    optimized = 0
    optimizer_steps = 0
    all_logits: list[Tensor] = []
    all_labels: list[Tensor] = []
    all_known: list[Tensor] = []
    start = time.perf_counter()
    model.train()
    for start_index in range(0, len(ordered), batch_size):
        batch = ordered[start_index : start_index + batch_size]
        decoded: list[DecodedStudy] = []
        decode_start = time.perf_counter()
        for study in batch:
            attempted += 1
            item = decode_study(
                indexed[study.study_uid], preprocessing, labels=study.labels
            )
            decode_errors.update(item.decode_error_counts)
            valid_slices += item.valid_slice_count
            if item.valid_slice_count:
                decoded.append(item)
        decode_seconds += time.perf_counter() - decode_start
        if not decoded:
            continue
        pixels, slice_mask, series_mask, plane_indices, labels, known = collate_studies(
            decoded
        )
        pixels = pixels.to(device)
        slice_mask = slice_mask.to(device)
        series_mask = series_mask.to(device)
        plane_indices = plane_indices.to(device)
        labels = labels.to(device)
        known = known.to(device)
        compute_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=mixed_precision and device.type == "cuda",
        ):
            logits = model(pixels, slice_mask, series_mask, plane_indices)
            loss = masked_binary_cross_entropy(logits, labels, known)
        scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_backward_seconds += time.perf_counter() - compute_start
        optimized += len(decoded)
        optimizer_steps += 1
        losses.append(float(loss.detach().cpu()))
        if collect_diagnostics:
            all_logits.append(logits.detach().float().cpu())
            all_labels.append(labels.detach().cpu())
            all_known.append(known.detach().cpu())
    training_seconds = time.perf_counter() - start
    if not losses:
        raise RuntimeError("No training study completed an optimizer update")
    record: dict[str, Any] = {
        "training_seconds": training_seconds,
        "decode_seconds": decode_seconds,
        "forward_backward_seconds": forward_backward_seconds,
        "attempted_study_count": attempted,
        "optimized_study_count": optimized,
        "optimizer_step_count": optimizer_steps,
        "valid_slice_count": valid_slices,
        "mean_loss": sum(losses) / len(losses),
        "decode_errors": dict(sorted(decode_errors.items())),
    }
    if collect_diagnostics:
        record.update(
            {
                "batch_losses": losses,
                "logits": torch.cat(all_logits),
                "labels": torch.cat(all_labels),
                "known": torch.cat(all_known),
            }
        )
    return record


def predict_partition(
    *,
    model: GoldResNet18,
    indexed: Mapping[str, IndexedStudy],
    studies: Sequence[PartitionStudy],
    preprocessing: PreprocessingConfig,
    mixed_precision: bool,
    device: torch.device,
) -> PartitionPrediction:
    """Predict one labeled partition with shared decode and padding semantics."""
    probabilities: list[tuple[float, ...]] = []
    successful: list[PartitionStudy] = []
    decode_errors: Counter[str] = Counter()
    decode_seconds = 0.0
    forward_seconds = 0.0
    valid_slices = 0
    model.eval()
    with torch.no_grad():
        for study in studies:
            decode_start = time.perf_counter()
            decoded = decode_study(
                indexed[study.study_uid], preprocessing, labels=study.labels
            )
            decode_seconds += time.perf_counter() - decode_start
            decode_errors.update(decoded.decode_error_counts)
            valid_slices += decoded.valid_slice_count
            if not decoded.valid_slice_count:
                raise RuntimeError("Prediction study has no decodable selected slices")
            pixels = decoded.pixels.unsqueeze(0).to(device)
            slice_mask = decoded.slice_mask.unsqueeze(0).to(device)
            series_mask = decoded.series_mask.unsqueeze(0).to(device)
            plane_indices = (
                decoded.plane_indices
                if decoded.plane_indices is not None
                else torch.full_like(
                    decoded.series_mask, UNKNOWN_PLANE_INDEX, dtype=torch.long
                )
            )
            plane_indices = plane_indices.unsqueeze(0).to(device)
            forward_start = time.perf_counter()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=mixed_precision and device.type == "cuda",
            ):
                output = torch.sigmoid(
                    model(pixels, slice_mask, series_mask, plane_indices)
                )[0]
            if device.type == "cuda":
                torch.cuda.synchronize()
            forward_seconds += time.perf_counter() - forward_start
            probabilities.append(tuple(float(value) for value in output.float().cpu()))
            successful.append(study)
    return PartitionPrediction(
        studies=tuple(successful),
        probabilities=tuple(probabilities),
        decode_seconds=decode_seconds,
        forward_seconds=forward_seconds,
        valid_slice_count=valid_slices,
        decode_errors=dict(sorted(decode_errors.items())),
    )


def require_approved_cuda(*, approved: bool) -> dict[str, object]:
    """Refuse GPU access without approval and prove a real CUDA operation."""
    if not approved:
        raise PermissionError("GPU execution requires explicit approval")
    if not torch.cuda.is_available():
        raise RuntimeError("Approved smoke training requires an available CUDA device")
    device = torch.device("cuda")
    result = (torch.ones(1, device=device) * 2).item()
    torch.cuda.synchronize()
    if result != 2.0:
        raise RuntimeError("CUDA readiness operation returned an unexpected result")
    properties = torch.cuda.get_device_properties(device)
    return {
        "approved": True,
        "operation": "(torch.ones(1, device='cuda') * 2).item()",
        "result": result,
        "device_name": properties.name,
        "device_total_memory_bytes": properties.total_memory,
        "cuda_version": torch.version.cuda,
    }


def run_gold_smoke_training(
    *,
    input_root: Path,
    work_root: Path,
    config: SmokeTrainingConfig,
    approved_gpu: bool = False,
    run_id: str | None = None,
    source_provenance: Mapping[str, object] | None = None,
    notebook: Mapping[str, object] | None = None,
) -> TrainingOutcome:
    """Train one approved epoch, reload complete state, and validate predictions."""
    cuda_evidence = require_approved_cuda(approved=approved_gpu)
    inputs = discover_training_inputs(input_root)
    submission_inputs = discover_submission_inputs(input_root)
    if submission_inputs.competition_root != inputs.competition_root:
        raise ValueError("Training and submission competition roots differ")
    cohort = derive_gold_training_cohort(
        inputs.competition_root,
        inputs.index_artifact_dir,
        allow_verified_rebuilt_index=(
            inputs.index_verification == "rebuilt_verified_artifact"
        ),
    )
    if cohort.target_names != submission_inputs.contract.target_names:
        raise ValueError("Training head target order differs from submission contract")

    training_index = load_indexed_studies(
        inputs.competition_root,
        inputs.index_artifact_dir,
        [study.study_uid for study in cohort.studies],
        split="train",
    )
    if any(not training_index[study.study_uid].series for study in cohort.studies):
        raise ValueError("Canonical cohort contains a study with no selected series")
    test_index = load_indexed_studies(
        inputs.competition_root,
        inputs.index_artifact_dir,
        submission_inputs.entity_ids,
        split="test",
    )

    seed_everything(config.seed)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model, initialization = build_initialized_model(
        inputs.weight_path, cohort.target_names
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.GradScaler("cuda", enabled=config.mixed_precision)

    priors = tuple(
        cohort.per_target[target]["positive"] / cohort.per_target[target]["known"]
        for target in cohort.target_names
    )
    training_record = train_epoch(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        indexed=training_index,
        studies=cohort.studies,
        preprocessing=config.preprocessing,
        batch_size=config.batch_size,
        seed=config.seed,
        mixed_precision=config.mixed_precision,
        device=device,
        epoch=0,
        collect_diagnostics=True,
    )
    epoch_losses = cast(list[float], training_record["batch_losses"])
    decode_errors = cast(dict[str, int], training_record["decode_errors"])
    attempted_studies = int(training_record["attempted_study_count"])
    optimized_studies = int(training_record["optimized_study_count"])
    valid_slices = int(training_record["valid_slice_count"])
    training_seconds = float(training_record["training_seconds"])
    metrics = _smoke_metrics(
        cast(Tensor, training_record["logits"]),
        cast(Tensor, training_record["labels"]),
        cast(Tensor, training_record["known"]),
        cohort.target_names,
    )

    artifact_dir = work_root / _ARTIFACT_DIRECTORY
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(f"Training artifact already exists: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / "checkpoint.pt"
    checkpoint_payload = build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        completed_epochs=1,
        target_names=cohort.target_names,
        priors=priors,
        preprocessing=config.preprocessing,
        cohort=cohort,
        initialization=initialization,
        configuration=_configuration_record(config),
        source_provenance=source_provenance or {},
        artifact_hashes={
            "index_manifest_sha256": _sha256(
                inputs.index_artifact_dir / "index-manifest.json"
            ),
            "initialization_artifact_sha256": initialization.sha256,
            "initialization_contract_sha256": initialization.contract_sha256,
            "gold_supervision_contract_sha256": (
                cohort.gold_supervision_contract_sha256
            ),
        },
    )
    save_checkpoint(checkpoint_path, checkpoint_payload)

    del model
    torch.cuda.empty_cache()
    reloaded, reloaded_optimizer, reloaded_scaler, loaded_checkpoint = (
        restore_training_state(checkpoint_path, device=device)
    )
    del reloaded_optimizer, reloaded_scaler
    if tuple(loaded_checkpoint["targets"]) != cohort.target_names:
        raise ValueError("Reloaded checkpoint target order mismatch")
    if loaded_checkpoint["cohort"]["membership_label_digest"] != cohort.digest:
        raise ValueError("Reloaded checkpoint cohort digest mismatch")

    prediction = _predict_submission_with_model(
        model=reloaded,
        submission_inputs=submission_inputs,
        indexed=test_index,
        work_root=artifact_dir,
        preprocessing=config.preprocessing,
        priors=priors,
        mixed_precision=config.mixed_precision,
        device=device,
    )
    submission = prediction.artifacts

    report_path = artifact_dir / "training-report.json"
    report = {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "artifact_type": "rsna_knee_gold_training_smoke",
        "scope": (
            "Engineering smoke diagnostics only; losses and metrics do not claim "
            "model quality, statistical adequacy, or competitive performance."
        ),
        "supervision": {
            "source": "organizer gold train.csv only",
            "report_derived_labels_used": False,
            "unknown_labels_masked_in_loss": True,
            "cohort_selection_rule_version": _COHORT_RULE_VERSION,
            "cohort_size": len(cohort.studies),
            "membership_label_digest": cohort.digest,
            "target_order": list(cohort.target_names),
            "per_target": cohort.per_target,
        },
        "initialization": {
            "identity": config.weight_identity,
            "artifact_sha256": initialization.sha256,
            "artifact_size_bytes": initialization.size_bytes,
            "contract_sha256": initialization.contract_sha256,
            "validated_before_model_construction": True,
            "network_fallback": False,
        },
        "execution": {
            "approved_gpu": True,
            "cuda_readiness": cuda_evidence,
            "epochs_completed": 1,
            "mixed_precision": config.mixed_precision,
            "attempted_cohort_studies": attempted_studies,
            "optimized_cohort_studies": optimized_studies,
            "training_seconds": training_seconds,
            "inference_seconds": prediction.inference_seconds,
            "valid_training_slices": valid_slices,
            "training_slices_per_second": (
                valid_slices / training_seconds if training_seconds else None
            ),
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
            "training_decode_errors": dict(sorted(decode_errors.items())),
            "inference_decode_errors": prediction.decode_errors,
            "mean_training_loss": sum(epoch_losses) / len(epoch_losses),
            "batch_losses": epoch_losses,
            "smoke_metrics": metrics,
        },
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": _sha256(checkpoint_path),
            "reloaded_before_inference": True,
            "optimizer_and_scaler_reloaded": True,
            "completed_epochs": loaded_checkpoint["completed_epochs"],
        },
        "submission": {
            "path": submission.submission_path.name,
            "validation_path": submission.validation_report_path.name,
            "fallback_count": prediction.fallback_count,
            "fallback_strategy": "per-target organizer-gold cohort priors",
            "prediction_count": prediction.prediction_count,
            "reloaded_checkpoint_used": True,
            "validator_passed": True,
        },
        "runtime": runtime_record(),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path = artifact_dir / "run-manifest.json"
    write_run_manifest(
        manifest_path,
        run_id=run_id or create_run_id("gold-training-smoke"),
        source={
            **dict(source_provenance or {}),
            "competition_contract_sha256": cohort.gold_contract_sha256,
            "gold_supervision_contract_sha256": (
                cohort.gold_supervision_contract_sha256
            ),
            "index_manifest_sha256": _sha256(
                inputs.index_artifact_dir / "index-manifest.json"
            ),
        },
        configuration=_configuration_record(config),
        notebook=notebook or {"kind": "approved_kaggle_gpu_gold_smoke"},
        artifacts=[
            checkpoint_path,
            report_path,
            submission.submission_path,
            submission.fallbacks_path,
            submission.validation_report_path,
        ],
        artifact_root=artifact_dir,
    )
    return TrainingOutcome(
        artifact_dir=artifact_dir,
        checkpoint_path=checkpoint_path,
        report_path=report_path,
        submission_path=submission.submission_path,
        validation_report_path=submission.validation_report_path,
        manifest_path=manifest_path,
    )


def predict_submission_from_checkpoint(
    *,
    input_root: Path,
    work_root: Path,
    checkpoint_path: Path,
    approved_gpu: bool = False,
    index_artifact_dir: Path | None = None,
) -> CheckpointPredictionOutcome:
    """Reload a gold checkpoint and predict through the shared submission path."""
    require_approved_cuda(approved=approved_gpu)
    inputs = discover_training_inputs(input_root, index_artifact_dir=index_artifact_dir)
    submission_inputs = discover_submission_inputs(input_root)
    if submission_inputs.competition_root != inputs.competition_root:
        raise ValueError("Training and submission competition roots differ")

    device = torch.device("cuda")
    model, checkpoint = load_training_checkpoint(checkpoint_path, device=device)
    targets = checkpoint.get("targets")
    if targets != list(submission_inputs.contract.target_names):
        raise ValueError("Checkpoint target order differs from submission contract")
    label_contract = _mapping(checkpoint, "label_contract")
    if (
        label_contract.get("report_derived_labels_used") is not False
        or label_contract.get("unknown_labels_masked") is not True
    ):
        raise ValueError("Checkpoint violates the gold-only label contract")
    artifact_hashes = _mapping(checkpoint, "artifact_hashes")
    if artifact_hashes.get("index_manifest_sha256") != _sha256(
        inputs.index_artifact_dir / "index-manifest.json"
    ):
        raise ValueError("Checkpoint and accepted index artifact differ")

    preprocessing_raw = _mapping(checkpoint, "preprocessing")
    preprocessing = PreprocessingConfig(
        slice_count=_positive_int(preprocessing_raw, "slice_count"),
        lower_percentile=_number(preprocessing_raw, "lower_percentile"),
        upper_percentile=_number(preprocessing_raw, "upper_percentile"),
        image_size=_positive_int(preprocessing_raw, "image_size"),
    )
    training = _mapping(_mapping(checkpoint, "configuration"), "training")
    mixed_precision = _boolean(training, "mixed_precision")
    raw_priors = checkpoint.get("priors")
    if not isinstance(raw_priors, list) or len(raw_priors) != len(targets):
        raise TypeError("Checkpoint fallback priors are malformed")
    priors = tuple(float(value) for value in raw_priors)
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in priors):
        raise ValueError("Checkpoint fallback priors are outside [0,1]")

    test_index = load_indexed_studies(
        inputs.competition_root,
        inputs.index_artifact_dir,
        submission_inputs.entity_ids,
        split="test",
    )
    prediction = _predict_submission_with_model(
        model=model,
        submission_inputs=submission_inputs,
        indexed=test_index,
        work_root=work_root,
        preprocessing=preprocessing,
        priors=priors,
        mixed_precision=mixed_precision,
        device=device,
    )
    return CheckpointPredictionOutcome(
        submission_path=prediction.artifacts.submission_path,
        fallbacks_path=prediction.artifacts.fallbacks_path,
        validation_report_path=prediction.artifacts.validation_report_path,
        inference_seconds=prediction.inference_seconds,
        prediction_count=prediction.prediction_count,
        fallback_count=prediction.fallback_count,
        decode_errors=prediction.decode_errors,
        checkpoint_sha256=_sha256(checkpoint_path),
    )


def _predict_submission_with_model(
    *,
    model: GoldResNet18,
    submission_inputs: SubmissionInputs,
    indexed: Mapping[str, IndexedStudy],
    work_root: Path,
    preprocessing: PreprocessingConfig,
    priors: Sequence[float],
    mixed_precision: bool,
    device: torch.device,
) -> _SubmissionPrediction:
    predictions: dict[str, tuple[float, ...]] = {}
    failures: dict[str, str] = {}
    decode_errors: Counter[str] = Counter()
    model.eval()
    inference_start = time.perf_counter()
    with torch.no_grad():
        for study_uid in submission_inputs.entity_ids:
            indexed_study = indexed.get(study_uid)
            if indexed_study is None or not indexed_study.series:
                raise RuntimeError(
                    "A missing selected index series cannot be used as a scoring fallback"
                )
            decoded = decode_study(indexed_study, preprocessing, labels=None)
            decode_errors.update(decoded.decode_error_counts)
            if decoded.valid_slice_count == 0:
                failures[study_uid] = "no_decodable_selected_slices"
                continue
            pixels = decoded.pixels.unsqueeze(0).to(device)
            slice_mask = decoded.slice_mask.unsqueeze(0).to(device)
            series_mask = decoded.series_mask.unsqueeze(0).to(device)
            plane_indices = (
                decoded.plane_indices
                if decoded.plane_indices is not None
                else torch.full_like(
                    decoded.series_mask, UNKNOWN_PLANE_INDEX, dtype=torch.long
                )
            )
            plane_indices = plane_indices.unsqueeze(0).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=mixed_precision and device.type == "cuda",
            ):
                probabilities = torch.sigmoid(
                    model(pixels, slice_mask, series_mask, plane_indices)
                )[0]
            predictions[study_uid] = tuple(
                float(value) for value in probabilities.float().cpu()
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_start
    artifacts = write_submission(
        inputs=submission_inputs,
        work_root=work_root,
        predictions=predictions,
        failures=failures,
        fallback_probability=priors,
    )
    return _SubmissionPrediction(
        artifacts=artifacts,
        inference_seconds=inference_seconds,
        prediction_count=len(predictions),
        fallback_count=len(failures),
        decode_errors=dict(sorted(decode_errors.items())),
    )


def run_local_cpu_check(
    *,
    weight_path: Path,
    work_root: Path,
    target_names: Sequence[str],
    initialization_contract_path: Path | None = None,
) -> dict[str, object]:
    """Exercise forward/backward/optimizer/checkpoint/reload/predict on CPU."""
    if not target_names:
        raise ValueError("CPU check requires at least one target")
    seed_everything(20260812)
    model, initialization = build_initialized_model(
        weight_path,
        target_names,
        contract_path=initialization_contract_path,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.GradScaler("cuda", enabled=False)
    model.train()
    pixels = torch.rand(1, 2, 2, 224, 224)
    slice_mask = torch.tensor([[[True, True], [False, False]]])
    series_mask = torch.tensor([[True, False]])
    labels = torch.tensor([[float(index % 2) for index in range(len(target_names))]])
    known = torch.tensor(
        [[index != len(target_names) - 1 for index in range(len(target_names))]]
    )
    optimizer.zero_grad(set_to_none=True)
    logits = model(pixels, slice_mask, series_mask)
    loss = masked_binary_cross_entropy(logits, labels, known)
    loss.backward()  # type: ignore[no-untyped-call]
    optimizer.step()

    work_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = work_root / "local-cpu-checkpoint.pt"
    preprocessing = PreprocessingConfig(8, 1, 99, 224)
    dummy_counts = {
        target: {"known": 1, "missing": 0, "positive": 1, "negative": 0}
        for target in target_names
    }
    cohort = GoldTrainingCohort(
        studies=(GoldStudy("synthetic-study", tuple(0 for _ in target_names)),),
        target_names=tuple(target_names),
        digest="synthetic-local-cpu-check",
        per_target=dummy_counts,
        gold_contract_sha256="synthetic",
        gold_supervision_contract_sha256="synthetic",
    )
    config = SmokeTrainingConfig(
        preprocessing=preprocessing,
        epochs=1,
        batch_size=1,
        learning_rate=1e-4,
        weight_decay=0,
        seed=20260812,
        mixed_precision=False,
        architecture="torchvision.models.resnet18",
        weight_identity="ResNet18_Weights.IMAGENET1K_V1",
    )
    payload = build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        completed_epochs=1,
        target_names=target_names,
        priors=(0.5,) * len(target_names),
        preprocessing=preprocessing,
        cohort=cohort,
        initialization=initialization,
        configuration=_configuration_record(config),
        source_provenance={"kind": "local_synthetic_cpu_check"},
        artifact_hashes={"initialization_artifact_sha256": initialization.sha256},
    )
    save_checkpoint(checkpoint_path, payload)
    reloaded, reloaded_optimizer, reloaded_scaler, reloaded_payload = (
        restore_training_state(checkpoint_path, device=torch.device("cpu"))
    )
    del reloaded_scaler
    reloaded.eval()
    with torch.no_grad():
        prediction = torch.sigmoid(reloaded(pixels, slice_mask, series_mask))
    if not bool(torch.isfinite(prediction).all()):
        raise RuntimeError("Reloaded local CPU prediction is not finite")
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "forward_shape": list(logits.shape),
        "loss": float(loss.detach()),
        "optimizer_state_entries": len(reloaded_payload["optimizer_state_dict"]),
        "optimizer_param_groups": len(reloaded_optimizer.param_groups),
        "prediction_shape": list(prediction.shape),
        "prediction_finite": True,
        "reloaded": True,
    }


def load_training_checkpoint(
    path: Path, *, device: torch.device
) -> tuple[GoldResNet18, dict[str, Any]]:
    """Strictly reload complete state without consulting a network or weight enum."""
    raw = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError("Training checkpoint must be a mapping")
    checkpoint = cast(dict[str, Any], raw)
    required = {
        "checkpoint_schema_version",
        "model_state_dict",
        "optimizer_state_dict",
        "scaler_state_dict",
        "completed_epochs",
        "targets",
        "priors",
        "preprocessing",
        "cohort",
        "label_contract",
        "initialization",
        "configuration",
        "code",
        "runtime",
        "artifact_hashes",
        "rng_state",
    }
    if set(checkpoint) != required:
        raise ValueError("Training checkpoint fields are incomplete or unexpected")
    if checkpoint["checkpoint_schema_version"] != _CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported training checkpoint schema")
    targets = checkpoint["targets"]
    if (
        not isinstance(targets, list)
        or not targets
        or not all(isinstance(target, str) and target for target in targets)
    ):
        raise TypeError("Checkpoint targets are malformed")
    model = _new_uninitialized_model(len(targets))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    return model, checkpoint


def restore_training_state(
    path: Path, *, device: torch.device
) -> tuple[
    GoldResNet18,
    torch.optim.Optimizer,
    torch.GradScaler,
    dict[str, Any],
]:
    """Restore model, optimizer, scaler, and RNG state for exact resumption."""
    model, checkpoint = load_training_checkpoint(path, device=device)
    configuration = _mapping(checkpoint, "configuration")
    training = _mapping(configuration, "training")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=_positive_number(training, "learning_rate"),
        weight_decay=_nonnegative_number(training, "weight_decay"),
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    mixed_precision = _boolean(training, "mixed_precision")
    scaler = torch.GradScaler("cuda", enabled=device.type == "cuda" and mixed_precision)
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    _restore_rng_state(_mapping(checkpoint, "rng_state"), device=device)
    return model, optimizer, scaler, checkpoint


def verify_training_artifact(artifact_dir: Path) -> dict[str, object]:
    """Prove the remote artifact against frozen contracts and its actual bytes."""
    manifest_path = artifact_dir / "run-manifest.json"
    report_path = artifact_dir / "training-report.json"
    validation_path = artifact_dir / "submission-validation.json"
    for path in (manifest_path, report_path, validation_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing training artifact: {path}")
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text()))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported run manifest schema")
    _verify_manifest_artifacts(artifact_dir, manifest)

    gold_contract = load_verified_gold_contract()
    submission_contract = load_verified_submission_contract()
    supervision_contract, supervision_sha256 = _load_json_contract(
        "gold-supervision.json", None
    )
    initialization_contract, initialization_sha256 = _load_json_contract(
        "resnet18-imagenet1k-v1.json", None
    )
    data_index_contract, _ = _load_json_contract("data-index.json", None)
    _validate_frozen_supervision(supervision_contract, gold_contract)
    canonical = _mapping(supervision_contract, "canonical_cohort")
    expected_cohort_size = _positive_int(canonical, "cohort_size")
    expected_digest = _text(_mapping(canonical, "digest"), "value")

    report = cast(dict[str, Any], json.loads(report_path.read_text()))
    if (
        report.get("schema_version") != _REPORT_SCHEMA_VERSION
        or report.get("artifact_type") != "rsna_knee_gold_training_smoke"
    ):
        raise ValueError("Training report identity is invalid")
    execution = _mapping(report, "execution")
    supervision = _mapping(report, "supervision")
    submission = _mapping(report, "submission")
    _verify_execution_report(
        execution,
        expected_cohort_size=expected_cohort_size,
        expected_target_counts=_mapping(canonical, "per_target"),
    )
    if (
        supervision.get("report_derived_labels_used") is not False
        or supervision.get("unknown_labels_masked_in_loss") is not True
        or supervision.get("cohort_selection_rule_version") != _COHORT_RULE_VERSION
        or supervision.get("cohort_size") != expected_cohort_size
        or supervision.get("membership_label_digest") != expected_digest
        or supervision.get("target_order") != list(gold_contract.target_names)
        or supervision.get("per_target") != canonical.get("per_target")
    ):
        raise ValueError("Training report violates the gold-only supervision contract")
    if (
        submission.get("reloaded_checkpoint_used") is not True
        or submission.get("validator_passed") is not True
        or submission.get("fallback_strategy")
        != "per-target organizer-gold cohort priors"
    ):
        raise ValueError("Training report lacks reloaded submission evidence")

    checkpoint_path = artifact_dir / "checkpoint.pt"
    _, checkpoint = load_training_checkpoint(
        checkpoint_path, device=torch.device("cpu")
    )
    expected_index_manifest = _text(
        _mapping(_mapping(data_index_contract, "run_provenance"), "outputs"),
        "manifest_sha256",
    )
    _verify_checkpoint_contract(
        checkpoint,
        report=report,
        gold_contract=gold_contract,
        supervision_sha256=supervision_sha256,
        expected_cohort_size=expected_cohort_size,
        expected_digest=expected_digest,
        initialization_contract=initialization_contract,
        initialization_sha256=initialization_sha256,
        expected_index_manifest=expected_index_manifest,
    )
    if _mapping(report, "checkpoint").get("sha256") != _sha256(checkpoint_path):
        raise ValueError("Training report checkpoint hash differs from actual bytes")
    validation = _verify_submission_artifacts(
        artifact_dir,
        submission=submission,
        expected_columns=submission_contract.ordered_columns,
        competition_contract_sha256=submission_contract.sha256,
    )
    source = _mapping(manifest, "source")
    if (
        source.get("competition_contract_sha256") != gold_contract.sha256
        or source.get("gold_supervision_contract_sha256") != supervision_sha256
        or source.get("index_manifest_sha256") != expected_index_manifest
    ):
        raise ValueError("Run manifest source contracts do not match frozen inputs")
    return {
        "valid": True,
        "run_id": manifest.get("run_id"),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "cohort_digest": supervision.get("membership_label_digest"),
        "target_count": len(checkpoint["targets"]),
        "submission_sha256": validation.get("submission_sha256"),
    }


def _verify_manifest_artifacts(artifact_dir: Path, manifest: Mapping[str, Any]) -> None:
    records = manifest.get("artifacts")
    if not isinstance(records, list):
        raise TypeError("Training manifest artifact inventory is malformed")
    expected = {
        "checkpoint.pt",
        "training-report.json",
        "submission.csv",
        "submission-fallbacks.csv",
        "submission-validation.json",
    }
    paths: set[str] = set()
    for raw_record in records:
        if not isinstance(raw_record, dict):
            raise TypeError("Training manifest artifact record is malformed")
        relative = raw_record.get("path")
        if not isinstance(relative, str) or relative in paths:
            raise ValueError(
                "Training manifest artifact paths are malformed or duplicate"
            )
        paths.add(relative)
        path = (artifact_dir / relative).resolve()
        path.relative_to(artifact_dir.resolve())
        if (
            not path.is_file()
            or raw_record.get("sha256") != _sha256(path)
            or raw_record.get("size_bytes") != path.stat().st_size
        ):
            raise ValueError(f"Training artifact does not match manifest: {relative}")
    if paths != expected:
        raise ValueError(
            "Training manifest artifact inventory is incomplete or unexpected"
        )


def _verify_execution_report(
    execution: Mapping[str, Any],
    *,
    expected_cohort_size: int,
    expected_target_counts: Mapping[str, Any],
) -> None:
    cuda = _mapping(execution, "cuda_readiness")
    if (
        execution.get("approved_gpu") is not True
        or execution.get("epochs_completed") != 1
        or execution.get("mixed_precision") is not True
        or execution.get("attempted_cohort_studies") != expected_cohort_size
        or execution.get("optimized_cohort_studies") != expected_cohort_size
        or cuda.get("approved") is not True
        or cuda.get("operation") != "(torch.ones(1, device='cuda') * 2).item()"
        or cuda.get("result") != 2.0
        or not _is_positive_number(cuda.get("device_total_memory_bytes"))
        or not isinstance(cuda.get("device_name"), str)
        or not cuda.get("device_name")
    ):
        raise ValueError("Training report lacks complete approved CUDA evidence")
    numeric_positive = (
        "training_seconds",
        "inference_seconds",
        "valid_training_slices",
        "training_slices_per_second",
        "peak_cuda_memory_bytes",
    )
    if any(not _is_positive_number(execution.get(key)) for key in numeric_positive):
        raise ValueError(
            "Training report has invalid time, memory, or throughput evidence"
        )
    loss = execution.get("mean_training_loss")
    batch_losses = execution.get("batch_losses")
    if (
        not _is_nonnegative_number(loss)
        or not isinstance(batch_losses, list)
        or len(batch_losses) != expected_cohort_size
        or any(not _is_nonnegative_number(value) for value in batch_losses)
    ):
        raise ValueError("Training report loss evidence is invalid")
    for key in ("training_decode_errors", "inference_decode_errors"):
        values = execution.get(key)
        if not isinstance(values, dict) or any(
            not isinstance(code, str)
            or not code
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            for code, count in values.items()
        ):
            raise ValueError("Training report decode-error evidence is invalid")
    metrics = _mapping(_mapping(execution, "smoke_metrics"), "per_target")
    if set(metrics) != set(expected_target_counts):
        raise ValueError("Training report smoke metrics do not cover every target")
    for target in expected_target_counts:
        expected = _mapping(expected_target_counts, target)
        metric = _mapping(metrics, target)
        if (
            metric.get("known") != expected.get("known")
            or metric.get("positive") != expected.get("positive")
            or metric.get("negative") != expected.get("negative")
            or not _is_nonnegative_number(metric.get("binary_cross_entropy"))
            or not _is_probability(metric.get("accuracy_at_0_5"))
            or not _is_probability(metric.get("roc_auc"))
        ):
            raise ValueError(f"Training report smoke metric is invalid: {target}")


def _verify_checkpoint_contract(
    checkpoint: Mapping[str, Any],
    *,
    report: Mapping[str, Any],
    gold_contract: VerifiedGoldContract,
    supervision_sha256: str,
    expected_cohort_size: int,
    expected_digest: str,
    initialization_contract: Mapping[str, Any],
    initialization_sha256: str,
    expected_index_manifest: str,
) -> None:
    if checkpoint.get("completed_epochs") != 1:
        raise ValueError("Training checkpoint does not contain one completed epoch")
    if checkpoint.get("targets") != list(gold_contract.target_names):
        raise ValueError("Training checkpoint target order differs from contract")
    priors = checkpoint.get("priors")
    if (
        not isinstance(priors, list)
        or len(priors) != len(gold_contract.target_names)
        or any(not _is_probability(value) for value in priors)
    ):
        raise ValueError("Training checkpoint priors are invalid")
    cohort = _mapping(checkpoint, "cohort")
    label_contract = _mapping(checkpoint, "label_contract")
    if (
        cohort.get("selection_rule_version") != _COHORT_RULE_VERSION
        or cohort.get("membership_label_digest") != expected_digest
        or cohort.get("size") != expected_cohort_size
        or label_contract.get("sha256") != gold_contract.sha256
        or label_contract.get("unknown_labels_masked") is not True
        or label_contract.get("report_derived_labels_used") is not False
    ):
        raise ValueError("Training checkpoint supervision contract is invalid")
    preprocessing = _mapping(checkpoint, "preprocessing")
    if (
        preprocessing.get("slice_count") != 8
        or preprocessing.get("image_size") != 224
        or preprocessing.get("shared_contract") != "preprocess_series_result"
        or preprocessing.get("channel_adapter") != "grayscale_repeat_to_rgb"
    ):
        raise ValueError("Training checkpoint preprocessing contract is invalid")
    initialization = _mapping(checkpoint, "initialization")
    expected_artifact = _mapping(initialization_contract, "artifact")
    if (
        initialization.get("contract") != initialization_contract
        or initialization.get("contract_sha256") != initialization_sha256
        or initialization.get("artifact_sha256") != expected_artifact.get("sha256")
        or initialization.get("artifact_size_bytes")
        != expected_artifact.get("size_bytes")
    ):
        raise ValueError("Training checkpoint initialization contract is invalid")
    configuration = _mapping(_mapping(checkpoint, "configuration"), "training")
    if (
        configuration.get("epochs") != 1
        or configuration.get("batch_size") != 1
        or configuration.get("mixed_precision") is not True
        or configuration.get("architecture") != "torchvision.models.resnet18"
        or configuration.get("weight_identity") != "ResNet18_Weights.IMAGENET1K_V1"
    ):
        raise ValueError("Training checkpoint configuration is invalid")
    optimizer = _mapping(checkpoint, "optimizer_state_dict")
    scaler = _mapping(checkpoint, "scaler_state_dict")
    if (
        not isinstance(optimizer.get("state"), dict)
        or not optimizer.get("state")
        or not isinstance(optimizer.get("param_groups"), list)
        or not optimizer.get("param_groups")
        or set(scaler)
        != {
            "scale",
            "growth_factor",
            "backoff_factor",
            "growth_interval",
            "_growth_tracker",
        }
    ):
        raise ValueError("Training checkpoint optimizer or scaler state is incomplete")
    code = _mapping(checkpoint, "code")
    if (
        not _is_sha256(code.get("jupytext_source_sha256"))
        or not _is_sha256(code.get("package_archive_sha256"))
        or not _is_sha256(code.get("training_config_sha256"))
        or code.get("weight_dataset_id") != "plthon/torchvision-resnet18-imagenet1k-v1"
    ):
        raise ValueError("Training checkpoint code provenance is incomplete")
    artifact_hashes = _mapping(checkpoint, "artifact_hashes")
    if artifact_hashes != {
        "index_manifest_sha256": expected_index_manifest,
        "initialization_artifact_sha256": expected_artifact.get("sha256"),
        "initialization_contract_sha256": initialization_sha256,
        "gold_supervision_contract_sha256": supervision_sha256,
    }:
        raise ValueError("Training checkpoint artifact hashes are invalid")
    checkpoint_report = _mapping(report, "checkpoint")
    checkpoint_path = Path(_text(checkpoint_report, "path"))
    if (
        checkpoint_path != Path("checkpoint.pt")
        or checkpoint_report.get("reloaded_before_inference") is not True
        or checkpoint_report.get("optimizer_and_scaler_reloaded") is not True
        or checkpoint_report.get("completed_epochs") != 1
    ):
        raise ValueError("Training report lacks complete checkpoint reload evidence")
    if checkpoint.get("runtime") != report.get("runtime"):
        raise ValueError("Checkpoint and report runtime provenance differ")


def _verify_submission_artifacts(
    artifact_dir: Path,
    *,
    submission: Mapping[str, Any],
    expected_columns: Sequence[str],
    competition_contract_sha256: str,
) -> dict[str, Any]:
    submission_path = artifact_dir / "submission.csv"
    validation_path = artifact_dir / "submission-validation.json"
    fallbacks_path = artifact_dir / "submission-fallbacks.csv"
    validation = cast(dict[str, Any], json.loads(validation_path.read_text()))
    if (
        validation.get("valid") is not True
        or validation.get("competition_contract_sha256") != competition_contract_sha256
        or validation.get("columns") != list(expected_columns)
        or validation.get("target_count") != len(expected_columns) - 1
        or validation.get("submission_sha256") != _sha256(submission_path)
    ):
        raise ValueError("Submission validation report is inconsistent with contracts")
    with submission_path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.reader(file))
    if not rows or tuple(rows[0]) != tuple(expected_columns):
        raise ValueError("Reloaded submission columns differ from contract")
    data_rows = rows[1:]
    if len(data_rows) != validation.get("row_count"):
        raise ValueError("Reloaded submission row count differs from validation")
    identifiers: list[str] = []
    probabilities: list[float] = []
    for row in data_rows:
        if len(row) != len(expected_columns) or not row[0].strip():
            raise ValueError("Reloaded submission contains a malformed row")
        identifiers.append(row[0].strip())
        try:
            values = [float(value) for value in row[1:]]
        except ValueError as error:
            raise ValueError(
                "Reloaded submission contains a nonnumeric value"
            ) from error
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("Reloaded submission probability is outside [0,1]")
        probabilities.extend(values)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Reloaded submission contains duplicate identifiers")
    probability_summary = _mapping(validation, "probabilities")
    if probabilities and (
        probability_summary.get("minimum") != min(probabilities)
        or probability_summary.get("maximum") != max(probabilities)
    ):
        raise ValueError("Reloaded submission probability summary is inconsistent")
    with fallbacks_path.open(newline="", encoding="utf-8-sig") as file:
        fallback_rows = list(csv.reader(file))
    if not fallback_rows or fallback_rows[0] != [expected_columns[0], "reason"]:
        raise ValueError("Submission fallback ledger columns are invalid")
    fallback_count = len(fallback_rows) - 1
    fallback_validation = _mapping(validation, "fallbacks")
    prediction_count = submission.get("prediction_count")
    if (
        not isinstance(prediction_count, int)
        or isinstance(prediction_count, bool)
        or prediction_count < 0
        or fallback_count != submission.get("fallback_count")
        or fallback_count != fallback_validation.get("count")
        or fallback_validation.get("path") != "submission-fallbacks.csv"
        or prediction_count + fallback_count != len(data_rows)
    ):
        raise ValueError("Submission fallback ledger is inconsistent")
    return validation


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _is_positive_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value > 0
    )


def _is_nonnegative_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value >= 0
    )


def _is_probability(value: object) -> bool:
    return _is_nonnegative_number(value) and cast(float, value) <= 1


def _new_uninitialized_model(
    target_count: int, *, aggregation: StudyAggregation = "mean"
) -> GoldResNet18:
    backbone = resnet18(weights=None)
    feature_count = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return GoldResNet18(backbone, feature_count, target_count, aggregation=aggregation)


def decode_study(
    indexed: IndexedStudy,
    preprocessing: PreprocessingConfig,
    *,
    labels: Sequence[Label] | None,
) -> DecodedStudy:
    if not indexed.series:
        return DecodedStudy(
            pixels=torch.zeros(0, preprocessing.slice_count, 224, 224),
            slice_mask=torch.zeros(0, preprocessing.slice_count, dtype=torch.bool),
            series_mask=torch.zeros(0, dtype=torch.bool),
            labels=None,
            label_mask=None,
            valid_slice_count=0,
            decode_error_counts={},
            plane_indices=torch.empty(0, dtype=torch.long),
        )
    pixels: list[Tensor] = []
    masks: list[Tensor] = []
    errors: Counter[str] = Counter()
    for series in indexed.series:
        result = preprocess_series_result(series.paths, preprocessing)
        pixels.append(torch.from_numpy(result.pixels))
        masks.append(torch.from_numpy(result.mask))
        errors.update(failure.code for failure in result.failures)
    pixel_tensor = torch.stack(pixels)
    mask_tensor = torch.stack(masks)
    label_tensor: Tensor | None = None
    label_mask: Tensor | None = None
    if labels is not None:
        label_tensor = torch.tensor(
            [0.0 if value is None else float(value) for value in labels]
        )
        label_mask = torch.tensor([value is not None for value in labels])
    return DecodedStudy(
        pixels=pixel_tensor,
        slice_mask=mask_tensor,
        series_mask=mask_tensor.any(dim=1),
        labels=label_tensor,
        label_mask=label_mask,
        valid_slice_count=int(mask_tensor.sum()),
        decode_error_counts=dict(errors),
        plane_indices=torch.tensor(
            [
                UNKNOWN_PLANE_INDEX
                if series.plane is None
                else ANATOMICAL_PLANES.index(series.plane)
                for series in indexed.series
            ],
            dtype=torch.long,
        ),
    )


def collate_studies(
    studies: Sequence[DecodedStudy],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    if not studies or any(item.labels is None for item in studies):
        raise ValueError("Training collation requires labeled studies")
    maximum_series = max(item.pixels.shape[0] for item in studies)
    slice_count = studies[0].pixels.shape[1]
    height, width = studies[0].pixels.shape[-2:]
    pixels = torch.zeros(len(studies), maximum_series, slice_count, height, width)
    slice_mask = torch.zeros(
        len(studies), maximum_series, slice_count, dtype=torch.bool
    )
    series_mask = torch.zeros(len(studies), maximum_series, dtype=torch.bool)
    plane_indices = torch.full(
        (len(studies), maximum_series), UNKNOWN_PLANE_INDEX, dtype=torch.long
    )
    for index, item in enumerate(studies):
        series_count = item.pixels.shape[0]
        pixels[index, :series_count] = item.pixels
        slice_mask[index, :series_count] = item.slice_mask
        series_mask[index, :series_count] = item.series_mask
        if item.plane_indices is not None:
            plane_indices[index, :series_count] = item.plane_indices
    labels = torch.stack([cast(Tensor, item.labels) for item in studies])
    known = torch.stack([cast(Tensor, item.label_mask) for item in studies])
    return pixels, slice_mask, series_mask, plane_indices, labels, known


def build_checkpoint_payload(
    *,
    model: GoldResNet18,
    optimizer: torch.optim.Optimizer,
    scaler: torch.GradScaler,
    completed_epochs: int,
    target_names: Sequence[str],
    priors: Sequence[float],
    preprocessing: PreprocessingConfig,
    cohort: GoldTrainingCohort,
    initialization: InitializationArtifact,
    configuration: Mapping[str, object],
    source_provenance: Mapping[str, object],
    artifact_hashes: Mapping[str, str],
) -> dict[str, object]:
    return {
        "checkpoint_schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "completed_epochs": completed_epochs,
        "targets": list(target_names),
        "priors": list(priors),
        "preprocessing": {
            **asdict(preprocessing),
            "shared_contract": "preprocess_series_result",
            "channel_adapter": "grayscale_repeat_to_rgb",
            "imagenet_mean": [0.485, 0.456, 0.406],
            "imagenet_std": [0.229, 0.224, 0.225],
        },
        "cohort": {
            "selection_rule_version": _COHORT_RULE_VERSION,
            "membership_label_digest": cohort.digest,
            "size": len(cohort.studies),
        },
        "label_contract": {
            "sha256": cohort.gold_contract_sha256,
            "unknown_labels_masked": True,
            "report_derived_labels_used": False,
        },
        "initialization": {
            "contract": initialization.contract,
            "contract_sha256": initialization.contract_sha256,
            "artifact_sha256": initialization.sha256,
            "artifact_size_bytes": initialization.size_bytes,
        },
        "configuration": dict(configuration),
        "code": dict(source_provenance),
        "runtime": runtime_record(),
        "artifact_hashes": dict(artifact_hashes),
        "rng_state": {
            "python": random.getstate(),
            "numpy": _numpy_rng_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else [],
        },
    }


def save_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _configuration_record(config: SmokeTrainingConfig) -> dict[str, object]:
    return {
        "preprocessing": asdict(config.preprocessing),
        "training": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "seed": config.seed,
            "mixed_precision": config.mixed_precision,
            "architecture": config.architecture,
            "weight_identity": config.weight_identity,
        },
    }


def runtime_record() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torchvision": importlib.metadata.version("torchvision"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }


def _numpy_rng_state() -> dict[str, object]:
    algorithm, values, position, has_gaussian, cached = cast(
        tuple[str, np.ndarray, int, int, float], np.random.get_state()
    )
    return {
        "algorithm": algorithm,
        "values": values.tolist(),
        "position": position,
        "has_gaussian": has_gaussian,
        "cached_gaussian": cached,
    }


def _restore_rng_state(state: Mapping[str, Any], *, device: torch.device) -> None:
    python_state = state.get("python")
    numpy_state = state.get("numpy")
    torch_cpu = state.get("torch_cpu")
    torch_cuda = state.get("torch_cuda")
    if not isinstance(python_state, tuple):
        raise TypeError("Checkpoint Python RNG state is malformed")
    if not isinstance(numpy_state, dict):
        raise TypeError("Checkpoint NumPy RNG state is malformed")
    if not isinstance(torch_cpu, Tensor):
        raise TypeError("Checkpoint torch CPU RNG state is malformed")
    values = numpy_state.get("values")
    if not isinstance(values, list) or not all(
        isinstance(value, int) for value in values
    ):
        raise TypeError("Checkpoint NumPy RNG values are malformed")
    random.setstate(python_state)
    np.random.set_state(
        (
            _text(numpy_state, "algorithm"),
            np.asarray(values, dtype=np.uint32),
            _nonnegative_int(numpy_state, "position"),
            int(numpy_state.get("has_gaussian", 0)),
            _number(numpy_state, "cached_gaussian"),
        )
    )
    torch.set_rng_state(torch_cpu.cpu())
    if device.type == "cuda":
        if not isinstance(torch_cuda, list) or not all(
            isinstance(value, Tensor) for value in torch_cuda
        ):
            raise TypeError("Checkpoint torch CUDA RNG state is malformed")
        torch.cuda.set_rng_state_all([value.cpu() for value in torch_cuda])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _smoke_metrics(
    logits: Tensor,
    labels: Tensor,
    known: Tensor,
    targets: Sequence[str],
) -> dict[str, object]:
    result: dict[str, object] = {
        "scope": "online training diagnostics; not a quality estimate",
        "per_target": {},
    }
    per_target = cast(dict[str, object], result["per_target"])
    for index, target in enumerate(targets):
        mask = known[:, index].bool()
        target_logits = logits[mask, index]
        target_labels = labels[mask, index]
        positives = target_logits[target_labels == 1]
        negatives = target_logits[target_labels == 0]
        auc: float | None = None
        if len(positives) and len(negatives):
            comparisons = (positives[:, None] > negatives[None, :]).float() + 0.5 * (
                positives[:, None] == negatives[None, :]
            ).float()
            auc = float(comparisons.mean())
        target_loss = F.binary_cross_entropy_with_logits(target_logits, target_labels)
        per_target[target] = {
            "known": int(mask.sum()),
            "positive": int((target_labels == 1).sum()),
            "negative": int((target_labels == 0).sum()),
            "binary_cross_entropy": float(target_loss),
            "accuracy_at_0_5": float(
                ((target_logits >= 0) == target_labels.bool()).float().mean()
            ),
            "roc_auc": auc,
        }
    return result


def _validate_frozen_supervision(
    supervision: Mapping[str, Any], gold_contract: VerifiedGoldContract
) -> None:
    if supervision.get("schema_version") != 1:
        raise ValueError("Unsupported gold-supervision contract schema")
    label_contract = _mapping(supervision, "label_contract")
    canonical = _mapping(supervision, "canonical_cohort")
    sufficiency = _mapping(supervision, "smoke_sufficiency")
    if label_contract.get("target_order") != list(gold_contract.target_names):
        raise ValueError("#7 target order differs from verified competition contract")
    if label_contract.get("status") != "verified":
        raise ValueError("#7 label contract is not verified")
    if sufficiency.get("decision") != "smoke-sufficient":
        raise ValueError("#7 did not classify the cohort as smoke-sufficient")
    if sufficiency.get("unsupported_targets") != []:
        raise ValueError("#7 reports unsupported smoke targets")
    if canonical.get("selection_rule_version") != _COHORT_RULE_VERSION:
        raise ValueError("#7 cohort selection rule version is unsupported")
    digest = _mapping(canonical, "digest")
    if digest.get("canonicalization_version") != _COHORT_DIGEST_VERSION:
        raise ValueError("#7 cohort canonicalization version is unsupported")
    version = _mapping(canonical, "label_contract_version")
    if version.get("sha256") != gold_contract.sha256:
        raise ValueError("#7 label-contract digest differs from current contract")


def _verify_accepted_training_index(
    artifact_dir: Path,
    gold_contract: VerifiedGoldContract,
    *,
    data_index_contract_path: Path | None,
) -> None:
    index_contract, _ = _load_json_contract("data-index.json", data_index_contract_path)
    current = _mapping(index_contract, "current_competition_contract")
    if current.get("sha256") != gold_contract.sha256:
        raise ValueError("Accepted index is incompatible with gold contract")
    expected_manifest = _text(
        _mapping(_mapping(index_contract, "run_provenance"), "outputs"),
        "manifest_sha256",
    )
    manifest_path = artifact_dir / "index-manifest.json"
    if not manifest_path.is_file() or _sha256(manifest_path) != expected_manifest:
        raise ValueError("Index manifest does not match the accepted Kaggle output")
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text()))
    inventory = _mapping(manifest, "files")
    required_columns = {
        "studies.csv": _STUDY_COLUMNS,
        "series.csv": _SERIES_COLUMNS,
        "slices.csv": _SLICE_COLUMNS,
        "selected_slices.csv": _SELECTED_COLUMNS,
    }
    for name, columns in required_columns.items():
        record = _mapping(inventory, name)
        path = artifact_dir / name
        if (
            not path.is_file()
            or record.get("sha256") != _sha256(path)
            or record.get("size_bytes") != path.stat().st_size
            or record.get("columns") != list(columns)
        ):
            raise ValueError(f"Accepted index file is invalid: {name}")


def _target_counts(
    studies: Sequence[GoldStudy], targets: Sequence[str]
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for index, target in enumerate(targets):
        values = [study.labels[index] for study in studies]
        positive = sum(value == 1 for value in values)
        negative = sum(value == 0 for value in values)
        result[target] = {
            "known": positive + negative,
            "missing": sum(value is None for value in values),
            "positive": positive,
            "negative": negative,
        }
    return result


def _cohort_digest(studies: Sequence[GoldStudy], targets: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(
        _canonical_json({"schema": _COHORT_DIGEST_VERSION, "targets": list(targets)})
    )
    digest.update(b"\n")
    for study in sorted(studies, key=lambda item: item.study_uid.encode("utf-8")):
        digest.update(_canonical_json([study.study_uid, list(study.labels)]))
        digest.update(b"\n")
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _parse_label(raw: str) -> Label:
    stripped = raw.strip()
    if not stripped:
        return None
    try:
        value = Decimal(stripped)
    except InvalidOperation as error:
        raise ValueError("Organizer label is not numeric") from error
    if not value.is_finite() or value not in {Decimal(0), Decimal(1)}:
        raise ValueError("Organizer label is outside {0,1,missing}")
    return 1 if value == 1 else 0


def _index_row_is_usable(row: Mapping[str, str]) -> bool:
    return int(row["usable_series_count"]) > 0 and int(row["usable_slice_count"]) > 0


def _read_csv(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != tuple(columns):
            raise ValueError(f"Unexpected CSV columns for {path.name}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV row in {path.name}")
    return rows


def _load_json_contract(name: str, path: Path | None) -> tuple[dict[str, Any], str]:
    if path is not None:
        raw = path.read_bytes()
    else:
        repository_path = Path(__file__).resolve().parents[2] / "contracts" / name
        raw = (
            repository_path.read_bytes()
            if repository_path.is_file()
            else files("rsna_knee").joinpath(name).read_bytes()
        )
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise TypeError(f"Contract {name} must be a JSON object")
    return cast(dict[str, Any], parsed), hashlib.sha256(raw).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _base_version(version: str) -> str:
    return version.split("+", 1)[0]


def _mapping(parent: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Contract field {key} must be an object")
    return cast(dict[str, Any], value)


def _text(parent: Mapping[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"Field {key} must be a nonempty string")
    return value


def _positive_int(parent: Mapping[str, Any], key: str) -> int:
    value = parent.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TypeError(f"Field {key} must be a positive integer")
    return value


def _nonnegative_int(parent: Mapping[str, Any], key: str) -> int:
    value = parent.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TypeError(f"Field {key} must be a nonnegative integer")
    return value


def _number(parent: Mapping[str, Any], key: str) -> float:
    value = parent.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"Field {key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Field {key} must be finite")
    return result


def _positive_number(parent: Mapping[str, Any], key: str) -> float:
    value = _number(parent, key)
    if value <= 0:
        raise ValueError(f"Field {key} must be positive")
    return value


def _nonnegative_number(parent: Mapping[str, Any], key: str) -> float:
    value = _number(parent, key)
    if value < 0:
        raise ValueError(f"Field {key} must be nonnegative")
    return value


def _boolean(parent: Mapping[str, Any], key: str) -> bool:
    value = parent.get(key)
    if not isinstance(value, bool):
        raise TypeError(f"Field {key} must be boolean")
    return value
