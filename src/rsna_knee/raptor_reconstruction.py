"""Best-evidence reconstruction of the public Raptor CoAtNet training recipe."""

from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.metadata
import json
import platform
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset

from .holdout import evaluate_point_predictions
from .provenance import create_run_id, write_run_manifest
from .raptor_reference import (
    ARCHITECTURE,
    MODEL_INPUT_SIZE,
    TARGET_NAMES,
    RaptorCheckpointMetadata,
    RaptorClassifier,
    load_raptor_training_reference_contract,
)
from .training import require_approved_cuda, save_checkpoint

type ReconstructionStage = Literal["smoke", "full"]
type StateDict = dict[str, Tensor]
type BackboneFactory = Callable[[], nn.Module]

_CHECKPOINT_KEYS = {
    "model",
    "gold_auc",
    "aucs",
    "src",
    "res",
    "arch",
    "lab",
    "epoch",
}
_REPORT_NAME = "raptor-reconstruction-report.json"
_BEST_CHECKPOINT_NAME = "raptor-reconstruction-best.pt"
_SWA_CHECKPOINT_NAME = "raptor-reconstruction-swa.pt"
_MANIFEST_NAME = "run-manifest.json"
_EVIDENCE_NAMES = (
    _BEST_CHECKPOINT_NAME,
    _SWA_CHECKPOINT_NAME,
    _REPORT_NAME,
    _MANIFEST_NAME,
)


@dataclass(frozen=True, slots=True)
class RaptorReconstructionConfig:
    architecture: str
    resolution: int
    epochs: int
    batch_size: int
    validation_batch_size: int
    training_windows: int
    evaluation_windows: int
    backbone_learning_rate: float
    head_learning_rate: float
    weight_decay: float
    workers: int
    topk: int
    seed: int
    scheduler_pct_start: float
    gradient_clip_norm: float
    gain_minimum: float
    gain_maximum: float

    def record(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True, slots=True)
class PretrainedWeightIdentity:
    repository: str
    requested_revision: str | None
    resolved_revision: str
    filename: str
    sha256: str
    size_bytes: int
    cache_path: Path

    def record(self) -> dict[str, object]:
        return {
            "evidence_semantic": "reconstruction_specific_pinned_identity",
            "repository": self.repository,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "filename": self.filename,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class CorpusPart:
    ids: tuple[str, ...]
    masks: np.ndarray
    volumes: np.ndarray


@dataclass(frozen=True, slots=True)
class RaptorCorpus:
    parts: tuple[CorpusPart, ...]
    ids: tuple[str, ...]

    def sample(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        offset = index
        for part in self.parts:
            if offset < len(part.ids):
                return part.volumes[offset], part.masks[offset]
            offset -= len(part.ids)
        raise IndexError(index)


@dataclass(frozen=True, slots=True)
class RaptorCohorts:
    training_indices: tuple[int, ...]
    training_targets: np.ndarray
    gold_indices: tuple[int, ...]
    gold_targets: np.ndarray


@dataclass(frozen=True, slots=True)
class RankedState:
    macro_auc: float
    epoch: int
    aucs: tuple[float, ...]
    state: StateDict


@dataclass(frozen=True, slots=True)
class RaptorReconstructionOutcome:
    best_checkpoint_path: Path
    averaged_checkpoint_path: Path
    report_path: Path
    manifest_path: Path


class RaptorWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        *,
        corpus: RaptorCorpus,
        indices: Sequence[int],
        targets: np.ndarray,
        window_count: int,
        resolution: int,
        training: bool,
        gain_minimum: float = 0.9,
        gain_maximum: float = 1.1,
    ) -> None:
        if len(indices) != len(targets):
            raise ValueError("Raptor dataset indices and targets differ in length")
        self.corpus = corpus
        self.indices = tuple(indices)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.window_count = window_count
        self.resolution = resolution
        self.training = training
        self.gain_minimum = gain_minimum
        self.gain_maximum = gain_maximum

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        volume, mask = self.corpus.sample(self.indices[index])
        centers = (
            training_centers(mask, self.window_count)
            if self.training
            else evaluation_centers(mask, self.window_count)
        )
        gain = (
            random.uniform(self.gain_minimum, self.gain_maximum)
            if self.training
            else None
        )
        windows = make_training_windows(
            volume,
            centers,
            resolution=self.resolution,
            gain=gain,
        )
        return windows, torch.from_numpy(self.targets[index])


def reconstruction_config(
    contract: Mapping[str, Any] | None = None,
) -> RaptorReconstructionConfig:
    if contract is None:
        contract, _ = load_raptor_training_reference_contract()
    configurations = _mapping(contract, "configurations")
    raw = _mapping(configurations, "reconstruction_configuration")
    augmentation = _mapping(raw, "augmentation")
    return RaptorReconstructionConfig(
        architecture=_text(raw, "architecture"),
        resolution=_integer(raw, "resolution"),
        epochs=_integer(raw, "epochs"),
        batch_size=_integer(raw, "batch_size"),
        validation_batch_size=_integer(raw, "validation_batch_size"),
        training_windows=_integer(raw, "training_windows"),
        evaluation_windows=_integer(raw, "evaluation_windows"),
        backbone_learning_rate=_number(raw, "backbone_learning_rate"),
        head_learning_rate=_number(raw, "head_learning_rate"),
        weight_decay=_number(raw, "weight_decay"),
        workers=_integer(raw, "workers"),
        topk=_integer(raw, "topk"),
        seed=_integer(raw, "seed"),
        scheduler_pct_start=_number(raw, "scheduler_pct_start"),
        gradient_clip_norm=_number(raw, "gradient_clip_norm"),
        gain_minimum=_number(augmentation, "multiplicative_gain_minimum"),
        gain_maximum=_number(augmentation, "multiplicative_gain_maximum"),
    )


def required_pinned_label_identity(
    contract: Mapping[str, Any],
) -> dict[str, object]:
    labels = _mapping(contract, "labels")
    selection = _mapping(labels, "reconstruction_selection")
    identity = _mapping(selection, "reconstruction_identity")
    if identity.get("status") != "pinned_after_approved_read_only_retrieval":
        raise PermissionError(
            "Raptor reconstruction requires the approved label identity pin"
        )
    return {
        "sha256": _text(identity, "sha256"),
        "size_bytes": _integer(identity, "size_bytes"),
        "columns": identity.get("columns"),
        "row_count": _integer(identity, "row_count"),
    }


def valid_centers(mask: np.ndarray) -> list[int]:
    values = np.flatnonzero(np.asarray(mask, dtype=bool))
    if len(values) >= 3:
        lower = max(1, int(values[0]) + 1)
        upper = min(len(mask) - 2, int(values[-1]) - 1)
        centers = list(range(lower, upper + 1)) if lower <= upper else []
    else:
        centers = []
    if len(centers) < 3:
        centers = list(range(1, len(mask) - 1))
    if not centers:
        raise ValueError("Raptor volume has no valid adjacent-slice center")
    return centers


def training_centers(mask: np.ndarray, count: int) -> tuple[int, ...]:
    centers = valid_centers(mask)
    random.shuffle(centers)
    if len(centers) < count:
        repeats = (count + len(centers) - 1) // len(centers)
        centers = (centers * repeats)[:count]
    return tuple(centers[:count])


def evaluation_centers(mask: np.ndarray, count: int) -> tuple[int, ...]:
    centers = valid_centers(mask)
    positions = np.rint(np.linspace(0, len(centers) - 1, count)).astype(np.int64)
    return tuple(centers[int(position)] for position in positions)


def make_training_windows(
    volume: np.ndarray,
    centers: Sequence[int],
    *,
    resolution: int,
    gain: float | None,
) -> Tensor:
    source = np.asarray(volume)
    if source.ndim != 3 or source.dtype != np.uint8:
        raise ValueError("Raptor training volume must be a uint8 slice stack")
    triplets = np.stack([source[center - 1 : center + 2] for center in centers])
    windows = torch.from_numpy(triplets.astype(np.float32) / 255.0)
    if windows.shape[-2:] != (resolution, resolution):
        windows = F.interpolate(
            windows,
            size=(resolution, resolution),
            mode="bilinear",
            align_corners=False,
        )
    if gain is not None:
        windows = (windows * gain).clamp(0, 1)
    return windows


def prepare_cohorts(
    *,
    corpus_ids: Sequence[str],
    soft_ids: Sequence[str],
    soft_targets: np.ndarray,
    gold_ids: Sequence[str],
    gold_targets: np.ndarray,
    expected_soft_count: int = 4349,
    expected_gold_count: int = 58,
) -> RaptorCohorts:
    corpus = tuple(corpus_ids)
    soft = tuple(soft_ids)
    gold = tuple(gold_ids)
    soft_values = np.asarray(soft_targets, dtype=np.float32)
    gold_values = np.asarray(gold_targets, dtype=np.float32)
    target_count = len(TARGET_NAMES)
    if len(corpus) != len(set(corpus)):
        raise ValueError("Raptor corpus contains duplicate study identifiers")
    if len(soft) != expected_soft_count or len(set(soft)) != len(soft):
        raise ValueError(
            "Raptor soft labels do not contain the expected unique studies"
        )
    if len(gold) != expected_gold_count or len(set(gold)) != len(gold):
        raise ValueError(
            "Raptor gold labels do not contain the expected unique studies"
        )
    if soft_values.shape != (len(soft), target_count):
        raise ValueError("Raptor soft-label target shape is invalid")
    if gold_values.shape != (len(gold), target_count):
        raise ValueError("Raptor gold-label target shape is invalid")
    if not np.isfinite(soft_values).all() or bool(
        ((soft_values < 0) | (soft_values > 1)).any()
    ):
        raise ValueError("Raptor soft labels must be finite probabilities")
    if not np.isfinite(gold_values).all() or bool(
        ((gold_values != 0) & (gold_values != 1)).any()
    ):
        raise ValueError("Raptor gold labels must be finite binary values")
    soft_set = set(soft)
    gold_set = set(gold)
    if soft_set & gold_set:
        raise ValueError(
            "Raptor gold studies must be excluded from soft-label training"
        )
    if soft_set | gold_set != set(corpus):
        raise ValueError("Raptor label cohorts do not exactly cover the corpus")
    index_by_id = {study_id: index for index, study_id in enumerate(corpus)}
    return RaptorCohorts(
        training_indices=tuple(index_by_id[study_id] for study_id in soft),
        training_targets=soft_values,
        gold_indices=tuple(index_by_id[study_id] for study_id in gold),
        gold_targets=gold_values,
    )


def prevalence_positive_weights(targets: np.ndarray) -> Tensor:
    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(TARGET_NAMES):
        raise ValueError("Raptor positive-weight targets have the wrong shape")
    prevalence = np.clip(values.mean(axis=0), 0.03, 0.7)
    weights = np.clip((1.0 - prevalence) / prevalence, 1.0, 10.0)
    return torch.tensor(weights, dtype=torch.float32)


def build_optimizer_and_scheduler(
    model: RaptorClassifier,
    config: RaptorReconstructionConfig,
    *,
    steps_per_epoch: int,
) -> tuple[Optimizer, OneCycleLR]:
    backbone = [
        parameter
        for parameter in model.backbone.parameters()
        if parameter.requires_grad
    ]
    backbone_ids = {id(parameter) for parameter in backbone}
    head = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    if not backbone or not head or backbone_ids & {id(parameter) for parameter in head}:
        raise ValueError("Raptor optimizer parameter groups are invalid")
    all_trainable = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if backbone_ids | {id(parameter) for parameter in head} != all_trainable:
        raise ValueError("Raptor optimizer groups do not cover trainable parameters")
    optimizer = AdamW(
        [
            {"params": backbone, "lr": config.backbone_learning_rate},
            {"params": head, "lr": config.head_learning_rate},
        ],
        weight_decay=config.weight_decay,
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[config.backbone_learning_rate, config.head_learning_rate],
        total_steps=steps_per_epoch * config.epochs,
        pct_start=config.scheduler_pct_start,
    )
    return optimizer, scheduler


def is_strict_improvement(score: float, best: RankedState | None) -> bool:
    return best is None or score > best.macro_auc


def retain_top_states(
    states: Sequence[RankedState],
    candidate: RankedState,
    *,
    topk: int,
) -> tuple[RankedState, ...]:
    ranked = [*states, candidate]
    ranked.sort(key=lambda item: item.macro_auc, reverse=True)
    return tuple(ranked[:topk])


def average_ranked_states(states: Sequence[RankedState]) -> StateDict:
    if not states:
        raise ValueError("Raptor state averaging requires at least one state")
    keys = tuple(states[0].state)
    if any(tuple(item.state) != keys for item in states[1:]):
        raise ValueError("Raptor top states have different parameter keys")
    averaged: StateDict = {}
    for key in keys:
        tensors = [item.state[key] for item in states]
        if torch.is_floating_point(tensors[0]):
            averaged[key] = torch.stack(tensors).mean(dim=0)
        else:
            averaged[key] = tensors[0].clone()
    return averaged


def checkpoint_payload(
    *,
    state: Mapping[str, Tensor],
    macro_auc: float,
    aucs: Sequence[float],
    epoch: int,
    source: str,
) -> dict[str, object]:
    return {
        "model": dict(state),
        "gold_auc": macro_auc,
        "aucs": list(aucs),
        "src": source,
        "res": MODEL_INPUT_SIZE,
        "arch": ARCHITECTURE,
        "lab": list(TARGET_NAMES),
        "epoch": epoch,
    }


def load_raptor_reconstruction_checkpoint(
    checkpoint_path: Path,
    *,
    expected_sha256: str,
    device: torch.device,
    backbone_factory: BackboneFactory | None = None,
) -> tuple[RaptorClassifier, RaptorCheckpointMetadata]:
    digest = _sha256(checkpoint_path)
    if digest != expected_sha256:
        raise ValueError("Raptor reconstruction checkpoint hash differs")
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or set(raw) != _CHECKPOINT_KEYS:
        raise ValueError("Raptor reconstruction checkpoint schema differs")
    if raw.get("arch") != ARCHITECTURE or raw.get("res") != MODEL_INPUT_SIZE:
        raise ValueError("Raptor reconstruction architecture identity differs")
    labels = raw.get("lab")
    state = raw.get("model")
    if not isinstance(labels, list) or tuple(labels) != TARGET_NAMES:
        raise ValueError("Raptor reconstruction target order differs")
    if not isinstance(state, dict):
        raise TypeError("Raptor reconstruction model state is malformed")
    backbone = (
        backbone_factory() if backbone_factory is not None else _new_backbone(False)
    )
    feature_count = getattr(backbone, "num_features", None)
    if not isinstance(feature_count, int):
        raise TypeError("Raptor reconstruction backbone has no feature count")
    model = RaptorClassifier(
        backbone, feature_count=feature_count, target_count=len(TARGET_NAMES)
    )
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model, RaptorCheckpointMetadata(
        sha256=digest,
        architecture=ARCHITECTURE,
        resolution=MODEL_INPUT_SIZE,
        targets=TARGET_NAMES,
        epoch=int(raw["epoch"]),
        source=str(raw["src"]),
        gold_auc=float(raw["gold_auc"]),
    )


def run_raptor_reconstruction(
    *,
    input_root: Path,
    work_root: Path,
    stage: ReconstructionStage,
    approved_gpu: bool,
    source_provenance: Mapping[str, object],
    notebook: Mapping[str, object],
) -> RaptorReconstructionOutcome:
    if stage not in {"smoke", "full"}:
        raise ValueError("Raptor reconstruction stage must be smoke or full")
    started = time.perf_counter()
    contract, contract_sha256 = load_raptor_training_reference_contract()
    pinned_label = required_pinned_label_identity(contract)
    config = reconstruction_config(contract)
    cuda = require_approved_cuda(approved=approved_gpu)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    _seed_source_generators(config.seed)
    corpus = _discover_corpus(input_root, contract)
    soft_ids, soft_targets, label_record = _read_soft_labels(
        _unique_input(input_root, "labels_llm_soft.parquet", "rsna-knee-labels")
    )
    if any(label_record[key] != pinned_label[key] for key in pinned_label):
        raise ValueError("Raptor label artifact differs from its pinned identity")
    gold_ids, gold_targets = _read_gold_labels(
        _unique_input(input_root, "train.csv", "rsna-knee-abnormality-detection")
    )
    cohorts = prepare_cohorts(
        corpus_ids=corpus.ids,
        soft_ids=soft_ids,
        soft_targets=soft_targets,
        gold_ids=gold_ids,
        gold_targets=gold_targets,
    )
    model, pretrained = _initialize_pretrained_model(device)
    train_dataset = RaptorWindowDataset(
        corpus=corpus,
        indices=cohorts.training_indices,
        targets=cohorts.training_targets,
        window_count=config.training_windows,
        resolution=config.resolution,
        training=True,
        gain_minimum=config.gain_minimum,
        gain_maximum=config.gain_maximum,
    )
    gold_dataset = RaptorWindowDataset(
        corpus=corpus,
        indices=cohorts.gold_indices,
        targets=cohorts.gold_targets,
        window_count=config.evaluation_windows,
        resolution=config.resolution,
        training=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
    )
    gold_loader = DataLoader(
        gold_dataset,
        batch_size=config.validation_batch_size,
        shuffle=False,
        num_workers=config.workers,
    )
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, config, steps_per_epoch=len(train_loader)
    )
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=prevalence_positive_weights(cohorts.training_targets).to(device)
    )
    epoch_records: list[dict[str, object]] = []
    top_states: tuple[RankedState, ...] = ()
    best: RankedState | None = None
    epochs_to_run = 1 if stage == "smoke" else config.epochs
    best_path = work_root / _BEST_CHECKPOINT_NAME
    swa_path = work_root / _SWA_CHECKPOINT_NAME
    work_root.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs_to_run + 1):
        loss, steps = _train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_function,
            device=device,
            gradient_clip_norm=config.gradient_clip_norm,
            maximum_steps=1 if stage == "smoke" else None,
        )
        metric, aucs = _evaluate_gold(model, gold_loader, cohorts.gold_targets, device)
        macro_auc = _required_macro_auc(metric)
        candidate = RankedState(
            macro_auc=macro_auc,
            epoch=epoch,
            aucs=aucs,
            state=_cpu_state(model),
        )
        top_states = retain_top_states(top_states, candidate, topk=config.topk)
        if is_strict_improvement(macro_auc, best):
            best = candidate
            save_checkpoint(
                best_path,
                checkpoint_payload(
                    state=best.state,
                    macro_auc=best.macro_auc,
                    aucs=best.aucs,
                    epoch=best.epoch,
                    source="timm_best_evidence_reconstruction",
                ),
            )
        epoch_records.append(
            {
                "epoch": epoch,
                "training_loss": loss,
                "optimizer_steps": steps,
                "gold_macro_auc": macro_auc,
                "gold_per_target_auc": list(aucs),
            }
        )
    if best is None:
        raise RuntimeError("Raptor reconstruction produced no evaluated epoch")
    averaged_state = average_ranked_states(top_states)
    model.load_state_dict(averaged_state, strict=True)
    averaged_metric, averaged_aucs = _evaluate_gold(
        model, gold_loader, cohorts.gold_targets, device
    )
    averaged_macro = _required_macro_auc(averaged_metric)
    save_checkpoint(
        swa_path,
        checkpoint_payload(
            state=averaged_state,
            macro_auc=averaged_macro,
            aucs=averaged_aucs,
            epoch=top_states[0].epoch,
            source="timm_best_evidence_reconstruction_top3_average",
        ),
    )
    model.load_state_dict(best.state, strict=True)
    runtime = _runtime_evidence(device, cuda)
    report_path = work_root / _REPORT_NAME
    report = {
        "schema_version": 1,
        "stage": stage,
        "evidence_semantic": "reconstruction_specific_pinned_identity",
        "training_reference_contract_sha256": contract_sha256,
        "configuration": config.record(),
        "cohorts": {
            "corpus_study_count": len(corpus.ids),
            "soft_training_study_count": len(cohorts.training_indices),
            "gold_validation_study_count": len(cohorts.gold_indices),
            "gold_excluded_from_training": True,
        },
        "label_source": label_record,
        "pretrained_weight": pretrained.record(),
        "epochs": epoch_records,
        "checkpoint_selection": {
            "strict_improvement": True,
            "best_epoch": best.epoch,
            "best_gold_macro_auc": best.macro_auc,
            "top_epochs": [item.epoch for item in top_states],
            "averaged_gold_macro_auc": averaged_macro,
        },
        "comparison_reference": {
            "released_best_epoch": 12,
            "released_gold_macro_auc": 0.9167492483170476,
            "role": "comparison_only_not_a_tuning_target",
        },
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "smoke_limits_training_batches_only": stage == "smoke",
            "hidden_test_inference": False,
            "competition_submission": False,
        },
        "runtime": runtime,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    attached_data = _mapping(contract, "attached_data")
    manifest_path = write_run_manifest(
        work_root / _MANIFEST_NAME,
        run_id=create_run_id("raptor-reconstruction"),
        source={
            **dict(source_provenance),
            "training_reference_contract_sha256": contract_sha256,
            "corpus": {
                "concatenation_order": attached_data["concatenation_order"],
                "datasets": attached_data["datasets"],
            },
            "label_source": label_record,
            "pretrained_weight": pretrained.record(),
        },
        configuration={**config.record(), "stage": stage},
        notebook=notebook,
        artifacts=[best_path, swa_path, report_path],
        artifact_root=work_root,
        runtime=runtime,
    )
    outcome = RaptorReconstructionOutcome(
        best_checkpoint_path=best_path,
        averaged_checkpoint_path=swa_path,
        report_path=report_path,
        manifest_path=manifest_path,
    )
    verify_raptor_reconstruction_artifact(work_root)
    return outcome


def verify_raptor_reconstruction_artifact(artifact_dir: Path) -> dict[str, object]:
    evidence = {name: artifact_dir / name for name in _EVIDENCE_NAMES}
    for path in evidence.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing Raptor reconstruction artifact: {path}")
    manifest = _json_object(evidence[_MANIFEST_NAME])
    report = _json_object(evidence[_REPORT_NAME])
    records = manifest.get("artifacts")
    if manifest.get("schema_version") != 1 or not isinstance(records, list):
        raise ValueError("Raptor reconstruction manifest is malformed")
    expected = set(_EVIDENCE_NAMES[:-1])
    actual: set[str] = set()
    for raw in records:
        if not isinstance(raw, dict):
            raise TypeError("Raptor reconstruction artifact record is malformed")
        relative = raw.get("path")
        if not isinstance(relative, str) or relative not in expected:
            raise ValueError("Raptor reconstruction artifact inventory differs")
        path = evidence[relative]
        if (
            raw.get("sha256") != _sha256(path)
            or raw.get("size_bytes") != path.stat().st_size
        ):
            raise ValueError(f"Raptor reconstruction artifact hash differs: {relative}")
        actual.add(relative)
    if actual != expected:
        raise ValueError("Raptor reconstruction artifact inventory is incomplete")
    if (
        report.get("schema_version") != 1
        or report.get("stage") not in {"smoke", "full"}
        or _mapping(report, "execution").get("hidden_test_inference") is not False
        or _mapping(report, "execution").get("competition_submission") is not False
    ):
        raise ValueError("Raptor reconstruction report crosses its evidence boundary")
    return {
        "stage": report["stage"],
        "artifact_count": len(actual),
        "best_checkpoint_sha256": _sha256(evidence[_BEST_CHECKPOINT_NAME]),
        "averaged_checkpoint_sha256": _sha256(evidence[_SWA_CHECKPOINT_NAME]),
    }


def _discover_corpus(input_root: Path, contract: Mapping[str, Any]) -> RaptorCorpus:
    attached = _mapping(contract, "attached_data")
    datasets = attached.get("datasets")
    if not isinstance(datasets, list):
        raise TypeError("Raptor training datasets are malformed")
    part_names = (
        ("all_ids.npy", "all_masks.npy", "all_vols.npy"),
        ("extra_ids.npy", "extra_masks.npy", "extra_vols.npy"),
    )
    parts: list[CorpusPart] = []
    combined_ids: list[str] = []
    for raw_dataset, names in zip(datasets, part_names, strict=True):
        if not isinstance(raw_dataset, dict):
            raise TypeError("Raptor training dataset record is malformed")
        expected_files: dict[str, int] = {}
        for record in cast(list[dict[str, object]], raw_dataset["files"]):
            name = record.get("name")
            size_bytes = record.get("size_bytes")
            if not isinstance(name, str) or type(size_bytes) is not int:
                raise TypeError("Raptor corpus file identity is malformed")
            expected_files[name] = size_bytes
        paths = tuple(
            _unique_input(input_root, name, str(raw_dataset["slug"]).split("/")[-1])
            for name in names
        )
        for path in paths:
            if path.stat().st_size != expected_files[path.name]:
                raise ValueError(f"Raptor corpus file size differs: {path.name}")
        ids_array = np.load(paths[0], allow_pickle=False)
        masks = np.load(paths[1], mmap_mode="r", allow_pickle=False)
        volumes = np.load(paths[2], mmap_mode="r", allow_pickle=False)
        ids = tuple(_study_id(value) for value in ids_array.tolist())
        if volumes.shape != (len(ids), 44, 336, 336) or volumes.dtype != np.uint8:
            raise ValueError("Raptor corpus volume shape or dtype differs")
        if masks.shape != (len(ids), 44):
            raise ValueError("Raptor corpus mask shape differs")
        parts.append(CorpusPart(ids=ids, masks=masks, volumes=volumes))
        combined_ids.extend(ids)
    if len(combined_ids) != 4407:
        raise ValueError("Raptor corpus study count differs")
    return RaptorCorpus(parts=tuple(parts), ids=tuple(combined_ids))


def _read_soft_labels(
    path: Path,
) -> tuple[tuple[str, ...], np.ndarray, dict[str, object]]:
    pandas = importlib.import_module("pandas")
    frame = pandas.read_parquet(path)
    expected_columns = ["StudyInstanceUID", *TARGET_NAMES]
    if list(frame.columns) != expected_columns:
        raise ValueError("Raptor soft-label columns or target order differ")
    ids = tuple(str(value) for value in frame["StudyInstanceUID"].tolist())
    targets = np.asarray(frame[list(TARGET_NAMES)].to_numpy(), dtype=np.float32)
    return (
        ids,
        targets,
        {
            "evidence_semantic": "reconstruction_specific_pinned_identity",
            "historical_attachment_proven": False,
            "dataset": "dreaddevelopment/rsna-knee-labels",
            "dataset_version": 3,
            "filename": path.name,
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "columns": expected_columns,
            "row_count": len(ids),
        },
    )


def _read_gold_labels(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    ids: list[str] = []
    targets: list[list[float]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(
            target not in reader.fieldnames for target in TARGET_NAMES
        ):
            raise ValueError("Competition train.csv target columns differ")
        for row in reader:
            values = [row.get(target, "") for target in TARGET_NAMES]
            if all(value not in {None, ""} for value in values):
                study_id = row.get("StudyInstanceUID")
                if not study_id:
                    raise ValueError("Gold row has no study identifier")
                ids.append(study_id)
                targets.append([float(cast(str, value)) for value in values])
    return tuple(ids), np.asarray(targets, dtype=np.float32)


def _initialize_pretrained_model(
    device: torch.device,
) -> tuple[RaptorClassifier, PretrainedWeightIdentity]:
    identity = _resolve_pretrained_weight()
    backbone = _new_backbone(True, checkpoint_path=identity.cache_path)
    setter = getattr(backbone, "set_grad_checkpointing", None)
    if not callable(setter):
        raise TypeError("Raptor backbone does not support gradient checkpointing")
    setter(True)
    feature_count = getattr(backbone, "num_features", None)
    if feature_count != 1024:
        raise ValueError("Raptor reconstruction backbone feature count differs")
    model = RaptorClassifier(
        backbone, feature_count=feature_count, target_count=len(TARGET_NAMES)
    )
    return model.to(device), identity


def _resolve_pretrained_weight() -> PretrainedWeightIdentity:
    repository = "timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
    filename = "model.safetensors"
    hub = importlib.import_module("huggingface_hub")
    info = hub.HfApi().model_info(repository)
    revision = getattr(info, "sha", None)
    if not isinstance(revision, str) or not revision:
        raise ValueError("Hugging Face did not resolve the Raptor weight revision")
    raw_path = hub.hf_hub_download(
        repo_id=repository, filename=filename, revision=revision
    )
    path = Path(raw_path).resolve()
    if not path.is_file() or revision not in path.parts:
        raise ValueError(
            "Raptor pretrained cache path does not pin the resolved revision"
        )
    return PretrainedWeightIdentity(
        repository=repository,
        requested_revision=None,
        resolved_revision=revision,
        filename=filename,
        sha256=_sha256(path),
        size_bytes=path.stat().st_size,
        cache_path=path,
    )


def _new_backbone(
    pretrained: bool, *, checkpoint_path: Path | None = None
) -> nn.Module:
    timm = importlib.import_module("timm")
    kwargs: dict[str, object] = {
        "pretrained": pretrained,
        "num_classes": 0,
        "in_chans": 3,
        "global_pool": "avg",
    }
    if checkpoint_path is not None:
        kwargs["pretrained_cfg_overlay"] = {"file": str(checkpoint_path)}
    return cast(nn.Module, timm.create_model(ARCHITECTURE, **kwargs))


def _train_epoch(
    model: RaptorClassifier,
    loader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: Optimizer,
    scheduler: OneCycleLR,
    loss_function: nn.Module,
    *,
    device: torch.device,
    gradient_clip_norm: float,
    maximum_steps: int | None,
) -> tuple[float, int]:
    model.train()
    losses: list[float] = []
    for step, (inputs, targets) in enumerate(loader, 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(inputs.to(device, non_blocking=True))
            loss = loss_function(logits, targets.to(device, non_blocking=True))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().cpu()))
        if maximum_steps is not None and step >= maximum_steps:
            break
    if not losses:
        raise RuntimeError("Raptor training loader produced no batches")
    return float(np.mean(losses)), len(losses)


@torch.no_grad()
def _evaluate_gold(
    model: RaptorClassifier,
    loader: DataLoader[tuple[Tensor, Tensor]],
    labels: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, Any], tuple[float, ...]]:
    model.eval()
    probabilities: list[np.ndarray] = []
    for inputs, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(inputs.to(device, non_blocking=True))
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
    predictions = np.concatenate(probabilities, axis=0)
    metric = evaluate_point_predictions(
        labels=labels.tolist(),
        predictions=predictions.tolist(),
        target_names=TARGET_NAMES,
    )
    per_target = _mapping(metric, "per_target")
    aucs: list[float] = []
    for target in TARGET_NAMES:
        target_metric = _mapping(per_target, target)
        auc = target_metric.get("roc_auc")
        if not isinstance(auc, (int, float)):
            raise TypeError(f"Raptor gold ROC AUC is undefined for {target}")
        aucs.append(float(auc))
    return metric, tuple(aucs)


def _required_macro_auc(metric: Mapping[str, Any]) -> float:
    value = metric.get("official_macro_auc")
    if not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError("Raptor gold macro ROC AUC is undefined")
    return float(value)


def _cpu_state(model: nn.Module) -> StateDict:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def _seed_source_generators(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _runtime_evidence(
    device: torch.device, cuda: Mapping[str, object]
) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(device)
    packages = {
        name: _distribution_version(name)
        for name in (
            "numpy",
            "pandas",
            "pyarrow",
            "huggingface-hub",
            "timm",
            "torch",
            "torchvision",
        )
    }
    cudnn_version = cast(Callable[[], int | None], torch.backends.cudnn.version)()
    return {
        "evidence_semantic": "reconstruction_specific_pinned_identity",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda": {
            **dict(cuda),
            "cudnn_version": cudnn_version,
            "device_capability": list(torch.cuda.get_device_capability(device)),
            "device_total_memory_bytes": properties.total_memory,
        },
    }


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _unique_input(root: Path, name: str, hint: str) -> Path:
    matches = [path for path in root.rglob(name) if hint in path.as_posix()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {hint}/{name} input, found {len(matches)}"
        )
    return matches[0]


def _study_id(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _mapping(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise TypeError(f"Raptor reconstruction {key} is malformed")
    return cast(dict[str, Any], item)


def _text(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        raise TypeError(f"Raptor reconstruction {key} must be text")
    return item


def _integer(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if type(item) is not int:
        raise TypeError(f"Raptor reconstruction {key} must be an integer")
    return item


def _number(value: Mapping[str, Any], key: str) -> float:
    item = value.get(key)
    if not isinstance(item, (int, float)):
        raise TypeError(f"Raptor reconstruction {key} must be numeric")
    return float(item)
