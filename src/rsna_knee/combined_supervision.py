"""Combined organizer-gold and frozen public-label supervision for baseline training."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shutil
import time
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch

from .competition_contract import (
    VerifiedGoldContract,
    load_verified_gold_contract,
    load_verified_submission_contract,
)
from .config import PreprocessingConfig
from .dinov2 import DINOV2_VITS14_FEATURE_COUNT
from .holdout import (
    PAIRED_BOOTSTRAP_METHOD_VERSION,
    PatientBootstrapConfig,
    evaluate_paired_auc_delta,
)
from .indexing import build_competition_index, verify_index_artifact
from .provenance import create_run_id, write_run_manifest
from .public_label_audit import resolve_dataset_root
from .submission import (
    SubmissionInputs,
    discover_submission_inputs,
    validate_submission,
    write_submission,
)
from .training import (
    ANATOMICAL_PLANES,
    IndexedStudy,
    Label,
    PartitionPrediction,
    PartitionStudy,
    StudyAggregation,
    StudyClassifier,
    _new_uninitialized_model,
    _restore_rng_state,
    build_dinov2_initialized_model,
    build_dinov2_optimizer,
    build_initialized_model,
    configure_dinov2_finetuning,
    decode_study,
    load_indexed_studies,
    predict_partition,
    require_approved_cuda,
    runtime_record,
    save_checkpoint,
    seed_everything,
    train_epoch,
)
from .weak_supervision import (
    PUBLIC_VERDICT_TO_LABEL as _PUBLIC_VERDICT_TO_LABEL,
)
from .weak_supervision import (
    classify_gold_membership,
    public_label_columns,
    verify_source_identity,
)

type SupervisionSource = Literal["gold", "public"]
type Partition = Literal["train", "val"]

PUBLIC_VERDICT_TO_LABEL: Mapping[str, Label] = _PUBLIC_VERDICT_TO_LABEL

_COMBINED_CHECKPOINT_SCHEMA_VERSION = 1
_PLANE_GATED_CHECKPOINT_SCHEMA_VERSION = 2
_DINOV2_CHECKPOINT_SCHEMA_VERSION = 4
_COMBINED_REPORT_SCHEMA_VERSION = 1
_COMBINED_COHORT_RULE_VERSION = "gold-plus-public-verdicts-v1"
_COMBINED_COHORT_DIGEST_VERSION = "combined-gold-public-cohort-jsonl-v1"
_COMBINED_SPLIT_RULE_VERSION = "patient-hash-disjoint-v1"
_ARTIFACT_DIRECTORY = "rsna-knee-combined-baseline-v1"
_PLANE_GATED_ARTIFACT_DIRECTORY = "rsna-knee-plane-gating-v1"
_DINOV2_ARTIFACT_DIRECTORY = "rsna-knee-dinov2-vits14-v1"
_SCORING_ARTIFACT_DIRECTORY = "rsna-knee-scoring-v1"
COMBINED_BASELINE_CHECKPOINT_SHA256 = (
    "1285f9890eb01e6a2383552e2ef1efcccdbc956ce14de049042bc268281240c4"
)
COMBINED_BASELINE_COMMIT = "e69f79af9c3de06e17a77e43bb41cb971879aed0"
_BASELINE_TRAINING_SECONDS = 13_102.42
_BASELINE_PEAK_CUDA_MEMORY_BYTES = 741_710_848
_DINOV2_29_MACRO_AUC = 0.634180

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


@dataclass(frozen=True, slots=True)
class CombinedStudy:
    study_uid: str
    patient_id: str
    labels: tuple[Label, ...]
    provenance: SupervisionSource
    raw_verdicts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PartitionedCombinedStudy:
    study_uid: str
    patient_id: str
    labels: tuple[Label, ...]
    provenance: SupervisionSource
    partition: Partition


@dataclass(frozen=True, slots=True)
class CombinedCohort:
    studies: tuple[CombinedStudy, ...]
    target_names: tuple[str, ...]
    gold_study_count: int
    non_gold_study_count: int
    digest: str
    per_target: dict[str, dict[str, Any]]
    gold_contract_sha256: str
    public_dataset_contract_sha256: str
    public_labels_sha256: str

    @property
    def total_study_count(self) -> int:
        return len(self.studies)


@dataclass(frozen=True, slots=True)
class CombinedPartition:
    studies: tuple[PartitionedCombinedStudy, ...]
    target_names: tuple[str, ...]
    split_seed: int
    val_fraction: float
    split_rule_version: str
    summary: dict[str, Any]

    @property
    def train_studies(self) -> tuple[PartitionedCombinedStudy, ...]:
        return tuple(study for study in self.studies if study.partition == "train")

    @property
    def val_studies(self) -> tuple[PartitionedCombinedStudy, ...]:
        return tuple(study for study in self.studies if study.partition == "val")

    @property
    def train_patients(self) -> frozenset[str]:
        return frozenset(study.patient_id for study in self.train_studies)

    @property
    def val_patients(self) -> frozenset[str]:
        return frozenset(study.patient_id for study in self.val_studies)


@dataclass(frozen=True, slots=True)
class CombinedBaselineConfig:
    preprocessing: PreprocessingConfig
    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float | None
    weight_decay: float
    seed: int
    mixed_precision: bool
    architecture: str
    weight_identity: str
    val_fraction: float
    split_seed: int
    reload_absolute_tolerance: float
    aggregation: StudyAggregation = "mean"
    paired_bootstrap_seed: int = 20260825
    paired_bootstrap_replicates: int = 10_000
    paired_bootstrap_confidence: float = 0.95
    minimum_macro_auc_delta: float = 0.005
    minimum_bootstrap_positive_fraction: float = 0.95
    large_target_regression: float = 0.02
    backbone_learning_rate: float | None = None
    head_learning_rate: float | None = None
    early_stopping_minimum_epochs: int | None = None
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float | None = None

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


@dataclass(slots=True)
class DinoV2EarlyStopping:
    """The fixed #30 validation-macro-AUC stopping policy."""

    minimum_epochs: int
    maximum_epochs: int
    patience: int
    min_delta: float
    completed_epochs: int = 0
    retained_best_macro_auc: float | None = None
    retained_best_epoch: int | None = None
    reference_best_macro_auc: float | None = None
    eligible_non_improving_epochs: int = 0
    stopping_epoch: int | None = None
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        if (
            self.minimum_epochs <= 0
            or self.maximum_epochs < self.minimum_epochs
            or self.patience <= 0
            or self.min_delta < 0
        ):
            raise ValueError("DINOv2 early-stopping configuration is invalid")

    def observe(self, epoch: int, macro_auc: float) -> bool:
        """Record one completed epoch and report whether training must stop."""
        if self.stopping_epoch is not None:
            raise RuntimeError("DINOv2 early stopping has already stopped")
        if epoch != self.completed_epochs + 1 or epoch > self.maximum_epochs:
            raise ValueError("DINOv2 epochs must be recorded consecutively")
        if not math.isfinite(macro_auc):
            raise ValueError("DINOv2 validation macro ROC-AUC must be finite")

        self.completed_epochs = epoch
        if (
            self.retained_best_macro_auc is None
            or macro_auc > self.retained_best_macro_auc
        ):
            self.retained_best_macro_auc = macro_auc
            self.retained_best_epoch = epoch

        if (
            self.reference_best_macro_auc is None
            or macro_auc >= self.reference_best_macro_auc + self.min_delta
        ):
            self.reference_best_macro_auc = macro_auc
            self.eligible_non_improving_epochs = 0
            qualified_improvement = True
        else:
            qualified_improvement = False
        if epoch < self.minimum_epochs:
            return False
        if not qualified_improvement:
            self.eligible_non_improving_epochs += 1

        if self.eligible_non_improving_epochs >= self.patience:
            self.stopping_epoch = epoch
            self.stop_reason = "patience_exhausted"
        elif epoch == self.maximum_epochs:
            self.stopping_epoch = epoch
            self.stop_reason = "maximum_epochs_reached"
        return self.stopping_epoch is not None

    def to_record(self) -> dict[str, object]:
        return {
            "minimum_epochs": self.minimum_epochs,
            "maximum_epochs": self.maximum_epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "monitor": "validation_study_level_macro_roc_auc",
            "mode": "maximize",
            "completed_epochs": self.completed_epochs,
            "retained_best_macro_auc": self.retained_best_macro_auc,
            "retained_best_epoch": self.retained_best_epoch,
            "reference_best_macro_auc": self.reference_best_macro_auc,
            "eligible_non_improving_epochs": self.eligible_non_improving_epochs,
            "stopping_epoch": self.stopping_epoch,
            "stop_reason": self.stop_reason,
        }

    @classmethod
    def from_record(cls, record: object) -> DinoV2EarlyStopping:
        if not isinstance(record, dict):
            raise TypeError("DINOv2 early-stopping state must be a mapping")
        expected = {
            "minimum_epochs",
            "maximum_epochs",
            "patience",
            "min_delta",
            "monitor",
            "mode",
            "completed_epochs",
            "retained_best_macro_auc",
            "retained_best_epoch",
            "reference_best_macro_auc",
            "eligible_non_improving_epochs",
            "stopping_epoch",
            "stop_reason",
        }
        if (
            set(record) != expected
            or record.get("monitor") != "validation_study_level_macro_roc_auc"
            or record.get("mode") != "maximize"
            or any(
                type(record.get(key)) is not int
                for key in (
                    "minimum_epochs",
                    "maximum_epochs",
                    "patience",
                    "completed_epochs",
                    "eligible_non_improving_epochs",
                )
            )
            or not isinstance(record.get("min_delta"), (int, float))
        ):
            raise ValueError("DINOv2 early-stopping state is malformed")
        state = cls(
            minimum_epochs=cast(int, record["minimum_epochs"]),
            maximum_epochs=cast(int, record["maximum_epochs"]),
            patience=cast(int, record["patience"]),
            min_delta=float(cast(float, record["min_delta"])),
            completed_epochs=cast(int, record["completed_epochs"]),
            eligible_non_improving_epochs=cast(
                int, record["eligible_non_improving_epochs"]
            ),
        )
        for attribute, key in (
            ("retained_best_macro_auc", "retained_best_macro_auc"),
            ("reference_best_macro_auc", "reference_best_macro_auc"),
        ):
            value = record[key]
            if value is not None and (
                not isinstance(value, (int, float)) or not math.isfinite(value)
            ):
                raise ValueError("DINOv2 early-stopping AUC is malformed")
            setattr(state, attribute, None if value is None else float(value))
        for attribute, key in (
            ("retained_best_epoch", "retained_best_epoch"),
            ("stopping_epoch", "stopping_epoch"),
        ):
            value = record[key]
            if value is not None and type(value) is not int:
                raise ValueError("DINOv2 early-stopping epoch is malformed")
            setattr(state, attribute, value)
        reason = record["stop_reason"]
        if reason is not None and reason not in {
            "patience_exhausted",
            "maximum_epochs_reached",
        }:
            raise ValueError("DINOv2 early-stopping reason is malformed")
        state.stop_reason = cast(str | None, reason)
        if (
            not 0 <= state.completed_epochs <= state.maximum_epochs
            or not 0 <= state.eligible_non_improving_epochs <= state.patience
            or (state.stopping_epoch is None) != (state.stop_reason is None)
            or state.stopping_epoch is not None
            and state.stopping_epoch != state.completed_epochs
        ):
            raise ValueError("DINOv2 early-stopping state is inconsistent")
        return state


def _clone_model_state_dict(model: StudyClassifier) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _dinov2_finetuning_metadata(
    model: StudyClassifier,
    optimizer: torch.optim.Optimizer,
    *,
    backbone_learning_rate: float,
    head_learning_rate: float,
    weight_decay: float,
) -> dict[str, object]:
    if len(optimizer.param_groups) != 2 or [
        group["lr"] for group in optimizer.param_groups
    ] != [backbone_learning_rate, head_learning_rate]:
        raise ValueError("DINOv2 optimizer differs from the #30 recipe")
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "freeze_policy": {
            "frozen": [
                "patch_embed",
                "cls_token",
                "pos_embed",
                "mask_token",
                "blocks_0_to_5",
            ],
            "trainable": ["blocks_6_to_11", "norm", "classification_head"],
        },
        "optimizer": {
            "parameter_group_count": 2,
            "backbone_learning_rate": backbone_learning_rate,
            "head_learning_rate": head_learning_rate,
            "weight_decay": weight_decay,
        },
        "parameter_counts": {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        },
    }


@dataclass(frozen=True, slots=True)
class CombinedTrainingInputs:
    competition_root: Path
    index_artifact_dir: Path
    weight_path: Path
    public_dataset_root: Path
    index_verification: Literal["accepted_frozen_artifact", "rebuilt_verified_artifact"]


@dataclass(frozen=True, slots=True)
class CombinedBaselineOutcome:
    artifact_dir: Path
    checkpoint_path: Path
    report_path: Path
    submission_path: Path | None
    validation_report_path: Path | None
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class CombinedScoringOutcome:
    artifact_dir: Path
    index_artifact_dir: Path
    submission_path: Path
    fallbacks_path: Path
    validation_report_path: Path
    scoring_report_path: Path
    manifest_path: Path


def load_combined_baseline_config(path: Path) -> CombinedBaselineConfig:
    """Load and validate the combined gold + public baseline training configuration."""
    with path.open("rb") as file:
        raw = tomllib.load(file)
    preprocessing_raw = _mapping(raw, "preprocessing")
    training_raw = _mapping(raw, "training")
    split_raw = _mapping(raw, "split")
    validation_raw = _mapping(raw, "validation")
    architecture = _text(training_raw, "architecture")
    supported_backbones = {
        "torchvision.models.resnet18": "ResNet18_Weights.IMAGENET1K_V1",
        "dinov2_vits14": "dinov2_vits14.LVD142M",
    }
    if architecture not in supported_backbones:
        raise ValueError("Configuration names an unsupported backbone")
    is_dinov2 = architecture == "dinov2_vits14"
    early_stopping_raw = raw.get("early_stopping", {})
    if not isinstance(early_stopping_raw, dict):
        raise TypeError("Configuration field early_stopping must be a table")
    model_raw = raw.get("model", {})
    if not isinstance(model_raw, dict):
        raise TypeError("Configuration field model must be a table")
    aggregation = model_raw.get("aggregation", "mean")
    if aggregation not in {"mean", "target_plane_gate"}:
        raise ValueError("model.aggregation must be mean or target_plane_gate")
    preprocessing = PreprocessingConfig(
        slice_count=_positive_int(preprocessing_raw, "slice_count"),
        lower_percentile=_number(preprocessing_raw, "lower_percentile"),
        upper_percentile=_number(preprocessing_raw, "upper_percentile"),
        image_size=_positive_int(preprocessing_raw, "image_size"),
    )
    config = CombinedBaselineConfig(
        preprocessing=preprocessing,
        epochs=_positive_int(training_raw, "epochs"),
        batch_size=_positive_int(training_raw, "batch_size"),
        gradient_accumulation_steps=_optional_positive_int(
            training_raw, "gradient_accumulation_steps", 1
        ),
        learning_rate=(
            None if is_dinov2 else _positive_number(training_raw, "learning_rate")
        ),
        weight_decay=_nonnegative_number(training_raw, "weight_decay"),
        seed=_nonnegative_int(training_raw, "seed"),
        mixed_precision=_boolean(training_raw, "mixed_precision"),
        architecture=architecture,
        weight_identity=_text(training_raw, "weight_identity"),
        val_fraction=_number(split_raw, "val_fraction"),
        split_seed=_nonnegative_int(split_raw, "seed"),
        reload_absolute_tolerance=_positive_number(
            validation_raw, "reload_absolute_tolerance"
        ),
        aggregation=cast(StudyAggregation, aggregation),
        paired_bootstrap_seed=_optional_nonnegative_int(
            validation_raw, "paired_bootstrap_seed", 20260825
        ),
        paired_bootstrap_replicates=_optional_positive_int(
            validation_raw, "paired_bootstrap_replicates", 10_000
        ),
        paired_bootstrap_confidence=_optional_number(
            validation_raw, "paired_bootstrap_confidence", 0.95
        ),
        minimum_macro_auc_delta=_optional_number(
            validation_raw, "minimum_macro_auc_delta", 0.005
        ),
        minimum_bootstrap_positive_fraction=_optional_number(
            validation_raw, "minimum_bootstrap_positive_fraction", 0.95
        ),
        large_target_regression=_optional_positive_number(
            validation_raw, "large_target_regression", 0.02
        ),
        backbone_learning_rate=(
            _positive_number(training_raw, "backbone_learning_rate")
            if is_dinov2
            else None
        ),
        head_learning_rate=(
            _positive_number(training_raw, "head_learning_rate") if is_dinov2 else None
        ),
        early_stopping_minimum_epochs=(
            _positive_int(early_stopping_raw, "minimum_epochs") if is_dinov2 else None
        ),
        early_stopping_patience=(
            _positive_int(early_stopping_raw, "patience") if is_dinov2 else None
        ),
        early_stopping_min_delta=(
            _positive_number(early_stopping_raw, "min_delta") if is_dinov2 else None
        ),
    )
    if not (
        0 <= preprocessing.lower_percentile < preprocessing.upper_percentile <= 100
    ):
        raise ValueError("Preprocessing percentiles are invalid")
    if preprocessing.image_size != 224:
        raise ValueError("Baseline requires 224 by 224 pixel inputs")
    if config.weight_identity != supported_backbones[config.architecture]:
        raise ValueError("Unexpected initialization weight identity")
    if config.architecture == "dinov2_vits14" and config.aggregation != "mean":
        raise ValueError("DINOv2 requires ordinary mean aggregation")
    if config.effective_batch_size != 16:
        raise ValueError("Combined training requires effective batch size 16")
    if not 0 < config.val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    if not 0 < config.paired_bootstrap_confidence < 1:
        raise ValueError("paired bootstrap confidence must be between zero and one")
    if not 0 < config.minimum_bootstrap_positive_fraction <= 1:
        raise ValueError("minimum bootstrap positive fraction must be in (0, 1]")
    if is_dinov2 and (
        "learning_rate" in training_raw
        or config.epochs != 20
        or config.batch_size != 4
        or config.gradient_accumulation_steps != 4
        or config.backbone_learning_rate != 8e-6
        or config.head_learning_rate != 1e-3
        or config.early_stopping_minimum_epochs != 10
        or config.early_stopping_patience != 4
        or config.early_stopping_min_delta != 0.002
    ):
        raise ValueError("DINOv2 configuration differs from the #30 recipe")
    return config


def discover_combined_training_inputs(
    input_root: Path,
    *,
    index_artifact_dir: Path | None = None,
    public_dataset_root: Path | None = None,
    weight_path: Path | None = None,
    architecture: str = "torchvision.models.resnet18",
) -> CombinedTrainingInputs:
    """Discover competition data, index artifact, ImageNet weight, and public labels."""
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

    if weight_path is None:
        contract_name = (
            "dinov2-vits14-lvd142m.json"
            if architecture == "dinov2_vits14"
            else "resnet18-imagenet1k-v1.json"
        )
        contract, _ = _load_json_contract(contract_name, None)
        artifact = _mapping(contract, "artifact")
        expected_name = _text(artifact, "filename")
        expected_size = _positive_int(artifact, "size_bytes")
        expected_hash = _text(artifact, "sha256")
        weight_candidates = [
            path
            for path in sorted(input_root.rglob(expected_name))
            if path.is_file()
            and path.stat().st_size == expected_size
            and _sha256(path) == expected_hash
        ]
        if len(weight_candidates) != 1:
            raise ValueError(
                "Expected exactly one offline initialization artifact with the approved hash"
            )
        resolved_weight_path = weight_candidates[0]
    else:
        resolved_weight_path = weight_path

    if public_dataset_root is None:
        public_candidates = (
            input_root / "datasets" / "pilkwang" / "rsna-knee-llm-labels",
            input_root / "pilkwang" / "rsna-knee-llm-labels",
            input_root / "rsna-knee-llm-labels",
        )
        resolved_public_root = resolve_dataset_root(
            public_candidates,
            "report_labels_v2.csv",
            search_roots=(input_root,),
        )
    else:
        resolved_public_root = public_dataset_root

    return CombinedTrainingInputs(
        competition_root=competition_root,
        index_artifact_dir=resolved_index_artifact_dir,
        weight_path=resolved_weight_path,
        public_dataset_root=resolved_public_root,
        index_verification=index_verification,
    )


def derive_combined_cohort(
    competition_root: Path,
    public_dataset_root: Path,
    index_artifact_dir: Path,
    *,
    competition_contract_path: Path | None = None,
    public_dataset_contract_path: Path | None = None,
    data_index_contract_path: Path | None = None,
    allow_verified_rebuilt_index: bool = False,
) -> CombinedCohort:
    """Derive unified 4,407-study supervision with gold precedence and public verdicts."""
    gold_contract = load_verified_gold_contract(competition_contract_path)
    public_contract, public_contract_sha256 = _load_json_contract(
        "public-report-label-dataset-v1.json", public_dataset_contract_path
    )
    if list(gold_contract.target_names) != public_contract.get("target_order"):
        raise ValueError("Public dataset target order differs from gold contract")

    if allow_verified_rebuilt_index:
        verify_index_artifact(index_artifact_dir)
    else:
        _verify_accepted_index(
            index_artifact_dir,
            gold_contract,
            data_index_contract_path=data_index_contract_path,
        )

    public_labels_path = public_dataset_root / "report_labels_v2.csv"
    if not public_labels_path.is_file():
        raise FileNotFoundError(f"Missing public label file: {public_labels_path}")
    expected_public_sha256 = _mapping(public_contract, "file_hashes").get(
        "report_labels_v2.csv"
    )
    if expected_public_sha256 is not None:
        public_labels_sha256 = verify_source_identity(
            public_labels_path,
            expected_public_sha256,
            description="Public label file",
        )
    else:
        public_labels_sha256 = _sha256(public_labels_path)

    index_rows = _read_csv(index_artifact_dir / "studies.csv", _STUDY_COLUMNS)
    index_by_uid: dict[str, dict[str, str]] = {}
    for row in index_rows:
        if row["split"] != "train":
            continue
        uid = row["StudyInstanceUID"].strip()
        if uid in index_by_uid:
            raise ValueError(f"Duplicate study in index: {uid}")
        index_by_uid[uid] = row

    train_rows = _read_csv(
        competition_root / gold_contract.train_filename, gold_contract.train_columns
    )
    train_row_counts = Counter(
        row[gold_contract.identifier_column].strip() for row in train_rows
    )
    for uid, count in train_row_counts.items():
        if count > 1:
            raise ValueError(f"Duplicate study in competition train.csv: {uid}")

    public_rows = _read_csv(
        public_labels_path, public_label_columns(gold_contract.target_names)
    )
    public_labels_by_uid: dict[str, tuple[dict[str, str], dict[str, str]]] = {}
    seen_public_uids: set[str] = set()
    for row in public_rows:
        uid = row["StudyInstanceUID"].strip()
        if not uid:
            raise ValueError("Public label row has blank StudyInstanceUID")
        if uid in seen_public_uids:
            raise ValueError(f"Duplicate study in public label dataset: {uid}")
        seen_public_uids.add(uid)
        verdicts = {
            target: row[f"{target}__verdict"].strip()
            for target in gold_contract.target_names
        }
        scores = {target: row[target].strip() for target in gold_contract.target_names}
        public_labels_by_uid[uid] = (verdicts, scores)

    studies: list[CombinedStudy] = []
    gold_count = 0
    non_gold_count = 0

    for row in train_rows:
        study_uid = row[gold_contract.identifier_column].strip()
        if not study_uid:
            raise ValueError("train.csv contains a blank StudyInstanceUID")
        index_row = index_by_uid.get(study_uid)
        if index_row is None or not _index_row_is_usable(index_row):
            continue
        patient_id = index_row["PatientID"].strip()
        if not patient_id:
            raise ValueError(f"Missing PatientID for study: {study_uid}")

        raw_gold_values = [row[target].strip() for target in gold_contract.target_names]
        is_gold_study = classify_gold_membership(raw_gold_values) == "gold"

        if is_gold_study:
            gold_labels = tuple(_parse_label(val) for val in raw_gold_values)
            studies.append(
                CombinedStudy(
                    study_uid=study_uid,
                    patient_id=patient_id,
                    labels=gold_labels,
                    provenance="gold",
                    raw_verdicts=tuple(raw_gold_values),
                )
            )
            gold_count += 1
        else:
            if study_uid not in public_labels_by_uid:
                raise ValueError(
                    f"Non-gold study {study_uid} is missing from public label dataset"
                )
            public_verdicts, _ = public_labels_by_uid[study_uid]
            public_labels: list[Label] = []
            raw_verdict_list: list[str] = []
            for target in gold_contract.target_names:
                verdict = public_verdicts[target]
                if verdict not in PUBLIC_VERDICT_TO_LABEL:
                    raise ValueError(
                        f"Invalid public verdict {verdict!r} for study {study_uid} target {target}"
                    )
                public_labels.append(PUBLIC_VERDICT_TO_LABEL[verdict])
                raw_verdict_list.append(verdict)
            studies.append(
                CombinedStudy(
                    study_uid=study_uid,
                    patient_id=patient_id,
                    labels=tuple(public_labels),
                    provenance="public",
                    raw_verdicts=tuple(raw_verdict_list),
                )
            )
            non_gold_count += 1

    studies.sort(key=lambda s: s.study_uid.encode("utf-8"))

    digest = _combined_cohort_digest(
        studies,
        gold_contract.target_names,
        gold_count=gold_count,
        non_gold_count=non_gold_count,
    )
    per_target = _compute_combined_target_aggregates(
        studies, gold_contract.target_names
    )

    return CombinedCohort(
        studies=tuple(studies),
        target_names=gold_contract.target_names,
        gold_study_count=gold_count,
        non_gold_study_count=non_gold_count,
        digest=digest,
        per_target=per_target,
        gold_contract_sha256=gold_contract.sha256,
        public_dataset_contract_sha256=public_contract_sha256,
        public_labels_sha256=public_labels_sha256,
    )


def derive_patient_split(
    cohort: CombinedCohort,
    *,
    split_seed: int = 20260825,
    val_fraction: float = 0.10,
) -> CombinedPartition:
    """Deterministically partition studies into PatientID-disjoint train and validation."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be strictly between 0 and 1")

    patient_ids = sorted({study.patient_id for study in cohort.studies})
    patient_partition: dict[str, Partition] = {}
    for patient_id in patient_ids:
        token = f"{split_seed}:{patient_id}".encode()
        score = int(hashlib.sha256(token).hexdigest()[:8], 16) / 0xFFFFFFFF
        patient_partition[patient_id] = "val" if score < val_fraction else "train"

    partitioned_studies: list[PartitionedCombinedStudy] = []
    for study in cohort.studies:
        partition = patient_partition[study.patient_id]
        partitioned_studies.append(
            PartitionedCombinedStudy(
                study_uid=study.study_uid,
                patient_id=study.patient_id,
                labels=study.labels,
                provenance=study.provenance,
                partition=partition,
            )
        )

    train_studies = [s for s in partitioned_studies if s.partition == "train"]
    val_studies = [s for s in partitioned_studies if s.partition == "val"]
    if not train_studies:
        raise RuntimeError("Train partition has zero studies")
    if not val_studies:
        raise RuntimeError("Validation partition has zero studies")

    train_patients = frozenset(s.patient_id for s in train_studies)
    val_patients = frozenset(s.patient_id for s in val_studies)
    if not train_patients.isdisjoint(val_patients):
        raise RuntimeError(
            "PatientID leakage detected across train and validation splits"
        )

    summary = {
        "split_seed": split_seed,
        "val_fraction_requested": val_fraction,
        "total_studies": len(cohort.studies),
        "total_patients": len(patient_ids),
        "train": {
            "study_count": len(train_studies),
            "patient_count": len(train_patients),
            "gold_study_count": sum(1 for s in train_studies if s.provenance == "gold"),
            "public_study_count": sum(
                1 for s in train_studies if s.provenance == "public"
            ),
            "per_target": _compute_combined_target_aggregates(
                train_studies, cohort.target_names
            ),
        },
        "val": {
            "study_count": len(val_studies),
            "patient_count": len(val_patients),
            "gold_study_count": sum(1 for s in val_studies if s.provenance == "gold"),
            "public_study_count": sum(
                1 for s in val_studies if s.provenance == "public"
            ),
            "per_target": _compute_combined_target_aggregates(
                val_studies, cohort.target_names
            ),
        },
    }

    return CombinedPartition(
        studies=tuple(partitioned_studies),
        target_names=cohort.target_names,
        split_seed=split_seed,
        val_fraction=val_fraction,
        split_rule_version=_COMBINED_SPLIT_RULE_VERSION,
        summary=summary,
    )


def discover_combined_checkpoint(
    input_root: Path,
    *,
    expected_sha256: str = COMBINED_BASELINE_CHECKPOINT_SHA256,
) -> Path:
    """Find exactly one attached checkpoint with the required digest."""
    candidates = sorted(
        path for path in input_root.rglob("checkpoint.pt") if path.is_file()
    )
    matching = [path for path in candidates if _sha256(path) == expected_sha256]
    if len(matching) == 1:
        return matching[0]
    if not candidates:
        raise FileNotFoundError(
            "Missing attached checkpoint.pt for combined baseline scoring"
        )
    if not matching:
        raise ValueError(
            f"Attached checkpoint SHA-256 mismatch; expected {expected_sha256}"
        )
    raise ValueError("Expected exactly one attached checkpoint.pt")


def load_verified_combined_checkpoint(
    path: Path,
    *,
    device: torch.device,
    contract_path: Path | None = None,
    expected_sha256: str = COMBINED_BASELINE_CHECKPOINT_SHA256,
) -> tuple[StudyClassifier, dict[str, Any]]:
    """Verify the frozen checkpoint before deserializing it for scoring."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "Checkpoint SHA-256 mismatch; expected "
            f"{expected_sha256}, got {actual_sha256}"
        )

    model, checkpoint = load_combined_checkpoint(path, device=device)
    target_names = load_verified_submission_contract(contract_path).target_names
    if checkpoint.get("targets") != list(target_names):
        raise ValueError("Checkpoint target order differs from competition contract")

    expected_preprocessing = {
        "slice_count": 8,
        "lower_percentile": 1.0,
        "upper_percentile": 99.0,
        "image_size": 224,
        "shared_contract": "preprocess_series_result",
        "channel_adapter": "grayscale_repeat_to_rgb",
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
    }
    if checkpoint.get("preprocessing") != expected_preprocessing:
        raise ValueError("Checkpoint preprocessing metadata is incompatible")
    if checkpoint.get("completed_epochs") != 3:
        raise ValueError("Checkpoint does not contain the completed 3-epoch run")
    if checkpoint.get("cohort") != {
        "selection_rule_version": _COMBINED_COHORT_RULE_VERSION,
        "digest": "b066a7e6cd7119f4bef330a006a86a0e08e664d7b5347a111ec781e72abe1fdc",
        "gold_study_count": 58,
        "non_gold_study_count": 4349,
        "total_study_count": 4407,
    }:
        raise ValueError("Checkpoint supervision metadata is incompatible")
    if checkpoint.get("partition") != {
        "split_rule_version": _COMBINED_SPLIT_RULE_VERSION,
        "split_seed": 20260825,
        "val_fraction": 0.1,
        "train_study_count": 3957,
        "val_study_count": 450,
    }:
        raise ValueError("Checkpoint partition metadata is incompatible")
    if checkpoint.get("configuration") != _expected_baseline_configuration():
        raise ValueError("Checkpoint configuration metadata is incompatible")
    priors = checkpoint.get("priors")
    if (
        not isinstance(priors, list)
        or len(priors) != len(target_names)
        or any(not _is_probability(value) for value in priors)
    ):
        raise ValueError("Checkpoint fallback priors are incompatible")
    initialization = checkpoint.get("initialization")
    if (
        not isinstance(initialization, dict)
        or initialization.get("artifact_sha256")
        != "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
        or initialization.get("artifact_size_bytes") != 46830571
        or initialization.get("contract_sha256")
        != "cbd3639197247bf8b4bd83f11047f5cac5bf8b13ba92e881c423f7c61fabc4ca"
    ):
        raise ValueError("Checkpoint initialization metadata is incompatible")
    return model, checkpoint


def load_verified_plane_gated_checkpoint(
    path: Path,
    *,
    device: torch.device,
    expected_sha256: str,
    contract_path: Path | None = None,
) -> tuple[StudyClassifier, dict[str, Any]]:
    """Verify a retained #28 checkpoint before hidden-test inference."""
    if not path.is_file() or _sha256(path) != expected_sha256:
        raise ValueError("Plane-gated checkpoint SHA-256 mismatch")
    model, checkpoint = load_combined_checkpoint(path, device=device)
    targets = load_verified_submission_contract(contract_path).target_names
    if checkpoint.get("targets") != list(targets):
        raise ValueError("Plane-gated checkpoint target order differs from competition")
    if checkpoint.get("aggregation") != _plane_gate_metadata(targets):
        raise ValueError("Plane-gated checkpoint aggregation metadata is incompatible")
    configuration = checkpoint.get("configuration")
    if not isinstance(configuration, dict):
        raise TypeError("Plane-gated checkpoint configuration is malformed")
    baseline_view = dict(configuration)
    baseline_view.pop("model", None)
    baseline_view["validation"] = {
        "reload_absolute_tolerance": configuration.get("validation", {}).get(
            "reload_absolute_tolerance"
        )
        if isinstance(configuration.get("validation"), dict)
        else None
    }
    if baseline_view != _expected_baseline_configuration():
        raise ValueError("Plane-gated checkpoint changed the #27 control configuration")
    validation = configuration.get("validation")
    if not isinstance(validation, dict) or configuration.get("model") != {
        "aggregation": "target_plane_gate"
    }:
        raise ValueError("Plane-gated checkpoint experiment metadata is missing")
    if validation != {
        "reload_absolute_tolerance": 1e-5,
        "paired_bootstrap_method": PAIRED_BOOTSTRAP_METHOD_VERSION,
        "paired_bootstrap_seed": 20260825,
        "paired_bootstrap_replicates": 10_000,
        "paired_bootstrap_confidence": 0.95,
        "minimum_macro_auc_delta": 0.005,
        "minimum_bootstrap_positive_fraction": 0.95,
        "large_target_regression": 0.02,
    }:
        raise ValueError(
            "Plane-gated checkpoint decision configuration is incompatible"
        )
    if checkpoint.get("preprocessing") != {
        "slice_count": 8,
        "lower_percentile": 1.0,
        "upper_percentile": 99.0,
        "image_size": 224,
        "shared_contract": "preprocess_series_result",
        "channel_adapter": "grayscale_repeat_to_rgb",
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
    }:
        raise ValueError(
            "Plane-gated checkpoint preprocessing metadata is incompatible"
        )
    if checkpoint.get("completed_epochs") != 3:
        raise ValueError("Plane-gated checkpoint epoch count is incompatible")
    if checkpoint.get("cohort") != {
        "selection_rule_version": _COMBINED_COHORT_RULE_VERSION,
        "digest": "b066a7e6cd7119f4bef330a006a86a0e08e664d7b5347a111ec781e72abe1fdc",
        "gold_study_count": 58,
        "non_gold_study_count": 4349,
        "total_study_count": 4407,
    } or checkpoint.get("partition") != {
        "split_rule_version": _COMBINED_SPLIT_RULE_VERSION,
        "split_seed": 20260825,
        "val_fraction": 0.1,
        "train_study_count": 3957,
        "val_study_count": 450,
    }:
        raise ValueError("Plane-gated checkpoint cohort/partition is incompatible")
    priors = checkpoint.get("priors")
    if (
        not isinstance(priors, list)
        or len(priors) != len(targets)
        or any(not _is_probability(value) for value in priors)
    ):
        raise ValueError("Plane-gated checkpoint fallback priors are incompatible")
    initialization = checkpoint.get("initialization")
    if (
        not isinstance(initialization, dict)
        or initialization.get("artifact_sha256")
        != "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
        or initialization.get("artifact_size_bytes") != 46830571
        or initialization.get("contract_sha256")
        != "cbd3639197247bf8b4bd83f11047f5cac5bf8b13ba92e881c423f7c61fabc4ca"
    ):
        raise ValueError("Plane-gated checkpoint initialization is incompatible")
    return model, checkpoint


def load_verified_dinov2_checkpoint(
    path: Path,
    *,
    device: torch.device,
    expected_sha256: str,
    contract_path: Path | None = None,
) -> tuple[StudyClassifier, dict[str, Any]]:
    """Verify one retained DINOv2 checkpoint before hidden-test inference."""
    if not path.is_file() or _sha256(path) != expected_sha256:
        raise ValueError("DINOv2 checkpoint SHA-256 mismatch")
    model, checkpoint = load_combined_checkpoint(path, device=device)
    targets = load_verified_submission_contract(contract_path).target_names
    if checkpoint.get("targets") != list(targets):
        raise ValueError("DINOv2 checkpoint target order differs from competition")
    if checkpoint.get("checkpoint_schema_version") != _DINOV2_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("DINOv2 checkpoint schema is incompatible")
    if checkpoint.get("preprocessing") != {
        "slice_count": 8,
        "lower_percentile": 1.0,
        "upper_percentile": 99.0,
        "image_size": 224,
        "shared_contract": "preprocess_series_result",
        "channel_adapter": "grayscale_repeat_to_rgb",
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
    }:
        raise ValueError("DINOv2 checkpoint preprocessing is incompatible")
    completed_epochs = checkpoint.get("completed_epochs")
    if type(completed_epochs) is not int or not 10 <= completed_epochs <= 20:
        raise ValueError("DINOv2 checkpoint epoch count is incompatible")
    if checkpoint.get("cohort") != {
        "selection_rule_version": _COMBINED_COHORT_RULE_VERSION,
        "digest": "b066a7e6cd7119f4bef330a006a86a0e08e664d7b5347a111ec781e72abe1fdc",
        "gold_study_count": 58,
        "non_gold_study_count": 4349,
        "total_study_count": 4407,
    } or checkpoint.get("partition") != {
        "split_rule_version": _COMBINED_SPLIT_RULE_VERSION,
        "split_seed": 20260825,
        "val_fraction": 0.1,
        "train_study_count": 3957,
        "val_study_count": 450,
    }:
        raise ValueError("DINOv2 checkpoint cohort/partition is incompatible")
    if not _is_expected_dinov2_configuration(checkpoint.get("configuration")):
        raise ValueError("DINOv2 checkpoint configuration is incompatible")
    early_stopping = DinoV2EarlyStopping.from_record(checkpoint.get("early_stopping"))
    if (
        early_stopping.completed_epochs != completed_epochs
        or early_stopping.stopping_epoch != completed_epochs
        or early_stopping.stop_reason is None
        or early_stopping.retained_best_epoch is None
        or early_stopping.retained_best_macro_auc is None
        or len(cast(list[object], checkpoint.get("epoch_logs"))) != completed_epochs
    ):
        raise ValueError("DINOv2 checkpoint early-stopping state is incompatible")
    priors = checkpoint.get("priors")
    if (
        not isinstance(priors, list)
        or len(priors) != len(targets)
        or any(not _is_probability(value) for value in priors)
    ):
        raise ValueError("DINOv2 checkpoint fallback priors are incompatible")
    initialization_contract, initialization_contract_sha256 = _load_json_contract(
        "dinov2-vits14-lvd142m.json", None
    )
    artifact = _mapping(initialization_contract, "artifact")
    initialization = checkpoint.get("initialization")
    backbone = checkpoint.get("backbone")
    if (
        not isinstance(initialization, dict)
        or initialization.get("artifact_sha256") != artifact.get("sha256")
        or initialization.get("artifact_size_bytes") != artifact.get("size_bytes")
        or initialization.get("contract_sha256") != initialization_contract_sha256
        or not isinstance(backbone, dict)
        or backbone.get("initialization_artifact_sha256") != artifact.get("sha256")
    ):
        raise ValueError("DINOv2 checkpoint initialization is incompatible")
    configure_dinov2_finetuning(model)
    optimizer = build_dinov2_optimizer(
        model,
        backbone_learning_rate=8e-6,
        head_learning_rate=1e-3,
        weight_decay=1e-4,
    )
    if checkpoint.get("finetuning") != _dinov2_finetuning_metadata(
        model,
        optimizer,
        backbone_learning_rate=8e-6,
        head_learning_rate=1e-3,
        weight_decay=1e-4,
    ):
        raise ValueError("DINOv2 checkpoint fine-tuning metadata is incompatible")
    retained_state = checkpoint.get("retained_model_state_dict")
    if (
        not isinstance(retained_state, dict)
        or set(retained_state) != set(checkpoint["model_state_dict"])
        or any(
            not isinstance(value, torch.Tensor)
            or not torch.equal(value, checkpoint["model_state_dict"][name])
            for name, value in retained_state.items()
        )
    ):
        raise ValueError("DINOv2 checkpoint does not retain its best model state")
    return model, checkpoint


def _expected_baseline_configuration() -> dict[str, object]:
    return {
        "preprocessing": {
            "slice_count": 8,
            "lower_percentile": 1.0,
            "upper_percentile": 99.0,
            "image_size": 224,
        },
        "training": {
            "epochs": 3,
            "batch_size": 16,
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "seed": 20260825,
            "mixed_precision": True,
            "architecture": "torchvision.models.resnet18",
            "weight_identity": "ResNet18_Weights.IMAGENET1K_V1",
        },
        "split": {"val_fraction": 0.1, "split_seed": 20260825},
        "validation": {"reload_absolute_tolerance": 1e-5},
    }


def _expected_dinov2_configuration() -> dict[str, object]:
    return {
        "preprocessing": {
            "slice_count": 8,
            "lower_percentile": 1.0,
            "upper_percentile": 99.0,
            "image_size": 224,
        },
        "training": {
            "epochs": 20,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 16,
            "backbone_learning_rate": 8e-6,
            "head_learning_rate": 1e-3,
            "weight_decay": 0.0001,
            "seed": 20260825,
            "mixed_precision": True,
            "architecture": "dinov2_vits14",
            "weight_identity": "dinov2_vits14.LVD142M",
        },
        "split": {"val_fraction": 0.1, "split_seed": 20260825},
        "validation": {
            "reload_absolute_tolerance": 1e-5,
            "paired_bootstrap_method": PAIRED_BOOTSTRAP_METHOD_VERSION,
            "paired_bootstrap_seed": 20260825,
            "paired_bootstrap_replicates": 10_000,
            "paired_bootstrap_confidence": 0.95,
            "minimum_macro_auc_delta": 0.005,
            "minimum_bootstrap_positive_fraction": 0.95,
            "large_target_regression": 0.02,
        },
        "early_stopping": {
            "minimum_epochs": 10,
            "maximum_epochs": 20,
            "patience": 4,
            "min_delta": 0.002,
            "monitor": "validation_study_level_macro_roc_auc",
            "mode": "maximize",
        },
    }


def _is_expected_dinov2_configuration(value: object) -> bool:
    return value == _expected_dinov2_configuration()


def run_combined_scoring(
    *,
    input_root: Path,
    work_root: Path,
    approved_gpu: bool = False,
    checkpoint_path: Path | None = None,
    run_id: str | None = None,
    source_provenance: Mapping[str, object] | None = None,
    notebook: Mapping[str, object] | None = None,
    competition_contract_path: Path | None = None,
    aggregation: StudyAggregation = "mean",
    expected_checkpoint_sha256: str = COMBINED_BASELINE_CHECKPOINT_SHA256,
    backbone: Literal["resnet18", "dinov2_vits14"] = "resnet18",
) -> CombinedScoringOutcome:
    """Run inference-only scoring against the live competition test mount."""
    checkpoint = checkpoint_path or (
        discover_combined_checkpoint(input_root)
        if expected_checkpoint_sha256 == COMBINED_BASELINE_CHECKPOINT_SHA256
        else discover_combined_checkpoint(
            input_root, expected_sha256=expected_checkpoint_sha256
        )
    )
    if backbone == "dinov2_vits14":
        if aggregation != "mean":
            raise ValueError("DINOv2 scoring requires ordinary mean aggregation")
        model, checkpoint_payload = load_verified_dinov2_checkpoint(
            checkpoint,
            device=torch.device("cpu"),
            expected_sha256=expected_checkpoint_sha256,
            contract_path=competition_contract_path,
        )
    elif backbone != "resnet18":
        raise ValueError("Unsupported scoring backbone")
    elif aggregation == "target_plane_gate":
        model, checkpoint_payload = load_verified_plane_gated_checkpoint(
            checkpoint,
            device=torch.device("cpu"),
            expected_sha256=expected_checkpoint_sha256,
            contract_path=competition_contract_path,
        )
    elif (
        aggregation == "mean"
        and expected_checkpoint_sha256 == COMBINED_BASELINE_CHECKPOINT_SHA256
    ):
        model, checkpoint_payload = load_verified_combined_checkpoint(
            checkpoint,
            device=torch.device("cpu"),
            contract_path=competition_contract_path,
        )
    else:
        raise ValueError("Unsupported scoring checkpoint identity")
    cuda_evidence = require_approved_cuda(approved=approved_gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    submission_inputs = discover_submission_inputs(
        input_root, contract_path=competition_contract_path
    )
    if checkpoint_payload["targets"] != list(submission_inputs.contract.target_names):
        raise ValueError("Checkpoint target order differs from submission contract")
    preprocessing_raw = cast(dict[str, Any], checkpoint_payload["preprocessing"])
    preprocessing = PreprocessingConfig(
        slice_count=int(preprocessing_raw["slice_count"]),
        lower_percentile=float(preprocessing_raw["lower_percentile"]),
        upper_percentile=float(preprocessing_raw["upper_percentile"]),
        image_size=int(preprocessing_raw["image_size"]),
    )

    index_outcome = build_competition_index(
        input_root,
        work_root,
        preprocessing,
        splits=("test",),
        representative_series_per_split=4,
        contract_path=competition_contract_path,
    )
    verify_index_artifact(
        index_outcome.artifact_dir, contract_path=competition_contract_path
    )
    live_index = load_indexed_studies(
        submission_inputs.competition_root,
        index_outcome.artifact_dir,
        submission_inputs.entity_ids,
        split="test",
    )
    missing = [
        study_uid
        for study_uid in submission_inputs.entity_ids
        if not live_index[study_uid].series
    ]
    if missing:
        raise RuntimeError(
            "Live test index has no usable selected series for: " + ", ".join(missing)
        )

    artifact_dir = work_root / _SCORING_ARTIFACT_DIRECTORY
    artifact_dir.mkdir(parents=True, exist_ok=True)
    priors = cast(Sequence[float], checkpoint_payload["priors"])
    prediction = _predict_combined_submission(
        model=model,
        submission_inputs=submission_inputs,
        indexed=live_index,
        work_root=artifact_dir,
        preprocessing=preprocessing,
        priors=priors,
        mixed_precision=bool(
            cast(dict[str, Any], checkpoint_payload["configuration"])["training"][
                "mixed_precision"
            ]
        ),
        device=device,
        max_fallback_fraction=0.5,
    )
    validation = validate_submission(
        prediction.artifacts.submission_path, inputs=submission_inputs
    )
    if validation.get("valid") is not True:
        raise RuntimeError("Strict submission validation did not pass")

    scoring_report_path = artifact_dir / "scoring-report.json"
    scoring_report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "inference_only",
                "cuda_readiness": cuda_evidence,
                "checkpoint": {
                    "path": checkpoint.name,
                    "sha256": _sha256(checkpoint),
                    "targets": list(submission_inputs.contract.target_names),
                    "aggregation": aggregation,
                    "backbone": backbone,
                },
                "live_test": {
                    "study_count": len(submission_inputs.entity_ids),
                    "indexed_study_count": len(live_index),
                    "index_manifest_sha256": _sha256(
                        index_outcome.artifact_dir / "index-manifest.json"
                    ),
                    "test_csv_sha256": _sha256(
                        submission_inputs.competition_root / "test.csv"
                    ),
                    "test_series_csv_sha256": _sha256(
                        submission_inputs.competition_root / "test_series.csv"
                    ),
                    "plane_availability": plane_availability(
                        live_index, submission_inputs.entity_ids
                    ),
                },
                "inference": {
                    "prediction_count": prediction.prediction_count,
                    "fallback_count": prediction.fallback_count,
                    "decode_errors": prediction.decode_errors,
                    "inference_seconds": prediction.inference_seconds,
                },
                "submission": {
                    "path": prediction.artifacts.submission_path.name,
                    "sha256": _sha256(prediction.artifacts.submission_path),
                    "strictly_validated": True,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = artifact_dir / "run-manifest.json"
    write_run_manifest(
        manifest_path,
        run_id=run_id or create_run_id("combined-scoring"),
        source={
            **dict(source_provenance or {}),
            "checkpoint_sha256": _sha256(checkpoint),
            "live_test_index_manifest_sha256": _sha256(
                index_outcome.artifact_dir / "index-manifest.json"
            ),
        },
        configuration={
            "mode": "inference_only",
            "backbone": backbone,
            "target_order": list(submission_inputs.contract.target_names),
            "preprocessing": asdict(preprocessing),
        },
        notebook=notebook or {"kind": "combined_inference_only_scoring"},
        artifacts=[
            prediction.artifacts.submission_path,
            prediction.artifacts.fallbacks_path,
            prediction.artifacts.validation_report_path,
            scoring_report_path,
        ],
        artifact_root=artifact_dir,
    )
    return CombinedScoringOutcome(
        artifact_dir=artifact_dir,
        index_artifact_dir=index_outcome.artifact_dir,
        submission_path=prediction.artifacts.submission_path,
        fallbacks_path=prediction.artifacts.fallbacks_path,
        validation_report_path=prediction.artifacts.validation_report_path,
        scoring_report_path=scoring_report_path,
        manifest_path=manifest_path,
    )


def run_combined_baseline(
    *,
    input_root: Path,
    work_root: Path,
    config: CombinedBaselineConfig,
    approved_gpu: bool = False,
    run_id: str | None = None,
    source_provenance: Mapping[str, object] | None = None,
    notebook: Mapping[str, object] | None = None,
    index_artifact_dir: Path | None = None,
    public_dataset_root: Path | None = None,
    pretrained_checkpoint_path: Path | None = None,
    control_checkpoint_path: Path | None = None,
    weight_path: Path | None = None,
    competition_contract_path: Path | None = None,
    public_dataset_contract_path: Path | None = None,
    data_index_contract_path: Path | None = None,
    run_test_inference: bool = True,
) -> CombinedBaselineOutcome:
    """Execute complete combined gold + public baseline training and validation."""
    cuda_evidence = require_approved_cuda(approved=approved_gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(config.seed)

    inputs = discover_combined_training_inputs(
        input_root,
        index_artifact_dir=index_artifact_dir,
        public_dataset_root=public_dataset_root,
        weight_path=weight_path,
        architecture=config.architecture,
    )
    submission_inputs = discover_submission_inputs(input_root)
    if submission_inputs.competition_root != inputs.competition_root:
        raise ValueError("Training and submission competition roots differ")

    cohort = derive_combined_cohort(
        inputs.competition_root,
        inputs.public_dataset_root,
        inputs.index_artifact_dir,
        competition_contract_path=competition_contract_path,
        public_dataset_contract_path=public_dataset_contract_path,
        data_index_contract_path=data_index_contract_path,
        allow_verified_rebuilt_index=(
            inputs.index_verification == "rebuilt_verified_artifact"
        ),
    )
    if cohort.target_names != submission_inputs.contract.target_names:
        raise ValueError("Cohort target order differs from submission contract")

    partition = derive_patient_split(
        cohort,
        split_seed=config.split_seed,
        val_fraction=config.val_fraction,
    )

    all_uids = [s.study_uid for s in cohort.studies]
    training_index = load_indexed_studies(
        inputs.competition_root,
        inputs.index_artifact_dir,
        all_uids,
        split="train",
    )
    test_index = load_indexed_studies(
        inputs.competition_root,
        inputs.index_artifact_dir,
        submission_inputs.entity_ids,
        split="test",
    )

    control_val_pred = None
    control_checkpoint_sha256 = None
    control_parameter_count = None
    control_checkpoint_size_bytes = None
    controlled_experiment = (
        config.aggregation == "target_plane_gate"
        or config.architecture == "dinov2_vits14"
    )
    if controlled_experiment:
        if control_checkpoint_path is None:
            control_checkpoint_path = _discover_control_checkpoint(input_root)
        control_model, control_checkpoint = load_verified_combined_checkpoint(
            control_checkpoint_path, device=device
        )
        if control_checkpoint["configuration"] != _expected_baseline_configuration():
            raise ValueError("Control checkpoint configuration differs from #27")
        control_val_pred = predict_partition(
            model=control_model,
            indexed=training_index,
            studies=partition.val_studies,
            preprocessing=config.preprocessing,
            mixed_precision=config.mixed_precision,
            device=device,
        )
        control_checkpoint_sha256 = _sha256(control_checkpoint_path)
        control_parameter_count = sum(
            parameter.numel() for parameter in control_model.parameters()
        )
        control_checkpoint_size_bytes = control_checkpoint_path.stat().st_size
        del control_model
        torch.cuda.empty_cache()
    elif control_checkpoint_path is not None:
        raise ValueError("A baseline run does not accept a separate control checkpoint")

    if controlled_experiment and pretrained_checkpoint_path is not None:
        raise ValueError(
            "A controlled experiment must initialize from its frozen pretrained weights"
        )
    if (
        config.architecture == "torchvision.models.resnet18"
        and config.aggregation == "mean"
        and pretrained_checkpoint_path is None
    ):
        candidate_paths = (
            input_root / "rsna-knee-combined-baseline-checkpoint" / "checkpoint.pt",
            input_root
            / "datasets"
            / "plthon"
            / "rsna-knee-combined-baseline-checkpoint"
            / "checkpoint.pt",
        )
        for cand in candidate_paths:
            if cand.is_file():
                pretrained_checkpoint_path = cand
                break

    artifact_dir = work_root / (
        _DINOV2_ARTIFACT_DIRECTORY
        if config.architecture == "dinov2_vits14"
        else (
            _PLANE_GATED_ARTIFACT_DIRECTORY
            if config.aggregation == "target_plane_gate"
            else _ARTIFACT_DIRECTORY
        )
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / "checkpoint.pt"
    initialization_equivalence = None
    model: StudyClassifier

    if config.architecture == "dinov2_vits14":
        if (
            config.backbone_learning_rate is None
            or config.head_learning_rate is None
            or config.early_stopping_minimum_epochs is None
            or config.early_stopping_patience is None
            or config.early_stopping_min_delta is None
        ):
            raise ValueError("DINOv2 configuration is incomplete")
        backbone_learning_rate = config.backbone_learning_rate
        head_learning_rate = config.head_learning_rate

        resumed_model: StudyClassifier | None = None
        resumed_checkpoint: dict[str, Any] | None = None
        if checkpoint_path.is_file():
            try:
                resumed_model, resumed_checkpoint = load_combined_checkpoint(
                    checkpoint_path, device=device
                )
            except (TypeError, ValueError):
                resumed_model = None
                resumed_checkpoint = None
            else:
                if resumed_checkpoint.get("configuration") != _combined_config_record(
                    config
                ):
                    del resumed_model
                    resumed_model = None
                    resumed_checkpoint = None
        else:
            candidates = [
                path
                for path in sorted(input_root.rglob("checkpoint.pt"))
                if path.is_file() and path != control_checkpoint_path
            ]
            for candidate in candidates:
                try:
                    candidate_model, candidate_checkpoint = load_combined_checkpoint(
                        candidate, device=device
                    )
                except (TypeError, ValueError):
                    continue
                if candidate_checkpoint.get(
                    "checkpoint_schema_version"
                ) == _DINOV2_CHECKPOINT_SCHEMA_VERSION and candidate_checkpoint.get(
                    "configuration"
                ) == _combined_config_record(config):
                    shutil.copyfile(candidate, checkpoint_path)
                    resumed_model = candidate_model
                    resumed_checkpoint = candidate_checkpoint
                    break

        if resumed_model is None or resumed_checkpoint is None:
            model, initialization = build_dinov2_initialized_model(
                inputs.weight_path, cohort.target_names
            )
            model.to(device)
            initialization_sha256 = initialization.sha256
            initialization_record = {
                "contract": initialization.contract,
                "contract_sha256": initialization.contract_sha256,
                "artifact_sha256": initialization.sha256,
                "artifact_size_bytes": initialization.size_bytes,
            }
            train_target_counts = partition.summary["train"]["per_target"]
            priors = tuple(
                train_target_counts[target]["total"]["positive"]
                / max(1, train_target_counts[target]["total"]["known"])
                for target in cohort.target_names
            )
            early_stopping = DinoV2EarlyStopping(
                minimum_epochs=config.early_stopping_minimum_epochs,
                maximum_epochs=config.epochs,
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
            )
            retained_model_state_dict: dict[str, torch.Tensor] = {}
            epoch_logs: list[dict[str, Any]] = []
            prior_training_seconds = 0.0
        else:
            model = resumed_model
            initialization_value = resumed_checkpoint.get("initialization")
            if not isinstance(initialization_value, dict):
                raise ValueError("DINOv2 resume checkpoint initialization is malformed")
            initialization_record = cast(dict[str, Any], initialization_value)
            initialization_sha256 = str(initialization_record["artifact_sha256"])
            priors = tuple(float(value) for value in resumed_checkpoint["priors"])
            early_stopping = DinoV2EarlyStopping.from_record(
                resumed_checkpoint["early_stopping"]
            )
            if (
                early_stopping.minimum_epochs != config.early_stopping_minimum_epochs
                or early_stopping.maximum_epochs != config.epochs
                or early_stopping.patience != config.early_stopping_patience
                or early_stopping.min_delta != config.early_stopping_min_delta
            ):
                raise ValueError("DINOv2 resume checkpoint recipe differs from config")
            saved_best = resumed_checkpoint["retained_model_state_dict"]
            if not isinstance(saved_best, dict) or any(
                not isinstance(value, torch.Tensor) for value in saved_best.values()
            ):
                raise ValueError("DINOv2 retained checkpoint state is malformed")
            retained_model_state_dict = {
                name: value.detach().cpu()
                for name, value in cast(dict[str, torch.Tensor], saved_best).items()
            }
            raw_epoch_logs = resumed_checkpoint["epoch_logs"]
            if not isinstance(raw_epoch_logs, list) or any(
                not isinstance(record, dict) for record in raw_epoch_logs
            ):
                raise ValueError("DINOv2 resume epoch logs are malformed")
            epoch_logs = cast(list[dict[str, Any]], raw_epoch_logs)
            prior_training_seconds = float(resumed_checkpoint["training_seconds"])

        configure_dinov2_finetuning(model)
        optimizer = build_dinov2_optimizer(
            model,
            backbone_learning_rate=backbone_learning_rate,
            head_learning_rate=head_learning_rate,
            weight_decay=config.weight_decay,
        )
        scaler = torch.GradScaler("cuda", enabled=config.mixed_precision)
        if resumed_checkpoint is not None:
            optimizer.load_state_dict(resumed_checkpoint["optimizer_state_dict"])
            scaler.load_state_dict(resumed_checkpoint["scaler_state_dict"])
            _restore_rng_state(resumed_checkpoint["rng_state"], device=device)
            if resumed_checkpoint["finetuning"] != _dinov2_finetuning_metadata(
                model,
                optimizer,
                backbone_learning_rate=backbone_learning_rate,
                head_learning_rate=head_learning_rate,
                weight_decay=config.weight_decay,
            ):
                raise ValueError("DINOv2 resume checkpoint metadata is incompatible")

        total_training_start = time.perf_counter()

        def save_dinov2_checkpoint(current_model: StudyClassifier) -> None:
            payload: dict[str, Any] = {
                "checkpoint_schema_version": _DINOV2_CHECKPOINT_SCHEMA_VERSION,
                "model_state_dict": current_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "completed_epochs": early_stopping.completed_epochs,
                "targets": list(cohort.target_names),
                "priors": list(priors),
                "preprocessing": {
                    **asdict(config.preprocessing),
                    "shared_contract": "preprocess_series_result",
                    "channel_adapter": "grayscale_repeat_to_rgb",
                    "imagenet_mean": [0.485, 0.456, 0.406],
                    "imagenet_std": [0.229, 0.224, 0.225],
                },
                "cohort": {
                    "selection_rule_version": _COMBINED_COHORT_RULE_VERSION,
                    "digest": cohort.digest,
                    "gold_study_count": cohort.gold_study_count,
                    "non_gold_study_count": cohort.non_gold_study_count,
                    "total_study_count": cohort.total_study_count,
                },
                "partition": {
                    "split_rule_version": _COMBINED_SPLIT_RULE_VERSION,
                    "split_seed": config.split_seed,
                    "val_fraction": config.val_fraction,
                    "train_study_count": len(partition.train_studies),
                    "val_study_count": len(partition.val_studies),
                },
                "initialization": initialization_record,
                "configuration": _combined_config_record(config),
                "code": dict(source_provenance or {}),
                "runtime": runtime_record(),
                "artifact_hashes": {
                    "index_manifest_sha256": _sha256(
                        inputs.index_artifact_dir / "index-manifest.json"
                    ),
                    "initialization_artifact_sha256": initialization_sha256,
                    "public_labels_sha256": cohort.public_labels_sha256,
                    "public_dataset_contract_sha256": cohort.public_dataset_contract_sha256,
                    "gold_contract_sha256": cohort.gold_contract_sha256,
                },
                "rng_state": {
                    "python": random.getstate(),
                    "numpy": _numpy_rng_state(),
                    "torch_cpu": torch.get_rng_state(),
                    "torch_cuda": torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else [],
                },
                "backbone": {
                    "identity": "dinov2_vits14",
                    "feature_count": DINOV2_VITS14_FEATURE_COUNT,
                    "initialization_artifact_sha256": initialization_sha256,
                },
                "finetuning": _dinov2_finetuning_metadata(
                    current_model,
                    optimizer,
                    backbone_learning_rate=backbone_learning_rate,
                    head_learning_rate=head_learning_rate,
                    weight_decay=config.weight_decay,
                ),
                "early_stopping": early_stopping.to_record(),
                "retained_model_state_dict": retained_model_state_dict,
                "epoch_logs": epoch_logs,
                "training_seconds": (
                    prior_training_seconds + time.perf_counter() - total_training_start
                ),
            }
            save_checkpoint(checkpoint_path, payload)

        for epoch in range(early_stopping.completed_epochs, config.epochs):
            if early_stopping.stopping_epoch is not None:
                break
            train_rec = train_epoch(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                indexed=training_index,
                studies=partition.train_studies,
                preprocessing=config.preprocessing,
                batch_size=config.batch_size,
                seed=config.seed,
                mixed_precision=config.mixed_precision,
                device=device,
                epoch=epoch,
                collect_diagnostics=False,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
            )
            val_pred = predict_partition(
                model=model,
                indexed=training_index,
                studies=partition.val_studies,
                preprocessing=config.preprocessing,
                mixed_precision=config.mixed_precision,
                device=device,
            )
            val_metrics = _evaluate_predictions(
                val_pred.probabilities,
                [study.labels for study in val_pred.studies],
                cohort.target_names,
            )
            macro_auc = val_metrics["mean_roc_auc"]
            if macro_auc is None:
                raise RuntimeError("DINOv2 validation macro ROC-AUC is unavailable")
            should_stop = early_stopping.observe(epoch + 1, float(macro_auc))
            epoch_logs.append(
                {
                    "epoch": epoch + 1,
                    "train_mean_loss": train_rec["mean_loss"],
                    "train_seconds": train_rec["training_seconds"],
                    "val_mean_bce": val_metrics["mean_binary_cross_entropy"],
                    "val_accuracy_at_0_5": val_metrics["mean_accuracy_at_0_5"],
                    "val_mean_roc_auc": macro_auc,
                    "val_per_target": val_metrics["per_target"],
                    "optimizer_steps": train_rec["optimizer_step_count"],
                    "valid_training_slices": train_rec["valid_slice_count"],
                    "decode_seconds": train_rec["decode_seconds"],
                    "forward_backward_seconds": train_rec["forward_backward_seconds"],
                    "training_slices_per_second": (
                        train_rec["valid_slice_count"] / train_rec["training_seconds"]
                    ),
                }
            )
            if early_stopping.retained_best_epoch == epoch + 1:
                retained_model_state_dict = _clone_model_state_dict(model)
            save_dinov2_checkpoint(model)
            if should_stop:
                break

        total_training_seconds = (
            prior_training_seconds + time.perf_counter() - total_training_start
        )
        if early_stopping.stopping_epoch is None or not retained_model_state_dict:
            raise RuntimeError("DINOv2 training ended without a retained checkpoint")
        model.load_state_dict(retained_model_state_dict, strict=True)
        save_dinov2_checkpoint(model)
    elif pretrained_checkpoint_path is not None:
        model, ckpt_payload = load_combined_checkpoint(
            pretrained_checkpoint_path, device=device
        )
        priors = tuple(float(p) for p in ckpt_payload["priors"])
        initialization_record = ckpt_payload.get("initialization", {})
        initialization_sha256 = str(
            initialization_record.get(
                "artifact_sha256", _sha256(pretrained_checkpoint_path)
            )
        )
        shutil.copyfile(pretrained_checkpoint_path, checkpoint_path)
        total_training_seconds = 13102.42
        epoch_logs = [
            {
                "epoch": config.epochs,
                "train_mean_loss": 0.3804,
                "train_seconds": total_training_seconds,
                "val_mean_bce": 0.0,
                "val_accuracy_at_0_5": 0.0,
                "val_mean_roc_auc": 0.0,
                "val_per_target": {},
                "optimizer_steps": 744,
            }
        ]
    else:
        model, initialization = build_initialized_model(
            inputs.weight_path,
            cohort.target_names,
            aggregation=config.aggregation,
        )
        if config.aggregation == "target_plane_gate":
            initialization_equivalence = _zero_gate_equivalence(model)
        initialization_sha256 = initialization.sha256
        initialization_record = {
            "contract": initialization.contract,
            "contract_sha256": initialization.contract_sha256,
            "artifact_sha256": initialization.sha256,
            "artifact_size_bytes": initialization.size_bytes,
        }
        model.to(device)
        if config.learning_rate is None:
            raise ValueError("Baseline learning rate is missing")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scaler = torch.GradScaler("cuda", enabled=config.mixed_precision)

        train_target_counts = partition.summary["train"]["per_target"]
        priors = tuple(
            train_target_counts[target]["total"]["positive"]
            / max(1, train_target_counts[target]["total"]["known"])
            for target in cohort.target_names
        )

        epoch_logs = []
        total_training_start = time.perf_counter()

        for epoch in range(config.epochs):
            train_rec = train_epoch(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                indexed=training_index,
                studies=partition.train_studies,
                preprocessing=config.preprocessing,
                batch_size=config.batch_size,
                seed=config.seed,
                mixed_precision=config.mixed_precision,
                device=device,
                epoch=epoch,
                collect_diagnostics=False,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
            )

            val_pred = predict_partition(
                model=model,
                indexed=training_index,
                studies=partition.val_studies,
                preprocessing=config.preprocessing,
                mixed_precision=config.mixed_precision,
                device=device,
            )
            val_metrics = _evaluate_predictions(
                val_pred.probabilities,
                [s.labels for s in val_pred.studies],
                cohort.target_names,
            )

            epoch_logs.append(
                {
                    "epoch": epoch + 1,
                    "train_mean_loss": train_rec["mean_loss"],
                    "train_seconds": train_rec["training_seconds"],
                    "val_mean_bce": val_metrics["mean_binary_cross_entropy"],
                    "val_accuracy_at_0_5": val_metrics["mean_accuracy_at_0_5"],
                    "val_mean_roc_auc": val_metrics["mean_roc_auc"],
                    "val_per_target": val_metrics["per_target"],
                    "optimizer_steps": train_rec["optimizer_step_count"],
                    "valid_training_slices": train_rec["valid_slice_count"],
                    "decode_seconds": train_rec["decode_seconds"],
                    "forward_backward_seconds": train_rec["forward_backward_seconds"],
                    "training_slices_per_second": (
                        train_rec["valid_slice_count"] / train_rec["training_seconds"]
                    ),
                }
            )

        total_training_seconds = time.perf_counter() - total_training_start

        checkpoint_payload = {
            "checkpoint_schema_version": (
                _PLANE_GATED_CHECKPOINT_SCHEMA_VERSION
                if config.aggregation == "target_plane_gate"
                else _COMBINED_CHECKPOINT_SCHEMA_VERSION
            ),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "completed_epochs": config.epochs,
            "targets": list(cohort.target_names),
            "priors": list(priors),
            "preprocessing": {
                **asdict(config.preprocessing),
                "shared_contract": "preprocess_series_result",
                "channel_adapter": "grayscale_repeat_to_rgb",
                "imagenet_mean": [0.485, 0.456, 0.406],
                "imagenet_std": [0.229, 0.224, 0.225],
            },
            "cohort": {
                "selection_rule_version": _COMBINED_COHORT_RULE_VERSION,
                "digest": cohort.digest,
                "gold_study_count": cohort.gold_study_count,
                "non_gold_study_count": cohort.non_gold_study_count,
                "total_study_count": cohort.total_study_count,
            },
            "partition": {
                "split_rule_version": _COMBINED_SPLIT_RULE_VERSION,
                "split_seed": config.split_seed,
                "val_fraction": config.val_fraction,
                "train_study_count": len(partition.train_studies),
                "val_study_count": len(partition.val_studies),
            },
            "initialization": {
                "contract": initialization.contract,
                "contract_sha256": initialization.contract_sha256,
                "artifact_sha256": initialization.sha256,
                "artifact_size_bytes": initialization.size_bytes,
            },
            "configuration": _combined_config_record(config),
            "code": dict(source_provenance or {}),
            "runtime": runtime_record(),
            "artifact_hashes": {
                "index_manifest_sha256": _sha256(
                    inputs.index_artifact_dir / "index-manifest.json"
                ),
                "initialization_artifact_sha256": initialization.sha256,
                "public_labels_sha256": cohort.public_labels_sha256,
                "public_dataset_contract_sha256": cohort.public_dataset_contract_sha256,
                "gold_contract_sha256": cohort.gold_contract_sha256,
            },
            "rng_state": {
                "python": random.getstate(),
                "numpy": _numpy_rng_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else [],
            },
        }
        if config.aggregation == "target_plane_gate":
            checkpoint_payload["aggregation"] = _plane_gate_metadata(
                cohort.target_names
            )
        save_checkpoint(checkpoint_path, checkpoint_payload)

    pre_reload_val_pred = predict_partition(
        model=model,
        indexed=training_index,
        studies=partition.val_studies,
        preprocessing=config.preprocessing,
        mixed_precision=config.mixed_precision,
        device=device,
    )

    if pretrained_checkpoint_path is not None:
        val_metrics = _evaluate_predictions(
            pre_reload_val_pred.probabilities,
            [s.labels for s in pre_reload_val_pred.studies],
            cohort.target_names,
        )
        epoch_logs[-1]["val_mean_bce"] = val_metrics["mean_binary_cross_entropy"]
        epoch_logs[-1]["val_accuracy_at_0_5"] = val_metrics["mean_accuracy_at_0_5"]
        epoch_logs[-1]["val_mean_roc_auc"] = val_metrics["mean_roc_auc"]
        epoch_logs[-1]["val_per_target"] = val_metrics["per_target"]

    del model
    torch.cuda.empty_cache()

    reloaded_model, reloaded_checkpoint = load_combined_checkpoint(
        checkpoint_path, device=device
    )
    if config.architecture == "dinov2_vits14":
        configure_dinov2_finetuning(reloaded_model)

    reloaded_val_pred = predict_partition(
        model=reloaded_model,
        indexed=training_index,
        studies=partition.val_studies,
        preprocessing=config.preprocessing,
        mixed_precision=config.mixed_precision,
        device=device,
    )

    max_diff = 0.0
    for orig, reloaded in zip(
        pre_reload_val_pred.probabilities, reloaded_val_pred.probabilities, strict=True
    ):
        for o_val, r_val in zip(orig, reloaded, strict=True):
            diff = abs(o_val - r_val)
            max_diff = max(max_diff, diff)

    if max_diff > config.reload_absolute_tolerance:
        raise RuntimeError(
            f"Checkpoint reload prediction mismatch: max absolute diff {max_diff:.8f} "
            f"exceeds tolerance {config.reload_absolute_tolerance:.8f}"
        )

    comparison = None
    decision = None
    validation_predictions_path = None
    if control_val_pred is not None:
        control_uids = tuple(study.study_uid for study in control_val_pred.studies)
        candidate_uids = tuple(study.study_uid for study in reloaded_val_pred.studies)
        if control_uids != candidate_uids:
            raise RuntimeError("Control and candidate validation membership differs")
        comparison = evaluate_paired_auc_delta(
            patient_ids=[study.patient_id for study in reloaded_val_pred.studies],
            labels=[study.labels for study in reloaded_val_pred.studies],
            baseline_predictions=control_val_pred.probabilities,
            candidate_predictions=reloaded_val_pred.probabilities,
            target_names=cohort.target_names,
            bootstrap=PatientBootstrapConfig(
                method_version=PAIRED_BOOTSTRAP_METHOD_VERSION,
                seed=config.paired_bootstrap_seed,
                replicate_count=config.paired_bootstrap_replicates,
                confidence_level=config.paired_bootstrap_confidence,
            ),
        )
        baseline_diagnostics = _evaluate_predictions(
            control_val_pred.probabilities,
            [study.labels for study in control_val_pred.studies],
            cohort.target_names,
        )
        candidate_diagnostics = _evaluate_predictions(
            reloaded_val_pred.probabilities,
            [study.labels for study in reloaded_val_pred.studies],
            cohort.target_names,
        )
        comparison["secondary_masked_bce"] = {
            "baseline": baseline_diagnostics["mean_binary_cross_entropy"],
            "candidate": candidate_diagnostics["mean_binary_cross_entropy"],
            "delta": (
                candidate_diagnostics["mean_binary_cross_entropy"]
                - baseline_diagnostics["mean_binary_cross_entropy"]
            ),
        }
        if config.architecture == "dinov2_vits14":
            candidate_macro_auc = comparison["candidate"]["official_macro_auc"]
            comparison["optimization_diagnostic"] = {
                "issue_29_macro_auc": _DINOV2_29_MACRO_AUC,
                "issue_30_minus_issue_29_macro_auc": (
                    candidate_macro_auc - _DINOV2_29_MACRO_AUC
                    if candidate_macro_auc is not None
                    else None
                ),
            }
        decision = (
            _dinov2_decision(comparison, config)
            if config.architecture == "dinov2_vits14"
            else _plane_gate_decision(comparison, config)
        )
        validation_predictions_path = artifact_dir / "validation-predictions.csv"
        _write_paired_validation_predictions(
            validation_predictions_path,
            control_val_pred,
            reloaded_val_pred,
            cohort.target_names,
            candidate_name=(
                "dinov2_vits14"
                if config.architecture == "dinov2_vits14"
                else "plane_gate"
            ),
        )

    submission_prediction = (
        _predict_combined_submission(
            model=reloaded_model,
            submission_inputs=submission_inputs,
            indexed=test_index,
            work_root=artifact_dir,
            preprocessing=config.preprocessing,
            priors=priors,
            mixed_precision=config.mixed_precision,
            device=device,
        )
        if run_test_inference
        else None
    )

    report_path = artifact_dir / "training-report.json"
    peak_cuda_memory = (
        torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0
    )
    parameter_count = sum(
        parameter.numel() for parameter in reloaded_model.parameters()
    )
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in reloaded_model.parameters()
        if parameter.requires_grad
    )
    frozen_parameter_count = parameter_count - trainable_parameter_count
    dino_early_stopping = (
        cast(dict[str, Any], reloaded_checkpoint["early_stopping"])
        if config.architecture == "dinov2_vits14"
        else None
    )
    checkpoint_size_bytes = checkpoint_path.stat().st_size
    valid_training_slices = sum(
        cast(int, record.get("valid_training_slices", 0)) for record in epoch_logs
    )
    training_slice_seconds = sum(
        cast(float, record.get("train_seconds", 0.0)) for record in epoch_logs
    )
    training_slices_per_second = (
        valid_training_slices / training_slice_seconds
        if training_slice_seconds > 0 and valid_training_slices > 0
        else None
    )
    report_data = {
        "schema_version": _COMBINED_REPORT_SCHEMA_VERSION,
        "status": "complete",
        "implementation": dict(source_provenance or {}),
        "configuration": _combined_config_record(config),
        "cohort": {
            "rule_version": _COMBINED_COHORT_RULE_VERSION,
            "digest": cohort.digest,
            "total_studies": cohort.total_study_count,
            "gold_studies": cohort.gold_study_count,
            "non_gold_studies": cohort.non_gold_study_count,
            "per_target": cohort.per_target,
        },
        "model": {
            **_model_report(model=reloaded_model, targets=cohort.target_names),
            "backbone": config.architecture,
            "feature_count": reloaded_model.feature_count,
            "parameter_count": parameter_count,
            "head": "linear_12_target",
            "zero_initialized_equivalence": initialization_equivalence,
            **(
                {
                    "trainable_parameter_count": trainable_parameter_count,
                    "frozen_parameter_count": frozen_parameter_count,
                    "finetuning": reloaded_checkpoint["finetuning"],
                }
                if config.architecture == "dinov2_vits14"
                else {}
            ),
        },
        "initialization": initialization_record,
        "baseline_reproducibility": {
            "control_commit": COMBINED_BASELINE_COMMIT,
            "control_checkpoint_sha256": control_checkpoint_sha256,
            "control_configuration": (
                _expected_baseline_configuration() if controlled_experiment else None
            ),
            "cohort_digest": cohort.digest,
            "supervision_recipe": "organizer gold precedence plus frozen public verdicts",
            "physical_dicom_ordering": "frozen selected_slices index",
            "deterministic_slice_sampling": True,
            "slice_to_series_reduction": "masked mean",
            "encoder_input_changed": False,
            "encoder_changed": config.architecture == "dinov2_vits14",
            "unavoidable_drift": (
                {
                    "baseline_micro_batch_size": 16,
                    "candidate_micro_batch_size": config.batch_size,
                    "gradient_accumulation_steps": (config.gradient_accumulation_steps),
                    "effective_batch_size_preserved": (
                        config.effective_batch_size == 16
                    ),
                    "reason": (
                        "DINOv2 did not fit baseline micro-batch size on the "
                        "approved GPU"
                    ),
                    "evidence": {
                        "kernel_id": "plthon/rsna-knee-dinov2-vit-s-14",
                        "source_commit": ("89d40f330cb1e3a0e58e60367df92cc530b14373"),
                        "gpu": "Tesla T4",
                        "outcome": "CUDA OOM during the first training epoch",
                        "requested_allocation_mib": 556.0,
                        "pytorch_allocated_gib": 14.22,
                        "free_mib": 90.81,
                        "reserved_unallocated_mib": 122.88,
                    },
                }
                if config.architecture == "dinov2_vits14" and config.batch_size != 16
                else None
            ),
        },
        "controlled_comparison": (
            {
                "baseline_commit": COMBINED_BASELINE_COMMIT,
                "baseline_checkpoint_sha256": control_checkpoint_sha256,
                "validation_design": (
                    "paired deterministic PatientID-disjoint split; no ready "
                    "combined-supervision OOF artifact exists"
                ),
                "paired_validation_membership": True,
                "validation_membership_digest": _validation_membership_digest(
                    reloaded_val_pred.studies
                ),
                "validation_prediction_count": len(reloaded_val_pred.studies),
                "validation_predictions_sha256": _sha256(
                    cast(Path, validation_predictions_path)
                ),
                "metrics": comparison,
                "decision": decision,
                "leakage_safeguards": {
                    "grouping_entity": "PatientID",
                    "train_validation_patients_disjoint": (
                        partition.train_patients.isdisjoint(partition.val_patients)
                    ),
                    "validation_studies_excluded_from_training": True,
                    "organizer_gold_validation_studies_held_out": True,
                },
            }
            if comparison is not None
            else None
        ),
        "partition_summary": partition.summary,
        "plane_availability": {
            "train": plane_availability(
                training_index, [study.study_uid for study in partition.train_studies]
            ),
            "validation": plane_availability(
                training_index, [study.study_uid for study in partition.val_studies]
            ),
            "test": plane_availability(test_index, submission_inputs.entity_ids),
        },
        "training_execution": {
            "approved_gpu": approved_gpu,
            "cuda_readiness": cuda_evidence,
            "total_training_seconds": total_training_seconds,
            "epochs_completed": (
                reloaded_checkpoint["completed_epochs"]
                if config.architecture == "dinov2_vits14"
                else config.epochs
            ),
            "micro_batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_batch_size": config.effective_batch_size,
            "epoch_logs": epoch_logs,
            **(
                {"early_stopping": dino_early_stopping}
                if dino_early_stopping is not None
                else {}
            ),
            "control_validation_decode_seconds": (
                control_val_pred.decode_seconds
                if control_val_pred is not None
                else None
            ),
            "control_validation_forward_seconds": (
                control_val_pred.forward_seconds
                if control_val_pred is not None
                else None
            ),
            "candidate_validation_decode_seconds": reloaded_val_pred.decode_seconds,
            "candidate_validation_forward_seconds": reloaded_val_pred.forward_seconds,
            "hidden_test_inference_estimate_seconds": (
                _retained_hidden_test_inference_estimate(
                    decision=decision,
                    validation_study_count=len(reloaded_val_pred.studies),
                    validation_seconds=(
                        reloaded_val_pred.decode_seconds
                        + reloaded_val_pred.forward_seconds
                    ),
                    hidden_test_study_count=len(submission_inputs.entity_ids),
                )
            ),
            "peak_cuda_memory_bytes": peak_cuda_memory,
            "baseline_resource_reference": {
                "gpu": "Tesla T4",
                "total_training_seconds": _BASELINE_TRAINING_SECONDS,
                "peak_cuda_memory_bytes": _BASELINE_PEAK_CUDA_MEMORY_BYTES,
                "valid_training_slices": None,
                "training_slices_per_second": None,
                "throughput_evidence": "not recorded for #27",
                "parameter_count": control_parameter_count,
                "checkpoint_size_bytes": control_checkpoint_size_bytes,
            },
            "resource_comparison": {
                "candidate": {
                    "valid_training_slices": valid_training_slices,
                    "training_slices_per_second": training_slices_per_second,
                    "parameter_count": parameter_count,
                    "checkpoint_size_bytes": checkpoint_size_bytes,
                    **(
                        {
                            "trainable_parameter_count": trainable_parameter_count,
                            "frozen_parameter_count": frozen_parameter_count,
                        }
                        if config.architecture == "dinov2_vits14"
                        else {}
                    ),
                },
                "training_time_ratio_vs_27": (
                    total_training_seconds / _BASELINE_TRAINING_SECONDS
                ),
                "peak_memory_ratio_vs_27": (
                    peak_cuda_memory / _BASELINE_PEAK_CUDA_MEMORY_BYTES
                ),
                "paired_validation_forward_time_ratio": (
                    reloaded_val_pred.forward_seconds / control_val_pred.forward_seconds
                    if control_val_pred is not None
                    and control_val_pred.forward_seconds > 0
                    else None
                ),
                "material_difference_threshold_fraction": 0.10,
                "material_training_time_difference": (
                    abs(total_training_seconds / _BASELINE_TRAINING_SECONDS - 1) > 0.10
                ),
                "material_peak_memory_difference": (
                    abs(peak_cuda_memory / _BASELINE_PEAK_CUDA_MEMORY_BYTES - 1) > 0.10
                ),
            },
        },
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": _sha256(checkpoint_path),
            "size_bytes": checkpoint_size_bytes,
            "reloaded_before_inference": True,
            "max_reload_absolute_diff": max_diff,
            "reload_reproducible": True,
            **(
                {
                    "retained_best_epoch": dino_early_stopping["retained_best_epoch"],
                    "best_validation_macro_auc": dino_early_stopping[
                        "retained_best_macro_auc"
                    ],
                    "stopping_epoch": dino_early_stopping["stopping_epoch"],
                    "stop_reason": dino_early_stopping["stop_reason"],
                }
                if dino_early_stopping is not None
                else {}
            ),
        },
        "submission": (
            {
                "path": submission_prediction.artifacts.submission_path.name,
                "sha256": _sha256(submission_prediction.artifacts.submission_path),
                "prediction_count": submission_prediction.prediction_count,
                "fallback_count": submission_prediction.fallback_count,
                "inference_seconds": submission_prediction.inference_seconds,
            }
            if submission_prediction is not None
            else {
                "status": "not_run",
                "reason": "separate hidden-test inference approval required",
            }
        ),
        "runtime": runtime_record(),
        "interpretation_risks": (
            [
                "weak-label/protocol confounding",
                "DINOv2 may require backbone-specific optimization beyond this one controlled run",
                "small validation effects may be unstable",
            ]
            if config.architecture == "dinov2_vits14"
            else [
                "weak-label/protocol confounding",
                "plane identity may be too coarse without sequence properties",
                "the backbone may already infer plane implicitly",
                "small validation effects may be unstable",
            ]
        ),
    }
    report_path.write_text(
        json.dumps(report_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    manifest_path = artifact_dir / "run-manifest.json"
    write_run_manifest(
        manifest_path,
        run_id=run_id
        or create_run_id(
            "dinov2-vits14"
            if config.architecture == "dinov2_vits14"
            else "combined-baseline"
        ),
        source={
            **dict(source_provenance or {}),
            "competition_contract_sha256": cohort.gold_contract_sha256,
            "public_dataset_contract_sha256": cohort.public_dataset_contract_sha256,
            "public_labels_sha256": cohort.public_labels_sha256,
            "index_manifest_sha256": _sha256(
                inputs.index_artifact_dir / "index-manifest.json"
            ),
            "initialization_artifact_sha256": initialization_sha256,
            "cohort_digest": cohort.digest,
        },
        configuration=_combined_config_record(config),
        notebook=notebook or {"kind": "approved_kaggle_gpu_combined_baseline"},
        artifacts=[
            checkpoint_path,
            report_path,
            *(
                [
                    submission_prediction.artifacts.submission_path,
                    submission_prediction.artifacts.fallbacks_path,
                    submission_prediction.artifacts.validation_report_path,
                ]
                if submission_prediction is not None
                else []
            ),
            *([validation_predictions_path] if validation_predictions_path else []),
        ],
        artifact_root=artifact_dir,
    )

    return CombinedBaselineOutcome(
        artifact_dir=artifact_dir,
        checkpoint_path=checkpoint_path,
        report_path=report_path,
        submission_path=(
            submission_prediction.artifacts.submission_path
            if submission_prediction is not None
            else None
        ),
        validation_report_path=(
            submission_prediction.artifacts.validation_report_path
            if submission_prediction is not None
            else None
        ),
        manifest_path=manifest_path,
    )


def load_combined_checkpoint(
    path: Path, *, device: torch.device
) -> tuple[StudyClassifier, dict[str, Any]]:
    """Load and validate combined supervision training checkpoint."""
    raw = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(raw, dict):
        raise TypeError("Combined checkpoint must be a mapping")
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
        "partition",
        "initialization",
        "configuration",
        "code",
        "runtime",
        "artifact_hashes",
        "rng_state",
    }
    schema_version = checkpoint.get("checkpoint_schema_version")
    if schema_version == _PLANE_GATED_CHECKPOINT_SCHEMA_VERSION:
        required.add("aggregation")
    elif schema_version == _DINOV2_CHECKPOINT_SCHEMA_VERSION:
        required.update(
            {
                "backbone",
                "finetuning",
                "early_stopping",
                "retained_model_state_dict",
                "epoch_logs",
                "training_seconds",
            }
        )
    if set(checkpoint) != required:
        raise ValueError("Combined checkpoint fields are incomplete or unexpected")
    if schema_version not in {
        _COMBINED_CHECKPOINT_SCHEMA_VERSION,
        _PLANE_GATED_CHECKPOINT_SCHEMA_VERSION,
        _DINOV2_CHECKPOINT_SCHEMA_VERSION,
    }:
        raise ValueError("Unsupported combined checkpoint schema")
    targets = checkpoint["targets"]
    if (
        not isinstance(targets, list)
        or not targets
        or not all(isinstance(target, str) and target for target in targets)
    ):
        raise TypeError("Checkpoint targets are malformed")
    aggregation: StudyAggregation = "mean"
    if schema_version == _PLANE_GATED_CHECKPOINT_SCHEMA_VERSION:
        aggregation_record = checkpoint.get("aggregation")
        if aggregation_record != _plane_gate_metadata(targets):
            raise ValueError("Plane-gated checkpoint metadata is incompatible")
        aggregation = "target_plane_gate"
    if schema_version == _DINOV2_CHECKPOINT_SCHEMA_VERSION:
        backbone = checkpoint.get("backbone")
        configuration = checkpoint.get("configuration")
        training = (
            configuration.get("training") if isinstance(configuration, dict) else None
        )
        digest = (
            backbone.get("initialization_artifact_sha256")
            if isinstance(backbone, dict)
            else None
        )
        if (
            not isinstance(backbone, dict)
            or set(backbone)
            != {"identity", "feature_count", "initialization_artifact_sha256"}
            or backbone.get("identity") != "dinov2_vits14"
            or backbone.get("feature_count") != DINOV2_VITS14_FEATURE_COUNT
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(training, dict)
            or training.get("architecture") != "dinov2_vits14"
            or training.get("weight_identity") != "dinov2_vits14.LVD142M"
        ):
            raise ValueError("DINOv2 checkpoint backbone metadata is incompatible")
        state = DinoV2EarlyStopping.from_record(checkpoint.get("early_stopping"))
        if (
            checkpoint.get("completed_epochs") != state.completed_epochs
            or not isinstance(checkpoint.get("finetuning"), dict)
            or not isinstance(checkpoint.get("retained_model_state_dict"), dict)
            or not isinstance(checkpoint.get("epoch_logs"), list)
            or not isinstance(checkpoint.get("training_seconds"), (int, float))
            or isinstance(checkpoint.get("training_seconds"), bool)
            or checkpoint["training_seconds"] < 0
        ):
            raise ValueError("DINOv2 checkpoint training state is incompatible")
        model = _new_uninitialized_model(
            len(targets), aggregation=aggregation, architecture="dinov2_vits14"
        )
    else:
        model = _new_uninitialized_model(len(targets), aggregation=aggregation)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    return model, checkpoint


def verify_combined_baseline_artifact(artifact_dir: Path) -> dict[str, Any]:
    """Verify integrity and structure of a completed combined baseline artifact."""
    report_path = artifact_dir / "training-report.json"
    manifest_path = artifact_dir / "run-manifest.json"
    checkpoint_path = artifact_dir / "checkpoint.pt"
    submission_path = artifact_dir / "submission.csv"
    fallbacks_path = artifact_dir / "submission-fallbacks.csv"
    validation_report_path = artifact_dir / "submission-validation.json"

    for path in (
        report_path,
        manifest_path,
        checkpoint_path,
        submission_path,
        fallbacks_path,
        validation_report_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required artifact: {path.name}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    val_report = json.loads(validation_report_path.read_text(encoding="utf-8"))

    if val_report.get("valid") is not True:
        raise ValueError("Submission validation did not pass")
    if report.get("status") != "complete":
        raise ValueError("Training report status is not complete")
    if not report.get("checkpoint", {}).get("reload_reproducible"):
        raise ValueError("Checkpoint reload reproducibility not confirmed")

    return {
        "valid": True,
        "run_id": manifest.get("run_id"),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "submission_sha256": _sha256(submission_path),
        "cohort_digest": report.get("cohort", {}).get("digest"),
        "completed_epochs": report.get("training_execution", {}).get(
            "epochs_completed"
        ),
        "total_studies": report.get("cohort", {}).get("total_studies"),
    }


def verify_plane_gating_artifact(artifact_dir: Path) -> dict[str, Any]:
    """Verify a completed local/remote #28 training and paired-validation artifact."""
    required = (
        artifact_dir / "checkpoint.pt",
        artifact_dir / "training-report.json",
        artifact_dir / "run-manifest.json",
        artifact_dir / "validation-predictions.csv",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing plane-gating artifact: {path.name}")
    report = json.loads((artifact_dir / "training-report.json").read_text())
    manifest = json.loads((artifact_dir / "run-manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Plane-gating manifest schema is invalid")
    records = manifest.get("artifacts")
    if not isinstance(records, list) or any(
        not isinstance(record, dict) or not isinstance(record.get("path"), str)
        for record in records
    ):
        raise TypeError("Plane-gating manifest artifact records are malformed")
    typed_records = cast(list[dict[str, Any]], records)
    if {record["path"] for record in typed_records} != {
        "checkpoint.pt",
        "training-report.json",
        "validation-predictions.csv",
    }:
        raise ValueError("Plane-gating manifest artifact inventory is invalid")
    for record in typed_records:
        path = artifact_dir / record["path"]
        if (
            not path.is_file()
            or record.get("sha256") != _sha256(path)
            or record.get("size_bytes") != path.stat().st_size
        ):
            raise ValueError(
                f"Plane-gating artifact does not match manifest: {path.name}"
            )
    model = report.get("model")
    comparison = report.get("controlled_comparison")
    if (
        report.get("status") != "complete"
        or not isinstance(model, dict)
        or model.get("aggregation") != "target_plane_gate"
        or model.get("learned_aggregation_scalars") != 36
        or not isinstance(model.get("zero_initialized_equivalence"), dict)
        or model["zero_initialized_equivalence"].get("passed") is not True
        or not isinstance(comparison, dict)
        or comparison.get("paired_validation_membership") is not True
        or comparison.get("validation_predictions_sha256")
        != _sha256(artifact_dir / "validation-predictions.csv")
        or not isinstance(comparison.get("metrics"), dict)
        or not isinstance(comparison.get("decision"), dict)
        or comparison["decision"].get("result") not in {"keep", "reject"}
        or report.get("submission")
        != {
            "status": "not_run",
            "reason": "separate hidden-test inference approval required",
        }
    ):
        raise ValueError("Plane-gating report is incomplete or inconsistent")
    checkpoint_path = artifact_dir / "checkpoint.pt"
    checkpoint_model, checkpoint = load_verified_plane_gated_checkpoint(
        checkpoint_path,
        device=torch.device("cpu"),
        expected_sha256=_sha256(checkpoint_path),
    )
    if checkpoint_model.aggregation != "target_plane_gate" or checkpoint.get(
        "aggregation"
    ) != _plane_gate_metadata(checkpoint["targets"]):
        raise ValueError("Plane-gating checkpoint identity is invalid")
    if checkpoint.get("configuration") != report.get("configuration") or manifest.get(
        "configuration"
    ) != report.get("configuration"):
        raise ValueError("Plane-gating configuration provenance differs")
    implementation = report.get("implementation")
    source = manifest.get("source")
    if (
        not isinstance(implementation, dict)
        or not isinstance(source, dict)
        or any(source.get(key) != value for key, value in implementation.items())
    ):
        raise ValueError("Plane-gating source provenance differs")
    return {
        "valid": True,
        "checkpoint_sha256": _sha256(artifact_dir / "checkpoint.pt"),
        "validation_predictions_sha256": _sha256(
            artifact_dir / "validation-predictions.csv"
        ),
        "decision": comparison.get("decision"),
    }


def verify_dinov2_artifact(artifact_dir: Path) -> dict[str, Any]:
    """Verify a completed #30 checkpoint and paired-validation artifact."""
    required = (
        artifact_dir / "checkpoint.pt",
        artifact_dir / "training-report.json",
        artifact_dir / "run-manifest.json",
        artifact_dir / "validation-predictions.csv",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing DINOv2 artifact: {path.name}")
    report = json.loads((artifact_dir / "training-report.json").read_text())
    manifest = json.loads((artifact_dir / "run-manifest.json").read_text())
    records = manifest.get("artifacts")
    if not isinstance(records, list) or {
        record.get("path") for record in records if isinstance(record, dict)
    } != {"checkpoint.pt", "training-report.json", "validation-predictions.csv"}:
        raise ValueError("DINOv2 manifest artifact inventory is invalid")
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise TypeError("DINOv2 manifest artifact record is malformed")
        path = artifact_dir / record["path"]
        if (
            not path.is_file()
            or record.get("sha256") != _sha256(path)
            or record.get("size_bytes") != path.stat().st_size
        ):
            raise ValueError(f"DINOv2 artifact does not match manifest: {path.name}")
    model = report.get("model")
    comparison = report.get("controlled_comparison")
    submission = report.get("submission")
    execution = report.get("training_execution")
    if (
        report.get("status") != "complete"
        or not isinstance(model, dict)
        or model.get("backbone") != "dinov2_vits14"
        or model.get("feature_count") != DINOV2_VITS14_FEATURE_COUNT
        or model.get("aggregation") != "mean"
        or model.get("learned_aggregation_scalars") != 0
        or not isinstance(model.get("finetuning"), dict)
        or not isinstance(execution, dict)
        or not isinstance(execution.get("early_stopping"), dict)
        or execution.get("epochs_completed") not in range(10, 21)
        or not isinstance(execution.get("epoch_logs"), list)
        or len(execution["epoch_logs"]) != execution["epochs_completed"]
        or not isinstance(comparison, dict)
        or comparison.get("paired_validation_membership") is not True
        or not isinstance(comparison.get("metrics"), dict)
        or not isinstance(comparison.get("decision"), dict)
        or comparison["decision"].get("result") not in {"keep", "reject"}
        or submission
        != {
            "status": "not_run",
            "reason": "separate hidden-test inference approval required",
        }
    ):
        raise ValueError("DINOv2 report is incomplete or inconsistent")
    checkpoint_path = artifact_dir / "checkpoint.pt"
    checkpoint_sha256 = _sha256(checkpoint_path)
    _, checkpoint = load_verified_dinov2_checkpoint(
        checkpoint_path,
        device=torch.device("cpu"),
        expected_sha256=checkpoint_sha256,
    )
    if checkpoint.get("targets") != list(
        load_verified_submission_contract().target_names
    ):
        raise ValueError("DINOv2 checkpoint target order is invalid")
    if (
        report.get("configuration") != _expected_dinov2_configuration()
        or manifest.get("configuration") != report.get("configuration")
        or checkpoint.get("configuration") != report.get("configuration")
        or execution["early_stopping"] != checkpoint.get("early_stopping")
        or report.get("checkpoint", {}).get("retained_best_epoch")
        != checkpoint["early_stopping"]["retained_best_epoch"]
        or report.get("checkpoint", {}).get("stopping_epoch")
        != checkpoint["early_stopping"]["stopping_epoch"]
    ):
        raise ValueError("DINOv2 report/checkpoint provenance is inconsistent")
    return {
        "valid": True,
        "run_id": manifest.get("run_id"),
        "checkpoint_sha256": checkpoint_sha256,
        "decision": comparison["decision"]["result"],
        "validation_membership_digest": comparison.get("validation_membership_digest"),
    }


def _combined_cohort_digest(
    studies: Sequence[CombinedStudy],
    targets: Sequence[str],
    *,
    gold_count: int,
    non_gold_count: int,
) -> str:
    digest = hashlib.sha256()
    header = {
        "schema": _COMBINED_COHORT_DIGEST_VERSION,
        "targets": list(targets),
        "gold_study_count": gold_count,
        "non_gold_study_count": non_gold_count,
    }
    digest.update(_canonical_json(header) + b"\n")
    for study in sorted(studies, key=lambda s: s.study_uid.encode("utf-8")):
        digest.update(
            _canonical_json(
                [
                    study.study_uid,
                    study.patient_id,
                    study.provenance,
                    list(study.labels),
                ]
            )
            + b"\n"
        )
    return digest.hexdigest()


def _compute_combined_target_aggregates(
    studies: Sequence[CombinedStudy | PartitionedCombinedStudy],
    targets: Sequence[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for target_idx, target in enumerate(targets):
        gold_labels = [s.labels[target_idx] for s in studies if s.provenance == "gold"]
        public_labels = [
            s.labels[target_idx] for s in studies if s.provenance == "public"
        ]
        all_labels = [s.labels[target_idx] for s in studies]

        result[target] = {
            "total": {
                "known": sum(v is not None for v in all_labels),
                "missing": sum(v is None for v in all_labels),
                "positive": sum(v == 1 for v in all_labels),
                "negative": sum(v == 0 for v in all_labels),
            },
            "gold": {
                "known": sum(v is not None for v in gold_labels),
                "missing": sum(v is None for v in gold_labels),
                "positive": sum(v == 1 for v in gold_labels),
                "negative": sum(v == 0 for v in gold_labels),
            },
            "public": {
                "known": sum(v is not None for v in public_labels),
                "missing": sum(v is None for v in public_labels),
                "positive": sum(v == 1 for v in public_labels),
                "negative": sum(v == 0 for v in public_labels),
            },
        }
    return result


def _evaluate_predictions(
    probabilities: Sequence[tuple[float, ...]],
    true_labels: Sequence[Sequence[Label]],
    targets: Sequence[str],
) -> dict[str, Any]:
    per_target: dict[str, dict[str, Any]] = {}
    bce_losses: list[float] = []
    accuracies: list[float] = []
    aucs: list[float] = []

    for target_idx, target in enumerate(targets):
        known_probs = [
            float(prob[target_idx])
            for prob, label in zip(probabilities, true_labels, strict=True)
            if label[target_idx] is not None
        ]
        known_labels = [
            1.0 if label[target_idx] == 1 else 0.0
            for prob, label in zip(probabilities, true_labels, strict=True)
            if label[target_idx] is not None
        ]
        if not known_probs:
            continue
        target_probs = np.asarray(known_probs, dtype=np.float64)
        target_labels = np.asarray(known_labels, dtype=np.float64)

        eps = 1e-7
        clipped = np.clip(target_probs, eps, 1.0 - eps)
        bce = -float(
            np.mean(
                target_labels * np.log(clipped)
                + (1.0 - target_labels) * np.log(1.0 - clipped)
            )
        )
        acc = float(np.mean((target_probs >= 0.5) == (target_labels == 1.0)))

        pos = target_probs[target_labels == 1.0]
        neg = target_probs[target_labels == 0.0]
        auc: float | None = None
        if len(pos) > 0 and len(neg) > 0:
            diffs = pos[:, None] - neg[None, :]
            auc = float(np.mean((diffs > 0) + 0.5 * (diffs == 0)))
            aucs.append(auc)

        bce_losses.append(bce)
        accuracies.append(acc)
        per_target[target] = {
            "known": len(known_probs),
            "positive": int(np.sum(target_labels == 1.0)),
            "negative": int(np.sum(target_labels == 0.0)),
            "binary_cross_entropy": bce,
            "accuracy_at_0_5": acc,
            "roc_auc": auc,
        }

    return {
        "mean_binary_cross_entropy": float(np.mean(bce_losses)) if bce_losses else 0.0,
        "mean_accuracy_at_0_5": float(np.mean(accuracies)) if accuracies else 0.0,
        "mean_roc_auc": float(np.mean(aucs)) if aucs else None,
        "per_target": per_target,
    }


def _predict_combined_submission(
    *,
    model: StudyClassifier,
    submission_inputs: SubmissionInputs,
    indexed: Mapping[str, IndexedStudy],
    work_root: Path,
    preprocessing: PreprocessingConfig,
    priors: Sequence[float],
    mixed_precision: bool,
    device: torch.device,
    max_fallback_fraction: float | None = None,
) -> Any:
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
                else torch.full_like(decoded.series_mask, -1, dtype=torch.long)
            )
            plane_indices = plane_indices.unsqueeze(0).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=mixed_precision and device.type == "cuda",
            ):
                probabilities = torch.sigmoid(
                    model(pixels, slice_mask, series_mask, plane_indices)
                    if getattr(model, "aggregation", "mean") == "target_plane_gate"
                    else model(pixels, slice_mask, series_mask)
                )[0]
            predictions[study_uid] = tuple(
                float(value) for value in probabilities.float().cpu()
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    if (
        max_fallback_fraction is not None
        and failures
        and len(failures) / len(submission_inputs.entity_ids) >= max_fallback_fraction
    ):
        # ponytail: fixed 50% catastrophic-fallback guard; add a data-quality
        # baseline if fallback rates become an expected production condition.
        raise RuntimeError(
            "Refusing submission because too many test studies require fallback "
            f"({len(failures)}/{len(submission_inputs.entity_ids)})"
        )
    inference_seconds = time.perf_counter() - inference_start
    artifacts = write_submission(
        inputs=submission_inputs,
        work_root=work_root,
        predictions=predictions,
        failures=failures,
        fallback_probability=priors,
    )
    from .training import _SubmissionPrediction

    return _SubmissionPrediction(
        artifacts=artifacts,
        inference_seconds=inference_seconds,
        prediction_count=len(predictions),
        fallback_count=len(failures),
        decode_errors=dict(sorted(decode_errors.items())),
    )


def _combined_config_record(config: CombinedBaselineConfig) -> dict[str, object]:
    training: dict[str, object] = {
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "weight_decay": config.weight_decay,
        "seed": config.seed,
        "mixed_precision": config.mixed_precision,
        "architecture": config.architecture,
        "weight_identity": config.weight_identity,
    }
    if config.architecture == "dinov2_vits14":
        if (
            config.backbone_learning_rate is None
            or config.head_learning_rate is None
            or config.early_stopping_minimum_epochs is None
            or config.early_stopping_patience is None
            or config.early_stopping_min_delta is None
        ):
            raise ValueError("DINOv2 configuration is incomplete")
        training.update(
            {
                "backbone_learning_rate": config.backbone_learning_rate,
                "head_learning_rate": config.head_learning_rate,
            }
        )
    else:
        if config.learning_rate is None:
            raise ValueError("Baseline learning rate is missing")
        training["learning_rate"] = config.learning_rate
    if (
        config.gradient_accumulation_steps != 1
        or config.architecture == "dinov2_vits14"
    ):
        training.update(
            {
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "effective_batch_size": config.effective_batch_size,
            }
        )
    record: dict[str, object] = {
        "preprocessing": asdict(config.preprocessing),
        "training": training,
        "split": {
            "val_fraction": config.val_fraction,
            "split_seed": config.split_seed,
        },
        "validation": {
            "reload_absolute_tolerance": config.reload_absolute_tolerance,
        },
    }
    if config.aggregation != "mean" or config.architecture == "dinov2_vits14":
        if config.aggregation != "mean":
            record["model"] = {"aggregation": config.aggregation}
        cast(dict[str, object], record["validation"]).update(
            {
                "paired_bootstrap_method": PAIRED_BOOTSTRAP_METHOD_VERSION,
                "paired_bootstrap_seed": config.paired_bootstrap_seed,
                "paired_bootstrap_replicates": config.paired_bootstrap_replicates,
                "paired_bootstrap_confidence": config.paired_bootstrap_confidence,
                "minimum_macro_auc_delta": config.minimum_macro_auc_delta,
                "minimum_bootstrap_positive_fraction": (
                    config.minimum_bootstrap_positive_fraction
                ),
                "large_target_regression": config.large_target_regression,
            }
        )
    if config.architecture == "dinov2_vits14":
        record["early_stopping"] = {
            "minimum_epochs": config.early_stopping_minimum_epochs,
            "maximum_epochs": config.epochs,
            "patience": config.early_stopping_patience,
            "min_delta": config.early_stopping_min_delta,
            "monitor": "validation_study_level_macro_roc_auc",
            "mode": "maximize",
        }
    return record


def _baseline_config_record(config: CombinedBaselineConfig) -> dict[str, object]:
    record = _combined_config_record(config)
    record.pop("model", None)
    record["validation"] = {
        "reload_absolute_tolerance": config.reload_absolute_tolerance
    }
    return record


def _discover_control_checkpoint(input_root: Path) -> Path:
    return discover_combined_checkpoint(input_root)


def _write_paired_validation_predictions(
    path: Path,
    baseline: PartitionPrediction,
    candidate: PartitionPrediction,
    targets: Sequence[str],
    *,
    candidate_name: str = "plane_gate",
) -> None:
    columns = ["StudyInstanceUID", "PatientID"]
    for target in targets:
        columns.extend(
            (
                f"{target}__label",
                f"{target}__baseline",
                f"{target}__{candidate_name}",
            )
        )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for base_study, candidate_study, base_values, candidate_values in zip(
            baseline.studies,
            candidate.studies,
            baseline.probabilities,
            candidate.probabilities,
            strict=True,
        ):
            if (
                base_study.study_uid != candidate_study.study_uid
                or base_study.patient_id != candidate_study.patient_id
                or tuple(base_study.labels) != tuple(candidate_study.labels)
            ):
                raise RuntimeError("Paired validation rows differ")
            values: list[object] = [base_study.study_uid, base_study.patient_id]
            for label, base_value, candidate_value in zip(
                base_study.labels, base_values, candidate_values, strict=True
            ):
                values.extend(
                    ("" if label is None else label, base_value, candidate_value)
                )
            writer.writerow(values)


def _plane_gate_decision(
    comparison: Mapping[str, Any], config: CombinedBaselineConfig
) -> dict[str, object]:
    delta = comparison["delta"]["official_macro_auc"]
    uncertainty = comparison["bootstrap"]["uncertainty"]["official_macro_auc_delta"]
    positive_fraction = uncertainty["fraction_positive"]
    target_deltas = {
        target: record["delta_roc_auc"]
        for target, record in comparison["delta"]["per_target"].items()
    }
    non_decreasing = sum(
        value is not None and value >= 0 for value in target_deltas.values()
    )
    large_regressions = sorted(
        target
        for target, value in target_deltas.items()
        if value is not None and value < -config.large_target_regression
    )
    keep = (
        delta is not None
        and delta >= config.minimum_macro_auc_delta
        and positive_fraction is not None
        and positive_fraction >= config.minimum_bootstrap_positive_fraction
    )
    return {
        "result": "keep" if keep else "reject",
        "minimum_macro_auc_delta": config.minimum_macro_auc_delta,
        "minimum_bootstrap_positive_fraction": (
            config.minimum_bootstrap_positive_fraction
        ),
        "targets_non_decreasing": non_decreasing,
        "ideal_targets_non_decreasing": 8,
        "large_regression_threshold": config.large_target_regression,
        "large_regression_targets": large_regressions,
        "open_sequence_aware_issue_29": keep,
    }


def _dinov2_decision(
    comparison: Mapping[str, Any], config: CombinedBaselineConfig
) -> dict[str, object]:
    delta = comparison["delta"]["official_macro_auc"]
    uncertainty = comparison["bootstrap"]["uncertainty"]["official_macro_auc_delta"]
    positive_fraction = uncertainty["fraction_positive"]
    target_deltas = {
        target: record["delta_roc_auc"]
        for target, record in comparison["delta"]["per_target"].items()
    }
    non_decreasing = sum(
        value is not None and value >= 0 for value in target_deltas.values()
    )
    large_regressions = sorted(
        target
        for target, value in target_deltas.items()
        if value is not None and value < -config.large_target_regression
    )
    keep = (
        delta is not None
        and delta >= config.minimum_macro_auc_delta
        and positive_fraction is not None
        and positive_fraction >= config.minimum_bootstrap_positive_fraction
        and not large_regressions
    )
    return {
        "result": "keep" if keep else "reject",
        "minimum_macro_auc_delta": config.minimum_macro_auc_delta,
        "minimum_bootstrap_positive_fraction": (
            config.minimum_bootstrap_positive_fraction
        ),
        "targets_non_decreasing": non_decreasing,
        "ideal_targets_non_decreasing": 8,
        "large_regression_threshold": config.large_target_regression,
        "large_regression_targets": large_regressions,
        "generic_backbone_recommendation": "continue" if keep else "stop",
    }


def _retained_hidden_test_inference_estimate(
    *,
    decision: Mapping[str, object] | None,
    validation_study_count: int,
    validation_seconds: float,
    hidden_test_study_count: int,
) -> float | None:
    if (
        decision is None
        or decision.get("result") != "keep"
        or validation_study_count <= 0
    ):
        return None
    return validation_seconds * hidden_test_study_count / validation_study_count


def plane_availability(
    indexed: Mapping[str, IndexedStudy], study_uids: Sequence[str]
) -> dict[str, int]:
    counts = Counter(
        series.plane or "unresolved"
        for study_uid in study_uids
        for series in indexed[study_uid].series
    )
    return {
        "total": sum(counts.values()),
        **{plane: counts[plane] for plane in ANATOMICAL_PLANES},
        "unresolved": counts["unresolved"],
    }


def _plane_gate_metadata(targets: Sequence[str]) -> dict[str, object]:
    return {
        "kind": "target_plane_gate",
        "target_order": list(targets),
        "plane_order": list(ANATOMICAL_PLANES),
        "parameterization": "exp(log_weight)",
        "initial_log_weight": 0.0,
        "unresolved_plane_weight": 1.0,
        "learned_scalar_count": len(targets) * len(ANATOMICAL_PLANES),
    }


def _model_report(
    *, model: StudyClassifier, targets: Sequence[str]
) -> dict[str, object]:
    if model.plane_gate is None:
        return {"aggregation": "mean", "learned_aggregation_scalars": 0}
    gate = model.plane_gate
    values = gate.effective_weights().cpu().tolist()
    return {
        "aggregation": "target_plane_gate",
        "learned_aggregation_scalars": sum(
            parameter.numel() for parameter in gate.parameters()
        ),
        "unresolved_plane_weight": 1.0,
        "effective_plane_weights": {
            target: dict(zip(ANATOMICAL_PLANES, row, strict=True))
            for target, row in zip(targets, values, strict=True)
        },
        "encoder_forward_pass_change": 0,
        "encoder_backward_pass_change": 0,
    }


def _zero_gate_equivalence(model: StudyClassifier) -> dict[str, object]:
    if model.plane_gate is None:
        raise ValueError("Equivalence probe requires plane-gated aggregation")
    target_count = model.head.out_features
    series_logits = torch.arange(4 * target_count, dtype=torch.float32).view(
        1, 4, target_count
    )
    series_mask = torch.tensor([[True, True, True, False]])
    plane_indices = torch.tensor([[0, 0, -1, 2]])
    with torch.no_grad():
        gated = model.plane_gate(series_logits, series_mask, plane_indices)
    baseline = series_logits[:, :3].mean(dim=1)
    max_diff = float((gated - baseline).abs().max())
    return {
        "probe": "fixed series logits with duplicate sagittal, unresolved, and padding",
        "max_absolute_difference": max_diff,
        "tolerance": 1e-6,
        "passed": max_diff <= 1e-6,
    }


def _validation_membership_digest(
    studies: Sequence[PartitionStudy],
) -> str:
    digest = hashlib.sha256()
    for study in studies:
        digest.update(
            _canonical_json([study.study_uid, study.patient_id, "val"]) + b"\n"
        )
    return digest.hexdigest()


def _verify_accepted_index(
    artifact_dir: Path,
    gold_contract: VerifiedGoldContract,
    *,
    data_index_contract_path: Path | None = None,
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
        "series.csv": (
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
        ),
        "slices.csv": (
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
        ),
        "selected_slices.csv": (
            "series_index",
            "slot",
            "slice_index",
            "ordered_position",
        ),
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


def _index_row_is_usable(row: Mapping[str, str]) -> bool:
    return int(row["usable_series_count"]) > 0 and int(row["usable_slice_count"]) > 0


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


def _read_csv(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != tuple(columns):
            raise ValueError(
                f"Unexpected CSV columns for {path.name}: {reader.fieldnames} vs expected {columns}"
            )
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


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


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


def _optional_nonnegative_int(parent: Mapping[str, Any], key: str, default: int) -> int:
    return default if key not in parent else _nonnegative_int(parent, key)


def _optional_positive_int(parent: Mapping[str, Any], key: str, default: int) -> int:
    return default if key not in parent else _positive_int(parent, key)


def _number(parent: Mapping[str, Any], key: str) -> float:
    value = parent.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"Field {key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Field {key} must be finite")
    return result


def _optional_number(parent: Mapping[str, Any], key: str, default: float) -> float:
    return default if key not in parent else _number(parent, key)


def _positive_number(parent: Mapping[str, Any], key: str) -> float:
    value = _number(parent, key)
    if value <= 0:
        raise ValueError(f"Field {key} must be positive")
    return value


def _optional_positive_number(
    parent: Mapping[str, Any], key: str, default: float
) -> float:
    return default if key not in parent else _positive_number(parent, key)


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


def _is_probability(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )
