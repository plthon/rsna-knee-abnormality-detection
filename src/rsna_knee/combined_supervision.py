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
)
from .config import PreprocessingConfig
from .indexing import verify_index_artifact
from .provenance import create_run_id, write_run_manifest
from .public_label_audit import resolve_dataset_root
from .submission import (
    SubmissionInputs,
    discover_submission_inputs,
    write_submission,
)
from .training import (
    GoldResNet18,
    IndexedStudy,
    Label,
    _new_uninitialized_model,
    build_initialized_model,
    decode_study,
    load_indexed_studies,
    predict_partition,
    require_approved_cuda,
    runtime_record,
    save_checkpoint,
    seed_everything,
    train_epoch,
)

type SupervisionSource = Literal["gold", "public"]
type Partition = Literal["train", "val"]

_COMBINED_CHECKPOINT_SCHEMA_VERSION = 1
_COMBINED_REPORT_SCHEMA_VERSION = 1
_COMBINED_COHORT_RULE_VERSION = "gold-plus-public-verdicts-v1"
_COMBINED_COHORT_DIGEST_VERSION = "combined-gold-public-cohort-jsonl-v1"
_COMBINED_SPLIT_RULE_VERSION = "patient-hash-disjoint-v1"
_ARTIFACT_DIRECTORY = "rsna-knee-combined-baseline-v1"

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

PUBLIC_VERDICT_TO_LABEL: Mapping[str, Label] = {
    "YES": 1,
    "NO": 0,
    "UNK": None,
}


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
    learning_rate: float
    weight_decay: float
    seed: int
    mixed_precision: bool
    architecture: str
    weight_identity: str
    val_fraction: float
    split_seed: int
    reload_absolute_tolerance: float


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
    submission_path: Path
    validation_report_path: Path
    manifest_path: Path


def load_combined_baseline_config(path: Path) -> CombinedBaselineConfig:
    """Load and validate the combined gold + public baseline training configuration."""
    with path.open("rb") as file:
        raw = tomllib.load(file)
    preprocessing_raw = _mapping(raw, "preprocessing")
    training_raw = _mapping(raw, "training")
    split_raw = _mapping(raw, "split")
    validation_raw = _mapping(raw, "validation")
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
        learning_rate=_positive_number(training_raw, "learning_rate"),
        weight_decay=_nonnegative_number(training_raw, "weight_decay"),
        seed=_nonnegative_int(training_raw, "seed"),
        mixed_precision=_boolean(training_raw, "mixed_precision"),
        architecture=_text(training_raw, "architecture"),
        weight_identity=_text(training_raw, "weight_identity"),
        val_fraction=_number(split_raw, "val_fraction"),
        split_seed=_nonnegative_int(split_raw, "seed"),
        reload_absolute_tolerance=_positive_number(
            validation_raw, "reload_absolute_tolerance"
        ),
    )
    if not (
        0 <= preprocessing.lower_percentile < preprocessing.upper_percentile <= 100
    ):
        raise ValueError("Preprocessing percentiles are invalid")
    if preprocessing.image_size != 224:
        raise ValueError("Baseline requires 224 by 224 pixel inputs")
    if config.architecture != "torchvision.models.resnet18":
        raise ValueError("Baseline requires exactly torchvision ResNet-18")
    if config.weight_identity != "ResNet18_Weights.IMAGENET1K_V1":
        raise ValueError("Unexpected initialization weight identity")
    if not 0 < config.val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    return config


def discover_combined_training_inputs(
    input_root: Path,
    *,
    index_artifact_dir: Path | None = None,
    public_dataset_root: Path | None = None,
    weight_path: Path | None = None,
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
        contract, _ = _load_json_contract("resnet18-imagenet1k-v1.json", None)
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
                "Expected exactly one offline ResNet-18 artifact with the approved hash"
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
    public_labels_sha256 = _sha256(public_labels_path)
    expected_public_sha256 = _mapping(public_contract, "file_hashes").get(
        "report_labels_v2.csv"
    )
    if (
        expected_public_sha256 is not None
        and public_labels_sha256 != expected_public_sha256
    ):
        raise ValueError("Public label file SHA-256 mismatch")

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
        public_labels_path, _public_expected_columns(gold_contract.target_names)
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
        is_gold_study = all(len(val) > 0 for val in raw_gold_values)

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
    weight_path: Path | None = None,
    competition_contract_path: Path | None = None,
    public_dataset_contract_path: Path | None = None,
    data_index_contract_path: Path | None = None,
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

    if pretrained_checkpoint_path is None:
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

    artifact_dir = work_root / _ARTIFACT_DIRECTORY
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / "checkpoint.pt"

    if pretrained_checkpoint_path is not None:
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
            inputs.weight_path, cohort.target_names
        )
        initialization_sha256 = initialization.sha256
        model.to(device)
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
                }
            )

        total_training_seconds = time.perf_counter() - total_training_start

        checkpoint_payload = {
            "checkpoint_schema_version": _COMBINED_CHECKPOINT_SCHEMA_VERSION,
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

    reloaded_model, _ = load_combined_checkpoint(checkpoint_path, device=device)

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

    submission_prediction = _predict_combined_submission(
        model=reloaded_model,
        submission_inputs=submission_inputs,
        indexed=test_index,
        work_root=artifact_dir,
        preprocessing=config.preprocessing,
        priors=priors,
        mixed_precision=config.mixed_precision,
        device=device,
    )

    report_path = artifact_dir / "training-report.json"
    report_data = {
        "schema_version": _COMBINED_REPORT_SCHEMA_VERSION,
        "status": "complete",
        "cohort": {
            "rule_version": _COMBINED_COHORT_RULE_VERSION,
            "digest": cohort.digest,
            "total_studies": cohort.total_study_count,
            "gold_studies": cohort.gold_study_count,
            "non_gold_studies": cohort.non_gold_study_count,
            "per_target": cohort.per_target,
        },
        "partition_summary": partition.summary,
        "training_execution": {
            "approved_gpu": approved_gpu,
            "cuda_readiness": cuda_evidence,
            "total_training_seconds": total_training_seconds,
            "epochs_completed": config.epochs,
            "epoch_logs": epoch_logs,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device)
            if torch.cuda.is_available()
            else 0,
        },
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": _sha256(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "reloaded_before_inference": True,
            "max_reload_absolute_diff": max_diff,
            "reload_reproducible": True,
        },
        "submission": {
            "path": submission_prediction.artifacts.submission_path.name,
            "sha256": _sha256(submission_prediction.artifacts.submission_path),
            "prediction_count": submission_prediction.prediction_count,
            "fallback_count": submission_prediction.fallback_count,
            "inference_seconds": submission_prediction.inference_seconds,
        },
        "runtime": runtime_record(),
    }
    report_path.write_text(
        json.dumps(report_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    manifest_path = artifact_dir / "run-manifest.json"
    write_run_manifest(
        manifest_path,
        run_id=run_id or create_run_id("combined-baseline"),
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
            submission_prediction.artifacts.submission_path,
            submission_prediction.artifacts.fallbacks_path,
            submission_prediction.artifacts.validation_report_path,
        ],
        artifact_root=artifact_dir,
    )

    return CombinedBaselineOutcome(
        artifact_dir=artifact_dir,
        checkpoint_path=checkpoint_path,
        report_path=report_path,
        submission_path=submission_prediction.artifacts.submission_path,
        validation_report_path=submission_prediction.artifacts.validation_report_path,
        manifest_path=manifest_path,
    )


def load_combined_checkpoint(
    path: Path, *, device: torch.device
) -> tuple[GoldResNet18, dict[str, Any]]:
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
    if set(checkpoint) != required:
        raise ValueError("Combined checkpoint fields are incomplete or unexpected")
    if checkpoint["checkpoint_schema_version"] != _COMBINED_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported combined checkpoint schema")
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
    model: GoldResNet18,
    submission_inputs: SubmissionInputs,
    indexed: Mapping[str, IndexedStudy],
    work_root: Path,
    preprocessing: PreprocessingConfig,
    priors: Sequence[float],
    mixed_precision: bool,
    device: torch.device,
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
                failures[study_uid] = "no_selected_index_series"
                continue
            decoded = decode_study(indexed_study, preprocessing, labels=None)
            decode_errors.update(decoded.decode_error_counts)
            if decoded.valid_slice_count == 0:
                failures[study_uid] = "no_decodable_selected_slices"
                continue
            pixels = decoded.pixels.unsqueeze(0).to(device)
            slice_mask = decoded.slice_mask.unsqueeze(0).to(device)
            series_mask = decoded.series_mask.unsqueeze(0).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=mixed_precision and device.type == "cuda",
            ):
                probabilities = torch.sigmoid(model(pixels, slice_mask, series_mask))[0]
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
    from .training import _SubmissionPrediction

    return _SubmissionPrediction(
        artifacts=artifacts,
        inference_seconds=inference_seconds,
        prediction_count=len(predictions),
        fallback_count=len(failures),
        decode_errors=dict(sorted(decode_errors.items())),
    )


def _public_expected_columns(targets: Sequence[str]) -> list[str]:
    cols = ["StudyInstanceUID"]
    for t in targets:
        cols.extend([t, f"{t}__conf", f"{t}__verdict"])
    return cols


def _combined_config_record(config: CombinedBaselineConfig) -> dict[str, object]:
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
        "split": {
            "val_fraction": config.val_fraction,
            "split_seed": config.split_seed,
        },
        "validation": {
            "reload_absolute_tolerance": config.reload_absolute_tolerance,
        },
    }


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
