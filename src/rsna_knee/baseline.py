"""Complete leakage-safe organizer-gold holdout baseline for issue #9."""

from __future__ import annotations

import csv
import hashlib
import json
import time
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from .config import PreprocessingConfig
from .holdout import (
    BOOTSTRAP_METHOD_VERSION,
    GoldEvaluationPartition,
    Label,
    PatientBootstrapConfig,
    counts_with_prevalence,
    derive_frozen_evaluation_partition,
    evaluate_holdout_predictions,
    evaluate_point_predictions,
)
from .provenance import create_run_id, write_run_manifest
from .training import (
    GoldTrainingCohort,
    InitializationArtifact,
    PartitionPrediction,
    StudyClassifier,
    TrainingInputs,
    build_checkpoint_payload,
    build_initialized_model,
    derive_gold_training_cohort,
    discover_training_inputs,
    load_indexed_studies,
    predict_partition,
    require_approved_cuda,
    restore_training_state,
    runtime_record,
    save_checkpoint,
    seed_everything,
    train_epoch,
)

_ARTIFACT_DIRECTORY = "rsna-knee-gold-baseline-v1"
_REPORT_SCHEMA_VERSION = 1
_CONFIG_SCHEMA_VERSION = 1
_BEST_CHECKPOINT = "best-checkpoint.pt"
_LAST_CHECKPOINT = "last-checkpoint.pt"
_PREDICTIONS = "holdout-predictions-private.csv"
_REPORT = "baseline-report.json"
_MANIFEST = "run-manifest.json"


@dataclass(frozen=True, slots=True)
class BaselineTrainingConfig:
    preprocessing: PreprocessingConfig
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int
    mixed_precision: bool
    architecture: str
    weight_identity: str
    bootstrap: PatientBootstrapConfig
    reload_absolute_tolerance: float


@dataclass(frozen=True, slots=True)
class BaselineOutcome:
    artifact_dir: Path
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    predictions_path: Path
    report_path: Path
    manifest_path: Path


def load_baseline_training_config(path: Path) -> BaselineTrainingConfig:
    """Load the immutable issue #9 baseline and predeclared bootstrap settings."""
    with path.open("rb") as file:
        raw = tomllib.load(file)
    preprocessing_raw = _mapping(raw, "preprocessing")
    training = _mapping(raw, "training")
    bootstrap = _mapping(raw, "bootstrap")
    validation = _mapping(raw, "validation")
    preprocessing = PreprocessingConfig(
        slice_count=_positive_int(preprocessing_raw, "slice_count"),
        lower_percentile=_number(preprocessing_raw, "lower_percentile"),
        upper_percentile=_number(preprocessing_raw, "upper_percentile"),
        image_size=_positive_int(preprocessing_raw, "image_size"),
    )
    config = BaselineTrainingConfig(
        preprocessing=preprocessing,
        epochs=_positive_int(training, "epochs"),
        batch_size=_positive_int(training, "batch_size"),
        learning_rate=_positive_number(training, "learning_rate"),
        weight_decay=_nonnegative_number(training, "weight_decay"),
        seed=_nonnegative_int(training, "seed"),
        mixed_precision=_boolean(training, "mixed_precision"),
        architecture=_text(training, "architecture"),
        weight_identity=_text(training, "weight_identity"),
        bootstrap=PatientBootstrapConfig(
            method_version=_text(bootstrap, "method_version"),
            seed=_nonnegative_int(bootstrap, "seed"),
            replicate_count=_positive_int(bootstrap, "replicate_count"),
            confidence_level=_number(bootstrap, "confidence_level"),
        ),
        reload_absolute_tolerance=_positive_number(
            validation, "reload_absolute_tolerance"
        ),
    )
    if not (
        0 <= preprocessing.lower_percentile < preprocessing.upper_percentile <= 100
    ):
        raise ValueError("Preprocessing percentiles are invalid")
    if preprocessing.image_size != 224:
        raise ValueError("Issue #9 reuses the 224 by 224 preprocessing contract")
    if config.architecture != "torchvision.models.resnet18":
        raise ValueError("Issue #9 reuses exactly torchvision ResNet-18")
    if config.weight_identity != "ResNet18_Weights.IMAGENET1K_V1":
        raise ValueError("Issue #9 reuses the frozen initialization release")
    if config.bootstrap.method_version != BOOTSTRAP_METHOD_VERSION:
        raise ValueError("Issue #9 requires the shared PatientID bootstrap version")
    if not 0 < config.bootstrap.confidence_level < 1:
        raise ValueError("Bootstrap confidence level must be between zero and one")
    return config


def run_gold_holdout_baseline(
    *,
    input_root: Path,
    work_root: Path,
    config: BaselineTrainingConfig,
    approved_gpu: bool = False,
    run_id: str | None = None,
    source_provenance: Mapping[str, object] | None = None,
    notebook: Mapping[str, object] | None = None,
    index_artifact_dir: Path | None = None,
) -> BaselineOutcome:
    """Consume the prior duration probe and evaluate the complete baseline."""
    cuda_evidence = require_approved_cuda(approved=approved_gpu)
    inputs = discover_training_inputs(input_root, index_artifact_dir=index_artifact_dir)
    if inputs.index_verification == "rebuilt_verified_artifact":
        _verify_rebuilt_index_configuration(inputs.index_artifact_dir, config)
    cohort = derive_gold_training_cohort(
        inputs.competition_root,
        inputs.index_artifact_dir,
        allow_verified_rebuilt_index=(
            inputs.index_verification == "rebuilt_verified_artifact"
        ),
    )
    partition = derive_frozen_evaluation_partition(cohort, inputs.index_artifact_dir)
    train_studies = partition.train
    holdout_studies = partition.holdout
    if not train_studies or not holdout_studies:
        raise ValueError("Frozen evaluation partition must contain train and holdout")
    train_patients = {study.patient_id for study in train_studies}
    holdout_patients = {study.patient_id for study in holdout_studies}
    if train_patients & holdout_patients:
        raise ValueError("PatientID crosses the frozen evaluation partition")

    indexed = load_indexed_studies(
        inputs.competition_root,
        inputs.index_artifact_dir,
        [study.study_uid for study in partition.studies],
        split="train",
    )
    if any(not indexed[study.study_uid].series for study in partition.studies):
        raise ValueError("Frozen partition contains a study with no selected series")
    device = torch.device("cuda")

    duration_probe = _duration_estimate_from_issue_8(
        cohort=cohort,
        partition=partition,
        config=config,
    )

    seed_everything(config.seed)
    torch.cuda.reset_peak_memory_stats(device)
    model, initialization = build_initialized_model(
        inputs.weight_path, cohort.target_names
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.GradScaler("cuda", enabled=config.mixed_precision)
    train_counts = cast(
        dict[str, dict[str, int]], partition.per_partition["train"]["per_target"]
    )
    priors = tuple(
        train_counts[target]["positive"] / train_counts[target]["known"]
        for target in cohort.target_names
    )

    artifact_dir = work_root / _ARTIFACT_DIRECTORY
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(f"Baseline artifact already exists: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = artifact_dir / _BEST_CHECKPOINT
    last_checkpoint_path = artifact_dir / _LAST_CHECKPOINT
    predictions_path = artifact_dir / _PREDICTIONS
    report_path = artifact_dir / _REPORT
    manifest_path = artifact_dir / _MANIFEST

    epoch_records: list[dict[str, Any]] = []
    total_training_start = time.perf_counter()
    best_metric = float("-inf")
    best_epoch = 0
    best_predictions: PartitionPrediction | None = None
    best_point: dict[str, Any] | None = None
    for epoch in range(1, config.epochs + 1):
        train_record = train_epoch(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            indexed=indexed,
            studies=train_studies,
            preprocessing=config.preprocessing,
            batch_size=config.batch_size,
            seed=config.seed,
            mixed_precision=config.mixed_precision,
            device=device,
            epoch=epoch,
        )
        validation = predict_partition(
            model=model,
            indexed=indexed,
            studies=holdout_studies,
            preprocessing=config.preprocessing,
            mixed_precision=config.mixed_precision,
            device=device,
        )
        point = evaluate_point_predictions(
            labels=[study.labels for study in validation.studies],
            predictions=validation.probabilities,
            target_names=cohort.target_names,
        )
        macro = point["official_macro_auc"]
        if not isinstance(macro, float):
            raise TypeError("Frozen holdout official macro AUC is undefined")
        checkpoint = _baseline_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            completed_epochs=epoch,
            cohort=cohort,
            partition=partition,
            initialization=initialization,
            config=config,
            priors=priors,
            source_provenance=source_provenance or {},
            inputs=inputs,
            validation_metric=macro,
        )
        save_checkpoint(last_checkpoint_path, checkpoint)
        if macro > best_metric:
            best_metric = macro
            best_epoch = epoch
            best_predictions = validation
            best_point = point
            save_checkpoint(best_checkpoint_path, checkpoint)
        epoch_records.append(
            {
                "epoch": epoch,
                **train_record,
                "validation": {
                    "official_macro_auc": macro,
                    "decode_seconds": validation.decode_seconds,
                    "forward_seconds": validation.forward_seconds,
                    "valid_slice_count": validation.valid_slice_count,
                    "decode_errors": validation.decode_errors,
                },
            }
        )
    torch.cuda.synchronize()
    complete_run_seconds = time.perf_counter() - total_training_start
    if best_predictions is None or best_point is None:
        raise RuntimeError("Complete baseline did not produce a best checkpoint")
    for record in epoch_records:
        record["is_best"] = record["epoch"] == best_epoch

    del model
    torch.cuda.empty_cache()
    reloaded, reloaded_optimizer, reloaded_scaler, best_checkpoint = (
        restore_training_state(best_checkpoint_path, device=device)
    )
    if best_checkpoint["completed_epochs"] != best_epoch:
        raise ValueError("Best checkpoint epoch does not match selected epoch")
    if (
        best_checkpoint["cohort"].get("evaluation_assignment_digest")
        != partition.assignment_digest
    ):
        raise ValueError("Reloaded best checkpoint split digest mismatch")
    del reloaded_optimizer, reloaded_scaler
    reloaded_predictions = predict_partition(
        model=reloaded,
        indexed=indexed,
        studies=holdout_studies,
        preprocessing=config.preprocessing,
        mixed_precision=config.mixed_precision,
        device=device,
    )
    maximum_reload_difference = _maximum_prediction_difference(
        best_predictions.probabilities, reloaded_predictions.probabilities
    )
    if maximum_reload_difference > config.reload_absolute_tolerance:
        raise ValueError("Reloaded validation predictions exceed numerical tolerance")
    _, last_optimizer, last_scaler, last_checkpoint = restore_training_state(
        last_checkpoint_path, device=device
    )
    del last_optimizer, last_scaler
    if last_checkpoint["completed_epochs"] != config.epochs:
        raise ValueError("Last checkpoint is not resumable from the final epoch")

    evaluation = evaluate_holdout_predictions(
        patient_ids=[study.patient_id for study in reloaded_predictions.studies],
        labels=[study.labels for study in reloaded_predictions.studies],
        predictions=reloaded_predictions.probabilities,
        target_names=cohort.target_names,
        bootstrap=config.bootstrap,
    )
    if evaluation["metric"]["point_estimate"] != best_point:
        raise ValueError("Reloaded holdout metric differs from best-epoch metric")
    _write_private_predictions(
        predictions_path, reloaded_predictions, cohort.target_names
    )

    epoch_training_seconds = sum(
        float(record["training_seconds"]) for record in epoch_records
    )
    epoch_decode_seconds = sum(
        float(record["decode_seconds"]) for record in epoch_records
    )
    epoch_forward_backward_seconds = sum(
        float(record["forward_backward_seconds"]) for record in epoch_records
    )
    valid_training_slices = sum(
        int(record["valid_slice_count"]) for record in epoch_records
    )
    training_decode_errors: Counter[str] = Counter()
    for record in epoch_records:
        training_decode_errors.update(cast(dict[str, int], record["decode_errors"]))

    report = {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "artifact_type": "rsna_knee_gold_holdout_baseline",
        "scope": (
            "Gold-only frozen-holdout evidence for the limited issue #11 one-change "
            "decision; no claim of general statistical sufficiency or competitive "
            "performance."
        ),
        "supervision": {
            "source": "organizer gold train.csv only",
            "report_derived_labels_used": False,
            "missing_labels_masked_in_loss": True,
            "target_order": list(cohort.target_names),
            "canonical_cohort_digest": cohort.digest,
        },
        "split": {
            "grouping_entity": "PatientID",
            "report_hash_grouping_applied": False,
            "allocation_rule_version": partition.allocation_rule_version,
            "allocation_seed": partition.allocation_seed,
            "assignment_digest": partition.assignment_digest,
            "independently_rederived_before_training": True,
            "digest_verified_before_training": True,
            "patient_overlap_count": 0,
            "partitions": {
                name: {
                    "study_count": partition.per_partition[name]["study_count"],
                    "patient_count": partition.per_partition[name]["patient_count"],
                    "per_target": counts_with_prevalence(
                        cast(
                            dict[str, dict[str, int]],
                            partition.per_partition[name]["per_target"],
                        ),
                        cohort.target_names,
                    ),
                }
                for name in ("train", "holdout")
            },
        },
        "model": {
            "architecture": config.architecture,
            "image_size": config.preprocessing.image_size,
            "slice_to_series_pooling": "masked mean",
            "series_to_study_pooling": "masked mean",
            "initialization_identity": config.weight_identity,
            "initialization_artifact_sha256": initialization.sha256,
            "initialization_validated_before_model_construction": True,
            "network_fallback": False,
        },
        "configuration": _configuration_record(config),
        "duration_probe": duration_probe,
        "execution": {
            "approved_gpu": True,
            "cuda_readiness": cuda_evidence,
            "gpu": cuda_evidence["device_name"],
            "epochs_completed": config.epochs,
            "batch_size": config.batch_size,
            "mixed_precision": config.mixed_precision,
            "complete_run_seconds": complete_run_seconds,
            "epoch_training_seconds": epoch_training_seconds,
            "training_decode_seconds": epoch_decode_seconds,
            "training_forward_backward_seconds": epoch_forward_backward_seconds,
            "valid_training_slices": valid_training_slices,
            "training_slices_per_second": (
                valid_training_slices / epoch_training_seconds
                if epoch_training_seconds
                else None
            ),
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
            "training_decode_errors": dict(sorted(training_decode_errors.items())),
            "best_reload_validation_decode_errors": (
                reloaded_predictions.decode_errors
            ),
            "epoch_records": epoch_records,
        },
        "evaluation": evaluation,
        "checkpoints": {
            "retention_policy": "best and last resumable checkpoints only",
            "retained_count": 2,
            "best": {
                "path": best_checkpoint_path.name,
                "epoch": best_epoch,
                "sha256": _sha256(best_checkpoint_path),
                "size_bytes": best_checkpoint_path.stat().st_size,
                "selection_metric": "official_macro_auc",
                "selection_value": best_metric,
            },
            "last": {
                "path": last_checkpoint_path.name,
                "epoch": config.epochs,
                "sha256": _sha256(last_checkpoint_path),
                "size_bytes": last_checkpoint_path.stat().st_size,
            },
            "best_and_last_resumable": True,
            "best_reloaded_for_reported_predictions": True,
            "reload_prediction_absolute_tolerance": (config.reload_absolute_tolerance),
            "reload_prediction_maximum_absolute_difference": (
                maximum_reload_difference
            ),
            "reload_predictions_within_tolerance": True,
        },
        "data_loading": {
            "training_decode_seconds": epoch_decode_seconds,
            "training_forward_backward_seconds": epoch_forward_backward_seconds,
            "decode_share_of_measured_epoch_training": (
                epoch_decode_seconds / epoch_training_seconds
                if epoch_training_seconds
                else None
            ),
            "validation_decode_seconds_after_best_reload": (
                reloaded_predictions.decode_seconds
            ),
            "validation_forward_seconds_after_best_reload": (
                reloaded_predictions.forward_seconds
            ),
            "cache_or_embedding_recommendation": {
                "decision": "not_recommended",
                "rationale": (
                    "This single run measures decoding but does not measure reuse "
                    "benefit or the cost and size of a cache; no material benefit is "
                    "therefore established. This baseline creates neither decoded-"
                    "pixel nor embedding caches."
                ),
            },
        },
        "output": {
            "predictions_path": predictions_path.name,
            "predictions_sha256": _sha256(predictions_path),
            "predictions_size_bytes": predictions_path.stat().st_size,
            "retained_checkpoint_and_prediction_bytes": sum(
                path.stat().st_size
                for path in (
                    best_checkpoint_path,
                    last_checkpoint_path,
                    predictions_path,
                )
            ),
        },
        "adequacy_for_issue_11": _adequacy_decision(partition, evaluation),
        "runtime": runtime_record(),
        "exclusions": {
            "leaderboard_evidence_used": False,
            "report_derived_labels_used": False,
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    accepted_index_manifest_sha256 = _text(
        _mapping(
            _mapping(_load_json_contract("data-index.json")[0], "run_provenance"),
            "outputs",
        ),
        "manifest_sha256",
    )
    write_run_manifest(
        manifest_path,
        run_id=run_id or create_run_id("gold-holdout-baseline"),
        source={
            **dict(source_provenance or {}),
            "competition_contract_sha256": cohort.gold_contract_sha256,
            "gold_supervision_contract_sha256": (
                cohort.gold_supervision_contract_sha256
            ),
            "gold_training_contract_sha256": duration_probe["source_contract_sha256"],
            "index_manifest_sha256": _sha256(
                inputs.index_artifact_dir / "index-manifest.json"
            ),
            "index_verification": inputs.index_verification,
            "accepted_index_manifest_sha256": accepted_index_manifest_sha256,
            "train_csv_sha256": _sha256(inputs.competition_root / "train.csv"),
            "initialization_artifact_sha256": initialization.sha256,
            "evaluation_assignment_digest": partition.assignment_digest,
        },
        configuration=_configuration_record(config),
        notebook=notebook or {"kind": "approved_kaggle_gpu_gold_holdout_baseline"},
        artifacts=[
            best_checkpoint_path,
            last_checkpoint_path,
            predictions_path,
            report_path,
        ],
        artifact_root=artifact_dir,
    )
    return BaselineOutcome(
        artifact_dir=artifact_dir,
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
        predictions_path=predictions_path,
        report_path=report_path,
        manifest_path=manifest_path,
    )


def verify_baseline_artifact(artifact_dir: Path) -> dict[str, Any]:
    """Verify issue #9 output bytes, metrics, split, checkpoint, and decisions."""
    report_path = artifact_dir / _REPORT
    manifest_path = artifact_dir / _MANIFEST
    predictions_path = artifact_dir / _PREDICTIONS
    for path in (report_path, manifest_path, predictions_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing baseline artifact: {path}")
    report = cast(dict[str, Any], json.loads(report_path.read_text()))
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text()))
    if (
        report.get("schema_version") != _REPORT_SCHEMA_VERSION
        or report.get("artifact_type") != "rsna_knee_gold_holdout_baseline"
    ):
        raise ValueError("Baseline report identity is invalid")
    if manifest.get("schema_version") != 1:
        raise ValueError("Baseline manifest schema is invalid")
    manifest_source = _mapping(manifest, "source")
    required_source_hashes = (
        "jupytext_source_sha256",
        "package_archive_sha256",
        "baseline_config_sha256",
        "competition_contract_sha256",
        "gold_supervision_contract_sha256",
        "gold_training_contract_sha256",
        "index_manifest_sha256",
        "train_csv_sha256",
        "initialization_artifact_sha256",
        "evaluation_assignment_digest",
    )
    if any(not _is_sha256(manifest_source.get(key)) for key in required_source_hashes):
        raise ValueError("Baseline manifest input or source hashes are incomplete")
    _verify_frozen_sources(manifest_source, report)
    if manifest.get("configuration") != report.get("configuration"):
        raise ValueError("Baseline manifest configuration differs from report")
    notebook = _mapping(manifest, "notebook")
    if (
        notebook.get("kind")
        not in {
            "private_approved_kaggle_gpu_gold_holdout_baseline",
            "private_review_kaggle_gpu_publication_reproduction",
        }
        or notebook.get("internet_enabled") is not False
        or not isinstance(notebook.get("kernel_id"), str)
    ):
        raise ValueError("Baseline notebook provenance is incomplete")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise TypeError("Baseline manifest artifact inventory is malformed")
    expected_paths = {
        _BEST_CHECKPOINT,
        _LAST_CHECKPOINT,
        _PREDICTIONS,
        _REPORT,
    }
    seen: set[str] = set()
    for raw in artifacts:
        if not isinstance(raw, dict):
            raise TypeError("Baseline manifest artifact record is malformed")
        relative = raw.get("path")
        if not isinstance(relative, str) or relative in seen:
            raise ValueError("Baseline manifest artifact path is invalid")
        path = (artifact_dir / relative).resolve()
        path.relative_to(artifact_dir.resolve())
        if (
            not path.is_file()
            or raw.get("sha256") != _sha256(path)
            or raw.get("size_bytes") != path.stat().st_size
        ):
            raise ValueError("Baseline manifest artifact bytes do not match")
        seen.add(relative)
    if seen != expected_paths:
        raise ValueError("Baseline manifest artifact inventory is incomplete")
    retained_checkpoints = {path.name for path in artifact_dir.glob("*.pt")}
    if retained_checkpoints != {_BEST_CHECKPOINT, _LAST_CHECKPOINT}:
        raise ValueError("Only best and last checkpoints may be retained")

    supervision = _mapping(report, "supervision")
    split = _mapping(report, "split")
    if (
        supervision.get("report_derived_labels_used") is not False
        or supervision.get("missing_labels_masked_in_loss") is not True
        or split.get("grouping_entity") != "PatientID"
        or split.get("report_hash_grouping_applied") is not False
        or split.get("independently_rederived_before_training") is not True
        or split.get("digest_verified_before_training") is not True
        or split.get("patient_overlap_count") != 0
    ):
        raise ValueError("Baseline supervision or split contract is invalid")
    targets = supervision.get("target_order")
    if not isinstance(targets, list) or not all(
        isinstance(target, str) and target for target in targets
    ):
        raise TypeError("Baseline target order is malformed")
    typed_targets = cast(list[str], targets)
    partitions = _mapping(split, "partitions")
    for name in ("train", "holdout"):
        partition_record = _mapping(partitions, name)
        per_target = _mapping(partition_record, "per_target")
        for target in typed_targets:
            counts = _mapping(per_target, target)
            if counts.get("positive", 0) < 1 or counts.get("negative", 0) < 1:
                raise ValueError("Baseline partition lacks both target classes")

    checkpoint_record = _mapping(report, "checkpoints")
    if (
        checkpoint_record.get("retained_count") != 2
        or checkpoint_record.get("best_and_last_resumable") is not True
        or checkpoint_record.get("best_reloaded_for_reported_predictions") is not True
        or checkpoint_record.get("reload_predictions_within_tolerance") is not True
    ):
        raise ValueError("Baseline checkpoint or reload evidence is incomplete")
    for name in ("best", "last"):
        record = _mapping(checkpoint_record, name)
        path = artifact_dir / _text(record, "path")
        if (
            record.get("sha256") != _sha256(path)
            or record.get("size_bytes") != path.stat().st_size
        ):
            raise ValueError("Baseline checkpoint record differs from actual bytes")
        _, optimizer, scaler, checkpoint = restore_training_state(
            path, device=torch.device("cpu")
        )
        del optimizer, scaler
        if checkpoint["cohort"].get("evaluation_assignment_digest") != split.get(
            "assignment_digest"
        ):
            raise ValueError("Baseline checkpoint split digest differs from report")
        if checkpoint.get("configuration") != report.get("configuration"):
            raise ValueError("Baseline checkpoint configuration differs from report")
        if checkpoint.get("runtime") != report.get("runtime"):
            raise ValueError("Baseline checkpoint runtime differs from report")
        if _mapping(checkpoint, "artifact_hashes").get(
            "index_manifest_sha256"
        ) != manifest_source.get("index_manifest_sha256"):
            raise ValueError(
                "Baseline checkpoint index provenance differs from manifest"
            )
        code = _mapping(checkpoint, "code")
        if any(
            code.get(key) != manifest_source.get(key)
            for key in (
                "git_commit",
                "git_dirty",
                "jupytext_source_sha256",
                "package_archive_sha256",
                "baseline_config_sha256",
                "weight_dataset_id",
            )
        ):
            raise ValueError(
                "Baseline checkpoint code provenance differs from manifest"
            )

    patient_ids, labels, predictions = _read_private_predictions(
        predictions_path, typed_targets
    )
    configuration = _mapping(report, "configuration")
    bootstrap_raw = _mapping(configuration, "bootstrap")
    recalculated = evaluate_holdout_predictions(
        patient_ids=patient_ids,
        labels=labels,
        predictions=predictions,
        target_names=typed_targets,
        bootstrap=PatientBootstrapConfig(
            method_version=_text(bootstrap_raw, "method_version"),
            seed=_nonnegative_int(bootstrap_raw, "seed"),
            replicate_count=_positive_int(bootstrap_raw, "replicate_count"),
            confidence_level=_number(bootstrap_raw, "confidence_level"),
        ),
    )
    if report.get("evaluation") != recalculated:
        raise ValueError("Baseline metrics or uncertainty differ from predictions")
    bootstrap = _mapping(recalculated, "bootstrap")
    uncertainty = _mapping(bootstrap, "uncertainty")
    macro = _mapping(uncertainty, "official_macro_auc")
    if macro.get("valid_replicate_count", 0) + macro.get(
        "undefined_replicate_count", 0
    ) != bootstrap.get("replicate_count"):
        raise ValueError("Baseline macro bootstrap accounting is incomplete")
    for target in typed_targets:
        target_uncertainty = _mapping(_mapping(uncertainty, "per_target"), target)
        if target_uncertainty.get("valid_replicate_count", 0) + target_uncertainty.get(
            "undefined_replicate_count", 0
        ) != bootstrap.get("replicate_count"):
            raise ValueError("Baseline target bootstrap accounting is incomplete")
    adequacy = _mapping(report, "adequacy_for_issue_11")
    if adequacy.get("decision") not in {"adequate", "inadequate"} or not isinstance(
        adequacy.get("rationale"), str
    ):
        raise ValueError("Issue #11 adequacy decision is not explicit")
    exclusions = _mapping(report, "exclusions")
    if exclusions != {
        "leaderboard_evidence_used": False,
        "report_derived_labels_used": False,
    }:
        raise ValueError("Baseline evidence uses a prohibited source")
    data_loading = _mapping(report, "data_loading")
    cache_decision = _mapping(data_loading, "cache_or_embedding_recommendation")
    if cache_decision.get("decision") not in {
        "recommended",
        "not_recommended",
    } or not isinstance(cache_decision.get("rationale"), str):
        raise ValueError("Baseline cache recommendation is not evidence-addressed")
    return {
        "valid": True,
        "run_id": manifest.get("run_id"),
        "assignment_digest": split.get("assignment_digest"),
        "best_checkpoint_sha256": _sha256(artifact_dir / _BEST_CHECKPOINT),
        "last_checkpoint_sha256": _sha256(artifact_dir / _LAST_CHECKPOINT),
        "official_macro_auc": recalculated["metric"]["point_estimate"][
            "official_macro_auc"
        ],
        "macro_valid_bootstrap_replicates": macro.get("valid_replicate_count"),
        "macro_undefined_bootstrap_replicates": macro.get("undefined_replicate_count"),
        "adequacy_for_issue_11": adequacy.get("decision"),
    }


def _verify_frozen_sources(
    manifest_source: Mapping[str, Any], report: Mapping[str, Any]
) -> None:
    competition, competition_sha256 = _load_json_contract("competition.json")
    supervision, supervision_sha256 = _load_json_contract("gold-supervision.json")
    prior_training, prior_training_sha256 = _load_json_contract("gold-training.json")
    initialization, _ = _load_json_contract("resnet18-imagenet1k-v1.json")
    data_index, _ = _load_json_contract("data-index.json")
    expected_index_manifest = _text(
        _mapping(_mapping(data_index, "run_provenance"), "outputs"),
        "manifest_sha256",
    )
    index_verification = manifest_source.get(
        "index_verification", "accepted_frozen_artifact"
    )
    index_matches_contract = (
        index_verification == "accepted_frozen_artifact"
        and manifest_source.get("index_manifest_sha256") == expected_index_manifest
    ) or (
        index_verification == "rebuilt_verified_artifact"
        and _is_sha256(manifest_source.get("index_manifest_sha256"))
        and manifest_source.get("accepted_index_manifest_sha256")
        == expected_index_manifest
    )
    target_order = _mapping(competition, "targets").get("order")
    evaluation = _mapping(supervision, "evaluation_eligibility")
    assignment_digest = _mapping(evaluation, "assignment_digest").get("value")
    initialization_artifact = _mapping(initialization, "artifact")
    if (
        manifest_source.get("competition_contract_sha256") != competition_sha256
        or manifest_source.get("gold_supervision_contract_sha256") != supervision_sha256
        or manifest_source.get("gold_training_contract_sha256") != prior_training_sha256
        or not index_matches_contract
        or manifest_source.get("initialization_artifact_sha256")
        != initialization_artifact.get("sha256")
        or manifest_source.get("evaluation_assignment_digest") != assignment_digest
        or _mapping(report, "supervision").get("target_order") != target_order
        or _mapping(report, "split").get("assignment_digest") != assignment_digest
    ):
        raise ValueError("Baseline sources differ from the current frozen contracts")
    duration_probe = _mapping(report, "duration_probe")
    prior_execution = _mapping(prior_training, "execution")
    prior_kernel = _mapping(prior_execution, "kernel")
    prior_training_record = _mapping(prior_execution, "training")
    if (
        duration_probe.get("source_contract_sha256") != prior_training_sha256
        or duration_probe.get("approved_gpu") is not True
        or duration_probe.get("reviewed_before_predictions") is not True
        or duration_probe.get("kernel_id") != prior_kernel.get("id")
        or duration_probe.get("kernel_version")
        != prior_kernel.get("successful_version")
        or duration_probe.get("source_epoch_duration_seconds")
        != prior_training_record.get("training_seconds")
    ):
        raise ValueError("Baseline duration estimate differs from approved #8 evidence")


def _verify_rebuilt_index_configuration(
    artifact_dir: Path, config: BaselineTrainingConfig
) -> None:
    manifest = cast(
        dict[str, Any],
        json.loads((artifact_dir / "index-manifest.json").read_text()),
    )
    preprocessing = _mapping(manifest, "preprocessing")
    expected = {
        **asdict(config.preprocessing),
        "normalization_status": "provisional",
        "resize": "bilinear_aspect_preserving_center_pad",
    }
    if preprocessing != expected:
        raise ValueError("Rebuilt index preprocessing differs from baseline contract")


def export_public_baseline_contract(
    artifact_dir: Path, destination: Path
) -> dict[str, Any]:
    """Export public-safe aggregate issue #9 evidence without private rows."""
    verified = verify_baseline_artifact(artifact_dir)
    report = cast(dict[str, Any], json.loads((artifact_dir / _REPORT).read_text()))
    manifest = cast(dict[str, Any], json.loads((artifact_dir / _MANIFEST).read_text()))
    contract = {
        "schema_version": 1,
        "artifact_type": "rsna_knee_gold_holdout_baseline",
        "scope": report["scope"],
        "source_contracts": {
            key: value
            for key, value in _mapping(manifest, "source").items()
            if key
            in {
                "competition_contract_sha256",
                "gold_supervision_contract_sha256",
                "gold_training_contract_sha256",
                "index_manifest_sha256",
                "initialization_artifact_sha256",
                "evaluation_assignment_digest",
            }
        },
        "supervision": report["supervision"],
        "split": report["split"],
        "model": report["model"],
        "configuration": report["configuration"],
        "duration_probe": report["duration_probe"],
        "execution": {
            key: value
            for key, value in _mapping(report, "execution").items()
            if key != "epoch_records"
        },
        "evaluation": report["evaluation"],
        "checkpoints": report["checkpoints"],
        "data_loading": report["data_loading"],
        "output": {
            **_mapping(report, "output"),
            "registered_artifact_size_bytes": sum(
                path.stat().st_size for path in artifact_dir.iterdir() if path.is_file()
            ),
        },
        "adequacy_for_issue_11": report["adequacy_for_issue_11"],
        "runtime": report["runtime"],
        "exclusions": report["exclusions"],
        "provenance": {
            "run_id": manifest["run_id"],
            "git_commit": _mapping(manifest, "source").get("git_commit"),
            "source_worktree_dirty": _mapping(manifest, "source").get("git_dirty"),
            "jupytext_source_sha256": _mapping(manifest, "source").get(
                "jupytext_source_sha256"
            ),
            "package_archive_sha256": _mapping(manifest, "source").get(
                "package_archive_sha256"
            ),
            "baseline_config_sha256": _mapping(manifest, "source").get(
                "baseline_config_sha256"
            ),
            "artifact_hashes": {
                record["path"]: record["sha256"]
                for record in cast(list[dict[str, Any]], manifest["artifacts"])
            },
        },
        "privacy": {
            "competition_data_downloaded": False,
            "row_level_membership_labels_patient_ids_or_predictions_committed": False,
            "private_artifacts_registered_under_ignored_path": True,
        },
        "verification": verified,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return contract


def _duration_estimate_from_issue_8(
    *,
    cohort: GoldTrainingCohort,
    partition: GoldEvaluationPartition,
    config: BaselineTrainingConfig,
) -> dict[str, Any]:
    contract, contract_sha256 = _load_json_contract("gold-training.json")
    if contract.get("artifact_type") != "rsna_knee_gold_training_smoke":
        raise ValueError("#8 duration source has an invalid artifact type")
    supervision = _mapping(contract, "supervision")
    model = _mapping(contract, "model")
    initialization = _mapping(contract, "offline_initialization")
    execution = _mapping(contract, "execution")
    training = _mapping(execution, "training")
    kernel = _mapping(execution, "kernel")
    if (
        supervision.get("report_derived_labels_used") is not False
        or supervision.get("membership_label_digest") != cohort.digest
        or supervision.get("target_order") != list(cohort.target_names)
        or model.get("architecture") != config.architecture
        or model.get("image_size") != config.preprocessing.image_size
        or initialization.get("identity") != config.weight_identity
        or initialization.get("validated_before_model_construction") is not True
        or execution.get("approved_gpu") is not True
        or training.get("epochs_completed") != 1
        or training.get("attempted_cohort_studies") != len(cohort.studies)
    ):
        raise ValueError("#8 duration source is not comparable to issue #9")
    training_seconds = _positive_number(training, "training_seconds")
    source_studies = _positive_int(training, "attempted_cohort_studies")
    train_studies = _positive_int(partition.per_partition["train"], "study_count")
    return {
        "approved_gpu": True,
        "kind": "approved one-epoch smoke run completed before the issue #9 run",
        "source_issue": 8,
        "source_contract": "gold-training.json",
        "source_contract_sha256": contract_sha256,
        "kernel_id": kernel.get("id"),
        "kernel_version": kernel.get("successful_version"),
        "gpu": kernel.get("machine_shape"),
        "source_epoch_study_count": source_studies,
        "source_epoch_duration_seconds": training_seconds,
        "source_training_slices_per_second": training.get("training_slices_per_second"),
        "source_peak_cuda_memory_bytes": training.get("peak_cuda_memory_bytes"),
        "target_train_study_count": train_studies,
        "estimated_complete_epoch_seconds": (
            training_seconds * train_studies / source_studies
        ),
        "estimate_method": (
            "linear study-count scaling of the comparable #8 one-epoch smoke run"
        ),
        "reviewed_before_predictions": True,
    }


def _baseline_checkpoint_payload(
    *,
    model: StudyClassifier,
    optimizer: torch.optim.Optimizer,
    scaler: torch.GradScaler,
    completed_epochs: int,
    cohort: GoldTrainingCohort,
    partition: GoldEvaluationPartition,
    initialization: InitializationArtifact,
    config: BaselineTrainingConfig,
    priors: Sequence[float],
    source_provenance: Mapping[str, object],
    inputs: TrainingInputs,
    validation_metric: float,
) -> dict[str, object]:
    payload = build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        completed_epochs=completed_epochs,
        target_names=cohort.target_names,
        priors=priors,
        preprocessing=config.preprocessing,
        cohort=cohort,
        initialization=initialization,
        configuration=_configuration_record(config),
        source_provenance=source_provenance,
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
    payload["configuration"] = _configuration_record(config)
    checkpoint_cohort = cast(dict[str, object], payload["cohort"])
    checkpoint_cohort.update(
        {
            "evaluation_assignment_digest": partition.assignment_digest,
            "allocation_rule_version": partition.allocation_rule_version,
            "allocation_seed": partition.allocation_seed,
            "train_patient_count": partition.per_partition["train"]["patient_count"],
            "holdout_patient_count": partition.per_partition["holdout"][
                "patient_count"
            ],
            "validation_metric": {
                "name": "official_macro_auc",
                "value": validation_metric,
            },
        }
    )
    return payload


def _write_private_predictions(
    path: Path, predictions: PartitionPrediction, targets: Sequence[str]
) -> None:
    columns = [
        "StudyInstanceUID",
        "PatientID",
        *(f"label::{target}" for target in targets),
        *(f"prediction::{target}" for target in targets),
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for study, values in zip(
            predictions.studies, predictions.probabilities, strict=True
        ):
            row: dict[str, object] = {
                "StudyInstanceUID": study.study_uid,
                "PatientID": study.patient_id,
            }
            row.update(
                {
                    f"label::{target}": "" if label is None else label
                    for target, label in zip(targets, study.labels, strict=True)
                }
            )
            row.update(
                {
                    f"prediction::{target}": value
                    for target, value in zip(targets, values, strict=True)
                }
            )
            writer.writerow(row)


def _read_private_predictions(
    path: Path, targets: Sequence[str]
) -> tuple[list[str], list[tuple[Label, ...]], list[tuple[float, ...]]]:
    columns = [
        "StudyInstanceUID",
        "PatientID",
        *(f"label::{target}" for target in targets),
        *(f"prediction::{target}" for target in targets),
    ]
    patient_ids: list[str] = []
    labels: list[tuple[Label, ...]] = []
    predictions: list[tuple[float, ...]] = []
    seen_studies: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != columns:
            raise ValueError("Private holdout prediction columns are invalid")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Private holdout prediction row is malformed")
            study_uid = row["StudyInstanceUID"]
            patient_id = row["PatientID"]
            if not study_uid or not patient_id or study_uid in seen_studies:
                raise ValueError("Private holdout prediction identifiers are invalid")
            seen_studies.add(study_uid)
            patient_ids.append(patient_id)
            labels.append(
                tuple(
                    None
                    if not row[f"label::{target}"]
                    else cast(Label, int(row[f"label::{target}"]))
                    for target in targets
                )
            )
            predictions.append(
                tuple(float(row[f"prediction::{target}"]) for target in targets)
            )
    if not patient_ids:
        raise ValueError("Private holdout predictions are empty")
    return patient_ids, labels, predictions


def _maximum_prediction_difference(
    first: Sequence[Sequence[float]], second: Sequence[Sequence[float]]
) -> float:
    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    if first_values.shape != second_values.shape or not first_values.size:
        raise ValueError("Reloaded validation prediction shape differs")
    return float(np.abs(first_values - second_values).max())


def _adequacy_decision(
    partition: GoldEvaluationPartition, evaluation: Mapping[str, Any]
) -> dict[str, object]:
    holdout = partition.per_partition["holdout"]
    bootstrap = _mapping(evaluation, "bootstrap")
    uncertainty = _mapping(bootstrap, "uncertainty")
    macro = _mapping(uncertainty, "official_macro_auc")
    undefined_rate = float(macro["undefined_replicate_rate"])
    interval = macro.get("interval")
    return {
        "decision": "inadequate",
        "purpose": "motivate and evaluate exactly one model change in issue #11",
        "rationale": (
            f"The frozen holdout contains only {holdout['patient_count']} PatientIDs "
            f"and {holdout['study_count']} studies; several targets have only one "
            "example of a class. Under the predeclared PatientID bootstrap, "
            f"{undefined_rate:.1%} of macro-AUC replicates are undefined and the "
            f"defined-replicate percentile interval is {interval}. This evidence is "
            "too fragile to attribute a measured failure to the baseline or to "
            "evaluate one model change. The decision is limited to issue #11 and "
            "does not assert a general support or interval-width threshold."
        ),
        "leaderboard_evidence_used": False,
        "claims_general_statistical_sufficiency": False,
        "uses_support_or_interval_threshold": False,
        "roadmap_effect": "issue #11 must stop for explicit roadmap realignment",
    }


def _configuration_record(config: BaselineTrainingConfig) -> dict[str, object]:
    return {
        "schema_version": _CONFIG_SCHEMA_VERSION,
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
        "bootstrap": asdict(config.bootstrap),
        "validation": {"reload_absolute_tolerance": config.reload_absolute_tolerance},
    }


def _mapping(parent: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Configuration field {key} must be an object")
    return cast(dict[str, Any], value)


def _text(parent: Mapping[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"Configuration field {key} must be a nonempty string")
    return value


def _positive_int(parent: Mapping[str, Any], key: str) -> int:
    value = parent.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TypeError(f"Configuration field {key} must be a positive integer")
    return value


def _nonnegative_int(parent: Mapping[str, Any], key: str) -> int:
    value = parent.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TypeError(f"Configuration field {key} must be a nonnegative integer")
    return value


def _number(parent: Mapping[str, Any], key: str) -> float:
    value = parent.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"Configuration field {key} must be numeric")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"Configuration field {key} must be finite")
    return number


def _positive_number(parent: Mapping[str, Any], key: str) -> float:
    value = _number(parent, key)
    if value <= 0:
        raise ValueError(f"Configuration field {key} must be positive")
    return value


def _nonnegative_number(parent: Mapping[str, Any], key: str) -> float:
    value = _number(parent, key)
    if value < 0:
        raise ValueError(f"Configuration field {key} must be nonnegative")
    return value


def _boolean(parent: Mapping[str, Any], key: str) -> bool:
    value = parent.get(key)
    if not isinstance(value, bool):
        raise TypeError(f"Configuration field {key} must be boolean")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_contract(name: str) -> tuple[dict[str, Any], str]:
    source_path = Path(__file__).resolve().parents[2] / "contracts" / name
    raw = (
        source_path.read_bytes()
        if source_path.is_file()
        else files("rsna_knee").joinpath(name).read_bytes()
    )
    return cast(dict[str, Any], json.loads(raw)), hashlib.sha256(raw).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
