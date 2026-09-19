"""Bounded resource measurements for the resolution-only #35 experiment."""

import gc
import json
import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch

from .combined_supervision import (
    PHYSICAL_CROP_CHECKPOINT_SHA256,
    CombinedBaselineConfig,
    _combined_config_record,
    _expected_resolution_configuration,
    _validation_membership_digest,
    derive_combined_cohort,
    derive_patient_split,
    discover_combined_checkpoint,
    discover_combined_training_inputs,
    load_physical_crop_validation_predictions,
    load_verified_physical_crop_checkpoint,
)
from .training import (
    IndexedStudy,
    PartitionStudy,
    build_initialized_model,
    load_indexed_studies,
    predict_partition,
    require_approved_cuda,
    seed_everything,
    train_epoch,
)

SMOKE_SCHEMA = "rsna-knee-resolution-resource-smoke-v1"
VALIDATION_DIGEST = "8b1240fc2456779ba64fa115b23956adb131fb387f01c0f0424a933a2096ee88"
RUNTIME_LIMIT_SECONDS = 9 * 3600


def project_resolution_runtime(
    *,
    training_seconds: float,
    training_studies: int,
    training_anchors: int,
    validation_seconds: float,
    validation_studies: int,
    validation_anchors: int,
    total_training_anchors: int,
    total_validation_anchors: int,
    setup_seconds: float,
) -> dict[str, Any]:
    """Use the slower of study- and anchor-scaled estimates, with explicit margin."""
    positive = (
        training_seconds,
        training_studies,
        training_anchors,
        validation_seconds,
        validation_studies,
        validation_anchors,
        total_training_anchors,
        total_validation_anchors,
    )
    if any(not math.isfinite(value) or value <= 0 for value in positive) or not (
        math.isfinite(setup_seconds) and setup_seconds >= 0
    ):
        raise ValueError("Resource projection requires finite positive measurements")
    training = (
        3
        * training_seconds
        * max(3957 / training_studies, total_training_anchors / training_anchors)
    )
    validation = (
        5
        * validation_seconds
        * max(450 / validation_studies, total_validation_anchors / validation_anchors)
    )
    total = (training + validation + setup_seconds + 1800) * 1.25
    return {
        "three_epoch_training_seconds": training,
        "five_validation_passes_seconds": validation,
        "setup_seconds": setup_seconds,
        "checkpoint_bootstrap_reporting_allowance_seconds": 1800,
        "safety_multiplier": 1.25,
        "projected_total_seconds": total,
        "runtime_limit_seconds": RUNTIME_LIMIT_SECONDS,
        "within_limit": total < RUNTIME_LIMIT_SECONDS,
        "hidden_inference_included": False,
    }


def require_resolution_smoke(
    path: Path | None, config: CombinedBaselineConfig, provenance: Mapping[str, object]
) -> dict[str, Any]:
    if path is None:
        raise ValueError("A successful resource smoke is required before full training")
    report: dict[str, Any] = json.loads(path.read_text())
    attempts = report.get("attempts", [])
    if (
        report.get("schema") != SMOKE_SCHEMA
        or report.get("status") != "safe_for_separate_full_approval"
        or report.get("configuration") != _combined_config_record(config)
        or report.get("validation_membership_digest") != VALIDATION_DIGEST
        or report.get("control_checkpoint_sha256") != PHYSICAL_CROP_CHECKPOINT_SHA256
        or report.get("source", {}).get("git_commit") != provenance.get("git_commit")
        or not provenance.get("git_commit")
        or provenance.get("git_dirty") is not False
        or report.get("source", {}).get("git_dirty") is not False
        or not attempts
        or attempts[-1].get("status") != "safe"
        or [attempt.get("physical_batch_size") for attempt in attempts]
        != [16, 8, 4, 2, 1][: len(attempts)]
        or any(
            attempt.get("status") not in {"cuda_oom", "memory_unsafe"}
            for attempt in attempts[:-1]
        )
        or attempts[-1].get("physical_batch_size") != config.batch_size
    ):
        raise ValueError("Resolution resource smoke provenance or batch policy differs")
    measured = report["measurements"]
    projection = project_resolution_runtime(**measured)
    if projection != report.get("projection") or not projection["within_limit"]:
        raise ValueError("Resolution resource smoke exceeds the nine-hour boundary")
    last = attempts[-1]
    total_memory = report["cuda"]["device_total_memory_bytes"]
    if not (0 < last["peak_reserved_bytes"] < 0.9 * total_memory):
        raise ValueError("Resolution resource smoke memory is unsafe")
    return report


def require_smoke_hardware(
    report: Mapping[str, Any], cuda: Mapping[str, object]
) -> None:
    for key in ("device_name", "device_total_memory_bytes", "cuda_version"):
        if report["cuda"][key] != cuda[key]:
            raise ValueError(
                "Resolution resource smoke accelerator differs from full run"
            )


def _anchor_count(study: IndexedStudy) -> int:
    return sum(min(8, len(series.paths)) for series in study.series)


def _attempt(
    *,
    config: CombinedBaselineConfig,
    weight_path: Path,
    targets: Sequence[str],
    indexed: Mapping[str, IndexedStudy],
    train: Sequence[PartitionStudy],
    stress: Sequence[PartitionStudy],
    validation: Sequence[PartitionStudy],
    deadline: float,
) -> dict[str, Any]:
    seed_everything(config.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, initialization = build_initialized_model(weight_path, targets)
    model.to(torch.device("cuda"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
    scaler = torch.GradScaler("cuda", enabled=config.mixed_precision)
    records = []
    for studies in (stress, train):
        records.append(
            train_epoch(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                indexed=indexed,
                studies=studies,
                preprocessing=config.preprocessing,
                batch_size=config.batch_size,
                seed=config.seed,
                mixed_precision=config.mixed_precision,
                device=torch.device("cuda"),
                epoch=0,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                deadline=deadline,
            )
        )
    prediction = predict_partition(
        model=model,
        indexed=indexed,
        studies=validation,
        preprocessing=config.preprocessing,
        mixed_precision=config.mixed_precision,
        device=torch.device("cuda"),
        deadline=deadline,
    )
    training = records[1]
    return {
        "physical_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "effective_batch_size": 16,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "training": training,
        "stress": records[0],
        "training_seconds_per_optimizer_step": training["training_seconds"]
        / training["optimizer_step_count"],
        "training_anchor_inputs_per_second": training["valid_slice_count"]
        / training["training_seconds"],
        "validation_decode_seconds": prediction.decode_seconds,
        "validation_forward_seconds": prediction.forward_seconds,
        "validation_valid_anchors": prediction.valid_slice_count,
        "validation_anchor_inputs_per_second": prediction.valid_slice_count
        / (prediction.decode_seconds + prediction.forward_seconds),
        "validation_spatial_counts": prediction.spatial_counts,
        "initialization_sha256": initialization.sha256,
    }


def run_resolution_smoke(
    *,
    input_root: Path,
    work_root: Path,
    config: CombinedBaselineConfig,
    approved_gpu: bool = False,
    source_provenance: Mapping[str, object],
) -> Path:
    """Measure a fixed 16-study stress batch, 64 train and 16 validation studies.

    Retry only memory failures, in descending divisors of effective batch 16.
    No checkpoint, validation selection, hidden inference, or full run is produced.
    """
    if _combined_config_record(config) != _expected_resolution_configuration():
        raise ValueError("Resource smoke must start with the fixed 336 batch-16 recipe")
    if source_provenance.get("git_dirty") is not False or not source_provenance.get(
        "git_commit"
    ):
        raise ValueError("Resource smoke requires committed source provenance")
    started = time.perf_counter()
    deadline = started + 3600
    cuda = require_approved_cuda(approved=approved_gpu)
    inputs = discover_combined_training_inputs(input_root)
    cohort = derive_combined_cohort(
        inputs.competition_root,
        inputs.public_dataset_root,
        inputs.index_artifact_dir,
        allow_verified_rebuilt_index=inputs.index_verification
        == "rebuilt_verified_artifact",
    )
    partition = derive_patient_split(
        cohort, split_seed=config.split_seed, val_fraction=config.val_fraction
    )
    if (
        len(partition.train_studies) != 3957
        or len(partition.val_studies) != 450
        or _validation_membership_digest(partition.val_studies) != VALIDATION_DIGEST
    ):
        raise ValueError("Resource smoke partition differs from #31")
    control_path = discover_combined_checkpoint(
        input_root, expected_sha256=PHYSICAL_CROP_CHECKPOINT_SHA256
    )
    control, _ = load_verified_physical_crop_checkpoint(
        control_path,
        device=torch.device("cpu"),
        expected_sha256=PHYSICAL_CROP_CHECKPOINT_SHA256,
    )
    del control
    if (
        load_physical_crop_validation_predictions(
            input_root, partition.val_studies, cohort.target_names
        )
        is None
    ):
        raise ValueError("Resource smoke requires verified cached #31 predictions")
    indexed = load_indexed_studies(
        inputs.competition_root,
        inputs.index_artifact_dir,
        [s.study_uid for s in cohort.studies],
        split="train",
    )
    train = list(partition.train_studies)
    validation = list(partition.val_studies)
    random.Random(config.seed).shuffle(train)
    random.Random(config.seed).shuffle(validation)
    stress = sorted(
        train, key=lambda s: (-_anchor_count(indexed[s.study_uid]), s.study_uid)
    )[:16]
    setup_seconds = time.perf_counter() - started
    report: dict[str, Any] = {
        "schema": SMOKE_SCHEMA,
        "status": "resource_infeasible",
        "source": dict(source_provenance),
        "configuration": _combined_config_record(config),
        "control_checkpoint_sha256": PHYSICAL_CROP_CHECKPOINT_SHA256,
        "validation_membership_digest": VALIDATION_DIGEST,
        "cuda": cuda,
        "attempts": [],
        "sampling": {
            "training_studies": 64,
            "stress_studies": 16,
            "validation_studies": 16,
            "stress_selection": "largest indexed anchor counts",
            "seed": config.seed,
        },
        "full_training_approved": False,
        "hidden_inference": False,
        "competition_submission": False,
    }
    for batch_size in (16, 8, 4, 2, 1):
        candidate = replace(
            config, batch_size=batch_size, gradient_accumulation_steps=16 // batch_size
        )
        attempt: dict[str, Any]
        try:
            attempt = _attempt(
                config=candidate,
                weight_path=inputs.weight_path,
                targets=cohort.target_names,
                indexed=indexed,
                train=train[:64],
                stress=stress,
                validation=validation[:16],
                deadline=deadline,
            )
        except torch.cuda.OutOfMemoryError:
            attempt = {
                "physical_batch_size": batch_size,
                "status": "cuda_oom",
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        except TimeoutError:
            report["status"] = "smoke_time_limit"
            break
        finally:
            gc.collect()
            torch.cuda.empty_cache()
        if attempt.get("status") != "cuda_oom":
            attempt["status"] = (
                "safe"
                if attempt["peak_reserved_bytes"]
                < 0.9 * cast(int, cuda["device_total_memory_bytes"])
                else "memory_unsafe"
            )
        report["attempts"].append(attempt)
        if attempt["status"] != "safe":
            continue
        training = attempt["training"]
        measurements = {
            "training_seconds": training["training_seconds"],
            "training_studies": training["attempted_study_count"],
            "training_anchors": training["valid_slice_count"],
            "validation_seconds": attempt["validation_decode_seconds"]
            + attempt["validation_forward_seconds"],
            "validation_studies": 16,
            "validation_anchors": attempt["validation_valid_anchors"],
            "total_training_anchors": sum(
                _anchor_count(indexed[s.study_uid]) for s in train
            ),
            "total_validation_anchors": sum(
                _anchor_count(indexed[s.study_uid]) for s in validation
            ),
            "setup_seconds": setup_seconds,
        }
        report.update(
            configuration=_combined_config_record(candidate),
            measurements=measurements,
            projection=project_resolution_runtime(**measurements),
        )
        if report["projection"]["within_limit"]:
            report["status"] = "safe_for_separate_full_approval"
        break
    report["elapsed_seconds"] = time.perf_counter() - started
    work_root.mkdir(parents=True, exist_ok=True)
    path = work_root / "resolution-resource-smoke.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return path
