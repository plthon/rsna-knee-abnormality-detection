"""Issue #38 batch-4/accumulation-2 derivative of the frozen Raptor recipe."""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Mapping
from dataclasses import replace
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from . import raptor_reconstruction as parent
from .provenance import create_run_id, write_run_manifest
from .raptor_reference import RaptorClassifier, load_raptor_training_reference_contract
from .training import require_approved_cuda, save_checkpoint

type Stage = Literal["smoke", "full"]

_BEST = "raptor-t4-best.pt"
_AVERAGED = "raptor-t4-swa.pt"
_RESUME = "raptor-t4-resume.pt"
_REPORT = "raptor-t4-report.json"
_MANIFEST = "run-manifest.json"


def derivative_config() -> tuple[
    parent.RaptorReconstructionConfig, dict[str, Any], str
]:
    contract, parent_sha = load_raptor_training_reference_contract()
    path = files("rsna_knee").joinpath("raptor-t4-derivative-v1.json")
    try:
        contract_bytes = path.read_bytes()
    except FileNotFoundError:
        contract_bytes = (
            Path(__file__).parents[2] / "contracts" / "raptor-t4-derivative-v1.json"
        ).read_bytes()
    raw = json.loads(contract_bytes)
    if not isinstance(raw, dict):
        raise TypeError("Raptor T4 derivative contract is malformed")
    if (
        raw.get("schema_version") != 1
        or raw.get("kind") != "issue_38_kaggle_t4_derivative"
        or raw.get("parent_sha256") != parent_sha
        or raw.get("physical_batch_size") != 4
        or raw.get("gradient_accumulation_steps") != 2
        or raw.get("effective_batch_size") != 8
        or raw.get("smoke_optimizer_updates") != 3
        or raw.get("pretrained_revision") != "2c31d82376cc244c907c064332941e6c65be81f1"
        or raw.get("pretrained_sha256")
        != "91570c7af10fea0af03c1b0a49be3e4775596239890eb4193c75129b3045623e"
    ):
        raise ValueError("Raptor T4 derivative configuration has drifted")
    original = parent.reconstruction_config(contract)
    if original.batch_size != 8:
        raise ValueError("Raptor T4 parent batch size has drifted")
    return (
        replace(original, batch_size=4),
        raw,
        hashlib.sha256(contract_bytes).hexdigest(),
    )


def optimizer_updates(micro_batches: int, accumulation_steps: int = 2) -> int:
    if micro_batches < 1 or accumulation_steps != 2:
        raise ValueError(
            "Raptor T4 update accounting requires positive batches and accumulation 2"
        )
    return (micro_batches + accumulation_steps - 1) // accumulation_steps


def train_epoch(
    model: RaptorClassifier,
    loader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: Optimizer,
    scheduler: OneCycleLR,
    loss_function: nn.Module,
    *,
    device: torch.device,
    gradient_clip_norm: float,
    maximum_updates: int | None = None,
    progress: dict[str, object] | None = None,
) -> dict[str, object]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    micro_seconds: list[float] = []
    update_seconds: list[float] = []
    update_studies: list[int] = []
    if progress is not None:
        progress.update(
            micro_batch_seconds=micro_seconds,
            optimizer_update_seconds=update_seconds,
            update_study_counts=update_studies,
        )
    pending = 0
    pending_studies = 0
    group_started = time.perf_counter()

    def step() -> None:
        nonlocal pending, pending_studies, group_started
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        update_seconds.append(time.perf_counter() - group_started)
        update_studies.append(pending_studies)
        pending = 0
        pending_studies = 0
        group_started = time.perf_counter()

    for inputs, targets in loader:
        started = time.perf_counter()
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(inputs.to(device, non_blocking=True))
            loss = loss_function(logits, targets.to(device, non_blocking=True))
        (loss / 2).backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        micro_seconds.append(time.perf_counter() - started)
        losses.append(float(loss.detach().cpu()))
        pending += 1
        pending_studies += len(inputs)
        if pending == 2:
            step()
            if maximum_updates is not None and len(update_seconds) >= maximum_updates:
                break
    if not losses:
        raise RuntimeError("Raptor T4 training loader produced no batches")
    # A final odd micro-batch is stepped with the same loss / 2 rule.
    if pending:
        step()
    return {
        "training_loss": float(np.mean(losses)),
        "micro_batches": len(losses),
        "optimizer_updates": len(update_seconds),
        "scheduler_steps": len(update_seconds),
        "update_study_counts": update_studies,
        "micro_batch_seconds": micro_seconds,
        "optimizer_update_seconds": update_seconds,
    }


def _rng_state() -> dict[str, object]:
    numpy = cast(tuple[str, np.ndarray, int, int, float], np.random.get_state())
    return {
        "python": random.getstate(),
        "numpy": (numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng(raw: Mapping[str, Any]) -> None:
    numpy = raw["numpy"]
    random.setstate(raw["python"])
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(raw["torch"])
    torch.cuda.set_rng_state_all(raw["cuda"])


def _save_resume(
    path: Path,
    *,
    model: RaptorClassifier,
    optimizer: Optimizer,
    scheduler: OneCycleLR,
    epoch: int,
    updates: int,
    records: list[dict[str, object]],
    resume_boundaries: list[dict[str, object]],
    top_states: tuple[parent.RankedState, ...],
    best: parent.RankedState,
    contract_sha256: str,
    source_commit: str,
    label_sha256: str,
    pretrained_sha256: str,
) -> None:
    payload = {
        "kind": "issue_38_raptor_t4_resume_v1",
        "contract_sha256": contract_sha256,
        "source_commit": source_commit,
        "label_sha256": label_sha256,
        "pretrained_sha256": pretrained_sha256,
        "epoch": epoch,
        "optimizer_updates": updates,
        "model": parent._cpu_state(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "records": records,
        "resume_boundaries": resume_boundaries,
        "top_states": [
            (state.macro_auc, state.epoch, state.aucs, state.state)
            for state in top_states
        ],
        "best": (best.macro_auc, best.epoch, best.aucs, best.state),
        "rng": _rng_state(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_resume(
    path: Path,
    *,
    model: RaptorClassifier,
    optimizer: Optimizer,
    scheduler: OneCycleLR,
    contract_sha256: str,
    source_commit: str,
    label_sha256: str,
    pretrained_sha256: str,
    epochs: int,
    updates_per_epoch: int,
) -> tuple[
    int,
    int,
    list[dict[str, object]],
    list[dict[str, object]],
    tuple[parent.RankedState, ...],
    parent.RankedState,
]:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if (
        raw.get("kind") != "issue_38_raptor_t4_resume_v1"
        or raw.get("contract_sha256") != contract_sha256
        or raw.get("source_commit") != source_commit
        or raw.get("label_sha256") != label_sha256
        or raw.get("pretrained_sha256") != pretrained_sha256
        or type(raw.get("epoch")) is not int
        or not 1 <= raw["epoch"] < epochs
        or raw.get("optimizer_updates") != raw["epoch"] * updates_per_epoch
        or len(raw.get("records", [])) != raw["epoch"]
    ):
        raise ValueError("Raptor T4 resume identity or progress differs")
    model.load_state_dict(raw["model"], strict=True)
    optimizer.load_state_dict(raw["optimizer"])
    scheduler.load_state_dict(raw["scheduler"])
    if scheduler.last_epoch != raw["optimizer_updates"]:
        raise ValueError("Raptor T4 scheduler progress differs from optimizer updates")
    _restore_rng(raw["rng"])
    states = tuple(parent.RankedState(*item) for item in raw["top_states"])
    best = parent.RankedState(*raw["best"])
    return (
        raw["epoch"],
        raw["optimizer_updates"],
        raw["records"],
        raw["resume_boundaries"],
        states,
        best,
    )


def run_raptor_t4_derivative(
    *,
    input_root: Path,
    work_root: Path,
    stage: Stage,
    approved_gpu: bool,
    source_provenance: Mapping[str, object],
    notebook: Mapping[str, object],
    resume_path: Path | None = None,
) -> Path:
    if stage not in {"smoke", "full"} or (resume_path is not None and stage != "full"):
        raise ValueError("Raptor T4 stage must be smoke or full; only full can resume")
    started = time.perf_counter()
    contract, parent_sha = load_raptor_training_reference_contract()
    config, derivative, derivative_sha = derivative_config()
    pinned_label = parent.required_pinned_label_identity(contract)
    cuda = require_approved_cuda(approved=approved_gpu)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    parent._seed_source_generators(config.seed)
    corpus = parent._discover_corpus(input_root, contract)
    soft_ids, soft_targets, label_record = parent._read_soft_labels(
        parent._unique_input(input_root, "labels_llm_soft.parquet", "rsna-knee-labels")
    )
    if any(label_record[key] != pinned_label[key] for key in pinned_label):
        raise ValueError("Raptor T4 label artifact differs from its pinned identity")
    gold_ids, gold_targets = parent._read_gold_labels(
        parent._unique_input(input_root, "train.csv", "rsna-knee-abnormality-detection")
    )
    cohorts = parent.prepare_cohorts(
        corpus_ids=corpus.ids,
        soft_ids=soft_ids,
        soft_targets=soft_targets,
        gold_ids=gold_ids,
        gold_targets=gold_targets,
    )
    model, pretrained = parent._initialize_pretrained_model(
        device,
        revision=derivative["pretrained_revision"],
        expected_sha256=derivative["pretrained_sha256"],
    )
    train_loader = DataLoader(
        parent.RaptorWindowDataset(
            corpus=corpus,
            indices=cohorts.training_indices,
            targets=cohorts.training_targets,
            window_count=config.training_windows,
            resolution=config.resolution,
            training=True,
            gain_minimum=config.gain_minimum,
            gain_maximum=config.gain_maximum,
        ),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
    )
    gold_loader = DataLoader(
        parent.RaptorWindowDataset(
            corpus=corpus,
            indices=cohorts.gold_indices,
            targets=cohorts.gold_targets,
            window_count=config.evaluation_windows,
            resolution=config.resolution,
            training=False,
        ),
        batch_size=config.validation_batch_size,
        shuffle=False,
        num_workers=config.workers,
    )
    updates_per_epoch = optimizer_updates(len(train_loader))
    optimizer, scheduler = parent.build_optimizer_and_scheduler(
        model, config, steps_per_epoch=updates_per_epoch
    )
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=parent.prevalence_positive_weights(cohorts.training_targets).to(
            device
        )
    )
    records: list[dict[str, object]] = []
    top_states: tuple[parent.RankedState, ...] = ()
    best: parent.RankedState | None = None
    completed_epoch = 0
    total_updates = 0
    resume_boundaries: list[dict[str, object]] = []
    if resume_path is not None:
        completed_epoch, total_updates, records, resume_boundaries, top_states, best = (
            _load_resume(
                resume_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                contract_sha256=derivative_sha,
                source_commit=str(source_provenance["git_commit"]),
                label_sha256=str(label_record["sha256"]),
                pretrained_sha256=pretrained.sha256,
                epochs=config.epochs,
                updates_per_epoch=updates_per_epoch,
            )
        )
        resume_boundaries.append(
            {"from_epoch": completed_epoch, "sha256": parent._sha256(resume_path)}
        )
    work_root.mkdir(parents=True, exist_ok=True)
    if best is not None:
        save_checkpoint(
            work_root / _BEST,
            parent.checkpoint_payload(
                state=best.state,
                macro_auc=best.macro_auc,
                aucs=best.aucs,
                epoch=best.epoch,
                source="issue_38_raptor_t4_derivative",
            ),
        )

    def record_oom(progress: dict[str, object]) -> None:
        failure = {
            "schema_version": 1,
            "stage": stage,
            "status": "resource_infeasible",
            "source": dict(source_provenance),
            "notebook": dict(notebook),
            "parent_reference_sha256": parent_sha,
            "derivative_sha256": derivative_sha,
            "configuration": {
                **config.record(),
                "gradient_accumulation_steps": 2,
                "effective_batch_size": 8,
            },
            "pretrained_weight": pretrained.record(),
            "progress": progress,
            "optimizer_update_count": scheduler.last_epoch,
            "scheduler_step_count": scheduler.last_epoch,
            "execution": {
                "elapsed_seconds": time.perf_counter() - started,
                "peak_allocated_gpu_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
                "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved(device)),
                "oom_or_retry": True,
                "hidden_test_inference": False,
                "competition_submission": False,
            },
            "runtime": parent._runtime_evidence(device, cuda),
        }
        (work_root / _REPORT).write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n"
        )

    target_epoch = 1 if stage == "smoke" else min(completed_epoch + 2, config.epochs)
    for epoch in range(completed_epoch + 1, target_epoch + 1):
        progress: dict[str, object] = {}
        try:
            training = train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                loss_function,
                device=device,
                gradient_clip_norm=config.gradient_clip_norm,
                maximum_updates=3 if stage == "smoke" else None,
                progress=progress,
            )
        except torch.OutOfMemoryError:
            record_oom(progress)
            raise
        updates = training["optimizer_updates"]
        assert isinstance(updates, int)
        total_updates += updates
        try:
            metric, aucs = parent._evaluate_gold(
                model, gold_loader, cohorts.gold_targets, device
            )
        except torch.OutOfMemoryError:
            record_oom(progress)
            raise
        macro_auc = parent._required_macro_auc(metric)
        candidate = parent.RankedState(macro_auc, epoch, aucs, parent._cpu_state(model))
        top_states = parent.retain_top_states(top_states, candidate, topk=config.topk)
        if parent.is_strict_improvement(macro_auc, best):
            best = candidate
            save_checkpoint(
                work_root / _BEST,
                parent.checkpoint_payload(
                    state=best.state,
                    macro_auc=best.macro_auc,
                    aucs=best.aucs,
                    epoch=best.epoch,
                    source="issue_38_raptor_t4_derivative",
                ),
            )
        records.append(
            {
                "epoch": epoch,
                **training,
                "gold_macro_auc": macro_auc,
                "gold_per_target_auc": list(aucs),
            }
        )
        if stage == "full" and best is not None:
            _save_resume(
                work_root / _RESUME,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                updates=total_updates,
                records=records,
                resume_boundaries=resume_boundaries,
                top_states=top_states,
                best=best,
                contract_sha256=derivative_sha,
                source_commit=str(source_provenance["git_commit"]),
                label_sha256=str(label_record["sha256"]),
                pretrained_sha256=pretrained.sha256,
            )
    if best is None:
        raise RuntimeError("Raptor T4 derivative produced no evaluated epoch")
    expected_updates = 3 if stage == "smoke" else target_epoch * updates_per_epoch
    if total_updates != expected_updates or scheduler.last_epoch != total_updates:
        raise RuntimeError("Raptor T4 optimizer or scheduler update count differs")
    averaged_state = parent.average_ranked_states(top_states)
    model.load_state_dict(averaged_state, strict=True)
    try:
        averaged_metric, averaged_aucs = parent._evaluate_gold(
            model, gold_loader, cohorts.gold_targets, device
        )
    except torch.OutOfMemoryError:
        record_oom({"phase": "averaged_checkpoint_validation"})
        raise
    averaged_macro = parent._required_macro_auc(averaged_metric)
    save_checkpoint(
        work_root / _AVERAGED,
        parent.checkpoint_payload(
            state=averaged_state,
            macro_auc=averaged_macro,
            aucs=averaged_aucs,
            epoch=top_states[0].epoch,
            source="issue_38_raptor_t4_derivative_top3_average",
        ),
    )
    runtime = parent._runtime_evidence(device, cuda)
    report = {
        "schema_version": 1,
        "stage": stage,
        "notebook": dict(notebook),
        "parent_reference_sha256": parent_sha,
        "derivative_sha256": derivative_sha,
        "configuration": {
            **config.record(),
            "gradient_accumulation_steps": 2,
            "effective_batch_size": 8,
        },
        "cohorts": {
            "soft_training_study_count": len(cohorts.training_indices),
            "gold_validation_study_count": len(cohorts.gold_indices),
            "gold_excluded_from_training": True,
        },
        "label_source": label_record,
        "pretrained_weight": pretrained.record(),
        "epochs": records,
        "completed_epochs": target_epoch,
        "full_training_complete": stage == "full" and target_epoch == config.epochs,
        "optimizer_update_count": total_updates,
        "scheduler_step_count": scheduler.last_epoch,
        "resume_boundaries": resume_boundaries,
        "checkpoint_selection": {
            "best_epoch": best.epoch,
            "best_gold_macro_auc": best.macro_auc,
            "top_epochs": [item.epoch for item in top_states],
            "averaged_gold_macro_auc": averaged_macro,
        },
        "comparison_reference": {
            "released_best_epoch": 12,
            "released_gold_macro_auc": 0.9167492483170476,
            "released_public_leaderboard": 0.924,
            "role": "comparison_only_not_a_tuning_target",
        },
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved(device)),
            "oom_or_retry": False,
            "hidden_test_inference": False,
            "competition_submission": False,
        },
        "runtime": runtime,
    }
    report_path = work_root / _REPORT
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    artifacts = [work_root / _BEST, work_root / _AVERAGED, report_path]
    if (work_root / _RESUME).exists():
        artifacts.append(work_root / _RESUME)
    write_run_manifest(
        work_root / _MANIFEST,
        run_id=create_run_id("raptor-t4-derivative"),
        source={
            **dict(source_provenance),
            "parent_reference_sha256": parent_sha,
            "derivative_sha256": derivative_sha,
            "label_source": label_record,
            "pretrained_weight": pretrained.record(),
            "corpus": parent._mapping(contract, "attached_data"),
        },
        configuration={
            **config.record(),
            "gradient_accumulation_steps": 2,
            "effective_batch_size": 8,
            "stage": stage,
            "partial_group": derivative["partial_group"],
        },
        notebook=notebook,
        artifacts=artifacts,
        artifact_root=work_root,
        runtime=runtime,
    )
    verify_raptor_t4_artifact(work_root)
    return report_path


def verify_raptor_t4_artifact(root: Path) -> dict[str, object]:
    report = parent._json_object(root / _REPORT)
    manifest = parent._json_object(root / _MANIFEST)
    if (
        report.get("schema_version") != 1
        or report.get("stage") not in {"smoke", "full"}
        or report.get("configuration", {}).get("batch_size") != 4
        or report.get("configuration", {}).get("gradient_accumulation_steps") != 2
        or report.get("execution", {}).get("hidden_test_inference") is not False
        or report.get("execution", {}).get("competition_submission") is not False
    ):
        raise ValueError("Raptor T4 report violates derivative boundary")
    expected = {_BEST, _AVERAGED, _REPORT}
    if (root / _RESUME).exists():
        expected.add(_RESUME)
    records = manifest.get("artifacts")
    if (
        not isinstance(records, list)
        or {item.get("path") for item in records} != expected
    ):
        raise ValueError("Raptor T4 manifest inventory differs")
    for item in records:
        path = root / item["path"]
        if (
            item.get("sha256") != parent._sha256(path)
            or item.get("size_bytes") != path.stat().st_size
        ):
            raise ValueError(f"Raptor T4 artifact hash differs: {path.name}")
    return {"stage": report["stage"], "artifact_count": len(expected)}
