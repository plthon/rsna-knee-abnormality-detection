"""Public summary and reproduction helpers for the gold-only reference baseline."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast


def load_reference_baseline_summary(
    path: Path | None = None,
) -> dict[str, Any]:
    """Return the reference run in reader-facing terms after validating it."""
    evidence = load_public_baseline_evidence(path)
    supervision = _mapping(evidence, "supervision")
    split = _mapping(evidence, "split")
    configuration = _mapping(evidence, "configuration")
    evaluation = _mapping(evidence, "evaluation")
    bootstrap = _mapping(evaluation, "bootstrap")
    uncertainty = _mapping(bootstrap, "uncertainty")
    macro_uncertainty = _mapping(uncertainty, "official_macro_auc")
    point_estimate = _mapping(_mapping(evaluation, "metric"), "point_estimate")
    partitions = _mapping(split, "partitions")
    holdout = _mapping(partitions, "holdout")

    return {
        "targets": list(cast(list[str], supervision["target_order"])),
        "supervision": {
            "source": "Organizer-provided train.csv labels only",
            "report_labels_used": supervision["report_derived_labels_used"],
            "unknown_labels_masked_in_loss": supervision[
                "missing_labels_masked_in_loss"
            ],
        },
        "split": {
            "grouping_column": split["grouping_entity"],
            "patient_overlap": split["patient_overlap_count"],
            "fingerprint": split["assignment_digest"],
            "partitions": partitions,
        },
        "preprocessing": _mapping(configuration, "preprocessing"),
        "model": _mapping(evidence, "model"),
        "training": _mapping(configuration, "training"),
        "results": {
            "macro_auc": point_estimate["official_macro_auc"],
            "per_target": point_estimate["per_target"],
            "per_target_bootstrap": _mapping(uncertainty, "per_target"),
        },
        "runtime": _mapping(evidence, "execution"),
        "checkpoint": _mapping(evidence, "checkpoints"),
        "limitations": {
            "holdout_patient_count": holdout["patient_count"],
            "bootstrap_replicate_count": bootstrap["replicate_count"],
            "defined_macro_replicate_count": macro_uncertainty["valid_replicate_count"],
            "undefined_macro_replicate_count": macro_uncertainty[
                "undefined_replicate_count"
            ],
            "undefined_macro_replicate_rate": macro_uncertainty[
                "undefined_replicate_rate"
            ],
            "supports_model_comparison": False,
        },
    }


def reproduce_reference_baseline(
    *,
    input_root: Path,
    work_parent: Path,
    config_bytes: bytes,
    approved_gpu: bool,
    source_provenance: Mapping[str, object],
    kernel_id: str,
) -> dict[str, object]:
    """Train, reload, predict, validate, and remove private intermediate files."""
    if not isinstance(kernel_id, str) or not kernel_id:
        raise ValueError("Kaggle kernel ID must be nonempty")
    notebook = {
        "kernel_id": kernel_id,
        "kind": "private_review_kaggle_gpu_publication_reproduction",
        "internet_enabled": False,
    }

    from .baseline import (
        export_public_baseline_contract,
        load_baseline_training_config,
        run_gold_holdout_baseline,
        verify_baseline_artifact,
    )
    from .indexing import build_competition_index, verify_index_artifact
    from .submission import discover_submission_inputs, validate_submission
    from .training import predict_submission_from_checkpoint

    reference = load_reference_baseline_summary()
    with TemporaryDirectory(prefix="rsna-knee-reproduction-", dir=work_parent) as temp:
        private_root = Path(temp)
        config_path = private_root / "baseline.toml"
        config_path.write_bytes(config_bytes)
        config = load_baseline_training_config(config_path)

        index_outcome = build_competition_index(
            input_root=input_root,
            work_root=private_root,
            config=config.preprocessing,
        )
        verify_index_artifact(index_outcome.artifact_dir)
        outcome = run_gold_holdout_baseline(
            input_root=input_root,
            work_root=private_root,
            config=config,
            index_artifact_dir=index_outcome.artifact_dir,
            approved_gpu=approved_gpu,
            source_provenance=source_provenance,
            notebook=notebook,
        )
        verified = verify_baseline_artifact(outcome.artifact_dir)
        rerun = export_public_baseline_contract(
            outcome.artifact_dir, private_root / "reproduction-summary.json"
        )
        if rerun["supervision"]["target_order"] != reference["targets"]:
            raise RuntimeError("Reproduction changed the target order")
        if rerun["split"]["assignment_digest"] != reference["split"]["fingerprint"]:
            raise RuntimeError("Reproduction changed the patient split")

        prediction = predict_submission_from_checkpoint(
            input_root=input_root,
            work_root=private_root / "submission-check",
            checkpoint_path=outcome.best_checkpoint_path,
            approved_gpu=approved_gpu,
            index_artifact_dir=index_outcome.artifact_dir,
        )
        submission_validation = validate_submission(
            prediction.submission_path,
            inputs=discover_submission_inputs(input_root),
        )
        if submission_validation["valid"] is not True:
            raise RuntimeError("Generated predictions failed submission validation")

        summary: dict[str, object] = {
            "training_outputs_verified": verified["valid"],
            "patient_split_reproduced": (
                verified["assignment_digest"] == reference["split"]["fingerprint"]
            ),
            "best_checkpoint_used_for_test_predictions": True,
            "submission_valid": submission_validation["valid"],
            "submission_rows": submission_validation["row_count"],
            "test_predictions": prediction.prediction_count,
            "fallback_predictions": prediction.fallback_count,
            "decode_failures": prediction.decode_errors,
        }

    if private_root.exists():
        raise RuntimeError("Temporary reproduction files were not removed")
    summary["temporary_files_removed"] = True
    return summary


def load_public_baseline_evidence(
    path: Path | None = None,
) -> dict[str, Any]:
    """Load and cross-check the aggregate evidence packaged in the wheel."""
    baseline = _load_json("gold-baseline.json", path)
    competition = _load_json("competition.json")
    supervision = _load_json("gold-supervision.json")
    training = _load_json("gold-training.json")
    initialization = _load_json("resnet18-imagenet1k-v1.json")

    if (
        baseline.get("schema_version") != 1
        or baseline.get("artifact_type") != "rsna_knee_gold_holdout_baseline"
        or _mapping(baseline, "verification").get("valid") is not True
    ):
        raise ValueError("Public baseline evidence identity is invalid")

    targets = _mapping(competition, "targets").get("order")
    baseline_supervision = _mapping(baseline, "supervision")
    if (
        not isinstance(targets, list)
        or baseline_supervision.get("target_order") != targets
        or baseline_supervision.get("source") != "organizer gold train.csv only"
        or baseline_supervision.get("report_derived_labels_used") is not False
        or baseline_supervision.get("missing_labels_masked_in_loss") is not True
    ):
        raise ValueError("Public baseline supervision or target order has drifted")

    evaluation_witness = _mapping(supervision, "evaluation_eligibility")
    assignment = _mapping(evaluation_witness, "assignment_digest")
    split = _mapping(baseline, "split")
    if (
        evaluation_witness.get("decision") != "Candidate"
        or split.get("grouping_entity") != "PatientID"
        or split.get("patient_overlap_count") != 0
        or split.get("independently_rederived_before_training") is not True
        or split.get("digest_verified_before_training") is not True
        or split.get("assignment_digest") != assignment.get("value")
    ):
        raise ValueError("Public baseline frozen split evidence has drifted")

    partitions = _mapping(split, "partitions")
    for partition_name in ("train", "holdout"):
        partition = _mapping(partitions, partition_name)
        per_target = _mapping(partition, "per_target")
        if set(per_target) != set(cast(list[str], targets)):
            raise ValueError("Public baseline class-support targets have drifted")
        for target in cast(list[str], targets):
            counts = _mapping(per_target, target)
            if counts.get("positive", 0) < 1 or counts.get("negative", 0) < 1:
                raise ValueError("Public baseline partition lacks both classes")

    model = _mapping(baseline, "model")
    initialization_artifact = _mapping(initialization, "artifact")
    if (
        model.get("architecture") != "torchvision.models.resnet18"
        or model.get("image_size") != 224
        or model.get("slice_to_series_pooling") != "masked mean"
        or model.get("series_to_study_pooling") != "masked mean"
        or model.get("initialization_identity") != "ResNet18_Weights.IMAGENET1K_V1"
        or model.get("initialization_artifact_sha256")
        != initialization_artifact.get("sha256")
        or model.get("initialization_validated_before_model_construction") is not True
        or model.get("network_fallback") is not False
    ):
        raise ValueError("Public baseline model contract has drifted")

    bootstrap = _mapping(_mapping(baseline, "evaluation"), "bootstrap")
    uncertainty = _mapping(bootstrap, "uncertainty")
    macro = _mapping(uncertainty, "official_macro_auc")
    if (
        bootstrap.get("sampling_unit") != "PatientID"
        or bootstrap.get("replicate_count") != 10_000
        or macro.get("valid_replicate_count") != 935
        or macro.get("undefined_replicate_count") != 9_065
    ):
        raise ValueError("Public baseline bootstrap evidence has drifted")

    adequacy = _mapping(baseline, "adequacy_for_issue_11")
    if (
        adequacy.get("decision") != "inadequate"
        or adequacy.get("claims_general_statistical_sufficiency") is not False
        or adequacy.get("leaderboard_evidence_used") is not False
    ):
        raise ValueError("Public baseline must not claim support for model comparison")
    if _mapping(baseline, "exclusions") != {
        "leaderboard_evidence_used": False,
        "report_derived_labels_used": False,
    }:
        raise ValueError("Public baseline includes a prohibited evidence source")

    checkpoints = _mapping(baseline, "checkpoints")
    training_submission = _mapping(training, "submission")
    if (
        checkpoints.get("best_reloaded_for_reported_predictions") is not True
        or checkpoints.get("reload_predictions_within_tolerance") is not True
        or training_submission.get("reloaded_checkpoint_used") is not True
        or training_submission.get("validator_passed") is not True
    ):
        raise ValueError("Checkpoint reload or submission evidence is incomplete")

    source_contracts = _mapping(baseline, "source_contracts")
    expected_source_hashes = {
        "competition_contract_sha256": _resource_sha256("competition.json"),
        "gold_supervision_contract_sha256": _resource_sha256("gold-supervision.json"),
        "gold_training_contract_sha256": _resource_sha256("gold-training.json"),
        "initialization_artifact_sha256": initialization_artifact.get("sha256"),
        "evaluation_assignment_digest": assignment.get("value"),
    }
    if any(
        source_contracts.get(key) != value
        for key, value in expected_source_hashes.items()
    ):
        raise ValueError("Public baseline source provenance has drifted")

    privacy = _mapping(baseline, "privacy")
    if privacy != {
        "competition_data_downloaded": False,
        "private_artifacts_registered_under_ignored_path": True,
        "row_level_membership_labels_patient_ids_or_predictions_committed": False,
    }:
        raise ValueError("Public baseline privacy boundary has drifted")
    return baseline


def _load_json(name: str, path: Path | None = None) -> dict[str, Any]:
    raw = (path.read_bytes() if path else _resource_bytes(name)).decode("utf-8")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError(f"Packaged contract must be an object: {name}")
    return cast(dict[str, Any], value)


def _resource_sha256(name: str) -> str:
    return hashlib.sha256(_resource_bytes(name)).hexdigest()


def _resource_bytes(name: str) -> bytes:
    repository_path = Path(__file__).resolve().parents[2] / "contracts" / name
    return (
        repository_path.read_bytes()
        if repository_path.is_file()
        else files("rsna_knee").joinpath(name).read_bytes()
    )


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Public baseline field {key} must be an object")
    return cast(dict[str, Any], value)
