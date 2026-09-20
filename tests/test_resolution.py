import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest


@pytest.mark.parametrize("matching_copies", [True, False])
def test_resolution_smoke_discovers_duplicate_control_attachments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, matching_copies: bool
) -> None:
    import hashlib
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    from rsna_knee import resolution
    from rsna_knee.combined_supervision import load_combined_baseline_config

    # Keep discovery, hashing, and checkpoint loading real. A deliberately
    # incomplete payload stops the smoke before any training can start.
    control = tmp_path / "control.pt"
    torch.save({}, control)
    control_bytes = control.read_bytes()
    monkeypatch.setattr(
        resolution,
        "PHYSICAL_CROP_CHECKPOINT_SHA256",
        hashlib.sha256(control_bytes).hexdigest(),
    )
    for attachment in ("retained", "smoke-output"):
        path = tmp_path / attachment / "checkpoint.pt"
        path.parent.mkdir()
        path.write_bytes(control_bytes if matching_copies else b"wrong checkpoint")
    unrelated = tmp_path / "other" / "checkpoint.pt"
    unrelated.parent.mkdir()
    unrelated.write_bytes(b"unrelated checkpoint")

    # Only bypass the unavailable CUDA and competition-cohort prerequisites;
    # the regression exercises run_resolution_smoke's own discovery call.
    monkeypatch.setattr(resolution, "require_approved_cuda", lambda **_: {})
    monkeypatch.setattr(
        resolution,
        "discover_combined_training_inputs",
        lambda _: SimpleNamespace(
            competition_root=tmp_path,
            public_dataset_root=tmp_path,
            index_artifact_dir=tmp_path,
            index_verification="verified_artifact",
        ),
    )
    monkeypatch.setattr(resolution, "derive_combined_cohort", lambda *_, **__: None)
    monkeypatch.setattr(
        resolution,
        "derive_patient_split",
        lambda *_, **__: SimpleNamespace(
            train_studies=[None] * 3957, val_studies=[None] * 450
        ),
    )
    monkeypatch.setattr(
        resolution,
        "_validation_membership_digest",
        lambda _: resolution.VALIDATION_DIGEST,
    )
    config = load_combined_baseline_config(
        Path(__file__).parents[1] / "configs/resolution-336.toml"
    )
    expected = (
        "Combined checkpoint fields are incomplete or unexpected"
        if matching_copies
        else "Attached checkpoint SHA-256 mismatch"
    )
    with pytest.raises(ValueError, match=expected):
        resolution.run_resolution_smoke(
            input_root=tmp_path,
            work_root=tmp_path / "working",
            config=config,
            source_provenance={"git_dirty": False, "git_commit": "test-commit"},
        )


def test_control_checkpoint_discovery_remains_strict_by_default(tmp_path: Path) -> None:
    import hashlib

    pytest.importorskip("torch")
    from rsna_knee.combined_supervision import discover_combined_checkpoint

    for attachment in ("retained", "smoke-output"):
        path = tmp_path / attachment / "checkpoint.pt"
        path.parent.mkdir()
        path.write_bytes(b"identical control")
    with pytest.raises(ValueError, match="Expected exactly one attached checkpoint"):
        discover_combined_checkpoint(
            tmp_path, expected_sha256=hashlib.sha256(b"identical control").hexdigest()
        )


def test_resolution_config_changes_only_final_resize() -> None:
    pytest.importorskip("torch")
    from rsna_knee.combined_supervision import load_combined_baseline_config

    root = Path(__file__).parents[1]
    control = load_combined_baseline_config(root / "configs/physical-crop.toml")
    candidate = load_combined_baseline_config(root / "configs/resolution-336.toml")
    assert candidate.preprocessing.image_size == 336
    assert replace(candidate, preprocessing=control.preprocessing) == control
    assert replace(candidate.preprocessing, image_size=224) == control.preprocessing


def test_resource_projection_covers_all_five_validation_passes() -> None:
    from rsna_knee.resolution import project_resolution_runtime

    result = project_resolution_runtime(
        training_seconds=64,
        training_studies=64,
        training_anchors=512,
        validation_seconds=16,
        validation_studies=16,
        validation_anchors=128,
        total_training_anchors=3957 * 8,
        total_validation_anchors=450 * 8,
        setup_seconds=100,
    )
    assert result["three_epoch_training_seconds"] == 11871
    assert result["five_validation_passes_seconds"] == 2250
    assert result["projected_total_seconds"] == (11871 + 2250 + 100 + 1800) * 1.25
    assert result["within_limit"] is True
    slow = project_resolution_runtime(
        training_seconds=640,
        training_studies=64,
        training_anchors=512,
        validation_seconds=160,
        validation_studies=16,
        validation_anchors=128,
        total_training_anchors=3957 * 8,
        total_validation_anchors=450 * 8,
        setup_seconds=100,
    )
    assert slow["within_limit"] is False


def test_resolution_bundle_separates_smoke_from_full_approval(tmp_path: Path) -> None:
    from rsna_knee.kaggle import (
        _notebook_source,
        build_plane_gating_bundle,
        load_kaggle_bundle,
    )

    root = Path(__file__).parents[1]
    bundle = build_plane_gating_bundle(
        root, tmp_path / "smoke", resolution_stage="smoke"
    )
    load_kaggle_bundle(bundle.root)
    metadata = json.loads(bundle.kernel_metadata_path.read_text())
    assert metadata["enable_gpu"] == "false"
    assert "plthon/rsna-knee-physical-crop" in metadata["kernel_sources"]
    source = _notebook_source(bundle.notebook_path.read_text())
    assert "run_resolution_smoke" in source
    assert 'STAGE = "smoke"' in source
    with pytest.raises(ValueError, match="resource smoke"):
        build_plane_gating_bundle(root, tmp_path / "full", resolution_stage="full")


def _smoke_report(batch_size: int = 16) -> dict[str, Any]:
    from rsna_knee.combined_supervision import (
        PHYSICAL_CROP_CHECKPOINT_SHA256,
        _combined_config_record,
        load_combined_baseline_config,
    )
    from rsna_knee.resolution import (
        SMOKE_SCHEMA,
        VALIDATION_DIGEST,
        project_resolution_runtime,
    )

    config = load_combined_baseline_config(
        Path(__file__).parents[1] / "configs/resolution-336.toml"
    )
    config = replace(
        config, batch_size=batch_size, gradient_accumulation_steps=16 // batch_size
    )
    measurements = {
        "training_seconds": 64,
        "training_studies": 64,
        "training_anchors": 512,
        "validation_seconds": 16,
        "validation_studies": 16,
        "validation_anchors": 128,
        "total_training_anchors": 31656,
        "total_validation_anchors": 3600,
        "setup_seconds": 100,
    }
    attempts = [
        {"physical_batch_size": size, "status": "cuda_oom"}
        for size in (16, 8, 4, 2, 1)
        if size > batch_size
    ]
    attempts.append(
        {
            "physical_batch_size": batch_size,
            "status": "safe",
            "peak_reserved_bytes": 1000,
        }
    )
    return {
        "schema": SMOKE_SCHEMA,
        "status": "safe_for_separate_full_approval",
        "source": {"git_commit": "a" * 40, "git_dirty": False},
        "configuration": _combined_config_record(config),
        "validation_membership_digest": VALIDATION_DIGEST,
        "control_checkpoint_sha256": PHYSICAL_CROP_CHECKPOINT_SHA256,
        "attempts": attempts,
        "measurements": measurements,
        "projection": project_resolution_runtime(**measurements),
        "cuda": {
            "device_name": "synthetic-test-device",
            "device_total_memory_bytes": 2000,
            "cuda_version": "test",
        },
    }


def test_full_bundle_embeds_verified_smoke_without_enabling_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rsna_knee.kaggle import (
        _notebook_source,
        build_plane_gating_bundle,
        load_kaggle_bundle,
    )

    report = _smoke_report(8)
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(report))
    monkeypatch.setattr("rsna_knee.kaggle._git_state", lambda root: ("a" * 40, False))
    bundle = build_plane_gating_bundle(
        Path(__file__).parents[1],
        tmp_path / "full",
        resolution_stage="full",
        resource_smoke_path=path,
    )
    load_kaggle_bundle(bundle.root)
    metadata = json.loads(bundle.kernel_metadata_path.read_text())
    assert metadata["enable_gpu"] == "false"
    source = _notebook_source(bundle.notebook_path.read_text())
    assert 'STAGE = "full"' in source
    assert "resource_smoke_path=SMOKE_PATH" in source
    assert "run_test_inference=False" in source
    assert "verify_physical_crop_artifact" in source


@pytest.mark.parametrize("batch_size", [16, 8, 4, 2, 1])
def test_smoke_evidence_permits_only_minimal_memory_adaptation(
    tmp_path: Path, batch_size: int
) -> None:
    from rsna_knee.combined_supervision import load_combined_baseline_config
    from rsna_knee.resolution import require_resolution_smoke

    config = load_combined_baseline_config(
        Path(__file__).parents[1] / "configs/resolution-336.toml"
    )
    config = replace(
        config, batch_size=batch_size, gradient_accumulation_steps=16 // batch_size
    )
    report = _smoke_report(batch_size)
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(report))
    assert require_resolution_smoke(path, config, report["source"]) == report
    if batch_size != 16:
        report["attempts"][0]["status"] = "safe"
        path.write_text(json.dumps(report))
        with pytest.raises(ValueError, match="batch policy"):
            require_resolution_smoke(path, config, report["source"])


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_sha",
        "dirty",
        "wrong_split",
        "unsafe_memory",
        "failed",
        "optimistic_projection",
        "wrong_batch",
    ],
)
def test_smoke_gate_rejects_incompatible_or_unsafe_evidence(
    tmp_path: Path, mutation: str
) -> None:
    from rsna_knee.combined_supervision import load_combined_baseline_config
    from rsna_knee.resolution import require_resolution_smoke

    config = load_combined_baseline_config(
        Path(__file__).parents[1] / "configs/resolution-336.toml"
    )
    report = _smoke_report()
    provenance = dict(report["source"])
    if mutation == "wrong_sha":
        report["source"]["git_commit"] = "b" * 40
    elif mutation == "dirty":
        report["source"]["git_dirty"] = True
    elif mutation == "wrong_split":
        report["validation_membership_digest"] = "0" * 64
    elif mutation == "unsafe_memory":
        report["attempts"][-1]["peak_reserved_bytes"] = 1900
    elif mutation == "failed":
        report["status"] = "resource_infeasible"
    elif mutation == "optimistic_projection":
        report["projection"]["projected_total_seconds"] = 1
    else:
        report["configuration"]["training"]["batch_size"] = 8
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="resource smoke"):
        require_resolution_smoke(path, config, provenance)


@pytest.mark.parametrize(
    ("delta", "positive", "regression", "expected"),
    [
        (0.005, 0.95, 0.0, "keep"),
        (0.0049, 0.99, 0.0, "reject"),
        (0.01, 0.9499, 0.0, "reject"),
        (0.01, 0.99, -0.0201, "reject"),
        (0.01, 0.99, None, "reject"),
    ],
)
def test_resolution_keep_rule_is_precommitted(
    delta: float, positive: float, regression: float | None, expected: str
) -> None:
    from rsna_knee.combined_supervision import (
        _resolution_decision,
        load_combined_baseline_config,
    )

    config = load_combined_baseline_config(
        Path(__file__).parents[1] / "configs/resolution-336.toml"
    )
    result = _resolution_decision(
        {
            "delta": {
                "official_macro_auc": delta,
                "per_target": {"target": {"delta_roc_auc": regression}},
            },
            "bootstrap": {
                "uncertainty": {
                    "official_macro_auc_delta": {"fraction_positive": positive}
                }
            },
        },
        config,
    )
    assert result["result"] == expected
    assert result["retained_resolution"] == (336 if expected == "keep" else 224)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("slice_count = 8", "slice_count = 3"),
        ("image_size = 336", "image_size = 320"),
        ('aggregation = "mean"', 'aggregation = "target_slot_attention"'),
        ("physical_fov_mm = 130.0", "physical_fov_mm = 150.0"),
        ("batch_size = 16", "batch_size = 12"),
        ("learning_rate = 0.0001", "learning_rate = 0.0002"),
        ('spatial_mode = "physical_crop"', 'spatial_mode = "whole_image"'),
        (
            'spacing_fallback = "whole_image_resize_pad"',
            'spacing_fallback = "whole_image_resize_pad"\ninput_representation = "adjacent_slice_2_5d"',
        ),
    ],
)
def test_resolution_rejects_other_interventions(
    tmp_path: Path, old: str, new: str
) -> None:
    from rsna_knee.combined_supervision import load_combined_baseline_config

    path = tmp_path / "changed.toml"
    path.write_text(
        (Path(__file__).parents[1] / "configs/resolution-336.toml")
        .read_text()
        .replace(old, new)
    )
    with pytest.raises(ValueError):
        load_combined_baseline_config(path)


def test_resolution_inference_requires_validation_keep_before_discovery(
    tmp_path: Path,
) -> None:
    from rsna_knee.combined_supervision import run_combined_scoring

    with pytest.raises(ValueError, match="validation-retained"):
        run_combined_scoring(
            input_root=tmp_path,
            work_root=tmp_path,
            resolution_336=True,
            physical_crop=True,
        )


def test_full_resolution_run_requires_smoke_before_cuda(tmp_path: Path) -> None:
    from rsna_knee.combined_supervision import (
        load_combined_baseline_config,
        run_combined_baseline,
    )

    config = load_combined_baseline_config(
        Path(__file__).parents[1] / "configs/resolution-336.toml"
    )
    with pytest.raises(ValueError, match="resource smoke"):
        run_combined_baseline(
            input_root=tmp_path,
            work_root=tmp_path,
            config=config,
            run_test_inference=False,
        )


def test_resolution_training_cannot_run_hidden_inference_by_default(
    tmp_path: Path,
) -> None:
    from rsna_knee.combined_supervision import (
        load_combined_baseline_config,
        run_combined_baseline,
    )

    config = load_combined_baseline_config(
        Path(__file__).parents[1] / "configs/resolution-336.toml"
    )
    with pytest.raises(ValueError, match="hidden inference requires separate approval"):
        run_combined_baseline(input_root=tmp_path, work_root=tmp_path, config=config)


@pytest.mark.parametrize("outcome", ["timeout", "oom", "unsafe", "safe", "slow"])
def test_smoke_orchestration_preserves_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    from types import SimpleNamespace

    import torch

    from rsna_knee import resolution
    from rsna_knee.combined_supervision import load_combined_baseline_config
    from rsna_knee.training import ResourceSmokeTimeout

    study = SimpleNamespace(study_uid="synthetic")
    monkeypatch.setattr(
        resolution,
        "require_approved_cuda",
        lambda **_: {
            "device_total_memory_bytes": 2000,
        },
    )
    monkeypatch.setattr(
        resolution,
        "discover_combined_training_inputs",
        lambda _: SimpleNamespace(
            competition_root=tmp_path,
            public_dataset_root=tmp_path,
            index_artifact_dir=tmp_path,
            index_verification="verified_artifact",
            weight_path=tmp_path / "weights.pt",
        ),
    )
    monkeypatch.setattr(
        resolution,
        "derive_combined_cohort",
        lambda *_, **__: SimpleNamespace(
            studies=[study],
            target_names=["target"],
        ),
    )
    monkeypatch.setattr(
        resolution,
        "derive_patient_split",
        lambda *_, **__: SimpleNamespace(
            train_studies=[study] * 3957,
            val_studies=[study] * 450,
        ),
    )
    monkeypatch.setattr(
        resolution,
        "_validation_membership_digest",
        lambda _: resolution.VALIDATION_DIGEST,
    )
    monkeypatch.setattr(
        resolution,
        "discover_combined_checkpoint",
        lambda *_, **__: tmp_path / "control.pt",
    )
    monkeypatch.setattr(
        resolution,
        "load_verified_physical_crop_checkpoint",
        lambda *_, **__: (None, {}),
    )
    monkeypatch.setattr(
        resolution,
        "load_physical_crop_validation_predictions",
        lambda *_, **__: object(),
    )
    monkeypatch.setattr(
        resolution,
        "load_indexed_studies",
        lambda *_, **__: {
            "synthetic": SimpleNamespace(series=[SimpleNamespace(paths=["slice"] * 8)]),
        },
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 800)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 1000)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    current_batch = 16
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(resolution, "seed_everything", lambda _: None)
    monkeypatch.setattr(
        resolution,
        "build_initialized_model",
        lambda *_, **__: (
            SimpleNamespace(to=lambda _: None, parameters=list),
            SimpleNamespace(sha256="test-weights"),
        ),
    )
    monkeypatch.setattr(torch.optim, "AdamW", lambda *_, **__: None)
    monkeypatch.setattr(torch, "GradScaler", lambda *_, **__: None)
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_reserved",
        lambda: 1900 if outcome == "unsafe" and current_batch == 16 else 1000,
    )

    def train(**kwargs: Any) -> dict[str, Any]:
        nonlocal current_batch
        current_batch = kwargs["batch_size"]
        count = len(kwargs["studies"])
        # The real _attempt persists the completed stress phase before training.
        if count == 64 and outcome == "timeout":
            raise ResourceSmokeTimeout(
                "training deadline",
                {
                    "attempted_study_count": 16,
                    "optimizer_step_count": 1,
                    "training_seconds": 12,
                    "valid_slice_count": 128,
                },
            )
        if outcome == "oom" and current_batch == 16:
            raise torch.cuda.OutOfMemoryError()
        return {
            "training_seconds": count * (10 if outcome == "slow" else 1),
            "attempted_study_count": count,
            "valid_slice_count": count * 8,
            "optimizer_step_count": count // 16,
        }

    monkeypatch.setattr(resolution, "train_epoch", train)
    monkeypatch.setattr(
        resolution,
        "predict_partition",
        lambda **_: SimpleNamespace(
            decode_seconds=8,
            forward_seconds=8,
            valid_slice_count=128,
            spatial_counts={},
        ),
    )
    config = load_combined_baseline_config(
        Path(__file__).parents[1] / "configs/resolution-336.toml"
    )
    path = resolution.run_resolution_smoke(
        input_root=tmp_path,
        work_root=tmp_path,
        config=config,
        source_provenance={"git_commit": "test", "git_dirty": False},
    )
    report = json.loads(path.read_text())
    assert report["setup_seconds"] >= 0
    assert "control_checkpoint" in report["setup_phase_seconds"]
    last = report["attempts"][-1]
    assert last["peak_reserved_bytes"] == 1000
    if outcome == "timeout":
        assert report["status"] == "smoke_time_limit"
        assert len(report["attempts"]) == 1
        assert last["status"] == "timeout"
        assert last["phase"] == "training"
        assert last["stress"]["attempted_study_count"] == 16
        assert last["partial_measurements"]["optimizer_step_count"] == 1
        assert "projection" not in report
    elif outcome == "slow":
        assert report["status"] == "resource_infeasible"
        assert len(report["attempts"]) == 1
        assert not report["projection"]["within_limit"]
    else:
        batch = 8 if outcome in {"oom", "unsafe"} else 16
        assert [a["physical_batch_size"] for a in report["attempts"]] == (
            [16, 8] if batch == 8 else [16]
        )
        if batch == 8:
            assert report["attempts"][0]["status"] == (
                "cuda_oom" if outcome == "oom" else "memory_unsafe"
            )
        config = replace(
            config, batch_size=batch, gradient_accumulation_steps=16 // batch
        )
        assert (
            resolution.require_resolution_smoke(path, config, report["source"])
            == report
        )
        return
    with pytest.raises(ValueError, match="resource smoke"):
        resolution.require_resolution_smoke(path, config, report["source"])
