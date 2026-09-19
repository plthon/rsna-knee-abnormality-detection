import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest


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
    from rsna_knee.kaggle import build_resolution_bundle, load_kaggle_bundle

    root = Path(__file__).parents[1]
    bundle = build_resolution_bundle(root, tmp_path / "smoke", stage="smoke")
    load_kaggle_bundle(bundle.root)
    metadata = json.loads(bundle.kernel_metadata_path.read_text())
    assert metadata["enable_gpu"] == "false"
    assert "plthon/rsna-knee-physical-crop" in metadata["kernel_sources"]
    source = bundle.notebook_path.read_text()
    assert "run_resolution_smoke" in source
    assert "run_combined_baseline" not in source
    with pytest.raises(ValueError, match="resource smoke"):
        build_resolution_bundle(root, tmp_path / "full", stage="full")


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
    from rsna_knee.kaggle import build_resolution_bundle, load_kaggle_bundle

    report = _smoke_report(8)
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(report))
    monkeypatch.setattr("rsna_knee.kaggle._git_state", lambda root: ("a" * 40, False))
    bundle = build_resolution_bundle(
        Path(__file__).parents[1],
        tmp_path / "full",
        stage="full",
        resource_smoke_path=path,
    )
    load_kaggle_bundle(bundle.root)
    metadata = json.loads(bundle.kernel_metadata_path.read_text())
    assert metadata["enable_gpu"] == "false"
    source = bundle.notebook_path.read_text()
    assert "resource_smoke_path=SMOKE_PATH" in source
    assert "run_test_inference=False" in source
    assert "verify_resolution_artifact" in source


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
    ],
)
def test_resolution_keep_rule_is_precommitted(
    delta: float, positive: float, regression: float, expected: str
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
