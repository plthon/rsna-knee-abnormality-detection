import hashlib
import io
import json
from collections.abc import Sequence
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor, nn

import rsna_knee.training as training_module
from rsna_knee.combined_supervision import (
    DinoV2EarlyStopping,
    _dinov2_decision,
    _dinov2_finetuning_metadata,
    _retained_hidden_test_inference_estimate,
    discover_combined_checkpoint,
    load_combined_baseline_config,
    load_combined_checkpoint,
    load_verified_combined_checkpoint,
    load_verified_dinov2_checkpoint,
)
from rsna_knee.config import PreprocessingConfig
from rsna_knee.dinov2 import (
    DINOV2_VITS14_FEATURE_COUNT,
    DinoV2ViTS14,
    dinov2_vits14,
)
from rsna_knee.training import (
    DecodedStudy,
    GoldResNet18,
    GoldStudy,
    IndexedStudy,
    Label,
    _new_uninitialized_model,
    build_dinov2_initialized_model,
    build_dinov2_optimizer,
    configure_dinov2_finetuning,
    train_epoch,
    verify_dinov2_initialization_artifact,
)

PROJECT_ROOT = Path(__file__).parents[1]
TARGETS = tuple(
    json.loads((PROJECT_ROOT / "contracts/competition.json").read_text())["targets"][
        "order"
    ]
)


def test_dinov2_config_names_the_only_supported_candidate(tmp_path: Path) -> None:
    config = load_combined_baseline_config(PROJECT_ROOT / "configs/dinov2.toml")
    assert config.architecture == "dinov2_vits14"
    assert config.weight_identity == "dinov2_vits14.LVD142M"
    assert config.aggregation == "mean"
    assert config.batch_size == 4
    assert config.gradient_accumulation_steps == 4
    assert config.effective_batch_size == 16
    assert config.epochs == 20
    assert config.backbone_learning_rate == 8e-6
    assert config.head_learning_rate == 1e-3
    assert config.early_stopping_minimum_epochs == 10
    assert config.early_stopping_patience == 4
    assert config.early_stopping_min_delta == 0.002

    unsupported = (
        (PROJECT_ROOT / "configs/dinov2.toml")
        .read_text()
        .replace('architecture = "dinov2_vits14"', 'architecture = "dinov2_vitb14"')
    )
    path = tmp_path / "unsupported.toml"
    path.write_text(unsupported)
    with pytest.raises(ValueError, match="supported backbone"):
        load_combined_baseline_config(path)


def test_dinov2_finetuning_has_two_disjoint_groups_and_frozen_parameters_stay_fixed() -> (
    None
):
    model = GoldResNet18(
        dinov2_vits14(), DINOV2_VITS14_FEATURE_COUNT, target_count=len(TARGETS)
    )
    configure_dinov2_finetuning(model)
    optimizer = build_dinov2_optimizer(
        model,
        backbone_learning_rate=8e-6,
        head_learning_rate=1e-3,
        weight_decay=1e-4,
    )
    backbone = cast(DinoV2ViTS14, model.backbone)

    assert all(
        not parameter.requires_grad
        for parameter in (
            *backbone.patch_embed.parameters(),
            backbone.cls_token,
            backbone.pos_embed,
            backbone.mask_token,
            *(
                parameter
                for block in backbone.blocks[:6]
                for parameter in block.parameters()
            ),
        )
    )
    assert all(
        parameter.requires_grad
        for parameter in (
            *(
                parameter
                for block in backbone.blocks[6:]
                for parameter in block.parameters()
            ),
            *backbone.norm.parameters(),
            *model.head.parameters(),
        )
    )
    assert len(optimizer.param_groups) == 2
    assert [group["lr"] for group in optimizer.param_groups] == [8e-6, 1e-3]
    grouped = [
        parameter
        for group in optimizer.param_groups
        for parameter in cast(list[Tensor], group["params"])
    ]
    assert len({id(parameter) for parameter in grouped}) == len(grouped)
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }

    frozen_before = backbone.patch_embed.proj.weight.detach().clone()
    trainable_parameter = next(backbone.blocks[6].parameters())
    trainable_before = trainable_parameter.detach().clone()
    trainable_parameter.sum().backward()  # type: ignore[no-untyped-call]
    model.head.weight.grad = torch.ones_like(model.head.weight)
    optimizer.step()

    assert torch.equal(backbone.patch_embed.proj.weight, frozen_before)
    assert trainable_parameter.grad is not None
    assert model.head.weight.grad is not None
    assert not torch.equal(trainable_parameter, trainable_before)


def test_dinov2_early_stopping_tracks_best_checkpoint_and_eligible_patience() -> None:
    state = DinoV2EarlyStopping(
        minimum_epochs=10,
        maximum_epochs=20,
        patience=4,
        min_delta=0.002,
    )
    for epoch in range(1, 10):
        assert state.observe(epoch, 0.5) is False
    assert state.observe(10, 0.5) is False
    resumed = DinoV2EarlyStopping.from_record(state.to_record())
    assert resumed.to_record() == state.to_record()
    assert state.observe(11, 0.501) is resumed.observe(11, 0.501) is False
    assert resumed.to_record() == state.to_record()
    assert state.retained_best_epoch == 11
    assert state.eligible_non_improving_epochs == 2
    assert state.observe(12, 0.502) is False
    assert state.reference_best_macro_auc == 0.502
    assert state.eligible_non_improving_epochs == 0
    assert state.observe(13, 0.502) is False
    assert state.observe(14, 0.501) is False
    assert state.observe(15, 0.501) is False
    assert state.observe(16, 0.501) is True
    assert state.stop_reason == "patience_exhausted"
    assert state.stopping_epoch == 16

    ties = DinoV2EarlyStopping(10, 20, 4, 0.002)
    for epoch, auc in enumerate((0.4,) * 11, start=1):
        ties.observe(epoch, auc)
    assert ties.retained_best_epoch == 1

    maximum = DinoV2EarlyStopping(10, 20, 4, 0.002)
    for epoch in range(1, 21):
        assert maximum.observe(epoch, epoch / 100) is (epoch == 20)
    assert maximum.stop_reason == "maximum_epochs_reached"


@pytest.mark.training
def test_dinov2_encoder_returns_the_cls_feature() -> None:
    model = dinov2_vits14()
    with torch.no_grad():
        features = model(torch.zeros(1, 3, 224, 224))
    assert features.shape == (1, DINOV2_VITS14_FEATURE_COUNT)
    assert DINOV2_VITS14_FEATURE_COUNT == 384


def test_dinov2_contract_pins_official_implementation_and_weights() -> None:
    contract = json.loads(
        (PROJECT_ROOT / "contracts/dinov2-vits14-lvd142m.json").read_text()
    )
    assert contract["identity"] == {
        "backbone": "dinov2_vits14",
        "feature_count": 384,
        "implementation_commit": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
        "implementation_entrypoint": "dinov2.hub.backbones.dinov2_vits14",
        "pretraining": "LVD-142M",
        "register_tokens": 0,
    }
    assert contract["artifact"] == {
        "filename": "dinov2_vits14_pretrain.pth",
        "sha256": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
        "size_bytes": 88283115,
        "source_url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
    }


@pytest.mark.training
def test_dinov2_offline_build_has_linear_head_and_partial_finetuning_gradients(
    tmp_path: Path,
) -> None:
    weight_path = tmp_path / "dinov2_vits14_pretrain.pth"
    torch.save(dinov2_vits14().state_dict(), weight_path)
    contract = json.loads(
        (PROJECT_ROOT / "contracts/dinov2-vits14-lvd142m.json").read_text()
    )
    contract["artifact"]["size_bytes"] = weight_path.stat().st_size
    contract["artifact"]["sha256"] = hashlib.sha256(
        weight_path.read_bytes()
    ).hexdigest()
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))

    model, artifact = build_dinov2_initialized_model(
        weight_path,
        tuple(f"target-{index}" for index in range(12)),
        contract_path=contract_path,
    )
    assert artifact.sha256 == contract["artifact"]["sha256"]
    assert model.head.out_features == 12
    assert model.plane_gate is None
    configure_dinov2_finetuning(model)
    optimizer = build_dinov2_optimizer(
        model,
        backbone_learning_rate=8e-6,
        head_learning_rate=1e-3,
        weight_decay=1e-4,
    )

    logits = model(
        torch.zeros(1, 1, 1, 224, 224),
        torch.ones(1, 1, 1, dtype=torch.bool),
        torch.ones(1, 1, dtype=torch.bool),
    )
    assert logits.shape == (1, 12)
    logits.sum().backward()
    backbone = cast(DinoV2ViTS14, model.backbone)
    assert backbone.patch_embed.proj.weight.grad is None
    assert any(
        parameter.grad is not None for parameter in backbone.blocks[6].parameters()
    )
    assert model.head.weight.grad is not None

    early_stopping = DinoV2EarlyStopping(10, 20, 4, 0.002)
    for epoch in range(1, 14):
        early_stopping.observe(epoch, 0.5)

    checkpoint_path = tmp_path / "checkpoint.pt"
    payload: dict[str, Any] = {
        "checkpoint_schema_version": 4,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": {},
        "completed_epochs": early_stopping.completed_epochs,
        "targets": [f"target-{index}" for index in range(12)],
        "priors": [0.5] * 12,
        "preprocessing": {},
        "cohort": {},
        "partition": {},
        "initialization": {},
        "configuration": {
            "training": {
                "architecture": "dinov2_vits14",
                "weight_identity": "dinov2_vits14.LVD142M",
            }
        },
        "code": {},
        "runtime": {},
        "artifact_hashes": {},
        "rng_state": {},
        "backbone": {
            "identity": "dinov2_vits14",
            "feature_count": 384,
            "initialization_artifact_sha256": artifact.sha256,
        },
        "finetuning": _dinov2_finetuning_metadata(
            model,
            optimizer,
            backbone_learning_rate=8e-6,
            head_learning_rate=1e-3,
            weight_decay=1e-4,
        ),
        "early_stopping": early_stopping.to_record(),
        "retained_model_state_dict": model.state_dict(),
        "epoch_logs": [
            {"epoch": epoch} for epoch in range(1, early_stopping.completed_epochs + 1)
        ],
        "training_seconds": 1.0,
    }
    torch.save(payload, checkpoint_path)
    reloaded, reloaded_payload = load_combined_checkpoint(
        checkpoint_path, device=torch.device("cpu")
    )
    with torch.no_grad():
        assert torch.equal(
            reloaded(
                torch.zeros(1, 1, 1, 224, 224),
                torch.ones(1, 1, 1, dtype=torch.bool),
                torch.ones(1, 1, dtype=torch.bool),
            ),
            logits.detach(),
        )
    assert reloaded_payload["backbone"]["identity"] == "dinov2_vits14"

    official_hash = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
    payload["targets"] = list(TARGETS)
    payload["preprocessing"] = {
        "slice_count": 8,
        "lower_percentile": 1.0,
        "upper_percentile": 99.0,
        "image_size": 224,
        "shared_contract": "preprocess_series_result",
        "channel_adapter": "grayscale_repeat_to_rgb",
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
    }
    payload["cohort"] = {
        "selection_rule_version": "gold-plus-public-verdicts-v1",
        "digest": "b066a7e6cd7119f4bef330a006a86a0e08e664d7b5347a111ec781e72abe1fdc",
        "gold_study_count": 58,
        "non_gold_study_count": 4349,
        "total_study_count": 4407,
    }
    payload["partition"] = {
        "split_rule_version": "patient-hash-disjoint-v1",
        "split_seed": 20260825,
        "val_fraction": 0.1,
        "train_study_count": 3957,
        "val_study_count": 450,
    }
    payload["initialization"] = {
        "contract_sha256": hashlib.sha256(
            (PROJECT_ROOT / "contracts/dinov2-vits14-lvd142m.json").read_bytes()
        ).hexdigest(),
        "artifact_sha256": official_hash,
        "artifact_size_bytes": 88283115,
    }
    payload["configuration"] = {
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
            "paired_bootstrap_method": "paired-patient-id-nonparametric-bootstrap-v1",
            "paired_bootstrap_seed": 20260825,
            "paired_bootstrap_replicates": 10000,
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
    payload["backbone"] = {
        **payload["backbone"],
        "initialization_artifact_sha256": official_hash,
    }
    torch.save(payload, checkpoint_path)
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    verified, verified_payload = load_verified_dinov2_checkpoint(
        checkpoint_path,
        device=torch.device("cpu"),
        expected_sha256=checkpoint_hash,
        contract_path=PROJECT_ROOT / "contracts/competition.json",
    )
    assert verified.head.out_features == len(TARGETS)
    assert verified_payload["targets"] == list(TARGETS)
    assert verified_payload["finetuning"]["freeze_policy"] == {
        "frozen": [
            "patch_embed",
            "cls_token",
            "pos_embed",
            "mask_token",
            "blocks_0_to_5",
        ],
        "trainable": ["blocks_6_to_11", "norm", "classification_head"],
    }
    assert verified_payload["finetuning"]["optimizer"] == {
        "parameter_group_count": 2,
        "backbone_learning_rate": 8e-6,
        "head_learning_rate": 1e-3,
        "weight_decay": 1e-4,
    }
    assert verified_payload["early_stopping"]["stopping_epoch"] == 13
    assert verified_payload["early_stopping"]["retained_best_epoch"] == 1

    with pytest.raises(ValueError, match="spatial schema is incompatible"):
        load_verified_combined_checkpoint(
            checkpoint_path,
            device=torch.device("cpu"),
            expected_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            contract_path=PROJECT_ROOT / "contracts/competition.json",
        )

    resnet_path = tmp_path / "resnet-checkpoint.pt"
    resnet_payload = dict(payload)
    resnet_payload["checkpoint_schema_version"] = 1
    for key in (
        "backbone",
        "finetuning",
        "early_stopping",
        "retained_model_state_dict",
        "epoch_logs",
        "training_seconds",
    ):
        resnet_payload.pop(key)
    resnet_payload["model_state_dict"] = _new_uninitialized_model(
        len(TARGETS)
    ).state_dict()
    resnet_payload["configuration"] = {
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
    torch.save(resnet_payload, resnet_path)
    with pytest.raises(ValueError, match="DINOv2 checkpoint schema"):
        load_verified_dinov2_checkpoint(
            resnet_path,
            device=torch.device("cpu"),
            expected_sha256=hashlib.sha256(resnet_path.read_bytes()).hexdigest(),
            contract_path=PROJECT_ROOT / "contracts/competition.json",
        )

    payload["targets"] = list(reversed(TARGETS))
    torch.save(payload, checkpoint_path)
    with pytest.raises(ValueError, match="target order"):
        load_verified_dinov2_checkpoint(
            checkpoint_path,
            device=torch.device("cpu"),
            expected_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            contract_path=PROJECT_ROOT / "contracts/competition.json",
        )

    payload["backbone"] = {
        **payload["backbone"],
        "identity": "torchvision.models.resnet18",
    }
    torch.save(payload, checkpoint_path)
    with pytest.raises(ValueError, match="backbone metadata"):
        load_combined_checkpoint(checkpoint_path, device=torch.device("cpu"))

    weight_path.write_bytes(weight_path.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_dinov2_initialization_artifact(weight_path, contract_path=contract_path)


@pytest.mark.training
def test_gradient_accumulation_matches_one_effective_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MeanBackbone(nn.Module):
        def forward(self, values: Tensor) -> Tensor:
            return values.mean(dim=(2, 3))[:, :2]

    studies = tuple(
        GoldStudy(f"study-{index}", (cast(Label, index % 2),)) for index in range(4)
    )
    indexed = {study.study_uid: IndexedStudy(study.study_uid, ()) for study in studies}

    def decode(
        study: IndexedStudy, preprocessing: object, *, labels: object
    ) -> DecodedStudy:
        del preprocessing
        typed_labels = tuple(cast(Sequence[Label], labels))
        return DecodedStudy(
            pixels=torch.full((1, 1, 2, 2), float(study.study_uid[-1]) / 4),
            slice_mask=torch.ones(1, 1, dtype=torch.bool),
            series_mask=torch.ones(1, dtype=torch.bool),
            labels=torch.tensor(typed_labels, dtype=torch.float32),
            label_mask=torch.ones(len(typed_labels), dtype=torch.bool),
            valid_slice_count=1,
            decode_error_counts={},
        )

    monkeypatch.setattr(training_module, "decode_study", decode)
    full = GoldResNet18(MeanBackbone(), feature_count=2, target_count=1)
    accumulated = GoldResNet18(MeanBackbone(), feature_count=2, target_count=1)
    accumulated.load_state_dict(full.state_dict())
    for model in (full, accumulated):
        model.imagenet_mean.zero_()
        model.imagenet_std.fill_(1)
    common: dict[str, Any] = {
        "indexed": indexed,
        "studies": studies,
        "preprocessing": PreprocessingConfig(1, 0, 100, 224),
        "seed": 7,
        "mixed_precision": False,
        "device": torch.device("cpu"),
        "epoch": 0,
    }
    full_record = train_epoch(
        model=full,
        optimizer=torch.optim.SGD(full.parameters(), lr=0.1),
        scaler=torch.GradScaler("cuda", enabled=False),
        batch_size=4,
        gradient_accumulation_steps=1,
        **common,
    )
    accumulated_record = train_epoch(
        model=accumulated,
        optimizer=torch.optim.SGD(accumulated.parameters(), lr=0.1),
        scaler=torch.GradScaler("cuda", enabled=False),
        batch_size=1,
        gradient_accumulation_steps=4,
        **common,
    )

    assert full_record["optimizer_step_count"] == 1
    assert accumulated_record["optimizer_step_count"] == 1
    for full_parameter, accumulated_parameter in zip(
        full.parameters(), accumulated.parameters(), strict=True
    ):
        assert torch.allclose(full_parameter, accumulated_parameter, atol=1e-7)


def test_dinov2_decision_requires_effect_size_evidence_and_no_severe_regression() -> (
    None
):
    config = load_combined_baseline_config(PROJECT_ROOT / "configs/dinov2.toml")
    comparison: dict[str, Any] = {
        "delta": {
            "official_macro_auc": 0.006,
            "per_target": {target: {"delta_roc_auc": 0.001} for target in TARGETS},
        },
        "bootstrap": {
            "uncertainty": {"official_macro_auc_delta": {"fraction_positive": 0.96}}
        },
    }
    assert _dinov2_decision(comparison, config)["result"] == "keep"

    comparison["delta"]["per_target"][TARGETS[0]]["delta_roc_auc"] = -0.021
    rejected = _dinov2_decision(comparison, config)
    assert rejected["result"] == "reject"
    assert rejected["generic_backbone_recommendation"] == "stop"


def test_hidden_inference_estimate_is_only_reported_for_a_retained_candidate() -> None:
    assert (
        _retained_hidden_test_inference_estimate(
            decision={"result": "reject"},
            validation_study_count=10,
            validation_seconds=5.0,
            hidden_test_study_count=20,
        )
        is None
    )
    assert (
        _retained_hidden_test_inference_estimate(
            decision={"result": "keep"},
            validation_study_count=10,
            validation_seconds=5.0,
            hidden_test_study_count=20,
        )
        == 10.0
    )


def test_dinov2_model_uses_ordinary_two_stage_mean_reduction_without_gating() -> None:
    class DummyBackbone(nn.Module):
        def forward(self, values: Tensor) -> Tensor:
            flat = values.mean(dim=(1, 2, 3), keepdim=True).flatten(1)
            return flat.expand(-1, DINOV2_VITS14_FEATURE_COUNT)

    model = GoldResNet18(
        DummyBackbone(),
        feature_count=DINOV2_VITS14_FEATURE_COUNT,
        target_count=len(TARGETS),
        aggregation="mean",
    )
    assert model.aggregation == "mean"
    assert model.plane_gate is None

    pixels = torch.ones((1, 2, 2, 224, 224), dtype=torch.float32)
    slice_mask = torch.ones((1, 2, 2), dtype=torch.bool)
    series_mask = torch.ones((1, 2), dtype=torch.bool)

    output = model(pixels, slice_mask, series_mask)
    assert output.shape == (1, len(TARGETS))


@pytest.mark.training
def test_train_epoch_skips_batches_without_known_labels_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MeanBackbone(nn.Module):
        def forward(self, values: Tensor) -> Tensor:
            return values.mean(dim=(2, 3))[:, :2]

    studies = (
        GoldStudy("study-0", (cast(Label, 1),)),
        GoldStudy("study-1", (cast(Label, None),)),
    )
    indexed = {study.study_uid: IndexedStudy(study.study_uid, ()) for study in studies}

    def decode(
        study: IndexedStudy, preprocessing: object, *, labels: object
    ) -> DecodedStudy:
        del preprocessing
        typed_labels = tuple(cast(Sequence[Label], labels))
        return DecodedStudy(
            pixels=torch.full((1, 1, 2, 2), 1.0),
            slice_mask=torch.ones(1, 1, dtype=torch.bool),
            series_mask=torch.ones(1, dtype=torch.bool),
            labels=torch.tensor([0.0 if l is None else float(l) for l in typed_labels]),
            label_mask=torch.tensor(
                [l is not None for l in typed_labels], dtype=torch.bool
            ),
            valid_slice_count=1,
            decode_error_counts={},
        )

    monkeypatch.setattr(training_module, "decode_study", decode)
    model = GoldResNet18(MeanBackbone(), feature_count=2, target_count=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.GradScaler("cuda", enabled=False)

    record = train_epoch(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        indexed=indexed,
        studies=studies,
        preprocessing=PreprocessingConfig(1, 0, 100, 224),
        batch_size=1,
        seed=42,
        mixed_precision=False,
        device=torch.device("cpu"),
        epoch=0,
        gradient_accumulation_steps=1,
    )
    assert record["optimized_study_count"] == 1
    assert record["optimizer_step_count"] == 1


def test_discover_combined_checkpoint_allows_other_checkpoints_when_one_matches(
    tmp_path: Path,
) -> None:
    control_dir = tmp_path / "control"
    control_dir.mkdir()
    control_ckpt = control_dir / "checkpoint.pt"
    control_ckpt.write_bytes(b"matching-control-bytes")
    expected_hash = hashlib.sha256(b"matching-control-bytes").hexdigest()

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_ckpt = other_dir / "checkpoint.pt"
    other_ckpt.write_bytes(b"different-bytes")

    found = discover_combined_checkpoint(tmp_path, expected_sha256=expected_hash)
    assert found == control_ckpt


def test_training_notebook_displays_report_summary() -> None:
    metrics = {
        "baseline": {"official_macro_auc": 0.838277},
        "candidate": {"official_macro_auc": 0.847875},
        "delta": {"official_macro_auc": 0.009598},
        "optimization_diagnostic": {"issue_30_minus_issue_29_macro_auc": 0.213695},
        "secondary_masked_bce": {"candidate": 0.493540},
        "bootstrap": {
            "uncertainty": {
                "official_macro_auc_delta": {
                    "interval": {"lower": -0.001906, "upper": 0.021668},
                    "fraction_positive": 0.9473,
                }
            }
        },
    }
    comparison = {"metrics": metrics, "decision": {"result": "reject"}}
    report = {
        "controlled_comparison": comparison,
        "checkpoint": {
            "retained_best_epoch": 7,
            "stopping_epoch": 13,
            "stop_reason": "patience_exhausted",
        },
        "training_execution": {"resource_comparison": {}},
    }
    source = (Path(__file__).parents[1] / "notebooks/train_dinov2.py").read_text()
    summary = source[source.index('comparison = report["controlled_comparison"]') :]
    output = io.StringIO()
    with redirect_stdout(output):
        exec(  # noqa: S102 - execute the repository's actual notebook display code
            summary, {"report": report, "verification": {"valid": True}, "json": json}
        )
    assert "'lower': -0.001906, 'upper': 0.021668" in output.getvalue()
    assert "Decision: reject" in output.getvalue()
