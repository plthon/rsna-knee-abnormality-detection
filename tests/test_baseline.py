import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

import torch
from torch import Tensor, nn

import rsna_knee.baseline as baseline_module
from rsna_knee.baseline import (
    _duration_estimate_from_issue_8,
    _predict_partition,
    _train_epoch,
    load_baseline_training_config,
)
from rsna_knee.holdout import BOOTSTRAP_METHOD_VERSION, PartitionedGoldStudy
from rsna_knee.training import GoldResNet18, IndexedStudy, _DecodedStudy

PROJECT_ROOT = Path(__file__).parents[1]


def test_baseline_configuration_reuses_model_contract_and_freezes_uncertainty() -> None:
    config = load_baseline_training_config(
        PROJECT_ROOT / "configs" / "kaggle-baseline.toml"
    )

    assert config.preprocessing.image_size == 224
    assert config.preprocessing.slice_count == 8
    assert config.architecture == "torchvision.models.resnet18"
    assert config.weight_identity == "ResNet18_Weights.IMAGENET1K_V1"
    assert config.epochs == 10
    assert config.bootstrap.method_version == BOOTSTRAP_METHOD_VERSION
    assert config.bootstrap.seed == 20260812
    assert config.bootstrap.replicate_count == 10000
    assert config.bootstrap.confidence_level == 0.95


def test_baseline_configuration_rejects_uncertainty_method_drift(
    tmp_path: Path,
) -> None:
    source = (PROJECT_ROOT / "configs" / "kaggle-baseline.toml").read_text()
    path = tmp_path / "config.toml"
    path.write_text(source.replace(BOOTSTRAP_METHOD_VERSION, "different-version"))

    with pytest.raises(ValueError, match="shared PatientID bootstrap"):
        load_baseline_training_config(path)


def test_public_gold_contract_still_declares_candidate_witness() -> None:
    contract = json.loads(
        (PROJECT_ROOT / "contracts" / "gold-supervision.json").read_text()
    )
    evaluation = contract["evaluation_eligibility"]
    assert evaluation["decision"] == "Candidate"
    assert evaluation["assignment_digest"]["value"] == (
        "8dccc31852a3e26ec677062a289301004fdce29eb64a556bea8922dfd2bb263d"
    )


def test_baseline_consumes_approved_issue_8_epoch_duration_before_predictions() -> None:
    config = load_baseline_training_config(
        PROJECT_ROOT / "configs" / "kaggle-baseline.toml"
    )
    training_contract = json.loads(
        (PROJECT_ROOT / "contracts" / "gold-training.json").read_text()
    )
    supervision = training_contract["supervision"]
    cohort = cast(
        Any,
        SimpleNamespace(
            digest=supervision["membership_label_digest"],
            target_names=tuple(supervision["target_order"]),
            studies=tuple(range(supervision["cohort_size"])),
        ),
    )
    partition = cast(
        Any,
        SimpleNamespace(per_partition={"train": {"study_count": 54}}),
    )

    estimate = _duration_estimate_from_issue_8(
        cohort=cohort,
        partition=partition,
        config=config,
    )

    assert estimate["source_issue"] == 8
    assert estimate["approved_gpu"] is True
    assert estimate["reviewed_before_predictions"] is True
    assert estimate["source_epoch_study_count"] == 58
    assert estimate["target_train_study_count"] == 54
    assert estimate["estimated_complete_epoch_seconds"] == pytest.approx(
        58.533929398000055 * 54 / 58
    )


class _MeanBackbone(nn.Module):
    def forward(self, values: Tensor) -> Tensor:
        means = values.mean(dim=(1, 2, 3))
        return torch.stack((means, means * 2), dim=1)


def test_baseline_epoch_and_holdout_prediction_run_on_synthetic_cpu_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        load_baseline_training_config(
            PROJECT_ROOT / "configs" / "kaggle-baseline.toml"
        ),
        mixed_precision=False,
        epochs=1,
    )
    studies = (
        PartitionedGoldStudy("study-a", "patient-a", (0, 1), "train"),
        PartitionedGoldStudy("study-b", "patient-b", (1, 0), "train"),
    )
    indexed = {
        study.study_uid: IndexedStudy(study_uid=study.study_uid, series=())
        for study in studies
    }

    def decode(
        study: IndexedStudy, preprocessing: object, *, labels: object
    ) -> _DecodedStudy:
        del preprocessing
        value = 0.2 if study.study_uid == "study-a" else 0.8
        typed_labels = cast(tuple[int | None, ...], labels)
        return _DecodedStudy(
            pixels=torch.full((1, 1, 2, 2), value),
            slice_mask=torch.ones((1, 1), dtype=torch.bool),
            series_mask=torch.ones(1, dtype=torch.bool),
            labels=torch.tensor(
                [0.0 if item is None else item for item in typed_labels],
                dtype=torch.float32,
            ),
            label_mask=torch.tensor(
                [item is not None for item in typed_labels], dtype=torch.bool
            ),
            valid_slice_count=1,
            decode_error_counts={},
        )

    monkeypatch.setattr(baseline_module, "_decode_study", decode)
    model = GoldResNet18(_MeanBackbone(), feature_count=2, target_count=2)
    model.imagenet_mean.zero_()
    model.imagenet_std.fill_(1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.GradScaler("cuda", enabled=False)

    record = _train_epoch(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        indexed=indexed,
        studies=studies,
        config=config,
        device=torch.device("cpu"),
        epoch=1,
    )
    predictions = _predict_partition(
        model=model,
        indexed=indexed,
        studies=studies,
        preprocessing=config.preprocessing,
        mixed_precision=False,
        device=torch.device("cpu"),
    )

    assert record["optimized_study_count"] == 2
    assert record["optimizer_step_count"] == 2
    assert record["valid_slice_count"] == 2
    assert len(predictions.probabilities) == 2
    assert all(0 <= value <= 1 for row in predictions.probabilities for value in row)
