from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import Tensor, nn


def test_frozen_slot_mapping() -> None:
    from rsna_knee.sequence_slots import resolve_sequence_slot

    for plane, fluid, expected in [
        ("Axial", "Yes", 0),
        ("sagittal", "true", 1),
        (" CORONAL ", "1", 2),
        ("Sagittal", "No", 3),
        ("axial", "false", -1),
        ("unknown", "Yes", -1),
        ("sagittal", "", -1),
        ("sagittal", "garbage", -1),
    ]:
        assert resolve_sequence_slot(plane, fluid) == expected


def test_equal_slot_pooling_and_target_specific_masking() -> None:
    torch = pytest.importorskip("torch")
    from rsna_knee.sequence_slots import TargetSpecificSlotAttention

    attention = TargetSpecificSlotAttention(1, 2)
    head = torch.nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        attention.projection.weight.copy_(torch.tensor([[1.0], [-1.0]]))
        attention.projection.bias.zero_()
        head.weight.fill_(1)
    features = torch.tensor([[[1.0], [3.0], [4.0], [999.0]]])
    logits, weights = attention(
        features,
        torch.ones(1, 4, dtype=torch.bool),
        torch.tensor([[0, 0, 1, -1]]),
        head,
    )
    assert weights.shape == (1, 4, 2)
    assert torch.equal(weights[:, 2:], torch.zeros(1, 2, 2))
    torch.testing.assert_close(weights.sum(1), torch.ones(1, 2))
    torch.testing.assert_close(weights[0, 0], torch.tensor([0.11920292, 0.88079708]))
    torch.testing.assert_close(logits, torch.tensor([[3.761594, 2.238406]]))
    _, single = attention(
        features,
        torch.ones(1, 4, dtype=torch.bool),
        torch.tensor([[3, 3, -1, -1]]),
        head,
    )
    assert torch.equal(single[0, 3], torch.ones(2))
    zero_logits, zero = attention(
        features, torch.ones(1, 4, dtype=torch.bool), torch.full((1, 4), -1), head
    )
    assert torch.equal(zero, torch.zeros_like(zero))
    assert torch.isfinite(zero_logits).all()


def test_control_features_preprocessing_fallback_and_reload(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from rsna_knee.combined_supervision import (
        _combined_config_record,
        load_combined_baseline_config,
        load_combined_checkpoint,
    )
    from rsna_knee.competition_contract import load_verified_submission_contract
    from rsna_knee.sequence_slots import SLOT_SCHEMA, slot_metadata
    from rsna_knee.training import StudyClassifier

    root = Path(__file__).parents[1]
    control_config = load_combined_baseline_config(root / "configs/physical-crop.toml")
    config = load_combined_baseline_config(root / "configs/slot-attention.toml")
    assert config.preprocessing == control_config.preprocessing
    assert config.preprocessing.slice_count == 8
    assert config.preprocessing.image_size == 224
    assert config.preprocessing.physical_fov_mm == 130
    assert config.preprocessing.input_representation == "same_slice_rgb"
    assert config.preprocessing.sampling_layout == "full_span"
    assert config.architecture == control_config.architecture
    assert config.weight_identity == control_config.weight_identity
    from rsna_knee.training import _new_uninitialized_model

    targets = list(load_verified_submission_contract().target_names)
    model = _new_uninitialized_model(len(targets), aggregation="target_slot_attention")
    model.eval()
    with torch.no_grad():
        model.fallback_logits.fill_(-1.38629436112)
    pixels = torch.rand(1, 2, 8, 16, 16)
    mask = torch.ones(1, 2, 8, dtype=torch.bool)
    series_mask = mask.any(2)
    slots = torch.tensor([[0, 1]])
    with torch.no_grad():
        expected = model(pixels, mask, series_mask, slot_indices=slots)
        torch.testing.assert_close(
            model(pixels, mask, series_mask, slot_indices=slots), expected
        )
        fallback = model(pixels, mask, series_mask, slot_indices=torch.full((1, 2), -1))
        torch.testing.assert_close(fallback.sigmoid(), torch.full((1, 12), 0.2))
    payload = {
        "checkpoint_schema_version": SLOT_SCHEMA,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "scaler_state_dict": {},
        "completed_epochs": 3,
        "targets": targets,
        "priors": [0.2] * 12,
        "preprocessing": {
            **config.preprocessing.record(),
            "shared_contract": "preprocess_series_result",
            "channel_adapter": "grayscale_repeat_to_rgb",
            "imagenet_mean": [0.485, 0.456, 0.406],
            "imagenet_std": [0.229, 0.224, 0.225],
        },
        "configuration": _combined_config_record(config),
        "aggregation": slot_metadata(targets),
        "cohort": {},
        "partition": {},
        "initialization": {},
        "code": {},
        "runtime": {},
        "artifact_hashes": {},
        "rng_state": {},
    }
    checkpoint = tmp_path / "candidate.pt"
    torch.save(payload, checkpoint)
    loaded, _ = load_combined_checkpoint(checkpoint, device=torch.device("cpu"))
    loaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(pixels, mask, series_mask, slot_indices=slots),
            expected,
            atol=1e-5,
            rtol=0,
        )
    for key, value in [
        ("checkpoint_schema_version", "rsna-knee-physical-crop-v1"),
        ("targets", list(reversed(targets))),
        ("aggregation", {}),
    ]:
        bad = {**payload, key: value}
        torch.save(bad, checkpoint)
        with pytest.raises(ValueError):
            load_combined_checkpoint(checkpoint, device=torch.device("cpu"))
    # A mean checkpoint cannot acquire attention semantics by changing only its schema.
    bad = {**payload, "configuration": _combined_config_record(control_config)}
    torch.save(bad, checkpoint)
    with pytest.raises(ValueError):
        load_combined_checkpoint(checkpoint, device=torch.device("cpu"))

    # With two singleton slots and zero attention scores, #31's feature mean agrees.
    control = StudyClassifier(torch.nn.Identity(), 3, 12)
    candidate = StudyClassifier(
        torch.nn.Identity(), 3, 12, aggregation="target_slot_attention"
    )
    candidate.head.load_state_dict(control.head.state_dict())
    assert candidate.slot_attention is not None
    with torch.no_grad():
        candidate.slot_attention.projection.weight.zero_()
        candidate.slot_attention.projection.bias.zero_()

    # Identity needs flattened channel means as its encoder.
    class ChannelMean(nn.Module):
        def forward(self, value: Tensor) -> Tensor:
            return value.mean((2, 3))

    control.backbone = ChannelMean()
    candidate.backbone = ChannelMean()
    torch.testing.assert_close(
        control(pixels, mask, series_mask),
        candidate(pixels, mask, series_mask, slot_indices=slots),
    )


def test_live_metadata_decode_training_validation_and_hidden_inference(
    tmp_path: Path,
) -> None:
    import csv
    from dataclasses import dataclass

    import numpy as np
    from pydicom import dcmread
    from test_synthetic_pipeline import _write_dicom

    from rsna_knee.combined_supervision import (
        _predict_combined_submission,
        slot_evidence,
    )
    from rsna_knee.competition_contract import load_verified_submission_contract
    from rsna_knee.config import PreprocessingConfig
    from rsna_knee.indexing import build_competition_index
    from rsna_knee.submission import SubmissionInputs
    from rsna_knee.training import (
        Label,
        StudyClassifier,
        decode_study,
        load_indexed_studies,
        predict_partition,
        train_epoch,
    )

    root = tmp_path / "input"
    root.mkdir()
    # File names are deliberately the reverse of physical order; metadata alone assigns slots.
    with (root / "test_series.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "Fluid_Sensitive",
                "Fat_Suppression",
                "Anatomical_Plane",
            ]
        )
        writer.writerows(
            [
                ("mapped", "series1", "Yes", "No", "Axial"),
                ("empty", "series2", "No", "Yes", "Coronal"),
            ]
        )
    (root / "train.csv").write_text("StudyInstanceUID\n")
    (root / "test.csv").write_text("StudyInstanceUID\nmapped\nempty\n")
    contract = load_verified_submission_contract()
    with (root / "sample_submission.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["StudyInstanceUID", *contract.target_names])
        writer.writerows([(uid, *([0.5] * 12)) for uid in ("mapped", "empty")])
    for uid, series in (("mapped", "series1"), ("empty", "series2")):
        directory = root / "test_series" / uid / series
        directory.mkdir(parents=True)
        for position in range(12):
            path = directory / f"{12 - position:02}.dcm"
            _write_dicom(path, np.arange(16 * 16).reshape(16, 16) + position)
            ds = dcmread(path)
            ds.PixelSpacing = [1, 1]
            ds.ImagePositionPatient = [0, 0, position]
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            ds.save_as(path, enforce_file_format=True)
    preprocessing = PreprocessingConfig(8, 1, 99, 224, "physical_crop")
    artifact = build_competition_index(
        root,
        tmp_path / "index",
        preprocessing,
        splits=("test",),
        representative_series_per_split=0,
    )
    indexed = load_indexed_studies(
        root, artifact.artifact_dir, ("mapped", "empty"), split="test"
    )
    assert indexed["mapped"].series[0].slot_index == 0
    assert indexed["empty"].series[0].slot_index == -1
    assert [p.name for p in indexed["mapped"].series[0].paths] == [
        f"{i:02}.dcm" for i in range(12, 0, -1)
    ]
    decoded = decode_study(indexed["mapped"], preprocessing, labels=None)
    from rsna_knee.preprocessing import preprocess_series_result

    control = preprocess_series_result(indexed["mapped"].series[0].paths, preprocessing)
    assert control.source_indices == (0, 1, 3, 4, 6, 7, 9, 11)
    np.testing.assert_array_equal(decoded.pixels[0].numpy(), control.pixels)
    assert decoded.pixels.shape == (1, 8, 224, 224)

    model = StudyClassifier(
        nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten()),
        3,
        12,
        aggregation="target_slot_attention",
    )
    with torch.no_grad():
        model.fallback_logits.fill_(float(torch.logit(torch.tensor(0.25))))

    @dataclass
    class Study:
        study_uid: str
        patient_id: str
        labels: tuple[Label, ...] = (1,) * 12

    studies = [Study("mapped", "one"), Study("empty", "two")]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    training = train_epoch(
        model=model,
        optimizer=optimizer,
        scaler=torch.GradScaler("cpu", enabled=False),
        indexed=indexed,
        studies=studies,
        preprocessing=preprocessing,
        batch_size=16,
        seed=20260825,
        mixed_precision=False,
        device=torch.device("cpu"),
        epoch=0,
    )
    assert training["optimized_study_count"] == 1
    assert training["spatial_counts"]["slots:zero_available"] == 1
    prediction = predict_partition(
        model=model,
        indexed=indexed,
        studies=studies,
        preprocessing=preprocessing,
        mixed_precision=False,
        device=torch.device("cpu"),
    )
    assert prediction.probabilities[1] == (0.25,) * 12
    evidence = slot_evidence([training], prediction, contract.target_names)
    assert evidence["validation"] == {
        "counts": {
            "studies": 2,
            "unmatched_series": 1,
            "zero_available": 1,
            "pattern:1000": 1,
            "pattern:0000": 1,
            "available:0": 1,
            "available:1": 0,
            "available:2": 0,
            "available:3": 0,
            "multiplicity:0:1": 1,
            "multiplicity:0:0": 1,
            "multiplicity:1:0": 2,
            "multiplicity:2:0": 2,
            "multiplicity:3:0": 2,
        },
        "availability_rates": {
            "axial_fluid_sensitive": 0.5,
            "sagittal_fluid_sensitive": 0,
            "coronal_fluid_sensitive": 0,
            "sagittal_non_fluid_sensitive": 0,
        },
    }
    hidden = _predict_combined_submission(
        model=model,
        submission_inputs=SubmissionInputs(root, ("mapped", "empty"), contract),
        indexed=indexed,
        work_root=tmp_path / "submission",
        preprocessing=preprocessing,
        priors=[0.25] * 12,
        mixed_precision=False,
        device=torch.device("cpu"),
    )
    assert hidden.fallback_count == 1
    assert "no_available_candidate_slots" in hidden.artifacts.fallbacks_path.read_text()
    with hidden.artifacts.submission_path.open() as file:
        rows = list(csv.DictReader(file))
    for row, values in zip(rows, prediction.probabilities, strict=True):
        np.testing.assert_allclose(
            [float(row[t]) for t in contract.target_names], values, atol=1e-6
        )


def test_local_bundle_is_guarded(tmp_path: Path) -> None:
    import json

    from rsna_knee.kaggle import build_plane_gating_bundle, load_kaggle_bundle

    bundle = build_plane_gating_bundle(
        Path(__file__).parents[1],
        tmp_path,
        slot_attention=True,
        kernel_id="plthon/rsna-knee-slot-attention",
        gpu_approved=False,
    )
    load_kaggle_bundle(tmp_path)
    metadata = json.loads(bundle.kernel_metadata_path.read_text())
    assert metadata["enable_gpu"] == "false"
    assert "plthon/rsna-knee-physical-crop" in metadata["kernel_sources"]
    assert bundle is not None


def test_cached_control_predictions_require_provenance_and_aligned_labels(
    tmp_path: Path,
) -> None:
    import csv
    import hashlib
    import json
    from dataclasses import dataclass

    from rsna_knee.combined_supervision import (
        PHYSICAL_CROP_CHECKPOINT_SHA256,
        PHYSICAL_CROP_COMMIT,
        _expected_physical_crop_configuration,
        _validation_membership_digest,
        load_physical_crop_validation_predictions,
    )
    from rsna_knee.training import GoldStudy

    @dataclass(frozen=True)
    class Study(GoldStudy):
        patient_id: str

    studies = [Study("one", (1, None), "patient")]
    path = tmp_path / "validation-predictions.csv"
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "StudyInstanceUID",
                "PatientID",
                "a__label",
                "a__baseline",
                "a__physical_crop",
                "b__label",
                "b__baseline",
                "b__physical_crop",
            ]
        )
        writer.writerow(["one", "patient", 1, 0.1, 0.8, "", 0.3, 0.4])
    report = {
        "checkpoint": {"sha256": PHYSICAL_CROP_CHECKPOINT_SHA256},
        "implementation": {"git_commit": PHYSICAL_CROP_COMMIT},
        "configuration": _expected_physical_crop_configuration(),
        "controlled_comparison": {
            "validation_predictions_sha256": hashlib.sha256(
                path.read_bytes()
            ).hexdigest(),
            "validation_membership_digest": _validation_membership_digest(studies),
        },
    }
    (tmp_path / "training-report.json").write_text(json.dumps(report))
    prediction = load_physical_crop_validation_predictions(
        tmp_path, studies, ("a", "b")
    )
    assert prediction is not None and prediction.probabilities == ((0.8, 0.4),)
    with pytest.raises(ValueError, match="labels"):
        load_physical_crop_validation_predictions(
            tmp_path, [Study("one", (0, None), "patient")], ("a", "b")
        )
    path.write_text(path.read_text().replace("0.8", "0.9"))
    with pytest.raises(ValueError, match="provenance"):
        load_physical_crop_validation_predictions(tmp_path, studies, ("a", "b"))


def test_attention_masks_padding_and_backpropagates_only_available_slots() -> None:
    from rsna_knee.sequence_slots import TargetSpecificSlotAttention

    attention = TargetSpecificSlotAttention(1, 2)
    head = nn.Linear(1, 2)
    features = torch.tensor(
        [[[1.0], [3.0], [1000.0]], [[1.0], [2.0], [3.0]]], requires_grad=True
    )
    logits, weights = attention(
        features,
        torch.tensor([[True, True, False], [True, True, True]]),
        torch.tensor([[0, 1, 2], [-1, -1, -1]]),
        head,
    )
    logits.sum().backward()
    assert features.grad is not None
    assert features.grad[0, 2].item() == 0
    assert torch.equal(features.grad[1], torch.zeros(3, 1))
    assert attention.projection.weight.grad is not None
    assert torch.isfinite(attention.projection.weight.grad).all()
    assert not weights[:, 2:].any()
    with pytest.raises(ValueError, match="Slot indices"):
        attention(
            features, torch.ones(2, 3, dtype=torch.bool), torch.full((2, 3), 4), head
        )
