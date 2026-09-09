from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pydicom import dcmread
from test_synthetic_pipeline import _write_dicom

from rsna_knee.config import PreprocessingConfig
from rsna_knee.preprocessing import (
    centered_crop_pad,
    physical_crop_shape,
    preprocess_series_result,
)


@pytest.mark.parametrize(
    ("spacing", "expected"),
    [
        ([1, 1], (130, 130)),
        ([2, 0.5], (65, 260)),
        ([4, 8], (32, 16)),
        ([130 / 33.5, 130 / 34.5], (34, 34)),
    ],
)
def test_physical_crop_dimensions(
    spacing: list[float], expected: tuple[int, int]
) -> None:
    assert physical_crop_shape(spacing) == (expected, None)


@pytest.mark.parametrize(
    ("spacing", "reason"),
    [
        (None, "missing_pixel_spacing"),
        ([0, 1], "nonpositive_pixel_spacing"),
        ([-1, 1], "nonpositive_pixel_spacing"),
        ([float("nan"), 1], "nonfinite_pixel_spacing"),
        ([1, float("inf")], "nonfinite_pixel_spacing"),
        (["bad", 1], "malformed_pixel_spacing"),
        ([1], "malformed_pixel_spacing"),
        ("1\\1", "malformed_pixel_spacing"),
    ],
)
def test_spacing_fallback_reasons(spacing: object, reason: str) -> None:
    assert physical_crop_shape(spacing) == (None, reason)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((2, 2), [[5, 6], [9, 10]]),
        ((3, 3), [[0, 1, 2], [4, 5, 6], [8, 9, 10]]),
        (
            (5, 5),
            [
                [0, 1, 2, 3, 0],
                [4, 5, 6, 7, 0],
                [8, 9, 10, 11, 0],
                [12, 13, 14, 15, 0],
                [0, 0, 0, 0, 0],
            ],
        ),
    ],
)
def test_centered_crop_coordinates(
    shape: tuple[int, int], expected: list[list[int]]
) -> None:
    np.testing.assert_array_equal(
        centered_crop_pad(np.arange(16).reshape(4, 4), shape), expected
    )


def test_odd_source_and_symmetric_neutral_padding() -> None:
    image = np.arange(25).reshape(5, 5)
    np.testing.assert_array_equal(centered_crop_pad(image, (2, 2)), [[6, 7], [11, 12]])
    padded = centered_crop_pad(np.ones((2, 3), dtype=np.float32), (6, 7))
    assert padded.shape == (6, 7)
    np.testing.assert_array_equal(padded[2:4, 2:5], 1)
    assert padded.sum() == 6
    assert not padded[:2].any() and not padded[:, :2].any()


def test_dicom_crop_preserves_percentiles_sampling_and_fallback(tmp_path: Path) -> None:
    paths = []
    for i in range(12):
        path = tmp_path / f"{i}.dcm"
        _write_dicom(path, np.arange(200 * 160).reshape(200, 160) + i)
        ds = dcmread(path)
        if i != 0:
            ds.PixelSpacing = [1, 2]
        ds.save_as(path, enforce_file_format=True)
        paths.append(path)
    baseline_config = PreprocessingConfig(8, 1, 99)
    candidate_config = replace(baseline_config, spatial_mode="physical_crop")
    baseline = preprocess_series_result(paths, baseline_config)
    candidate = preprocess_series_result(paths, candidate_config)
    assert (
        candidate.source_indices == baseline.source_indices == (0, 1, 3, 4, 6, 7, 9, 11)
    )
    assert (candidate.lower_value, candidate.upper_value) == (
        baseline.lower_value,
        baseline.upper_value,
    )
    assert candidate.pixels.shape == (8, 224, 224)
    assert candidate.mask.all()
    assert candidate.spatial_evidence[0].fallback_reason == "missing_pixel_spacing"
    np.testing.assert_array_equal(candidate.pixels[0], baseline.pixels[0])
    assert {e.crop_shape for e in candidate.spatial_evidence[1:]} == {(130, 65)}
    assert not np.array_equal(candidate.pixels[1], baseline.pixels[1])
    np.testing.assert_array_equal(
        preprocess_series_result(paths, baseline_config).pixels, baseline.pixels
    )


def test_crop_config_preserves_frozen_baseline_and_rejects_other_interventions() -> (
    None
):
    pytest.importorskip("torch")
    from rsna_knee.combined_supervision import load_combined_baseline_config

    root = Path(__file__).parents[1]
    candidate = load_combined_baseline_config(root / "configs/physical-crop.toml")
    control = load_combined_baseline_config(root / "configs/combined-baseline.toml")
    assert (
        replace(candidate.preprocessing, spatial_mode="whole_image")
        == control.preprocessing
    )
    assert (
        candidate.aggregation == "mean"
        and candidate.architecture == control.architecture
    )
    assert candidate.preprocessing.record()["physical_fov_mm"] == 130.0
    assert (
        candidate.preprocessing.record()["spacing_fallback"] == "whole_image_resize_pad"
    )
    assert set(control.preprocessing.record()) == {
        "slice_count",
        "image_size",
        "lower_percentile",
        "upper_percentile",
    }


def test_crop_checkpoint_reload_and_schema_target_protection(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from rsna_knee.combined_supervision import (
        _combined_config_record,
        load_combined_baseline_config,
        load_combined_checkpoint,
    )
    from rsna_knee.competition_contract import load_verified_submission_contract
    from rsna_knee.training import (
        IndexedSeries,
        IndexedStudy,
        _new_uninitialized_model,
        decode_study,
    )

    config = load_combined_baseline_config(
        Path(__file__).parents[1] / "configs/physical-crop.toml"
    )
    targets = list(load_verified_submission_contract().target_names)
    model = _new_uninitialized_model(len(targets))
    model.eval()
    path = tmp_path / "slice.dcm"
    _write_dicom(path, np.arange(160 * 160).reshape(160, 160))
    ds = dcmread(path)
    ds.PixelSpacing = [1, 1]
    ds.save_as(path, enforce_file_format=True)
    decoded = decode_study(
        IndexedStudy("study", (IndexedSeries(0, (path,)),)),
        config.preprocessing,
        labels=None,
    )
    assert decoded.spatial_counts["slices_crop"] == 1
    assert decoded.spatial_counts["series_crop"] == 1
    assert decoded.spatial_counts["studies_crop"] == 1
    assert model.aggregation == "mean" and model.plane_gate is None
    payload = {
        "checkpoint_schema_version": "rsna-knee-physical-crop-v1",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "scaler_state_dict": {},
        "completed_epochs": 3,
        "targets": targets,
        "priors": [0.5] * len(targets),
        "preprocessing": {
            **config.preprocessing.record(),
            "shared_contract": "preprocess_series_result",
            "channel_adapter": "grayscale_repeat_to_rgb",
            "imagenet_mean": [0.485, 0.456, 0.406],
            "imagenet_std": [0.229, 0.224, 0.225],
        },
        "configuration": _combined_config_record(config),
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
    loaded, metadata = load_combined_checkpoint(checkpoint, device=torch.device("cpu"))
    loaded.eval()
    inputs = (
        decoded.pixels.unsqueeze(0),
        decoded.slice_mask.unsqueeze(0),
        decoded.series_mask.unsqueeze(0),
    )
    with torch.no_grad():
        torch.testing.assert_close(loaded(*inputs), model(*inputs), atol=1e-5, rtol=0)
    assert metadata["preprocessing"]["physical_fov_mm"] == 130.0
    payload["checkpoint_schema_version"] = 1
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="spatial identity"):
        load_combined_checkpoint(checkpoint, device=torch.device("cpu"))
    payload["checkpoint_schema_version"] = "rsna-knee-physical-crop-v1"
    payload["targets"] = list(reversed(targets))
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="target order"):
        load_combined_checkpoint(checkpoint, device=torch.device("cpu"))


def test_crop_fallback_evidence_aggregates_mixed_studies(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from rsna_knee.combined_supervision import spatial_summary
    from rsna_knee.training import IndexedSeries, IndexedStudy, decode_study

    paths = [tmp_path / "missing.dcm", tmp_path / "valid.dcm"]
    for path in paths:
        _write_dicom(path, np.ones((10, 10)))
    ds = dcmread(paths[1])
    ds.PixelSpacing = [1, 1]
    ds.save_as(paths[1], enforce_file_format=True)
    result = decode_study(
        IndexedStudy("study", (IndexedSeries(0, tuple(paths)),)),
        PreprocessingConfig(8, 1, 99, 224, "physical_crop"),
        labels=None,
    )
    summary = spatial_summary(result.spatial_counts)
    assert summary["slices"] == {
        "processed": 2,
        "crop_fraction": 0.5,
        "fallback_fraction": 0.5,
    }
    assert summary["series"] == {
        "processed": 1,
        "crop_fraction": 1.0,
        "fallback_fraction": 1.0,
    }
    assert result.spatial_counts["reason:missing_pixel_spacing"] == 1
    assert result.spatial_counts["studies_fallback"] == 1
    assert not result.pixels.any()
    missing = decode_study(
        IndexedStudy("undecodable", (IndexedSeries(0, (tmp_path / "absent.dcm",)),)),
        PreprocessingConfig(8, 1, 99, 224, "physical_crop"),
        labels=None,
    )
    assert spatial_summary(missing.spatial_counts)["studies"] == {
        "processed": 0,
        "crop_fraction": None,
        "fallback_fraction": None,
    }


def test_physical_crop_bundle_is_local_unapproved_training_only(tmp_path: Path) -> None:
    import json

    from rsna_knee.kaggle import build_plane_gating_bundle

    bundle = build_plane_gating_bundle(
        Path(__file__).parents[1],
        tmp_path / "bundle",
        kernel_id="plthon/rsna-knee-physical-crop",
        physical_crop=True,
    )
    metadata = json.loads(bundle.kernel_metadata_path.read_text())
    assert metadata["enable_gpu"] == "false"
    source = bundle.notebook_path.read_text()
    assert "run_test_inference=False" in source
    assert "physical_crop" in source


@pytest.mark.parametrize(
    "spacing", ([0, 1], [-1, 1], [float("nan"), 1], [float("inf"), 1], ["bad", "1"])
)
def test_invalid_dicom_spacing_uses_recorded_unchanged_fallback(
    tmp_path: Path, spacing: list[object]
) -> None:
    path = tmp_path / "invalid-spacing.dcm"
    _write_dicom(path, np.arange(60).reshape(6, 10))
    ds = dcmread(path)
    # LO permits a deliberately malformed spacing value in this synthetic file.
    ds.add_new(0x00280030, "LO", [str(value) for value in spacing])
    ds.save_as(path, enforce_file_format=True)
    baseline = preprocess_series_result([path], PreprocessingConfig(8, 1, 99))
    candidate = preprocess_series_result(
        [path], PreprocessingConfig(8, 1, 99, 224, "physical_crop")
    )
    assert candidate.mask[0]
    assert candidate.spatial_evidence[0].fallback_reason is not None
    np.testing.assert_array_equal(candidate.pixels, baseline.pixels)


@pytest.mark.parametrize(
    ("old", "wrong"),
    [
        ("verify_physical_crop_artifact", "verify_plane_gating_artifact"),
        ('config.aggregation != "mean"', 'config.aggregation != "target_plane_gate"'),
    ],
)
def test_crop_bundle_rejects_other_experiment_safeguards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, old: str, wrong: str
) -> None:
    from rsna_knee.kaggle import build_plane_gating_bundle

    original_read = Path.read_text

    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        text = original_read(path)  # source fixture boundary
        return (
            text.replace(old, wrong) if path.name == "train_physical_crop.py" else text
        )

    monkeypatch.setattr(Path, "read_text", read_text)
    with pytest.raises(ValueError, match="safeguard"):
        build_plane_gating_bundle(
            Path(__file__).parents[1],
            tmp_path / "bundle",
            kernel_id="plthon/rsna-knee-physical-crop",
            physical_crop=True,
        )


def test_physical_crop_scoring_bundle_pins_retained_checkpoint(tmp_path: Path) -> None:
    import json

    from rsna_knee.kaggle import build_physical_crop_scoring_bundle, load_kaggle_bundle

    bundle = build_physical_crop_scoring_bundle(
        Path(__file__).parents[1], tmp_path / "scoring"
    )
    load_kaggle_bundle(bundle.root)
    metadata = json.loads(bundle.kernel_metadata_path.read_text())
    assert metadata["kernel_sources"] == ["plthon/rsna-knee-physical-crop"]
    assert metadata["dataset_sources"] == []
    assert metadata["enable_gpu"] == "false"
    notebook = json.loads(bundle.notebook_path.read_text())
    source = "\n".join(
        cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        for cell in notebook["cells"]
    )
    assert "55e9da7854d284a91c2569e25c799281d09301bae61dc5adcecebbd17209f013" in source
    assert "physical_crop=True" in source
    assert 'aggregation="mean"' in source
    assert "run_combined_baseline" not in source
    assert "train_epoch" not in source


def test_scoring_notebook_selects_identical_checkpoint_copies_and_rejects_wrong_hash(
    tmp_path: Path,
) -> None:
    import ast
    import hashlib

    path = Path(__file__).parents[1] / "notebooks/score_physical_crop.py"
    tree = ast.parse(path.read_text())
    # Exercise the notebook's checkpoint-selection cell with synthetic output files.
    selection: list[ast.stmt] = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "checkpoint_path"
                for target in node.targets
            )
        )
        or (
            isinstance(node, ast.If)
            and ast.unparse(node.test) == "checkpoint_path is None"
        )
    ]
    compiled = compile(ast.Module(body=selection, type_ignores=[]), str(path), "exec")
    root_copy = tmp_path / "checkpoint.pt"
    nested_copy = tmp_path / "nested/checkpoint.pt"
    nested_copy.parent.mkdir()
    root_copy.write_bytes(b"retained")
    nested_copy.write_bytes(b"retained")
    namespace = {
        "input_root": tmp_path,
        "hashlib": hashlib,
        "EXPECTED_CHECKPOINT_SHA256": hashlib.sha256(b"retained").hexdigest(),
    }
    exec(compiled, namespace)  # noqa: S102 - execute only the trusted repository selection cell
    assert namespace["checkpoint_path"] in {root_copy, nested_copy}
    root_copy.write_bytes(b"wrong")
    nested_copy.write_bytes(b"wrong")
    with pytest.raises(FileNotFoundError, match="missing or changed"):
        exec(compiled, namespace)  # noqa: S102 - execute only the trusted repository selection cell
