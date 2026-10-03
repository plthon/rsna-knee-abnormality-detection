from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from dicom_support import write_ordered_series
from pydicom import dcmread

from rsna_knee.config import PreprocessingConfig
from rsna_knee.preprocessing import clipped_window_indices, preprocess_series_result


def test_frozen_clipped_window_centers() -> None:
    # Floor both fractional endpoints and linspace positions; clamp to [1,n-2].
    for length, expected in [
        (0, ()),
        (1, (0,)),
        (2, (0, 1)),
        (3, (1,)),
        (4, (1, 2)),
        (5, (1, 2, 3)),
        (10, (1, 4, 8)),
        (11, (1, 5, 9)),
        (20, (1, 9, 17)),
        (21, (2, 10, 18)),
    ]:
        assert clipped_window_indices(length) == expected


CONFIG = PreprocessingConfig(
    3,
    1,
    99,
    224,
    "physical_crop",
    input_representation="adjacent_slice_2_5d",
    sampling_layout="central_80_three_anchor",
)


def test_clipped_triplets_share_center_statistics_and_crop_every_source(
    tmp_path: Path,
) -> None:
    paths = write_ordered_series(tmp_path, list(range(21)))
    result = preprocess_series_result(paths, CONFIG)
    assert result.source_indices == (2, 10, 18)
    assert result.clipped_window == (2, 18)
    assert result.source_triplets == ((1, 2, 3), (9, 10, 11), (17, 18, 19))
    assert (result.lower_value, result.upper_value) == (2, 18)
    assert result.pixels.shape == (3, 3, 224, 224)
    assert result.mask.tolist() == [True, True, True]
    np.testing.assert_allclose(
        result.pixels[:, :, 112, 112],
        [[0, 0, 1 / 16], [7 / 16, 8 / 16, 9 / 16], [15 / 16, 1, 1]],
    )
    assert result.preprocessing_counts["encoder_inputs"] == 3
    assert result.preprocessing_counts["unique_source_slices_decoded"] == 9
    assert len(result.spatial_evidence) == 9
    assert all(item.crop_shape == (130, 130) for item in result.spatial_evidence)
    assert result.preprocessing_counts["short_series"] == 0
    assert result.preprocessing_counts["boundary_clamp_previous"] == 0
    assert result.preprocessing_counts["boundary_clamp_next"] == 0
    # Changing an unrequested slice cannot affect anchors or normalization.
    paths[5].write_bytes(b"unrequested")
    np.testing.assert_array_equal(
        preprocess_series_result(paths, CONFIG).pixels, result.pixels
    )


@pytest.mark.parametrize(
    "length,valid,duplicates,clamps",
    [(1, 1, 0, 1), (2, 2, 0, 1), (3, 1, 2, 0), (4, 2, 1, 0)],
)
def test_short_series_deduplicates_and_masks_padding(
    tmp_path: Path,
    length: int,
    valid: int,
    duplicates: int,
    clamps: int,
) -> None:
    result = preprocess_series_result(
        write_ordered_series(tmp_path, list(range(length))), CONFIG
    )
    assert result.mask.sum() == valid
    assert result.pixels.shape == (3, 3, 224, 224)
    assert result.preprocessing_counts["short_series"] == 1
    assert result.preprocessing_counts["duplicate_anchors_removed"] == duplicates
    assert result.preprocessing_counts["boundary_clamp_previous"] == clamps
    assert result.preprocessing_counts["boundary_clamp_next"] == clamps
    assert result.preprocessing_counts["unique_source_slices_decoded"] == length


@pytest.mark.parametrize(
    "failed,slot,side", [(9, 1, "previous"), (11, 1, "next"), (10, 1, "center")]
)
def test_clipped_failure_policy(
    tmp_path: Path, failed: int, slot: int, side: str
) -> None:
    paths = write_ordered_series(tmp_path, list(range(21)))
    paths[failed].write_bytes(b"broken")
    result = preprocess_series_result(paths, CONFIG)
    assert result.source_triplets[slot] == (9, 10, 11)
    assert len(result.failures) == 1
    assert result.failures[0].path == paths[failed]
    if side == "center":
        assert result.mask.tolist() == [True, False, True]
        assert result.preprocessing_counts["unique_source_slices_attempted"] == 7
    else:
        np.testing.assert_array_equal(
            result.pixels[slot, 0 if side == "previous" else 2], result.pixels[slot, 1]
        )
        assert (
            result.preprocessing_counts[
                f"neighbor_substitution_{side}:pixel_decode_failed"
            ]
            == 1
        )
        assert result.preprocessing_counts["unique_source_slices_attempted"] == 9


def test_clipped_all_centers_fail_without_neighbor_search(tmp_path: Path) -> None:
    paths = write_ordered_series(tmp_path, list(range(21)))
    for center in (2, 10, 18):
        paths[center].write_bytes(b"broken")
    result = preprocess_series_result(paths, CONFIG)
    assert not result.mask.any()
    assert result.lower_value is None
    assert result.preprocessing_counts["unique_source_slices_attempted"] == 3
    assert result.source_indices == (2, 10, 18)


def test_clipped_neighbor_spacing_fallback(tmp_path: Path) -> None:
    paths = write_ordered_series(tmp_path, list(range(21)))
    dataset = dcmread(paths[9])
    del dataset.PixelSpacing
    dataset.save_as(paths[9], enforce_file_format=True)
    result = preprocess_series_result(paths, CONFIG)
    evidence = {item.source_index: item for item in result.spatial_evidence}
    assert evidence[9].fallback_reason == "missing_pixel_spacing"
    assert evidence[10].crop_shape == (130, 130)


def test_clipped_config_preserves_retained_recipe() -> None:
    pytest.importorskip("torch")
    from rsna_knee.combined_supervision import load_combined_baseline_config

    root = Path(__file__).parents[1]
    control = load_combined_baseline_config(root / "configs/physical-crop.toml")
    candidate = load_combined_baseline_config(root / "configs/clipped-window.toml")
    assert candidate.preprocessing == CONFIG
    assert replace(candidate, preprocessing=control.preprocessing) == control
    assert (
        replace(
            CONFIG,
            slice_count=8,
            input_representation="same_slice_rgb",
            sampling_layout="full_span",
        )
        == control.preprocessing
    )
    assert (
        CONFIG.record()["normalization_population"] == "three_selected_centers_only_v1"
    )
    for changes in (
        {"slice_count": 8},
        {"physical_fov_mm": 140},
        {"image_size": 336},
        {"input_representation": "same_slice_rgb"},
    ):
        with pytest.raises(ValueError):
            replace(CONFIG, **changes)


def test_three_real_encoder_inputs_and_equal_series_mean(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from torch import Tensor, nn

    from rsna_knee.training import (
        IndexedSeries,
        IndexedStudy,
        StudyClassifier,
        decode_study,
    )

    class CaptureMean(nn.Module):
        def forward(self, pixels: Tensor) -> Tensor:
            assert pixels.shape == (3, 3, 224, 224)
            return pixels.mean(dim=(2, 3))

    paths = write_ordered_series(tmp_path, list(range(21)))
    decoded = decode_study(
        IndexedStudy("synthetic", (IndexedSeries(0, tuple(paths)),)),
        CONFIG,
        labels=None,
    )
    assert decoded.pixels.shape == (1, 3, 3, 224, 224)
    model = StudyClassifier(CaptureMean(), 3, 1)
    with torch.no_grad():
        model.head.weight.fill_(1)
        model.head.bias.zero_()
        actual = model(
            decoded.pixels.unsqueeze(0),
            decoded.slice_mask.unsqueeze(0),
            decoded.series_mask.unsqueeze(0),
        )
        normalized = (
            decoded.pixels[0] - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        ) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        torch.testing.assert_close(actual[0, 0], normalized.mean(dim=(0, 2, 3)).sum())


def test_clipped_bundle_is_guarded_and_round_trips(tmp_path: Path) -> None:
    import json

    from rsna_knee.kaggle import (
        build_plane_gating_bundle,
        load_kaggle_bundle,
        push_kaggle_bundle,
    )

    bundle = build_plane_gating_bundle(
        Path(__file__).parents[1],
        tmp_path / "bundle",
        kernel_id="plthon/rsna-knee-clipped-window",
        clipped_window=True,
    )
    metadata = json.loads(bundle.kernel_metadata_path.read_text())
    assert metadata["enable_gpu"] == "false"
    assert "plthon/rsna-knee-physical-crop" in metadata["kernel_sources"]
    assert load_kaggle_bundle(bundle.root).root == bundle.root
    notebook = json.loads(bundle.notebook_path.read_text())
    source = "\n".join(
        cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        for cell in notebook["cells"]
    )
    assert "run_test_inference=False" in source
    assert "clipped_window=True" in source
    assert "central_80_three_anchor" in source
    with pytest.raises((ValueError, PermissionError), match="approv"):
        push_kaggle_bundle(bundle, approved=False, gpu_approved=False)


@pytest.mark.parametrize(
    "delta,positive,regression,expected",
    [
        (0.005, 0.95, 0.0, "keep"),
        (0.0049, 0.99, 0.0, "reject"),
        (0.01, 0.949, 0.0, "reject"),
        (0.01, 0.99, -0.021, "reject"),
    ],
)
def test_clipped_keep_rule_is_frozen(
    delta: float, positive: float, regression: float, expected: str
) -> None:
    pytest.importorskip("torch")
    from rsna_knee.combined_supervision import (
        _adjacent_slice_decision,
        load_combined_baseline_config,
    )

    config = load_combined_baseline_config(
        Path(__file__).parents[1] / "configs/clipped-window.toml"
    )
    result = _adjacent_slice_decision(
        {
            "delta": {
                "official_macro_auc": delta,
                "per_target": {"ACL": {"delta_roc_auc": regression}},
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
    assert result["next_experiment"] == (
        "fixed_clipped_window_336px"
        if expected == "keep"
        else "plane_or_sequence_aware_aggregation"
    )
