import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from dicom_support import write_ordered_series
from pydicom import dcmread

from rsna_knee.config import PreprocessingConfig
from rsna_knee.preprocessing import (
    adjacent_triplet_indices,
    preprocess_series_result,
)


@pytest.mark.parametrize(
    ("length", "anchor", "expected"),
    [
        (5, 2, (1, 2, 3)),
        (5, 0, (0, 0, 1)),
        (5, 4, (3, 4, 4)),
        (1, 0, (0, 0, 0)),
        (2, 0, (0, 0, 1)),
        (2, 1, (0, 1, 1)),
    ],
)
def test_adjacent_triplet_indices(
    length: int, anchor: int, expected: tuple[int, int, int]
) -> None:
    assert adjacent_triplet_indices(length, anchor) == expected


def test_adjacent_representation_preserves_anchors_percentiles_crop_and_order(
    tmp_path: Path,
) -> None:
    paths = write_ordered_series(tmp_path, list(range(12)))
    control_config = PreprocessingConfig(8, 1, 99, 224, "physical_crop")
    candidate_config = replace(
        control_config, input_representation="adjacent_slice_2_5d"
    )
    control = preprocess_series_result(paths, control_config)
    candidate = preprocess_series_result(paths, candidate_config)

    assert (
        candidate.source_indices
        == control.source_indices
        == (
            0,
            1,
            3,
            4,
            6,
            7,
            9,
            11,
        )
    )
    assert (candidate.lower_value, candidate.upper_value) == (
        control.lower_value,
        control.upper_value,
    )
    assert candidate.pixels.shape == (8, 3, 224, 224)
    assert candidate.mask.all()
    assert np.all(candidate.pixels[2, 0] < candidate.pixels[2, 1])
    assert np.all(candidate.pixels[2, 1] < candidate.pixels[2, 2])
    assert candidate.preprocessing_counts["boundary_clamp_previous"] == 1
    assert candidate.preprocessing_counts["boundary_clamp_next"] == 1
    assert candidate.preprocessing_counts["encoder_inputs"] == 8
    assert candidate.preprocessing_counts["unique_source_slices_decoded"] == 12
    assert all(item.crop_shape == (130, 130) for item in candidate.spatial_evidence)


def test_same_slice_control_is_unchanged(tmp_path: Path) -> None:
    paths = write_ordered_series(tmp_path, [0, 1, 2])
    legacy = PreprocessingConfig(3, 1, 99, 224, "physical_crop")
    explicit = replace(legacy, input_representation="same_slice_rgb")
    old = preprocess_series_result(paths, legacy)
    control = preprocess_series_result(paths, explicit)
    np.testing.assert_array_equal(control.pixels, old.pixels)
    np.testing.assert_array_equal(control.mask, old.mask)
    assert control.pixels.shape == (3, 224, 224)
    assert "input_representation" not in legacy.record()


def test_adjacent_config_changes_only_channel_representation() -> None:
    pytest.importorskip("torch")
    from rsna_knee.combined_supervision import load_combined_baseline_config

    root = Path(__file__).parents[1]
    control = load_combined_baseline_config(root / "configs/physical-crop.toml")
    candidate = load_combined_baseline_config(root / "configs/adjacent-slice.toml")
    assert (
        replace(candidate.preprocessing, input_representation="same_slice_rgb")
        == control.preprocessing
    )
    assert replace(candidate, preprocessing=control.preprocessing) == control
    assert candidate.preprocessing.record()["channel_order"] == [
        "previous",
        "center",
        "next",
    ]


@pytest.mark.parametrize(("failed", "side"), [(0, "previous"), (2, "next")])
def test_failed_neighbor_substitutes_center_without_farther_search(
    tmp_path: Path, failed: int, side: str
) -> None:
    paths = write_ordered_series(tmp_path, [0, 1, 2])
    paths[failed].write_bytes(b"not a dicom")
    result = preprocess_series_result(
        paths,
        PreprocessingConfig(
            3,
            1,
            99,
            224,
            "physical_crop",
            input_representation="adjacent_slice_2_5d",
        ),
    )
    channel = 0 if side == "previous" else 2
    np.testing.assert_array_equal(result.pixels[1, channel], result.pixels[1, 1])
    assert result.preprocessing_counts[f"neighbor_substitution_{side}"] == 1
    assert (
        result.preprocessing_counts[f"neighbor_substitution_{side}:pixel_decode_failed"]
        == 1
    )
    assert not result.mask[failed]


def test_neighbor_spacing_fallback_is_recorded_separately(tmp_path: Path) -> None:
    paths = write_ordered_series(tmp_path, [0, 1, 2])
    dataset = dcmread(paths[0])
    del dataset.PixelSpacing
    dataset.save_as(paths[0], enforce_file_format=True)
    result = preprocess_series_result(
        paths,
        PreprocessingConfig(
            3,
            1,
            99,
            224,
            "physical_crop",
            input_representation="adjacent_slice_2_5d",
        ),
    )
    evidence = {item.source_index: item for item in result.spatial_evidence}
    assert evidence[0].fallback_reason == "missing_pixel_spacing"
    assert result.preprocessing_counts["unique_source_slices_decoded"] == 3


def test_model_accepts_pre_stacked_channels_without_changing_mean_pooling() -> None:
    torch = pytest.importorskip("torch")
    from torch import Tensor, nn

    from rsna_knee.training import StudyClassifier

    class ChannelMean(nn.Module):
        def forward(self, pixels: Tensor) -> Tensor:
            return pixels.mean(dim=(2, 3))

    model = StudyClassifier(ChannelMean(), 3, 1)
    model.imagenet_mean.zero_()
    model.imagenet_std.fill_(1)
    with torch.no_grad():
        model.head.weight.fill_(1)
        model.head.bias.zero_()
    pixels = torch.tensor(
        [[[[[[1.0]], [[2.0]], [[3.0]]], [[[3.0]], [[4.0]], [[5.0]]]]]]
    )
    output = model(
        pixels,
        torch.ones(1, 1, 2, dtype=torch.bool),
        torch.ones(1, 1, dtype=torch.bool),
    )
    torch.testing.assert_close(output, torch.tensor([[9.0]]))


def test_training_loader_retains_full_index_order_for_neighbor_lookup(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from rsna_knee.training import load_indexed_studies

    competition = tmp_path / "competition"
    series_root = competition / "train_images/study/series"
    series_root.mkdir(parents=True)
    ordered_names = ["z.dcm", "a.dcm", "y.dcm", "b.dcm"]
    for name in ordered_names:
        (series_root / name).touch()
    index = tmp_path / "index"
    index.mkdir()

    def write(
        name: str, columns: tuple[str, ...], rows: list[tuple[object, ...]]
    ) -> None:
        with (index / name).open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            writer.writerows(rows)

    write(
        "studies.csv",
        (
            "study_index",
            "split",
            "StudyInstanceUID",
            "PatientID",
            "series_count",
            "usable_series_count",
            "slice_count",
            "usable_slice_count",
        ),
        [(0, "train", "study", "patient", 1, 1, 4, 4)],
    )
    write(
        "series.csv",
        (
            "series_index",
            "study_index",
            "SeriesInstanceUID",
            "relative_path",
            "Fluid_Sensitive",
            "Fat_Suppression",
            "Anatomical_Plane",
            "slice_count",
            "usable_slice_count",
            "ordering_method",
            "selection_mask",
        ),
        [
            (
                0,
                0,
                "series",
                "train_images/study/series",
                "",
                "",
                "Sagittal",
                4,
                4,
                "geometry",
                "11110000",
            )
        ],
    )
    slice_columns = (
        "slice_index",
        "series_index",
        "SOPInstanceUID",
        "filename",
        "ordered_position",
        "rows",
        "columns",
        "instance_number",
        "geometry_coordinate",
        "transfer_syntax",
        "photometric_interpretation",
        "rescale_slope",
        "rescale_intercept",
    )
    write(
        "slices.csv",
        slice_columns,
        [
            (
                index_value,
                0,
                f"sop-{index_value}",
                name,
                index_value,
                2,
                2,
                99 if index_value < 2 else 500 + index_value,
                index_value,
                "",
                "MONOCHROME2",
                1,
                0,
            )
            for index_value, name in enumerate(ordered_names)
        ],
    )
    write(
        "selected_slices.csv",
        ("series_index", "slot", "slice_index", "ordered_position"),
        [(0, position, position, position) for position in range(4)],
    )

    loaded = load_indexed_studies(competition, index, ["study"], split="train")
    assert [path.name for path in loaded["study"].series[0].paths] == ordered_names


@pytest.mark.parametrize("clipped_window", [False, True])
def test_adjacent_checkpoint_reload_and_representation_protection(
    tmp_path: Path,
    clipped_window: bool,
) -> None:
    torch = pytest.importorskip("torch")
    from rsna_knee.combined_supervision import (
        _combined_config_record,
        load_combined_baseline_config,
        load_combined_checkpoint,
    )
    from rsna_knee.competition_contract import load_verified_submission_contract
    from rsna_knee.training import _new_uninitialized_model

    config = load_combined_baseline_config(
        Path(__file__).parents[1]
        / (
            "configs/clipped-window.toml"
            if clipped_window
            else "configs/adjacent-slice.toml"
        )
    )
    targets = list(load_verified_submission_contract().target_names)
    model = _new_uninitialized_model(len(targets))
    payload = {
        "checkpoint_schema_version": (
            "rsna-knee-clipped-window-2-5d-v1"
            if clipped_window
            else "rsna-knee-adjacent-slice-2-5d-v1"
        ),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "scaler_state_dict": {},
        "completed_epochs": 3,
        "targets": targets,
        "priors": [0.5] * len(targets),
        "preprocessing": {
            **config.preprocessing.record(),
            "shared_contract": "preprocess_series_result",
            "channel_adapter": "prestacked_previous_center_next",
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
    pixels = torch.rand(1, 1, 3, 3, 224, 224)
    mask = torch.ones(1, 1, 3, dtype=torch.bool)
    series = torch.ones(1, 1, dtype=torch.bool)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(pixels, mask, series), model(pixels, mask, series), atol=1e-5, rtol=0
        )
    assert metadata["preprocessing"]["input_representation"] == ("adjacent_slice_2_5d")

    original_schema = payload["checkpoint_schema_version"]
    for wrong_schema in {
        "rsna-knee-physical-crop-v1",
        "rsna-knee-adjacent-slice-2-5d-v1",
        "rsna-knee-clipped-window-2-5d-v1",
    } - {original_schema}:
        payload["checkpoint_schema_version"] = wrong_schema
        torch.save(payload, checkpoint)
        with pytest.raises(ValueError, match="identity|configuration|preprocessing"):
            load_combined_checkpoint(checkpoint, device=torch.device("cpu"))
    payload["checkpoint_schema_version"] = original_schema
    payload["targets"] = list(reversed(targets))
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="target order"):
        load_combined_checkpoint(checkpoint, device=torch.device("cpu"))


def test_adjacent_bundle_is_local_clean_training_only(tmp_path: Path) -> None:
    import json

    from rsna_knee.kaggle import build_plane_gating_bundle

    bundle = build_plane_gating_bundle(
        Path(__file__).parents[1],
        tmp_path / "bundle",
        kernel_id="plthon/rsna-knee-adjacent-slice",
        adjacent_slice=True,
    )
    metadata = json.loads(bundle.kernel_metadata_path.read_text())
    assert metadata["enable_gpu"] == "false"
    assert "plthon/rsna-knee-physical-crop" in metadata["kernel_sources"]
    notebook = json.loads(bundle.notebook_path.read_text())
    source = "\n".join(
        cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        for cell in notebook["cells"]
    )
    assert "run_test_inference=False" in source
    assert "adjacent_slice_2_5d" in source
    assert "verify_adjacent_slice_artifact" in source
    assert '"competition_submission": False' in source


def test_retained_control_discovery_accepts_identical_output_copies(
    tmp_path: Path,
) -> None:
    import hashlib

    from rsna_knee.combined_supervision import discover_combined_checkpoint

    first = tmp_path / "a/checkpoint.pt"
    second = tmp_path / "b/checkpoint.pt"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"retained #31")
    second.write_bytes(first.read_bytes())
    digest = hashlib.sha256(first.read_bytes()).hexdigest()
    assert (
        discover_combined_checkpoint(
            tmp_path,
            expected_sha256=digest,
            allow_identical_copies=True,
        )
        == first
    )
