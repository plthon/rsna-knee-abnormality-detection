from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from dicom_support import write_dicom as _write_dicom
from pydicom import dcmread

from rsna_knee.config import PreprocessingConfig
from rsna_knee.preprocessing import preprocess_series_result


@pytest.mark.parametrize(
    "spacing", [[1, 2], [0.5, 0.5], None, [0, 1], [float("nan"), 1], ["bad", 1]]
)
def test_resolution_preserves_sampling_percentiles_crop_and_fallback(
    tmp_path: Path, spacing: list[object] | None
) -> None:
    paths = []
    for i in range(12):
        path = tmp_path / f"{i}.dcm"
        _write_dicom(
            path,
            np.arange(200 * 160).reshape(200, 160) + i,
            slope=2,
            intercept=-10,
            photometric_interpretation="MONOCHROME1",
        )
        if spacing is not None:
            ds = dcmread(path)
            ds.add_new(0x00280030, "LO", [str(value) for value in spacing])
            ds.save_as(path, enforce_file_format=True)
        paths.append(path)
    control_config = PreprocessingConfig(8, 1, 99, 224, "physical_crop")
    control = preprocess_series_result(paths, control_config)
    candidate = preprocess_series_result(paths, replace(control_config, image_size=336))
    assert (
        control.source_indices == candidate.source_indices == (0, 1, 3, 4, 6, 7, 9, 11)
    )
    assert (control.lower_value, control.upper_value) == (
        candidate.lower_value,
        candidate.upper_value,
    )
    assert control.spatial_evidence == candidate.spatial_evidence
    assert not candidate.source_triplets and candidate.clipped_window is None
    assert candidate.pixels.shape == (8, 336, 336)
    np.testing.assert_array_equal(control.mask, candidate.mask)
    np.testing.assert_array_equal(
        control.pixels, preprocess_series_result(paths, control_config).pixels
    )
    if candidate.spatial_evidence[0].fallback_reason:
        whole = preprocess_series_result(
            paths, replace(control_config, image_size=336, spatial_mode="whole_image")
        )
        np.testing.assert_array_equal(candidate.pixels, whole.pixels)


@pytest.mark.parametrize("image_size", [224, 336])
@pytest.mark.parametrize("padding", [False, True])
def test_both_resolutions_resize_the_identical_known_physical_crop(
    tmp_path: Path, image_size: int, padding: bool
) -> None:
    torch = pytest.importorskip("torch")
    from torch.nn.functional import interpolate

    source = np.arange(200 * 160).reshape(200, 160)
    path = tmp_path / "source.dcm"
    _write_dicom(path, source)
    ds = dcmread(path)
    ds.PixelSpacing = [0.5, 0.5] if padding else [1, 2]
    ds.save_as(path, enforce_file_format=True)
    result = preprocess_series_result(
        [path], PreprocessingConfig(8, 1, 99, image_size, "physical_crop")
    )
    assert result.lower_value == pytest.approx(319.99)
    assert result.upper_value == pytest.approx(31679.01)
    normalized = np.clip(
        (source.astype(np.float32) - 319.99) / (31679.01 - 319.99), 0, 1
    )
    # Known #31 coordinates: 130x65 at (35,47), or 260x260 with neutral padding.
    crop = (
        np.pad(normalized, ((30, 30), (50, 50)))
        if padding
        else normalized[35:165, 47:112]
    )
    expected = interpolate(
        torch.from_numpy(crop).double()[None, None],
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=True,
    )
    np.testing.assert_allclose(
        result.pixels[0], expected[0, 0].numpy(), atol=3e-7, rtol=0
    )


def test_candidate_encoder_receives_eight_same_slice_rgb_anchors_and_equal_means() -> (
    None
):
    torch = pytest.importorskip("torch")
    from torch import Tensor, nn

    from rsna_knee.training import StudyClassifier

    class ChannelMean(nn.Module):
        def forward(self, pixels: Tensor) -> Tensor:
            assert pixels.shape == (8, 3, 336, 336)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            grayscale = pixels * std + mean
            torch.testing.assert_close(
                grayscale[:, 0], grayscale[:, 1], atol=1e-6, rtol=0
            )
            torch.testing.assert_close(
                grayscale[:, 0], grayscale[:, 2], atol=1e-6, rtol=0
            )
            return cast(Tensor, grayscale.mean((2, 3)))

    model = StudyClassifier(ChannelMean(), feature_count=3, target_count=1)
    assert (
        model.aggregation == "mean"
        and model.plane_gate is None
        and model.slot_attention is None
    )
    with torch.no_grad():
        model.head.weight.fill_(1 / 3)
        model.head.bias.zero_()
    pixels = torch.zeros(1, 2, 8, 336, 336)
    pixels[:, 0] = 0.2
    pixels[:, 1] = 0.8
    mask = torch.zeros(1, 2, 8, dtype=torch.bool)
    mask[:, 0, :6] = True
    mask[:, 1, :2] = True
    output = model(pixels, mask, torch.ones(1, 2, dtype=torch.bool))
    torch.testing.assert_close(output, torch.tensor([[0.5]]))


def test_offline_resnet_weights_and_architecture_are_identical_at_336() -> None:
    torch = pytest.importorskip("torch")
    from rsna_knee.competition_contract import load_verified_submission_contract
    from rsna_knee.training import build_initialized_model

    path = Path(__file__).parents[1] / ".artifacts/weights/resnet18-f37072fd.pth"
    if not path.is_file():
        pytest.skip("Exact offline ImageNet artifact is not installed")
    model, initialization = build_initialized_model(
        path, load_verified_submission_contract().target_names
    )
    assert (
        initialization.sha256
        == "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
    )
    assert sum(p.numel() for p in model.parameters()) == 11182668
    weights = torch.load(path, map_location="cpu", weights_only=True)
    from torch import nn

    for key, value in model.backbone.state_dict().items():
        if key.endswith("num_batches_tracked"):
            assert value.item() == 0
        else:
            torch.testing.assert_close(value, weights[key], atol=0, rtol=0)
    assert cast(nn.Conv2d, model.backbone.conv1).kernel_size == (7, 7)
    assert cast(nn.Conv2d, model.backbone.conv1).stride == (2, 2)
    assert cast(nn.AdaptiveAvgPool2d, model.backbone.avgpool).output_size == (1, 1)
    model.eval()
    for size in (224, 336):
        with torch.no_grad():
            output = model(
                torch.zeros(1, 1, 1, size, size),
                torch.ones(1, 1, 1, dtype=torch.bool),
                torch.ones(1, 1, dtype=torch.bool),
            )
        assert output.shape == (1, 12)
