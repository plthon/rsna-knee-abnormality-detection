import os
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    SecondaryCaptureImageStorage,
    generate_uid,
)

from rsna_knee import decode_pixels, load_config, preprocess_series, uniform_indices


def _write_dicom(
    path: Path,
    pixels: np.ndarray,
    *,
    slope: float = 1.0,
    intercept: float = 0.0,
    photometric_interpretation: str = "MONOCHROME2",
) -> None:
    sop_instance_uid = generate_uid()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dataset = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = sop_instance_uid
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = photometric_interpretation
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.RescaleSlope = slope
    dataset.RescaleIntercept = intercept
    dataset.PixelData = pixels.astype(np.uint16).tobytes()
    dataset.save_as(path, enforce_file_format=True)


def test_generated_dicom_pipeline_runs_on_cpu(tmp_path: Path) -> None:
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""

    input_root = tmp_path / "dicoms"
    input_root.mkdir()
    config_path = tmp_path / "pipeline.toml"
    config_path.write_text(
        f"""
[paths]
mode = "local"
input_root = "{input_root}"
work_root = "{tmp_path / "artifacts"}"

[preprocessing]
slice_count = 4
lower_percentile = 0
upper_percentile = 100
""".strip()
    )

    base_pixels = np.array([[0, 1], [2, 3]], dtype=np.uint16)
    for index in range(10):
        _write_dicom(input_root / f"{index:02}.dcm", base_pixels + index)

    inverted_path = input_root / "monochrome1.dcm"
    _write_dicom(
        inverted_path,
        base_pixels,
        slope=2,
        intercept=-10,
        photometric_interpretation="MONOCHROME1",
    )
    np.testing.assert_array_equal(
        decode_pixels(inverted_path),
        np.array([[-4, -6], [-8, -10]], dtype=np.float32),
    )

    config = load_config(config_path)
    series = sorted(config.paths.input_root.glob("[0-9][0-9].dcm"))
    normalized, mask = preprocess_series(series, config.preprocessing)

    assert normalized.shape == (4, 2, 2)
    np.testing.assert_array_equal(mask, [True, True, True, True])
    assert np.isfinite(normalized).all()
    assert normalized.min() == 0
    assert normalized.max() == 1

    short_normalized, short_mask = preprocess_series(series[:3], config.preprocessing)
    assert short_normalized.shape == (4, 2, 2)
    np.testing.assert_array_equal(short_mask, [True, True, True, False])
    assert not short_normalized[3].any()


def test_uniform_sampling_handles_long_and_short_series() -> None:
    assert uniform_indices(length=10, count=4) == (0, 3, 6, 9)
    assert uniform_indices(length=3, count=8) == (0, 1, 2)


def test_local_and_kaggle_modes_are_resolved_from_config() -> None:
    local = load_config(Path("configs/local.toml"))
    kaggle = load_config(Path("configs/kaggle.toml"))

    assert local.paths.mode == "local"
    assert local.paths.input_root == Path("data")
    assert kaggle.paths.mode == "kaggle"
    assert kaggle.paths.input_root == Path("/kaggle/input")
    assert kaggle.paths.work_root == Path("/kaggle/working")
