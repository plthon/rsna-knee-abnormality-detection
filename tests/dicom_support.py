from pathlib import Path

import numpy as np
from pydicom import dcmread
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    SecondaryCaptureImageStorage,
    generate_uid,
)


def write_dicom(
    path: Path,
    pixels: np.ndarray,
    *,
    slope: float | None = 1.0,
    intercept: float | None = 0.0,
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
    if slope is not None:
        dataset.RescaleSlope = slope
    if intercept is not None:
        dataset.RescaleIntercept = intercept
    dataset.PixelData = pixels.astype(np.uint16).tobytes()
    dataset.save_as(path, enforce_file_format=True)


def write_ordered_series(root: Path, values: list[int]) -> list[Path]:
    paths = []
    for index, value in enumerate(values):
        # Deliberately reverse filenames and duplicate/non-consecutive InstanceNumber.
        path = root / f"slice-{len(values) - index}.dcm"
        write_dicom(path, np.full((140, 140), value, dtype=np.uint16))
        dataset = dcmread(path)
        dataset.PixelSpacing = [1.0, 1.0]
        dataset.InstanceNumber = 99 if index < 2 else 500 + index
        dataset.save_as(path, enforce_file_format=True)
        paths.append(path)
    return paths
