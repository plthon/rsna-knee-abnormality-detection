import ast
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from pydicom import dcmread
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.pixels.processing import apply_modality_lut
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid


def _public_interfaces(*names: str) -> dict[str, Any]:
    source_path = Path("notebooks/public_dicom_ordering.py")
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    selected: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            selected.append(node)
    namespace: dict[str, Any] = {
        "Any": Any,
        "Literal": Literal,
        "Path": Path,
        "Sequence": Sequence,
        "cast": cast,
        "dataclass": dataclass,
        "dcmread": dcmread,
        "np": np,
        "apply_modality_lut": apply_modality_lut,
    }
    exec(  # noqa: S102 - exercise the public functions authored in the notebook
        compile(ast.Module(body=selected, type_ignores=[]), source_path, "exec"),
        namespace,
    )
    return namespace


def test_reusable_public_interfaces_prefer_geometry_and_sample_without_duplicates(
    tmp_path: Path,
) -> None:
    public = _public_interfaces(
        "SliceRecord",
        "OrderedSeries",
        "geometry_coordinates",
        "order_series",
        "uniform_indices",
    )
    slice_record = public["SliceRecord"]
    records = tuple(
        slice_record(
            path=tmp_path / filename,
            instance_number=instance_number,
            orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            position=(0.0, 0.0, z),
        )
        for filename, instance_number, z in (
            ("a.dcm", 1, 20.0),
            ("b.dcm", 2, 0.0),
            ("c.dcm", 3, 10.0),
        )
    )

    ordered = public["order_series"](records)

    assert ordered.method == "geometry"
    assert [record.path.name for record in ordered.records] == [
        "b.dcm",
        "c.dcm",
        "a.dcm",
    ]

    without_geometry = tuple(
        slice_record(
            path=tmp_path / filename,
            instance_number=instance_number,
            orientation=None,
            position=None,
        )
        for filename, instance_number in (("c.dcm", 2), ("a.dcm", 3), ("b.dcm", 1))
    )
    instance_ordered = public["order_series"](without_geometry)
    assert instance_ordered.method == "instance_number"
    assert [record.path.name for record in instance_ordered.records] == [
        "b.dcm",
        "c.dcm",
        "a.dcm",
    ]

    without_instances = tuple(
        slice_record(
            path=tmp_path / filename,
            instance_number=None,
            orientation=None,
            position=None,
        )
        for filename in ("c.dcm", "a.dcm", "b.dcm")
    )
    filename_ordered = public["order_series"](without_instances)
    assert filename_ordered.method == "filename"
    assert [record.path.name for record in filename_ordered.records] == [
        "a.dcm",
        "b.dcm",
        "c.dcm",
    ]

    assert public["uniform_indices"](10, 4) == (0, 3, 6, 9)
    assert public["uniform_indices"](3, 8) == (0, 1, 2)


def test_reusable_public_preprocessing_rescales_inverts_and_normalizes(
    tmp_path: Path,
) -> None:
    public = _public_interfaces("decode_pixels", "normalize_images")
    path = tmp_path / "monochrome1.dcm"
    pixels = np.array([[0, 1], [2, 3]], dtype=np.uint16)
    sop_uid = generate_uid()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = MRImageStorage
    dataset.SOPInstanceUID = sop_uid
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME1"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.RescaleSlope = 2
    dataset.RescaleIntercept = -10
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(path, enforce_file_format=True)

    decoded = public["decode_pixels"](path)
    normalized = public["normalize_images"]([decoded], 0, 100)

    np.testing.assert_array_equal(
        decoded, np.array([[-4, -6], [-8, -10]], dtype=np.float32)
    )
    assert len(normalized) == 1
    assert np.isfinite(normalized[0]).all()
    assert float(normalized[0].min()) == 0
    assert float(normalized[0].max()) == 1
