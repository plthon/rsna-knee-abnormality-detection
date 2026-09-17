import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    SecondaryCaptureImageStorage,
    generate_uid,
)

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from torch import Tensor, nn

from rsna_knee.combined_supervision import (
    _predict_combined_submission,
    load_verified_combined_checkpoint,
    run_combined_scoring,
)
from rsna_knee.competition_contract import load_verified_submission_contract
from rsna_knee.config import PreprocessingConfig
from rsna_knee.submission import SubmissionInputs
from rsna_knee.training import DecodedStudy, GoldResNet18, IndexedSeries, IndexedStudy

PROJECT_ROOT = Path(__file__).parents[1]
TARGETS = tuple(
    json.loads((PROJECT_ROOT / "contracts/competition.json").read_text())["targets"][
        "order"
    ]
)


class _MeanModel(nn.Module):
    def forward(
        self, pixels: Tensor, slice_mask: Tensor, series_mask: Tensor
    ) -> Tensor:
        del slice_mask, series_mask
        value = pixels.mean(dim=tuple(range(1, pixels.ndim)))
        return value.unsqueeze(1).repeat(1, len(TARGETS))


@pytest.mark.parametrize(
    "backbone,physical_crop,adjacent_slice",
    (
        ("resnet18", False, False),
        ("dinov2_vits14", False, False),
        ("resnet18", True, False),
        ("resnet18", True, True),
    ),
)
def test_scoring_uses_live_hidden_test_index_and_never_trains(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backbone: Literal["resnet18", "dinov2_vits14"],
    physical_crop: bool,
    adjacent_slice: bool,
) -> None:
    input_root = tmp_path / "input"
    competition_root = input_root / "competitions" / "rsna-knee-abnormality-detection"
    competition_root.mkdir(parents=True)
    hidden_ids = ("hidden-X", "hidden-Y", "hidden-Z")
    frozen_ids = ("visible-A", "visible-B", "visible-C")

    _write_csv(
        competition_root / "test.csv",
        ("StudyInstanceUID",),
        [(study_id,) for study_id in hidden_ids],
    )
    _write_csv(
        competition_root / "sample_submission.csv",
        ("StudyInstanceUID", *TARGETS),
        [(study_id, *(0.5 for _ in TARGETS)) for study_id in hidden_ids],
    )
    series_rows = []
    for study_id in hidden_ids:
        series_id = f"series-{study_id}"
        series_rows.append((study_id, series_id, "Yes", "No", "Sagittal"))
        series_root = competition_root / "test_series" / study_id / series_id
        series_root.mkdir(parents=True)
        _write_dicom(series_root / "slice.dcm")
    _write_csv(
        competition_root / "test_series.csv",
        (
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "Fluid_Sensitive",
            "Fat_Suppression",
            "Anatomical_Plane",
        ),
        series_rows,
    )

    frozen_index = input_root / "datasets" / "plthon" / "rsna-knee-data-index"
    frozen_index.mkdir(parents=True)
    (frozen_index / "studies.csv").write_text(
        "study_index,split,StudyInstanceUID,PatientID,series_count,usable_series_count,"
        "slice_count,usable_slice_count\n"
        + "\n".join(
            f"{index},test,{study_id},patient-{index},1,1,1,1"
            for index, study_id in enumerate(frozen_ids)
        )
        + "\n"
    )

    checkpoint_path = input_root / "checkpoint" / "checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"verified-by-test-double")
    payload = {
        "targets": list(TARGETS),
        "priors": [0.5] * len(TARGETS),
        "preprocessing": {
            "slice_count": 8,
            "lower_percentile": 1.0,
            "upper_percentile": 99.0,
            "image_size": 224,
        },
        "configuration": {"training": {"mixed_precision": False}},
    }
    if physical_crop:
        payload["preprocessing"] = PreprocessingConfig(
            8,
            1,
            99,
            224,
            "physical_crop",
            input_representation=(
                "adjacent_slice_2_5d" if adjacent_slice else "same_slice_rgb"
            ),
        ).record()
        for path in competition_root.rglob("*.dcm"):
            from pydicom import dcmread

            dataset = dcmread(path)
            dataset.PixelSpacing = [1.0, 1.0]
            dataset.save_as(path, enforce_file_format=True)
    monkeypatch.setattr(
        "rsna_knee.combined_supervision.load_verified_physical_crop_checkpoint",
        lambda path, *, device, expected_sha256, contract_path=None: (
            _MeanModel(),
            payload,
        ),
    )
    monkeypatch.setattr(
        "rsna_knee.combined_supervision.load_verified_adjacent_slice_checkpoint",
        lambda path, *, device, expected_sha256, contract_path=None: (
            _MeanModel(),
            payload,
        ),
    )
    monkeypatch.setattr(
        "rsna_knee.combined_supervision.discover_combined_checkpoint",
        lambda _: checkpoint_path,
    )
    monkeypatch.setattr(
        "rsna_knee.combined_supervision.load_verified_combined_checkpoint",
        lambda path, *, device, contract_path=None: (_MeanModel(), payload),
    )
    monkeypatch.setattr(
        "rsna_knee.combined_supervision.load_verified_dinov2_checkpoint",
        lambda path, *, device, expected_sha256, contract_path=None: (
            _MeanModel(),
            payload,
        ),
    )
    monkeypatch.setattr(
        "rsna_knee.combined_supervision.require_approved_cuda",
        lambda approved: {"approved": approved, "device_name": "CPU"},
    )
    monkeypatch.setattr(
        "rsna_knee.combined_supervision.train_epoch",
        lambda *args, **kwargs: pytest.fail("scoring must never train"),
    )

    outcome = run_combined_scoring(
        input_root=input_root,
        work_root=tmp_path / "working",
        approved_gpu=True,
        checkpoint_path=None,
        competition_contract_path=PROJECT_ROOT / "contracts/competition.json",
        backbone=backbone,
        physical_crop=physical_crop,
        adjacent_slice=adjacent_slice,
    )

    if physical_crop:
        report = json.loads(outcome.scoring_report_path.read_text())
        counts = report["inference"]["spatial_evidence"]["counts"]
        assert counts["slices_crop"] == 3
        assert counts["shape:130x130"] == 3
        assert counts["studies_crop"] == 3
        if adjacent_slice:
            assert counts["representation:encoder_inputs"] == 3
    indexed_rows = _read_csv(outcome.index_artifact_dir / "studies.csv")
    assert {row["StudyInstanceUID"] for row in indexed_rows} == set(hidden_ids)
    assert not set(frozen_ids) & {row["StudyInstanceUID"] for row in indexed_rows}

    with outcome.submission_path.open(newline="") as file:
        submission_rows = list(csv.reader(file))
    assert submission_rows[0] == ["StudyInstanceUID", *TARGETS]
    assert [row[0] for row in submission_rows[1:]] == list(hidden_ids)
    assert not set(frozen_ids) & {row[0] for row in submission_rows[1:]}
    assert json.loads(outcome.scoring_report_path.read_text())["mode"] == (
        "inference_only"
    )
    assert (
        json.loads(outcome.scoring_report_path.read_text())["checkpoint"]["backbone"]
        == backbone
    )
    assert outcome.fallbacks_path.read_text() == "StudyInstanceUID,reason\n"


def test_scoring_checkpoint_hash_fails_before_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"wrong checkpoint")
    monkeypatch.setattr(
        "rsna_knee.combined_supervision.load_combined_checkpoint",
        lambda *args, **kwargs: pytest.fail("hash must be checked first"),
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_verified_combined_checkpoint(
            checkpoint_path,
            device=torch.device("cpu"),
        )


def test_scoring_refuses_catastrophic_fallback_fraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = load_verified_submission_contract(
        PROJECT_ROOT / "contracts/competition.json"
    )
    entity_ids = ("hidden-X", "hidden-Y", "hidden-Z")
    inputs = SubmissionInputs(tmp_path, entity_ids, contract)
    indexed = {
        entity_id: IndexedStudy(
            entity_id,
            (IndexedSeries(0, (tmp_path / f"{entity_id}.dcm",)),),
        )
        for entity_id in entity_ids
    }
    outcomes = iter((False, False, True))

    def fake_decode(*args: object, **kwargs: object) -> DecodedStudy:
        del args, kwargs
        if not next(outcomes):
            return DecodedStudy(
                pixels=torch.empty(0, 8, 224, 224),
                slice_mask=torch.empty(0, 8, dtype=torch.bool),
                series_mask=torch.empty(0, dtype=torch.bool),
                labels=None,
                label_mask=None,
                valid_slice_count=0,
                decode_error_counts={"pixel_decode_failed": 1},
            )
        return DecodedStudy(
            pixels=torch.zeros(1, 8, 224, 224),
            slice_mask=torch.tensor([[True] + [False] * 7]),
            series_mask=torch.tensor([True]),
            labels=None,
            label_mask=None,
            valid_slice_count=1,
            decode_error_counts={},
        )

    monkeypatch.setattr("rsna_knee.combined_supervision.decode_study", fake_decode)
    with pytest.raises(RuntimeError, match="too many test studies"):
        _predict_combined_submission(
            model=cast(GoldResNet18, _MeanModel()),
            submission_inputs=inputs,
            indexed=indexed,
            work_root=tmp_path / "output",
            preprocessing=PreprocessingConfig(
                slice_count=8,
                lower_percentile=1.0,
                upper_percentile=99.0,
                image_size=224,
            ),
            priors=(0.5,) * len(TARGETS),
            mixed_precision=False,
            device=torch.device("cpu"),
            max_fallback_fraction=0.5,
        )
    assert not (tmp_path / "output" / "submission.csv").exists()


def test_plane_gated_hidden_inference_uses_decoded_plane_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ScalarBackbone(nn.Module):
        def forward(self, values: Tensor) -> Tensor:
            return values.mean(dim=(1, 2, 3)).unsqueeze(1)

    model = GoldResNet18(
        ScalarBackbone(),
        feature_count=1,
        target_count=len(TARGETS),
        aggregation="target_plane_gate",
    )
    model.imagenet_mean.zero_()
    model.imagenet_std.fill_(1)
    assert model.plane_gate is not None
    with torch.no_grad():
        model.head.weight.fill_(1)
        model.head.bias.zero_()
        model.plane_gate.log_weights[0, 0] = torch.log(torch.tensor(4.0))
        model.plane_gate.log_weights[1, 2] = torch.log(torch.tensor(4.0))
    decoded = DecodedStudy(
        pixels=torch.tensor([[[[1.0]]], [[[3.0]]]]),
        slice_mask=torch.ones(2, 1, dtype=torch.bool),
        series_mask=torch.ones(2, dtype=torch.bool),
        labels=None,
        label_mask=None,
        valid_slice_count=2,
        decode_error_counts={},
        plane_indices=torch.tensor([0, 2]),
    )
    monkeypatch.setattr(
        "rsna_knee.combined_supervision.decode_study", lambda *args, **kwargs: decoded
    )
    contract = load_verified_submission_contract(
        PROJECT_ROOT / "contracts/competition.json"
    )
    inputs = SubmissionInputs(tmp_path, ("hidden-X",), contract)
    indexed = {
        "hidden-X": IndexedStudy(
            "hidden-X", (IndexedSeries(0, (tmp_path / "slice.dcm",), "sagittal"),)
        )
    }

    outcome = _predict_combined_submission(
        model=model,
        submission_inputs=inputs,
        indexed=indexed,
        work_root=tmp_path / "output",
        preprocessing=PreprocessingConfig(1, 1, 99, 1),
        priors=(0.5,) * len(TARGETS),
        mixed_precision=False,
        device=torch.device("cpu"),
    )

    with outcome.artifacts.submission_path.open(newline="") as file:
        row = list(csv.reader(file))[1]
    assert float(row[1]) < float(row[2])


def _write_csv(
    path: Path, columns: Sequence[str], rows: Sequence[Sequence[object]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)


def _write_dicom(path: Path) -> None:
    sop_uid = generate_uid()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = sop_uid
    dataset.Rows = 8
    dataset.Columns = 8
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = np.arange(64, dtype=np.uint16).tobytes()
    dataset.save_as(path, enforce_file_format=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))
