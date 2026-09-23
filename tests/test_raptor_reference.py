from __future__ import annotations

import csv
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from rsna_knee import raptor_reference as raptor

PROJECT_ROOT = Path(__file__).parents[1]
REFERENCE_CONTRACT = PROJECT_ROOT / "contracts" / "raptor-reference-v1.json"
TRAINING_REFERENCE_CONTRACT = (
    PROJECT_ROOT / "contracts" / "raptor-training-reference-v1.json"
)
CHECKPOINT = (
    PROJECT_ROOT
    / ".artifacts/raptor-reference/model-dataset/raptor_ft_coatnet_v4_full.pt"
)


def test_reference_contract_freezes_public_source_and_exact_layout() -> None:
    contract, digest = raptor.load_reference_contract(REFERENCE_CONTRACT)

    assert len(digest) == 64
    assert contract["public_notebook"] == {
        **contract["public_notebook"],
        "version": 7,
        "source_sha256": "859aae12ecea593b45ab0a5f6463fa573cd3083f340c161e920f10fbba29e81b",
    }
    assert contract["model_dataset"]["version"] == 2
    assert contract["checkpoint"]["sha256"] == raptor.CHECKPOINT_SHA256
    assert contract["checkpoint"]["metadata"]["arch"] == raptor.ARCHITECTURE
    assert tuple(contract["checkpoint"]["metadata"]["lab"]) == raptor.TARGET_NAMES
    assert contract["model"]["cuda_backend"] == {
        "cudnn_benchmark": True,
        "matmul_allow_tf32": True,
    }
    assert (
        tuple(
            (
                slot["plane"],
                slot["preferred_fluid_sensitive"],
                slot["slice_count"],
            )
            for slot in contract["study_stack"]["slots"]
        )
        == raptor.SLOTS
    )
    assert [slot[2] for slot in raptor.SLOTS] == [18, 14, 12, 8, 12]
    assert raptor.STACK_SIZE == 64
    assert raptor.WINDOW_COUNT == 42


def test_training_reference_contract_freezes_blocked_public_source() -> None:
    contract, digest = raptor.load_raptor_training_reference_contract(
        TRAINING_REFERENCE_CONTRACT
    )

    assert len(digest) == 64
    assert contract["status"] == "blocked_pending_authoritative_training_provenance"
    notebook = contract["public_notebook"]
    assert notebook["url"] == (
        "https://www.kaggle.com/code/dreaddevelopment/"
        "knee-mri-training-the-twelve-finding-model"
    )
    assert notebook["kaggle_id"] == 131724100
    assert notebook["current_version"] == 8
    assert notebook["source_sha256"] == (
        "6ea209b03c33bf21d50d89e9e9696e2aecf90c4bcccdd1037c0b3711ab3d3699"
    )
    assert notebook["source_size_bytes"] == 35181
    assert notebook["source_line_count"] == 670
    assert notebook["docker_image"] == (
        "gcr.io/kaggle-images/python@sha256:"
        "dafd4ce5668bbf1ad422e4c109e0f18c9623c3a7c7f48b0235f13142755c40b9"
    )
    datasets = contract["attached_data"]["datasets"]
    assert [(dataset["dataset_id"], dataset["version"]) for dataset in datasets] == [
        (11703975, 2),
        (11735963, 1),
    ]
    assert [
        (file["name"], file["size_bytes"], file["etag_md5"])
        for file in datasets[0]["files"]
    ] == [
        ("all_ids.npy", 819328, "cac84b408590d45d67420884cb0d7d03"),
        ("all_masks.npy", 140928, "8aeb911a4ee5590d97d0c7497cead92c"),
        ("all_vols.npy", 15895756928, "e57b2d99849172f3c36d4cea40e05edc"),
        ("window_index.npy", 783936, "5da25640a4fa5851430bfc82a725fa15"),
    ]
    assert [
        (file["name"], file["size_bytes"], file["etag_md5"])
        for file in datasets[1]["files"]
    ] == [
        ("extra_ids.npy", 309120, "9f031e86bf4114f1a80a6419a0af325d"),
        ("extra_masks.npy", 53236, "b98fb526138cbe781820f08b493a3066"),
        ("extra_meta.json", 85, "ce68ba631254c68ce98bef58216a8aca"),
        ("extra_vols.npy", 5995680896, "3d867ebf417469e70e77b39d0608e01d"),
    ]
    assert contract["attached_data"]["aggregate_volume_shape"] == [
        4407,
        44,
        336,
        336,
    ]
    assert tuple(contract["labels"]["target_order"]) == raptor.TARGET_NAMES
    assert contract["labels"]["authoritative_source"] is None
    assert contract["labels"]["candidate"]["authority"] == (
        "non_authoritative_candidate"
    )
    configurations = contract["configurations"]
    assert configurations["executable_defaults"]["architecture"] == (
        "vit_small_patch16_224"
    )
    assert configurations["documented_coatnet_example"]["architecture"] == (
        raptor.ARCHITECTURE
    )
    assert configurations["actual_run_configuration"] is None
    assert contract["checkpointing"]["public_training_checkpoint"] is None
    assert contract["unresolved_authoritative_identities"] == [
        "The required soft-label parquet source/version/hash is not attached or pinned.",
        "The actual executed architecture and command-line arguments are not exposed.",
        "No completed public training output or checkpoint is retained.",
        (
            "Python, PyTorch, timm, NumPy, pandas, scikit-learn, and CUDA "
            "runtime package versions are not captured."
        ),
        "The timm pretrained weight URL and byte digest are not pinned.",
        (
            "The competition input has no immutable dataset version exposed by the "
            "observed API."
        ),
        (
            "The current-version source hash could not be independently retrieved "
            "through an explicit version-8 endpoint because that request returned "
            "HTTP 403."
        ),
    ]
    assert contract["boundaries"] == {
        "authoritative_training_recipe_complete": False,
        "resource_infeasible": False,
        "training_run_completed": False,
    }
    packaged_contract, packaged_digest = (
        raptor.load_raptor_training_reference_contract()
    )
    assert packaged_contract == contract
    assert packaged_digest == digest


@pytest.mark.parametrize(
    "mutation",
    (
        "status",
        "source_digest",
        "target_order",
        "candidate_authority",
        "authoritative_label_source",
        "merged_configuration",
        "actual_run_configuration",
        "public_training_checkpoint",
        "resource_infeasible",
        "completed_training",
    ),
)
def test_training_reference_contract_rejects_unfounded_claims(
    tmp_path: Path,
    mutation: str,
) -> None:
    contract = json.loads(TRAINING_REFERENCE_CONTRACT.read_text())
    if mutation == "status":
        contract["status"] = "training_reproduced"
    elif mutation == "source_digest":
        contract["public_notebook"]["source_sha256"] = "0" * 64
    elif mutation == "target_order":
        contract["labels"]["target_order"] = list(reversed(raptor.TARGET_NAMES))
    elif mutation == "candidate_authority":
        contract["labels"]["candidate"]["authority"] = "authoritative"
    elif mutation == "authoritative_label_source":
        contract["labels"]["authoritative_source"] = {
            "dataset_id": 11640568,
            "version": 3,
        }
    elif mutation == "merged_configuration":
        contract["configurations"]["documented_coatnet_example"]["kind"] = (
            "executable_cli_defaults"
        )
    elif mutation == "actual_run_configuration":
        contract["configurations"]["actual_run_configuration"] = {
            "architecture": raptor.ARCHITECTURE
        }
    elif mutation == "public_training_checkpoint":
        contract["checkpointing"]["public_training_checkpoint"] = {
            "filename": "fabricated.pt"
        }
    elif mutation == "resource_infeasible":
        contract["boundaries"]["resource_infeasible"] = True
    else:
        contract["boundaries"]["training_run_completed"] = True

    mutated_path = tmp_path / "raptor-training-reference-v1.json"
    mutated_path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="training reference contract"):
        raptor.load_raptor_training_reference_contract(mutated_path)


def test_series_selection_preserves_row_order_and_preference() -> None:
    rows: list[dict[str, object]] = [
        {
            "SeriesInstanceUID": "sag-other",
            "Anatomical_Plane": "Sagittal",
            "Fluid_Sensitive": 0,
        },
        {
            "SeriesInstanceUID": "sag-fluid-b",
            "Anatomical_Plane": "Sagittal",
            "Fluid_Sensitive": 1,
        },
        {
            "SeriesInstanceUID": "sag-fluid-c",
            "Anatomical_Plane": "Sagittal",
            "Fluid_Sensitive": 1,
        },
        {
            "SeriesInstanceUID": "axial",
            "Anatomical_Plane": "Axial",
            "Fluid_Sensitive": 0,
        },
    ]
    used: set[str] = set()

    first = raptor.select_series_for_slot(rows, "Sagittal", 1, used)
    assert first is rows[1]
    used.add("sag-fluid-b")
    assert raptor.select_series_for_slot(rows, "Sagittal", 0, used) is rows[0]
    assert raptor.select_series_for_slot(rows, "Axial", -1, used) is rows[3]
    assert raptor.select_series_for_slot(rows, "Coronal", 1, used) is None


def test_sampling_endpoints_and_short_series_repetition_are_exact() -> None:
    assert raptor.sampled_slice_indices(10, 4) == (0, 3, 5, 8)
    assert raptor.sampled_slice_indices(3, 5) == (0, 0, 0, 1, 1)
    assert raptor.sampled_slice_indices(2, 5) == (0, 0, 0, 0, 0)
    assert raptor.sampled_slice_indices(1, 4) == (0, 0, 0, 0)


def test_study_stack_uses_five_slots_and_zero_fills_missing_slots(
    tmp_path: Path,
) -> None:
    selected: list[str] = []
    rows = [
        {
            "SeriesInstanceUID": "sag-fluid",
            "Anatomical_Plane": "Sagittal",
            "Fluid_Sensitive": "1",
        },
        {
            "SeriesInstanceUID": "sag-other",
            "Anatomical_Plane": "Sagittal",
            "Fluid_Sensitive": "0",
        },
        {
            "SeriesInstanceUID": "cor-fluid",
            "Anatomical_Plane": "Coronal",
            "Fluid_Sensitive": "1",
        },
        {
            "SeriesInstanceUID": "cor-other",
            "Anatomical_Plane": "Coronal",
            "Fluid_Sensitive": "0",
        },
        {
            "SeriesInstanceUID": "axial",
            "Anatomical_Plane": "Axial",
            "Fluid_Sensitive": "0",
        },
    ]

    def order(series_dir: Path) -> raptor.OrderedSeries:
        selected.append(series_dir.name)
        return ([(series_dir / "slice.dcm", 1.0)], 1.0)

    def read(_: Path) -> np.ndarray:
        return np.arange(16, dtype=np.float32).reshape(4, 4)

    def resize(image: np.ndarray, _: float) -> np.ndarray:
        return np.resize(image, (raptor.INTERMEDIATE_SIZE, raptor.INTERMEDIATE_SIZE))

    stack = raptor.build_study_stack(
        "study", {"study": rows}, tmp_path, reader=(order, read, resize)
    )
    assert selected == ["sag-fluid", "sag-other", "cor-fluid", "cor-other", "axial"]
    assert stack.volume.shape == (64, 336, 336)
    assert stack.mask.shape == (64,)
    assert stack.mask.all()

    selected.clear()
    missing = raptor.build_study_stack(
        "study", {"study": rows[:1]}, tmp_path, reader=(order, read, resize)
    )
    assert selected == ["sag-fluid"]
    assert missing.mask[:18].all()
    assert not missing.mask[18:].any()
    assert not missing.volume[18:].any()


def test_decode_failure_zero_fills_only_failed_sampled_slices(tmp_path: Path) -> None:
    rows = [
        {
            "SeriesInstanceUID": "sag-fluid",
            "Anatomical_Plane": "Sagittal",
            "Fluid_Sensitive": "1",
        }
    ]
    crop_spacings: list[float] = []

    def order(series_dir: Path) -> raptor.OrderedSeries:
        return (
            [
                (series_dir / "bad.dcm", 0.5),
                (series_dir / "good.dcm", -1.0),
                (series_dir / "unused.dcm", 0.5),
            ],
            0.75,
        )

    def read(path: Path) -> np.ndarray:
        if path.name == "bad.dcm":
            raise ValueError("synthetic decode failure")
        return np.arange(16, dtype=np.float32).reshape(4, 4)

    def resize(image: np.ndarray, spacing: float) -> np.ndarray:
        crop_spacings.append(spacing)
        return np.resize(image, (336, 336))

    stack = raptor.build_study_stack(
        "study", {"study": rows}, tmp_path, reader=(order, read, resize)
    )

    assert not stack.mask[0]
    assert stack.mask[17]
    assert not stack.volume[0].any()
    assert crop_spacings and set(crop_spacings) == {0.75}


def test_reader_orders_by_geometry_and_uses_first_pixel_spacing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pydicom = pytest.importorskip("pydicom")
    first = tmp_path / "first.dcm"
    second = tmp_path / "second.dcm"
    first.touch()
    second.touch()
    headers = {
        first: SimpleNamespace(
            ImageOrientationPatient=[1, 0, 0, 0, 1, 0],
            ImagePositionPatient=[0, 0, 4],
            PixelSpacing=[0.7, 9.9],
        ),
        second: SimpleNamespace(
            ImageOrientationPatient=[1, 0, 0, 0, 1, 0],
            ImagePositionPatient=[0, 0, 2],
            PixelSpacing=[0.8, 8.8],
        ),
    }
    monkeypatch.setattr(pydicom, "dcmread", lambda path, **_kwargs: headers[Path(path)])

    ordered, median = raptor.make_reader()[0](tmp_path)

    assert ordered == [(second, 0.8), (first, 0.7)]
    assert median == pytest.approx(0.75)


def test_public_crop_uses_first_spacing_square_clamp_and_no_padding() -> None:
    cv2 = pytest.importorskip("cv2")
    crop_resize = raptor.make_reader()[2]
    image = np.arange(35, dtype=np.float32).reshape(5, 7)

    actual = crop_resize(image, 28.0)
    expected = cv2.resize(
        image[:, 1:6],
        (336, 336),
        interpolation=cv2.INTER_AREA,
    )
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (336, 336)


def test_windows_have_exact_count_shape_channel_order_and_internal_gap_behavior() -> (
    None
):
    volume = np.zeros((64, 336, 336), dtype=np.uint8)
    for index in range(64):
        volume[index].fill(index)
    mask = np.ones(64, dtype=np.uint8)
    mask[20:30] = 0

    centers = raptor.eval_centers(mask, 64, 42)
    windows = raptor.eval_windows(volume, mask, count=1, resolution=336)
    denormalized = windows * raptor.IMAGENET_STD + raptor.IMAGENET_MEAN

    assert len(centers) == 42
    assert any(20 <= center < 30 for center in centers)
    assert windows.shape == (1, 3, 336, 336)
    torch.testing.assert_close(
        denormalized[0, :, 0, 0], torch.tensor([0.0, 1.0, 2.0]) / 255.0
    )
    assert raptor.eval_centers(np.zeros(64, dtype=np.uint8), 64, 3) == (1, 1, 1)


class _IdentityBackbone(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


def test_attention_pooling_is_per_target_unmasked_and_deterministic() -> None:
    model = raptor.RaptorClassifier(
        _IdentityBackbone(), feature_count=4, target_count=2, dropout=0.2
    ).eval()
    with torch.no_grad():
        for parameter in model.att.parameters():
            parameter.zero_()
        model.clsW.copy_(torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]]))
        model.clsb.zero_()
    features = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [4.0, 1.0, 0.0, 2.0]]])

    first = model.head(features)
    second = model.head(features)
    normalized = model.norm(features)
    expected = torch.stack(
        [normalized[:, :, 0].mean(1), normalized[:, :, 1].mean(1)], dim=1
    )

    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, first, atol=0, rtol=0)


def test_rank_transform_matches_public_argsort_semantics() -> None:
    probabilities = np.array([[0.9, 0.1], [0.2, 0.1], [0.5, 0.7]], dtype=np.float32)
    np.testing.assert_array_equal(
        raptor.rank_percentiles(probabilities),
        np.array([[1.0, 0.0], [0.0, 0.5], [0.5, 1.0]]),
    )


def _write_submission_inputs(root: Path) -> None:
    competition = root / "competition"
    competition.mkdir(parents=True)
    (competition / "test.csv").write_text("StudyInstanceUID\nstudy-a\nstudy-b\n")
    with (competition / "sample_submission.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("StudyInstanceUID", *raptor.TARGET_NAMES))
        writer.writerow(("study-a", *(0.5 for _ in raptor.TARGET_NAMES)))
        writer.writerow(("study-b", *(0.5 for _ in raptor.TARGET_NAMES)))
    with (competition / "test_series.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            (
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "Fluid_Sensitive",
                "Fat_Suppression",
                "Anatomical_Plane",
            )
        )
        writer.writerow(("study-a", "series-a", 1, 0, "Sagittal"))
        writer.writerow(("study-b", "series-b", 1, 0, "Sagittal"))


def test_reference_run_routes_ranked_predictions_through_strict_submission_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_submission_inputs(tmp_path / "input")
    metadata = raptor.RaptorCheckpointMetadata(
        sha256=raptor.CHECKPOINT_SHA256,
        architecture=raptor.ARCHITECTURE,
        resolution=384,
        targets=raptor.TARGET_NAMES,
        epoch=12,
        source="timm-pretrained",
        gold_auc=0.9167492483170476,
    )
    outputs = iter(
        [
            np.full(12, 0.2, dtype=np.float32),
            np.full(12, 0.8, dtype=np.float32),
        ]
    )
    monkeypatch.setattr(
        raptor,
        "load_raptor_model",
        lambda *_args, **_kwargs: (_IdentityBackbone(), metadata),
    )
    monkeypatch.setattr(raptor, "make_reader", lambda: (None, None, None))
    monkeypatch.setattr(
        raptor,
        "build_study_stack",
        lambda *_args, **_kwargs: raptor.RaptorStudyStack(
            np.zeros((64, 336, 336), dtype=np.uint8), np.zeros(64, dtype=np.uint8)
        ),
    )
    monkeypatch.setattr(
        raptor, "eval_windows", lambda *_args, **_kwargs: torch.zeros(1, 3, 1, 1)
    )
    monkeypatch.setattr(
        raptor, "infer_probabilities", lambda *_args, **_kwargs: next(outputs)
    )

    outcome = raptor.run_raptor_reference(
        input_root=tmp_path / "input",
        work_root=tmp_path / "work",
        checkpoint_path=tmp_path / raptor.CHECKPOINT_FILENAME,
        device=torch.device("cpu"),
        contract_path=PROJECT_ROOT / "contracts" / "competition.json",
        run_id="raptor-reference-test",
        source_provenance={
            "git_commit": "c" * 40,
            "git_dirty": False,
            "jupytext_source_sha256": "d" * 64,
            "kernel_id": "plthon/raptor-scoring-test",
            "rendered_notebook_sha256": "e" * 64,
        },
        notebook={
            "competition_submission": False,
            "hidden_test_inference": True,
            "kernel_id": "plthon/raptor-scoring-test",
            "kind": "approved_kaggle_raptor_reference_hidden_scoring",
            "training": False,
        },
    )

    validation = json.loads(outcome.submission.validation_report_path.read_text())
    rows = list(csv.reader(outcome.submission.submission_path.open()))
    report = json.loads(outcome.report_path.read_text())
    manifest = json.loads(outcome.submission.manifest_path.read_text())
    reference, reference_digest = raptor.load_reference_contract(REFERENCE_CONTRACT)
    assert validation["valid"] is True
    assert rows[0] == ["StudyInstanceUID", *raptor.TARGET_NAMES]
    assert [float(value) for value in rows[1][1:]] == [0.0] * 12
    assert [float(value) for value in rows[2][1:]] == [1.0] * 12
    assert report["output"]["submission_transform"] == "per-target percentile rank"
    assert manifest["source"]["raptor_reference"] == {
        "checkpoint": {
            "filename": reference["checkpoint"]["filename"],
            "sha256": reference["checkpoint"]["sha256"],
            "size_bytes": reference["checkpoint"]["size_bytes"],
        },
        "model_dataset": reference["model_dataset"],
        "public_notebook": reference["public_notebook"],
    }
    assert manifest["source"]["raptor_reference_contract_sha256"] == reference_digest
    assert outcome.raw_failure_count == 0

    report["inference"].update(
        {
            "cuda_backend": {
                "cudnn_benchmark": True,
                "matmul_allow_tf32": True,
            },
            "device": "cuda",
            "peak_gpu_memory_bytes": 123,
        }
    )
    outcome.report_path.write_text(json.dumps(report))
    for record in manifest["artifacts"]:
        if record["path"] == outcome.report_path.name:
            record["sha256"] = hashlib.sha256(
                outcome.report_path.read_bytes()
            ).hexdigest()
            record["size_bytes"] = outcome.report_path.stat().st_size
    outcome.submission.manifest_path.write_text(json.dumps(manifest))

    verified = raptor.verify_raptor_reference_scoring_artifact(tmp_path / "work")
    assert verified["valid"] is True
    assert verified["row_count"] == 2
    assert verified["raw_failure_count"] == 0
    assert verified["submission_sha256"] == validation["submission_sha256"]

    submission_bytes = outcome.submission.submission_path.read_bytes()
    outcome.submission.submission_path.write_bytes(submission_bytes + b"tampered")
    with pytest.raises(ValueError, match="artifact hash differs"):
        raptor.verify_raptor_reference_scoring_artifact(tmp_path / "work")
    outcome.submission.submission_path.write_bytes(submission_bytes)

    manifest_bytes = outcome.submission.manifest_path.read_bytes()
    changed_manifest = json.loads(manifest_bytes)
    changed_manifest["notebook"]["competition_submission"] = True
    outcome.submission.manifest_path.write_text(json.dumps(changed_manifest))
    with pytest.raises(ValueError, match="boundary flags"):
        raptor.verify_raptor_reference_scoring_artifact(tmp_path / "work")
    outcome.submission.manifest_path.write_bytes(manifest_bytes)

    validation["valid"] = False
    outcome.submission.validation_report_path.write_text(json.dumps(validation))
    changed_manifest = json.loads(manifest_bytes)
    for record in changed_manifest["artifacts"]:
        if record["path"] == outcome.submission.validation_report_path.name:
            record["sha256"] = hashlib.sha256(
                outcome.submission.validation_report_path.read_bytes()
            ).hexdigest()
            record["size_bytes"] = (
                outcome.submission.validation_report_path.stat().st_size
            )
    outcome.submission.manifest_path.write_text(json.dumps(changed_manifest))
    with pytest.raises(ValueError, match="validation evidence is invalid"):
        raptor.verify_raptor_reference_scoring_artifact(tmp_path / "work")


def test_classifier_checkpoint_reload_preserves_predictions(tmp_path: Path) -> None:
    torch.manual_seed(0)
    inputs = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    first_model = raptor.RaptorClassifier(
        _IdentityBackbone(), feature_count=4, target_count=2
    ).eval()
    expected = torch.sigmoid(first_model(inputs))
    checkpoint = tmp_path / "classifier.pt"
    torch.save(first_model.state_dict(), checkpoint)

    second_model = raptor.RaptorClassifier(
        _IdentityBackbone(), feature_count=4, target_count=2
    ).eval()
    second_model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    actual = torch.sigmoid(second_model(inputs))

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.training
def test_published_checkpoint_reload_is_exact_and_outputs_probabilities() -> None:
    if not CHECKPOINT.is_file():
        pytest.skip("Downloaded public checkpoint is not present")
    windows = torch.linspace(-1.0, 1.0, 3 * 384 * 384, dtype=torch.float32).reshape(
        1, 3, 384, 384
    )

    first_model, first_metadata = raptor.load_raptor_model(
        CHECKPOINT, device=torch.device("cpu")
    )
    first = raptor.infer_probabilities(first_model, windows, torch.device("cpu"))
    del first_model
    gc.collect()
    second_model, second_metadata = raptor.load_raptor_model(
        CHECKPOINT, device=torch.device("cpu")
    )
    second = raptor.infer_probabilities(second_model, windows, torch.device("cpu"))

    assert first_metadata == second_metadata
    assert first_metadata.targets == raptor.TARGET_NAMES
    assert first_metadata.architecture == raptor.ARCHITECTURE
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()
    assert ((0 <= first) & (first <= 1)).all()
