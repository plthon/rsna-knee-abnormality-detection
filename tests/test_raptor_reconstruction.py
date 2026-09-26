from __future__ import annotations

import importlib
import json
import random
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from rsna_knee import raptor_reconstruction as reconstruction
from rsna_knee import raptor_reference as reference
from rsna_knee.provenance import write_run_manifest


class TinyBackbone(nn.Module):
    num_features = 4

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, self.num_features)

    def forward(self, inputs: Tensor) -> Tensor:
        return cast(Tensor, self.projection(inputs.mean(dim=(-2, -1))))


def _config() -> reconstruction.RaptorReconstructionConfig:
    return reconstruction.RaptorReconstructionConfig(
        architecture=reference.ARCHITECTURE,
        resolution=384,
        epochs=16,
        batch_size=8,
        validation_batch_size=4,
        training_windows=12,
        evaluation_windows=24,
        backbone_learning_rate=3e-5,
        head_learning_rate=1e-3,
        weight_decay=0.02,
        workers=4,
        topk=3,
        seed=42,
        scheduler_pct_start=0.15,
        gradient_clip_norm=3.0,
        gain_minimum=0.9,
        gain_maximum=1.1,
    )


def test_reconstruction_config_is_exact_and_evidence_scoped() -> None:
    contract, _ = reference.load_raptor_training_reference_contract()
    config = reconstruction.reconstruction_config(contract)

    assert config == _config()
    assert contract["status"] == "phase_1_success_phase_2_resource_infeasible"
    assert contract["labels"]["candidate"]["authority"] == (
        "non_authoritative_candidate"
    )
    selection = contract["labels"]["reconstruction_selection"]
    assert selection["evidence_semantic"] == "best_evidence_inferred"
    assert selection["historical_attachment_proven"] is False
    assert selection["reconstruction_identity"]["sha256"] == (
        "15f7339514ff9af961da64c4b7eb40cb3c231aa6aaac93c99cde42201a2f9373"
    )
    assert contract["configurations"]["actual_run_configuration"] is None
    assert contract["checkpointing"]["public_training_checkpoint"] is None
    with pytest.raises(PermissionError, match="resource infeasibility"):
        reconstruction.require_reconstruction_open(contract)


def test_cohorts_require_exact_disjoint_corpus_coverage() -> None:
    soft_ids = ("s0", "s1", "s2")
    gold_ids = ("g0", "g1")
    soft = np.full((3, len(reference.TARGET_NAMES)), 0.25, dtype=np.float32)
    gold = np.tile(
        np.asarray([0, 1], dtype=np.float32)[:, None],
        (1, len(reference.TARGET_NAMES)),
    )
    cohorts = reconstruction.prepare_cohorts(
        corpus_ids=(*soft_ids, *gold_ids),
        soft_ids=soft_ids,
        soft_targets=soft,
        gold_ids=gold_ids,
        gold_targets=gold,
        expected_soft_count=3,
        expected_gold_count=2,
    )
    assert cohorts.training_indices == (0, 1, 2)
    assert cohorts.gold_indices == (3, 4)

    with pytest.raises(ValueError, match="excluded"):
        reconstruction.prepare_cohorts(
            corpus_ids=(*soft_ids, "g0"),
            soft_ids=soft_ids,
            soft_targets=soft,
            gold_ids=("s0", "g0"),
            gold_targets=gold,
            expected_soft_count=3,
            expected_gold_count=2,
        )
    invalid = soft.copy()
    invalid[0, 0] = 1.1
    with pytest.raises(ValueError, match="finite probabilities"):
        reconstruction.prepare_cohorts(
            corpus_ids=(*soft_ids, *gold_ids),
            soft_ids=soft_ids,
            soft_targets=invalid,
            gold_ids=gold_ids,
            gold_targets=gold,
            expected_soft_count=3,
            expected_gold_count=2,
        )


def test_window_selection_scaling_and_gain_match_source() -> None:
    mask = np.zeros(10, dtype=np.uint8)
    mask[2:8] = 1
    assert reconstruction.valid_centers(mask) == [3, 4, 5, 6]
    assert reconstruction.evaluation_centers(mask, 4) == (3, 4, 5, 6)

    random.seed(42)
    centers = reconstruction.training_centers(mask, 4)
    assert centers == (5, 4, 6, 3)

    volume = np.arange(10 * 2 * 2, dtype=np.uint8).reshape(10, 2, 2)
    windows = reconstruction.make_training_windows(volume, (3,), resolution=2, gain=0.9)
    assert windows.shape == (1, 3, 2, 2)
    expected = torch.from_numpy(volume[2:5].astype(np.float32) / 255.0) * 0.9
    torch.testing.assert_close(windows[0], expected)
    assert float(windows.min()) >= 0
    assert float(windows.max()) <= 1


def test_soft_label_parquet_schema_and_identity_are_captured(tmp_path: Path) -> None:
    pandas = importlib.import_module("pandas")
    path = tmp_path / "labels_llm_soft.parquet"
    rows: dict[str, list[object]] = {"StudyInstanceUID": ["s0", "s1"]}
    rows.update({target: [0.25, 0.75] for target in reference.TARGET_NAMES})
    pandas.DataFrame(rows).to_parquet(path, index=False)

    ids, targets, identity = reconstruction._read_soft_labels(path)
    assert ids == ("s0", "s1")
    assert targets.shape == (2, len(reference.TARGET_NAMES))
    assert identity["row_count"] == 2
    assert identity["sha256"] == reconstruction._sha256(path)
    assert identity["historical_attachment_proven"] is False

    bad = pandas.DataFrame(rows)[
        ["StudyInstanceUID", *reversed(reference.TARGET_NAMES)]
    ]
    bad.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="columns or target order"):
        reconstruction._read_soft_labels(path)


def test_source_seed_capture_is_repeatable_without_extra_configuration() -> None:
    reconstruction._seed_source_generators(42)
    first = (random.random(), float(np.random.random()), torch.rand(1).item())
    reconstruction._seed_source_generators(42)
    second = (random.random(), float(np.random.random()), torch.rand(1).item())
    assert first == second


def test_positive_weights_optimizer_and_scheduler_are_exact() -> None:
    targets = np.full((10, len(reference.TARGET_NAMES)), 0.5, dtype=np.float32)
    targets[:, 0] = 0
    targets[:, 1] = 1
    weights = reconstruction.prevalence_positive_weights(targets)
    assert weights[0].item() == pytest.approx(10.0)
    assert weights[1].item() == pytest.approx(1.0)
    assert weights[2].item() == pytest.approx(1.0)

    model = reference.RaptorClassifier(TinyBackbone(), feature_count=4)
    optimizer, scheduler = reconstruction.build_optimizer_and_scheduler(
        model, _config(), steps_per_epoch=5
    )
    assert len(optimizer.param_groups) == 2
    assert [group["max_lr"] for group in optimizer.param_groups] == pytest.approx(
        [3e-5, 1e-3]
    )
    assert all(
        group["weight_decay"] == pytest.approx(0.02) for group in optimizer.param_groups
    )
    assert scheduler.total_steps == 80
    grouped = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert grouped == {id(parameter) for parameter in model.parameters()}


def test_top_states_average_floats_and_copy_best_nonfloats() -> None:
    first = reconstruction.RankedState(
        macro_auc=0.9,
        epoch=1,
        aucs=(0.9,),
        state={
            "weight": torch.tensor([1.0]),
            "count": torch.tensor([7], dtype=torch.int64),
        },
    )
    second = reconstruction.RankedState(
        macro_auc=0.8,
        epoch=2,
        aucs=(0.8,),
        state={
            "weight": torch.tensor([3.0]),
            "count": torch.tensor([11], dtype=torch.int64),
        },
    )
    assert reconstruction.is_strict_improvement(0.9, None)
    assert reconstruction.is_strict_improvement(0.91, first)
    assert not reconstruction.is_strict_improvement(0.9, first)
    ranked = reconstruction.retain_top_states((), second, topk=3)
    ranked = reconstruction.retain_top_states(ranked, first, topk=3)
    assert [item.epoch for item in ranked] == [1, 2]
    averaged = reconstruction.average_ranked_states(ranked)
    torch.testing.assert_close(averaged["weight"], torch.tensor([2.0]))
    torch.testing.assert_close(averaged["count"], torch.tensor([7]))


def test_reconstruction_checkpoint_round_trip_keeps_public_loader_strict(
    tmp_path: Path,
) -> None:
    model = reference.RaptorClassifier(TinyBackbone(), feature_count=4)
    model.eval()
    inputs = torch.rand(2, 3, 3, 4, 4)
    expected = model(inputs).detach()
    path = tmp_path / "reconstruction.pt"
    torch.save(
        reconstruction.checkpoint_payload(
            state=model.state_dict(),
            macro_auc=0.75,
            aucs=[0.75] * len(reference.TARGET_NAMES),
            epoch=3,
            source="timm_best_evidence_reconstruction",
        ),
        path,
    )
    digest = reconstruction._sha256(path)
    loaded, metadata = reconstruction.load_raptor_reconstruction_checkpoint(
        path,
        expected_sha256=digest,
        device=torch.device("cpu"),
        backbone_factory=TinyBackbone,
    )
    torch.testing.assert_close(loaded(inputs), expected)
    assert metadata.epoch == 3
    with pytest.raises(ValueError, match="frozen reference"):
        reference.load_raptor_model(path, device=torch.device("cpu"))


def test_manifest_runtime_extension_and_artifact_verifier(tmp_path: Path) -> None:
    best = tmp_path / "raptor-reconstruction-best.pt"
    averaged = tmp_path / "raptor-reconstruction-swa.pt"
    report = tmp_path / "raptor-reconstruction-report.json"
    best.write_bytes(b"best")
    averaged.write_bytes(b"average")
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "smoke",
                "execution": {
                    "hidden_test_inference": False,
                    "competition_submission": False,
                },
            }
        )
    )
    manifest = write_run_manifest(
        tmp_path / "run-manifest.json",
        run_id="raptor-test",
        source={},
        configuration={},
        notebook={},
        artifacts=[best, averaged, report],
        artifact_root=tmp_path,
        runtime={"packages": {"timm": "1.0.test"}, "cuda": {"version": "test"}},
    )
    raw = json.loads(manifest.read_text())
    assert raw["runtime"]["packages"]["timm"] == "1.0.test"
    assert raw["runtime"]["packages"]["numpy"] is not None
    verified = reconstruction.verify_raptor_reconstruction_artifact(tmp_path)
    assert verified["stage"] == "smoke"
    assert verified["artifact_count"] == 3

    report.write_text("{}")
    with pytest.raises(ValueError, match="hash differs"):
        reconstruction.verify_raptor_reconstruction_artifact(tmp_path)


def test_label_identity_lifecycle_gates_training(tmp_path: Path) -> None:
    contract, _ = reference.load_raptor_training_reference_contract()
    assert reconstruction.required_pinned_label_identity(contract) == {
        "sha256": ("15f7339514ff9af961da64c4b7eb40cb3c231aa6aaac93c99cde42201a2f9373"),
        "size_bytes": 176422,
        "columns": ["StudyInstanceUID", *reference.TARGET_NAMES],
        "row_count": 4349,
    }

    pending = json.loads(
        (
            Path(__file__).parents[1] / "contracts/raptor-training-reference-v1.json"
        ).read_text()
    )
    pending["status"] = "best_evidence_reconstruction_selected_pending_asset_pin"
    identity = pending["labels"]["reconstruction_selection"]["reconstruction_identity"]
    identity.update(
        {
            "status": "pending_approved_read_only_retrieval",
            "sha256": None,
            "size_bytes": None,
            "columns": None,
            "row_count": None,
        }
    )
    pending["boundaries"]["label_identity_pinned"] = False
    pending["boundaries"]["resource_infeasible"] = False
    pending["checkpointing"]["reconstruction_output"]["status"] = "pending_execution"
    pending.pop("resource_evidence")
    path = tmp_path / "pending-training-reference.json"
    path.write_text(json.dumps(pending))
    loaded, _ = reference.load_raptor_training_reference_contract(path)
    with pytest.raises(PermissionError, match="label identity pin"):
        reconstruction.required_pinned_label_identity(loaded)


def test_pretrained_cache_revision_is_checked_before_symlink_resolution(
    tmp_path: Path,
) -> None:
    revision = "abc123"
    blob = tmp_path / "blobs" / "weight-bytes"
    blob.parent.mkdir()
    blob.write_bytes(b"weights")
    snapshot = tmp_path / "snapshots" / revision / "model.safetensors"
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to(blob)

    validated = reconstruction._validated_pretrained_cache_path(snapshot, revision)
    assert validated == snapshot
    assert validated.suffix == ".safetensors"
