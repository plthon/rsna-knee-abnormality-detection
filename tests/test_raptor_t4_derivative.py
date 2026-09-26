from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor, nn
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset

from rsna_knee import raptor_reconstruction as parent
from rsna_knee import raptor_t4_derivative as derivative
from rsna_knee.kaggle import build_raptor_t4_derivative_bundle
from rsna_knee.raptor_reference import (
    RaptorClassifier,
    load_raptor_training_reference_contract,
)


class TinyBackbone(nn.Module):
    num_features = 4

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 4)

    def forward(self, inputs: Tensor) -> Tensor:
        return cast(Tensor, self.projection(inputs.mean(dim=(-2, -1))))


def test_derivative_changes_only_batch_and_accumulation() -> None:
    config, contract, _ = derivative.derivative_config()
    parent_contract, _ = load_raptor_training_reference_contract()
    original = parent.reconstruction_config(parent_contract)
    assert {
        key for key in config.record() if config.record()[key] != original.record()[key]
    } == {"batch_size"}
    assert config.batch_size == 4
    assert contract["gradient_accumulation_steps"] == 2
    assert contract["effective_batch_size"] == 8
    assert derivative.optimizer_updates(1088) == 544
    assert derivative.optimizer_updates(5) == 3


def test_accumulation_scales_loss_and_steps_only_after_clipping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    events: list[tuple[str, float | None]] = []

    class RecordingScheduler:
        last_epoch = 0

        def step(self) -> None:
            events.append(("scheduler", None))
            self.last_epoch += 1

    def clip(parameters: object, max_norm: float) -> Tensor:
        assert max_norm == 3
        assert model.weight.grad is not None
        events.append(("clip", float(model.weight.grad.item())))
        return torch.tensor(0.0)

    monkeypatch.setattr(nn.utils, "clip_grad_norm_", clip)
    optimizer = SGD(model.parameters(), lr=0.1)
    original_zero = optimizer.zero_grad

    def zero_grad(set_to_none: bool = True) -> None:
        events.append(("zero", None))
        original_zero(set_to_none=set_to_none)

    def step() -> None:
        assert model.weight.grad is not None
        events.append(("optimizer", float(model.weight.grad.item())))

    monkeypatch.setattr(optimizer, "zero_grad", zero_grad)
    monkeypatch.setattr(optimizer, "step", step)
    scheduler = RecordingScheduler()
    loader = DataLoader(
        TensorDataset(torch.ones(5, 1), torch.zeros(5, 1)), batch_size=1
    )
    result = derivative.train_epoch(
        cast(RaptorClassifier, model),
        cast(DataLoader[tuple[Tensor, Tensor]], loader),
        optimizer,
        cast(OneCycleLR, scheduler),
        nn.MSELoss(),
        device=torch.device("cpu"),
        gradient_clip_norm=3,
    )
    assert result["training_loss"] == pytest.approx(1.0)
    assert result["micro_batches"] == 5
    assert result["optimizer_updates"] == 3
    assert result["scheduler_steps"] == 3
    assert result["update_study_counts"] == [2, 2, 1]
    assert [name for name, _ in events] == [
        "zero",
        "clip",
        "optimizer",
        "scheduler",
        "zero",
        "clip",
        "optimizer",
        "scheduler",
        "zero",
        "clip",
        "optimizer",
        "scheduler",
        "zero",
    ]
    assert [value for name, value in events if name == "clip"] == [2, 2, 1]
    assert model.weight.grad is None


def test_oom_keeps_completed_update_timing() -> None:
    class OomModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1, 1))
            self.calls = 0

        def forward(self, inputs: Tensor) -> Tensor:
            self.calls += 1
            if self.calls == 3:
                raise torch.OutOfMemoryError("synthetic OOM")
            return inputs @ self.weight

    model = OomModel()
    optimizer = SGD(model.parameters(), lr=0.1)
    scheduler = OneCycleLR(optimizer, max_lr=0.1, total_steps=3)
    loader = DataLoader(
        TensorDataset(torch.ones(5, 1), torch.zeros(5, 1)), batch_size=1
    )
    progress: dict[str, object] = {}
    with pytest.raises(torch.OutOfMemoryError):
        derivative.train_epoch(
            cast(RaptorClassifier, model),
            cast(DataLoader[tuple[Tensor, Tensor]], loader),
            optimizer,
            scheduler,
            nn.MSELoss(),
            device=torch.device("cpu"),
            gradient_clip_norm=3,
            progress=progress,
        )
    assert len(cast(list[float], progress["micro_batch_seconds"])) == 2
    assert len(cast(list[float], progress["optimizer_update_seconds"])) == 1
    assert scheduler.last_epoch == 1


def test_resume_preserves_progress_and_checkpoint_layout(tmp_path: Path) -> None:
    model = RaptorClassifier(TinyBackbone(), feature_count=4)
    config, _, derivative_sha = derivative.derivative_config()
    optimizer, scheduler = parent.build_optimizer_and_scheduler(
        model, config, steps_per_epoch=2
    )
    model(torch.rand(2, 3, 3, 4, 4)).sum().backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    model(torch.rand(2, 3, 3, 4, 4)).sum().backward()
    optimizer.step()
    scheduler.step()
    state = parent._cpu_state(model)
    ranked = parent.RankedState(0.7, 1, (0.7,) * 12, state)
    path = tmp_path / "resume.pt"
    derivative._save_resume(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        updates=2,
        records=[{"epoch": 1}],
        resume_boundaries=[{"from_epoch": 0, "sha256": "prior"}],
        top_states=(ranked,),
        best=ranked,
        contract_sha256=derivative_sha,
        source_commit="c" * 40,
        label_sha256="label",
        pretrained_sha256="weight",
    )
    restored = RaptorClassifier(TinyBackbone(), feature_count=4)
    restored_optimizer, restored_scheduler = parent.build_optimizer_and_scheduler(
        restored, config, steps_per_epoch=2
    )
    epoch, updates, records, boundaries, top, best = derivative._load_resume(
        path,
        model=restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        contract_sha256=derivative_sha,
        source_commit="c" * 40,
        label_sha256="label",
        pretrained_sha256="weight",
        epochs=16,
        updates_per_epoch=2,
    )
    assert (epoch, updates, records) == (1, 2, [{"epoch": 1}])
    assert boundaries == [{"from_epoch": 0, "sha256": "prior"}]
    assert restored_scheduler.last_epoch == 2
    assert restored_optimizer.state_dict()["state"]
    assert top[0].epoch == best.epoch == 1
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], tensor)
    with pytest.raises(ValueError, match="identity or progress"):
        derivative._load_resume(
            path,
            model=restored,
            optimizer=restored_optimizer,
            scheduler=restored_scheduler,
            contract_sha256=derivative_sha,
            source_commit="d" * 40,
            label_sha256="label",
            pretrained_sha256="weight",
            epochs=16,
            updates_per_epoch=2,
        )

    checkpoint = tmp_path / "best.pt"
    torch.save(
        parent.checkpoint_payload(
            state=state,
            macro_auc=0.7,
            aucs=ranked.aucs,
            epoch=1,
            source="issue_38_raptor_t4_derivative",
        ),
        checkpoint,
    )
    loaded, metadata = parent.load_raptor_reconstruction_checkpoint(
        checkpoint,
        expected_sha256=parent._sha256(checkpoint),
        device=torch.device("cpu"),
        backbone_factory=TinyBackbone,
    )
    assert metadata.epoch == 1
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], tensor)


def test_derivative_bundle_is_separate_and_guarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = Path(__file__).parents[1]
    monkeypatch.setattr("rsna_knee.kaggle._git_state", lambda _: ("c" * 40, False))
    with pytest.raises(PermissionError, match="GPU approval"):
        build_raptor_t4_derivative_bundle(project, tmp_path / "blocked", stage="smoke")
    bundle = build_raptor_t4_derivative_bundle(
        project, tmp_path / "smoke", stage="smoke", gpu_approved=True
    )
    metadata = json.loads(bundle.kernel_metadata_path.read_text())
    assert metadata["enable_gpu"] == "true"
    assert metadata["competition_sources"] == ["rsna-knee-abnormality-detection"]
    assert metadata["dataset_sources"] == [
        "dreaddevelopment/knee-raptor-corpus",
        "dreaddevelopment/knee-raptor-corpus-ext",
        "dreaddevelopment/rsna-knee-labels",
    ]
    source = "\n".join(
        "".join(cell["source"])
        for cell in json.loads(bundle.notebook_path.read_text())["cells"]
    )
    assert 'STAGE_LITERAL = "smoke"' in source
    assert 'RESUME = json.loads("false")' in source
    assert "run_raptor_t4_derivative" in source
    assert "except torch.OutOfMemoryError:" in source
    assert '"kernel_version": KERNEL_VERSION' in source
    assert "test_series" not in source
    assert "submission.csv" not in source
    resumed = build_raptor_t4_derivative_bundle(
        project,
        tmp_path / "full",
        stage="full",
        gpu_approved=True,
        resume_dataset_id="plthon/raptor-t4-resume",
    )
    full_metadata = json.loads(resumed.kernel_metadata_path.read_text())
    assert full_metadata["dataset_sources"][-1] == "plthon/raptor-t4-resume"
