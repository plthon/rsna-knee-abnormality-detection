import csv
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from torch import Tensor, nn
from torchvision.models import (  # type: ignore[import-untyped]
    ResNet18_Weights,
    resnet18,
)

from rsna_knee.training import (
    ANATOMICAL_PLANES,
    GoldResNet18,
    TargetSpecificPlaneGate,
    derive_gold_training_cohort,
    load_indexed_studies,
    load_smoke_training_config,
    masked_binary_cross_entropy,
    predict_submission_from_checkpoint,
    require_approved_cuda,
    resolve_anatomical_plane,
    run_local_cpu_check,
    verify_initialization_artifact,
)

pytestmark = pytest.mark.training

PROJECT_ROOT = Path(__file__).parents[1]
COMPETITION_CONTRACT = PROJECT_ROOT / "contracts" / "competition.json"
TARGETS = tuple(json.loads(COMPETITION_CONTRACT.read_text())["targets"]["order"])
TRAIN_COLUMNS = tuple(
    json.loads(COMPETITION_CONTRACT.read_text())["data"]["tables"]["train.csv"][
        "columns"
    ]
)
STUDY_COLUMNS = (
    "study_index",
    "split",
    "StudyInstanceUID",
    "PatientID",
    "series_count",
    "usable_series_count",
    "slice_count",
    "usable_slice_count",
)
SERIES_COLUMNS = (
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
)
SLICE_COLUMNS = (
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
SELECTED_COLUMNS = ("series_index", "slot", "slice_index", "ordered_position")


class _MeanBackbone(nn.Module):
    def forward(self, values: Tensor) -> Tensor:
        means = values.mean(dim=(1, 2, 3))
        return cast(Tensor, torch.stack((means, means * 2), dim=1))


def test_masked_pooling_excludes_slice_and_series_padding() -> None:
    model = GoldResNet18(_MeanBackbone(), feature_count=2, target_count=1)
    model.imagenet_mean.zero_()
    model.imagenet_std.fill_(1)
    with torch.no_grad():
        model.head.weight.fill_(1)
        model.head.bias.zero_()
    pixels = torch.tensor(
        [
            [
                [[[1.0]], [[3.0]], [[100.0]]],
                [[[5.0]], [[200.0]], [[300.0]]],
                [[[400.0]], [[500.0]], [[600.0]]],
            ]
        ]
    )
    slice_mask = torch.tensor([[[True, True, False], [True, False, False], [True] * 3]])
    series_mask = torch.tensor([[True, True, False]])

    first = model(pixels, slice_mask, series_mask)
    changed_padding = pixels.clone()
    changed_padding[0, 0, 2] = -9999
    changed_padding[0, 1, 1:] = -9999
    changed_padding[0, 2] = -9999
    second = model(changed_padding, slice_mask, series_mask)

    assert torch.equal(first, second)
    assert first.item() == pytest.approx(10.5)


def test_zero_plane_gate_is_equal_mean_with_exactly_36_parameters() -> None:
    gate = TargetSpecificPlaneGate(target_count=12)
    logits = torch.arange(48, dtype=torch.float32).view(1, 4, 12)
    mask = torch.tensor([[True, True, True, True]])
    planes = torch.tensor([[0, 0, 2, -1]])

    output = gate(logits, mask, planes)

    assert sum(parameter.numel() for parameter in gate.parameters()) == 36
    assert torch.allclose(output, logits.mean(dim=1))


def test_plane_gate_uses_actual_series_and_neutral_unknown_weight() -> None:
    gate = TargetSpecificPlaneGate(target_count=2)
    with torch.no_grad():
        gate.log_weights.copy_(
            torch.log(torch.tensor([[2.0, 99.0, 4.0], [1.0, 99.0, 3.0]]))
        )
    logits = torch.tensor([[[1.0, 10.0], [3.0, 20.0], [5.0, 40.0], [999.0, 999.0]]])
    mask = torch.tensor([[True, True, True, False]])
    planes = torch.tensor([[0, 0, -1, 1]])

    output = gate(logits, mask, planes)

    assert output[0, 0].item() == pytest.approx((2 * 1 + 2 * 3 + 1 * 5) / 5)
    assert output[0, 1].item() == pytest.approx((1 * 10 + 1 * 20 + 1 * 40) / 3)


def test_plane_gate_is_target_specific_and_receives_gradients() -> None:
    gate = TargetSpecificPlaneGate(target_count=2)
    with torch.no_grad():
        gate.log_weights[0, 0] = torch.log(torch.tensor(4.0))
        gate.log_weights[1, 2] = torch.log(torch.tensor(4.0))
    logits = torch.tensor([[[1.0, 1.0], [3.0, 3.0]]], requires_grad=True)
    output = gate(
        logits,
        torch.tensor([[True, True]]),
        torch.tensor([[0, 2]]),
    )

    output.sum().backward()

    assert output[0, 0] < output[0, 1]
    assert gate.log_weights.grad is not None
    assert bool((gate.log_weights.grad != 0).any())


def test_plane_resolution_is_typed_and_deterministic() -> None:
    assert ANATOMICAL_PLANES == ("sagittal", "coronal", "axial")
    assert resolve_anatomical_plane(" Sagittal ") == "sagittal"
    assert resolve_anatomical_plane("CORONAL") == "coronal"
    assert resolve_anatomical_plane("axial") == "axial"
    assert resolve_anatomical_plane("") is None
    assert resolve_anatomical_plane("oblique") is None


def test_zero_initialized_gated_model_matches_baseline_model() -> None:
    baseline = GoldResNet18(_MeanBackbone(), feature_count=2, target_count=2)
    gated = GoldResNet18(
        _MeanBackbone(),
        feature_count=2,
        target_count=2,
        aggregation="target_plane_gate",
    )
    gated.head.load_state_dict(baseline.head.state_dict())
    for model in (baseline, gated):
        model.imagenet_mean.zero_()
        model.imagenet_std.fill_(1)
    pixels = torch.tensor([[[[[1.0]], [[3.0]]], [[[5.0]], [[7.0]]]]])
    slice_mask = torch.ones(1, 2, 2, dtype=torch.bool)
    series_mask = torch.ones(1, 2, dtype=torch.bool)

    baseline_output = baseline(pixels, slice_mask, series_mask)
    gated_output = gated(
        pixels,
        slice_mask,
        series_mask,
        torch.tensor([[0, 2]]),
    )

    assert torch.allclose(gated_output, baseline_output, atol=1e-6)


def test_masked_bce_ignores_unknown_targets() -> None:
    logits = torch.tensor([[0.0, 100.0]], requires_grad=True)
    labels = torch.tensor([[1.0, 0.0]])
    known = torch.tensor([[True, False]])

    loss = masked_binary_cross_entropy(logits, labels, known)
    loss.backward()  # type: ignore[no-untyped-call]

    assert loss.item() == pytest.approx(float(torch.log(torch.tensor(2.0))))
    assert logits.grad is not None
    assert logits.grad[0, 0] != 0
    assert logits.grad[0, 1] == 0


def test_gpu_execution_is_refused_before_cuda_is_inspected() -> None:
    with pytest.raises(PermissionError, match="explicit approval"):
        require_approved_cuda(approved=False)


def test_checkpoint_prediction_is_refused_without_gpu_approval(
    tmp_path: Path,
) -> None:
    with pytest.raises(PermissionError, match="explicit approval"):
        predict_submission_from_checkpoint(
            input_root=tmp_path / "input",
            work_root=tmp_path / "work",
            checkpoint_path=tmp_path / "checkpoint.pt",
        )


def test_initialization_contract_rejects_byte_drift(tmp_path: Path) -> None:
    weight_path, contract_path = _random_resnet_artifact(tmp_path)
    verified = verify_initialization_artifact(weight_path, contract_path=contract_path)
    assert verified.sha256 == _sha256(weight_path)

    weight_path.write_bytes(weight_path.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_initialization_artifact(weight_path, contract_path=contract_path)


def test_local_cpu_check_completes_checkpoint_reload_and_prediction(
    tmp_path: Path,
) -> None:
    weight_path, contract_path = _random_resnet_artifact(tmp_path)

    result = run_local_cpu_check(
        weight_path=weight_path,
        work_root=tmp_path / "cpu-check",
        target_names=TARGETS,
        initialization_contract_path=contract_path,
    )

    assert result["forward_shape"] == [1, len(TARGETS)]
    assert result["prediction_shape"] == [1, len(TARGETS)]
    assert result["prediction_finite"] is True
    assert result["reloaded"] is True
    assert Path(str(result["checkpoint_path"])).is_file()


def test_cohort_is_rederived_from_authoritative_rows_and_digest_checked(
    tmp_path: Path,
) -> None:
    competition_root, index_root, index_contract, supervision_contract = (
        _cohort_fixture(tmp_path)
    )

    cohort = derive_gold_training_cohort(
        competition_root,
        index_root,
        gold_supervision_contract_path=supervision_contract,
        data_index_contract_path=index_contract,
    )

    assert [study.study_uid for study in cohort.studies] == ["study-a", "study-b"]
    assert cohort.target_names == TARGETS
    assert all(counts["positive"] == 1 for counts in cohort.per_target.values())
    assert all(counts["negative"] == 1 for counts in cohort.per_target.values())
    resolved = load_indexed_studies(
        competition_root,
        index_root,
        ["study-a", "study-b"],
        split="train",
    )
    assert resolved["study-a"].series[0].paths[0].name == "slice-a.dcm"
    assert resolved["study-a"].series[0].plane == "sagittal"

    contract = json.loads(supervision_contract.read_text())
    contract["canonical_cohort"]["digest"]["value"] = "0" * 64
    supervision_contract.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="digest mismatch"):
        derive_gold_training_cohort(
            competition_root,
            index_root,
            gold_supervision_contract_path=supervision_contract,
            data_index_contract_path=index_contract,
        )


def test_smoke_configuration_is_exactly_one_epoch_at_224(tmp_path: Path) -> None:
    config_path = tmp_path / "smoke.toml"
    config_path.write_text(
        (PROJECT_ROOT / "configs/kaggle-baseline.toml")
        .read_text()
        .replace("epochs = 10", "epochs = 1")
        .split("[bootstrap]", 1)[0]
    )
    config = load_smoke_training_config(config_path)
    assert config.epochs == 1
    assert config.preprocessing.image_size == 224
    assert config.weight_identity == "ResNet18_Weights.IMAGENET1K_V1"
    assert config.mixed_precision is True


def test_frozen_initialization_contract_names_one_exact_offline_release() -> None:
    contract = json.loads(
        (PROJECT_ROOT / "contracts" / "resnet18-imagenet1k-v1.json").read_text()
    )
    assert contract["identity"] == {
        "architecture": "torchvision.models.resnet18",
        "torch_version_base": "2.10.0",
        "torchvision_version_base": "0.25.0",
        "weight_enum": "ResNet18_Weights.IMAGENET1K_V1",
        "weight_release": "IMAGENET1K_V1",
        "default_alias_accepted": False,
    }
    assert contract["artifact"]["filename"] == "resnet18-f37072fd.pth"
    assert contract["artifact"]["size_bytes"] == 46830571
    assert contract["artifact"]["sha256"] == (
        "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
    )
    assert contract["construction_policy"] == {
        "validate_before_model_construction": True,
        "network_download_allowed_at_runtime": False,
        "silent_weight_fallback_allowed": False,
        "random_initialization_fallback_allowed": False,
        "strict_state_dict_load": True,
    }
    assert contract["upstream"]["license"]["torchvision_code_spdx"] == ("BSD-3-Clause")
    assert "scope_note" in contract["upstream"]["license"]


def _random_resnet_artifact(root: Path) -> tuple[Path, Path]:
    weight_path = root / "test-resnet18.pth"
    torch.save(resnet18(weights=None).state_dict(), weight_path)
    contract = {
        "schema_version": 1,
        "identity": {
            "architecture": "torchvision.models.resnet18",
            "torch_version_base": torch.__version__.split("+", 1)[0],
            "torchvision_version_base": "0.25.0",
            "weight_enum": "ResNet18_Weights.IMAGENET1K_V1",
        },
        "artifact": {
            "filename": weight_path.name,
            "size_bytes": weight_path.stat().st_size,
            "sha256": _sha256(weight_path),
            "source_url": ResNet18_Weights.IMAGENET1K_V1.url,
        },
        "construction_policy": {
            "validate_before_model_construction": True,
            "network_download_allowed_at_runtime": False,
            "silent_weight_fallback_allowed": False,
            "random_initialization_fallback_allowed": False,
            "strict_state_dict_load": True,
        },
    }
    contract_path = root / "initialization.json"
    contract_path.write_text(json.dumps(contract))
    return weight_path, contract_path


def _cohort_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    competition_root = root / "competition"
    competition_root.mkdir()
    train_rows = [
        _train_row("study-b", [1] * len(TARGETS)),
        _train_row("study-unlabeled", [None] * len(TARGETS)),
        _train_row("study-a", [0] * len(TARGETS)),
    ]
    _write_csv(competition_root / "train.csv", TRAIN_COLUMNS, train_rows)

    index_root = root / "index"
    index_root.mkdir()
    study_rows = [
        _study_row(0, "study-b"),
        _study_row(1, "study-unlabeled"),
        _study_row(2, "study-a"),
    ]
    series_rows: list[dict[str, object]] = []
    slice_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for series_index, (study_index, suffix) in enumerate(
        ((0, "b"), (1, "u"), (2, "a"))
    ):
        series_rows.append(
            {
                "series_index": series_index,
                "study_index": study_index,
                "SeriesInstanceUID": f"series-{suffix}",
                "relative_path": f"train_series/study-{suffix}/series-{suffix}",
                "Fluid_Sensitive": "",
                "Fat_Suppression": "",
                "Anatomical_Plane": "Sagittal",
                "slice_count": 1,
                "usable_slice_count": 1,
                "ordering_method": "geometry",
                "selection_mask": "10000000",
            }
        )
        slice_rows.append(
            {
                "slice_index": series_index,
                "series_index": series_index,
                "SOPInstanceUID": f"sop-{suffix}",
                "filename": f"slice-{suffix}.dcm",
                "ordered_position": 0,
                "rows": 8,
                "columns": 8,
                "instance_number": 1,
                "geometry_coordinate": 0,
                "transfer_syntax": "1.2.840.10008.1.2.1",
                "photometric_interpretation": "MONOCHROME2",
                "rescale_slope": 1,
                "rescale_intercept": 0,
            }
        )
        selected_rows.append(
            {
                "series_index": series_index,
                "slot": 0,
                "slice_index": series_index,
                "ordered_position": 0,
            }
        )
    paths_and_columns = (
        (index_root / "studies.csv", STUDY_COLUMNS, study_rows),
        (index_root / "series.csv", SERIES_COLUMNS, series_rows),
        (index_root / "slices.csv", SLICE_COLUMNS, slice_rows),
        (index_root / "selected_slices.csv", SELECTED_COLUMNS, selected_rows),
    )
    for path, columns, rows in paths_and_columns:
        _write_csv(path, columns, rows)
    manifest = {
        "schema_version": 1,
        "decoded_images_persisted": False,
        "files": {
            path.name: {
                "columns": list(columns),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path, columns, _ in paths_and_columns
        },
    }
    manifest_path = index_root / "index-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    data_index = {
        "schema_version": 1,
        "current_competition_contract": {"sha256": _sha256(COMPETITION_CONTRACT)},
        "run_provenance": {"outputs": {"manifest_sha256": _sha256(manifest_path)}},
    }
    index_contract_path = root / "data-index.json"
    index_contract_path.write_text(json.dumps(data_index))

    selected = [
        ("study-a", [0] * len(TARGETS)),
        ("study-b", [1] * len(TARGETS)),
    ]
    digest = hashlib.sha256()
    digest.update(
        _canonical_json({"schema": "study-label-jsonl-v1", "targets": list(TARGETS)})
        + b"\n"
    )
    for study_uid, labels in selected:
        digest.update(_canonical_json([study_uid, labels]) + b"\n")
    per_target = {
        target: {"known": 2, "missing": 0, "positive": 1, "negative": 1}
        for target in TARGETS
    }
    supervision = {
        "schema_version": 1,
        "label_contract": {"status": "verified", "target_order": list(TARGETS)},
        "smoke_sufficiency": {
            "decision": "smoke-sufficient",
            "unsupported_targets": [],
        },
        "canonical_cohort": {
            "selection_rule_version": "organizer-gold-all-eligible-v1",
            "cohort_size": 2,
            "label_contract_version": {"sha256": _sha256(COMPETITION_CONTRACT)},
            "per_target": per_target,
            "digest": {
                "canonicalization_version": "study-label-jsonl-v1",
                "value": digest.hexdigest(),
            },
        },
    }
    supervision_path = root / "gold-supervision.json"
    supervision_path.write_text(json.dumps(supervision))
    return competition_root, index_root, index_contract_path, supervision_path


def _train_row(study_uid: str, labels: Sequence[int | None]) -> dict[str, object]:
    row: dict[str, object] = {column: "" for column in TRAIN_COLUMNS}
    row["StudyInstanceUID"] = study_uid
    row["Report"] = "private report text must not be consumed"
    row.update(
        {
            target: "" if label is None else label
            for target, label in zip(TARGETS, labels, strict=True)
        }
    )
    return row


def _study_row(index: int, study_uid: str) -> dict[str, object]:
    return {
        "study_index": index,
        "split": "train",
        "StudyInstanceUID": study_uid,
        "PatientID": f"patient-{index}",
        "series_count": 1,
        "usable_series_count": 1,
        "slice_count": 1,
        "usable_slice_count": 1,
    }


def _write_csv(
    path: Path, columns: Sequence[str], rows: Sequence[dict[str, object]]
) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_index_loading_does_not_resolve_shared_parents_per_slice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    from typing import Any

    from rsna_knee.preprocessing import uniform_indices

    root, index, _, _ = _cohort_fixture(tmp_path)
    with (index / "series.csv").open() as file:
        series_rows = list(csv.DictReader(file))
    series_rows[2].update(
        slice_count="32", usable_slice_count="32", selection_mask="11111111"
    )
    _write_csv(index / "series.csv", SERIES_COLUMNS, list(series_rows))
    with (index / "slices.csv").open() as file:
        template = list(csv.DictReader(file))[2]
    rows = [
        dict(
            template,
            slice_index=str(i),
            filename=f"slice-{i}.dcm",
            ordered_position=str(i),
        )
        for i in range(32)
    ]
    _write_csv(index / "slices.csv", SLICE_COLUMNS, rows)
    _write_csv(
        index / "selected_slices.csv",
        SELECTED_COLUMNS,
        [
            {"series_index": 2, "slot": slot, "slice_index": i, "ordered_position": i}
            for slot, i in enumerate(uniform_indices(32, 8))
        ],
    )
    series = root / "train_series/study-a/series-a"
    series.mkdir(parents=True)
    for row in rows:
        (series / row["filename"]).touch()
    expected = tuple((series / f"slice-{i}.dcm").resolve() for i in range(32))
    root = root.resolve()
    count = 0
    lstat = os.lstat

    def stat(path: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal count
        if Path(path) == root:
            count += 1
        return lstat(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", stat)
    result = load_indexed_studies(root, index, ["study-a"], split="train")
    assert result["study-a"].series[0].paths == expected
    assert count <= 3, f"Repeated common-parent checks: {count}"


@pytest.mark.parametrize(
    "escape", ["relative", "series_link", "slice_link", "inside_link"]
)
def test_index_paths_preserve_containment_and_symlink_semantics(
    tmp_path: Path, escape: str
) -> None:
    root, index, _, _ = _cohort_fixture(tmp_path)
    with (index / "series.csv").open() as file:
        rows = list(csv.DictReader(file))
    series = root / rows[2]["relative_path"]
    series.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    if escape == "relative":
        rows[2]["relative_path"] = "../outside"
        _write_csv(index / "series.csv", SERIES_COLUMNS, list(rows))
    elif escape == "series_link":
        series.symlink_to(outside, target_is_directory=True)
    else:
        series.mkdir()
        target = (root if escape == "inside_link" else outside) / "actual.dcm"
        target.touch()
        (series / "slice-a.dcm").symlink_to(target)
    if escape == "inside_link":
        resolved = load_indexed_studies(root, index, ["study-a"], split="train")
        assert resolved["study-a"].series[0].paths == ((root / "actual.dcm").resolve(),)
    else:
        with pytest.raises(ValueError):
            load_indexed_studies(root, index, ["study-a"], split="train")
