import hashlib
import json
from pathlib import Path
from typing import Any


def test_data_index_contract_records_complete_real_evidence() -> None:
    evidence: dict[str, Any] = json.loads(
        Path("contracts/data-index.json").read_text(encoding="utf-8")
    )
    competition_contract = Path("contracts/competition.json").read_bytes()

    assert evidence["schema_version"] == 1
    assert evidence["competition_contract"] == {
        "path": "contracts/competition.json",
        "sha256": "8ae8eda3d5f5a2c4d256ec23992a9a8bd9358f86a69c3f994d8586f7307e3880",
        "role": "exact contract embedded in and consumed by Kaggle kernel version 1",
    }
    assert evidence["current_competition_contract"] == {
        "path": "contracts/competition.json",
        "sha256": hashlib.sha256(competition_contract).hexdigest(),
        "reviewed_at": "2026-08-11",
        "compatibility": (
            "Issue #17 updated evidence classifications only; the table, join, "
            "path, and target fields consumed by the index are unchanged"
        ),
    }
    assert evidence["source_dataset"] == {
        "platform": "kaggle",
        "kind": "competition",
        "slug": "rsna-knee-abnormality-detection",
        "mount_path": "/kaggle/input/competitions/rsna-knee-abnormality-detection",
        "version": {
            "kind": "verified_snapshot",
            "id": (
                "sha256:8ae8eda3d5f5a2c4d256ec23992a9a8bd9358f86a69c3f994d8586f7307e3880"
            ),
            "platform_version_id": None,
            "platform_version_status": (
                "Kaggle competition mounts expose no immutable dataset version "
                "identifier"
            ),
            "verified_at": "2026-08-09",
            "file_count": 819640,
            "size_bytes": 569763893144,
        },
    }
    assert evidence["indexing_configuration"] == {
        "artifact_schema_version": 1,
        "series_tables": ["train_series.csv", "test_series.csv"],
        "join_fields": ["StudyInstanceUID", "SeriesInstanceUID"],
        "image_path_template": (
            "{train_series|test_series}/{StudyInstanceUID}/"
            "{SeriesInstanceUID}/{SOPInstanceUID}.dcm"
        ),
        "header_read": {
            "reader": "pydicom.dcmread",
            "stop_before_pixels": True,
            "specific_tags": [
                "SOPInstanceUID",
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "PatientID",
                "Rows",
                "Columns",
                "InstanceNumber",
                "ImagePositionPatient",
                "ImageOrientationPatient",
                "PhotometricInterpretation",
                "RescaleSlope",
                "RescaleIntercept",
            ],
        },
        "ordering_priority": ["geometry", "instance_number", "filename"],
        "selection": {
            "policy": "deterministic_uniform",
            "slice_count": 8,
            "short_series": "mask_without_duplication",
        },
        "representative_series_per_split": 4,
        "preprocessing": {
            "lower_percentile": 1,
            "upper_percentile": 99,
            "image_size": 224,
            "resize": "bilinear_aspect_preserving_center_pad",
            "normalization_status": "provisional",
        },
    }
    assert evidence["run_provenance"] == {
        "kind": "private_kaggle_kernel_output",
        "kernel_id": "plthon/rsna-knee-data-index",
        "kernel_version": 1,
        "status": "complete",
        "url": "https://www.kaggle.com/code/plthon/rsna-knee-data-index",
        "source": {
            "base_git_commit": "adf8cd9719239fde685dc915ff0ac5e88d7c29de",
            "worktree_dirty": True,
            "jupytext_source": "notebooks/index_real_data.py",
            "jupytext_source_sha256": (
                "20fc89256841148180cdf76de1d3b1b172ac90eceda3b03cafa27e3d1fcf6f5a"
            ),
            "rendered_notebook_sha256": (
                "7f57cf7d3f909a0aa4e21e49e4d1c93e87ffeab90cab5317ec17b873bf882f67"
            ),
            "kernel_metadata_sha256": (
                "98ec466e4a6b62d8a968427c868151ff67a8e9cdbe6550587bf1b7eb2193062f"
            ),
        },
        "execution": {
            "cpu_only": True,
            "gpu_enabled": False,
            "tpu_enabled": False,
            "internet_enabled": False,
            "runtime_seconds_to_verified_output": 10063.58520553,
        },
        "outputs": {
            "manifest_sha256": (
                "85970899c44c1325ab6d1fe8983dc9e472e6a854d8b6bb8fb65506ec91480caa"
            ),
            "statistics_sha256": (
                "4be3434249f73c287c611bb6de030b273975b3b311250395a676a87556ea917c"
            ),
            "log_sha256": (
                "6a5f89429ccba19a23ca6f2c925da075de27a27b0dc6d2b76714026970022de1"
            ),
        },
    }
    assert evidence["counts"] == {
        "studies": 4410,
        "series": 24386,
        "slices": 819635,
        "usable_slices": 819635,
        "selected_slices": 195088,
        "structured_errors": 0,
    }
    assert sum(evidence["ordering"].values()) == evidence["counts"]["series"]
    assert (
        sum(evidence["decoder"]["transfer_syntaxes"].values())
        == evidence["counts"]["slices"]
    )
    assert evidence["decoder"] == {
        "required_dependency": "pydicom",
        "optional_plugins": [],
        "sampled_decode_failures": 0,
        "transfer_syntaxes": {"1.2.840.10008.1.2.1": 819635},
    }
    assert evidence["preprocessing"]["representative_series_checked"] == 8
    assert evidence["preprocessing"]["representative_series_succeeded"] == 8
    assert evidence["preprocessing"]["all_outputs_finite"] is True
    assert evidence["preprocessing"]["output_shape"] == [8, 224, 224]
    assert evidence["artifact"]["decoded_images_persisted"] is False
    assert set(evidence["artifact"]["files"]) == {
        "studies.csv",
        "series.csv",
        "slices.csv",
        "selected_slices.csv",
        "statistics.json",
        "errors.csv",
    }
    assert (
        evidence["artifact"]["files"]["statistics.json"]["sha256"]
        == evidence["run_provenance"]["outputs"]["statistics_sha256"]
    )
    assert evidence["safeguards"] == {
        "cpu_only": True,
        "gpu_enabled": False,
        "internet_enabled": False,
        "headers_completed_before_pixel_decoding": True,
        "raw_dicoms_downloaded": False,
        "decoded_images_downloaded": False,
        "decoded_images_persisted": False,
    }
