import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
CONTRACT_PATH = PROJECT_ROOT / "contracts" / "competition.json"


def test_canonical_competition_contract_is_complete_and_machine_readable() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())

    assert contract["schema_version"] == 1
    assert contract["competition"]["slug"] == "rsna-knee-abnormality-detection"
    assert [criterion["id"] for criterion in contract["acceptance_criteria"]] == [
        "AC1",
        "AC2",
        "AC3",
        "AC4",
        "AC5",
        "AC6",
        "AC7",
        "AC8",
    ]
    assert {
        criterion["initial_classification"]
        for criterion in contract["acceptance_criteria"]
    } <= {
        "already_satisfied",
        "requires_additional_lightweight_inspection",
        "unresolved_by_available_evidence",
    }
    assert {
        criterion["final_status"] for criterion in contract["acceptance_criteria"]
    } == {"satisfied"}

    target_order = [
        "ACL",
        "MCL",
        "Medial Meniscus",
        "Lateral Meniscus",
        "Medial OA",
        "Lateral OA",
        "PF OA",
        "Effusion",
        "Synovitis",
        "Baker's",
        "Contusion",
        "Fracture",
    ]
    assert contract["targets"]["order"] == target_order
    assert contract["metric"] == {
        "aggregation": "unweighted_mean_across_12_targets",
        "name": "macro_averaged_roc_auc",
        "objective": "maximize",
        "status": "verified",
    }
    assert contract["submission"] == {
        "filename": "submission.csv",
        "format": "csv",
        "identifier_columns": ["StudyInstanceUID"],
        "inference_api": {
            "basis": "The official interface is a notebook-generated submission.csv; no inference API is specified.",
            "required": False,
            "status": "verified",
        },
        "ordered_columns": ["StudyInstanceUID", *target_order],
        "row_granularity": "study",
        "status": "verified",
    }

    assert contract["leakage"]["highest_safe_available_grouping_entity"] == "PatientID"
    assert set(contract["rules"]) == {
        "external_data",
        "external_report_processing",
        "generative_ai",
        "pretrained_weights",
    }
    assert {rule["status"] for rule in contract["rules"].values()} <= {
        "permitted",
        "prohibited",
        "unresolved",
    }

    runtime = contract["platform"]["observed_runtime"]
    assert set(runtime["packages"]) >= {"pydicom", "torch", "torchvision"}
    assert set(runtime) >= {
        "accelerator",
        "cuda_version",
        "kaggle_image_git_commit",
        "packages",
        "python",
    }

    assumption_statuses = {
        assumption["status"] for assumption in contract["assumptions"].values()
    }
    assert assumption_statuses <= {"verified", "rejected", "unresolved"}
    assert all(
        assumption.get("basis") for assumption in contract["assumptions"].values()
    )

    for table in contract["data"]["tables"].values():
        assert set(table) >= {
            "columns",
            "duplicate_identifier_rows",
            "identifier_columns",
            "missing_by_column",
            "row_count",
            "size_bytes",
        }

    train = contract["data"]["tables"]["train.csv"]
    assert train["missing_by_column"]["Report"] == 0
    assert {train["missing_by_column"][target] for target in target_order} == {4349}
    train_series_ids = contract["data"]["tables"]["train_series.csv"][
        "identifier_cardinality"
    ]["SeriesInstanceUID"]
    assert train_series_ids == {
        "duplicate_nonblank_rows": 0,
        "maximum_rows_per_value": 1,
        "unique_nonblank": 24371,
    }
    assert contract["reports"]["observed_languages"]["status"] == "unresolved"
    assert contract["dicom"]["transfer_syntaxes"]["mounted_snapshot"] == {
        "status": "verified",
        "headers_indexed": 819635,
        "frequencies": {"1.2.840.10008.1.2.1": 819635},
        "evidence": "full_corpus_data_index_v1",
    }
    assert (
        contract["assumptions"][
            "all_mounted_snapshot_dicoms_use_explicit_vr_little_endian"
        ]["status"]
        == "verified"
    )
    assert (
        contract["assumptions"]["all_dicoms_use_explicit_vr_little_endian"]["status"]
        == "unresolved"
    )
    assert (
        contract["assumptions"]["report_pseudo_labelling_is_needed"]["status"]
        == "unresolved"
    )
    assert (
        contract["assumptions"]["global_sop_instance_uid_uniqueness"]["status"]
        == "unresolved"
    )

    assert contract["data_governance"] == {
        "competition_data_committed": False,
        "competition_source_files_downloaded_by_issue_4": False,
        "derived_inspection_artifacts_committed": False,
        "derived_inspection_artifacts_preexisting_locally": True,
        "full_reports_downloaded": False,
        "raw_dicoms_downloaded": False,
    }
