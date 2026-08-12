import argparse
import json
from pathlib import Path

from rsna_knee.kaggle_run import download_and_register_run
from rsna_knee.training import verify_training_artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download, verify, and register the issue #8 Kaggle smoke run"
    )
    parser.add_argument("--kernel-id", default="plthon/rsna-knee-gold-model-smoke")
    parser.add_argument(
        "--download-root",
        type=Path,
        default=Path(".artifacts/kaggle-downloads/gold-model-smoke"),
    )
    parser.add_argument(
        "--registry-root",
        type=Path,
        default=Path(".artifacts/experiments"),
    )
    arguments = parser.parse_args()
    registered = download_and_register_run(
        kernel_id=arguments.kernel_id,
        download_root=arguments.download_root,
        registry_root=arguments.registry_root,
        selected_artifacts=(
            "checkpoint.pt",
            "training-report.json",
            "submission.csv",
            "submission-fallbacks.csv",
            "submission-validation.json",
        ),
    )
    summary = verify_training_artifact(registered.root)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
