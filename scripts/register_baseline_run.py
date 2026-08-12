import argparse
import json
from pathlib import Path

from rsna_knee.baseline import verify_baseline_artifact
from rsna_knee.kaggle_run import download_and_register_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download, verify, and register the issue #9 baseline"
    )
    parser.add_argument("--kernel-id", default="plthon/rsna-knee-gold-holdout-baseline")
    parser.add_argument(
        "--download-root",
        type=Path,
        default=Path(".artifacts/kaggle-downloads/gold-holdout-baseline"),
    )
    parser.add_argument(
        "--registry-root", type=Path, default=Path(".artifacts/experiments")
    )
    arguments = parser.parse_args()
    registered = download_and_register_run(
        kernel_id=arguments.kernel_id,
        download_root=arguments.download_root,
        registry_root=arguments.registry_root,
        selected_artifacts=(
            "best-checkpoint.pt",
            "last-checkpoint.pt",
            "holdout-predictions-private.csv",
            "baseline-report.json",
        ),
    )
    print(
        json.dumps(verify_baseline_artifact(registered.root), indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
