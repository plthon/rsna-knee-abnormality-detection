import argparse
from pathlib import Path

from rsna_knee.kaggle_run import load_run_bundle, push_run_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push the wheel Dataset and execute the guarded Kaggle run"
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("build/kaggle/synthetic-run"),
    )
    parser.add_argument(
        "--approve-remote",
        action="store_true",
        help="Explicitly approve private Dataset creation and notebook execution",
    )
    parser.add_argument(
        "--approve-gpu",
        action="store_true",
        help="Separately approve a GPU-enabled kernel if metadata requests one",
    )
    parser.add_argument(
        "--reuse-ready-dataset",
        action="store_true",
        help="Reuse an already-ready Dataset instead of attempting to recreate it",
    )
    arguments = parser.parse_args()
    try:
        push_run_bundle(
            load_run_bundle(arguments.bundle),
            approved=arguments.approve_remote,
            gpu_approved=arguments.approve_gpu,
            reuse_ready_dataset=arguments.reuse_ready_dataset,
        )
    except (FileNotFoundError, PermissionError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
