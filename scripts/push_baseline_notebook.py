import argparse
from pathlib import Path

from rsna_knee.baseline_bundle import load_baseline_bundle, push_baseline_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the guarded issue #9 Kaggle holdout baseline"
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("build/kaggle/gold-holdout-baseline"),
    )
    parser.add_argument("--approve-remote", action="store_true")
    parser.add_argument("--approve-gpu", action="store_true")
    parser.add_argument("--reuse-ready-dataset", action="store_true")
    arguments = parser.parse_args()
    try:
        push_baseline_bundle(
            load_baseline_bundle(arguments.bundle),
            approved=arguments.approve_remote,
            gpu_approved=arguments.approve_gpu,
            reuse_ready_dataset=arguments.reuse_ready_dataset,
        )
    except (
        FileNotFoundError,
        PermissionError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
