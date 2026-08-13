import argparse
from pathlib import Path

from rsna_knee.public_baseline_bundle import (
    load_public_baseline_bundle,
    push_public_baseline_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute the public MRI baseline privately on Kaggle"
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("build/kaggle/public-gold-baseline"),
    )
    parser.add_argument("--approve-remote", action="store_true")
    parser.add_argument("--approve-gpu", action="store_true")
    parser.add_argument("--reuse-ready-dataset", action="store_true")
    arguments = parser.parse_args()
    try:
        push_public_baseline_bundle(
            load_public_baseline_bundle(arguments.bundle),
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
