import argparse
from pathlib import Path

from rsna_knee.training_bundle import load_training_bundle, push_training_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the private weight Dataset and run the guarded GPU smoke"
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("build/kaggle/gold-model-smoke"),
    )
    parser.add_argument("--approve-remote", action="store_true")
    parser.add_argument("--approve-gpu", action="store_true")
    parser.add_argument(
        "--reuse-ready-dataset",
        action="store_true",
        help="Reuse an already-ready exact weight Dataset version",
    )
    arguments = parser.parse_args()
    try:
        push_training_bundle(
            load_training_bundle(arguments.bundle),
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
