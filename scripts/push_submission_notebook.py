import argparse
from pathlib import Path

from rsna_knee.kaggle import push_submission_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push the guarded Kaggle constant baseline submission"
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("build/kaggle/constant-baseline-submission"),
    )
    parser.add_argument(
        "--approve-remote",
        action="store_true",
        help="Confirm explicit approval to push and execute the private CPU notebook",
    )
    arguments = parser.parse_args()
    try:
        push_submission_bundle(arguments.bundle, approved=arguments.approve_remote)
    except PermissionError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
