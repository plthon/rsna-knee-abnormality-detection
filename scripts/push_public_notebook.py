import argparse
from pathlib import Path

from rsna_knee.kaggle import push_public_notebook_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute the public-facing notebook privately on Kaggle"
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("build/kaggle/public-dicom-ordering"),
    )
    parser.add_argument(
        "--approve-remote",
        action="store_true",
        help="Confirm approval to push and execute the private CPU notebook",
    )
    arguments = parser.parse_args()
    try:
        push_public_notebook_bundle(
            arguments.bundle,
            approved=arguments.approve_remote,
        )
    except PermissionError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
