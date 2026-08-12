import argparse
from pathlib import Path

from rsna_knee.kaggle import push_gold_audit_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push the guarded Kaggle organizer-gold audit"
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("build/kaggle/gold-supervision-audit"),
    )
    parser.add_argument(
        "--approve-remote",
        action="store_true",
        help="Confirm explicit approval to push and execute the private CPU audit",
    )
    arguments = parser.parse_args()
    try:
        push_gold_audit_bundle(arguments.bundle, approved=arguments.approve_remote)
    except PermissionError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
