import argparse
from pathlib import Path

from rsna_knee import export_public_gold_audit_contract


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export only the public-safe issue #7 aggregate contract"
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("contracts/gold-supervision.json"),
    )
    arguments = parser.parse_args()
    export_public_gold_audit_contract(arguments.artifact, arguments.output)
    print(arguments.output)


if __name__ == "__main__":
    main()
