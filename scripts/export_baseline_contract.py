import argparse
import json
from pathlib import Path

from rsna_knee.baseline import export_public_baseline_contract


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export public-safe aggregate issue #9 baseline evidence"
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("contracts/gold-baseline.json")
    )
    arguments = parser.parse_args()
    print(
        json.dumps(
            export_public_baseline_contract(arguments.artifact, arguments.output),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
