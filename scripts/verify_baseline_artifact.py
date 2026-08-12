import argparse
import json
from pathlib import Path

from rsna_knee.baseline import verify_baseline_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify an issue #9 baseline artifact")
    parser.add_argument("artifact", type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            verify_baseline_artifact(arguments.artifact), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
