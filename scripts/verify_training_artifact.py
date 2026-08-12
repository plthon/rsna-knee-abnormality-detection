import argparse
import json
from pathlib import Path

from rsna_knee.training import verify_training_artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a completed issue #8 Kaggle training artifact"
    )
    parser.add_argument("artifact", type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            verify_training_artifact(arguments.artifact),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
