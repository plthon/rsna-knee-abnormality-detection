import argparse
import json
from pathlib import Path

from rsna_knee.raptor_reference import verify_raptor_reference_scoring_artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify an issue #36 Raptor hidden-inference artifact"
    )
    parser.add_argument("artifact", type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            verify_raptor_reference_scoring_artifact(arguments.artifact),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
