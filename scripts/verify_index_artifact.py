import argparse
import json
from pathlib import Path

from rsna_knee import verify_index_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a downloaded data index")
    parser.add_argument("artifact", type=Path)
    arguments = parser.parse_args()
    statistics = verify_index_artifact(arguments.artifact)
    print(json.dumps(statistics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
