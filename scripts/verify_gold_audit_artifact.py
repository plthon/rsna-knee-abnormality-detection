import argparse
import json
from pathlib import Path

from rsna_knee import verify_gold_audit_artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a private issue #7 gold-audit artifact"
    )
    parser.add_argument("artifact", type=Path)
    arguments = parser.parse_args()
    summary = verify_gold_audit_artifact(arguments.artifact)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
