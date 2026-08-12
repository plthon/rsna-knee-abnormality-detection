import argparse
import json
from pathlib import Path

from rsna_knee.competition_contract import load_verified_submission_contract
from rsna_knee.training import run_local_cpu_check


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run issue #8's synthetic local CPU training/reload check"
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path(".artifacts/local-training-check")
    )
    arguments = parser.parse_args()
    targets = load_verified_submission_contract().target_names
    result = run_local_cpu_check(
        weight_path=arguments.weights,
        work_root=arguments.output,
        target_names=targets,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
