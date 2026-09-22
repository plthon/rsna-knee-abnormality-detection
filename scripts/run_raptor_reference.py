from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rsna_knee.raptor_reference import run_raptor_reference


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen public Raptor reference locally"
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    arguments = parser.parse_args()
    device = None if arguments.device == "auto" else torch.device(arguments.device)
    outcome = run_raptor_reference(
        input_root=arguments.input_root,
        work_root=arguments.work_root,
        checkpoint_path=arguments.checkpoint,
        device=device,
    )
    print(
        json.dumps(
            {
                "fallbacks": str(outcome.submission.fallbacks_path),
                "manifest": str(outcome.submission.manifest_path),
                "raw_failure_count": outcome.raw_failure_count,
                "report": str(outcome.report_path),
                "submission": str(outcome.submission.submission_path),
                "validation": str(outcome.submission.validation_report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
