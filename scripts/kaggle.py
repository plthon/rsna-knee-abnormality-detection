import argparse
import json
from pathlib import Path

from rsna_knee.kaggle import (
    build_public_baseline_bundle,
    build_public_label_audit_bundle,
    build_public_notebook_bundle,
    build_report_llm_bundle,
    build_reports_bundle,
    load_kaggle_bundle,
    pull_kaggle_output,
    push_kaggle_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or push a guarded Kaggle bundle"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="render a notebook bundle locally")
    build.add_argument(
        "target",
        choices=(
            "ordering",
            "baseline",
            "reports",
            "report-llm",
            "public-label-audit",
        ),
    )
    build.add_argument("--output", type=Path)
    build.add_argument("--stage", choices=("audit", "smoke", "extract"))
    build.add_argument(
        "--wheel-dataset-id", default="plthon/rsna-knee-gold-baseline-package"
    )
    build.add_argument(
        "--weight-dataset-id", default="plthon/torchvision-resnet18-imagenet1k-v1"
    )
    build.add_argument(
        "--kernel-id", default="plthon/rsna-knee-a-reproducible-mri-baseline"
    )
    build.add_argument("--approve-gpu", action="store_true")

    push = subparsers.add_parser("push", help="push a previously built bundle")
    push.add_argument("bundle", type=Path)
    push.add_argument("--approve-remote", action="store_true")
    push.add_argument("--approve-gpu", action="store_true")
    push.add_argument("--reuse-ready-dataset", action="store_true")

    pull = subparsers.add_parser("pull", help="retrieve selected kernel outputs")
    pull.add_argument("bundle", type=Path)
    pull.add_argument("--output", type=Path)
    pull.add_argument("--filename", action="append", required=True)

    arguments = parser.parse_args()
    project_root = Path.cwd()
    if arguments.command == "build":
        if arguments.output is None:
            names = {
                "ordering": "public-dicom-ordering",
                "baseline": "public-gold-baseline",
                "reports": f"reports-{arguments.stage}",
                "report-llm": f"report-llm-{arguments.stage}",
                "public-label-audit": "public-label-audit",
            }
            arguments.output = Path("build/kaggle") / names[arguments.target]
        if arguments.target == "ordering":
            bundle = build_public_notebook_bundle(project_root, arguments.output)
        elif arguments.target == "baseline":
            bundle = build_public_baseline_bundle(
                project_root,
                arguments.output,
                wheel_dataset_id=arguments.wheel_dataset_id,
                weight_dataset_id=arguments.weight_dataset_id,
                kernel_id=arguments.kernel_id,
                gpu_approved=arguments.approve_gpu,
            )
        elif arguments.target == "reports":
            if arguments.stage is None:
                parser.error("reports build requires --stage audit|extract")
            if arguments.stage not in {"audit", "extract"}:
                parser.error("reports build requires --stage audit|extract")
            bundle = build_reports_bundle(
                project_root, arguments.output, stage=arguments.stage
            )
        elif arguments.target == "report-llm":
            if arguments.stage not in {"smoke", "extract"}:
                parser.error("report-llm build requires --stage smoke|extract")
            bundle = build_report_llm_bundle(
                project_root, arguments.output, stage=arguments.stage
            )
        else:
            if arguments.stage != "audit":
                parser.error("public-label-audit build requires --stage audit")
            bundle = build_public_label_audit_bundle(
                project_root, arguments.output, stage=arguments.stage
            )
        print(bundle.root)
        return

    if arguments.command == "pull":
        bundle = load_kaggle_bundle(arguments.bundle)
        metadata = json.loads(bundle.kernel_metadata_path.read_text())
        kernel_id = metadata["id"]
        if not isinstance(kernel_id, str):
            raise TypeError("Kaggle kernel ID is malformed")
        output = arguments.output or (
            Path(".artifacts/kaggle") / kernel_id.partition("/")[2]
        )
        results = pull_kaggle_output(
            bundle,
            output,
            artifact_root=project_root / ".artifacts",
            filenames=tuple(arguments.filename),
        )
        print(output)
        for result in results:
            print(result)
        return

    dataset_result, kernel_result = push_kaggle_bundle(
        load_kaggle_bundle(arguments.bundle),
        approved=arguments.approve_remote,
        gpu_approved=arguments.approve_gpu,
        reuse_ready_dataset=arguments.reuse_ready_dataset,
    )
    print(kernel_result)
    if dataset_result is not None:
        print(dataset_result)


if __name__ == "__main__":
    main()
