import argparse
from pathlib import Path

from rsna_knee.kaggle import (
    build_public_baseline_bundle,
    build_public_notebook_bundle,
    load_kaggle_bundle,
    push_kaggle_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or push a guarded Kaggle bundle"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="render a notebook bundle locally")
    build.add_argument("target", choices=("ordering", "baseline"))
    build.add_argument("--output", type=Path)
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

    arguments = parser.parse_args()
    project_root = Path.cwd()
    if arguments.command == "build":
        if arguments.output is None:
            arguments.output = Path("build/kaggle") / (
                "public-dicom-ordering"
                if arguments.target == "ordering"
                else "public-gold-baseline"
            )
        if arguments.target == "ordering":
            bundle = build_public_notebook_bundle(project_root, arguments.output)
        else:
            bundle = build_public_baseline_bundle(
                project_root,
                arguments.output,
                wheel_dataset_id=arguments.wheel_dataset_id,
                weight_dataset_id=arguments.weight_dataset_id,
                kernel_id=arguments.kernel_id,
                gpu_approved=arguments.approve_gpu,
            )
        print(bundle.root)
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
