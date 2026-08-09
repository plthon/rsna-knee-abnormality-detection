import argparse
from pathlib import Path

from rsna_knee.kaggle_run import (
    build_run_bundle,
    load_run_bundle,
    render_run_kernel,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the offline wheel Dataset and guarded Kaggle run"
    )
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Resolved private Dataset ID in owner/slug form",
    )
    parser.add_argument(
        "--kernel-id",
        required=True,
        help="Resolved private notebook ID in owner/slug form",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/kaggle/synthetic-run"),
    )
    parser.add_argument("--enable-gpu", action="store_true")
    parser.add_argument(
        "--reuse-wheel-dataset",
        action="store_true",
        help="Rerender only the kernel while preserving the existing wheel",
    )
    parser.add_argument(
        "--approve-gpu",
        action="store_true",
        help="Separately approve generating GPU-enabled metadata",
    )
    arguments = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    try:
        if arguments.reuse_wheel_dataset:
            wheel_dataset = load_run_bundle(arguments.output).wheel_dataset
            bundle = render_run_kernel(
                project_root,
                arguments.output,
                wheel_dataset=wheel_dataset,
                kernel_id=arguments.kernel_id,
                enable_gpu=arguments.enable_gpu,
                gpu_approved=arguments.approve_gpu,
            )
        else:
            bundle = build_run_bundle(
                project_root,
                arguments.output,
                dataset_id=arguments.dataset_id,
                kernel_id=arguments.kernel_id,
                enable_gpu=arguments.enable_gpu,
                gpu_approved=arguments.approve_gpu,
            )
    except (FileNotFoundError, PermissionError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(f"Wheel Dataset: {bundle.wheel_dataset.root}")
    print(f"Kernel bundle: {bundle.kernel_root}")
    print(f"Wheel provenance: {bundle.wheel_dataset.provenance_path}")


if __name__ == "__main__":
    main()
