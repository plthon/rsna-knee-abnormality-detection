import argparse
from pathlib import Path

from rsna_knee.public_baseline_bundle import build_public_baseline_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the guarded public MRI baseline notebook"
    )
    parser.add_argument(
        "--wheel-dataset-id",
        default="plthon/rsna-knee-gold-baseline-package",
    )
    parser.add_argument(
        "--weight-dataset-id",
        default="plthon/torchvision-resnet18-imagenet1k-v1",
    )
    parser.add_argument(
        "--kernel-id",
        default="plthon/rsna-knee-a-reproducible-mri-baseline",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/kaggle/public-gold-baseline"),
    )
    parser.add_argument("--approve-gpu", action="store_true")
    arguments = parser.parse_args()
    try:
        bundle = build_public_baseline_bundle(
            Path(__file__).resolve().parents[1],
            arguments.output,
            wheel_dataset_id=arguments.wheel_dataset_id,
            weight_dataset_id=arguments.weight_dataset_id,
            kernel_id=arguments.kernel_id,
            gpu_approved=arguments.approve_gpu,
        )
    except (
        FileNotFoundError,
        PermissionError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        parser.error(str(error))
    print(f"Wheel Dataset: {bundle.wheel_dataset.root}")
    print(f"Notebook: {bundle.notebook_path}")
    print(f"Metadata: {bundle.kernel_metadata_path}")


if __name__ == "__main__":
    main()
