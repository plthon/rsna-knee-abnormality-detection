import argparse
from pathlib import Path

from rsna_knee.training_bundle import build_training_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the exact offline weight Dataset and guarded GPU smoke run"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to the exact ResNet18 IMAGENET1K_V1 state dict",
    )
    parser.add_argument(
        "--dataset-id",
        default="plthon/torchvision-resnet18-imagenet1k-v1",
    )
    parser.add_argument(
        "--kernel-id",
        default="plthon/rsna-knee-gold-model-smoke",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/kaggle/gold-model-smoke"),
    )
    parser.add_argument(
        "--approve-gpu",
        action="store_true",
        help="Explicitly approve rendering GPU-enabled metadata",
    )
    arguments = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    try:
        bundle = build_training_bundle(
            project_root,
            arguments.output,
            weight_path=arguments.weights,
            dataset_id=arguments.dataset_id,
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
    print(f"Weight Dataset: {bundle.weight_dataset.root}")
    print(f"Kernel bundle: {bundle.kernel_root}")


if __name__ == "__main__":
    main()
