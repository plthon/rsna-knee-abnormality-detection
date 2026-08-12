import argparse
from pathlib import Path

from rsna_knee.baseline_bundle import build_baseline_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the guarded issue #9 Kaggle holdout baseline"
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--dataset-id", default="plthon/torchvision-resnet18-imagenet1k-v1"
    )
    parser.add_argument("--kernel-id", default="plthon/rsna-knee-gold-holdout-baseline")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/kaggle/gold-holdout-baseline"),
    )
    parser.add_argument("--approve-gpu", action="store_true")
    arguments = parser.parse_args()
    try:
        bundle = build_baseline_bundle(
            Path(__file__).resolve().parents[1],
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
