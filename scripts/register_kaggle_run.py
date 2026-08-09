import argparse
from pathlib import Path

from rsna_knee.kaggle_run import download_and_register_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download selected Kaggle outputs into the local run registry"
    )
    parser.add_argument(
        "--kernel-id",
        required=True,
        help="Executed notebook ID in owner/slug form",
    )
    parser.add_argument(
        "--download-root",
        type=Path,
        default=Path("build/kaggle/downloaded-run"),
    )
    parser.add_argument(
        "--registry-root",
        type=Path,
        default=Path(".artifacts/experiments"),
    )
    parser.add_argument(
        "--artifact",
        action="append",
        dest="artifacts",
        help="Manifest-relative output to register; may be repeated",
    )
    arguments = parser.parse_args()
    try:
        registered = download_and_register_run(
            kernel_id=arguments.kernel_id,
            download_root=arguments.download_root,
            registry_root=arguments.registry_root,
            selected_artifacts=arguments.artifacts or ["synthetic-output.npz"],
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(f"Registered run: {registered.run_id}")
    print(f"Manifest: {registered.manifest_path}")
    print(f"Registry: {registered.registry_path}")


if __name__ == "__main__":
    main()
