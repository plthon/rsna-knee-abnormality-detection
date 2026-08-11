import argparse
from pathlib import Path

from rsna_knee.kaggle import build_public_notebook_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the standalone public-facing DICOM notebook"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/kaggle/public-dicom-ordering"),
    )
    arguments = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    bundle = build_public_notebook_bundle(project_root, arguments.output)
    print(f"Notebook: {bundle.notebook_path}")
    print(f"Metadata: {bundle.metadata_path}")


if __name__ == "__main__":
    main()
