import argparse
from pathlib import Path

from rsna_knee.kaggle import build_index_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the guarded Kaggle data index")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/kaggle/index-real-data"),
    )
    arguments = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    bundle = build_index_bundle(project_root, arguments.output)
    print(f"Notebook: {bundle.notebook_path}")
    print(f"Metadata: {bundle.metadata_path}")


if __name__ == "__main__":
    main()
