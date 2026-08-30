import argparse
import json
from pathlib import Path

from rsna_knee.kaggle import (
    build_combined_baseline_bundle,
    build_combined_scoring_bundle,
    build_dinov2_scoring_bundle,
    build_dinov2_training_bundle,
    build_plane_gated_scoring_bundle,
    build_plane_gating_bundle,
    build_public_baseline_bundle,
    build_public_label_audit_bundle,
    build_public_notebook_bundle,
    build_report_llm_bundle,
    build_report_source_comparison_bundle,
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
            "report-source-comparison",
            "combined-baseline",
            "combined-scoring",
            "plane-gating",
            "plane-gating-scoring",
            "dinov2-training",
            "dinov2-scoring",
        ),
    )
    build.add_argument("--output", type=Path)
    build.add_argument("--stage", choices=("audit", "smoke", "extract", "compare"))
    build.add_argument(
        "--wheel-dataset-id", default="plthon/rsna-knee-gold-baseline-package"
    )
    build.add_argument(
        "--weight-dataset-id", default="plthon/torchvision-resnet18-imagenet1k-v1"
    )
    build.add_argument("--kernel-id", default=None)
    build.add_argument("--approve-gpu", action="store_true")
    build.add_argument("--checkpoint-dataset-id")
    build.add_argument("--checkpoint-sha256")
    build.add_argument("--approve-retained-gate", action="store_true")

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
                "report-source-comparison": "report-source-comparison",
                "combined-baseline": "gold-public-scoring",
                "combined-scoring": "gold-public-scoring",
                "plane-gating": "plane-gating",
                "plane-gating-scoring": "plane-gated-scoring",
                "dinov2-training": "dinov2-training",
                "dinov2-scoring": "dinov2-scoring",
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
                kernel_id=arguments.kernel_id
                or "plthon/rsna-knee-a-reproducible-mri-baseline",
                gpu_approved=arguments.approve_gpu,
            )
        elif arguments.target in {"plane-gating-scoring", "dinov2-scoring"}:
            if not arguments.checkpoint_dataset_id or not arguments.checkpoint_sha256:
                parser.error(
                    f"{arguments.target} requires --checkpoint-dataset-id and "
                    "--checkpoint-sha256"
                )
            scoring_builder = (
                build_dinov2_scoring_bundle
                if arguments.target == "dinov2-scoring"
                else build_plane_gated_scoring_bundle
            )
            bundle = scoring_builder(
                project_root,
                arguments.output,
                checkpoint_dataset_id=arguments.checkpoint_dataset_id,
                checkpoint_sha256=arguments.checkpoint_sha256,
                validation_kept=arguments.approve_retained_gate,
                kernel_id=arguments.kernel_id
                or (
                    "plthon/rsna-knee-dinov2-scoring"
                    if arguments.target == "dinov2-scoring"
                    else "plthon/rsna-knee-plane-gated-scoring"
                ),
                gpu_approved=arguments.approve_gpu,
            )
        elif arguments.target == "dinov2-training":
            bundle = build_dinov2_training_bundle(
                project_root,
                arguments.output,
                kernel_id=arguments.kernel_id or "plthon/rsna-knee-dinov2-vit-s-14",
                gpu_approved=arguments.approve_gpu,
            )
        elif arguments.target in {
            "combined-baseline",
            "combined-scoring",
            "plane-gating",
        }:
            builder = (
                build_plane_gating_bundle
                if arguments.target == "plane-gating"
                else (
                    build_combined_baseline_bundle
                    if arguments.target == "combined-baseline"
                    else build_combined_scoring_bundle
                )
            )
            bundle = builder(
                project_root,
                arguments.output,
                kernel_id=arguments.kernel_id
                or (
                    "plthon/rsna-knee-plane-gating"
                    if arguments.target == "plane-gating"
                    else "plthon/rsna-knee-gold-public-scoring"
                ),
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
        elif arguments.target == "report-source-comparison":
            if arguments.stage != "compare":
                parser.error("report-source-comparison build requires --stage compare")
            bundle = build_report_source_comparison_bundle(
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
