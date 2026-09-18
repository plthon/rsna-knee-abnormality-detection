import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    mode: Literal["local", "kaggle"]
    input_root: Path
    work_root: Path


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    slice_count: int
    lower_percentile: float
    upper_percentile: float
    image_size: int = 224
    spatial_mode: str = "whole_image"
    physical_fov_mm: float = 130.0
    spacing_fallback: str = "whole_image_resize_pad"
    input_representation: Literal["same_slice_rgb", "adjacent_slice_2_5d"] = (
        "same_slice_rgb"
    )

    sampling_layout: str = "full_span"

    def __post_init__(self) -> None:
        if self.sampling_layout not in {"full_span", "central_80_three_anchor"}:
            raise ValueError("Unsupported sampling layout")
        if self.sampling_layout == "central_80_three_anchor" and (
            self.slice_count != 3
            or self.input_representation != "adjacent_slice_2_5d"
            or self.spatial_mode != "physical_crop"
            or self.physical_fov_mm != 130
            or self.image_size != 224
        ):
            raise ValueError("Clipped window requires three 130 mm / 224 px triplets")
        if self.spatial_mode not in {"whole_image", "physical_crop"}:
            raise ValueError("Unknown spatial preprocessing mode")
        if not math.isfinite(self.physical_fov_mm) or self.physical_fov_mm <= 0:
            raise ValueError("Physical FOV must be finite and positive")
        if self.spacing_fallback != "whole_image_resize_pad":
            raise ValueError("Unsupported PixelSpacing fallback policy")
        if self.input_representation not in {
            "same_slice_rgb",
            "adjacent_slice_2_5d",
        }:
            raise ValueError("Unsupported input representation")

    def record(self) -> dict[str, object]:
        # Preserve the frozen baseline checkpoint identity.
        result: dict[str, object] = {
            "slice_count": self.slice_count,
            "lower_percentile": self.lower_percentile,
            "upper_percentile": self.upper_percentile,
            "image_size": self.image_size,
        }
        if self.spatial_mode == "physical_crop":
            result.update(
                spatial_mode=self.spatial_mode,
                physical_fov_mm=self.physical_fov_mm,
                spacing_fallback=self.spacing_fallback,
                crop_contract="center_round_ties_even_extra_bottom_right_zero_v1",
            )
        if self.input_representation == "adjacent_slice_2_5d":
            result.update(
                input_representation=self.input_representation,
                channel_order=["previous", "center", "next"],
                neighbor_policy="adjacent_ordered_position_clamp_v1",
                neighbor_failure_policy="substitute_processed_center_v1",
                normalization_population="sampled_center_anchors_only_v1",
            )
        if self.sampling_layout == "central_80_three_anchor":
            result.update(
                sampling_layout=self.sampling_layout,
                representation_identity="issue_33_clipped_window_2_5d_v1",
                sampling_window=[0.10, 0.90],
                anchor_policy="floor_fraction_n_minus_1_clamp_interior_linspace_floor_deduplicate_v1",
                normalization_population="three_selected_centers_only_v1",
            )
        return result


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    paths: ProjectPaths
    preprocessing: PreprocessingConfig


def load_config(path: Path) -> PipelineConfig:
    with path.open("rb") as file:
        values = tomllib.load(file)

    path_values = values["paths"]
    mode = path_values["mode"]
    if mode not in {"local", "kaggle"}:
        raise ValueError(f"Unsupported path mode: {mode}")

    preprocessing_values = values["preprocessing"]
    preprocessing = PreprocessingConfig(
        slice_count=preprocessing_values["slice_count"],
        lower_percentile=preprocessing_values["lower_percentile"],
        upper_percentile=preprocessing_values["upper_percentile"],
        image_size=preprocessing_values.get("image_size", 224),
        spatial_mode=preprocessing_values.get("spatial_mode", "whole_image"),
        physical_fov_mm=preprocessing_values.get("physical_fov_mm", 130.0),
        spacing_fallback=preprocessing_values.get(
            "spacing_fallback", "whole_image_resize_pad"
        ),
        input_representation=preprocessing_values.get(
            "input_representation", "same_slice_rgb"
        ),
        sampling_layout=preprocessing_values.get("sampling_layout", "full_span"),
    )
    if preprocessing.slice_count <= 0:
        raise ValueError("slice_count must be positive")
    if not 0 <= preprocessing.lower_percentile < preprocessing.upper_percentile <= 100:
        raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")
    if preprocessing.image_size <= 0:
        raise ValueError("image_size must be positive")

    return PipelineConfig(
        paths=ProjectPaths(
            mode=mode,
            input_root=Path(path_values["input_root"]),
            work_root=Path(path_values["work_root"]),
        ),
        preprocessing=preprocessing,
    )
