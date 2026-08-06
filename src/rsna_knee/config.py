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
    )
    if preprocessing.slice_count <= 0:
        raise ValueError("slice_count must be positive")
    if not 0 <= preprocessing.lower_percentile < preprocessing.upper_percentile <= 100:
        raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")

    return PipelineConfig(
        paths=ProjectPaths(
            mode=mode,
            input_root=Path(path_values["input_root"]),
            work_root=Path(path_values["work_root"]),
        ),
        preprocessing=preprocessing,
    )
