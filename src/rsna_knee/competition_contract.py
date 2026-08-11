import hashlib
import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

_EXPECTED_SERIES_COLUMNS = (
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "Fluid_Sensitive",
    "Fat_Suppression",
    "Anatomical_Plane",
)
_EXPECTED_IMAGE_PATH_TEMPLATE = (
    "{train_series|test_series}/{StudyInstanceUID}/"
    "{SeriesInstanceUID}/{SOPInstanceUID}.dcm"
)


@dataclass(frozen=True, slots=True)
class SeriesTableContract:
    split: str
    image_root: str
    filename: str
    columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VerifiedIndexContract:
    schema_version: int
    verified_at: str
    sha256: str
    series_tables: tuple[SeriesTableContract, ...]


def load_verified_index_contract(path: Path | None = None) -> VerifiedIndexContract:
    """Load and validate the canonical fields consumed by DICOM indexing."""
    raw = _contract_bytes(path)
    contract = cast(dict[str, Any], json.loads(raw))
    schema_version = _required_int(contract, "schema_version")
    competition = _required_dict(contract, "competition")
    verified_at = _required_str(competition, "contract_verified_at")
    data = _required_dict(contract, "data")
    if data.get("image_path_template") != _EXPECTED_IMAGE_PATH_TEMPLATE:
        raise ValueError("Competition image path template is not the verified layout")
    image_roots = _required_dict(data, "image_roots")
    tables = _required_dict(data, "tables")

    series_tables: list[SeriesTableContract] = []
    for split in ("train", "test"):
        image_root = f"{split}_series"
        filename = f"{split}_series.csv"
        if image_root not in image_roots:
            raise ValueError(f"Competition contract has no {image_root} image root")
        table = _required_dict(tables, filename)
        raw_columns = table.get("columns")
        if not isinstance(raw_columns, list) or not all(
            isinstance(column, str) for column in raw_columns
        ):
            raise TypeError(f"Competition contract has invalid columns for {filename}")
        columns = tuple(raw_columns)
        if columns != _EXPECTED_SERIES_COLUMNS:
            raise ValueError(
                f"Competition contract has unexpected columns for {filename}"
            )
        series_tables.append(
            SeriesTableContract(
                split=split,
                image_root=image_root,
                filename=filename,
                columns=columns,
            )
        )
    return VerifiedIndexContract(
        schema_version=schema_version,
        verified_at=verified_at,
        sha256=hashlib.sha256(raw).hexdigest(),
        series_tables=tuple(series_tables),
    )


def _contract_bytes(path: Path | None) -> bytes:
    if path is not None:
        return path.read_bytes()
    source_contract = (
        Path(__file__).resolve().parents[2] / "contracts" / "competition.json"
    )
    if source_contract.is_file():
        return source_contract.read_bytes()
    resource = files("rsna_knee").joinpath("competition.json")
    try:
        return resource.read_bytes()
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "The canonical competition contract is not available in the source tree "
            "or installed package"
        ) from error


def _required_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Competition contract field {key} must be an object")
    return cast(dict[str, Any], value)


def _required_str(parent: dict[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"Competition contract field {key} must be a nonempty string")
    return value


def _required_int(parent: dict[str, Any], key: str) -> int:
    value = parent.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"Competition contract field {key} must be an integer")
    return value
