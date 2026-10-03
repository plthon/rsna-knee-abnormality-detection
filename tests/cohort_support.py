import csv
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

COMPETITION_CONTRACT = Path(__file__).parents[1] / "contracts" / "competition.json"
_CONTRACT = json.loads(COMPETITION_CONTRACT.read_text())
TARGETS = tuple(_CONTRACT["targets"]["order"])
TRAIN_COLUMNS = tuple(_CONTRACT["data"]["tables"]["train.csv"]["columns"])
STUDY_COLUMNS = (
    "study_index",
    "split",
    "StudyInstanceUID",
    "PatientID",
    "series_count",
    "usable_series_count",
    "slice_count",
    "usable_slice_count",
)
SERIES_COLUMNS = (
    "series_index",
    "study_index",
    "SeriesInstanceUID",
    "relative_path",
    "Fluid_Sensitive",
    "Fat_Suppression",
    "Anatomical_Plane",
    "slice_count",
    "usable_slice_count",
    "ordering_method",
    "selection_mask",
)
SLICE_COLUMNS = (
    "slice_index",
    "series_index",
    "SOPInstanceUID",
    "filename",
    "ordered_position",
    "rows",
    "columns",
    "instance_number",
    "geometry_coordinate",
    "transfer_syntax",
    "photometric_interpretation",
    "rescale_slope",
    "rescale_intercept",
)
SELECTED_COLUMNS = ("series_index", "slot", "slice_index", "ordered_position")


def train_row(study_uid: str, labels: Sequence[int | None]) -> dict[str, object]:
    row: dict[str, object] = {column: "" for column in TRAIN_COLUMNS}
    row["StudyInstanceUID"] = study_uid
    row["Report"] = "private report text must not be consumed"
    row.update(
        {
            target: "" if label is None else label
            for target, label in zip(TARGETS, labels, strict=True)
        }
    )
    return row


def study_row(
    index: int, study_uid: str, patient_id: str, split: str = "train"
) -> dict[str, object]:
    return {
        "study_index": index,
        "split": split,
        "StudyInstanceUID": study_uid,
        "PatientID": patient_id,
        "series_count": 1,
        "usable_series_count": 1,
        "slice_count": 1,
        "usable_slice_count": 1,
    }


def write_csv(
    path: Path, columns: Sequence[str], rows: Sequence[dict[str, object]]
) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
