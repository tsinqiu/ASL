from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


DEFAULT_DATA_ROOT = Path(r"C:\ASL\asl-signs")
ROWS_PER_FRAME = 543
COORD_COLUMNS = ("x", "y", "z")
REQUIRED_PARQUET_COLUMNS = {"frame", "type", "landmark_index", "x", "y", "z"}
REQUIRED_CSV_COLUMNS = {"path", "participant_id", "sequence_id", "sign"}

TYPE_OFFSETS = {
    "face": 0,
    "left_hand": 468,
    "pose": 489,
    "right_hand": 522,
}
TYPE_COUNTS = {
    "face": 468,
    "left_hand": 21,
    "pose": 33,
    "right_hand": 21,
}

NOSE = [1, 2, 98, 327]
NOTEBOOK_CENTER = [17]
CENTER_MODES = {
    "notebook_strict": NOTEBOOK_CENTER,
    "nose_mean": NOSE,
}
LIP = [
    0,
    61,
    185,
    40,
    39,
    37,
    267,
    269,
    270,
    409,
    291,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    78,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
    95,
    88,
    178,
    87,
    14,
    317,
    402,
    318,
    324,
    308,
]
REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
LHAND = list(range(468, 489))
RHAND = list(range(522, 543))
POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE
NUM_NODES = len(POINT_LANDMARKS)
FEATURE_DIM = 6 * NUM_NODES


def resolve_center_landmarks(
    center_mode: str = "notebook_strict",
    center_landmarks: list[int] | tuple[int, ...] | None = None,
) -> list[int]:
    if center_landmarks is not None:
        return [int(index) for index in center_landmarks]
    if center_mode not in CENTER_MODES:
        raise ValueError(f"Unknown center_mode={center_mode!r}. Available: {sorted(CENTER_MODES)}")
    return CENTER_MODES[center_mode]


def validate_parquet_columns(df: pd.DataFrame, source: str | Path = "parquet") -> None:
    missing = REQUIRED_PARQUET_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            f"{source} does not match expected Kaggle ISLR long-table format. "
            f"Missing columns: {sorted(missing)}. Actual columns: {list(df.columns)}"
        )


def validate_csv_columns(df: pd.DataFrame, csv_path: Path) -> None:
    missing = REQUIRED_CSV_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: {sorted(missing)}. "
            f"Actual columns: {list(df.columns)}"
        )


def load_label_map(label_map_path: Path) -> dict[str, int]:
    with label_map_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(key): int(value) for key, value in raw.items()}


def resolve_parquet_path(data_root: Path, sample_path: str | Path) -> Path:
    path = Path(str(sample_path))
    return path if path.is_absolute() else data_root / path


def restore_full_landmark_tensor(parquet_or_df: str | Path | pd.DataFrame) -> np.ndarray:
    """Restore Kaggle's long-table parquet rows to [T, 543, 3]."""
    if isinstance(parquet_or_df, pd.DataFrame):
        df = parquet_or_df
        source: str | Path = "dataframe"
    else:
        source = Path(parquet_or_df)
        df = pd.read_parquet(source)

    validate_parquet_columns(df, source)
    frames = sorted(df["frame"].dropna().unique().tolist())
    full = np.full((len(frames), ROWS_PER_FRAME, 3), np.nan, dtype=np.float32)
    frame_to_pos = {frame: pos for pos, frame in enumerate(frames)}

    for row in df.itertuples(index=False):
        landmark_type = str(row.type)
        if landmark_type not in TYPE_OFFSETS:
            continue
        local_idx = int(row.landmark_index)
        if local_idx < 0 or local_idx >= TYPE_COUNTS[landmark_type]:
            continue
        frame_pos = frame_to_pos[row.frame]
        global_idx = TYPE_OFFSETS[landmark_type] + local_idx
        full[frame_pos, global_idx, 0] = np.float32(row.x) if not pd.isna(row.x) else np.nan
        full[frame_pos, global_idx, 1] = np.float32(row.y) if not pd.isna(row.y) else np.nan
        full[frame_pos, global_idx, 2] = np.float32(row.z) if not pd.isna(row.z) else np.nan

    return full


def nan_mean(array: np.ndarray, axis: Any = None, keepdims: bool = False) -> np.ndarray:
    valid = ~np.isnan(array)
    count = valid.sum(axis=axis, keepdims=keepdims)
    total = np.where(valid, array, 0.0).sum(axis=axis, keepdims=keepdims)
    with np.errstate(divide="ignore", invalid="ignore"):
        return total / count


def nan_std(array: np.ndarray, center: np.ndarray, axis: Any = None, keepdims: bool = False) -> np.ndarray:
    diff = array - center
    return np.sqrt(nan_mean(diff * diff, axis=axis, keepdims=keepdims))


def filter_empty_frames(full: np.ndarray, point_landmarks: Iterable[int] = POINT_LANDMARKS) -> np.ndarray:
    selected = full[:, list(point_landmarks), :]
    keep = ~np.all(np.isnan(selected), axis=(1, 2))
    return full[keep]


def first_place_preprocess_array(
    full: np.ndarray,
    max_len: int = 64,
    point_landmarks: list[int] | tuple[int, ...] = POINT_LANDMARKS,
    center_mode: str = "notebook_strict",
    center_landmarks: list[int] | tuple[int, ...] | None = None,
    fillna_value: float = 0.0,
    filter_empty: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply first-place-style landmark selection, normalization, deltas, and padding.

    Output is [max_len, 708] for the default 118 selected landmarks:
    xy features + first-order deltas + second-order deltas.
    """
    if full.ndim != 3 or full.shape[1:] != (ROWS_PER_FRAME, 3):
        raise ValueError(f"Expected full tensor shape [T, 543, 3], got {full.shape}")
    if max_len <= 0:
        raise ValueError(f"max_len must be positive, got {max_len}")

    x_full = full.astype(np.float32, copy=False)
    if filter_empty:
        x_full = filter_empty_frames(x_full, point_landmarks)

    selected = x_full[:, list(point_landmarks), :]
    resolved_center_landmarks = resolve_center_landmarks(center_mode, center_landmarks)
    center = nan_mean(x_full[:, resolved_center_landmarks, :], axis=(0, 1), keepdims=True)
    center = np.where(np.isnan(center), np.array(0.5, dtype=np.float32), center).astype(np.float32)
    std = nan_std(selected, center=center, axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.where((np.isnan(std)) | (std == 0), np.array(1.0, dtype=np.float32), std)

    selected = (selected - center) / std
    selected = selected[:max_len, :, :2]
    length = selected.shape[0]

    dx = np.zeros_like(selected, dtype=np.float32)
    if length > 1:
        dx[:-1] = selected[1:] - selected[:-1]

    dx2 = np.zeros_like(selected, dtype=np.float32)
    if length > 2:
        dx2[:-2] = selected[2:] - selected[:-2]

    features = np.concatenate(
        [
            selected.reshape(length, 2 * len(point_landmarks)),
            dx.reshape(length, 2 * len(point_landmarks)),
            dx2.reshape(length, 2 * len(point_landmarks)),
        ],
        axis=-1,
    ).astype(np.float32)
    features = np.nan_to_num(features, nan=fillna_value, posinf=fillna_value, neginf=fillna_value)

    padded = np.zeros((max_len, features.shape[-1]), dtype=np.float32)
    mask = np.zeros(max_len, dtype=bool)
    padded[:length] = features
    mask[:length] = True
    return padded, mask


def load_first_place_tensor(
    parquet_path: str | Path,
    max_len: int = 64,
    fillna_value: float = 0.0,
    center_mode: str = "notebook_strict",
    center_landmarks: list[int] | tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    full = restore_full_landmark_tensor(parquet_path)
    features, mask = first_place_preprocess_array(
        full,
        max_len=max_len,
        fillna_value=fillna_value,
        center_mode=center_mode,
        center_landmarks=center_landmarks,
    )
    return features, mask, full


def inspect_feature_tensor(tensor: np.ndarray | torch.Tensor) -> str:
    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu().numpy()
        dtype = str(tensor.dtype)
    else:
        array = tensor
        dtype = str(tensor.dtype)

    nan_count = int(np.isnan(array).sum())
    nonzero_ratio = float(np.count_nonzero(array) / array.size) if array.size else 0.0
    mean = float(np.mean(array)) if array.size else 0.0
    std = float(np.std(array)) if array.size else 0.0
    return "\n".join(
        [
            f"shape: {tuple(array.shape)}",
            f"dtype: {dtype}",
            f"nan count: {nan_count}",
            f"nonzero ratio: {nonzero_ratio:.6f}",
            f"mean: {mean:.6f}",
            f"std: {std:.6f}",
        ]
    )


class FirstPlaceISLRDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        csv_path: str | Path,
        label_map_path: str | Path | None = None,
        max_len: int = 64,
        fillna_value: float = 0.0,
        center_mode: str = "notebook_strict",
        center_landmarks: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.csv_path = Path(csv_path)
        self.label_map_path = (
            Path(label_map_path)
            if label_map_path is not None
            else self.data_root / "sign_to_prediction_index_map.json"
        )
        self.max_len = max_len
        self.fillna_value = fillna_value
        self.center_mode = center_mode
        self.center_landmarks = center_landmarks
        self.resolved_center_landmarks = resolve_center_landmarks(center_mode, center_landmarks)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        if not self.label_map_path.exists():
            raise FileNotFoundError(f"Label map not found: {self.label_map_path}")

        self.df = pd.read_csv(self.csv_path)
        validate_csv_columns(self.df, self.csv_path)
        self.label_map = load_label_map(self.label_map_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        sign = str(row["sign"])
        if sign not in self.label_map:
            raise KeyError(f"Sign '{sign}' is missing from {self.label_map_path}")

        parquet_path = resolve_parquet_path(self.data_root, row["path"])
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        x_np, mask_np, _ = load_first_place_tensor(
            parquet_path,
            max_len=self.max_len,
            fillna_value=self.fillna_value,
            center_mode=self.center_mode,
            center_landmarks=self.center_landmarks,
        )
        return {
            "x": torch.from_numpy(x_np).float(),
            "mask": torch.from_numpy(mask_np).bool(),
            "y": torch.tensor(self.label_map[sign], dtype=torch.long),
            "sign": sign,
            "path": str(row["path"]),
            "participant_id": int(row["participant_id"]),
            "sequence_id": int(row["sequence_id"]),
        }
