from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


DEFAULT_DATA_ROOT = Path(r"C:\ASL\asl-signs")
REQUIRED_PARQUET_COLUMNS = {"frame", "type", "landmark_index", "x", "y", "z"}
REQUIRED_CSV_COLUMNS = {"path", "participant_id", "sequence_id", "sign"}
COORD_COLUMNS = ("x", "y", "z")
DEFAULT_LANDMARK_COUNTS = {
    "face": 468,
    "left_hand": 21,
    "pose": 33,
    "right_hand": 21,
}


def load_label_map(label_map_path: Path) -> dict[str, int]:
    with label_map_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(key): int(value) for key, value in raw.items()}


def validate_csv_columns(df: pd.DataFrame, csv_path: Path) -> None:
    missing = REQUIRED_CSV_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: {sorted(missing)}. "
            f"Actual columns: {list(df.columns)}"
        )


def validate_parquet_columns(df: pd.DataFrame, parquet_path: Path) -> None:
    missing = REQUIRED_PARQUET_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            f"{parquet_path} does not match expected Kaggle ISLR long-table format. "
            f"Missing columns: {sorted(missing)}. Actual columns: {list(df.columns)}"
        )


def landmark_keys_for_types(
    df: pd.DataFrame,
    include_types: Iterable[str],
) -> list[tuple[str, int]]:
    keys: list[tuple[str, int]] = []
    for landmark_type in include_types:
        if landmark_type in DEFAULT_LANDMARK_COUNTS:
            indices = range(DEFAULT_LANDMARK_COUNTS[landmark_type])
        else:
            indices = (
                df.loc[df["type"] == landmark_type, "landmark_index"]
                .dropna()
                .astype(int)
                .sort_values()
                .unique()
                .tolist()
            )
        keys.extend((landmark_type, int(index)) for index in indices)
    return keys


def resolve_parquet_path(data_root: Path, sample_path: str | Path) -> Path:
    path = Path(str(sample_path))
    return path if path.is_absolute() else data_root / path


def load_landmark_tensor(
    parquet_path: str | Path,
    max_frames: int = 64,
    include_types: tuple[str, ...] = ("left_hand", "right_hand", "pose"),
    fillna_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one Kaggle ISLR parquet sample into a tensor and frame mask.

    Feature order is include_types order, then ascending landmark_index, then x/y/z.
    Padding frames are zero-filled and marked False in the returned mask.
    """
    if max_frames <= 0:
        raise ValueError(f"max_frames must be positive, got {max_frames}")

    parquet_path = Path(parquet_path)
    df = pd.read_parquet(parquet_path)
    validate_parquet_columns(df, parquet_path)

    filtered = df[df["type"].isin(include_types)].copy()
    if filtered.empty:
        raise ValueError(
            f"No rows found for include_types={include_types}. "
            f"Available types: {sorted(df['type'].dropna().astype(str).unique().tolist())}"
        )

    landmark_keys = landmark_keys_for_types(filtered, include_types)
    if not landmark_keys:
        raise ValueError(f"Could not infer landmark indices for include_types={include_types}")

    frame_values = sorted(filtered["frame"].dropna().unique().tolist())[:max_frames]
    feature_dim = len(landmark_keys) * len(COORD_COLUMNS)
    tensor = np.zeros((max_frames, feature_dim), dtype=np.float32)
    mask = np.zeros(max_frames, dtype=bool)
    key_to_offset = {key: idx * len(COORD_COLUMNS) for idx, key in enumerate(landmark_keys)}

    for out_frame_idx, frame in enumerate(frame_values):
        frame_df = filtered[filtered["frame"] == frame]
        frame_features = np.full(feature_dim, fillna_value, dtype=np.float32)
        for row in frame_df.itertuples(index=False):
            key = (str(row.type), int(row.landmark_index))
            offset = key_to_offset.get(key)
            if offset is None:
                continue
            for coord_idx, value in enumerate((row.x, row.y, row.z)):
                frame_features[offset + coord_idx] = (
                    np.float32(fillna_value) if pd.isna(value) else np.float32(value)
                )
        tensor[out_frame_idx] = frame_features
        mask[out_frame_idx] = True

    return tensor, mask


def inspect_tensor(tensor: np.ndarray | torch.Tensor) -> str:
    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu().numpy()
        dtype = str(tensor.dtype)
    else:
        array = tensor
        dtype = str(tensor.dtype)

    nan_count = int(np.isnan(array).sum())
    nonzero_ratio = float(np.count_nonzero(array) / array.size) if array.size else 0.0
    mean = float(np.nanmean(array)) if array.size else 0.0
    std = float(np.nanstd(array)) if array.size else 0.0
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


class ISLRDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        csv_path: str | Path,
        label_map_path: str | Path | None = None,
        max_frames: int = 64,
        include_types: tuple[str, ...] = ("left_hand", "right_hand", "pose"),
        fillna_value: float = 0.0,
    ) -> None:
        self.data_root = Path(data_root)
        self.csv_path = Path(csv_path)
        self.label_map_path = (
            Path(label_map_path)
            if label_map_path is not None
            else self.data_root / "sign_to_prediction_index_map.json"
        )
        self.max_frames = max_frames
        self.include_types = include_types
        self.fillna_value = fillna_value

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
            raise KeyError(f"Sign '{sign}' from {self.csv_path} is missing from {self.label_map_path}")

        parquet_path = resolve_parquet_path(self.data_root, row["path"])
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        x_np, mask_np = load_landmark_tensor(
            parquet_path,
            max_frames=self.max_frames,
            include_types=self.include_types,
            fillna_value=self.fillna_value,
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
