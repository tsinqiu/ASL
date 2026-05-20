from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_DATA_ROOT = Path(r"C:\ASL\asl-signs")
REQUIRED_COLUMNS = {"frame", "type", "landmark_index", "x", "y", "z"}
COORD_COLUMNS = ("x", "y", "z")


def _resolve_sample_from_index(data_root: Path, index: int) -> Path:
    train_csv = data_root / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv not found: {train_csv}")

    train_df = pd.read_csv(train_csv)
    if "path" not in train_df.columns:
        raise ValueError(f"train.csv has no 'path' column. Actual columns: {list(train_df.columns)}")
    if index < 0 or index >= len(train_df):
        raise IndexError(f"--index {index} is out of range for train.csv with {len(train_df)} rows")

    sample_path = Path(str(train_df.iloc[index]["path"]))
    return sample_path if sample_path.is_absolute() else data_root / sample_path


def _validate_landmark_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            "Parquet file does not match expected Kaggle ISLR long-table format. "
            f"Missing columns: {sorted(missing)}. Actual columns: {list(df.columns)}"
        )


def _ordered_landmark_keys(df: pd.DataFrame, include_types: Iterable[str]) -> list[tuple[str, int]]:
    keys: list[tuple[str, int]] = []
    for landmark_type in include_types:
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


def load_landmark_tensor(
    parquet_path: str | Path,
    max_frames: int = 64,
    include_types: tuple[str, ...] = ("left_hand", "right_hand", "pose"),
    fillna_value: float = 0.0,
) -> np.ndarray:
    """Load one Kaggle ISLR parquet sample as a fixed-length landmark tensor.

    The returned tensor is ordered by ascending frame, then by include_types order,
    then by ascending landmark_index, with x/y/z flattened for each landmark.
    """
    if max_frames <= 0:
        raise ValueError(f"max_frames must be positive, got {max_frames}")

    parquet_path = Path(parquet_path)
    df = pd.read_parquet(parquet_path)
    _validate_landmark_columns(df)

    filtered = df[df["type"].isin(include_types)].copy()
    if filtered.empty:
        raise ValueError(
            f"No landmarks found for include_types={include_types}. "
            f"Available types: {sorted(df['type'].dropna().unique().tolist())}"
        )

    landmark_keys = _ordered_landmark_keys(filtered, include_types)
    if not landmark_keys:
        raise ValueError(f"Could not infer landmark_index values for include_types={include_types}")

    frame_values = sorted(filtered["frame"].dropna().unique().tolist())[:max_frames]
    feature_dim = len(landmark_keys) * len(COORD_COLUMNS)
    tensor = np.zeros((max_frames, feature_dim), dtype=np.float32)

    key_to_offset = {key: idx * len(COORD_COLUMNS) for idx, key in enumerate(landmark_keys)}

    for out_frame_idx, frame in enumerate(frame_values):
        frame_df = filtered[filtered["frame"] == frame]
        frame_values_flat = np.full(feature_dim, fillna_value, dtype=np.float32)

        for row in frame_df.itertuples(index=False):
            key = (str(row.type), int(row.landmark_index))
            offset = key_to_offset.get(key)
            if offset is None:
                continue

            coords = (row.x, row.y, row.z)
            for coord_idx, value in enumerate(coords):
                frame_values_flat[offset + coord_idx] = (
                    fillna_value if pd.isna(value) else np.float32(value)
                )

        tensor[out_frame_idx] = frame_values_flat

    return tensor


def inspect_tensor(tensor: np.ndarray) -> str:
    """Return a compact sanity-check report for a landmark tensor."""
    nan_count = int(np.isnan(tensor).sum())
    nonzero_ratio = float(np.count_nonzero(tensor) / tensor.size) if tensor.size else 0.0
    mean = float(np.nanmean(tensor)) if tensor.size else 0.0
    std = float(np.nanstd(tensor)) if tensor.size else 0.0
    lines = [
        f"tensor shape: {tensor.shape}",
        f"tensor dtype: {tensor.dtype}",
        f"nan count: {nan_count}",
        f"nonzero ratio: {nonzero_ratio:.6f}",
        f"mean: {mean:.6f}",
        f"std: {std:.6f}",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert one Kaggle ISLR parquet sample to a tensor.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--index", type=int, default=0, help="Row index in train.csv to load.")
    parser.add_argument("--path", type=Path, default=None, help="Direct parquet path. Overrides --index.")
    parser.add_argument("--max-frames", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    parquet_path = args.path if args.path is not None else _resolve_sample_from_index(data_root, args.index)
    if not parquet_path.is_absolute():
        parquet_path = data_root / parquet_path

    print(f"DATA_ROOT: {data_root}")
    print(f"parquet_path: {parquet_path}")
    tensor = load_landmark_tensor(parquet_path, max_frames=args.max_frames)
    print(inspect_tensor(tensor))


if __name__ == "__main__":
    main()
