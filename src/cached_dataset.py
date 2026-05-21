from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from first_place_preprocess import DEFAULT_DATA_ROOT


DEFAULT_MAX_LEN = 64
DEFAULT_FEATURE_DIM = 708


def cache_file_name(participant_id: int, sequence_id: int, max_len: int | None = None) -> str:
    if max_len is None:
        return f"{int(participant_id)}_{int(sequence_id)}.npy"
    return f"{int(participant_id)}_{int(sequence_id)}_len{int(max_len)}.npy"


def legacy_cache_file_name(participant_id: int, sequence_id: int) -> str:
    return f"{int(participant_id)}_{int(sequence_id)}.npy"


def load_label_map(data_root: Path) -> dict[str, int]:
    label_map_path = data_root / "sign_to_prediction_index_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError(f"Label map not found: {label_map_path}")
    with label_map_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(key): int(value) for key, value in raw.items()}


class CachedISLRDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        cache_dir: str | Path,
        data_root: str | Path = DEFAULT_DATA_ROOT,
        label_map_path: str | Path | None = None,
        filter_missing_cache: bool = False,
        max_len: int = DEFAULT_MAX_LEN,
        feature_dim: int = DEFAULT_FEATURE_DIM,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.cache_dir = Path(cache_dir)
        self.data_root = Path(data_root)
        self.label_map_path = Path(label_map_path) if label_map_path is not None else self.data_root / "sign_to_prediction_index_map.json"
        self.filter_missing_cache = filter_missing_cache
        self.max_len = int(max_len)
        self.feature_dim = int(feature_dim)
        self.expected_shape = (self.max_len, self.feature_dim)

        if self.max_len <= 0:
            raise ValueError(f"max_len must be positive, got {self.max_len}")
        if self.feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {self.feature_dim}")

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        if not self.cache_dir.exists():
            raise FileNotFoundError(
                f"Cache directory not found: {self.cache_dir}. "
                "Run scripts/build_feature_cache.py manually before using dataset_mode='cache'."
            )

        self.df = pd.read_csv(self.csv_path)
        self.original_len = len(self.df)
        if self.filter_missing_cache:
            cache_exists = self.df.apply(lambda row: self._cache_path_for_row(row).exists(), axis=1)
            self.df = self.df[cache_exists].reset_index(drop=True)
            if self.df.empty:
                raise FileNotFoundError(
                    f"No cached samples from {self.csv_path} were found in {self.cache_dir}. "
                    "Build cache for this CSV first, or disable filter_missing_cache to fail on the first missing file."
                )
        self.filtered_missing_count = self.original_len - len(self.df)
        self.label_map = load_label_map(self.data_root)

    def __len__(self) -> int:
        return len(self.df)

    def _cache_path_for_row(self, row: pd.Series) -> Path:
        participant_id = int(row["participant_id"])
        sequence_id = int(row["sequence_id"])
        len_specific_path = self.cache_dir / cache_file_name(participant_id, sequence_id, self.max_len)
        if len_specific_path.exists():
            return len_specific_path
        return self.cache_dir / legacy_cache_file_name(participant_id, sequence_id)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        cache_path = self._cache_path_for_row(row)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Feature cache file not found: {cache_path}. "
                "Build it manually with scripts/build_feature_cache.py before cached training."
            )

        x_np = np.load(cache_path)
        if tuple(x_np.shape) != self.expected_shape:
            raise ValueError(
                f"Expected cached feature shape {self.expected_shape} "
                f"(max_len={self.max_len}, feature_dim={self.feature_dim}), got {tuple(x_np.shape)} at {cache_path}"
            )
        x_np = x_np.astype(np.float32, copy=False)
        mask_np = np.any(x_np != 0.0, axis=1)

        sign = str(row["sign"])
        if "label" in row and not pd.isna(row["label"]):
            label = int(row["label"])
        else:
            if sign not in self.label_map:
                raise KeyError(f"Sign {sign!r} is missing from {self.label_map_path}")
            label = self.label_map[sign]

        return {
            "x": torch.from_numpy(x_np).float(),
            "mask": torch.from_numpy(mask_np).bool(),
            "y": torch.tensor(label, dtype=torch.long),
            "sign": sign,
            "path": str(row["path"]),
            "participant_id": int(row["participant_id"]),
            "sequence_id": int(row["sequence_id"]),
        }
