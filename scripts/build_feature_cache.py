from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from first_place_preprocess import DEFAULT_DATA_ROOT, load_first_place_tensor, resolve_parquet_path


DEFAULT_CACHE_DIR = Path(r"C:\ASL\islr_feature_cache_fp16")
DEFAULT_CSV = Path("outputs") / "first_place_train_fold0.csv"
DEFAULT_METADATA = Path("outputs") / "cache_metadata.csv"


def load_json_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_label_map(data_root: Path) -> dict[str, int]:
    label_map_path = data_root / "sign_to_prediction_index_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError(f"Label map not found: {label_map_path}")
    with label_map_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(key): int(value) for key, value in raw.items()}


def cache_file_name(participant_id: int, sequence_id: int, max_len: int) -> str:
    return f"{int(participant_id)}_{int(sequence_id)}_len{int(max_len)}.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build .npy feature cache from first-place preprocessing.")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config file.")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="0 means no limit.")
    parser.add_argument("--max-len", type=int, default=None, help="Cached sequence length.")
    parser.add_argument("--dtype", choices=["float16", "float32"], default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def merged_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_json_config(args.config)
    return {
        "data_root": Path(args.data_root or config.get("data_root", DEFAULT_DATA_ROOT)),
        "csv": Path(args.csv or config.get("csv", DEFAULT_CSV)),
        "cache_dir": Path(args.cache_dir or config.get("cache_dir", DEFAULT_CACHE_DIR)),
        "max_samples": int(args.max_samples if args.max_samples is not None else config.get("max_samples", 0)),
        "max_len": int(args.max_len if args.max_len is not None else config.get("max_len", 64)),
        "dtype": str(args.dtype or config.get("dtype", "float16")),
        "overwrite": bool(args.overwrite or config.get("overwrite", False)),
    }


def main() -> None:
    args = parse_args()
    config = merged_config(args)
    data_root: Path = config["data_root"]
    csv_path: Path = config["csv"]
    cache_dir: Path = config["cache_dir"]
    max_samples: int = config["max_samples"]
    max_len: int = config["max_len"]
    dtype_name: str = config["dtype"]
    overwrite: bool = config["overwrite"]

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if max_samples < 0:
        raise ValueError(f"--max-samples must be >= 0, got {max_samples}")
    if max_len <= 0:
        raise ValueError(f"--max-len must be positive, got {max_len}")

    output_dtype = np.float16 if dtype_name == "float16" else np.float32
    label_map = load_label_map(data_root)
    df = pd.read_csv(csv_path)
    if max_samples > 0:
        df = df.head(max_samples).copy()

    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows: list[dict[str, Any]] = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="building feature cache"):
        sign = str(row.sign)
        if sign not in label_map:
            raise KeyError(f"Sign {sign!r} is missing from label map")

        cache_path = cache_dir / cache_file_name(int(row.participant_id), int(row.sequence_id), max_len)
        if cache_path.exists() and not overwrite:
            cache_exists = True
            try:
                cached = np.load(cache_path, mmap_mode="r")
                shape = tuple(cached.shape)
                saved_dtype = str(cached.dtype)
            except Exception:
                shape = ""
                saved_dtype = ""
        else:
            parquet_path = resolve_parquet_path(data_root, row.path)
            features, _, _ = load_first_place_tensor(parquet_path, max_len=max_len)
            if tuple(features.shape) != (max_len, 708):
                raise ValueError(
                    f"Expected generated feature shape {(max_len, 708)}, got {tuple(features.shape)} for {parquet_path}"
                )
            features = features.astype(output_dtype, copy=False)
            np.save(cache_path, features)
            cache_exists = cache_path.exists()
            shape = tuple(features.shape)
            saved_dtype = str(features.dtype)

        metadata_rows.append(
            {
                "path": row.path,
                "participant_id": int(row.participant_id),
                "sequence_id": int(row.sequence_id),
                "sign": sign,
                "label": label_map[sign],
                "cache_path": str(cache_path),
                "max_len": max_len,
                "shape": str(shape),
                "dtype": saved_dtype,
                "cache_exists": bool(cache_exists),
            }
        )

    DEFAULT_METADATA.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metadata_rows).to_csv(DEFAULT_METADATA, index=False)
    print(f"processed rows: {len(metadata_rows)}")
    print(f"cache_dir: {cache_dir}")
    print(f"max_len: {max_len}")
    print(f"metadata: {DEFAULT_METADATA.resolve()}")


if __name__ == "__main__":
    main()
