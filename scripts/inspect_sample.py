from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_DATA_ROOT = Path(r"C:\ASL\asl-signs")
OUTPUT_PATH = Path("outputs") / "sample_inspection.txt"


def add(lines: list[str], text: str = "") -> None:
    lines.append(text)


def resolve_parquet_path(data_root: Path, sample_path: Path) -> Path:
    return sample_path if sample_path.is_absolute() else data_root / sample_path


def find_row_for_path(train_df: pd.DataFrame, sample_path: Path) -> pd.Series | None:
    if "path" not in train_df.columns:
        return None
    normalized = sample_path.as_posix()
    matches = train_df[train_df["path"].astype(str).str.replace("\\", "/", regex=False) == normalized]
    if matches.empty:
        return None
    return matches.iloc[0]


def build_report(data_root: Path, index: int, direct_path: Path | None) -> str:
    lines: list[str] = []
    train_csv = data_root / "train.csv"
    train_df: pd.DataFrame | None = None
    selected_row: pd.Series | None = None

    add(lines, "Kaggle ISLR Sample Inspection")
    add(lines, "=" * 31)
    add(lines, f"DATA_ROOT: {data_root}")

    if train_csv.exists():
        train_df = pd.read_csv(train_csv)
    else:
        add(lines, f"WARNING: train.csv not found: {train_csv}")

    if direct_path is not None:
        sample_path = direct_path
        parquet_path = resolve_parquet_path(data_root, sample_path)
        if train_df is not None:
            selected_row = find_row_for_path(train_df, sample_path)
    else:
        if train_df is None:
            raise FileNotFoundError(f"train.csv is required when --path is not provided: {train_csv}")
        if "path" not in train_df.columns:
            raise ValueError(f"train.csv has no 'path' column. Actual columns: {list(train_df.columns)}")
        if index < 0 or index >= len(train_df):
            raise IndexError(f"--index {index} is out of range for train.csv with {len(train_df)} rows")
        selected_row = train_df.iloc[index]
        parquet_path = resolve_parquet_path(data_root, Path(str(selected_row["path"])))

    add(lines)
    add(lines, "1. Current train.csv row information:")
    if selected_row is not None:
        add(lines, selected_row.to_string())
    elif direct_path is not None:
        add(lines, "No matching train.csv row found for --path, or train.csv/path column is unavailable.")
    else:
        add(lines, "Unavailable.")

    add(lines)
    add(lines, f"2. parquet file path: {parquet_path}")
    add(lines, f"10. file exists: {parquet_path.exists()}")
    if parquet_path.exists():
        size_bytes = parquet_path.stat().st_size
        add(lines, f"10. file size: {size_bytes} bytes ({size_bytes / (1024 * 1024):.3f} MiB)")
    else:
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    add(lines, f"3. parquet shape: {df.shape}")
    add(lines, f"4. parquet columns: {list(df.columns)}")
    add(lines, "5. df.head():")
    add(lines, df.head().to_string(index=False))
    add(lines)

    if "type" in df.columns:
        add(lines, "6. type distribution:")
        add(lines, df["type"].value_counts(dropna=False).to_string())
    else:
        add(lines, "6. type distribution: unavailable; missing 'type' column")

    if "frame" in df.columns:
        add(lines, f"7. frame count: {df['frame'].nunique()}")
        add(lines, f"   frame min/max: {df['frame'].min()} / {df['frame'].max()}")
    else:
        add(lines, "7. frame count: unavailable; missing 'frame' column")

    coord_cols = [col for col in ("x", "y", "z") if col in df.columns]
    if coord_cols:
        add(lines, "8. x/y/z NaN ratio:")
        add(lines, df[coord_cols].isna().mean().to_string())
    else:
        add(lines, "8. x/y/z NaN ratio: unavailable; missing x/y/z columns")

    if {"type", "landmark_index"}.issubset(df.columns):
        add(lines, "9. landmark count by type:")
        add(lines, df.groupby("type")["landmark_index"].nunique().to_string())
    else:
        add(lines, "9. landmark count by type: unavailable; missing type or landmark_index column")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one Kaggle ISLR parquet sample.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--path", type=Path, default=None, help="Direct parquet path. Overrides --index.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.data_root, args.index, args.path)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nSaved report to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
