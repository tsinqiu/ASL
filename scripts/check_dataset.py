from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DATA_ROOT = Path(r"C:\ASL\asl-signs")
OUTPUT_PATH = Path("outputs") / "dataset_summary.txt"


def add(lines: list[str], text: str = "") -> None:
    lines.append(text)


def format_head(df: pd.DataFrame, rows: int = 5) -> str:
    return df.head(rows).to_string(index=False)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_report(data_root: Path) -> str:
    lines: list[str] = []
    train_csv = data_root / "train.csv"
    label_map_path = data_root / "sign_to_prediction_index_map.json"
    landmark_dir = data_root / "train_landmark_files"

    add(lines, "Kaggle ISLR Dataset Summary")
    add(lines, "=" * 32)
    add(lines, f"DATA_ROOT: {data_root}")
    add(lines, f"1. DATA_ROOT exists: {data_root.exists()}")
    add(lines, f"2. train.csv exists: {train_csv.exists()}")
    add(lines, f"3. sign_to_prediction_index_map.json exists: {label_map_path.exists()}")
    add(lines, f"4. train_landmark_files exists: {landmark_dir.exists()}")
    add(lines)

    train_df: pd.DataFrame | None = None
    if train_csv.exists():
        try:
            train_df = pd.read_csv(train_csv)
            add(lines, f"5. train.csv shape: {train_df.shape}")
            add(lines, f"6. train.csv columns: {list(train_df.columns)}")
            add(lines, "7. train.csv head:")
            add(lines, format_head(train_df))
            add(lines)

            if "sign" in train_df.columns:
                add(lines, f"8. sign class count in train.csv: {train_df['sign'].nunique()}")
            else:
                add(lines, "8. sign class count in train.csv: unavailable; missing 'sign' column")

            if "participant_id" in train_df.columns:
                add(lines, f"9. participant_id count: {train_df['participant_id'].nunique()}")
            else:
                add(lines, "9. participant_id count: unavailable; missing 'participant_id' column")

            if "sequence_id" in train_df.columns:
                add(lines, f"10. sequence_id count: {train_df['sequence_id'].nunique()}")
            else:
                add(lines, "10. sequence_id count: unavailable; missing 'sequence_id' column")
        except Exception as exc:
            add(lines, f"ERROR reading train.csv: {exc}")
    else:
        add(lines, "ERROR: train.csv is missing; cannot inspect table shape, labels, or sample paths.")

    add(lines)
    label_map: dict[str, Any] | None = None
    if label_map_path.exists():
        try:
            label_map = load_json(label_map_path)
            add(lines, f"11. JSON label class count: {len(label_map)}")
        except Exception as exc:
            add(lines, f"ERROR reading sign_to_prediction_index_map.json: {exc}")
    else:
        add(lines, "11. JSON label class count: unavailable; mapping file is missing")

    if train_df is not None and label_map is not None and "sign" in train_df.columns:
        train_signs = set(train_df["sign"].dropna().astype(str).unique())
        json_signs = set(str(key) for key in label_map.keys())
        add(lines, f"12. train.csv sign labels equal JSON keys: {train_signs == json_signs}")
        if train_signs != json_signs:
            add(lines, f"    labels in train.csv but not JSON: {sorted(train_signs - json_signs)[:20]}")
            add(lines, f"    labels in JSON but not train.csv: {sorted(json_signs - train_signs)[:20]}")
    else:
        add(lines, "12. train.csv sign labels equal JSON keys: unavailable")

    add(lines)
    add(lines, "13. First 5 parquet path checks:")
    if train_df is not None and "path" in train_df.columns:
        for idx, rel_path in enumerate(train_df["path"].head(5).astype(str).tolist()):
            sample_path = Path(rel_path)
            full_path = sample_path if sample_path.is_absolute() else data_root / sample_path
            add(lines, f"    [{idx}] {rel_path} -> exists: {full_path.exists()} ({full_path})")
    elif train_df is not None:
        add(lines, f"    unavailable; missing 'path' column. Actual columns: {list(train_df.columns)}")
    else:
        add(lines, "    unavailable; train.csv was not loaded")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Kaggle ISLR dataset structure.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.data_root)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nSaved report to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
