from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from first_place_preprocess import (
    FEATURE_DIM,
    FirstPlaceISLRDataset,
    DEFAULT_DATA_ROOT,
    CENTER_MODES,
    inspect_feature_tensor,
    load_first_place_tensor,
    resolve_center_landmarks,
    resolve_parquet_path,
)


FOLDS_CSV = Path("outputs") / "train_with_folds.csv"
OUTPUT_PATH = Path("outputs") / "first_place_preprocess_check.txt"


def add(lines: list[str], text: str = "") -> None:
    lines.append(text)


def summarize_sample(sample: dict[str, Any]) -> list[str]:
    x = sample["x"]
    mask = sample["mask"]
    return [
        f"x shape: {tuple(x.shape)}",
        f"mask shape: {tuple(mask.shape)}",
        f"y: {int(sample['y'].item())}",
        f"sign: {sample['sign']}",
        f"path: {sample['path']}",
        f"participant_id: {sample['participant_id']}",
        f"sequence_id: {sample['sequence_id']}",
        f"mask true count: {int(mask.sum().item())}",
        f"x NaN count: {int(torch.isnan(x).sum().item())}",
    ]


def summarize_batch(name: str, batch: dict[str, Any]) -> list[str]:
    x = batch["x"]
    mask = batch["mask"]
    y = batch["y"]
    signs = list(batch["sign"][: min(5, len(batch["sign"]))])
    return [
        f"{name} batch x shape: {tuple(x.shape)}",
        f"{name} batch mask shape: {tuple(mask.shape)}",
        f"{name} batch y shape: {tuple(y.shape)}",
        f"{name} batch sign examples: {signs}",
        f"{name} batch x NaN count: {int(torch.isnan(x).sum().item())}",
        f"{name} batch x nonzero ratio: {(torch.count_nonzero(x).item() / x.numel()):.6f}",
        f"{name} batch x mean: {x.mean().item():.6f}",
        f"{name} batch x std: {x.std(unbiased=False).item():.6f}",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check first-place-style preprocessing shapes.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--center-mode",
        choices=["both", *CENTER_MODES.keys()],
        default="both",
        help="Centering mode to check. Default checks notebook_strict and nose_mean.",
    )
    return parser.parse_args()


def build_report_for_mode(
    args: argparse.Namespace,
    fold_df: pd.DataFrame,
    center_mode: str,
    train_csv: Path,
    valid_csv: Path,
) -> list[str]:
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    row = fold_df.iloc[args.index]
    parquet_path = resolve_parquet_path(args.data_root, row["path"])
    features_np, mask_np, full_np = load_first_place_tensor(
        parquet_path,
        max_len=args.max_len,
        center_mode=center_mode,
    )

    train_dataset = FirstPlaceISLRDataset(
        args.data_root,
        train_csv,
        max_len=args.max_len,
        center_mode=center_mode,
    )
    valid_dataset = FirstPlaceISLRDataset(
        args.data_root,
        valid_csv,
        max_len=args.max_len,
        center_mode=center_mode,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    sample = train_dataset[0]
    train_batch = next(iter(train_loader))
    valid_batch = next(iter(valid_loader))

    lines: list[str] = []
    add(lines, f"Center mode: {center_mode}")
    add(lines, "-" * (13 + len(center_mode)))
    add(lines, f"center landmarks: {resolve_center_landmarks(center_mode)}")
    add(lines, f"expected feature_dim: {FEATURE_DIM}")
    add(lines, f"full tensor shape from sample parquet: {full_np.shape}")
    add(lines, f"single direct feature shape: {features_np.shape}")
    add(lines, f"single direct mask shape: {mask_np.shape}")
    add(lines, f"single direct mask true count: {int(mask_np.sum())}")
    add(lines, f"single direct feature NaN count: {int(pd.isna(features_np).sum())}")
    add(lines, "single direct feature stats:")
    add(lines, inspect_feature_tensor(features_np))
    add(lines)
    add(lines, "Train sample")
    add(lines, "-" * 12)
    lines.extend(summarize_sample(sample))
    add(lines)
    add(lines, "Train batch")
    add(lines, "-" * 11)
    lines.extend(summarize_batch("train", train_batch))
    add(lines)
    add(lines, "Valid batch")
    add(lines, "-" * 11)
    lines.extend(summarize_batch("valid", valid_batch))
    return lines


def main() -> None:
    args = parse_args()
    if not FOLDS_CSV.exists():
        raise FileNotFoundError(f"{FOLDS_CSV} does not exist. Run src/split.py first.")

    fold_df = pd.read_csv(FOLDS_CSV)
    if "fold" not in fold_df.columns:
        raise ValueError(f"{FOLDS_CSV} has no 'fold' column. Actual columns: {list(fold_df.columns)}")

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    train_csv = outputs_dir / f"first_place_train_fold{args.fold}.csv"
    valid_csv = outputs_dir / f"first_place_valid_fold{args.fold}.csv"
    train_df = fold_df[fold_df["fold"] != args.fold].reset_index(drop=True)
    valid_df = fold_df[fold_df["fold"] == args.fold].reset_index(drop=True)
    train_df.to_csv(train_csv, index=False)
    valid_df.to_csv(valid_csv, index=False)

    modes = list(CENTER_MODES.keys()) if args.center_mode == "both" else [args.center_mode]
    lines: list[str] = []
    add(lines, "First-Place-Style Preprocess Check")
    add(lines, "=" * 35)
    add(lines, f"DATA_ROOT: {args.data_root}")
    add(lines, f"fold: {args.fold}")
    add(lines, f"batch_size: {args.batch_size}")
    add(lines, f"max_len: {args.max_len}")
    add(lines, f"modes checked: {modes}")
    add(lines)

    for mode_idx, center_mode in enumerate(modes):
        if mode_idx:
            add(lines)
        lines.extend(build_report_for_mode(args, fold_df, center_mode, train_csv, valid_csv))

    report = "\n".join(lines)
    OUTPUT_PATH.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nsaved report to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
