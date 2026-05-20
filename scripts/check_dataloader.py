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

from dataset import ISLRDataset


DEFAULT_DATA_ROOT = Path(r"C:\ASL\asl-signs")
FOLDS_CSV = Path("outputs") / "train_with_folds.csv"
OUTPUT_PATH = Path("outputs") / "dataloader_check.txt"


def add(lines: list[str], text: str = "") -> None:
    lines.append(text)


def tensor_stats(prefix: str, x: torch.Tensor) -> list[str]:
    return [
        f"{prefix} NaN count: {int(torch.isnan(x).sum().item())}",
        f"{prefix} nonzero ratio: {(torch.count_nonzero(x).item() / x.numel()):.6f}",
        f"{prefix} mean: {x.mean().item():.6f}",
        f"{prefix} std: {x.std(unbiased=False).item():.6f}",
    ]


def summarize_sample(sample: dict[str, Any]) -> list[str]:
    x = sample["x"]
    mask = sample["mask"]
    lines = [
        f"x shape: {tuple(x.shape)}",
        f"mask shape: {tuple(mask.shape)}",
        f"y: {int(sample['y'].item())}",
        f"sign: {sample['sign']}",
        f"path: {sample['path']}",
        f"participant_id: {sample['participant_id']}",
        f"sequence_id: {sample['sequence_id']}",
        f"mask true count: {int(mask.sum().item())}",
        f"x has NaN: {bool(torch.isnan(x).any().item())}",
    ]
    return lines


def summarize_batch(name: str, batch: dict[str, Any]) -> list[str]:
    x = batch["x"]
    mask = batch["mask"]
    y = batch["y"]
    signs = batch["sign"]
    sign_preview = list(signs[: min(5, len(signs))])
    lines = [
        f"{name} batch x shape: {tuple(x.shape)}",
        f"{name} batch mask shape: {tuple(mask.shape)}",
        f"{name} batch y shape: {tuple(y.shape)}",
        f"{name} batch sign examples: {sign_preview}",
    ]
    lines.extend(tensor_stats(f"{name} batch x", x))
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check PyTorch DataLoader output for Kaggle ISLR.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not FOLDS_CSV.exists():
        raise FileNotFoundError(
            f"{FOLDS_CSV} does not exist. Run: python src/split.py --data-root {args.data_root} --n-splits 5"
        )

    fold_df = pd.read_csv(FOLDS_CSV)
    if "fold" not in fold_df.columns:
        raise ValueError(f"{FOLDS_CSV} has no 'fold' column. Actual columns: {list(fold_df.columns)}")
    if args.fold not in set(fold_df["fold"].astype(int).unique().tolist()):
        raise ValueError(f"--fold {args.fold} is not present in {FOLDS_CSV}")

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    train_csv = outputs_dir / f"train_fold{args.fold}.csv"
    valid_csv = outputs_dir / f"valid_fold{args.fold}.csv"

    train_df = fold_df[fold_df["fold"] != args.fold].reset_index(drop=True)
    valid_df = fold_df[fold_df["fold"] == args.fold].reset_index(drop=True)
    train_df.to_csv(train_csv, index=False)
    valid_df.to_csv(valid_csv, index=False)

    train_dataset = ISLRDataset(args.data_root, train_csv, max_frames=args.max_frames)
    valid_dataset = ISLRDataset(args.data_root, valid_csv, max_frames=args.max_frames)

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

    train_sample = train_dataset[0]
    train_batch = next(iter(train_loader))
    valid_batch = next(iter(valid_loader))

    lines: list[str] = []
    add(lines, "Kaggle ISLR DataLoader Check")
    add(lines, "=" * 30)
    add(lines, f"DATA_ROOT: {args.data_root}")
    add(lines, f"fold: {args.fold}")
    add(lines, f"batch_size: {args.batch_size}")
    add(lines, f"max_frames: {args.max_frames}")
    add(lines, f"num_workers: {args.num_workers}")
    add(lines, f"train samples: {len(train_dataset)}")
    add(lines, f"valid samples: {len(valid_dataset)}")
    add(lines, f"train csv: {train_csv.resolve()}")
    add(lines, f"valid csv: {valid_csv.resolve()}")
    add(lines)

    add(lines, "Train sample")
    add(lines, "-" * 12)
    lines.extend(summarize_sample(train_sample))
    add(lines)

    add(lines, "Train batch")
    add(lines, "-" * 11)
    lines.extend(summarize_batch("train", train_batch))
    add(lines)

    add(lines, "Valid batch")
    add(lines, "-" * 11)
    lines.extend(summarize_batch("valid", valid_batch))

    report = "\n".join(lines)
    OUTPUT_PATH.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nsaved report to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
