from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cached_dataset import CachedISLRDataset
from first_place_preprocess import DEFAULT_DATA_ROOT


def summarize_sample(sample: dict[str, Any]) -> list[str]:
    x = sample["x"]
    return [
        f"sample x shape: {tuple(x.shape)}",
        f"sample x dtype: {x.dtype}",
        f"sample x NaN count: {int(torch.isnan(x).sum().item())}",
        f"sample mask shape: {tuple(sample['mask'].shape)}",
        f"sample y: {int(sample['y'].item())}",
        f"sample sign: {sample['sign']}",
        f"sample path: {sample['path']}",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check cached ISLR feature DataLoader.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--filter-missing-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = CachedISLRDataset(
        args.csv,
        args.cache_dir,
        data_root=args.data_root,
        filter_missing_cache=args.filter_missing_cache,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    start = time.perf_counter()
    sample = dataset[0]
    sample_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    batch = next(iter(loader))
    batch_elapsed = time.perf_counter() - start

    x = batch["x"]
    lines = [
        "Cached ISLR DataLoader Check",
        "=" * 29,
        f"csv: {args.csv}",
        f"cache_dir: {args.cache_dir}",
        f"filter_missing_cache: {args.filter_missing_cache}",
        f"original csv rows: {dataset.original_len}",
        f"filtered missing rows: {dataset.filtered_missing_count}",
        f"dataset size: {len(dataset)}",
        f"batch_size: {args.batch_size}",
        "",
        *summarize_sample(sample),
        "",
        f"batch x shape: {tuple(x.shape)}",
        f"batch x dtype: {x.dtype}",
        f"batch x NaN count: {int(torch.isnan(x).sum().item())}",
        f"batch mask shape: {tuple(batch['mask'].shape)}",
        f"batch y shape: {tuple(batch['y'].shape)}",
        f"sample read seconds: {sample_elapsed:.4f}",
        f"batch read seconds: {batch_elapsed:.4f}",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
