from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cached_dataset import CachedISLRDataset
from first_place_preprocess import CENTER_MODES, FirstPlaceISLRDataset
from model_tiny import TinyISLRModel
from train_smoke import count_parameters, resolve_device


DEFAULT_CONFIG = Path("configs") / "tiny_baseline.json"


def add(lines: list[str], text: str = "") -> None:
    lines.append(text)


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = dict(config)
    for key in ("data_root", "device"):
        value = getattr(args, key)
        if value is not None:
            updated[key] = str(value)
    for key in ("epochs", "batch_size", "num_workers", "max_train_batches", "max_valid_batches"):
        value = getattr(args, key)
        if value is not None:
            updated[key] = value
    if args.lr is not None:
        updated["lr"] = args.lr
    if args.center_mode is not None:
        updated["center_mode"] = args.center_mode
    return updated


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "data_root",
        "train_csv",
        "valid_csv",
        "center_mode",
        "seed",
        "device",
        "epochs",
        "batch_size",
        "max_frames",
        "num_workers",
        "lr",
        "weight_decay",
        "model",
        "log_path",
        "best_checkpoint_path",
        "last_checkpoint_path",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Config missing required keys: {sorted(missing)}")
    if config["center_mode"] not in CENTER_MODES:
        raise ValueError(f"Unknown center_mode={config['center_mode']!r}. Available: {sorted(CENTER_MODES)}")
    if int(config["epochs"]) <= 0:
        raise ValueError("epochs must be positive")
    if int(config["batch_size"]) <= 0:
        raise ValueError("batch_size must be positive")
    dataset_mode = str(config.get("dataset_mode", "online"))
    if dataset_mode not in {"online", "cache"}:
        raise ValueError("dataset_mode must be either 'online' or 'cache'")
    if dataset_mode == "cache" and not config.get("cache_dir"):
        raise ValueError("cache_dir is required when dataset_mode='cache'")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["x"].to(device, non_blocking=True),
        batch["mask"].to(device, non_blocking=True),
        batch["y"].to(device, non_blocking=True),
    )


def make_loader(
    dataset: FirstPlaceISLRDataset | CachedISLRDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def train_one_epoch(
    model: TinyISLRModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    grad_clip_norm: float | None,
    max_batches: int = 0,
) -> tuple[float, float, tuple[int, ...], tuple[int, ...]]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    first_batch_shape: tuple[int, ...] | None = None
    first_logits_shape: tuple[int, ...] | None = None

    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        x, mask, y = move_batch_to_device(batch, device)
        if first_batch_shape is None:
            first_batch_shape = tuple(x.shape)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x, mask)
            loss = criterion(logits, y)
        if first_logits_shape is None:
            first_logits_shape = tuple(logits.shape)

        scaler.scale(loss).backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = x.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=-1) == y).sum().item())
        total_examples += batch_size

    if total_examples == 0:
        raise RuntimeError("No training examples were processed.")
    assert first_batch_shape is not None
    assert first_logits_shape is not None
    return total_loss / total_examples, total_correct / total_examples, first_batch_shape, first_logits_shape


@torch.no_grad()
def validate_one_epoch(
    model: TinyISLRModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    max_batches: int = 0,
) -> tuple[float, float, tuple[int, ...], tuple[int, ...]]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    first_batch_shape: tuple[int, ...] | None = None
    first_logits_shape: tuple[int, ...] | None = None

    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        x, mask, y = move_batch_to_device(batch, device)
        if first_batch_shape is None:
            first_batch_shape = tuple(x.shape)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x, mask)
            loss = criterion(logits, y)
        if first_logits_shape is None:
            first_logits_shape = tuple(logits.shape)

        batch_size = x.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=-1) == y).sum().item())
        total_examples += batch_size

    if total_examples == 0:
        raise RuntimeError("No validation examples were processed.")
    assert first_batch_shape is not None
    assert first_logits_shape is not None
    return total_loss / total_examples, total_correct / total_examples, first_batch_shape, first_logits_shape


def save_checkpoint(
    path: Path,
    model: TinyISLRModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    epoch: int,
    metrics: dict[str, float],
    param_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": config,
            "metrics": metrics,
            "param_count": param_count,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tiny single-fold ISLR baseline from JSON config.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-valid-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--center-mode", choices=sorted(CENTER_MODES), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    validate_config(config)

    data_root = Path(config["data_root"])
    train_csv = Path(config["train_csv"])
    valid_csv = Path(config["valid_csv"])
    if not train_csv.exists() or not valid_csv.exists():
        raise FileNotFoundError(
            "First-place train/valid CSV files are missing. Run "
            "`python scripts/check_first_place_preprocess.py --data-root C:\\ASL\\asl-signs --fold 0` first."
        )

    set_seed(int(config["seed"]))
    device = resolve_device(str(config["device"]))
    use_amp = bool(config.get("use_amp", True)) and device.type == "cuda"
    batch_size = int(config["batch_size"])
    num_workers = int(config["num_workers"])
    max_train_batches = int(config.get("max_train_batches", 0) or 0)
    max_valid_batches = int(config.get("max_valid_batches", 0) or 0)
    dataset_mode = str(config.get("dataset_mode", "online"))

    if dataset_mode == "cache":
        cache_dir = Path(config["cache_dir"])
        filter_missing_cache = bool(config.get("filter_missing_cache", False))
        if not cache_dir.exists():
            raise FileNotFoundError(
                f"Cache directory not found: {cache_dir}. "
                "Run `python scripts\\build_feature_cache.py ...` manually before cached training."
            )
        train_dataset = CachedISLRDataset(
            train_csv,
            cache_dir,
            data_root=data_root,
            filter_missing_cache=filter_missing_cache,
        )
        valid_dataset = CachedISLRDataset(
            valid_csv,
            cache_dir,
            data_root=data_root,
            filter_missing_cache=filter_missing_cache,
        )
    else:
        train_dataset = FirstPlaceISLRDataset(
            data_root,
            train_csv,
            max_len=int(config["max_frames"]),
            center_mode=str(config["center_mode"]),
        )
        valid_dataset = FirstPlaceISLRDataset(
            data_root,
            valid_csv,
            max_len=int(config["max_frames"]),
            center_mode=str(config["center_mode"]),
        )
    train_loader = make_loader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, device=device)
    valid_loader = make_loader(valid_dataset, batch_size, shuffle=False, num_workers=num_workers, device=device)

    model = TinyISLRModel(**config["model"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(enabled=use_amp)
    param_count = count_parameters(model)

    log_lines: list[str] = []
    add(log_lines, "Tiny ISLR Baseline Training")
    add(log_lines, "=" * 27)
    add(log_lines, f"config: {args.config.resolve()}")
    add(log_lines, f"device: {device}")
    add(log_lines, f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    add(log_lines, f"use_amp: {use_amp}")
    add(log_lines, f"dataset_mode: {dataset_mode}")
    if dataset_mode == "cache":
        add(log_lines, f"cache_dir: {Path(config['cache_dir'])}")
        add(log_lines, f"filter_missing_cache: {bool(config.get('filter_missing_cache', False))}")
        add(log_lines, f"train filtered missing rows: {getattr(train_dataset, 'filtered_missing_count', 0)}")
        add(log_lines, f"valid filtered missing rows: {getattr(valid_dataset, 'filtered_missing_count', 0)}")
    add(log_lines, f"center_mode: {config['center_mode']}")
    add(log_lines, f"epochs: {config['epochs']}")
    add(log_lines, f"batch_size: {batch_size}")
    add(log_lines, f"max_train_batches: {max_train_batches}")
    add(log_lines, f"max_valid_batches: {max_valid_batches}")
    add(log_lines, f"num_workers: {num_workers}")
    add(log_lines, f"lr: {config['lr']}")
    add(log_lines, f"weight_decay: {config['weight_decay']}")
    add(log_lines, f"model param count: {param_count}")
    add(log_lines, f"train samples: {len(train_dataset)}")
    add(log_lines, f"valid samples: {len(valid_dataset)}")
    add(log_lines)

    best_valid_loss = float("inf")
    best_epoch = -1
    best_metrics: dict[str, float] = {}
    train_batch_shape: tuple[int, ...] | None = None
    train_logits_shape: tuple[int, ...] | None = None
    valid_batch_shape: tuple[int, ...] | None = None
    valid_logits_shape: tuple[int, ...] | None = None

    for epoch in range(1, int(config["epochs"]) + 1):
        start_time = time.time()
        train_loss, train_acc, train_batch_shape, train_logits_shape = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            use_amp,
            float(config["grad_clip_norm"]) if config.get("grad_clip_norm") is not None else None,
            max_train_batches,
        )
        valid_loss, valid_acc, valid_batch_shape, valid_logits_shape = validate_one_epoch(
            model,
            valid_loader,
            criterion,
            device,
            use_amp,
            max_valid_batches,
        )
        elapsed = time.time() - start_time
        metrics = {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "valid_loss": valid_loss,
            "valid_accuracy": valid_acc,
        }
        is_best = valid_loss < best_valid_loss
        if is_best:
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_metrics = metrics
            save_checkpoint(
                Path(config["best_checkpoint_path"]),
                model,
                optimizer,
                scaler,
                config,
                epoch,
                metrics,
                param_count,
            )

        save_checkpoint(
            Path(config["last_checkpoint_path"]),
            model,
            optimizer,
            scaler,
            config,
            epoch,
            metrics,
            param_count,
        )
        add(
            log_lines,
            f"epoch {epoch}: train_loss={train_loss:.6f} train_acc={train_acc:.6f} "
            f"valid_loss={valid_loss:.6f} valid_acc={valid_acc:.6f} "
            f"elapsed_sec={elapsed:.1f} best={is_best}",
        )
        print(log_lines[-1], flush=True)

    add(log_lines)
    add(log_lines, f"train batch shape: {train_batch_shape}")
    add(log_lines, f"train logits shape: {train_logits_shape}")
    add(log_lines, f"valid batch shape: {valid_batch_shape}")
    add(log_lines, f"valid logits shape: {valid_logits_shape}")
    add(log_lines, f"best epoch: {best_epoch}")
    add(log_lines, f"best valid loss: {best_valid_loss:.6f}")
    add(log_lines, f"best valid accuracy: {best_metrics.get('valid_accuracy', 0.0):.6f}")
    add(log_lines, f"best checkpoint: {Path(config['best_checkpoint_path']).resolve()}")
    add(log_lines, f"last checkpoint: {Path(config['last_checkpoint_path']).resolve()}")

    log_path = Path(config["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print("\n".join(log_lines))
    print(f"saved log: {log_path.resolve()}")


if __name__ == "__main__":
    main()
