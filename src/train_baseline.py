from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from augment import apply_augmentation
from cached_dataset import CachedISLRDataset
from first_place_preprocess import CENTER_MODES, FirstPlaceISLRDataset
from model_small import SmallISLRModel
from model_tiny import TinyISLRModel
from train_smoke import count_parameters, resolve_device


DEFAULT_CONFIG = Path("configs") / "tiny_baseline.json"
DEFAULT_METRICS_CSV_PATH = Path("outputs") / "baseline_cached_metrics.csv"
METRICS_CSV_FIELDS = (
    "epoch",
    "train_loss",
    "train_acc",
    "valid_loss",
    "valid_acc",
    "elapsed_sec",
    "is_best",
    "lr",
    "augmentation_enabled",
)


def add(lines: list[str], text: str = "") -> None:
    lines.append(text)


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def initialize_metrics_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_CSV_FIELDS)
        writer.writeheader()


def append_metrics_csv(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_CSV_FIELDS)
        writer.writerow(row)


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
    best_metric = str(config.get("best_metric", "valid_loss"))
    best_mode = str(config.get("best_mode", "min"))
    if best_metric not in {"valid_loss", "valid_acc"}:
        raise ValueError("best_metric must be either 'valid_loss' or 'valid_acc'")
    if best_mode not in {"min", "max"}:
        raise ValueError("best_mode must be either 'min' or 'max'")
    label_smoothing = float(config.get("label_smoothing", 0.0))
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0.0, 1.0)")
    scheduler_config = config.get("scheduler", {"name": "none"}) or {"name": "none"}
    scheduler_name = str(scheduler_config.get("name", "none")).lower()
    if scheduler_name not in {"none", "cosine"}:
        raise ValueError("scheduler.name must be either 'none' or 'cosine'")
    if scheduler_name == "cosine":
        if int(scheduler_config.get("t_max", 0)) <= 0:
            raise ValueError("scheduler.t_max must be positive when scheduler.name='cosine'")
        if float(scheduler_config.get("eta_min", 0.0)) < 0.0:
            raise ValueError("scheduler.eta_min must be non-negative")


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


def build_model(config: dict[str, Any]) -> nn.Module:
    model_name = str(config.get("model_name", "tiny")).lower()
    if model_name == "tiny":
        return TinyISLRModel(**config["model"])
    if model_name == "small":
        return SmallISLRModel(**config["model"])
    raise ValueError("model_name must be either 'tiny' or 'small'")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> Any | None:
    scheduler_config = config.get("scheduler", {"name": "none"}) or {"name": "none"}
    scheduler_name = str(scheduler_config.get("name", "none")).lower()
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_config["t_max"]),
            eta_min=float(scheduler_config.get("eta_min", 0.0)),
        )
    raise ValueError("scheduler.name must be either 'none' or 'cosine'")


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def is_better_metric(current: float, best: float | None, mode: str) -> bool:
    if best is None:
        return True
    if mode == "min":
        return current < best
    if mode == "max":
        return current > best
    raise ValueError("best_mode must be either 'min' or 'max'")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    grad_clip_norm: float | None,
    augmentation_config: dict[str, Any] | None = None,
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
        x, mask = apply_augmentation(x, mask, augmentation_config)

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
    model: nn.Module,
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
    model: nn.Module,
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
    parser = argparse.ArgumentParser(description="Train a single-fold ISLR baseline from JSON config.")
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
    max_frames = int(config["max_frames"])
    dataset_mode = str(config.get("dataset_mode", "online"))
    metrics_csv_path = Path(config.get("metrics_csv_path") or DEFAULT_METRICS_CSV_PATH)
    best_metric = str(config.get("best_metric", "valid_loss"))
    best_mode = str(config.get("best_mode", "min"))
    label_smoothing = float(config.get("label_smoothing", 0.0))
    augmentation_config = config.get("augmentation", {"enabled": False}) or {"enabled": False}
    augmentation_enabled = bool(augmentation_config.get("enabled", False))
    scheduler_config = config.get("scheduler", {"name": "none"}) or {"name": "none"}
    scheduler_name = str(scheduler_config.get("name", "none")).lower()

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
            max_len=max_frames,
            feature_dim=int(config["model"].get("input_dim", 708)),
        )
        valid_dataset = CachedISLRDataset(
            valid_csv,
            cache_dir,
            data_root=data_root,
            filter_missing_cache=filter_missing_cache,
            max_len=max_frames,
            feature_dim=int(config["model"].get("input_dim", 708)),
        )
    else:
        train_dataset = FirstPlaceISLRDataset(
            data_root,
            train_csv,
            max_len=max_frames,
            center_mode=str(config["center_mode"]),
        )
        valid_dataset = FirstPlaceISLRDataset(
            data_root,
            valid_csv,
            max_len=max_frames,
            center_mode=str(config["center_mode"]),
        )
    train_loader = make_loader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, device=device)
    valid_loader = make_loader(valid_dataset, batch_size, shuffle=False, num_workers=num_workers, device=device)

    model_name = str(config.get("model_name", "tiny")).lower()
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    param_count = count_parameters(model)

    log_lines: list[str] = []
    add(log_lines, f"{model_name.title()} ISLR Baseline Training")
    add(log_lines, "=" * (len(log_lines[-1]) if log_lines else 28))
    add(log_lines, f"config: {args.config.resolve()}")
    add(log_lines, f"device: {device}")
    add(log_lines, f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    add(log_lines, f"use_amp: {use_amp}")
    add(log_lines, f"model_name: {model_name}")
    add(log_lines, f"dataset_mode: {dataset_mode}")
    if dataset_mode == "cache":
        add(log_lines, f"cache_dir: {Path(config['cache_dir'])}")
        add(log_lines, f"filter_missing_cache: {bool(config.get('filter_missing_cache', False))}")
        add(log_lines, f"train filtered missing rows: {getattr(train_dataset, 'filtered_missing_count', 0)}")
        add(log_lines, f"valid filtered missing rows: {getattr(valid_dataset, 'filtered_missing_count', 0)}")
    add(log_lines, f"center_mode: {config['center_mode']}")
    add(log_lines, f"epochs: {config['epochs']}")
    add(log_lines, f"batch_size: {batch_size}")
    add(log_lines, f"max_frames: {max_frames}")
    add(log_lines, f"max_train_batches: {max_train_batches}")
    add(log_lines, f"max_valid_batches: {max_valid_batches}")
    add(log_lines, f"num_workers: {num_workers}")
    add(log_lines, f"lr: {config['lr']}")
    add(log_lines, f"weight_decay: {config['weight_decay']}")
    add(log_lines, f"label_smoothing: {label_smoothing}")
    add(log_lines, f"augmentation_enabled: {augmentation_enabled}")
    if augmentation_enabled:
        for key in sorted(augmentation_config):
            add(log_lines, f"augmentation_{key}: {augmentation_config[key]}")
    add(log_lines, f"scheduler: {scheduler_name}")
    if scheduler_name == "cosine":
        add(log_lines, f"scheduler_t_max: {int(scheduler_config['t_max'])}")
        add(log_lines, f"scheduler_eta_min: {float(scheduler_config.get('eta_min', 0.0))}")
    add(log_lines, f"best_metric: {best_metric}")
    add(log_lines, f"best_mode: {best_mode}")
    add(log_lines, f"metrics_csv_path: {metrics_csv_path}")
    add(log_lines, f"model param count: {param_count}")
    add(log_lines, f"train samples: {len(train_dataset)}")
    add(log_lines, f"valid samples: {len(valid_dataset)}")
    add(log_lines)

    best_metric_value: float | None = None
    best_epoch = -1
    best_metrics: dict[str, float] = {}
    train_batch_shape: tuple[int, ...] | None = None
    train_logits_shape: tuple[int, ...] | None = None
    valid_batch_shape: tuple[int, ...] | None = None
    valid_logits_shape: tuple[int, ...] | None = None
    initialize_metrics_csv(metrics_csv_path)

    for epoch in range(1, int(config["epochs"]) + 1):
        start_time = time.time()
        lr = get_current_lr(optimizer)
        train_loss, train_acc, train_batch_shape, train_logits_shape = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            use_amp,
            float(config["grad_clip_norm"]) if config.get("grad_clip_norm") is not None else None,
            augmentation_config,
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
            "train_acc": train_acc,
            "train_accuracy": train_acc,
            "valid_loss": valid_loss,
            "valid_acc": valid_acc,
            "valid_accuracy": valid_acc,
        }
        current_best_metric_value = float(metrics[best_metric])
        is_best = is_better_metric(current_best_metric_value, best_metric_value, best_mode)
        if is_best:
            best_metric_value = current_best_metric_value
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

        if scheduler is not None:
            scheduler.step()

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
            f"lr={lr:.8f} elapsed_sec={elapsed:.1f} "
            f"best_metric={best_metric} best_metric_value={current_best_metric_value:.6f} "
            f"best={is_best}",
        )
        append_metrics_csv(
            metrics_csv_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "valid_loss": valid_loss,
                "valid_acc": valid_acc,
                "elapsed_sec": elapsed,
                "is_best": is_best,
                "lr": lr,
                "augmentation_enabled": augmentation_enabled,
            },
        )
        print(log_lines[-1], flush=True)

    add(log_lines)
    add(log_lines, f"train batch shape: {train_batch_shape}")
    add(log_lines, f"train logits shape: {train_logits_shape}")
    add(log_lines, f"valid batch shape: {valid_batch_shape}")
    add(log_lines, f"valid logits shape: {valid_logits_shape}")
    add(log_lines, f"best epoch: {best_epoch}")
    add(log_lines, f"best metric: {best_metric}")
    add(log_lines, f"best mode: {best_mode}")
    add(log_lines, f"best metric value: {(best_metric_value if best_metric_value is not None else 0.0):.6f}")
    add(log_lines, f"best valid loss: {best_metrics.get('valid_loss', 0.0):.6f}")
    add(log_lines, f"best valid accuracy: {best_metrics.get('valid_accuracy', 0.0):.6f}")
    add(log_lines, f"best checkpoint: {Path(config['best_checkpoint_path']).resolve()}")
    add(log_lines, f"last checkpoint: {Path(config['last_checkpoint_path']).resolve()}")
    add(log_lines, f"metrics csv: {metrics_csv_path.resolve()}")

    log_path = Path(config["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print("\n".join(log_lines))
    print(f"saved log: {log_path.resolve()}")


if __name__ == "__main__":
    main()
