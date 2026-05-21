from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from cached_dataset import CachedISLRDataset
from first_place_preprocess import FirstPlaceISLRDataset
from train_baseline import build_model, load_config, move_batch_to_device
from train_smoke import resolve_device


DEFAULT_CONFIG = Path("configs") / "tiny_baseline_cached.json"
DEFAULT_CHECKPOINT = Path("outputs") / "baseline_cached_tiny_fold0_best.pt"
DEFAULT_OUTPUT_JSON = Path("outputs") / "eval_tiny_baseline_valid.json"
DEFAULT_PER_CLASS_CSV = Path("outputs") / "eval_tiny_baseline_per_class.csv"
PER_CLASS_FIELDS = ("label", "sign", "support", "correct_top1", "top1_accuracy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained ISLR baseline checkpoint.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=("train", "valid"), default="valid")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--per-class-csv", type=Path, default=DEFAULT_PER_CLASS_CSV)
    return parser.parse_args()


def build_dataset(config: dict[str, Any], split: str) -> CachedISLRDataset | FirstPlaceISLRDataset:
    data_root = Path(config["data_root"])
    csv_path = Path(config[f"{split}_csv"])
    dataset_mode = str(config.get("dataset_mode", "online"))

    if dataset_mode == "cache":
        cache_dir = Path(config["cache_dir"])
        return CachedISLRDataset(
            csv_path,
            cache_dir,
            data_root=data_root,
            filter_missing_cache=bool(config.get("filter_missing_cache", False)),
            max_len=int(config["max_frames"]),
            feature_dim=int(config["model"].get("input_dim", 708)),
        )

    if dataset_mode == "online":
        print(
            "evaluate.py is optimized for dataset_mode='cache'; falling back to online parquet preprocessing.",
            flush=True,
        )
        return FirstPlaceISLRDataset(
            data_root,
            csv_path,
            max_len=int(config["max_frames"]),
            center_mode=str(config["center_mode"]),
        )

    raise ValueError(
        f"Unsupported dataset_mode={dataset_mode!r}. evaluate.py currently prioritizes dataset_mode='cache'."
    )


def make_loader(
    dataset: CachedISLRDataset | FirstPlaceISLRDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def load_checkpoint_model(checkpoint_path: Path, config: dict[str, Any], device: torch.device) -> nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}. "
            "Train the baseline first, or pass --checkpoint to an existing .pt file."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint is missing 'model_state_dict': {checkpoint_path}")

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[dict[str, float | int], list[dict[str, int | float | str]]]:
    model.eval()
    total_loss = 0.0
    total_correct_top1 = 0
    total_correct_top5 = 0
    total_examples = 0
    support = torch.zeros(num_classes, dtype=torch.long)
    correct_top1 = torch.zeros(num_classes, dtype=torch.long)
    label_to_sign = {int(label): str(sign) for sign, label in loader.dataset.label_map.items()}

    for batch in loader:
        x, mask, y = move_batch_to_device(batch, device)
        logits = model(x, mask)
        loss = criterion(logits, y)
        batch_size = x.shape[0]

        pred_top1 = logits.argmax(dim=-1)
        topk = min(5, logits.shape[-1])
        pred_top5 = logits.topk(k=topk, dim=-1).indices
        correct_top1_batch = pred_top1.eq(y)
        correct_top5_batch = pred_top5.eq(y.unsqueeze(1)).any(dim=1)

        total_loss += float(loss.item()) * batch_size
        total_correct_top1 += int(correct_top1_batch.sum().item())
        total_correct_top5 += int(correct_top5_batch.sum().item())
        total_examples += batch_size

        labels_cpu = y.detach().cpu()
        support += torch.bincount(labels_cpu, minlength=num_classes)
        correct_top1 += torch.bincount(
            labels_cpu[correct_top1_batch.detach().cpu()],
            minlength=num_classes,
        )

    if total_examples == 0:
        raise RuntimeError("No samples were processed during evaluation.")

    summary = {
        "loss": total_loss / total_examples,
        "top1_accuracy": total_correct_top1 / total_examples,
        "top5_accuracy": total_correct_top5 / total_examples,
        "num_samples": total_examples,
    }
    per_class_rows: list[dict[str, int | float | str]] = []
    for label in range(num_classes):
        class_support = int(support[label].item())
        if class_support == 0:
            continue
        class_correct = int(correct_top1[label].item())
        per_class_rows.append(
            {
                "label": label,
                "sign": label_to_sign.get(label, ""),
                "support": class_support,
                "correct_top1": class_correct,
                "top1_accuracy": class_correct / class_support,
            }
        )
    return summary, per_class_rows


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_per_class_csv(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_CLASS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    batch_size = int(args.batch_size or config["batch_size"])
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    device = resolve_device(str(config["device"]))
    dataset = build_dataset(config, args.split)
    loader = make_loader(dataset, batch_size, int(config.get("num_workers", 0)), device)
    model = load_checkpoint_model(args.checkpoint, config, device)
    criterion = nn.CrossEntropyLoss()
    summary, per_class_rows = evaluate_model(
        model,
        loader,
        criterion,
        device,
        num_classes=int(config["model"]["num_classes"]),
    )

    payload = {
        **summary,
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "split": args.split,
        "batch_size": batch_size,
        "device": str(device),
    }
    save_json(args.output_json, payload)
    save_per_class_csv(args.per_class_csv, per_class_rows)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"saved json: {args.output_json.resolve()}")
    print(f"saved per-class csv: {args.per_class_csv.resolve()}")


if __name__ == "__main__":
    main()
