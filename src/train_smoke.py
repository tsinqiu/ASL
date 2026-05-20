from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from first_place_preprocess import CENTER_MODES, DEFAULT_DATA_ROOT, FirstPlaceISLRDataset
from model_tiny import TinyISLRModel


def add(lines: list[str], text: str = "") -> None:
    lines.append(text)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def move_batch_to_device(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch["x"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    y = batch["y"].to(device, non_blocking=True)
    return x, mask, y


def run_train_epoch(
    model: TinyISLRModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int,
) -> tuple[float, tuple[int, ...], tuple[int, ...], int, int]:
    model.train()
    total_loss = 0.0
    total_examples = 0
    first_batch_shape: tuple[int, ...] | None = None
    first_logits_shape: tuple[int, ...] | None = None
    first_x_nan_count = 0
    first_logits_nan_count = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x, mask, y = move_batch_to_device(batch, device)
        if first_batch_shape is None:
            first_batch_shape = tuple(x.shape)
            first_x_nan_count = int(torch.isnan(x).sum().item())

        optimizer.zero_grad(set_to_none=True)
        logits = model(x, mask)
        if first_logits_shape is None:
            first_logits_shape = tuple(logits.shape)
            first_logits_nan_count = int(torch.isnan(logits).sum().item())
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    if total_examples == 0:
        raise RuntimeError("No training batches were processed.")
    assert first_batch_shape is not None
    assert first_logits_shape is not None
    return (
        total_loss / total_examples,
        first_batch_shape,
        first_logits_shape,
        first_x_nan_count,
        first_logits_nan_count,
    )


@torch.no_grad()
def run_valid_epoch(
    model: TinyISLRModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int,
) -> tuple[float, float, tuple[int, ...], tuple[int, ...], int, int]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    first_batch_shape: tuple[int, ...] | None = None
    first_logits_shape: tuple[int, ...] | None = None
    first_x_nan_count = 0
    first_logits_nan_count = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x, mask, y = move_batch_to_device(batch, device)
        if first_batch_shape is None:
            first_batch_shape = tuple(x.shape)
            first_x_nan_count = int(torch.isnan(x).sum().item())
        logits = model(x, mask)
        if first_logits_shape is None:
            first_logits_shape = tuple(logits.shape)
            first_logits_nan_count = int(torch.isnan(logits).sum().item())
        loss = criterion(logits, y)

        pred = logits.argmax(dim=-1)
        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += int((pred == y).sum().item())
        total_examples += batch_size

    if total_examples == 0:
        raise RuntimeError("No validation batches were processed.")
    assert first_batch_shape is not None
    assert first_logits_shape is not None
    return (
        total_loss / total_examples,
        total_correct / total_examples,
        first_batch_shape,
        first_logits_shape,
        first_x_nan_count,
        first_logits_nan_count,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny ISLR smoke training loop.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--max-train-batches", type=int, default=20)
    parser.add_argument("--max-valid-batches", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs") / "smoke_model.pt")
    parser.add_argument("--center-mode", choices=sorted(CENTER_MODES), default="notebook_strict")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be positive, got {args.epochs}")
    if args.max_train_batches <= 0 or args.max_valid_batches <= 0:
        raise ValueError("--max-train-batches and --max-valid-batches must be positive")

    train_csv = Path("outputs") / f"first_place_train_fold{args.fold}.csv"
    valid_csv = Path("outputs") / f"first_place_valid_fold{args.fold}.csv"
    if not train_csv.exists() or not valid_csv.exists():
        raise FileNotFoundError(
            "First-place fold CSV files are missing. Run "
            f"`python scripts/check_first_place_preprocess.py --data-root {args.data_root} --fold {args.fold}` first."
        )

    torch.manual_seed(123)
    device = resolve_device(args.device)
    train_dataset = FirstPlaceISLRDataset(
        args.data_root,
        train_csv,
        max_len=args.max_frames,
        center_mode=args.center_mode,
    )
    valid_dataset = FirstPlaceISLRDataset(
        args.data_root,
        valid_csv,
        max_len=args.max_frames,
        center_mode=args.center_mode,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = TinyISLRModel(input_dim=708, num_classes=250).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    param_count = count_parameters(model)

    lines: list[str] = []
    add(lines, "Tiny ISLR Smoke Training")
    add(lines, "=" * 24)
    add(lines, f"device: {device}")
    add(lines, f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    add(lines, f"center_mode: {args.center_mode}")
    add(lines, f"model param count: {param_count}")
    add(lines, f"train samples: {len(train_dataset)}")
    add(lines, f"valid samples: {len(valid_dataset)}")
    add(lines, f"batch_size: {args.batch_size}")
    add(lines, f"max_frames: {args.max_frames}")
    add(lines, f"max_train_batches: {args.max_train_batches}")
    add(lines, f"max_valid_batches: {args.max_valid_batches}")
    add(lines, f"epochs: {args.epochs}")
    add(lines)

    last_train_loss = 0.0
    last_valid_loss = 0.0
    last_valid_acc = 0.0
    train_batch_shape: tuple[int, ...] | None = None
    train_logits_shape: tuple[int, ...] | None = None
    valid_batch_shape: tuple[int, ...] | None = None
    valid_logits_shape: tuple[int, ...] | None = None
    train_x_nan = 0
    train_logits_nan = 0
    valid_x_nan = 0
    valid_logits_nan = 0

    for epoch in range(args.epochs):
        (
            last_train_loss,
            train_batch_shape,
            train_logits_shape,
            train_x_nan,
            train_logits_nan,
        ) = run_train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.max_train_batches,
        )
        (
            last_valid_loss,
            last_valid_acc,
            valid_batch_shape,
            valid_logits_shape,
            valid_x_nan,
            valid_logits_nan,
        ) = run_valid_epoch(
            model,
            valid_loader,
            criterion,
            device,
            args.max_valid_batches,
        )
        add(
            lines,
            f"epoch {epoch + 1}: train_loss={last_train_loss:.6f} "
            f"valid_loss={last_valid_loss:.6f} valid_accuracy={last_valid_acc:.6f}",
        )

    add(lines)
    add(lines, f"train batch shape: {train_batch_shape}")
    add(lines, f"train logits shape: {train_logits_shape}")
    add(lines, f"train batch x NaN count: {train_x_nan}")
    add(lines, f"train logits NaN count: {train_logits_nan}")
    add(lines, f"valid batch shape: {valid_batch_shape}")
    add(lines, f"valid logits shape: {valid_logits_shape}")
    add(lines, f"valid batch x NaN count: {valid_x_nan}")
    add(lines, f"valid logits NaN count: {valid_logits_nan}")
    add(lines, f"final train loss: {last_train_loss:.6f}")
    add(lines, f"final valid loss: {last_valid_loss:.6f}")
    add(lines, f"final valid accuracy: {last_valid_acc:.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "input_dim": 708,
                "num_classes": 250,
                "max_frames": args.max_frames,
                "center_mode": args.center_mode,
                "param_count": param_count,
            },
        },
        args.output,
    )
    add(lines, f"saved model: {args.output.resolve()}")

    report = "\n".join(lines)
    log_path = Path("outputs") / "smoke_train_log.txt"
    log_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"saved log: {log_path.resolve()}")


if __name__ == "__main__":
    main()
