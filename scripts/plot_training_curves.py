from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_CSV = Path("outputs") / "baseline_cached_metrics.csv"
DEFAULT_OUTPUT_DIR = Path("outputs") / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot tiny baseline training curves from metrics CSV.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_metrics(csv_path: Path) -> dict[str, list[float]]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Metrics CSV not found: {csv_path}. "
            "Run train_baseline.py manually first to generate the metrics CSV."
        )

    required_columns = {"epoch", "train_loss", "train_acc", "valid_loss", "valid_acc"}
    metrics = {column: [] for column in required_columns}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Metrics CSV is missing columns: {sorted(missing)}")
        for row in reader:
            for column in required_columns:
                metrics[column].append(float(row[column]))

    if not metrics["epoch"]:
        raise ValueError(f"Metrics CSV contains no epoch rows: {csv_path}")
    return metrics


def plot_curve(
    epochs: list[float],
    train_values: list[float],
    valid_values: list[float],
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plot_training_curves.py. Install it before plotting.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_values, marker="o", label="train")
    plt.plot(epochs, valid_values, marker="o", label="valid")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    metrics = load_metrics(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loss_path = args.output_dir / "tiny_baseline_loss_curve.png"
    acc_path = args.output_dir / "tiny_baseline_acc_curve.png"

    plot_curve(
        metrics["epoch"],
        metrics["train_loss"],
        metrics["valid_loss"],
        ylabel="Loss",
        title="Tiny Baseline Loss Curve",
        output_path=loss_path,
    )
    plot_curve(
        metrics["epoch"],
        metrics["train_acc"],
        metrics["valid_acc"],
        ylabel="Accuracy",
        title="Tiny Baseline Accuracy Curve",
        output_path=acc_path,
    )

    print(f"saved loss curve: {loss_path.resolve()}")
    print(f"saved accuracy curve: {acc_path.resolve()}")


if __name__ == "__main__":
    main()
