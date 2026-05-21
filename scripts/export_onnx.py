from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_small import SmallISLRModel
from model_tiny import TinyISLRModel


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    model_name = str(config.get("model_name", "small")).lower()
    if model_name == "small":
        return SmallISLRModel(**config["model"])
    if model_name == "tiny":
        return TinyISLRModel(**config["model"])
    raise ValueError(f"Unsupported model_name={model_name!r}; expected 'small' or 'tiny'")


def load_checkpoint_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise TypeError(
        "Unsupported checkpoint format. Expected key 'model_state_dict', key 'state_dict', "
        "or a raw PyTorch state dict."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an ISLR PyTorch checkpoint to ONNX.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'onnx' package is required by torch.onnx.export. "
            "Install it in this environment with: python -m pip install onnx"
        ) from exc

    config = load_json(args.config)
    model_name = str(config.get("model_name", "small")).lower()
    max_len = int(config["max_frames"])
    input_dim = int(config["model"]["input_dim"])
    num_classes = int(config["model"]["num_classes"])

    model = build_model(config)
    model.load_state_dict(load_checkpoint_state_dict(args.checkpoint))
    model.eval()

    dummy_x = torch.zeros((1, max_len, input_dim), dtype=torch.float32)
    dummy_mask = torch.ones((1, max_len), dtype=torch.bool)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_x, dummy_mask),
        args.output,
        input_names=["x", "mask"],
        output_names=["logits"],
        dynamic_axes={
            "x": {0: "batch"},
            "mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=int(args.opset),
    )

    print(f"config path: {args.config.resolve()}")
    print(f"checkpoint path: {args.checkpoint.resolve()}")
    print(f"output path: {args.output.resolve()}")
    print(f"max_len: {max_len}")
    print(f"input_dim: {input_dim}")
    print(f"num_classes: {num_classes}")
    print(f"model_name: {model_name}")


if __name__ == "__main__":
    main()
