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


DEFAULT_CONFIG = Path("configs") / "inference_small_v2.json"
DEFAULT_OUTPUT = Path("outputs") / "small_v2_len128.onnx"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_model(config: dict[str, Any], checkpoint_path: Path, device: torch.device) -> SmallISLRModel:
    if str(config.get("model_name", "small")).lower() != "small":
        raise ValueError("ONNX export currently expects model_name='small'.")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    model = SmallISLRModel(**config["model"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Small v2 checkpoint to ONNX for inference experiments.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    checkpoint_path = args.checkpoint or Path(config["checkpoint_path"])
    device = torch.device(args.device)
    model = load_model(config, checkpoint_path, device)

    max_len = int(config["max_frames"])
    input_dim = int(config["model"]["input_dim"])
    dummy_x = torch.zeros((1, max_len, input_dim), dtype=torch.float32, device=device)
    dummy_mask = torch.ones((1, max_len), dtype=torch.bool, device=device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_x, dummy_mask),
        args.output,
        input_names=["x", "mask"],
        output_names=["logits"],
        dynamic_axes={
            "x": {0: "batch", 1: "frames"},
            "mask": {0: "batch", 1: "frames"},
            "logits": {0: "batch"},
        },
        opset_version=int(args.opset),
    )
    print(f"exported ONNX: {args.output.resolve()}")


if __name__ == "__main__":
    main()
