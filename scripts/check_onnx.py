from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from export_onnx import build_model, load_checkpoint_state_dict


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_onnxruntime() -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError("onnxruntime is required for ONNX checking. Install it with: pip install onnxruntime") from exc
    return ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check an exported ISLR ONNX model on dummy input.")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--input-dim", type=int, default=708)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--compare-pytorch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ort = load_onnxruntime()
    if not args.onnx.exists():
        raise FileNotFoundError(f"ONNX file not found: {args.onnx}")

    x = np.zeros((1, int(args.max_len), int(args.input_dim)), dtype=np.float32)
    mask = np.ones((1, int(args.max_len)), dtype=bool)
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_names = [item.name for item in session.get_inputs()]
    output_names = [item.name for item in session.get_outputs()]
    logits = session.run(None, {"x": x, "mask": mask})[0]

    print(f"ONNX input names: {input_names}")
    print(f"ONNX output names: {output_names}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"NaN count: {int(np.isnan(logits).sum())}")
    if tuple(logits.shape) != (1, 250):
        raise ValueError(f"Expected logits shape (1, 250), got {tuple(logits.shape)}")

    if args.compare_pytorch:
        if args.config is None or args.checkpoint is None:
            raise ValueError("--compare-pytorch requires both --config and --checkpoint")
        config = load_json(args.config)
        model = build_model(config)
        model.load_state_dict(load_checkpoint_state_dict(args.checkpoint))
        model.eval()
        with torch.no_grad():
            torch_logits = model(torch.from_numpy(x), torch.from_numpy(mask)).detach().cpu().numpy()
        max_abs_diff = float(np.max(np.abs(torch_logits - logits)))
        print(f"max_abs_diff: {max_abs_diff:.8f}")


if __name__ == "__main__":
    main()
