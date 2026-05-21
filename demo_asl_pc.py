from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.first_place_preprocess import first_place_preprocess_array
from src.label_utils import format_sign_translation, load_label_zh_map
from src.model_small import SmallISLRModel
from src.realtime_mediapipe import MediaPipeHolisticConfig, MediaPipeHolisticExtractor


DEFAULT_CONFIG = Path("configs") / "inference_small_v2.json"


@dataclass
class Prediction:
    sign: str
    confidence: float


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_index_to_sign(label_map_path: str | Path) -> dict[int, str]:
    path = Path(label_map_path)
    if not path.exists():
        raise FileNotFoundError(f"Label map not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        sign_to_index = {str(key): int(value) for key, value in json.load(f).items()}
    return {index: sign for sign, index in sign_to_index.items()}


def load_model(config: dict[str, Any], device: torch.device) -> SmallISLRModel:
    if str(config.get("model_name", "small")).lower() != "small":
        raise ValueError("PC realtime demo currently expects model_name='small'.")

    checkpoint_path = Path(config["checkpoint_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}. "
            "Train or copy a Small v2 checkpoint first, then rerun the demo."
        )

    model = SmallISLRModel(**config["model"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def predict_signs(
    model: torch.nn.Module,
    frames: list[np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    index_to_sign: dict[int, str],
) -> list[Prediction]:
    if not frames:
        raise ValueError("No recorded frames. Press r to record before pressing s.")

    full = np.stack(frames, axis=0).astype(np.float32, copy=False)
    features, mask = first_place_preprocess_array(
        full,
        max_len=int(config["max_frames"]),
        center_mode=str(config.get("center_mode", "notebook_strict")),
    )
    if not bool(mask.any()):
        raise ValueError("No usable landmarks were recorded. Try recording with hands and face visible.")

    x = torch.from_numpy(features).unsqueeze(0).to(device=device, dtype=torch.float32)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(device=device, dtype=torch.bool)
    logits = model(x, mask_t)
    probs = torch.softmax(logits, dim=-1)[0]
    top_k = min(int(config.get("top_k", 5)), probs.numel())
    values, indices = torch.topk(probs, k=top_k)

    predictions: list[Prediction] = []
    for confidence, index in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()):
        predictions.append(Prediction(sign=index_to_sign.get(int(index), str(index)), confidence=float(confidence)))
    return predictions


def print_predictions(predictions: list[Prediction], zh_map: dict[str, str]) -> None:
    if not predictions:
        return

    top1 = predictions[0]
    print(f"Top1: {format_sign_translation(top1.sign, zh_map)}  confidence={top1.confidence:.4f}")
    print("Top5:")
    for rank, prediction in enumerate(predictions, start=1):
        print(f"{rank}. {format_sign_translation(prediction.sign, zh_map)}  confidence={prediction.confidence:.4f}")


def draw_overlay(frame: np.ndarray, recording: bool, frame_count: int, predictions: list[Prediction], zh_map: dict[str, str]) -> None:
    try:
        import cv2
    except ImportError:
        return

    status = "REC" if recording else "READY"
    cv2.putText(frame, f"{status} frames={frame_count}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, "r: record   s: stop+predict   q: quit", (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    if predictions:
        top1 = predictions[0]
        text = f"Top1: {format_sign_translation(top1.sign, zh_map)} {top1.confidence:.2f}"
        cv2.putText(frame, text, (16, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PC webcam demo for ASL isolated sign inference.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint override.")
    parser.add_argument("--camera-index", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    if args.checkpoint is not None:
        config["checkpoint_path"] = str(args.checkpoint)
    if args.camera_index is not None:
        config["camera_index"] = int(args.camera_index)

    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for the PC webcam demo. "
            "Install it manually: python -m pip install opencv-python mediapipe"
        ) from exc

    device = resolve_device(str(config.get("device", "auto")))
    model = load_model(config, device)
    index_to_sign = load_index_to_sign(config["label_map_path"])
    zh_map = load_label_zh_map(config.get("label_zh_map_path", "data/asl_label_zh_map.json"))

    mp_config_raw = config.get("mediapipe", {}) or {}
    mp_config = MediaPipeHolisticConfig(
        backend=str(mp_config_raw.get("backend", "auto")),
        model_asset_path=str(mp_config_raw.get("model_asset_path") or "") or None,
        model_complexity=int(mp_config_raw.get("model_complexity", 1)),
        smooth_landmarks=bool(mp_config_raw.get("smooth_landmarks", True)),
        refine_face_landmarks=bool(mp_config_raw.get("refine_face_landmarks", False)),
        min_detection_confidence=float(mp_config_raw.get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(mp_config_raw.get("min_tracking_confidence", 0.5)),
    )

    cap = cv2.VideoCapture(int(config.get("camera_index", 0)))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {config.get('camera_index', 0)}")

    recording = False
    recorded_frames: list[np.ndarray] = []
    last_predictions: list[Prediction] = []
    mirror_preview = bool(config.get("mirror_preview", True))

    print("PC ASL isolated sign demo")
    print("Press r to start recording, s to stop and predict, q to quit.")
    print("This recognizes ASL isolated signs and displays Chinese label meanings.")

    with MediaPipeHolisticExtractor(mp_config) as extractor:
        while True:
            ok, raw_frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read from camera.")

            landmark_frame = extractor.process_bgr(raw_frame)
            display_frame = cv2.flip(raw_frame, 1) if mirror_preview else raw_frame

            if recording:
                recorded_frames.append(landmark_frame)

            draw_overlay(display_frame, recording, len(recorded_frames), last_predictions, zh_map)
            cv2.imshow("ASL isolated sign PC demo", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                recorded_frames = []
                last_predictions = []
                recording = True
                print("Recording started.")
            if key == ord("s"):
                recording = False
                try:
                    last_predictions = predict_signs(model, recorded_frames, config, device, index_to_sign)
                    print_predictions(last_predictions, zh_map)
                except Exception as exc:
                    last_predictions = []
                    print(f"Prediction failed: {exc}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
