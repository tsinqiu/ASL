from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from label_utils import format_prediction, load_labels, load_zh_map
from preprocess_runtime import ROWS_PER_FRAME, first_place_preprocess_array


TYPE_OFFSETS = {
    "face": 0,
    "left_hand": 468,
    "pose": 489,
    "right_hand": 522,
}
TYPE_COUNTS = {
    "face": 468,
    "left_hand": 21,
    "pose": 33,
    "right_hand": 21,
}
DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path: str | Path, base_dir: Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else base_dir / resolved


def reorder_channels(frame: np.ndarray, input_order: str) -> np.ndarray:
    order = input_order.lower()
    channel_orders = {
        "rgb": (0, 1, 2),
        "bgr": (2, 1, 0),
        "gbr": (2, 0, 1),
        "grb": (1, 0, 2),
        "brg": (1, 2, 0),
        "rbg": (0, 2, 1),
    }
    if order not in channel_orders:
        valid = ", ".join(sorted(channel_orders))
        raise ValueError(f"camera_color_order must be one of: {valid}")
    return frame[..., channel_orders[order]]


def apply_camera_fixes(
    frame_rgb: np.ndarray,
    rotate_180: bool,
    swap_r_g: bool,
    camera_color_order: str,
) -> np.ndarray:
    fixed = reorder_channels(frame_rgb, camera_color_order)
    if rotate_180:
        fixed = np.rot90(fixed, 2)
    if swap_r_g:
        fixed = fixed[..., [1, 0, 2]]
    return np.ascontiguousarray(fixed)


class OpenCVCamera:
    def __init__(self, camera_id: int, width: int, height: int) -> None:
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap: Any | None = None

    def open(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("OpenCV is required. Install it with: pip install opencv-python") from exc
        self.cap = cv2.VideoCapture(self.camera_id)
        if self.width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera {self.camera_id} is not available. Check camera connection and permissions.")

    def read_rgb(self) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("OpenCV is required. Install it with: pip install opencv-python") from exc
        assert self.cap is not None
        ok, frame_bgr = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to read a frame from the OpenCV camera.")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.cap = None


class Picamera2Camera:
    def __init__(self, camera_id: int, width: int, height: int) -> None:
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.picam2: Any | None = None

    def open(self) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise ImportError(
                "Picamera2 is required for CSI camera input. On Raspberry Pi OS, install it with "
                "`sudo apt install python3-picamera2`, and create the venv with "
                "`python3 -m venv --system-site-packages .venv`."
            ) from exc
        self.picam2 = Picamera2(camera_num=self.camera_id)
        size = (self.width, self.height) if self.width > 0 and self.height > 0 else (640, 480)
        config = self.picam2.create_preview_configuration(main={"format": "RGB888", "size": size})
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(0.5)

    def read_rgb(self) -> np.ndarray:
        assert self.picam2 is not None
        return np.ascontiguousarray(self.picam2.capture_array())

    def release(self) -> None:
        if self.picam2 is not None:
            self.picam2.stop()
        self.picam2 = None


def open_camera(backend: str, camera_id: int, width: int, height: int) -> OpenCVCamera | Picamera2Camera:
    backend = backend.lower()
    if backend == "opencv":
        camera = OpenCVCamera(camera_id, width, height)
    elif backend == "picamera2":
        camera = Picamera2Camera(camera_id, width, height)
    else:
        raise ValueError("camera_backend must be either 'picamera2' or 'opencv'")
    camera.open()
    return camera


def landmarks_to_array(landmarks_obj: Any, count: int) -> np.ndarray:
    out = np.full((count, 3), np.nan, dtype=np.float32)
    if landmarks_obj is None:
        return out
    landmarks = getattr(landmarks_obj, "landmark", landmarks_obj)
    if isinstance(landmarks, list) and landmarks and isinstance(landmarks[0], list):
        landmarks = landmarks[0]
    for idx, landmark in enumerate(landmarks[:count]):
        out[idx, 0] = np.float32(landmark.x)
        out[idx, 1] = np.float32(landmark.y)
        out[idx, 2] = np.float32(landmark.z)
    return out


def results_to_landmark_frame(results: Any) -> np.ndarray:
    full = np.full((ROWS_PER_FRAME, 3), np.nan, dtype=np.float32)
    source_names = {
        "face": ("face_landmarks",),
        "left_hand": ("left_hand_landmarks",),
        "pose": ("pose_landmarks",),
        "right_hand": ("right_hand_landmarks",),
    }
    for landmark_type, names in source_names.items():
        source = None
        for name in names:
            source = getattr(results, name, None)
            if source is not None:
                break
        offset = TYPE_OFFSETS[landmark_type]
        count = TYPE_COUNTS[landmark_type]
        full[offset : offset + count] = landmarks_to_array(source, count)
    return full


class HolisticExtractor:
    def __init__(self, task_path: Path | None = None) -> None:
        self.task_path = task_path
        self.backend: str | None = None
        self.mp: Any | None = None
        self.model: Any | None = None

    def __enter__(self) -> "HolisticExtractor":
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def open(self) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError("MediaPipe is required. Install it with: pip install mediapipe") from exc

        self.mp = mp
        if hasattr(mp, "solutions"):
            self.backend = "solutions"
            self.model = mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                refine_face_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return

        if self.task_path is None or not self.task_path.exists():
            raise FileNotFoundError(
                "This MediaPipe install uses the Tasks API and needs a HolisticLandmarker .task file. "
                "Pass --mediapipe-task /path/to/holistic_landmarker.task, or install a MediaPipe build "
                "that provides mp.solutions.holistic."
            )

        from mediapipe.tasks import python
        from mediapipe.tasks.python.vision.holistic_landmarker import (
            HolisticLandmarker,
            HolisticLandmarkerOptions,
        )

        self.backend = "tasks"
        options = HolisticLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(self.task_path)),
            min_face_detection_confidence=0.5,
            min_face_landmarks_confidence=0.5,
            min_pose_detection_confidence=0.5,
            min_pose_landmarks_confidence=0.5,
            min_hand_landmarks_confidence=0.5,
        )
        self.model = HolisticLandmarker.create_from_options(options)

    def close(self) -> None:
        if self.model is not None:
            self.model.close()
        self.model = None

    def process_rgb(self, frame_rgb: np.ndarray) -> np.ndarray:
        if self.model is None:
            self.open()
        assert self.model is not None
        if self.backend == "tasks":
            assert self.mp is not None
            image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
            results = self.model.detect(image)
        else:
            results = self.model.process(frame_rgb)
        return results_to_landmark_frame(results)


def load_onnx_session(model_path: Path) -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required. Try: pip install onnxruntime. "
            "On Raspberry Pi, you may need an architecture-specific wheel if pip install fails."
        ) from exc
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def predict(
    session: Any,
    frames: list[np.ndarray],
    max_len: int,
    labels: dict[int, str],
    zh_map: dict[str, str],
    topk: int,
) -> list[tuple[str, float]]:
    if not frames:
        raise ValueError("No recorded frames. Press r to record before pressing s.")
    full = np.stack(frames, axis=0).astype(np.float32, copy=False)
    x, mask = first_place_preprocess_array(full, max_len=max_len)
    if not mask.any():
        raise ValueError("No usable landmarks were recorded. Keep face and hands visible.")

    inputs = {
        session.get_inputs()[0].name: x[None, :, :].astype(np.float32),
        session.get_inputs()[1].name: mask[None, :].astype(bool),
    }
    logits = session.run(None, inputs)[0][0]
    logits = logits.astype(np.float32)
    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    top_indices = np.argsort(probs)[::-1][:topk]
    predictions = [(labels.get(int(index), str(index)), float(probs[index])) for index in top_indices]

    print(f"Top1: {format_prediction(predictions[0][0], predictions[0][1], zh_map)}")
    print("Top5:")
    for rank, (sign, confidence) in enumerate(predictions, start=1):
        print(f"{rank}. {format_prediction(sign, confidence, zh_map)}")
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raspberry Pi ASL isolated sign ONNX demo.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--camera-backend", choices=("picamera2", "opencv"), default=None)
    parser.add_argument("--rotate-180", action="store_true", default=None)
    parser.add_argument("--no-rotate-180", action="store_false", dest="rotate_180")
    parser.add_argument("--swap-r-g", action="store_true", default=None)
    parser.add_argument("--no-swap-r-g", action="store_false", dest="swap_r_g")
    parser.add_argument("--camera-color-order", choices=("rgb", "bgr", "gbr", "grb", "brg", "rbg"), default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--zh-map", type=Path, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--mediapipe-task", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.config.resolve().parent
    config = load_config(args.config)
    model_path = args.model or resolve_path(config.get("model_path", "model.onnx"), base_dir)
    labels_path = args.labels or resolve_path(config.get("labels_path", "labels.json"), base_dir)
    zh_map_path = args.zh_map or resolve_path(config.get("zh_map_path", "asl_label_zh_map.json"), base_dir)
    camera_backend = str(args.camera_backend or config.get("camera_backend", "picamera2"))
    camera_id = int(args.camera if args.camera is not None else config.get("camera_id", 0))
    camera_width = int(config.get("camera_width", 640) or 0)
    camera_height = int(config.get("camera_height", 480) or 0)
    rotate_180 = bool(config.get("rotate_180", False) if args.rotate_180 is None else args.rotate_180)
    swap_r_g = bool(config.get("swap_r_g", False) if args.swap_r_g is None else args.swap_r_g)
    camera_color_order = str(args.camera_color_order or config.get("camera_color_order", "rgb"))
    max_len = int(args.max_len if args.max_len is not None else config.get("max_len", 64))
    topk = int(args.topk if args.topk is not None else config.get("topk", 5))
    fps_limit = float(config.get("record_fps_limit", 10) or 0)

    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required. Install it with: pip install opencv-python") from exc

    session = load_onnx_session(model_path)
    labels = load_labels(labels_path)
    zh_map = load_zh_map(zh_map_path)
    camera = open_camera(camera_backend, camera_id, camera_width, camera_height)

    recording = False
    recorded_frames: list[np.ndarray] = []
    last_record_time = 0.0
    task_path = args.mediapipe_task

    print("Raspberry Pi ASL isolated sign demo")
    print("r: start recording, s: stop and recognize, q: quit")
    print(
        f"camera_backend={camera_backend} rotate_180={rotate_180} "
        f"camera_color_order={camera_color_order} swap_r_g={swap_r_g}"
    )
    print("The Chinese text is only a meaning for the ASL English label, not Chinese Sign Language recognition.")

    try:
        with HolisticExtractor(task_path=task_path) as extractor:
            while True:
                frame_rgb = apply_camera_fixes(
                    camera.read_rgb(),
                    rotate_180=rotate_180,
                    swap_r_g=swap_r_g,
                    camera_color_order=camera_color_order,
                )

                if recording:
                    now = time.monotonic()
                    if fps_limit <= 0 or now - last_record_time >= 1.0 / fps_limit:
                        recorded_frames.append(extractor.process_rgb(frame_rgb))
                        last_record_time = now

                display_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    display_bgr,
                    f"{'REC' if recording else 'READY'} frames={len(recorded_frames)}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255) if recording else (255, 255, 255),
                    2,
                )
                cv2.imshow("ASL isolated sign", display_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    recorded_frames = []
                    recording = True
                    last_record_time = 0.0
                    print("Recording started.")
                if key == ord("s"):
                    recording = False
                    try:
                        predict(session, recorded_frames, max_len, labels, zh_map, topk)
                    except Exception as exc:
                        print(f"Prediction failed: {exc}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
