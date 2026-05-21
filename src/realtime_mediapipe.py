from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROWS_PER_FRAME = 543
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


def _unwrap_landmarks(landmark_list: Any) -> Any:
    if landmark_list is None:
        return None
    landmarks = getattr(landmark_list, "landmark", None)
    if landmarks is not None:
        return landmarks
    if isinstance(landmark_list, list) and landmark_list and isinstance(landmark_list[0], list):
        return landmark_list[0]
    return landmark_list


def landmark_list_to_array(landmark_list: Any, count: int) -> np.ndarray:
    values = np.full((count, 3), np.nan, dtype=np.float32)
    landmarks = _unwrap_landmarks(landmark_list)
    if landmarks is None:
        return values

    for idx, landmark in enumerate(landmarks[:count]):
        values[idx, 0] = np.float32(landmark.x)
        values[idx, 1] = np.float32(landmark.y)
        values[idx, 2] = np.float32(landmark.z)
    return values


def holistic_results_to_kaggle_frame(results: Any) -> np.ndarray:
    """Convert MediaPipe Holistic output to Kaggle [543, 3] landmark order."""
    full = np.full((ROWS_PER_FRAME, 3), np.nan, dtype=np.float32)
    landmark_sources = {
        "face": getattr(results, "face_landmarks", None),
        "left_hand": getattr(results, "left_hand_landmarks", None),
        "pose": getattr(results, "pose_landmarks", None),
        "right_hand": getattr(results, "right_hand_landmarks", None),
    }

    for landmark_type, landmark_list in landmark_sources.items():
        offset = TYPE_OFFSETS[landmark_type]
        count = TYPE_COUNTS[landmark_type]
        full[offset : offset + count] = landmark_list_to_array(landmark_list, count)
    return full


@dataclass
class MediaPipeHolisticConfig:
    backend: str = "auto"
    model_asset_path: str | None = None
    static_image_mode: bool = False
    model_complexity: int = 1
    smooth_landmarks: bool = True
    refine_face_landmarks: bool = False
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class MediaPipeHolisticExtractor:
    def __init__(self, config: MediaPipeHolisticConfig | None = None) -> None:
        self.config = config or MediaPipeHolisticConfig()
        self._backend: str | None = None
        self._mp: Any | None = None
        self._holistic: Any | None = None

    def __enter__(self) -> "MediaPipeHolisticExtractor":
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def open(self) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError(
                "mediapipe is required for realtime webcam inference. "
                "Install it manually, for example: python -m pip install mediapipe opencv-python"
            ) from exc

        self._mp = mp
        requested_backend = self.config.backend.lower()
        if requested_backend not in {"auto", "solutions", "tasks"}:
            raise ValueError("MediaPipe backend must be one of: auto, solutions, tasks")

        if requested_backend in {"auto", "solutions"} and hasattr(mp, "solutions"):
            self._open_solutions(mp)
            return

        if requested_backend == "solutions":
            raise AttributeError(
                "This mediapipe package does not expose mp.solutions.holistic. "
                "Use backend='tasks' with a holistic_landmarker.task model asset, or install a legacy "
                "MediaPipe build that includes mp.solutions."
            )

        self._open_tasks(mp)

    def _open_solutions(self, mp: Any) -> None:
        self._backend = "solutions"
        mp_holistic = mp.solutions.holistic
        self._holistic = mp_holistic.Holistic(
            static_image_mode=self.config.static_image_mode,
            model_complexity=self.config.model_complexity,
            smooth_landmarks=self.config.smooth_landmarks,
            refine_face_landmarks=self.config.refine_face_landmarks,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )

    def _open_tasks(self, mp: Any) -> None:
        model_asset_path = Path(self.config.model_asset_path or "")
        if not self.config.model_asset_path or not model_asset_path.exists():
            version = getattr(mp, "__version__", "unknown")
            raise FileNotFoundError(
                "Your installed mediapipe package uses the Tasks API "
                f"(version={version}) and does not include mp.solutions.holistic. "
                "Download a MediaPipe HolisticLandmarker .task model, then set "
                "configs/inference_small_v2.json -> mediapipe.model_asset_path to that file path. "
                "Alternatively install a legacy MediaPipe build that exposes mp.solutions.holistic."
            )

        from mediapipe.tasks import python
        from mediapipe.tasks.python.vision.holistic_landmarker import (
            HolisticLandmarker,
            HolisticLandmarkerOptions,
        )

        self._backend = "tasks"
        options = HolisticLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_asset_path)),
            min_face_detection_confidence=self.config.min_detection_confidence,
            min_face_landmarks_confidence=self.config.min_tracking_confidence,
            min_pose_detection_confidence=self.config.min_detection_confidence,
            min_pose_landmarks_confidence=self.config.min_tracking_confidence,
            min_hand_landmarks_confidence=self.config.min_tracking_confidence,
        )
        self._holistic = HolisticLandmarker.create_from_options(options)

    def close(self) -> None:
        if self._holistic is not None:
            self._holistic.close()
        self._holistic = None

    def process_rgb(self, frame_rgb: np.ndarray) -> np.ndarray:
        if self._holistic is None:
            self.open()
        assert self._holistic is not None
        if self._backend == "tasks":
            assert self._mp is not None
            image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(frame_rgb),
            )
            results = self._holistic.detect(image)
        else:
            results = self._holistic.process(frame_rgb)
        return holistic_results_to_kaggle_frame(results)

    def process_bgr(self, frame_bgr: np.ndarray) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "opencv-python is required for webcam frame conversion. "
                "Install it manually, for example: python -m pip install opencv-python"
            ) from exc

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.process_rgb(frame_rgb)
