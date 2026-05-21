import unittest

import numpy as np

from src.realtime_mediapipe import ROWS_PER_FRAME, holistic_results_to_kaggle_frame


class DummyLandmark:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class DummyLandmarkList:
    def __init__(self, count: int, value: float) -> None:
        self.landmark = [DummyLandmark(value, value + 1.0, value + 2.0) for _ in range(count)]


class DummyResults:
    face_landmarks = DummyLandmarkList(468, 1.0)
    left_hand_landmarks = DummyLandmarkList(21, 2.0)
    pose_landmarks = DummyLandmarkList(33, 3.0)
    right_hand_landmarks = DummyLandmarkList(21, 4.0)


class RealtimeMediaPipeTest(unittest.TestCase):
    def test_holistic_results_to_kaggle_frame_uses_expected_offsets(self) -> None:
        frame = holistic_results_to_kaggle_frame(DummyResults())

        self.assertEqual(frame.shape, (ROWS_PER_FRAME, 3))
        np.testing.assert_allclose(frame[0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(frame[468], [2.0, 3.0, 4.0])
        np.testing.assert_allclose(frame[489], [3.0, 4.0, 5.0])
        np.testing.assert_allclose(frame[522], [4.0, 5.0, 6.0])


if __name__ == "__main__":
    unittest.main()
