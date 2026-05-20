import unittest

import numpy as np
import pandas as pd

from src.first_place_preprocess import (
    CENTER_MODES,
    POINT_LANDMARKS,
    ROWS_PER_FRAME,
    first_place_preprocess_array,
    resolve_center_landmarks,
    restore_full_landmark_tensor,
)


class FirstPlacePreprocessTest(unittest.TestCase):
    def test_restore_full_landmark_tensor_uses_global_offsets(self) -> None:
        df = pd.DataFrame(
            [
                {"frame": 2, "type": "face", "landmark_index": 1, "x": 1.0, "y": 2.0, "z": 3.0},
                {"frame": 2, "type": "left_hand", "landmark_index": 0, "x": 4.0, "y": 5.0, "z": 6.0},
                {"frame": 2, "type": "pose", "landmark_index": 11, "x": 7.0, "y": 8.0, "z": 9.0},
                {"frame": 5, "type": "right_hand", "landmark_index": 20, "x": 10.0, "y": 11.0, "z": 12.0},
            ]
        )

        tensor = restore_full_landmark_tensor(df)

        self.assertEqual(tensor.shape, (2, ROWS_PER_FRAME, 3))
        np.testing.assert_allclose(tensor[0, 1], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(tensor[0, 468], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(tensor[0, 500], [7.0, 8.0, 9.0])
        np.testing.assert_allclose(tensor[1, 542], [10.0, 11.0, 12.0])

    def test_first_place_preprocess_outputs_padded_708_features_without_nan(self) -> None:
        full = np.full((3, ROWS_PER_FRAME, 3), np.nan, dtype=np.float32)
        for t in range(3):
            full[t, POINT_LANDMARKS, 0] = 0.1 + t
            full[t, POINT_LANDMARKS, 1] = 0.2 + 2 * t
            full[t, POINT_LANDMARKS, 2] = 0.3

        for center_mode in ("notebook_strict", "nose_mean"):
            tensor, mask = first_place_preprocess_array(full, max_len=4, center_mode=center_mode)

            self.assertEqual(tensor.shape, (4, 708))
            self.assertEqual(mask.shape, (4,))
            self.assertEqual(tensor.dtype, np.float32)
            self.assertFalse(np.isnan(tensor).any())
            np.testing.assert_array_equal(mask, np.array([True, True, True, False]))

    def test_default_center_mode_is_notebook_strict(self) -> None:
        self.assertEqual(resolve_center_landmarks(), [17])
        self.assertEqual(CENTER_MODES["nose_mean"], [1, 2, 98, 327])


if __name__ == "__main__":
    unittest.main()
