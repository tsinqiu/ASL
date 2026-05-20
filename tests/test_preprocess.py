from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.preprocess import load_landmark_tensor


class LoadLandmarkTensorTest(unittest.TestCase):
    def test_filters_types_pads_and_returns_float32(self) -> None:
        parquet_path = Path("outputs") / "_test_preprocess_sample.parquet"
        parquet_path.parent.mkdir(exist_ok=True)
        df = pd.DataFrame(
            [
                {"frame": 2, "type": "left_hand", "landmark_index": 0, "x": 1.0, "y": 2.0, "z": 3.0},
                {"frame": 2, "type": "left_hand", "landmark_index": 1, "x": np.nan, "y": 5.0, "z": 6.0},
                {"frame": 5, "type": "left_hand", "landmark_index": 0, "x": 7.0, "y": 8.0, "z": 9.0},
                {"frame": 5, "type": "right_hand", "landmark_index": 0, "x": 10.0, "y": 11.0, "z": 12.0},
                {"frame": 5, "type": "face", "landmark_index": 0, "x": 100.0, "y": 100.0, "z": 100.0},
            ]
        )
        df.to_parquet(parquet_path)

        tensor = load_landmark_tensor(
            parquet_path,
            max_frames=3,
            include_types=("left_hand", "right_hand"),
            fillna_value=-1.0,
        )

        self.assertEqual(tensor.shape, (3, 9))
        self.assertEqual(tensor.dtype, np.float32)
        np.testing.assert_allclose(tensor[0], [1.0, 2.0, 3.0, -1.0, 5.0, 6.0, -1.0, -1.0, -1.0])
        np.testing.assert_allclose(tensor[1], [7.0, 8.0, 9.0, -1.0, -1.0, -1.0, 10.0, 11.0, 12.0])
        np.testing.assert_allclose(tensor[2], [0.0] * 9)


if __name__ == "__main__":
    unittest.main()
