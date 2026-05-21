from __future__ import annotations

from typing import Iterable

import numpy as np


ROWS_PER_FRAME = 543
NOSE = [1, 2, 98, 327]
NOTEBOOK_CENTER = [17]
LIP = [
    0,
    61,
    185,
    40,
    39,
    37,
    267,
    269,
    270,
    409,
    291,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    78,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
    95,
    88,
    178,
    87,
    14,
    317,
    402,
    318,
    324,
    308,
]
REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
LHAND = list(range(468, 489))
RHAND = list(range(522, 543))
POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE
NUM_NODES = len(POINT_LANDMARKS)
FEATURE_DIM = 6 * NUM_NODES


def nan_mean(array: np.ndarray, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> np.ndarray:
    valid = ~np.isnan(array)
    count = valid.sum(axis=axis, keepdims=keepdims)
    total = np.where(valid, array, 0.0).sum(axis=axis, keepdims=keepdims)
    with np.errstate(divide="ignore", invalid="ignore"):
        return total / count


def nan_std(
    array: np.ndarray,
    center: np.ndarray,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> np.ndarray:
    diff = array - center
    return np.sqrt(nan_mean(diff * diff, axis=axis, keepdims=keepdims))


def filter_empty_frames(full: np.ndarray, point_landmarks: Iterable[int] = POINT_LANDMARKS) -> np.ndarray:
    selected = full[:, list(point_landmarks), :]
    keep = ~np.all(np.isnan(selected), axis=(1, 2))
    return full[keep]


def first_place_preprocess_array(full: np.ndarray, max_len: int = 64) -> tuple[np.ndarray, np.ndarray]:
    if full.ndim != 3 or full.shape[1:] != (ROWS_PER_FRAME, 3):
        raise ValueError(f"Expected full landmark sequence shape [T, 543, 3], got {full.shape}")
    if max_len <= 0:
        raise ValueError(f"max_len must be positive, got {max_len}")

    x_full = filter_empty_frames(full.astype(np.float32, copy=False), POINT_LANDMARKS)
    selected = x_full[:, POINT_LANDMARKS, :]

    center = nan_mean(x_full[:, NOTEBOOK_CENTER, :], axis=(0, 1), keepdims=True)
    center = np.where(np.isnan(center), np.array(0.5, dtype=np.float32), center).astype(np.float32)
    std = nan_std(selected, center=center, axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.where((np.isnan(std)) | (std == 0), np.array(1.0, dtype=np.float32), std)

    selected = (selected - center) / std
    selected = selected[:max_len, :, :2]
    length = selected.shape[0]

    dx = np.zeros_like(selected, dtype=np.float32)
    if length > 1:
        dx[:-1] = selected[1:] - selected[:-1]

    dx2 = np.zeros_like(selected, dtype=np.float32)
    if length > 2:
        dx2[:-2] = selected[2:] - selected[:-2]

    features = np.concatenate(
        [
            selected.reshape(length, 2 * NUM_NODES),
            dx.reshape(length, 2 * NUM_NODES),
            dx2.reshape(length, 2 * NUM_NODES),
        ],
        axis=-1,
    ).astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    x = np.zeros((max_len, FEATURE_DIM), dtype=np.float32)
    mask = np.zeros(max_len, dtype=bool)
    x[:length] = features
    mask[:length] = True
    return x, mask
