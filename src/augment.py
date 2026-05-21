from __future__ import annotations

from typing import Any

import torch


def _prob(config: dict[str, Any], key: str, default: float = 0.0) -> float:
    return float(config.get(key, default) or 0.0)


def _int(config: dict[str, Any], key: str, default: int = 0) -> int:
    return int(config.get(key, default) or 0)


def temporal_shift(
    x: torch.Tensor,
    mask: torch.Tensor,
    prob: float,
    max_shift: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prob <= 0.0 or max_shift <= 0:
        return x, mask

    batch_size, seq_len, _ = x.shape
    for i in range(batch_size):
        x[i, ~mask[i]] = 0.0
        if torch.rand((), device=x.device) >= prob:
            continue
        shift = int(torch.randint(-max_shift, max_shift + 1, (), device=x.device).item())
        if shift == 0:
            continue

        shifted_x = torch.zeros_like(x[i])
        shifted_mask = torch.zeros_like(mask[i])
        if shift > 0:
            shifted_x[shift:] = x[i, : seq_len - shift]
            shifted_mask[shift:] = mask[i, : seq_len - shift]
        else:
            offset = -shift
            shifted_x[: seq_len - offset] = x[i, offset:]
            shifted_mask[: seq_len - offset] = mask[i, offset:]
        shifted_x[~shifted_mask] = 0.0
        x[i] = shifted_x
        mask[i] = shifted_mask
    return x, mask


def temporal_mask(
    x: torch.Tensor,
    mask: torch.Tensor,
    prob: float,
    max_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prob <= 0.0 or max_width <= 0:
        return x, mask

    batch_size = x.shape[0]
    for i in range(batch_size):
        valid_len = int(mask[i].sum().item())
        if valid_len <= 0 or torch.rand((), device=x.device) >= prob:
            continue
        width = int(torch.randint(1, min(max_width, valid_len) + 1, (), device=x.device).item())
        start = int(torch.randint(0, valid_len - width + 1, (), device=x.device).item())
        x[i, start : start + width] = 0.0
    return x, mask


def feature_dropout(
    x: torch.Tensor,
    mask: torch.Tensor,
    prob: float,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prob <= 0.0 or width <= 0:
        return x, mask

    batch_size, _, feature_dim = x.shape
    width = min(width, feature_dim)
    for i in range(batch_size):
        if torch.rand((), device=x.device) >= prob:
            continue
        start = int(torch.randint(0, feature_dim - width + 1, (), device=x.device).item())
        x[i, :, start : start + width] = 0.0
    return x, mask


def gaussian_noise(
    x: torch.Tensor,
    mask: torch.Tensor,
    prob: float,
    std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prob <= 0.0 or std <= 0.0:
        return x, mask

    batch_size = x.shape[0]
    for i in range(batch_size):
        if torch.rand((), device=x.device) >= prob:
            continue
        noise = torch.randn_like(x[i]) * std
        valid = mask[i].unsqueeze(-1) & (x[i] != 0.0)
        x[i] = x[i] + noise * valid.to(dtype=x.dtype)
    return x, mask


def apply_augmentation(
    x: torch.Tensor,
    mask: torch.Tensor,
    config: dict[str, Any] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return x, mask

    x_aug = x.clone()
    mask_aug = mask.clone()
    x_aug[~mask_aug] = 0.0
    x_aug, mask_aug = temporal_shift(
        x_aug,
        mask_aug,
        prob=_prob(config, "temporal_shift_prob"),
        max_shift=_int(config, "temporal_shift_max"),
    )
    x_aug, mask_aug = temporal_mask(
        x_aug,
        mask_aug,
        prob=_prob(config, "temporal_mask_prob"),
        max_width=_int(config, "temporal_mask_max_width"),
    )
    x_aug, mask_aug = feature_dropout(
        x_aug,
        mask_aug,
        prob=_prob(config, "feature_dropout_prob"),
        width=_int(config, "feature_dropout_width", 32),
    )
    x_aug, mask_aug = gaussian_noise(
        x_aug,
        mask_aug,
        prob=_prob(config, "gaussian_noise_prob"),
        std=_prob(config, "gaussian_noise_std"),
    )
    return x_aug, mask_aug
