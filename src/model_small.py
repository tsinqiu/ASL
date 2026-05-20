from __future__ import annotations

import torch
from torch import nn


class Conv1DBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.bn(x)
        x = x.transpose(1, 2)
        x = self.act(x)
        x = self.dropout(x)
        return x + residual


class SmallISLRModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 708,
        num_classes: int = 250,
        d_model: int = 192,
        nhead: int = 4,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.d_model = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.conv_stage1 = nn.Sequential(
            Conv1DBlock(d_model, dropout=dropout),
            Conv1DBlock(d_model, dropout=dropout),
            Conv1DBlock(d_model, dropout=dropout),
        )
        self.encoder1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=1,
        )
        self.conv_stage2 = nn.Sequential(
            Conv1DBlock(d_model, dropout=dropout),
            Conv1DBlock(d_model, dropout=dropout),
            Conv1DBlock(d_model, dropout=dropout),
        )
        self.encoder2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=1,
        )
        self.pre_pool = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(d_model * 2, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x shape [B, T, C], got {tuple(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.shape[-1]}")

        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        else:
            mask = mask.to(device=x.device, dtype=torch.bool)
            if mask.shape != x.shape[:2]:
                raise ValueError(f"Expected mask shape {tuple(x.shape[:2])}, got {tuple(mask.shape)}")

        x = self.input_proj(x)
        x = self.conv_stage1(x)
        x = self.encoder1(x, src_key_padding_mask=~mask)
        x = self.conv_stage2(x)
        x = self.encoder2(x, src_key_padding_mask=~mask)
        x = self.pre_pool(x)

        mask_f = mask.unsqueeze(-1).to(dtype=x.dtype)
        lengths = mask_f.sum(dim=1).clamp_min(1.0)
        pooled = (x * mask_f).sum(dim=1) / lengths
        return self.head(pooled)
