"""Residual Calibrator (선택 모듈)."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .blocks import MLPBlock


class ResidualCalibrator(nn.Module):
    """DoA 예측을 보정하는 잔차 모듈."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            MLPBlock(feature_dim + 2, hidden_dim, dropout_p=dropout_p),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.xavier_uniform_(self.mlp[-1].weight)
        if self.mlp[-1].bias is not None:
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        mu: torch.Tensor,
        kappa: torch.Tensor,
        features: torch.Tensor,
        uncertainty_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """전방 계산.

        Args:
            mu: torch.FloatTensor[B,2]
            kappa: torch.FloatTensor[B,2]
            features: torch.FloatTensor[B,F]
            uncertainty_scale: torch.FloatTensor[B,1] or None
        Returns:
            torch.FloatTensor[B,2]
        """

        x = torch.cat([features, mu], dim=-1)
        residual = self.mlp(x)  # torch.FloatTensor[B,2]
        gate = torch.sigmoid(kappa.mean(dim=-1, keepdim=True))  # torch.FloatTensor[B,1]
        if uncertainty_scale is not None:
            if uncertainty_scale.ndim == 1:
                uncertainty_scale = uncertainty_scale.unsqueeze(-1)
            gate = gate * uncertainty_scale
        residual = residual * gate
        return residual


__all__ = ["ResidualCalibrator"]
