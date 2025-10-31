"""Decoder D: (h, mu) -> 4RSS_hat."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..physics.units import mw_to_dbm
from .blocks import MLPBlock


class DecoderD(nn.Module):
    """Latent 융합 표현과 추정 각도를 이용해 4RSS를 복원."""

    def __init__(
        self,
        latent_dim: int = 128,
        angle_dim: int = 2,
        hidden_dim: Optional[int] = None,
        dropout_p: float = 0.1,
        min_power_mw: float = 1e-6,
    ) -> None:
        super().__init__()
        hidden = hidden_dim or latent_dim
        in_dim = latent_dim + angle_dim
        self.trunk = nn.Sequential(
            MLPBlock(in_dim, hidden, dropout_p=dropout_p),
            MLPBlock(hidden, hidden, dropout_p=dropout_p),
        )
        self.head = nn.Linear(hidden, 4)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)
        self.min_power_mw = float(min_power_mw)

    def forward(self, h: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """전방 계산.

        Args:
            h: torch.FloatTensor[B,H]
            mu: torch.FloatTensor[B,2]
        Returns:
            torch.FloatTensor[B,4] dBm
        """

        x = torch.cat([h, mu], dim=-1)
        latent = self.trunk(x)
        power_raw = self.head(latent)  # torch.FloatTensor[B,4]
        power_mw = F.softplus(power_raw) + self.min_power_mw
        r4_hat_dbm = mw_to_dbm(power_mw)
        return r4_hat_dbm


__all__ = ["DecoderD"]
