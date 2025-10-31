"""인코더 및 어댑터 모듈."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .blocks import MLPBlock


class E4(nn.Module):
    """4RSS(dBm) 입력 인코더."""

    def __init__(self, latent_dim: int = 128, hidden_dim: Optional[int] = None, dropout_p: float = 0.1) -> None:
        super().__init__()
        hidden = hidden_dim or latent_dim
        self.net = nn.Sequential(
            MLPBlock(4, hidden, dropout_p=dropout_p),
            MLPBlock(hidden, latent_dim, dropout_p=dropout_p),
        )

    def forward(self, r4_dbm: torch.Tensor) -> torch.Tensor:
        """전방 계산.

        Args:
            r4_dbm: torch.FloatTensor[B,4]
        Returns:
            torch.FloatTensor[B,latent_dim]
        """

        return self.net(r4_dbm)


class E5(nn.Module):
    """z5d 입력 인코더."""

    def __init__(self, latent_dim: int = 128, hidden_dim: Optional[int] = None, dropout_p: float = 0.1) -> None:
        super().__init__()
        hidden = hidden_dim or latent_dim
        self.net = nn.Sequential(
            MLPBlock(5, hidden, dropout_p=dropout_p),
            MLPBlock(hidden, latent_dim, dropout_p=dropout_p),
        )

    def forward(self, z5d: torch.Tensor) -> torch.Tensor:
        """전방 계산.

        Args:
            z5d: torch.FloatTensor[B,5]
        Returns:
            torch.FloatTensor[B,latent_dim]
        """

        return self.net(z5d)


class Fuse(nn.Module):
    """HeMIS/ModDrop 스타일 모달리티 융합."""

    def __init__(self, latent_dim: int = 128, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout = nn.Dropout(dropout_p)
        self.norm = nn.LayerNorm(latent_dim)
        self.residual = MLPBlock(latent_dim, latent_dim, dropout_p=dropout_p)

    def forward(
        self,
        h5: torch.Tensor,
        h4_hat: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """전방 계산.

        Args:
            h5: torch.FloatTensor[B,H]
            h4_hat: torch.FloatTensor[B,H]
            mask: torch.FloatTensor[B,1], 1이면 신뢰도 높음
        Returns:
            torch.FloatTensor[B,H]
        """

        if mask.ndim == 1:
            mask = mask.unsqueeze(-1)
        mask = mask.clamp(0.0, 1.0)
        ones = torch.ones_like(mask)
        weights = torch.cat([ones, mask], dim=-1)  # torch.FloatTensor[B,2]
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        h_stack = torch.stack([h5, h4_hat], dim=1)  # torch.FloatTensor[B,2,H]
        fused = (h_stack * weights.unsqueeze(-1)).sum(dim=1)  # torch.FloatTensor[B,H]
        fused = self.norm(fused)
        fused = fused + self.dropout(self.residual(fused))
        return fused


class Adapter(nn.Module):
    """Latent h를 phi 공간으로 사상."""

    def __init__(self, latent_dim: int = 128, phi_dim: int = 16, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            MLPBlock(latent_dim, latent_dim, dropout_p=dropout_p),
            nn.Linear(latent_dim, phi_dim),
        )
        nn.init.xavier_uniform_(self.net[-1].weight)
        if self.net[-1].bias is not None:
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """전방 계산.

        Args:
            h: torch.FloatTensor[B,H]
        Returns:
            torch.FloatTensor[B,P]
        """

        return self.net(h)


__all__ = ["E4", "E5", "Fuse", "Adapter"]
