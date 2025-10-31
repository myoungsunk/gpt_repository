"""DoA Predictor (M2)."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import MLPBlock


class DoAPredictor(nn.Module):
    """von-Mises 기반 DoA 추정기."""

    def __init__(
        self,
        phi_dim: int = 16,
        latent_dim: int = 128,
        z_dim: int = 5,
        hidden_dim: Optional[int] = None,
        dropout_p: float = 0.1,
        mixture_components: int = 1,
        kappa_min: float = 1e-3,
    ) -> None:
        super().__init__()
        self.phi_dim = phi_dim
        self.z_dim = z_dim
        self.in_dim = z_dim + phi_dim
        hidden = hidden_dim or latent_dim
        self.backbone = nn.Sequential(
            MLPBlock(self.in_dim, hidden, dropout_p=dropout_p),
            MLPBlock(hidden, hidden, dropout_p=dropout_p),
        )
        self.mixture_components = mixture_components
        out_dim = 2 * mixture_components
        self.mu_head = nn.Linear(hidden, out_dim)
        self.kappa_head = nn.Linear(hidden, out_dim)
        nn.init.xavier_uniform_(self.mu_head.weight)
        nn.init.xavier_uniform_(self.kappa_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.kappa_head.bias)
        self.kappa_min = kappa_min
        if mixture_components > 1:
            self.logits_head = nn.Linear(hidden, mixture_components)
            nn.init.xavier_uniform_(self.logits_head.weight)
            nn.init.zeros_(self.logits_head.bias)
        else:
            self.logits_head = None

    def forward(
        self,
        z5d: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """전방 계산.

        Args:
            z5d: torch.FloatTensor[B,5]
            phi: torch.FloatTensor[B,P] or None
        Returns:
            mu: torch.FloatTensor[B,2] or [B,K,2]
            kappa: torch.FloatTensor[B,2] or [B,K,2]
            logits: torch.FloatTensor[B,K] or None
        """

        if phi is None:
            phi = z5d.new_zeros((z5d.size(0), self.phi_dim))
        if phi.size(-1) != self.phi_dim:
            raise ValueError(f"phi dimension mismatch: expected {self.phi_dim}, got {phi.size(-1)}")
        x = torch.cat([z5d, phi], dim=-1)
        hidden = self.backbone(x)
        mu_raw = self.mu_head(hidden)  # torch.FloatTensor[B,2*K]
        kappa_raw = self.kappa_head(hidden)  # torch.FloatTensor[B,2*K]
        mu = torch.tanh(mu_raw).view(z5d.size(0), self.mixture_components, 2) * math.pi
        kappa = F.softplus(kappa_raw).view(z5d.size(0), self.mixture_components, 2) + self.kappa_min
        if self.mixture_components == 1:
            mu = mu.squeeze(1)
            kappa = kappa.squeeze(1)
        logits: Optional[torch.Tensor]
        if self.logits_head is not None:
            logits = self.logits_head(hidden)
        else:
            logits = None
        return mu, kappa, logits


__all__ = ["DoAPredictor"]
