"""Residual calibrator with confidence gating."""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn


def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    """Wrap angles to (−π, π]."""

    return (x + math.pi) % (2 * math.pi) - math.pi


class ResidualCalibrator(nn.Module):
    """Predict residual angle corrections and confidence gates."""

    def __init__(
        self,
        in_dim: int,
        hidden: int = 128,
        dropout_p: float = 0.1,
        delta_max_rad: float = math.radians(10.0),
        gate_mode: str = "kappa",
        gate_tau: float = 1.0,
    ) -> None:
        super().__init__()
        if gate_mode not in {"none", "kappa", "mcdrop"}:
            raise ValueError(f"Unsupported gate mode: {gate_mode}")
        self.delta_max_rad = float(delta_max_rad)
        self.gate_mode = gate_mode
        self.gate_tau = float(gate_tau)
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.residual_head = nn.Linear(hidden, 2)
        self.gate_head = nn.Linear(hidden, 1)
        self.kappa_scale = nn.Parameter(torch.tensor(1.0))
        self.kappa_bias = nn.Parameter(torch.tensor(0.0))
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.gate_head.bias)

    def forward(
        self,
        features: torch.Tensor,  # torch.FloatTensor[B,D]
        mu: torch.Tensor,  # torch.FloatTensor[B,2]
        kappa: torch.Tensor,  # torch.FloatTensor[B,2]
        ramp: float = 1.0,
        extras: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute residual corrections."""

        if features.numel() == 0:
            raise ValueError("ResidualCalibrator requires non-empty features")
        ramp = float(ramp)
        x = self.layernorm(features)
        h = self.act(self.fc1(x))
        h = self.dropout(h)
        h = self.act(self.fc2(h))
        h = self.dropout(h)
        delta_raw = self.residual_head(h)
        delta = torch.clamp(delta_raw, min=-self.delta_max_rad, max=self.delta_max_rad)
        clip_mask = (delta_raw.abs() >= self.delta_max_rad - 1e-6)
        gate_logits = self.gate_head(h)
        aux: Dict[str, torch.Tensor] = {}
        if self.gate_mode == "none":
            gate = torch.ones_like(gate_logits)
        elif self.gate_mode == "kappa":
            norm_kappa = kappa.mean(dim=-1, keepdim=True)
            scaled = (norm_kappa * self.kappa_scale + self.kappa_bias) / max(self.gate_tau, 1e-6)
            gate = torch.sigmoid(gate_logits + scaled)
            aux["norm_kappa"] = norm_kappa.detach()
        elif self.gate_mode == "mcdrop":
            unc = None
            if extras is not None:
                unc = extras.get("uncertainty")
            if unc is None:
                aux["mcdrop_fallback"] = torch.ones(1, device=features.device)
                norm_kappa = kappa.mean(dim=-1, keepdim=True)
                scaled = (norm_kappa * self.kappa_scale + self.kappa_bias) / max(self.gate_tau, 1e-6)
                gate = torch.sigmoid(gate_logits + scaled)
            else:
                scaled_unc = -unc / max(self.gate_tau, 1e-6)
                gate = torch.sigmoid(gate_logits + scaled_unc)
                aux["uncertainty"] = unc.detach()
        gate = torch.clamp(gate * ramp, min=0.0, max=1.0)
        delta_effect = gate * delta
        mu_ref = _wrap_to_pi(mu + delta_effect)
        return {
            "delta_raw": delta_raw,
            "delta": delta,
            "gate": gate,
            "mu_ref": mu_ref,
            "delta_effect": delta_effect,
            "clip_mask": clip_mask,
            "aux": aux,
        }


__all__ = ["ResidualCalibrator"]
