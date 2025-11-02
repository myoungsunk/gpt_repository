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

    SUPPORTED_MODES = {"none", "kappa", "inv_kappa", "mcdrop"}

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
        if gate_mode not in self.SUPPORTED_MODES:
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
        delta_max: Optional[float] = None,
        gain: float = 1.0,
        gate_threshold: float = 0.5,
        extras: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute residual corrections."""

        if features.numel() == 0:
            raise ValueError("ResidualCalibrator requires non-empty features")
        ramp = float(ramp)
        gain = float(gain)
        effective_max = float(self.delta_max_rad if delta_max is None else delta_max)
        effective_max = max(0.0, min(self.delta_max_rad, effective_max))
        x = self.layernorm(features)
        h = self.act(self.fc1(x))
        h = self.dropout(h)
        h = self.act(self.fc2(h))
        h = self.dropout(h)
        delta_unit = torch.tanh(self.residual_head(h))  # torch.FloatTensor[B,2]
        delta = delta_unit * effective_max
        gate_logits = self.gate_head(h)
        aux: Dict[str, torch.Tensor] = {}
        if self.gate_mode == "none":
            gate_raw = torch.ones_like(gate_logits)
        elif self.gate_mode in {"kappa", "inv_kappa"}:
            sample_kappa = kappa.mean(dim=-1, keepdim=True)
            mean_kappa = sample_kappa.mean(dim=0, keepdim=True)
            std_kappa = sample_kappa.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-6)
            norm_kappa = (sample_kappa - mean_kappa) / std_kappa
            if self.gate_mode == "inv_kappa":
                norm_kappa = -norm_kappa
            aux["norm_kappa"] = norm_kappa.detach()
            scaled = (norm_kappa * self.kappa_scale + self.kappa_bias) / max(self.gate_tau, 1e-6)
            gate_raw = torch.sigmoid(gate_logits + scaled)
        else:  # mcdrop
            unc = None
            if extras is not None:
                unc = extras.get("uncertainty")
            if unc is None:
                aux["mcdrop_fallback"] = torch.ones(1, device=features.device)
                sample_kappa = kappa.mean(dim=-1, keepdim=True)
                mean_kappa = sample_kappa.mean(dim=0, keepdim=True)
                std_kappa = sample_kappa.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-6)
                norm_kappa = (sample_kappa - mean_kappa) / std_kappa
                scaled = (norm_kappa * self.kappa_scale + self.kappa_bias) / max(self.gate_tau, 1e-6)
                gate_raw = torch.sigmoid(gate_logits + scaled)
            else:
                scaled_unc = -unc / max(self.gate_tau, 1e-6)
                gate_raw = torch.sigmoid(gate_logits + scaled_unc)
                aux["uncertainty"] = unc.detach()
        gate = torch.clamp(gate_raw * ramp, min=0.0, max=1.0)
        keep_mask = gate >= gate_threshold
        delta_effect = gate * delta * gain
        if effective_max <= 1e-6 or gain <= 1e-6:
            clip_mask = torch.zeros_like(delta, dtype=torch.bool)
        else:
            clip_mask = keep_mask & (delta.abs() >= (effective_max - 1e-6))
        mu_ref = _wrap_to_pi(mu + delta_effect)
        return {
            "delta_unit": delta_unit,
            "delta": delta,
            "gate": gate,
            "gate_raw": gate_raw,
            "keep_mask": keep_mask,
            "mu_ref": mu_ref,
            "delta_effect": delta_effect,
            "clip_mask": clip_mask,
            "aux": aux,
        }


__all__ = ["ResidualCalibrator"]
