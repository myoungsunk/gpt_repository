"""안테나 이득 모델."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


def cos_power_gain(theta_rad: torch.Tensor, m: float) -> torch.Tensor:
    """cos^m 형태의 파라메트릭 이득."""

    return (torch.cos(theta_rad).clamp(min=0.0)) ** m


@dataclass
class AntennaGain:
    lut_theta_rad: Optional[torch.Tensor] = None  # torch.FloatTensor[N]
    lut_gain_db: Optional[torch.Tensor] = None  # torch.FloatTensor[N]
    fallback_m: float = 2.0
    fallback_gain_db: float = 0.0

    def lookup(self, theta_rad: torch.Tensor) -> torch.Tensor:
        """LUT 기반 이득 또는 파라메트릭 폴백."""

        if self.lut_theta_rad is not None and self.lut_gain_db is not None:
            theta_lut = self.lut_theta_rad.to(theta_rad.device)
            gain_lut = self.lut_gain_db.to(theta_rad.device)
            idx = torch.bucketize(theta_rad, theta_lut.clamp(min=theta_lut[0], max=theta_lut[-1]))
            idx0 = torch.clamp(idx - 1, min=0, max=theta_lut.numel() - 1)
            idx1 = torch.clamp(idx, min=0, max=theta_lut.numel() - 1)
            theta0 = theta_lut[idx0]
            theta1 = theta_lut[idx1]
            gain0 = gain_lut[idx0]
            gain1 = gain_lut[idx1]
            denom = torch.clamp(theta1 - theta0, min=1e-6)
            t = (theta_rad - theta0) / denom
            return gain0 + t * (gain1 - gain0)
        gain = cos_power_gain(theta_rad, self.fallback_m)
        gain_db = 10 * torch.log10(torch.clamp(gain, min=1e-6))
        return gain_db + self.fallback_gain_db

    def consistency_penalty(self, gain_a_db: torch.Tensor, gain_b_db: torch.Tensor) -> torch.Tensor:
        """두 안테나 간 상대 이득 일관성 페널티."""

        diff = gain_a_db - gain_b_db
        return F.mse_loss(diff, torch.zeros_like(diff))


__all__ = ["AntennaGain", "cos_power_gain"]
