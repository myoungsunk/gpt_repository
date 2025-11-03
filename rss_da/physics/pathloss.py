"""경로손실 기반 물리 제약."""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F


def ldpl_expected_rss(
    distance_m: torch.Tensor,  # torch.FloatTensor[B]
    tx_power_dbm: float,
    pathloss_exponent: float,
    reference_distance_m: float = 1.0,
    reference_loss_db: float = 40.0,
) -> torch.Tensor:
    """로그 거리 경로손실 모델 기대 RSS."""

    distance_safe = torch.clamp(distance_m, min=1e-3)
    return tx_power_dbm - (
        reference_loss_db + 10 * pathloss_exponent * torch.log10(distance_safe / reference_distance_m)
    )


@dataclass
class PathlossConstraint:
    tx_power_dbm: float = 0.0
    pathloss_exponent: float = 2.0
    reference_distance_m: float = 1.0
    reference_loss_db: float = 40.0
    slack_db: float = 6.0

    def penalty(self, predicted_dbm: torch.Tensor, distance_m: torch.Tensor) -> torch.Tensor:
        """예측 전력이 물리적 범위를 벗어났는지 페널티."""

        target = ldpl_expected_rss(
            distance_m,
            tx_power_dbm=self.tx_power_dbm,
            pathloss_exponent=self.pathloss_exponent,
            reference_distance_m=self.reference_distance_m,
            reference_loss_db=self.reference_loss_db,
        )
        diff = predicted_dbm - target
        margin = torch.clamp(diff.abs() - self.slack_db, min=0.0)
        return F.smooth_l1_loss(margin, torch.zeros_like(margin))


def friis_gain(
    distance_m: torch.Tensor,  # torch.FloatTensor[B]
    wavelength_m: float,
    tx_gain_db: float,
    rx_gain_db: float,
) -> torch.Tensor:
    """Friis 방정식 기반 수신 전력 dB."""

    distance_safe = torch.clamp(distance_m, min=1e-3)
    fspl = 20 * torch.log10(4 * torch.pi * distance_safe / wavelength_m)
    return tx_gain_db + rx_gain_db - fspl


def range_penalty(
    distance_m: torch.Tensor,  # torch.FloatTensor[B]
    min_distance_m: float,
    max_distance_m: float,
) -> torch.Tensor:
    """허용 거리 범위를 벗어나면 페널티."""

    low = F.relu(min_distance_m - distance_m)
    high = F.relu(distance_m - max_distance_m)
    return (low + high).mean()


__all__ = ["ldpl_expected_rss", "PathlossConstraint", "friis_gain", "range_penalty"]
