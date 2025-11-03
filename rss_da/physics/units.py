"""단위 변환 유틸."""
from __future__ import annotations

import torch


def dbm_to_mw(x_dbm: torch.Tensor) -> torch.Tensor:
    """dBm 값을 mW로 변환한다."""

    return 10 ** (x_dbm / 10)


def mw_to_dbm(x_mw: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """mW 값을 dBm으로 변환한다."""

    x_safe = torch.clamp(x_mw, min=eps)
    return 10 * torch.log10(x_safe)


__all__ = ["dbm_to_mw", "mw_to_dbm"]
