"""4RSS를 안테나별 합성 전력으로 결합하는 함수."""
from __future__ import annotations

import torch

from .units import dbm_to_mw, mw_to_dbm


def _rel_db_to_linear(x_db: torch.Tensor) -> torch.Tensor:
    return 10 ** (x_db / 10)


def _linear_to_rel_db(x_lin: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return 10 * torch.log10(torch.clamp(x_lin, min=eps))


def combine_r4_to_c(r4_dbm: torch.Tensor) -> torch.Tensor:
    """경로별 RSS(dBm)를 안테나별 합성 전력으로 변환한다."""

    if r4_dbm.ndim != 2 or r4_dbm.size(-1) != 4:
        raise ValueError("Expected input shape [B,4]")
    power_mw = dbm_to_mw(r4_dbm)  # torch.FloatTensor[B,4]
    sum_a = power_mw[:, :2].sum(dim=-1, keepdim=True)  # torch.FloatTensor[B,1]
    sum_b = power_mw[:, 2:].sum(dim=-1, keepdim=True)  # torch.FloatTensor[B,1]
    combined = torch.cat([sum_a, sum_b], dim=-1)  # torch.FloatTensor[B,2]
    return mw_to_dbm(combined)  # torch.FloatTensor[B,2]


def combine_r4_rel_to_c_rel(r4_rel_db: torch.Tensor) -> torch.Tensor:
    """상대 dB 스케일 4RSS를 상대 dB 합성 전력으로 변환."""

    if r4_rel_db.ndim != 2 or r4_rel_db.size(-1) != 4:
        raise ValueError("Expected input shape [B,4]")
    power_lin = _rel_db_to_linear(r4_rel_db)  # torch.FloatTensor[B,4]
    sum_a = power_lin[:, :2].sum(dim=-1, keepdim=True)
    sum_b = power_lin[:, 2:].sum(dim=-1, keepdim=True)
    combined = torch.cat([sum_a, sum_b], dim=-1)
    return _linear_to_rel_db(combined)


__all__ = ["combine_r4_to_c", "combine_r4_rel_to_c_rel"]
