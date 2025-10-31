from __future__ import annotations

import torch

from .units import dbm_to_mw, mw_to_dbm


def combine_r4_to_c(r4_dbm: torch.Tensor) -> torch.Tensor:
    """경로별 RSS(dBm)를 안테나별 합성 전력으로 변환한다."""

    if r4_dbm.ndim != 2 or r4_dbm.size(-1) != 4:
        raise ValueError("Expected input shape [B,4]")
    power_mw = dbm_to_mw(r4_dbm)  # torch.FloatTensor[B,4]
    # 두 경로씩 선형 전력 합(위상/간섭 없음)
    sum_a = power_mw[:, :2].sum(dim=-1, keepdim=True)  # torch.FloatTensor[B,1]
    sum_b = power_mw[:, 2:].sum(dim=-1, keepdim=True)  # torch.FloatTensor[B,1]
    combined = torch.cat([sum_a, sum_b], dim=-1)  # torch.FloatTensor[B,2]
    return mw_to_dbm(combined)  # torch.FloatTensor[B,2]


__all__ = ["combine_r4_to_c"]