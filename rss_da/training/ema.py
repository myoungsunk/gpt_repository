"""EMA Teacher 업데이트."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


def update_ema(student: nn.Module, teacher: nn.Module, decay: float) -> None:
    """파라미터 EMA 업데이트."""

    if not 0.0 <= decay <= 1.0:
        raise ValueError("decay must be within [0,1]")
    with torch.no_grad():
        for s_param, t_param in zip(student.parameters(), teacher.parameters()):
            t_param.copy_(decay * t_param + (1.0 - decay) * s_param)
        for s_buf, t_buf in zip(student.buffers(), teacher.buffers()):
            t_buf.copy_(decay * t_buf + (1.0 - decay) * s_buf)


def build_ema(model: nn.Module) -> nn.Module:
    """모델 복제 후 EMA 초기화."""

    ema = copy.deepcopy(model)
    for param in ema.parameters():
        param.requires_grad_(False)
    return ema


__all__ = ["update_ema", "build_ema"]
