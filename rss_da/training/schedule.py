"""스케줄 유틸."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class ScheduledSamplingConfig:
    start: float = 1.0
    end: float = 0.0
    warmup_steps: int = 0
    total_steps: int = 1000


class ScheduledSampling:
    """phi 경로 GT 사용 비율 스케줄."""

    def __init__(self, config: ScheduledSamplingConfig) -> None:
        self.config = config

    def __call__(self, step: int) -> float:
        cfg = self.config
        step = max(step, 0)
        if step < cfg.warmup_steps:
            return cfg.start
        progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps))
        return cfg.start + (cfg.end - cfg.start) * progress


class WarmupDecayScheduler:
    """학습률/가중 warmup+anneal."""

    def __init__(self, warmup_steps: int, total_steps: int, scheduler: Callable[[float], float]) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.scheduler = scheduler

    def scale(self, step: int) -> float:
        if step < self.warmup_steps:
            return (step + 1) / max(1, self.warmup_steps)
        t = min(1.0, (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps))
        return self.scheduler(t)


def cosine_anneal(t: float, min_scale: float = 0.0) -> float:
    """코사인 anneal."""

    return min_scale + 0.5 * (1 - min_scale) * (1 + torch.cos(torch.tensor(t * torch.pi))).item()


__all__ = ["ScheduledSamplingConfig", "ScheduledSampling", "WarmupDecayScheduler", "cosine_anneal"]
