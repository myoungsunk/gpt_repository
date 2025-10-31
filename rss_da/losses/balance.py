"""손실 가중 제어."""
from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn


class GradNormController(nn.Module):
    """GradNorm 기반 가중 조정."""

    def __init__(self, task_names: Iterable[str], alpha: float = 0.5, lr: float = 0.025) -> None:
        super().__init__()
        self.task_names: List[str] = list(task_names)
        if not self.task_names:
            raise ValueError("At least one task required")
        self.alpha = alpha
        self.lr = lr
        self.log_weights = nn.Parameter(torch.zeros(len(self.task_names)))
        self.register_buffer("initial_losses", torch.ones(len(self.task_names)))
        self._initialized = False

    def initialize(self, losses: Dict[str, torch.Tensor]) -> None:
        values = torch.stack([losses[name].detach() for name in self.task_names])
        self.initial_losses.copy_(values)
        self._initialized = True

    def forward(
        self,
        losses: Dict[str, torch.Tensor],
        shared_params: Iterable[torch.nn.Parameter],
    ) -> Dict[str, torch.Tensor]:
        """가중치 사전 반환."""

        if not self._initialized:
            self.initialize(losses)
        params = tuple(shared_params)
        weights = torch.softmax(self.log_weights, dim=0) * len(self.task_names)
        grad_norms = []
        for name, weight in zip(self.task_names, weights):
            loss = losses[name] * weight
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            total = torch.zeros(1, device=loss.device)
            for g in grads:
                if g is not None:
                    total = total + g.norm(p=2)
            grad_norms.append(total)
        grad_norms = torch.stack(grad_norms).view(-1)
        with torch.no_grad():
            loss_vec = torch.stack([losses[name].detach() for name in self.task_names])
            loss_ratio = loss_vec / self.initial_losses
            inverse_rate = loss_ratio / loss_ratio.mean()
            target = grad_norms.mean() * (inverse_rate ** self.alpha)
            diff = grad_norms - target
            self.log_weights.data = self.log_weights.data - self.lr * diff
        weights = torch.softmax(self.log_weights, dim=0)
        return {name: weight for name, weight in zip(self.task_names, weights)}


class UncertaintyWeighting(nn.Module):
    """Kendall 불확실도 가중."""

    def __init__(self, task_names: Iterable[str]) -> None:
        super().__init__()
        self.task_names: List[str] = list(task_names)
        if not self.task_names:
            raise ValueError("At least one task required")
        self.log_sigma = nn.Parameter(torch.zeros(len(self.task_names)))

    def forward(self, losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        weights = torch.exp(-self.log_sigma)
        weighted = {}
        for idx, name in enumerate(self.task_names):
            loss = losses[name]
            weighted[name] = weights[idx] * loss + self.log_sigma[idx]
        weighted["total"] = torch.stack(list(weighted.values())).sum()
        return weighted


__all__ = ["GradNormController", "UncertaintyWeighting"]
