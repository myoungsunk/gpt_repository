"""평가지표 유틸."""
from __future__ import annotations

from typing import Dict

import torch


def circular_mean_error_deg(pred_mu: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """원형 각도 오차(deg)."""

    diff = torch.atan2(torch.sin(pred_mu - target), torch.cos(pred_mu - target))
    return torch.rad2deg(diff.abs()).mean()


def ece_placeholder(logits: torch.Tensor, labels: torch.Tensor, num_bins: int = 10) -> Dict[str, torch.Tensor]:
    """ECE 스켈레톤."""

    confidences = torch.softmax(logits, dim=-1).max(dim=-1).values
    predictions = logits.argmax(dim=-1)
    accuracies = predictions.eq(labels)
    bins = torch.linspace(0, 1, steps=num_bins + 1, device=logits.device)
    total = logits.new_tensor(0.0)
    ece = logits.new_tensor(0.0)
    for i in range(num_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        count = mask.sum().float()
        if count == 0:
            continue
        total = total + count
        acc = accuracies[mask].float().mean()
        conf = confidences[mask].mean()
        ece = ece + count * (acc - conf).abs()
    ece = ece / torch.clamp(total, min=1.0)
    return {"ece": ece}


def aurc_placeholder(confidences: torch.Tensor, errors: torch.Tensor) -> torch.Tensor:
    """AURC 스켈레톤."""

    sorted_conf, indices = confidences.sort(descending=True)
    sorted_errors = errors[indices]
    cumulative_error = torch.cumsum(sorted_errors.float(), dim=0)
    coverage = torch.arange(1, confidences.numel() + 1, device=confidences.device)
    risk = cumulative_error / coverage
    aurc = torch.trapz(risk, coverage / coverage[-1])
    return aurc


def dann_accuracy_placeholder(domain_logits: torch.Tensor, domain_labels: torch.Tensor) -> torch.Tensor:
    """DANN domain classifier 정확도 placeholder."""

    preds = domain_logits.argmax(dim=-1)
    return preds.eq(domain_labels).float().mean()


__all__ = [
    "circular_mean_error_deg",
    "ece_placeholder",
    "aurc_placeholder",
    "dann_accuracy_placeholder",
]
