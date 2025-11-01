"""정렬 손실 모듈."""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def deep_coral_loss(h_s: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
    """Deep-CORAL 2차 통계 정렬."""

    if h_s.shape != h_t.shape:
        raise ValueError("Source/Target features must share shape [B,H]")
    bsz = h_s.size(0)
    mean_s = h_s.mean(dim=0, keepdim=True)
    mean_t = h_t.mean(dim=0, keepdim=True)
    c_s = (h_s - mean_s).t() @ (h_s - mean_s) / (bsz - 1)
    c_t = (h_t - mean_t).t() @ (h_t - mean_t) / (bsz - 1)
    loss = F.mse_loss(c_s, c_t)
    return loss


class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """nn.Module wrapper."""

    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(inputs, self.lambda_)


class DomainAdversarialLoss(nn.Module):
    """DANN 손실 래퍼."""

    def __init__(self, domain_classifier: nn.Module, lambda_: float = 1.0) -> None:
        super().__init__()
        self.domain_classifier = domain_classifier
        self.grl = GradientReversalLayer(lambda_)

    def forward(self, features: torch.Tensor, domain_labels: torch.Tensor) -> torch.Tensor:
        logits = self.domain_classifier(self.grl(features))  # torch.FloatTensor[B,num_domains]
        return F.cross_entropy(logits, domain_labels.long())


def cdan_loss(
    domain_classifier: nn.Module,
    features: torch.Tensor,  # torch.FloatTensor[B,H]
    class_logits: torch.Tensor,  # torch.FloatTensor[B,C]
    domain_labels: torch.Tensor,  # torch.LongTensor[B]
    lambda_: float = 1.0,
    entropy_weight: Optional[torch.Tensor] = None,  # torch.FloatTensor[B]
) -> torch.Tensor:
    """CDAN 조건부 적대 정렬."""

    probs = torch.softmax(class_logits, dim=-1)  # torch.FloatTensor[B,C]
    outer = torch.bmm(probs.unsqueeze(2), features.unsqueeze(1))  # torch.FloatTensor[B,C,H]
    joint = outer.view(features.size(0), -1)
    joint = GradientReversalFunction.apply(joint, lambda_)
    logits = domain_classifier(joint)
    loss = F.cross_entropy(logits, domain_labels.long(), reduction="none")
    if entropy_weight is not None:
        weights = entropy_weight.view(-1)
        loss = loss * weights
        loss = loss.sum() / torch.clamp(weights.sum(), min=1.0)
    else:
        loss = loss.mean()
    return loss


def apply_alignment(
    method: str,
    domain_classifier: Optional[nn.Module],
    features_s: torch.Tensor,
    features_t: torch.Tensor,
    domain_labels: Optional[torch.Tensor] = None,
    class_logits: Optional[torch.Tensor] = None,
    lambda_: float = 1.0,
    entropy_weight: Optional[torch.Tensor] = None,
    callback: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """정렬 손실 dispatcher."""

    method = (method or "").lower()
    if method == "coral":
        return deep_coral_loss(features_s, features_t)
    if method == "dann":
        if domain_classifier is None or domain_labels is None:
            raise ValueError("DANN requires domain classifier and labels")
        loss = DomainAdversarialLoss(domain_classifier, lambda_)(features_s, domain_labels)
        if callback is not None:
            callback(loss)
        return loss
    if method == "cdan":
        if domain_classifier is None or domain_labels is None or class_logits is None:
            raise ValueError("CDAN requires classifier, labels, logits")
        loss = cdan_loss(domain_classifier, features_s, class_logits, domain_labels, lambda_=lambda_, entropy_weight=entropy_weight)
        if callback is not None:
            callback(loss)
        return loss
    return torch.zeros(1, device=features_s.device)


__all__ = [
    "deep_coral_loss",
    "GradientReversalLayer",
    "DomainAdversarialLoss",
    "cdan_loss",
    "apply_alignment",
]
