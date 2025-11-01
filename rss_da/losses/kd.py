"""지식 증류 손실."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .vm_nll import scale_kappa


def _a_function(kappa: torch.Tensor) -> torch.Tensor:
    """A(kappa)=I1/I0."""

    kappa = torch.clamp(kappa, min=1e-6)
    return torch.i1(kappa) / (torch.i0(kappa) + 1e-8)


def output_kd_loss(
    mu_student: torch.Tensor,  # torch.FloatTensor[B,2]
    kappa_student: torch.Tensor,  # torch.FloatTensor[B,2]
    mu_teacher: torch.Tensor,  # torch.FloatTensor[B,2]
    kappa_teacher: torch.Tensor,  # torch.FloatTensor[B,2]
    temperature: float = 1.0,
    gating: Optional[torch.Tensor] = None,  # torch.FloatTensor[B,1] or [B]
    reduction: str = "mean",
) -> torch.Tensor:
    """von-Mises 출력 증류(KL)."""

    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if not (mu_student.shape == kappa_student.shape == mu_teacher.shape == kappa_teacher.shape):
        raise ValueError("mu/kappa tensors must share shape [B,2]")
    tau = float(temperature)
    kappa_s = scale_kappa(kappa_student, tau)  # torch.FloatTensor[B,2]
    kappa_t = scale_kappa(kappa_teacher, tau)
    a_t = _a_function(kappa_t)
    cos_delta = torch.cos(mu_student - mu_teacher)
    two_pi = torch.tensor(2 * torch.pi, device=mu_student.device, dtype=mu_student.dtype)
    ce = -kappa_s * a_t * cos_delta + torch.log(two_pi)
    ce = ce + torch.log(torch.i0(kappa_s) + torch.finfo(kappa_s.dtype).eps)
    entropy_t = torch.log(two_pi) + torch.log(torch.i0(kappa_t) + torch.finfo(kappa_t.dtype).eps)
    entropy_t = entropy_t - kappa_t * a_t
    kl = (ce - entropy_t).sum(dim=-1)  # torch.FloatTensor[B]
    if gating is not None:
        gate = gating.view(-1)
        kl = kl * gate
    if reduction == "mean":
        return kl.mean()
    if reduction == "sum":
        return kl.sum()
    return kl


def feature_kd_loss(
    h_student: torch.Tensor,  # torch.FloatTensor[B,H]
    h_teacher: torch.Tensor,  # torch.FloatTensor[B,H]
    attn_map_student: Optional[torch.Tensor] = None,  # torch.FloatTensor[B,C,H',W'] optional
    attn_map_teacher: Optional[torch.Tensor] = None,  # torch.FloatTensor[B,C,H',W'] optional
    gating: Optional[torch.Tensor] = None,  # torch.FloatTensor[B]
    reduction: str = "mean",
) -> torch.Tensor:
    """표현 증류(L2 + AT)."""

    if h_student.shape != h_teacher.shape:
        raise ValueError("Feature shapes must match")
    mse = F.mse_loss(h_student, h_teacher, reduction="none").mean(dim=-1)  # torch.FloatTensor[B]
    loss = mse
    if attn_map_student is not None and attn_map_teacher is not None:
        attn_s = F.normalize(attn_map_student.flatten(1), dim=-1)
        attn_t = F.normalize(attn_map_teacher.flatten(1), dim=-1)
        at_loss = F.mse_loss(attn_s, attn_t, reduction="none").mean(dim=-1)
        loss = loss + at_loss
    if gating is not None:
        loss = loss * gating.view(-1)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def kd_loss_bundle(
    outputs: Dict[str, torch.Tensor],
    temperature: float,
    gating: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """출력/표현 KD 손실 묶음."""

    mu_s = outputs["mu_student"]
    kappa_s = outputs["kappa_student"]
    mu_t = outputs["mu_teacher"]
    kappa_t = outputs["kappa_teacher"]
    kl = output_kd_loss(mu_s, kappa_s, mu_t, kappa_t, temperature=temperature, gating=gating)
    feat_loss = feature_kd_loss(outputs["h_student"], outputs["h_teacher"], gating=gating)
    return {"kd_out": kl, "kd_feat": feat_loss, "kd_total": kl + feat_loss}


__all__ = ["output_kd_loss", "feature_kd_loss", "kd_loss_bundle"]
