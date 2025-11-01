"""von-Mises NLL 및 κ 온도보정."""
from __future__ import annotations

import torch


def _log_modified_bessel0(x: torch.Tensor) -> torch.Tensor:
    """log I0(x) 안정화."""

    return torch.log(torch.i0(x) + torch.finfo(x.dtype).eps)


def von_mises_nll(
    mu_pred: torch.Tensor,  # torch.FloatTensor[B,2]
    kappa_pred: torch.Tensor,  # torch.FloatTensor[B,2]
    theta_gt: torch.Tensor,  # torch.FloatTensor[B,2]
    reduction: str = "mean",
) -> torch.Tensor:
    """von-Mises 음의 로그우도."""

    if mu_pred.shape != kappa_pred.shape or mu_pred.shape != theta_gt.shape:
        raise ValueError("mu, kappa, theta must share shape [B,2]")
    diff = torch.atan2(torch.sin(mu_pred - theta_gt), torch.cos(mu_pred - theta_gt))  # torch.FloatTensor[B,2]
    log_i0 = _log_modified_bessel0(torch.clamp(kappa_pred, min=1e-6))
    two_pi = torch.tensor(2 * torch.pi, device=mu_pred.device, dtype=mu_pred.dtype)
    nll = -kappa_pred * torch.cos(diff) + log_i0 + torch.log(two_pi)
    nll = nll.sum(dim=-1)  # torch.FloatTensor[B]
    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    return nll


def scale_kappa(kappa: torch.Tensor, temperature: float) -> torch.Tensor:
    """κ 온도보정."""

    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return kappa / temperature


__all__ = ["von_mises_nll", "scale_kappa"]
