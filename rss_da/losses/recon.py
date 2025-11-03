"""재구성 및 전방 일관성 손실."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from rss_da.physics.combine import combine_r4_rel_to_c_rel, combine_r4_to_c
from rss_da.physics.pathloss import PathlossConstraint


def _safe_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """shape 일치 확인 후 MSE."""

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")
    return F.mse_loss(pred, target)


def recon_loss(
    r4_hat_dbm: torch.Tensor,  # torch.FloatTensor[B,4]
    r4_gt_dbm: Optional[torch.Tensor] = None,  # torch.FloatTensor[B,4]
    c_meas_dbm: Optional[torch.Tensor] = None,  # torch.FloatTensor[B,2]
    c_meas_rel_db: Optional[torch.Tensor] = None,  # torch.FloatTensor[B,2]
    input_scale: str = "relative_db",
    physics_ctx: Optional[Dict[str, object]] = None,
    mix_variance_floor: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """전방 일관성 및 물리 제약 손실."""

    if r4_hat_dbm.ndim != 2 or r4_hat_dbm.size(-1) != 4:
        raise ValueError("r4_hat_dbm must have shape [B,4]")
    losses: Dict[str, torch.Tensor] = {}

    scale = input_scale.lower()
    if scale == "relative_db":
        if c_meas_rel_db is not None and c_meas_rel_db.numel() > 0:
            c_hat_rel = combine_r4_rel_to_c_rel(r4_hat_dbm)
            mix_raw = _safe_mse(c_hat_rel, c_meas_rel_db)
            variance = torch.var(c_meas_rel_db.detach().view(-1), unbiased=False)
            variance = torch.clamp(variance, min=mix_variance_floor)
            losses["mix_raw"] = mix_raw
            losses["mix_var"] = variance
            losses["mix"] = mix_raw / variance
        else:
            zero = r4_hat_dbm.new_zeros(1)
            losses["mix_raw"] = zero
            losses["mix_var"] = r4_hat_dbm.new_ones(1)
            losses["mix"] = zero
    else:
        if c_meas_dbm is not None:
            c_hat = combine_r4_to_c(r4_hat_dbm)  # torch.FloatTensor[B,2]
            mix_raw = _safe_mse(c_hat, c_meas_dbm)
            losses["mix_raw"] = mix_raw
            losses["mix_var"] = r4_hat_dbm.new_ones(1)
            losses["mix"] = mix_raw
        else:
            zero = r4_hat_dbm.new_zeros(1)
            losses["mix_raw"] = zero
            losses["mix_var"] = r4_hat_dbm.new_ones(1)
            losses["mix"] = zero

    if r4_gt_dbm is not None and r4_gt_dbm.numel() > 0:
        losses["data"] = _safe_mse(r4_hat_dbm, r4_gt_dbm)
    else:
        losses["data"] = torch.zeros(1, device=r4_hat_dbm.device)

    phys_loss = torch.zeros(1, device=r4_hat_dbm.device)
    if physics_ctx:
        pathloss_spec = physics_ctx.get("pathloss") if isinstance(physics_ctx, dict) else None
        if isinstance(pathloss_spec, dict):
            constraint = pathloss_spec.get("constraint")
            distance_m = pathloss_spec.get("distance_m")
            if isinstance(constraint, PathlossConstraint) and distance_m is not None:
                distance_m = distance_m.to(r4_hat_dbm.device)
                if distance_m.ndim == 2 and distance_m.size(-1) == r4_hat_dbm.size(-1):
                    distance_m = distance_m.mean(dim=-1)
                phys_loss = phys_loss + constraint.penalty(r4_hat_dbm.mean(dim=-1), distance_m)
        antenna_spec = physics_ctx.get("antenna") if isinstance(physics_ctx, dict) else None
        if isinstance(antenna_spec, dict):
            penalty_fn = antenna_spec.get("penalty_fn")
            if callable(penalty_fn):
                phys_loss = phys_loss + penalty_fn(r4_hat_dbm)
        extra_penalties = physics_ctx.get("extra") if isinstance(physics_ctx, dict) else None
        if isinstance(extra_penalties, (list, tuple)):
            for fn in extra_penalties:
                if callable(fn):
                    phys_loss = phys_loss + fn(r4_hat_dbm)
    losses["phys"] = phys_loss
    losses["total"] = losses["mix"] + losses["data"] + losses["phys"]
    return losses


__all__ = ["recon_loss"]
