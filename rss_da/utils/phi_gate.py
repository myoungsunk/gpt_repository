"""Phi-gating utilities based on decoder reconstruction errors.

This module provides `compute_phi_gate`, which turns per-sample reconstruction
errors into a binary mask for gating adapter features. It supports both
absolute-threshold and quantile-based gating, includes minimum-keep safety,
and returns lightweight statistics to aid diagnostics.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import math
import torch


def _safe_quantile(x: torch.Tensor, q: float) -> float:
    q = float(min(max(q, 0.0), 1.0))
    if x.numel() == 0:
        return 0.0
    return torch.quantile(x, q).item()


def compute_phi_gate(
    errors: torch.Tensor,
    *,
    enabled: bool = True,
    threshold: Optional[float] = None,
    quantile: Optional[float] = 0.6,
    min_keep: float = 0.0,
    min_samples_for_quantile: int = 8,
    ramp: float = 1.0,
    collapse_std_epsilon: float = 0.05,
    collapse_ramp_epsilon: float = 0.1,
) -> Tuple[torch.Tensor, float, Optional[float], Dict[str, float | bool]]:
    """Resolve a binary gate from reconstruction errors with diagnostics.

    Args:
        errors: Per-sample reconstruction errors ``[B]`` (smaller is better).
        enabled: Whether gating is enabled. When ``False`` every sample is kept.
        threshold: Absolute threshold. If provided it takes precedence over
            ``quantile``.
        quantile: Optional quantile ``Q_p`` for adaptive thresholding when
            ``threshold`` is ``None``.
        min_keep: Lower bound on the keep ratio to avoid all-zero gates.
        min_samples_for_quantile: Minimum batch size required to trust a quantile
            estimate. Smaller batches fall back to the mean error as a stable
            threshold.
        ramp: A [0,1] progress signal; when too small, quantile gating is
            effectively disabled to avoid early collapse.
        collapse_std_epsilon: If the resulting mask std is below this value,
            consider the gate distribution collapsed.
        collapse_ramp_epsilon: If ``ramp`` is below this value, quantile gating
            is treated as disabled.

    Returns:
        mask: ``torch.FloatTensor[B]`` containing zeros/ones.
        keep_ratio: Fraction of kept samples (after enforcing ``min_keep``).
        used_threshold: Threshold that produced the mask, or ``None`` if gating
            was effectively disabled.
        stats: Lightweight diagnostics for logging/analysis.
    """

    stats: Dict[str, float | bool] = {
        "gate_mean": 0.0,
        "gate_std": 0.0,
        "gate_p25": 0.0,
        "gate_p75": 0.0,
        "quantile_disabled": False,
        "ramp": float(ramp),
    }

    if errors.numel() == 0 or not enabled:
        mask = torch.ones_like(errors, dtype=errors.dtype)
        stats.update(
            {
                "gate_mean": float(mask.mean().item() if mask.numel() > 0 else 0.0),
                "gate_std": float(mask.std(unbiased=False).item() if mask.numel() > 1 else 0.0),
                "gate_p25": float(_safe_quantile(mask, 0.25)),
                "gate_p75": float(_safe_quantile(mask, 0.75)),
            }
        )
        return mask, 1.0 if errors.numel() > 0 else 0.0, None, stats

    errs = errors.detach()
    used_threshold: Optional[float] = None
    mask = torch.ones_like(errs, dtype=errs.dtype)

    quantile_disabled = ramp < collapse_ramp_epsilon

    if threshold is not None:
        used_threshold = float(threshold)
    elif quantile is not None and not quantile_disabled:
        q = float(min(max(quantile, 0.0), 1.0))
        if errs.numel() >= max(1, min_samples_for_quantile):
            used_threshold = torch.quantile(errs, q).item()
        else:
            used_threshold = errs.mean().item()
            quantile_disabled = True

    if used_threshold is not None:
        mask = (errs <= used_threshold).to(dtype=errs.dtype)

    keep_ratio = float(mask.mean().item() if mask.numel() > 0 else 0.0)

    if min_keep > 0.0 and mask.numel() > 0:
        target = float(min(max(min_keep, 0.0), 1.0))
        if keep_ratio < target:
            k = max(1, int(math.ceil(target * mask.numel())))
            _, indices = torch.topk(errs, k, largest=False)
            mask.zero_()
            mask[indices] = 1.0
            keep_ratio = float(k) / float(mask.numel())
            if used_threshold is not None:
                used_threshold = float(errs[indices].max().item())

    gate_std = float(mask.std(unbiased=False).item() if mask.numel() > 1 else 0.0)
    stats.update(
        {
            "gate_mean": keep_ratio,
            "gate_std": gate_std,
            "gate_p25": float(_safe_quantile(mask, 0.25)),
            "gate_p75": float(_safe_quantile(mask, 0.75)),
            "quantile_disabled": bool(quantile_disabled or (gate_std < collapse_std_epsilon)),
            "ramp": float(ramp),
        }
    )

    return mask, keep_ratio, used_threshold, stats


__all__ = ["compute_phi_gate"]

