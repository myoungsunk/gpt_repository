"""Utilities for φ quality gating based on decoder reconstruction error."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def compute_phi_gate(
    errors: torch.Tensor,
    *,
    enabled: bool = True,
    threshold: Optional[float] = None,
    quantile: Optional[float] = 0.6,
    min_keep: float = 0.0,
) -> Tuple[torch.Tensor, float, Optional[float]]:
    """Resolve a binary gate from reconstruction errors.

    Args:
        errors: Per-sample reconstruction errors ``[B]`` (smaller is better).
        enabled: Whether gating is enabled. When ``False`` every sample is kept.
        threshold: Absolute threshold ``τ``. If provided it takes precedence over
            ``quantile``.
        quantile: Optional quantile ``Q_p`` for adaptive thresholding when
            ``threshold`` is ``None``.
        min_keep: Lower bound on the keep ratio to avoid degenerate all-zero gates.

    Returns:
        mask: ``torch.FloatTensor[B]`` containing zeros/ones.
        keep_ratio: Fraction of kept samples (after enforcing ``min_keep``).
        used_threshold: Threshold that produced the mask, or ``None`` if gating
            was effectively disabled.
    """

    if errors.numel() == 0 or not enabled:
        mask = torch.ones_like(errors, dtype=errors.dtype)
        return mask, 1.0 if errors.numel() > 0 else 0.0, None

    errs = errors.detach()
    used_threshold: Optional[float] = None
    mask = torch.ones_like(errs, dtype=errs.dtype)

    if threshold is not None:
        used_threshold = float(threshold)
    elif quantile is not None:
        q = float(min(max(quantile, 0.0), 1.0))
        used_threshold = torch.quantile(errs, q).item()

    if used_threshold is not None:
        mask = (errs <= used_threshold).to(dtype=errs.dtype)

    keep_ratio = mask.mean().item() if mask.numel() > 0 else 0.0

    if min_keep > 0.0 and mask.numel() > 0:
        target = float(min(max(min_keep, 0.0), 1.0))
        if keep_ratio < target:
            k = max(1, int(math.ceil(target * mask.numel())))
            values, indices = torch.topk(errs, k, largest=False)
            mask.zero_()
            mask[indices] = 1.0
            keep_ratio = float(k) / float(mask.numel())
            used_threshold = values.max().item() if values.numel() > 0 else used_threshold

    return mask, keep_ratio, used_threshold


__all__ = ["compute_phi_gate"]

