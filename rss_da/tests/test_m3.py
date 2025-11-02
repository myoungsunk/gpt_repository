from __future__ import annotations

import math

import torch

from rss_da.losses.vm_nll import von_mises_nll
from rss_da.models.m3 import ResidualCalibrator


def test_residual_calibrator_outputs_shapes() -> None:
    calibrator = ResidualCalibrator(in_dim=21, hidden=16, dropout_p=0.0)
    features = torch.randn(8, 21)
    mu = torch.zeros(8, 2)
    kappa = torch.ones(8, 2) * 3.0
    out = calibrator(features, mu, kappa)
    assert out["delta"].shape == (8, 2)
    assert out["gate"].shape == (8, 1)
    assert out["mu_ref"].shape == (8, 2)
    assert out["clip_mask"].shape == (8, 2)


def test_mu_ref_within_bounds() -> None:
    calibrator = ResidualCalibrator(in_dim=10, hidden=8, dropout_p=0.0)
    features = torch.randn(4, 10)
    mu = torch.full((4, 2), math.pi - 0.1)
    kappa = torch.ones(4, 2) * 2.0
    out = calibrator(features, mu, kappa)
    mu_ref = out["mu_ref"]
    assert torch.all(mu_ref <= math.pi)
    assert torch.all(mu_ref > -math.pi)


def test_gate_monotonic_with_kappa() -> None:
    calibrator = ResidualCalibrator(in_dim=6, hidden=8, dropout_p=0.0)
    with torch.no_grad():
        calibrator.fc1.weight.zero_()
        calibrator.fc1.bias.zero_()
        calibrator.fc2.weight.zero_()
        calibrator.fc2.bias.zero_()
        calibrator.residual_head.weight.zero_()
        calibrator.residual_head.bias.zero_()
        calibrator.gate_head.weight.zero_()
        calibrator.gate_head.bias.zero_()
    features = torch.zeros(32, 6)
    mu = torch.zeros(32, 2)
    kappa = torch.ones(32, 2)
    kappa[16:] = 5.0
    out = calibrator(features, mu, kappa)
    gate = out["gate"].view(32)
    low_gate = gate[:16].mean().item()
    high_gate = gate[16:].mean().item()
    assert high_gate >= low_gate - 1e-5


def test_residual_reduces_nll_when_matching_target() -> None:
    calibrator = ResidualCalibrator(in_dim=4, hidden=4, dropout_p=0.0, gate_mode="none")
    with torch.no_grad():
        calibrator.fc1.weight.zero_()
        calibrator.fc1.bias.zero_()
        calibrator.fc2.weight.zero_()
        calibrator.fc2.bias.zero_()
        calibrator.gate_head.weight.zero_()
        calibrator.gate_head.bias.zero_()
    features = torch.zeros(1, 4)
    mu = torch.tensor([[0.0, 0.0]])
    theta = torch.tensor([[0.2, -0.15]])
    kappa = torch.ones(1, 2) * 4.0
    with torch.no_grad():
        calibrator.residual_head.weight.zero_()
        calibrator.residual_head.bias.copy_(theta.squeeze(0))
    base_nll = von_mises_nll(mu, kappa, theta)
    out = calibrator(features, mu, kappa)
    refined_nll = von_mises_nll(out["mu_ref"], kappa, theta)
    assert refined_nll.item() < base_nll.item()
