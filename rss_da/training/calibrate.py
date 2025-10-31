"""κ-Temperature scaling."""
from __future__ import annotations

from dataclasses import dataclass
import torch

from rss_da.losses.vm_nll import von_mises_nll
from rss_da.utils.metrics import ece_placeholder


@dataclass
class CalibrationResult:
    temperature: float
    nll: float
    ece: float


class KappaTemperatureCalibrator:
    """κ-Temperature scaling with calibration split."""

    def __init__(self, lr: float = 0.05, max_iter: int = 200) -> None:
        self.lr = lr
        self.max_iter = max_iter
        self.temperature: float = 1.0

    def fit(self, mu: torch.Tensor, kappa: torch.Tensor, theta: torch.Tensor) -> CalibrationResult:
        device = mu.device
        log_t = torch.nn.Parameter(torch.zeros(1, device=device))
        optimizer = torch.optim.Adam([log_t], lr=self.lr)
        for _ in range(self.max_iter):
            optimizer.zero_grad()
            temp = torch.exp(log_t)
            scaled_kappa = kappa / temp
            loss = von_mises_nll(mu, scaled_kappa, theta, reduction="mean")
            loss.backward()
            optimizer.step()
        self.temperature = torch.exp(log_t.detach()).item()
        return self.evaluate(mu, kappa, theta)

    def evaluate(self, mu: torch.Tensor, kappa: torch.Tensor, theta: torch.Tensor) -> CalibrationResult:
        temp = torch.tensor(self.temperature, device=mu.device, dtype=mu.dtype)
        scaled_kappa = kappa / temp
        nll = von_mises_nll(mu, scaled_kappa, theta, reduction="mean").item()
        logits = torch.stack([scaled_kappa * torch.cos(theta - mu), -scaled_kappa * torch.cos(theta - mu)], dim=-1)
        labels = (torch.cos(theta - mu) > 0).long()
        ece = ece_placeholder(logits.view(-1, 2), labels.view(-1))["ece"].item()
        return CalibrationResult(temperature=self.temperature, nll=nll, ece=ece)


def calibrate(mu: torch.Tensor, kappa: torch.Tensor, theta: torch.Tensor, lr: float = 0.05, max_iter: int = 200) -> CalibrationResult:
    calibrator = KappaTemperatureCalibrator(lr=lr, max_iter=max_iter)
    return calibrator.fit(mu, kappa, theta)


__all__ = ["KappaTemperatureCalibrator", "CalibrationResult", "calibrate"]
