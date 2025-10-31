"""Stage-1 학습 루프."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from rss_da.config import Config
from rss_da.losses.recon import recon_loss
from rss_da.losses.vm_nll import von_mises_nll
from rss_da.models.decoder import DecoderD
from rss_da.models.encoders import Adapter, E4, E5, Fuse
from rss_da.models.m2 import DoAPredictor
from rss_da.models.m3 import ResidualCalibrator
from rss_da.utils.metrics import circular_mean_error_deg


@dataclass
class Stage1Outputs:
    """단일 스텝 결과."""

    loss_total: float
    loss_sup0: float
    loss_sup1: float
    loss_mix: float
    loss_phys: float
    deg_rmse: float
    kappa_mean: float


def _ensure_two_dim(tensor: torch.Tensor) -> torch.Tensor:
    """혼합 모델 대응."""

    if tensor.ndim == 3:
        return tensor[:, 0, :]
    return tensor


class Stage1Trainer:
    """Stage-1 학습기."""

    def __init__(self, cfg: Config, device: Optional[torch.device] = None) -> None:
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        latent_dim = cfg.train.latent_dim
        phi_dim = cfg.train.phi_dim
        dropout = cfg.train.dropout_p
        self.e4 = E4(latent_dim=latent_dim, dropout_p=dropout).to(self.device)
        self.e5 = E5(latent_dim=latent_dim, dropout_p=dropout).to(self.device)
        self.fuse = Fuse(latent_dim=latent_dim, dropout_p=dropout).to(self.device)
        self.adapter = Adapter(latent_dim=latent_dim, phi_dim=phi_dim, dropout_p=dropout).to(self.device)
        self.decoder = DecoderD(latent_dim=latent_dim, dropout_p=dropout).to(self.device)
        self.m2 = DoAPredictor(phi_dim=phi_dim, latent_dim=latent_dim, dropout_p=dropout).to(self.device)
        self.m3: Optional[ResidualCalibrator]
        if cfg.train.use_m3:
            feat_dim = 5 + phi_dim
            self.m3 = ResidualCalibrator(feature_dim=feat_dim, hidden_dim=latent_dim, dropout_p=dropout).to(self.device)
        else:
            self.m3 = None
        params: Iterable[nn.Parameter] = chain(
            self.e4.parameters(),
            self.e5.parameters(),
            self.fuse.parameters(),
            self.adapter.parameters(),
            self.decoder.parameters(),
            self.m2.parameters(),
            self.m3.parameters() if self.m3 is not None else [],
        )
        self.optimizer = optim.Adam(
            params,
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
        self.global_step = 0

    def train(self, mode: bool = True) -> None:
        self.e4.train(mode)
        self.e5.train(mode)
        self.fuse.train(mode)
        self.adapter.train(mode)
        self.decoder.train(mode)
        self.m2.train(mode)
        if self.m3 is not None:
            self.m3.train(mode)

    def to(self, device: torch.device) -> None:
        self.device = device
        for module in (self.e4, self.e5, self.fuse, self.adapter, self.decoder, self.m2):
            module.to(device)
        if self.m3 is not None:
            self.m3.to(device)

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in batch.items()}

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Stage1Outputs:
        """단일 미니배치 학습."""

        self.train(True)
        batch = self._prepare_batch(batch)
        z5d = batch["z5d"]  # torch.FloatTensor[B,5]
        theta_gt = batch["theta_gt"]  # torch.FloatTensor[B,2]
        c_meas = batch["c_meas"]  # torch.FloatTensor[B,2]
        four_rss = batch["four_rss"]  # torch.FloatTensor[B,4] 또는 [B,0]
        self.optimizer.zero_grad()
        mu0, kappa0, _ = self.m2(z5d, None)
        mu0 = _ensure_two_dim(mu0)
        kappa0 = _ensure_two_dim(kappa0)
        loss_sup0 = von_mises_nll(mu0, kappa0, theta_gt)
        h5 = self.e5(z5d)  # torch.FloatTensor[B,H]
        r4_hat = self.decoder(h5.detach(), mu0)  # torch.FloatTensor[B,4]
        h4_hat = self.e4(r4_hat.detach())  # torch.FloatTensor[B,H]
        mask = torch.zeros(z5d.size(0), 1, device=self.device)
        h = self.fuse(h5, h4_hat, mask)
        phi = self.adapter(h).detach()
        mu1, kappa1, _ = self.m2(z5d, phi)
        mu1 = _ensure_two_dim(mu1)
        kappa1 = _ensure_two_dim(kappa1)
        if self.m3 is not None:
            features = torch.cat([z5d, phi], dim=-1)
            residual = self.m3(mu1, kappa1, features)
            mu1 = torch.atan2(torch.sin(mu1 + residual), torch.cos(mu1 + residual))
        loss_sup1 = von_mises_nll(mu1, kappa1, theta_gt)
        r4_gt = four_rss if four_rss.numel() == z5d.size(0) * 4 else None
        recon = recon_loss(r4_hat, r4_gt_dbm=r4_gt, c_meas_dbm=c_meas)
        w_sup, _, w_mix, _, w_phys = self.cfg.train.loss_weights
        loss_total = (
            w_sup * (loss_sup0 + loss_sup1)
            + w_mix * (recon["mix"] + recon["data"])
            + w_phys * recon["phys"]
        )
        loss_total.backward()
        if self.cfg.train.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                list(chain(
                    self.e4.parameters(),
                    self.e5.parameters(),
                    self.fuse.parameters(),
                    self.adapter.parameters(),
                    self.decoder.parameters(),
                    self.m2.parameters(),
                    self.m3.parameters() if self.m3 is not None else [],
                )),
                self.cfg.train.grad_clip,
            )
        self.optimizer.step()
        self.global_step += 1
        deg_rmse = circular_mean_error_deg(mu1.detach(), theta_gt.detach()).item()
        result = Stage1Outputs(
            loss_total=loss_total.detach().item(),
            loss_sup0=loss_sup0.detach().item(),
            loss_sup1=loss_sup1.detach().item(),
            loss_mix=recon["mix"].detach().item(),
            loss_phys=recon["phys"].detach().item(),
            deg_rmse=deg_rmse,
            kappa_mean=kappa1.detach().mean().item(),
        )
        return result

    def modules(self) -> Dict[str, nn.Module]:
        """모듈 접근."""

        modules: Dict[str, nn.Module] = {
            "e4": self.e4,
            "e5": self.e5,
            "fuse": self.fuse,
            "adapter": self.adapter,
            "decoder": self.decoder,
            "m2": self.m2,
        }
        if self.m3 is not None:
            modules["m3"] = self.m3
        return modules


__all__ = ["Stage1Trainer", "Stage1Outputs"]
