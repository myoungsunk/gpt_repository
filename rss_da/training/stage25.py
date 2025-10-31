"""Stage-2.5 학습 루프."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from rss_da.config import Config
from rss_da.losses.align import apply_alignment, deep_coral_loss
from rss_da.losses.balance import GradNormController, UncertaintyWeighting
from rss_da.losses.kd import kd_loss_bundle
from rss_da.losses.recon import recon_loss
from rss_da.losses.vm_nll import von_mises_nll
from rss_da.models.decoder import DecoderD
from rss_da.models.encoders import Adapter, E4, E5, Fuse
from rss_da.models.m2 import DoAPredictor
from rss_da.training.ema import build_ema, update_ema
from rss_da.utils.metrics import circular_mean_error_deg


@dataclass
class Stage25Outputs:
    """Stage-2.5 스텝 결과."""

    loss_total: float
    loss_sup: float
    loss_kd: float
    loss_data: float
    loss_mix: float
    loss_phys: float
    loss_align: float
    deg_rmse: float
    kappa_mean: float
    mix_weight: float
    mix_weighted: float


def _ensure_two_dim(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor[:, 0, :]
    return tensor


class Stage25Trainer:
    """Stage-2.5 도메인 적응 학습기."""

    def __init__(
        self,
        cfg: Config,
        device: Optional[torch.device] = None,
        teacher_modules: Optional[Dict[str, nn.Module]] = None,
    ) -> None:
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
        base_params = list(
            chain(
                self.e4.parameters(),
                self.e5.parameters(),
                self.fuse.parameters(),
                self.adapter.parameters(),
                self.decoder.parameters(),
                self.m2.backbone.parameters(),
            )
        )
        head_params = list(chain(self.m2.mu_head.parameters(), self.m2.kappa_head.parameters()))
        if self.m2.logits_head is not None:
            head_params.extend(self.m2.logits_head.parameters())
        nn.init.constant_(self.m2.kappa_head.bias, 0.5)
        self.optimizer = optim.Adam(
            [
                {"params": base_params},
                {
                    "params": head_params,
                    "lr": cfg.train.learning_rate * cfg.train.m2_head_lr_scale,
                },
            ],
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
        self.gradnorm: Optional[GradNormController]
        self.uncertainty: Optional[UncertaintyWeighting]
        task_names = ["sup", "kd", "mix", "phys"]
        if cfg.train.gradnorm:
            self.gradnorm = GradNormController(task_names)
        else:
            self.gradnorm = None
        if cfg.train.uncertainty_weighting:
            self.uncertainty = UncertaintyWeighting(task_names)
        else:
            self.uncertainty = None
        self.align_method = ""
        if cfg.train.use_coral:
            self.align_method = "coral"
        elif cfg.train.use_dann:
            self.align_method = "dann"
        elif cfg.train.use_cdan:
            self.align_method = "cdan"
        self.domain_classifier: Optional[nn.Module] = None
        if self.align_method in {"dann", "cdan"}:
            in_dim = latent_dim if self.align_method == "dann" else latent_dim * 4
            self.domain_classifier = nn.Sequential(
                nn.Linear(in_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, 2),
            ).to(self.device)
        self.teacher = self._build_teacher(teacher_modules)
        self.shared_parameters = tuple(
            chain(
                self.e5.parameters(),
                self.decoder.parameters(),
                self.m2.parameters(),
            )
        )
        self.global_step = 0
        self.mix_warmup_steps = cfg.data.mix_warmup_steps
        self.mix_ramp_steps = cfg.data.mix_ramp_steps

    def _build_teacher(self, teacher_modules: Optional[Dict[str, nn.Module]]) -> Dict[str, nn.Module]:
        teacher: Dict[str, nn.Module] = {}
        modules = {
            "e4": self.e4,
            "e5": self.e5,
            "fuse": self.fuse,
            "adapter": self.adapter,
            "decoder": self.decoder,
            "m2": self.m2,
        }
        if teacher_modules is not None:
            for name, module in modules.items():
                if name in teacher_modules:
                    module.load_state_dict(teacher_modules[name].state_dict())
        for name, module in modules.items():
            teacher[name] = build_ema(module).to(self.device)
        return teacher

    def train(self, mode: bool = True) -> None:
        for module in (self.e4, self.e5, self.fuse, self.adapter, self.decoder, self.m2):
            module.train(mode)
        if self.domain_classifier is not None:
            self.domain_classifier.train(mode)

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        def _to_device(value: torch.Tensor) -> torch.Tensor:
            return value.to(self.device)

        def _maybe_to_device(obj):
            if isinstance(obj, torch.Tensor):
                return _to_device(obj)
            if isinstance(obj, dict):
                return {key: _maybe_to_device(val) for key, val in obj.items()}
            return obj

        return {key: _maybe_to_device(value) for key, value in batch.items()}

    def _teacher_forward(self, z5d: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            mu0, kappa0, _ = self.teacher["m2"](z5d, None)
            mu0 = _ensure_two_dim(mu0)
            kappa0 = _ensure_two_dim(kappa0)
            h5 = self.teacher["e5"](z5d)
            r4_hat = self.teacher["decoder"](h5, mu0)
            h4_hat = self.teacher["e4"](r4_hat)
            mask = torch.zeros(z5d.size(0), 1, device=z5d.device)
            h = self.teacher["fuse"](h5, h4_hat, mask)
            phi = self.teacher["adapter"](h)
            mu1, kappa1, _ = self.teacher["m2"](z5d, phi.detach())
            mu1 = _ensure_two_dim(mu1)
            kappa1 = _ensure_two_dim(kappa1)
        return {
            "mu0": mu0.detach(),
            "kappa0": kappa0.detach(),
            "mu1": mu1.detach(),
            "kappa1": kappa1.detach(),
            "phi": phi.detach(),
            "h": h.detach(),
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Stage25Outputs:
        self.train(True)
        batch = self._prepare_batch(batch)
        z5d = batch["z5d"]
        theta_gt = batch["theta_gt"]
        c_meas = batch["c_meas"]  # 상대 dB
        if self.cfg.data.input_scale != "relative_db" and "c_meas_abs" in batch:
            c_meas = batch["c_meas_abs"]
        c_meas_rel = batch.get("c_meas_rel")
        if c_meas_rel is not None and c_meas_rel.numel() == 0:
            c_meas_rel = None
        self.optimizer.zero_grad()
        mu0_s, kappa0_s, _ = self.m2(z5d, None)
        mu0_s = _ensure_two_dim(mu0_s)
        kappa0_s = _ensure_two_dim(kappa0_s)
        h5_s = self.e5(z5d)
        r4_hat_s = self.decoder(h5_s.detach(), mu0_s)
        h4_hat_s = self.e4(r4_hat_s.detach())
        mask = torch.zeros(z5d.size(0), 1, device=self.device)
        h_s = self.fuse(h5_s, h4_hat_s, mask)
        phi_s = self.adapter(h_s).detach()
        mu1_s, kappa1_s, logits_s = self.m2(z5d, phi_s)
        mu1_s = _ensure_two_dim(mu1_s)
        kappa1_s = _ensure_two_dim(kappa1_s)
        loss_sup = von_mises_nll(mu1_s, kappa1_s, theta_gt)
        recon = recon_loss(
            r4_hat_s,
            r4_gt_dbm=None,
            c_meas_dbm=c_meas,
            c_meas_rel_db=c_meas_rel,
            input_scale=self.cfg.data.input_scale,
        )
        teacher_out = self._teacher_forward(z5d)
        gate = torch.sigmoid(teacher_out["kappa1"].mean(dim=-1, keepdim=True))
        kd_losses = kd_loss_bundle(
            {
                "mu_student": mu1_s,
                "kappa_student": kappa1_s,
                "mu_teacher": teacher_out["mu1"],
                "kappa_teacher": teacher_out["kappa1"],
                "h_student": h_s,
                "h_teacher": teacher_out["h"],
            },
            temperature=self.cfg.train.tau,
            gating=gate,
        )
        align_loss = torch.zeros(1, device=self.device)
        if self.align_method == "coral":
            align_loss = deep_coral_loss(h_s, teacher_out["h"])
        elif self.align_method == "dann" and self.domain_classifier is not None:
            domain_labels = torch.zeros(z5d.size(0), dtype=torch.long, device=self.device)
            align_loss = apply_alignment(
                "dann",
                self.domain_classifier,
                h_s,
                teacher_out["h"],
                domain_labels=domain_labels,
            )
        elif self.align_method == "cdan" and self.domain_classifier is not None:
            domain_labels = torch.zeros(z5d.size(0), dtype=torch.long, device=self.device)
            class_logits = torch.cat([torch.cos(mu1_s), torch.sin(mu1_s)], dim=-1)
            align_loss = apply_alignment(
                "cdan",
                self.domain_classifier,
                h_s,
                teacher_out["h"],
                domain_labels=domain_labels,
                class_logits=class_logits,
            )
        mix_scale = self._mix_weight()
        mix_weight = self.cfg.train.loss_weights[2] * mix_scale
        base_losses = {
            "sup": loss_sup,
            "kd": kd_losses["kd_total"],
            "mix": mix_weight * (recon["mix"] + recon["data"]),
            "phys": recon["phys"],
        }
        total_loss = None
        if self.uncertainty is not None:
            weighted = self.uncertainty(base_losses)
            total_loss = weighted["total"]
        elif self.gradnorm is not None:
            weights = self.gradnorm(base_losses, self.shared_parameters)
            total_loss = sum(weights[name] * base_losses[name] for name in base_losses)
        else:
            w_sup, w_kd, w_mix, w_align, w_phys = self.cfg.train.loss_weights
            total_loss = (
                w_sup * base_losses["sup"]
                + w_kd * base_losses["kd"]
                + w_mix * base_losses["mix"]
                + w_phys * base_losses["phys"]
            )
        w_align = self.cfg.train.loss_weights[3]
        total_loss = total_loss + w_align * align_loss
        total_loss.backward()
        if self.cfg.train.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                list(chain(
                    self.e4.parameters(),
                    self.e5.parameters(),
                    self.fuse.parameters(),
                    self.adapter.parameters(),
                    self.decoder.parameters(),
                    self.m2.parameters(),
                    self.domain_classifier.parameters() if self.domain_classifier is not None else [],
                )),
                self.cfg.train.grad_clip,
            )
        self.optimizer.step()
        decay = self.cfg.train.ema_decay
        for name, module in {
            "e4": self.e4,
            "e5": self.e5,
            "fuse": self.fuse,
            "adapter": self.adapter,
            "decoder": self.decoder,
            "m2": self.m2,
        }.items():
            update_ema(module, self.teacher[name], decay)
        self.global_step += 1
        deg_rmse = circular_mean_error_deg(mu1_s.detach(), theta_gt.detach()).item()
        outputs = Stage25Outputs(
            loss_total=total_loss.detach().item(),
            loss_sup=loss_sup.detach().item(),
            loss_kd=kd_losses["kd_total"].detach().item(),
            loss_data=recon["data"].detach().item(),
            loss_mix=recon["mix"].detach().item(),
            loss_phys=recon["phys"].detach().item(),
            loss_align=align_loss.detach().item(),
            deg_rmse=deg_rmse,
            kappa_mean=kappa1_s.detach().mean().item(),
            mix_weight=float(mix_weight),
            mix_weighted=(mix_weight * recon["mix"]).detach().item(),
        )
        return outputs

    def _mix_weight(self) -> float:
        warmup = self.mix_warmup_steps
        ramp = max(1, self.mix_ramp_steps)
        if self.global_step < warmup:
            return 0.0
        progress = min(1.0, (self.global_step - warmup) / ramp)
        return progress


__all__ = ["Stage25Trainer", "Stage25Outputs"]
