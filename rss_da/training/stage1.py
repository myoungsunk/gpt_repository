"""Stage-1 학습 루프."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from itertools import chain
from typing import Dict, Optional, Tuple

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
    sup0_nll: float
    sup1_nll: float
    recon_data_raw: float
    recon_mix_norm: float
    recon_mix_raw: float
    recon_phys: float
    deg_rmse: float
    kappa_mean: float
    mix_weight: float
    mix_weighted_norm: float
    mix_var: float
    m3_enabled: Optional[float] = None
    m3_gate_mean: Optional[float] = None
    m3_gate_p10: Optional[float] = None
    m3_gate_p90: Optional[float] = None
    m3_keep_ratio: Optional[float] = None
    m3_resid_abs_mean_deg: Optional[float] = None
    m3_resid_abs_p90_deg: Optional[float] = None
    m3_delta_clip_rate: Optional[float] = None
    m3_kappa_corr_spearman: Optional[float] = None
    m3_residual_penalty: Optional[float] = None
    m3_gate_entropy: Optional[float] = None
    m3_gate_threshold: Optional[float] = None


def _ensure_two_dim(tensor: torch.Tensor) -> torch.Tensor:
    """혼합 모델 대응."""

    if tensor.ndim == 3:
        return tensor[:, 0, :]
    return tensor


class Stage1Trainer:
    """Stage-1 학습기."""

    def __init__(self, cfg: Config, device: Optional[torch.device] = None) -> None:
        self.cfg = cfg
        self.phase = cfg.train.phase
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
        phase_enables_m3 = self.phase == "finetune_m3"
        self.m3_enabled = cfg.train.use_m3 or phase_enables_m3
        if self.phase == "pretrain_m2":
            self.m3_enabled = False
        self.m3: Optional[ResidualCalibrator]
        if self.m3_enabled:
            delta_max = math.radians(cfg.train.m3_delta_max_deg)
            feat_dim = self._m3_feature_dim(phi_dim)
            self.m3 = ResidualCalibrator(
                in_dim=feat_dim,
                hidden=latent_dim,
                dropout_p=dropout,
                delta_max_rad=delta_max,
                gate_mode=cfg.train.m3_gate_mode,
                gate_tau=cfg.train.m3_gate_tau,
            ).to(self.device)
        else:
            self.m3 = None
        self.m3_delta_max_rad = math.radians(cfg.train.m3_delta_max_deg)
        self.m3_delta_warmup_rad = min(self.m3_delta_max_rad, math.radians(cfg.train.m3_delta_warmup_deg))
        self.m3_gain_start = cfg.train.m3_gain_start
        self.m3_gain_end = cfg.train.m3_gain_end
        self.m3_gain_ramp_steps = max(0, cfg.train.m3_gain_ramp_steps)
        self.m3_apply_eval_only = cfg.train.m3_apply_eval_only
        self.m3_freeze_m2 = cfg.train.m3_freeze_m2 or (self.phase == "finetune_m3")
        base_params = list(
            chain(
                self.e4.parameters(),
                self.e5.parameters(),
                self.fuse.parameters(),
                self.adapter.parameters(),
                self.decoder.parameters(),
                self.m3.parameters() if self.m3 is not None else [],
            )
        )
        if not self.m3_freeze_m2:
            base_params.extend(self.m2.backbone.parameters())
        head_params = []
        if not self.m3_freeze_m2:
            head_params = list(chain(self.m2.mu_head.parameters(), self.m2.kappa_head.parameters()))
            if self.m2.logits_head is not None:
                head_params.extend(self.m2.logits_head.parameters())
        else:
            for param in self.m2.parameters():
                param.requires_grad = False
        nn.init.constant_(self.m2.kappa_head.bias, 0.5)
        param_groups = [{"params": base_params}]
        if head_params:
            param_groups.append(
                {
                    "params": head_params,
                    "lr": cfg.train.learning_rate * cfg.train.m2_head_lr_scale,
                }
            )
        self.optimizer = optim.Adam(
            param_groups,
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
        self.global_step = 0
        self._debug_stats_printed = False
        self.mix_warmup_steps = cfg.data.mix_warmup_steps
        self.mix_ramp_steps = cfg.data.mix_ramp_steps
        self.mix_weight_cap = cfg.data.mix_weight_max
        self.mix_variance_floor = cfg.data.mix_variance_floor
        total_span = max(1, self.mix_warmup_steps + self.mix_ramp_steps)
        warmup_steps = int(total_span * cfg.train.m3_warmup_frac)
        self.m3_warmup_steps = max(1, warmup_steps)
        detach_steps = int(total_span * cfg.train.m3_detach_warmup_epochs / max(cfg.train.epochs, 1))
        self.m3_detach_steps = max(1, detach_steps if detach_steps > 0 else self.m3_warmup_steps)
        self.m3_detach_m2 = cfg.train.m3_detach_m2
        self.m3_lambda_resid = cfg.train.m3_lambda_resid
        self.m3_lambda_gate = cfg.train.m3_lambda_gate_entropy
        self.m3_lambda_keep = cfg.train.m3_lambda_keep_target
        self.m3_gate_keep_threshold = cfg.train.m3_gate_keep_threshold
        self.m3_quantile_keep = cfg.train.m3_quantile_keep
        keep_steps = int(total_span * cfg.train.m3_keep_warmup_epochs / max(cfg.train.epochs, 1))
        self.m3_keep_schedule_steps = max(1, keep_steps if keep_steps > 0 else self.m3_warmup_steps)
        self.m3_target_keep_start = cfg.train.m3_target_keep_start
        self.m3_target_keep_end = cfg.train.m3_target_keep_end

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
        def _to_device(obj):
            if isinstance(obj, torch.Tensor):
                return obj.to(self.device)
            if isinstance(obj, dict):
                return {key: _to_device(val) for key, val in obj.items()}
            return obj

        return {key: _to_device(value) for key, value in batch.items()}

    @staticmethod
    def _m3_feature_dim(phi_dim: int) -> int:
        base_z5d = 5
        phi = phi_dim
        c_meas = 2
        four_rss = 4
        mu = 2
        cos_comp = 2
        sin_comp = 2
        kappa = 2
        return base_z5d + phi + c_meas + four_rss + mu + cos_comp + sin_comp + kappa

    @staticmethod
    def _spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float:
        if x.numel() < 8:
            return 0.0
        x = x.flatten().double()
        y = y.flatten().double()
        if torch.isnan(x).any() or torch.isnan(y).any():
            return 0.0
        x_rank = torch.argsort(torch.argsort(x))
        y_rank = torch.argsort(torch.argsort(y))
        x_rank = x_rank.double()
        y_rank = y_rank.double()
        x_rank = x_rank - x_rank.mean()
        y_rank = y_rank - y_rank.mean()
        denom = torch.sqrt((x_rank.pow(2).sum()) * (y_rank.pow(2).sum()))
        if denom <= 0:
            return 0.0
        return torch.clamp((x_rank * y_rank).sum() / denom, min=-1.0, max=1.0).item()

    def _assemble_m3_features(
        self,
        z5d: torch.Tensor,
        phi: torch.Tensor,
        c_meas: torch.Tensor,
        four_rss: torch.Tensor,
        mu: torch.Tensor,
        kappa: torch.Tensor,
    ) -> torch.Tensor:
        comps = [z5d]
        comps.append(phi)
        comps.append(c_meas)
        if four_rss.numel() == z5d.size(0) * 4:
            comps.append(four_rss)
        else:
            comps.append(torch.zeros(z5d.size(0), 4, device=z5d.device, dtype=z5d.dtype))
        comps.append(mu)
        comps.append(torch.cos(mu))
        comps.append(torch.sin(mu))
        comps.append(kappa)
        return torch.cat(comps, dim=-1)

    def _m3_weight(self) -> float:
        if not self.m3_enabled:
            return 0.0
        ramp = max(1, self.mix_ramp_steps)
        start = self.m3_warmup_steps
        if self.global_step < start:
            return 0.0
        progress = min(1.0, (self.global_step - start) / ramp)
        return progress

    def _m3_controls(self) -> Tuple[float, float, float, float]:
        ramp = self._m3_weight()
        base_delta = self.m3_delta_max_rad
        warm_delta = self.m3_delta_warmup_rad
        effective_delta = warm_delta + (base_delta - warm_delta) * ramp
        keep_progress = min(1.0, self.global_step / max(1, self.m3_keep_schedule_steps))
        if self.m3_gain_ramp_steps > 0:
            gain_progress = min(1.0, self.global_step / self.m3_gain_ramp_steps)
        else:
            gain_progress = keep_progress
        gain = self.m3_gain_start + (self.m3_gain_end - self.m3_gain_start) * gain_progress
        keep_target = self.m3_target_keep_start + (self.m3_target_keep_end - self.m3_target_keep_start) * keep_progress
        return ramp, effective_delta, gain, keep_target

    def train_step(self, batch: Dict[str, torch.Tensor], enable_pass1: bool = True) -> Stage1Outputs:
        """단일 미니배치 학습."""

        self.train(True)
        batch = self._prepare_batch(batch)
        z5d = batch["z5d"]  # torch.FloatTensor[B,5]
        theta_gt = batch["theta_gt"]  # torch.FloatTensor[B,2]
        c_meas = batch["c_meas"]  # torch.FloatTensor[B,2] 상대 dB
        if self.cfg.data.input_scale != "relative_db" and "c_meas_abs" in batch:
            c_meas = batch["c_meas_abs"]
        c_meas_rel = batch.get("c_meas_rel")
        if c_meas_rel is not None and c_meas_rel.numel() == 0:
            c_meas_rel = None
        four_rss = batch["four_rss"]  # torch.FloatTensor[B,4] 또는 [B,0]
        if not self._debug_stats_printed:
            stats = (
                z5d.mean().item(),
                z5d.std().item(),
                z5d.min().item(),
                z5d.max().item(),
            )
            logging.info("[Stage-1][debug] z5d stats mean=%.3f std=%.3f min=%.3f max=%.3f", *stats)
            self._debug_stats_printed = True
        self.optimizer.zero_grad()
        mu0, kappa0, _ = self.m2(z5d, None)
        mu0 = _ensure_two_dim(mu0)
        kappa0 = _ensure_two_dim(kappa0)
        loss_sup0 = von_mises_nll(mu0, kappa0, theta_gt)
        recon = {
            "mix": torch.zeros_like(loss_sup0),
            "mix_raw": torch.zeros_like(loss_sup0),
            "mix_var": torch.ones_like(loss_sup0),
            "data": torch.zeros_like(loss_sup0),
            "phys": torch.zeros_like(loss_sup0),
            "total": torch.zeros_like(loss_sup0),
        }
        mu1 = mu0
        kappa1 = kappa0
        loss_sup1 = torch.zeros_like(loss_sup0)
        m3_resid_penalty = torch.zeros_like(loss_sup0)
        m3_gate_penalty = torch.zeros_like(loss_sup0)
        m3_stats: Optional[Tuple[float, float, float, float, float, float, float, float, float, float, float]] = None
        mu_eval = mu0
        if enable_pass1:
            h5 = self.e5(z5d)  # torch.FloatTensor[B,H]
            r4_hat = self.decoder(h5.detach(), mu0)  # torch.FloatTensor[B,4]
            h4_hat = self.e4(r4_hat.detach())  # torch.FloatTensor[B,H]
            mask = torch.zeros(z5d.size(0), 1, device=self.device)
            h = self.fuse(h5, h4_hat, mask)
            phi = self.adapter(h).detach()
            mu1, kappa1, _ = self.m2(z5d, phi)
            mu1 = _ensure_two_dim(mu1)
            kappa1 = _ensure_two_dim(kappa1)
            mu_eval = mu1
            if self.m3_enabled and self.m3 is not None:
                c_feat = c_meas_rel if c_meas_rel is not None else c_meas
                features = self._assemble_m3_features(z5d, phi, c_feat, four_rss, mu1, kappa1)
                ramp, delta_cap, gain, keep_target = self._m3_controls()
                detached_for_warmup = False
                if self.m3_detach_m2 and self.global_step < self.m3_detach_steps:
                    detached_for_warmup = True
                detach_inputs = self.m3_freeze_m2 or detached_for_warmup
                features_in = features.detach() if detach_inputs else features
                mu_in = mu1.detach() if detach_inputs else mu1
                kappa_in = kappa1.detach() if detach_inputs else kappa1
                m3_out = self.m3(
                    features_in,
                    mu_in,
                    kappa_in,
                    ramp=ramp,
                    delta_max=delta_cap,
                    gain=gain,
                    gate_threshold=self.m3_gate_keep_threshold,
                    extras={},
                )
                mu_ref = m3_out["mu_ref"]
                gate = m3_out["gate"]
                gate_raw = m3_out["gate_raw"]
                delta_effect = m3_out["delta_effect"]
                delta_raw = m3_out["delta"]
                keep_mask_bool = m3_out["keep_mask"]
                threshold_used = float(self.m3_gate_keep_threshold)
                if self.m3_quantile_keep is not None and gate.numel() > 0:
                    q = float(min(max(self.m3_quantile_keep, 0.0), 1.0))
                    quantile_value = torch.quantile(gate.detach(), q)
                    threshold_used = quantile_value.item()
                    keep_mask_bool = gate >= quantile_value
                clip_mask = keep_mask_bool & (delta_raw.abs() >= max(0.0, delta_cap - 1e-6))
                keep_mask = keep_mask_bool.float()
                m3_resid_penalty = (delta_effect.pow(2).mean() * self.m3_lambda_resid)
                gate_clamped = torch.clamp(gate, min=1e-6, max=1 - 1e-6)
                gate_entropy = -(
                    gate_clamped * torch.log(gate_clamped)
                    + (1 - gate_clamped) * torch.log(1 - gate_clamped)
                ).mean()
                m3_gate_penalty = gate_entropy * self.m3_lambda_gate
                if self.m3_lambda_keep > 0.0:
                    keep_mean = keep_mask.mean()
                    keep_penalty = (keep_mean - keep_target) ** 2 * self.m3_lambda_keep
                    m3_gate_penalty = m3_gate_penalty + keep_penalty
                use_ref_for_loss = not (
                    detached_for_warmup or (self.m3_apply_eval_only and self.m3.training)
                )
                if use_ref_for_loss:
                    mu1 = mu_ref
                mu_eval = mu_ref
                with torch.no_grad():
                    gate_mean = gate.mean().item()
                    gate_p10 = torch.quantile(gate, 0.10).item()
                    gate_p90 = torch.quantile(gate, 0.90).item()
                    keep_ratio = keep_mask.mean().item()
                    resid_abs_deg = torch.rad2deg(delta_effect.abs())
                    resid_mean_deg = resid_abs_deg.mean().item()
                    resid_p90_deg = torch.quantile(resid_abs_deg, 0.90).item()
                    clip_rate = (clip_mask & (keep_mask.bool())).float().mean().item()
                    err = torch.atan2(torch.sin(mu_eval - theta_gt), torch.cos(mu_eval - theta_gt)).abs()
                    err_deg = torch.rad2deg(err).mean(dim=-1)
                    kappa_sample = kappa1.mean(dim=-1)
                    kappa_corr = self._spearman_corr(kappa_sample.detach(), err_deg.detach())
                    m3_stats = (
                        1.0,
                        gate_mean,
                        gate_p10,
                        gate_p90,
                        keep_ratio,
                        resid_mean_deg,
                        resid_p90_deg,
                        clip_rate,
                        kappa_corr,
                        m3_resid_penalty.detach().item(),
                        m3_gate_penalty.detach().item(),
                        threshold_used,
                    )
            loss_sup1 = von_mises_nll(mu1, kappa1, theta_gt)
            r4_gt = four_rss if four_rss.numel() == z5d.size(0) * 4 else None
            recon = recon_loss(
                r4_hat,
                r4_gt_dbm=r4_gt,
                c_meas_dbm=c_meas,
                c_meas_rel_db=c_meas_rel,
                input_scale=self.cfg.data.input_scale,
                mix_variance_floor=self.mix_variance_floor,
            )
        w_sup, _, w_mix, _, w_phys = self.cfg.train.loss_weights
        mix_scale = self._mix_weight()
        mix_weight = min(w_mix * mix_scale, self.mix_weight_cap)
        loss_total = w_sup * (loss_sup0 + loss_sup1)
        loss_total = (
            loss_total
            + mix_weight * (recon["mix"] + recon["data"])
            + w_phys * recon["phys"]
            + m3_resid_penalty
            + m3_gate_penalty
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
        deg_rmse = circular_mean_error_deg(mu_eval.detach(), theta_gt.detach()).item()
        if m3_stats is None and self.m3_enabled:
            m3_stats = (
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                m3_resid_penalty.detach().item(),
                m3_gate_penalty.detach().item(),
                float(self.m3_gate_keep_threshold),
            )
        result = Stage1Outputs(
            loss_total=loss_total.detach().item(),
            sup0_nll=loss_sup0.detach().item(),
            sup1_nll=loss_sup1.detach().item(),
            recon_data_raw=recon["data"].detach().item(),
            recon_mix_norm=recon["mix"].detach().item(),
            recon_mix_raw=recon["mix_raw"].detach().item(),
            recon_phys=recon["phys"].detach().item(),
            deg_rmse=deg_rmse,
            kappa_mean=kappa1.detach().mean().item(),
            mix_weight=float(mix_weight),
            mix_weighted_norm=(mix_weight * recon["mix"]).detach().item(),
            mix_var=recon["mix_var"].detach().item(),
            m3_enabled=m3_stats[0] if m3_stats is not None else None,
            m3_gate_mean=m3_stats[1] if m3_stats is not None else None,
            m3_gate_p10=m3_stats[2] if m3_stats is not None else None,
            m3_gate_p90=m3_stats[3] if m3_stats is not None else None,
            m3_keep_ratio=m3_stats[4] if m3_stats is not None else None,
            m3_resid_abs_mean_deg=m3_stats[5] if m3_stats is not None else None,
            m3_resid_abs_p90_deg=m3_stats[6] if m3_stats is not None else None,
            m3_delta_clip_rate=m3_stats[7] if m3_stats is not None else None,
            m3_kappa_corr_spearman=m3_stats[8] if m3_stats is not None else None,
            m3_residual_penalty=m3_stats[9] if m3_stats is not None else None,
            m3_gate_entropy=m3_stats[10] if m3_stats is not None else None,
            m3_gate_threshold=m3_stats[11] if m3_stats is not None else None,
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

    def _mix_weight(self) -> float:
        warmup = self.mix_warmup_steps
        ramp = max(1, self.mix_ramp_steps)
        if self.global_step < warmup:
            return 0.0
        progress = min(1.0, (self.global_step - warmup) / ramp)
        return progress * self.mix_weight_cap


__all__ = ["Stage1Trainer", "Stage1Outputs"]
