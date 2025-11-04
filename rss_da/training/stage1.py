"""Stage-1 """
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from itertools import chain
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rss_da.config import Config
from rss_da.losses.recon import recon_loss
from rss_da.losses.vm_nll import von_mises_nll
from rss_da.models.decoder import DecoderD
from rss_da.models.encoders import Adapter, E4, E5, Fuse
from rss_da.models.m2 import DoAPredictor
from rss_da.models.m3 import ResidualCalibrator
from rss_da.utils.metrics import circular_mean_error_deg
from rss_da.utils.phi_gate import compute_phi_gate


@dataclass
class Stage1Outputs:


    """??�� ??�� ����."""
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
    m3_gate_temp_tau: Optional[float] = None
    m3_delta_cap_deg_effective: Optional[float] = None
    m3_gain_applied: Optional[float] = None
    m3_resid_dir_agree_rate: Optional[float] = None
    m3_improve_rate_1deg: Optional[float] = None
    m3_worsen_rate_1deg: Optional[float] = None
    m3_err_delta_mean_deg: Optional[float] = None
    phi_quality_improve_corr: Optional[float] = None
    decoder_recon_mae_4rss: Optional[float] = None
    decoder_recon_mae_4rss_p90: Optional[float] = None
    decoder_recon_mae_4rss_std: Optional[float] = None
    decoder_recon_mae_4rss_min: Optional[float] = None
    decoder_recon_mae_4rss_max: Optional[float] = None
    decoder_recon_mae_4rss_std: Optional[float] = None
    decoder_recon_mae_4rss_min: Optional[float] = None
    decoder_recon_mae_4rss_max: Optional[float] = None
    phi_gate_keep_ratio: Optional[float] = None
    phi_gate_threshold: Optional[float] = None
    # phi-gate distribution stats
    phi_gate_mean: Optional[float] = None
    phi_gate_std: Optional[float] = None
    phi_gate_p25: Optional[float] = None
    phi_gate_p75: Optional[float] = None
    phi_gate_quantile_disabled: Optional[bool] = None
    phi_gate_ramp: Optional[float] = None
    recon_mae_theta_corr: Optional[float] = None
    forward_consistency: Optional[float] = None
    grad_norm_m3: Optional[float] = None
    grad_m3_spike_rate: Optional[float] = None
    grad_m3_spike_rate: Optional[float] = None
    # M3 gate distribution stats
    m3_gate_std: Optional[float] = None
    m3_gate_p25: Optional[float] = None
    m3_gate_p75: Optional[float] = None
    m3_quantile_disabled: Optional[bool] = None
    m3_ramp: Optional[float] = None
    # Residual distribution tails
    m3_resid_abs_p99_deg: Optional[float] = None
    # Correlation EMA (stability)
    kappa_corr_ema: Optional[float] = None
    phi_corr_ema: Optional[float] = None
    # Residual distribution tails
    m3_resid_abs_p99_deg: Optional[float] = None
    # Correlation EMA (stability)
    kappa_corr_ema: Optional[float] = None
    phi_corr_ema: Optional[float] = None


def _ensure_two_dim(tensor: torch.Tensor) -> torch.Tensor:
    """??�� ���� ????"""
    if tensor.ndim == 3:
        return tensor[:, 0, :]
    return tensor

class Stage1Trainer:
    """Stage-1 ??��??"""
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
        self.total_epochs = max(1, int(cfg.train.epochs))
        self.m3_delta_max_rad = math.radians(cfg.train.m3_delta_max_deg)
        self.m3_delta_warmup_rad = min(self.m3_delta_max_rad, math.radians(cfg.train.m3_delta_warmup_deg))
        self.m3_gain_start = cfg.train.m3_gain_start
        self.m3_gain_end = cfg.train.m3_gain_end
        self.m3_gain_ramp_steps = max(0, cfg.train.m3_gain_ramp_steps)
        self.m3_apply_eval_only = cfg.train.m3_apply_eval_only
        self.m3_freeze_m2 = cfg.train.m3_freeze_m2 or (self.phase == "finetune_m3")
        self.m3_gate_tau = cfg.train.m3_gate_tau
        self.m3_tau_end = getattr(cfg.train, "m3_tau_end", self.m3_gate_tau)
        self.m3_tau_ramp_steps = max(0, getattr(cfg.train, "m3_tau_ramp_steps", 0))
        self.m3_grad_clip = getattr(cfg.train, "m3_grad_clip", None)
        # EMA states for correlations and phi-gate EWM stats
        self._ema_alpha = 0.9
        self._kappa_corr_ema: Optional[float] = None
        self._phi_corr_ema: Optional[float] = None
        self._phi_ewm_mu: Optional[float] = None
        self._phi_ewm_var: Optional[float] = None
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
        # Store base lrs for scheduling
        self._base_lrs = [g["lr"] for g in self.optimizer.param_groups]
        self.use_cosine_lr = bool(cfg.train.use_cosine_lr)
        self.lr_warmup_steps = max(0, int(cfg.train.lr_warmup_steps))
        self.lr_min_ratio = float(max(0.0, min(1.0, cfg.train.lr_min_ratio)))
        self.global_step = 0
        # Optionally freeze BatchNorm statistics (Stage-3 stability)
        if cfg.train.freeze_batchnorm:
            self._freeze_batchnorm()
        self._debug_stats_printed = False
        self.mix_warmup_steps = cfg.data.mix_warmup_steps
        self.mix_ramp_steps = cfg.data.mix_ramp_steps
        self.mix_weight_cap = cfg.data.mix_weight_max
        self.mix_variance_floor = cfg.data.mix_variance_floor
        total_span = max(1, self.mix_warmup_steps + self.mix_ramp_steps)
        warmup_steps = int(total_span * cfg.train.m3_warmup_frac)
        self.m3_warmup_steps = max(1, warmup_steps)
        detach_steps = int(total_span * cfg.train.m3_detach_warmup_epochs / self.total_epochs)
        self.m3_detach_steps = max(1, detach_steps if detach_steps > 0 else self.m3_warmup_steps)
        self.m3_detach_m2 = cfg.train.m3_detach_m2
        self.m3_lambda_resid = cfg.train.m3_lambda_resid
        self.m3_lambda_gate = cfg.train.m3_lambda_gate_entropy
        self.m3_lambda_keep = cfg.train.m3_lambda_keep_target
        self.m3_gate_keep_threshold = cfg.train.m3_gate_keep_threshold
        self.m3_quantile_keep = cfg.train.m3_quantile_keep
        # Optional quantile warmup and minimum keep enforcement for M3
        self.m3_quantile_warmup_steps = getattr(cfg.train, "m3_quantile_warmup_steps", 0)
        self.m3_min_keep = getattr(cfg.train, "m3_min_keep", 0.0)
        keep_steps = int(total_span * cfg.train.m3_keep_warmup_epochs / self.total_epochs)
        self.m3_keep_schedule_steps = max(1, keep_steps if keep_steps > 0 else self.m3_warmup_steps)
        self.m3_target_keep_start = cfg.train.m3_target_keep_start
        self.m3_target_keep_end = cfg.train.m3_target_keep_end
        if self.m3_enabled:
            quantile_str = (
                f"{self.m3_quantile_keep:.2f}" if self.m3_quantile_keep is not None else "None"
            )
            logging.info(
                "[Stage-1][M3] resolved schedule: epochs=%d warmup_steps=%d detach_steps=%d keep_schedule_steps=%d gain_ramp_steps=%d",
                self.total_epochs,
                self.m3_warmup_steps,
                self.m3_detach_steps,
                self.m3_keep_schedule_steps,
                self.m3_gain_ramp_steps,
            )
            logging.info(
                "[Stage-1][M3] controls: gate_mode=%s gate_tau=%.2f->%.2f ramp_steps=%d keep_threshold=%.3f quantile_keep=%s gain_start=%.3f gain_end=%.3f delta_max_deg=%.2f",
                cfg.train.m3_gate_mode,
                self.m3_gate_tau,
                self.m3_tau_end,
                self.m3_tau_ramp_steps,
                self.m3_gate_keep_threshold,
                quantile_str,
                self.m3_gain_start,
                self.m3_gain_end,
                cfg.train.m3_delta_max_deg,
            )

    def _m3_tau(self) -> float:
        start = float(self.m3_gate_tau)
        end = float(self.m3_tau_end)
        steps = float(max(1, self.m3_tau_ramp_steps))
        if self.m3_tau_ramp_steps <= 0:
            return start
        t = min(1.0, max(0.0, self.global_step / steps))
        return float(start + (end - start) * t)

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
        """??�� �̴Ϲ�ġ ??��."""

        self.train(True)
        batch = self._prepare_batch(batch)
        z5d = batch["z5d"]  # torch.FloatTensor[B,5]
        theta_gt = batch["theta_gt"]  # torch.FloatTensor[B,2]
        c_meas = batch["c_meas"]  # torch.FloatTensor[B,2] ???? dB
        if self.cfg.data.input_scale != "relative_db" and "c_meas_abs" in batch:
            c_meas = batch["c_meas_abs"]
        c_meas_rel = batch.get("c_meas_rel")
        if c_meas_rel is not None and c_meas_rel.numel() == 0:
            c_meas_rel = None
        four_rss = batch["four_rss"]  # torch.FloatTensor[B,4] ??�� [B,0]
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
        # Update LR schedule per step
        self._update_lrs()
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
        m3_stats: Optional[Dict[str, Optional[float]]] = None
        mu_eval = mu0
        decoder_recon_mae_4rss: Optional[float] = None
        decoder_recon_mae_4rss_p90: Optional[float] = None
        decoder_mae_samples: Optional[torch.Tensor] = None
        phi_gate_keep_ratio: Optional[float] = None
        phi_gate_threshold: Optional[float] = None
        phi_gate_mean: Optional[float] = None
        phi_gate_std: Optional[float] = None
        phi_gate_p25: Optional[float] = None
        phi_gate_p75: Optional[float] = None
        phi_gate_quantile_disabled: Optional[bool] = None
        phi_gate_ramp: Optional[float] = None
        forward_consistency: Optional[float] = None
        recon_mae_theta_corr: Optional[float] = None
        if enable_pass1:
            h5 = self.e5(z5d)  # torch.FloatTensor[B,H]
            r4_hat = self.decoder(h5.detach(), mu0)  # torch.FloatTensor[B,4]
            h4_hat = self.e4(r4_hat.detach())  # torch.FloatTensor[B,H]
            batch_size = z5d.size(0)
            ones_mask = torch.ones(batch_size, 1, device=self.device, dtype=h5.dtype)
            r4_gt = four_rss if four_rss.numel() == batch_size * 4 else None
            phi_gate = torch.ones(batch_size, 1, device=self.device, dtype=h5.dtype)
            if r4_gt is not None:
                decoder_mae_samples = torch.abs(r4_hat.detach() - r4_gt.detach()).mean(dim=-1)
                decoder_recon_mae_4rss = decoder_mae_samples.mean().item()
                decoder_recon_mae_4rss_p90 = torch.quantile(decoder_mae_samples, 0.90).item()
                decoder_recon_mae_4rss_std = decoder_mae_samples.std(unbiased=False).item() if decoder_mae_samples.numel() > 1 else 0.0
                decoder_recon_mae_4rss_min = decoder_mae_samples.min().item()
                decoder_recon_mae_4rss_max = decoder_mae_samples.max().item()
                # EWM standardization so absolute threshold is interpreted as alpha in z-units
                ramp_val = self._phi_gate_ramp()
                if self._phi_ewm_mu is None:
                    self._phi_ewm_mu = float(decoder_recon_mae_4rss)
                    self._phi_ewm_var = float(max(1e-6, decoder_recon_mae_4rss_std ** 2))
                else:
                    mu = self._phi_ewm_mu
                    var = self._phi_ewm_var
                    mu_new = self._ema_alpha * mu + (1.0 - self._ema_alpha) * float(decoder_recon_mae_4rss)
                    diff = float(decoder_recon_mae_4rss) - mu
                    var_new = self._ema_alpha * var + (1.0 - self._ema_alpha) * (diff * diff)
                    self._phi_ewm_mu = mu_new
                    self._phi_ewm_var = max(1e-9, var_new)
                mu_ewm = torch.as_tensor(self._phi_ewm_mu, dtype=decoder_mae_samples.dtype, device=decoder_mae_samples.device)
                std_ewm = torch.sqrt(torch.as_tensor(self._phi_ewm_var, dtype=decoder_mae_samples.dtype, device=decoder_mae_samples.device))
                z_scores = (decoder_mae_samples - mu_ewm) / (std_ewm + 1e-6)
                thr_alpha = self.cfg.train.phi_gate_threshold if self.cfg.train.phi_gate_threshold is not None else None
                use_quantile = thr_alpha is None
                scores = decoder_mae_samples if use_quantile else z_scores
                gate_flat, phi_gate_keep_ratio, phi_gate_threshold, phi_stats = compute_phi_gate(
                    scores,
                    enabled=self.cfg.train.phi_gate_enabled,
                    threshold=thr_alpha,
                    quantile=self.cfg.train.phi_gate_quantile,
                    min_keep=self.cfg.train.phi_gate_min_keep,
                    ramp=ramp_val,
                )
                phi_gate = gate_flat.unsqueeze(-1).to(dtype=h5.dtype)
                phi_gate_mean = float(phi_stats.get("gate_mean", 0.0))
                phi_gate_std = float(phi_stats.get("gate_std", 0.0))
                phi_gate_p25 = float(phi_stats.get("gate_p25", 0.0))
                phi_gate_p75 = float(phi_stats.get("gate_p75", 0.0))
                phi_gate_quantile_disabled = bool(phi_stats.get("quantile_disabled", False))
                phi_gate_ramp = float(phi_stats.get("ramp", 0.0))
            elif self.cfg.train.phi_gate_enabled:
                phi_gate_keep_ratio = 1.0
            h_pass0 = self.fuse(h5, h4_hat, ones_mask)
            phi_pass0 = self.adapter(h_pass0).detach()
            h = self.fuse(h5, h4_hat, phi_gate)
            phi = (self.adapter(h) * phi_gate).detach()
            mu1, kappa1, _ = self.m2(z5d, phi)
            mu1 = _ensure_two_dim(mu1)
            kappa1 = _ensure_two_dim(kappa1)
            mu_eval = mu1
            phi_gt = None
            if r4_gt is not None:
                h4_gt = self.e4(r4_gt.detach())
                h_gt = self.fuse(h5, h4_gt, ones_mask)
                phi_gt = self.adapter(h_gt).detach()
                forward_consistency = F.cosine_similarity(phi_pass0, phi_gt, dim=-1).mean().item()
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
                mu_pre = mu1.detach()
                # Update M3 gate temperature by schedule before forward
                self.m3.gate_tau = float(self._m3_tau())
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
                    spread = (torch.quantile(gate.detach(), 0.90) - torch.quantile(gate.detach(), 0.10)).item()
                    if self.global_step < max(0, self.m3_quantile_warmup_steps) or spread < 1e-3:
                        keep_mask_bool = torch.ones_like(gate, dtype=torch.bool)
                        threshold_used = 0.0
                    else:
                        q = float(min(max(self.m3_quantile_keep, 0.0), 1.0))
                        quantile_value = torch.quantile(gate.detach(), q)
                        threshold_used = quantile_value.item()
                        keep_mask_bool = gate >= quantile_value
                    # Enforce minimum keep ratio
                    if self.m3_min_keep > 0.0:
                        keep_mean_now = keep_mask_bool.float().mean().item()
                        if keep_mean_now < self.m3_min_keep:
                            k = max(1, int(math.ceil(self.m3_min_keep * gate.numel())))
                            thr_relax = torch.topk(gate.flatten(), k, largest=True).values.min()
                            keep_mask_bool = gate >= thr_relax
                            threshold_used = float(thr_relax.item())
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
                    gate_std = gate.std(unbiased=False).item() if gate.numel() > 1 else 0.0
                    gate_p25 = torch.quantile(gate, 0.25).item()
                    gate_p75 = torch.quantile(gate, 0.75).item()
                    keep_ratio = keep_mask.mean().item()
                    resid_abs_deg = torch.rad2deg(delta_effect.abs())
                    resid_mean_deg = resid_abs_deg.mean().item()
                    resid_p90_deg = torch.quantile(resid_abs_deg, 0.90).item()
                    resid_p99_deg = torch.quantile(resid_abs_deg, 0.99).item()
                    clip_rate = (clip_mask & (keep_mask.bool())).float().mean().item()
                    err_pre = torch.atan2(torch.sin(mu_pre - theta_gt), torch.cos(mu_pre - theta_gt))
                    err_post = torch.atan2(torch.sin(mu_eval - theta_gt), torch.cos(mu_eval - theta_gt))
                    err_pre_deg = torch.rad2deg(err_pre.abs()).mean(dim=-1)
                    err_post_deg = torch.rad2deg(err_post.abs()).mean(dim=-1)
                    improvement = err_pre_deg - err_post_deg
                    improve_rate = (improvement >= 1.0).float().mean().item()
                    worsen_rate = (improvement <= -1.0).float().mean().item()
                    err_delta_mean = (err_post_deg - err_pre_deg).mean().item()
                    direction_agree = torch.cos(delta_effect - err_pre)
                    direction_agree_rate = (direction_agree > 0).float().mean().item()
                    kappa_sample = kappa1.mean(dim=-1)
                    kappa_corr = self._spearman_corr(kappa_sample.detach(), err_pre_deg.detach())
                    phi_corr = None
                    if decoder_mae_samples is not None and decoder_mae_samples.numel() == improvement.numel():
                        phi_quality = 1.0 / (1.0 + decoder_mae_samples.detach())
                        phi_corr = self._spearman_corr(phi_quality, improvement.detach())
                    # EMA updates for correlations
                    if kappa_corr is not None:
                        if self._kappa_corr_ema is None:
                            self._kappa_corr_ema = float(kappa_corr)
                        else:
                            self._kappa_corr_ema = float(self._ema_alpha * self._kappa_corr_ema + (1.0 - self._ema_alpha) * float(kappa_corr))
                    if phi_corr is not None:
                        if self._phi_corr_ema is None:
                            self._phi_corr_ema = float(phi_corr)
                        else:
                            self._phi_corr_ema = float(self._ema_alpha * self._phi_corr_ema + (1.0 - self._ema_alpha) * float(phi_corr))
                    m3_stats = {
                        "enabled": 1.0,
                        "gate_mean": gate_mean,
                        "gate_p10": gate_p10,
                        "gate_p90": gate_p90,
                        "gate_std": gate_std,
                        "gate_p25": gate_p25,
                        "gate_p75": gate_p75,
                        "quantile_disabled": bool(gate_std < 0.05 or ramp < 0.1),
                        "ramp": float(ramp),
                        "keep_ratio": keep_ratio,
                        "resid_mean_deg": resid_mean_deg,
                        "resid_p90_deg": resid_p90_deg,
                        "resid_p99_deg": resid_p99_deg,
                        "clip_rate": clip_rate,
                        "kappa_corr": kappa_corr,
                        "resid_penalty": m3_resid_penalty.detach().item(),
                        "gate_penalty": m3_gate_penalty.detach().item(),
                        "gate_threshold": threshold_used,
                        "gate_entropy": gate_entropy.detach().item(),
                        "delta_cap_deg": math.degrees(delta_cap),
                        "gain": gain,
                        "gate_tau": float(self._m3_tau()),
                        "resid_dir_agree_rate": direction_agree_rate,
                        "improve_rate_1deg": improve_rate,
                        "worsen_rate_1deg": worsen_rate,
                        "err_delta_mean_deg": err_delta_mean,
                        "phi_quality_corr": phi_corr,
                    }
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
        # Optional targeted clipping for M3 only
        if self.m3 is not None and getattr(self, "m3_grad_clip", None) not in (None, 0.0):
            torch.nn.utils.clip_grad_norm_(self.m3.parameters(), max_norm=float(self.m3_grad_clip))
        grad_norm_m3: Optional[float] = None
        if self.m3 is not None:
            total_sq = 0.0
            for param in self.m3.parameters():
                if param.grad is not None:
                    total_sq += float(param.grad.detach().pow(2).sum().item())
            if total_sq > 0.0:
                grad_norm_m3 = math.sqrt(total_sq)
            else:
                grad_norm_m3 = 0.0
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
        # Store last batch predictions/room_dim for plotting (Stage-1 only)
        self.last_theta_pred = mu_eval.detach().cpu()
        self.last_theta_gt = theta_gt.detach().cpu()
        try:
            z5d_named_batch = batch.get("z5d_named")
            if isinstance(z5d_named_batch, dict) and ("room_dim_meter" in z5d_named_batch):
                self.last_room_dim = z5d_named_batch["room_dim_meter"].detach().cpu()
        except Exception:
            pass
        if decoder_mae_samples is not None:
            err = torch.atan2(torch.sin(mu_eval.detach() - theta_gt.detach()), torch.cos(mu_eval.detach() - theta_gt.detach()))
            err_deg = torch.rad2deg(err.abs()).mean(dim=-1)
            recon_mae_theta_corr = self._spearman_corr(decoder_mae_samples.detach(), err_deg.detach())
        if m3_stats is None and self.m3_enabled:
            m3_stats = {
                "enabled": 1.0,
                "gate_mean": 0.0,
                "gate_p10": 0.0,
                "gate_p90": 0.0,
                "gate_std": 0.0,
                "gate_p25": 0.0,
                "gate_p75": 0.0,
                "quantile_disabled": False,
                "ramp": 0.0,
                "keep_ratio": 0.0,
                "resid_mean_deg": 0.0,
                "resid_p90_deg": 0.0,
                "clip_rate": 0.0,
                "kappa_corr": 0.0,
                "resid_penalty": m3_resid_penalty.detach().item(),
                "gate_penalty": m3_gate_penalty.detach().item(),
                "gate_threshold": float(self.m3_gate_keep_threshold),
                "gate_entropy": 0.0,
                "delta_cap_deg": math.degrees(self.m3_delta_warmup_rad),
                "gain": self.m3_gain_start,
                "gate_tau": self.m3_gate_tau,
                "resid_dir_agree_rate": 0.0,
                "improve_rate_1deg": 0.0,
                "worsen_rate_1deg": 0.0,
                "err_delta_mean_deg": 0.0,
                "phi_quality_corr": None,
            }
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
            m3_enabled=m3_stats["enabled"] if m3_stats is not None else None,
            m3_gate_mean=m3_stats["gate_mean"] if m3_stats is not None else None,
            m3_gate_p10=m3_stats["gate_p10"] if m3_stats is not None else None,
            m3_gate_p90=m3_stats["gate_p90"] if m3_stats is not None else None,
            m3_keep_ratio=m3_stats["keep_ratio"] if m3_stats is not None else None,
            m3_resid_abs_mean_deg=m3_stats["resid_mean_deg"] if m3_stats is not None else None,
            m3_resid_abs_p90_deg=m3_stats["resid_p90_deg"] if m3_stats is not None else None,
            m3_delta_clip_rate=m3_stats["clip_rate"] if m3_stats is not None else None,
            m3_kappa_corr_spearman=m3_stats["kappa_corr"] if m3_stats is not None else None,
            m3_residual_penalty=m3_stats["resid_penalty"] if m3_stats is not None else None,
            m3_gate_entropy=m3_stats["gate_entropy"] if m3_stats is not None else None,
            m3_gate_threshold=m3_stats["gate_threshold"] if m3_stats is not None else None,
            m3_gate_temp_tau=m3_stats["gate_tau"] if m3_stats is not None else None,
            m3_delta_cap_deg_effective=m3_stats["delta_cap_deg"] if m3_stats is not None else None,
            m3_gain_applied=m3_stats["gain"] if m3_stats is not None else None,
            m3_resid_dir_agree_rate=m3_stats["resid_dir_agree_rate"] if m3_stats is not None else None,
            m3_improve_rate_1deg=m3_stats["improve_rate_1deg"] if m3_stats is not None else None,
            m3_worsen_rate_1deg=m3_stats["worsen_rate_1deg"] if m3_stats is not None else None,
            m3_err_delta_mean_deg=m3_stats["err_delta_mean_deg"] if m3_stats is not None else None,
            phi_quality_improve_corr=m3_stats["phi_quality_corr"] if m3_stats is not None else None,
            decoder_recon_mae_4rss=decoder_recon_mae_4rss,
            decoder_recon_mae_4rss_p90=decoder_recon_mae_4rss_p90,
            phi_gate_keep_ratio=phi_gate_keep_ratio,
            phi_gate_threshold=phi_gate_threshold,
            phi_gate_mean=phi_gate_mean,
            phi_gate_std=phi_gate_std,
            phi_gate_p25=phi_gate_p25,
            phi_gate_p75=phi_gate_p75,
            phi_gate_quantile_disabled=phi_gate_quantile_disabled,
            phi_gate_ramp=phi_gate_ramp,
            recon_mae_theta_corr=recon_mae_theta_corr,
            forward_consistency=forward_consistency,
            grad_norm_m3=grad_norm_m3,
            m3_gate_std=m3_stats["gate_std"] if m3_stats is not None else None,
            m3_gate_p25=m3_stats["gate_p25"] if m3_stats is not None else None,
            m3_gate_p75=m3_stats["gate_p75"] if m3_stats is not None else None,
            m3_quantile_disabled=m3_stats["quantile_disabled"] if m3_stats is not None else None,
            m3_ramp=m3_stats["ramp"] if m3_stats is not None else None,
        )
        return result

    def modules(self) -> Dict[str, nn.Module]:
        """���� ??��."""

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

    def _phi_gate_ramp(self) -> float:
        """Ramp for phi-gate tied to mixing schedule (0..1)."""
        warmup = self.mix_warmup_steps
        ramp = max(1, self.mix_ramp_steps)
        if self.global_step < warmup:
            return 0.0
        progress = min(1.0, (self.global_step - warmup) / ramp)
        return float(progress)

    def _freeze_batchnorm(self) -> None:
        for module in [self.e4, self.e5, self.fuse, self.adapter, self.decoder, self.m2, self.m3]:
            if module is None:
                continue
            for m in module.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

    def _update_lrs(self) -> None:
        if not self.use_cosine_lr:
            return
        # Step-based cosine schedule tied to mix schedule span
        total_span = max(1, self.mix_warmup_steps + self.mix_ramp_steps)
        step = self.global_step
        if self.lr_warmup_steps > 0 and step < self.lr_warmup_steps:
            factor = step / float(max(1, self.lr_warmup_steps))
        else:
            # Cosine from 1.0 -> lr_min_ratio over total_span
            t = min(1.0, step / float(total_span))
            cosine = 0.5 * (1 + math.cos(math.pi * t))
            factor = self.lr_min_ratio + (1 - self.lr_min_ratio) * cosine
        for (pg, base) in zip(self.optimizer.param_groups, self._base_lrs):
            pg["lr"] = max(1e-8, float(base) * float(factor))


__all__ = ["Stage1Trainer", "Stage1Outputs"]
