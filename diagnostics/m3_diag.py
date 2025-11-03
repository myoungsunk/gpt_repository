"""Quick diagnostics for M3 residual calibrator."""
from __future__ import annotations

import argparse
from pathlib import Path
import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch

from rss_da.config import Config
from rss_da.train import _build_dataloader
from rss_da.training.stage1 import Stage1Trainer, _ensure_two_dim
from rss_da.training.stage25 import Stage25Trainer


def _prepare_batch(trainer: Stage1Trainer, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return trainer._prepare_batch(batch)  # type: ignore[attr-defined]


def _error_stats(mu: torch.Tensor, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    diff = torch.atan2(torch.sin(mu - theta), torch.cos(mu - theta)).abs()
    err_deg = torch.rad2deg(diff).mean(dim=-1)
    return diff, err_deg


def _reliability_bins(kappa: torch.Tensor, err_deg: torch.Tensor, bins: int = 5) -> Iterable[Tuple[float, float, float]]:
    kappa = kappa.detach()
    err_deg = err_deg.detach()
    quantiles = torch.linspace(0, 1, bins + 1, device=kappa.device)
    edges = torch.quantile(kappa, quantiles)
    results = []
    for idx in range(bins):
        lower = edges[idx]
        upper = edges[idx + 1]
        if idx == bins - 1:
            mask = (kappa >= lower) & (kappa <= upper)
        else:
            mask = (kappa >= lower) & (kappa < upper)
        if mask.any():
            mean_err = err_deg[mask].mean().item()
        else:
            mean_err = float("nan")
        results.append((lower.item(), upper.item(), mean_err))
    return results


def _summarize_m3(
    gate: torch.Tensor,
    keep_mask: torch.Tensor,
    delta_effect: torch.Tensor,
    clip_mask: torch.Tensor,
    theta: torch.Tensor,
    mu: torch.Tensor,
    kappa: torch.Tensor,
    corr_fn,
    threshold: float,
) -> Dict[str, float]:
    err_rad, err_deg = _error_stats(mu, theta)
    stats = {
        "gate_mean": gate.mean().item(),
        "gate_p10": torch.quantile(gate, 0.10).item(),
        "gate_p90": torch.quantile(gate, 0.90).item(),
        "keep_ratio": keep_mask.float().mean().item(),
        "resid_abs_mean_deg": torch.rad2deg(delta_effect.abs()).mean().item(),
        "resid_abs_p90_deg": torch.quantile(torch.rad2deg(delta_effect.abs()), 0.90).item(),
        "delta_clip_rate": (clip_mask & keep_mask.bool()).float().mean().item(),
        "gate_threshold": threshold,
    }
    kappa_sample = kappa.mean(dim=-1)
    kappa_corr = corr_fn(kappa_sample, err_deg)
    stats["kappa_corr"] = kappa_corr
    bins = _reliability_bins(kappa_sample, err_deg)
    stats["reliability_bins"] = bins  # type: ignore[assignment]
    return stats


def analyze_stage1(
    trainer: Stage1Trainer,
    batch: Dict[str, torch.Tensor],
    gate_threshold: float,
    quantile_keep: Optional[float],
) -> None:
    trainer.global_step = trainer.m3_warmup_steps + trainer.mix_ramp_steps + 1
    prepared = _prepare_batch(trainer, batch)
    z5d = prepared["z5d"]
    theta_gt = prepared["theta_gt"]
    c_meas = prepared["c_meas"]
    c_meas_rel = prepared.get("c_meas_rel")
    if c_meas_rel is not None and c_meas_rel.numel() == 0:
        c_meas_rel = None
    four_rss = prepared["four_rss"]
    with torch.no_grad():
        mu0, kappa0, _ = trainer.m2(z5d, None)
        mu0 = _ensure_two_dim(mu0)
        kappa0 = _ensure_two_dim(kappa0)
        h5 = trainer.e5(z5d)
        r4_hat = trainer.decoder(h5.detach(), mu0)
        h4 = trainer.e4(r4_hat.detach())
        mask = torch.zeros(z5d.size(0), 1, device=z5d.device)
        fused = trainer.fuse(h5, h4, mask)
        phi = trainer.adapter(fused).detach()
        mu1, kappa1, _ = trainer.m2(z5d, phi)
        mu1 = _ensure_two_dim(mu1)
        kappa1 = _ensure_two_dim(kappa1)
        if trainer.m3_enabled and trainer.m3 is not None:
            c_feat = c_meas_rel if c_meas_rel is not None else c_meas
            features = trainer._assemble_m3_features(z5d, phi, c_feat, four_rss, mu1, kappa1)
            m3_out = trainer.m3(
                features,
                mu1,
                kappa1,
                ramp=1.0,
                delta_max=trainer.m3_delta_max_rad,
                gain=1.0,
                gate_threshold=gate_threshold,
                extras={},
            )
            mu_ref = m3_out["mu_ref"]
            gate = m3_out["gate"]
            keep_mask = m3_out["keep_mask"]
            clip_mask = m3_out["clip_mask"]
            delta_effect = m3_out["delta_effect"]
            threshold_used = gate_threshold
            if quantile_keep is not None and gate.numel() > 0:
                q = float(min(max(quantile_keep, 0.0), 1.0))
                threshold_tensor = torch.quantile(gate.detach(), q)
                threshold_used = threshold_tensor.item()
                keep_mask = gate >= threshold_tensor
                clip_mask = keep_mask & (m3_out["delta"].abs() >= max(0.0, trainer.m3_delta_max_rad - 1e-6))
            stats = _summarize_m3(
                gate,
                keep_mask,
                delta_effect,
                clip_mask,
                theta_gt,
                mu_ref,
                kappa1,
                Stage1Trainer._spearman_corr,
                threshold_used,
            )
            print("[Stage-1] M3 metrics:")
            print(
                "  gate_mean={:.4f} keep_ratio={:.4f} resid_mean_deg={:.4f} clip_rate={:.4f} kappa_corr={:.4f}".format(
                    stats["gate_mean"],
                    stats["keep_ratio"],
                    stats["resid_abs_mean_deg"],
                    stats["delta_clip_rate"],
                    stats["kappa_corr"],
                )
            )
            print("  gate_p10={:.4f} gate_p90={:.4f} resid_p90_deg={:.4f} gate_thresh={:.4f}".format(
                stats["gate_p10"],
                stats["gate_p90"],
                stats["resid_abs_p90_deg"],
                stats["gate_threshold"],
            ))
            print("  reliability bins (kappa_low, kappa_high, mean_abs_err_deg):")
            for low, high, err in stats["reliability_bins"]:
                print("    [{:.3f}, {:.3f}) -> {:.4f}".format(low, high, err))
        else:
            print("[Stage-1] M3 disabled; no diagnostics available.")


def analyze_stage25(
    trainer: Stage25Trainer,
    batch: Dict[str, torch.Tensor],
    gate_threshold: float,
    quantile_keep: Optional[float],
) -> None:
    trainer.global_step = trainer.m3_warmup_steps + trainer.mix_ramp_steps + 1
    prepared = trainer._prepare_batch(batch)  # type: ignore[attr-defined]
    z5d = prepared["z5d"]
    theta_gt = prepared["theta_gt"]
    c_meas = prepared["c_meas"]
    c_meas_rel = prepared.get("c_meas_rel")
    if c_meas_rel is not None and c_meas_rel.numel() == 0:
        c_meas_rel = None
    four_rss = prepared.get("four_rss", torch.empty(0, device=z5d.device))
    with torch.no_grad():
        mu0, kappa0, _ = trainer.m2(z5d, None)
        mu0 = _ensure_two_dim(mu0)
        kappa0 = _ensure_two_dim(kappa0)
        h5 = trainer.e5(z5d)
        r4_hat = trainer.decoder(h5.detach(), mu0)
        h4 = trainer.e4(r4_hat.detach())
        mask = torch.zeros(z5d.size(0), 1, device=z5d.device)
        fused = trainer.fuse(h5, h4, mask)
        phi = trainer.adapter(fused).detach()
        mu1, kappa1, _ = trainer.m2(z5d, phi)
        mu1 = _ensure_two_dim(mu1)
        kappa1 = _ensure_two_dim(kappa1)
        if trainer.m3_enabled and trainer.m3 is not None:
            c_feat = c_meas_rel if c_meas_rel is not None else c_meas
            features = trainer._assemble_m3_features(z5d, phi, c_feat, four_rss, mu1, kappa1)
            m3_out = trainer.m3(
                features,
                mu1,
                kappa1,
                ramp=1.0,
                delta_max=trainer.m3_delta_max_rad,
                gain=1.0,
                gate_threshold=gate_threshold,
                extras={},
            )
            mu_ref = m3_out["mu_ref"]
            gate = m3_out["gate"]
            keep_mask = m3_out["keep_mask"]
            clip_mask = m3_out["clip_mask"]
            delta_effect = m3_out["delta_effect"]
            threshold_used = gate_threshold
            if quantile_keep is not None and gate.numel() > 0:
                q = float(min(max(quantile_keep, 0.0), 1.0))
                threshold_tensor = torch.quantile(gate.detach(), q)
                threshold_used = threshold_tensor.item()
                keep_mask = gate >= threshold_tensor
                clip_mask = keep_mask & (m3_out["delta"].abs() >= max(0.0, trainer.m3_delta_max_rad - 1e-6))
            stats = _summarize_m3(
                gate,
                keep_mask,
                delta_effect,
                clip_mask,
                theta_gt,
                mu_ref,
                kappa1,
                trainer._spearman_corr,
                threshold_used,
            )
            print("[Stage-2.5] M3 metrics:")
            print(
                "  gate_mean={:.4f} keep_ratio={:.4f} resid_mean_deg={:.4f} clip_rate={:.4f} kappa_corr={:.4f}".format(
                    stats["gate_mean"],
                    stats["keep_ratio"],
                    stats["resid_abs_mean_deg"],
                    stats["delta_clip_rate"],
                    stats["kappa_corr"],
                )
            )
            print("  gate_p10={:.4f} gate_p90={:.4f} resid_p90_deg={:.4f} gate_thresh={:.4f}".format(
                stats["gate_p10"],
                stats["gate_p90"],
                stats["resid_abs_p90_deg"],
                stats["gate_threshold"],
            ))
            print("  reliability bins (kappa_low, kappa_high, mean_abs_err_deg):")
            for low, high, err in stats["reliability_bins"]:
                print("    [{:.3f}, {:.3f}) -> {:.4f}".format(low, high, err))
        else:
            print("[Stage-2.5] M3 disabled; no diagnostics available.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M3 diagnostics")
    parser.add_argument("--stage", choices=["1", "2.5"], default="1")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--stage1_file", type=str, default="stage1_rel.csv")
    parser.add_argument("--stage25_file", type=str, default="stage25_rel.csv")
    parser.add_argument("--logdir", type=str, default="./runs/demo")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--enable_m3", action="store_true")
    parser.add_argument("--m3_gate_mode", choices=["none", "kappa", "inv_kappa", "mcdrop"], default="kappa")
    parser.add_argument("--m3_gate_threshold", type=float, default=0.5)
    parser.add_argument("--m3_gate_temp", type=float, default=1.0)
    parser.add_argument("--m3_quantile_keep", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()
    cfg.train.use_m3 = args.enable_m3
    cfg.train.m3_gate_mode = args.m3_gate_mode
    cfg.train.m3_gate_tau = args.m3_gate_temp
    cfg.train.m3_gate_keep_threshold = args.m3_gate_threshold
    cfg.train.m3_quantile_keep = args.m3_quantile_keep
    cfg.train.log_dir = args.logdir
    cfg.data.roots = [args.data_root]
    cfg.data.input_scale = "relative_db"
    data_root = Path(args.data_root)
    scaler_root = Path(args.logdir) / cfg.data.scaler_dir / "stage1" / "z5d_scaler.pkl"
    if args.stage == "1":
        loader = _build_dataloader(
            "1",
            args.batch_size,
            data_root,
            args.stage1_file,
            args.stage25_file,
            cfg,
            scaler_root,
            fit_scaler=False,
        )
        trainer = Stage1Trainer(cfg)
        trainer.train(False)
        batch = next(iter(loader))
        analyze_stage1(trainer, batch, args.m3_gate_threshold, args.m3_quantile_keep)
    else:
        scaler_path = scaler_root if scaler_root.exists() else None
        loader = _build_dataloader(
            "2.5",
            args.batch_size,
            data_root,
            args.stage1_file,
            args.stage25_file,
            cfg,
            scaler_path,
            fit_scaler=False,
        )
        trainer = Stage25Trainer(cfg, teacher_modules=None)
        trainer.train(False)
        batch = next(iter(loader))
        analyze_stage25(trainer, batch, args.m3_gate_threshold, args.m3_quantile_keep)


if __name__ == "__main__":
    main()
