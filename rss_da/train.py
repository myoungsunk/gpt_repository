"""엔드투엔드 학습 스크립트."""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from torch.utils.data import DataLoader

from rss_da.config import Config
from rss_da.data.dataset import RssDoADataset, Sample, collate_samples, load_standardized_csv
from rss_da.data.synth_generator import SynthConfig, build_samples, generate_synth_samples
from rss_da.training.stage1 import Stage1Trainer
from rss_da.training.stage25 import Stage25Trainer
from rss_da.utils.log import Logger
from rss_da.utils.seed import set_seed


def _create_samples(stage: str, num_samples: int) -> Iterable[Sample]:
    theta, arrays = generate_synth_samples(SynthConfig(num_samples=num_samples))
    samples = list(build_samples(arrays))
    if stage == "2.5":
        updated = []
        for sample in samples:
            updated.append(
                Sample(
                    z5d=sample.z5d,
                    c_meas=sample.c_meas,
                    theta_gt=sample.theta_gt,
                    four_rss=None,
                    mask_4rss_is_gt=0.0,
                )
            )
        samples = updated
    return samples


def _load_samples_from_root(
    stage: str,
    data_root: Path,
    stage1_file: str,
    stage25_file: str,
) -> Optional[List[Sample]]:
    """data_root에서 표준화 CSV를 찾는다."""

    if stage == "1":
        filename = stage1_file
    else:
        filename = stage25_file
    candidate = data_root / filename
    if not candidate.exists():
        logging.warning("%s not found. Falling back to synthetic samples.", candidate)
        return None
    logging.info("Loading %s dataset from %s", stage, candidate)
    samples = load_standardized_csv(candidate, stage=stage)
    if not samples:
        logging.warning("Dataset %s is empty. Falling back to synthetic samples.", candidate)
        return None
    return samples


def _compute_z5d_stats(samples: List[Sample]) -> Tuple[np.ndarray, np.ndarray]:
    matrix = np.stack([np.asarray(sample.z5d, dtype=np.float32) for sample in samples], axis=0)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def _apply_standardization(
    dataset: RssDoADataset,
    samples: List[Sample],
    scaler_path: Optional[Path],
    fit: bool,
) -> None:
    if fit:
        mean, std = _compute_z5d_stats(samples)
        dataset.set_standardization(mean, std)
        if scaler_path is not None:
            scaler_path.parent.mkdir(parents=True, exist_ok=True)
            with scaler_path.open("wb") as fp:
                pickle.dump({"mean": mean.tolist(), "std": std.tolist()}, fp)
            logging.info("Saved z5d scaler to %s", scaler_path.as_posix())
    else:
        loaded = False
        if scaler_path is not None and scaler_path.exists():
            with scaler_path.open("rb") as fp:
                payload = pickle.load(fp)
            dataset.set_standardization(np.asarray(payload["mean"], dtype=np.float32), np.asarray(payload["std"], dtype=np.float32))
            logging.info("Loaded z5d scaler from %s", scaler_path.as_posix())
            loaded = True
        if not loaded:
            logging.warning("Scaler not found. Fitting on-the-fly for current dataset.")
            mean, std = _compute_z5d_stats(samples)
            dataset.set_standardization(mean, std)


def _build_dataloader(
    stage: str,
    batch_size: int,
    data_root: Path,
    stage1_file: str,
    stage25_file: str,
    cfg: Config,
    scaler_path: Optional[Path],
    fit_scaler: bool,
) -> DataLoader:
    samples = _load_samples_from_root(stage, data_root, stage1_file, stage25_file)
    if samples is None:
        samples = list(_create_samples(stage, max(batch_size * 4, 128)))
        logging.info(
            "Generated synthetic %s dataset with %d samples for batch size %d",
            stage,
            len(samples),
            batch_size,
        )
    else:
        logging.info("Loaded %d samples for stage %s", len(samples), stage)
    dataset = RssDoADataset(list(samples), stage=stage, modality_dropout_p=0.0, training=True)
    _apply_standardization(dataset, samples, scaler_path, fit_scaler)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_samples,
        drop_last=False,
    )
    return loader


def _log_metrics(logger: Logger, step: int, prefix: str, metrics: Dict[str, float]) -> None:
    logger.add_scalars(step, {f"{prefix}/{k}": v for k, v in metrics.items()})


def _stage1_loop(
    cfg: Config,
    epochs: int,
    log_dir: Path,
    data_root: Path,
    stage1_file: str,
    stage25_file: str,
) -> Stage1Trainer:
    trainer = Stage1Trainer(cfg)
    logging.info("Stage-1 trainer initialized on device: %s", trainer.device)
    scaler_path = log_dir / cfg.data.scaler_dir / "stage1" / "z5d_scaler.pkl"
    loader = _build_dataloader(
        "1",
        cfg.train.batch_size,
        data_root,
        stage1_file,
        stage25_file,
        cfg,
        scaler_path,
        fit_scaler=True,
    )
    trainer.scaler_path = scaler_path  # type: ignore[attr-defined]
    logger = Logger(log_dir)
    global_step = 0
    try:
        for epoch in range(epochs):
            logging.info("[Stage-1] Starting epoch %d/%d", epoch + 1, epochs)
            for batch in loader:
                outputs = trainer.train_step(batch, enable_pass1=epoch > 0)
                metrics = {
                    "loss_total": outputs.loss_total,
                    "loss_sup0": outputs.loss_sup0,
                    "loss_sup1": outputs.loss_sup1,
                    "loss_data": outputs.loss_data,
                    "loss_mix": outputs.loss_mix,
                    "loss_phys": outputs.loss_phys,
                    "deg_rmse": outputs.deg_rmse,
                    "kappa_mean": outputs.kappa_mean,
                    "mix_weight": outputs.mix_weight,
                    "mix_weighted": outputs.mix_weighted,
                }
<<<<<<< Updated upstream
=======
                if outputs.decoder_recon_mae_4rss is not None:
                    metrics["decoder_recon_mae_4rss"] = outputs.decoder_recon_mae_4rss
<<<<<<< Updated upstream
                if outputs.decoder_recon_mae_4rss_p90 is not None:
                    metrics["decoder_recon_mae_4rss_p90"] = outputs.decoder_recon_mae_4rss_p90
                if outputs.phi_gate_keep_ratio is not None:
                    metrics["phi_gate_keep_ratio"] = outputs.phi_gate_keep_ratio
                if outputs.recon_mae_theta_corr is not None:
                    metrics["recon_mae_theta_corr"] = outputs.recon_mae_theta_corr
=======
>>>>>>> Stashed changes
                if outputs.m3_enabled is not None:
                    metrics.update(
                        {
                            "m3_enabled": outputs.m3_enabled,
                            "m3_gate_mean": outputs.m3_gate_mean or 0.0,
                            "m3_gate_p10": outputs.m3_gate_p10 or 0.0,
                            "m3_gate_p90": outputs.m3_gate_p90 or 0.0,
                            "m3_keep_ratio": outputs.m3_keep_ratio or 0.0,
                            "m3_resid_abs_mean_deg": outputs.m3_resid_abs_mean_deg or 0.0,
                            "m3_resid_abs_p90_deg": outputs.m3_resid_abs_p90_deg or 0.0,
                            "m3_delta_clip_rate": outputs.m3_delta_clip_rate or 0.0,
                            "m3_kappa_corr_spearman": outputs.m3_kappa_corr_spearman or 0.0,
                            "m3_residual_penalty": outputs.m3_residual_penalty or 0.0,
                            "m3_gate_entropy": outputs.m3_gate_entropy or 0.0,
                            "m3_gate_threshold": outputs.m3_gate_threshold or 0.0,
                            "m3_gate_temp_tau": outputs.m3_gate_temp_tau or 0.0,
                            "m3_delta_cap_deg_effective": outputs.m3_delta_cap_deg_effective or 0.0,
                            "m3_gain_applied": outputs.m3_gain_applied or 0.0,
                            "m3_resid_dir_agree_rate": outputs.m3_resid_dir_agree_rate or 0.0,
                            "m3_improve_rate_1deg": outputs.m3_improve_rate_1deg or 0.0,
                            "m3_worsen_rate_1deg": outputs.m3_worsen_rate_1deg or 0.0,
                            "m3_err_delta_mean_deg": outputs.m3_err_delta_mean_deg or 0.0,
                        }
                    )
                    if outputs.phi_quality_improve_corr is not None:
                        metrics["phi_quality_improve_corr"] = outputs.phi_quality_improve_corr
                    if outputs.grad_norm_m3 is not None:
                        metrics["grad_norm_m3"] = outputs.grad_norm_m3
>>>>>>> Stashed changes
                if global_step % cfg.train.log_interval == 0:
                    _log_metrics(logger, global_step, "train", metrics)
                    logging.info(
                        (
                            "[Stage-1][step=%d] loss=%.4f sup0=%.4f sup1=%.4f data=%.4f mix_raw=%.4f "
                            "mix_w=%.6f mix_weighted=%.4f phys=%.4f degRMSE=%.3f"
                        ),
                        global_step,
                        metrics["loss_total"],
                        metrics["loss_sup0"],
                        metrics["loss_sup1"],
                        metrics["loss_data"],
                        metrics["loss_mix"],
                        metrics["mix_weight"],
                        metrics["mix_weighted"],
                        metrics["loss_phys"],
                        metrics["deg_rmse"],
<<<<<<< Updated upstream
                    )
=======
                    ]
                    if outputs.decoder_recon_mae_4rss is not None:
                        base_msg += " decoder_mae4=%.4f"
                        args_list.append(metrics["decoder_recon_mae_4rss"])
<<<<<<< Updated upstream
                    if outputs.decoder_recon_mae_4rss_p90 is not None:
                        base_msg += " decoder_mae4_p90=%.4f"
                        args_list.append(metrics["decoder_recon_mae_4rss_p90"])
                    if outputs.phi_gate_keep_ratio is not None:
                        base_msg += " phi_keep=%.4f"
                        args_list.append(metrics["phi_gate_keep_ratio"])
                    if outputs.recon_mae_theta_corr is not None:
                        base_msg += " mae_theta_corr=%.4f"
                        args_list.append(metrics["recon_mae_theta_corr"])
=======
>>>>>>> Stashed changes
                    if outputs.m3_enabled is not None:
                        base_msg += (
                            " m3_enabled=%.0f gate_mean=%.4f keep=%.4f gate_p10=%.4f gate_p90=%.4f "
                            "thr=%.4f tau=%.2f gain=%.3f delta_cap=%.2f clip=%.4f gate_ent=%.4f "
                            "resid_mean=%.4f resid_p90=%.4f dir_agree=%.3f impro@1=%.3f "
                            "worsen@1=%.3f err_delta=%.3f kappa_corr=%.4f"
                        )
                        args_list.extend(
                            [
                                metrics["m3_enabled"],
                                metrics["m3_gate_mean"],
                                metrics["m3_keep_ratio"],
                                metrics["m3_gate_p10"],
                                metrics["m3_gate_p90"],
                                metrics["m3_gate_threshold"],
                                metrics["m3_gate_temp_tau"],
                                metrics["m3_gain_applied"],
                                metrics["m3_delta_cap_deg_effective"],
                                metrics["m3_delta_clip_rate"],
                                metrics["m3_gate_entropy"],
                                metrics["m3_resid_abs_mean_deg"],
                                metrics["m3_resid_abs_p90_deg"],
                                metrics["m3_resid_dir_agree_rate"],
                                metrics["m3_improve_rate_1deg"],
                                metrics["m3_worsen_rate_1deg"],
                                metrics["m3_err_delta_mean_deg"],
                                metrics["m3_kappa_corr_spearman"],
                            ]
                        )
                        if outputs.phi_quality_improve_corr is not None:
                            base_msg += " phi_corr=%.4f"
                            args_list.append(metrics["phi_quality_improve_corr"])
                        if outputs.grad_norm_m3 is not None:
                            base_msg += " grad_m3=%.4f"
                            args_list.append(metrics["grad_norm_m3"])
                    logging.info(base_msg, *args_list)
>>>>>>> Stashed changes
                global_step += 1
        return trainer
    finally:
        logger.close()


def _stage25_loop(
    cfg: Config,
    epochs: int,
    log_dir: Path,
    teacher: Stage1Trainer,
    data_root: Path,
    stage1_file: str,
    stage25_file: str,
    scaler_path: Optional[Path],
) -> Stage25Trainer:
    trainer = Stage25Trainer(cfg, teacher_modules=teacher.modules())
    logging.info("Stage-2.5 trainer initialized on device: %s", trainer.device)
    loader = _build_dataloader(
        "2.5",
        cfg.train.batch_size,
        data_root,
        stage1_file,
        stage25_file,
        cfg,
        scaler_path,
        fit_scaler=False,
    )
    logger = Logger(log_dir)
    global_step = 0
    try:
        for epoch in range(epochs):
            logging.info("[Stage-2.5] Starting epoch %d/%d", epoch + 1, epochs)
            for batch in loader:
                outputs = trainer.train_step(batch)
                metrics = {
                    "loss_total": outputs.loss_total,
                    "loss_sup": outputs.loss_sup,
                    "loss_kd": outputs.loss_kd,
                    "loss_data": outputs.loss_data,
                    "loss_mix": outputs.loss_mix,
                    "loss_phys": outputs.loss_phys,
                    "loss_align": outputs.loss_align,
                    "deg_rmse": outputs.deg_rmse,
                    "kappa_mean": outputs.kappa_mean,
                    "mix_weight": outputs.mix_weight,
                    "mix_weighted": outputs.mix_weighted,
                }
<<<<<<< Updated upstream
=======
                if outputs.decoder_recon_mae_4rss is not None:
                    metrics["decoder_recon_mae_4rss"] = outputs.decoder_recon_mae_4rss
                if outputs.decoder_recon_mae_4rss_p90 is not None:
                    metrics["decoder_recon_mae_4rss_p90"] = outputs.decoder_recon_mae_4rss_p90
                if outputs.phi_gate_keep_ratio is not None:
                    metrics["phi_gate_keep_ratio"] = outputs.phi_gate_keep_ratio
                if outputs.recon_mae_theta_corr is not None:
                    metrics["recon_mae_theta_corr"] = outputs.recon_mae_theta_corr
                if outputs.m3_enabled is not None:
                    metrics.update(
                        {
                            "m3_enabled": outputs.m3_enabled,
                            "m3_gate_mean": outputs.m3_gate_mean or 0.0,
                            "m3_gate_p10": outputs.m3_gate_p10 or 0.0,
                            "m3_gate_p90": outputs.m3_gate_p90 or 0.0,
                            "m3_keep_ratio": outputs.m3_keep_ratio or 0.0,
                            "m3_resid_abs_mean_deg": outputs.m3_resid_abs_mean_deg or 0.0,
                            "m3_resid_abs_p90_deg": outputs.m3_resid_abs_p90_deg or 0.0,
                            "m3_delta_clip_rate": outputs.m3_delta_clip_rate or 0.0,
                            "m3_kappa_corr_spearman": outputs.m3_kappa_corr_spearman or 0.0,
                            "m3_residual_penalty": outputs.m3_residual_penalty or 0.0,
                            "m3_gate_entropy": outputs.m3_gate_entropy or 0.0,
                            "m3_gate_threshold": outputs.m3_gate_threshold or 0.0,
                        }
                    )
>>>>>>> Stashed changes
                if global_step % cfg.train.log_interval == 0:
                    _log_metrics(logger, global_step, "train", metrics)
                    logging.info(
                        (
                            "[Stage-2.5][step=%d] loss=%.4f sup=%.4f kd=%.4f data=%.4f mix_raw=%.4f "
                            "mix_w=%.6f mix_weighted=%.4f align=%.4f degRMSE=%.3f"
                        ),
                        global_step,
                        metrics["loss_total"],
                        metrics["loss_sup"],
                        metrics["loss_kd"],
                        metrics["loss_data"],
                        metrics["loss_mix"],
                        metrics["mix_weight"],
                        metrics["mix_weighted"],
                        metrics["loss_align"],
                        metrics["deg_rmse"],
<<<<<<< Updated upstream
=======
                    ]
                    if outputs.decoder_recon_mae_4rss is not None:
                        base_msg += " decoder_mae4=%.4f"
                        args_list.append(metrics.get("decoder_recon_mae_4rss", 0.0))
                    if outputs.decoder_recon_mae_4rss_p90 is not None:
                        base_msg += " decoder_mae4_p90=%.4f"
                        args_list.append(metrics.get("decoder_recon_mae_4rss_p90", 0.0))
                    if outputs.phi_gate_keep_ratio is not None:
                        base_msg += " phi_keep=%.4f"
                        args_list.append(metrics.get("phi_gate_keep_ratio", 0.0))
                    if outputs.recon_mae_theta_corr is not None:
                        base_msg += " mae_theta_corr=%.4f"
                        args_list.append(metrics.get("recon_mae_theta_corr", 0.0))
                    if outputs.m3_enabled is not None:
                        base_msg += (
                            " m3_enabled=%.0f m3_gate_mean=%.4f m3_keep_ratio=%.4f "
                            "m3_resid_abs_mean_deg=%.4f m3_delta_clip_rate=%.4f m3_kappa_corr=%.4f "
                            "m3_gate_thresh=%.4f"
                        )
                        args_list.extend(
                            [
                                metrics["m3_enabled"],
                                metrics["m3_gate_mean"],
                                metrics["m3_keep_ratio"],
                                metrics["m3_resid_abs_mean_deg"],
                                metrics["m3_delta_clip_rate"],
                                metrics["m3_kappa_corr_spearman"],
                                metrics["m3_gate_threshold"],
                            ]
                        )
                    logging.info(base_msg, *args_list)
                global_step += 1
                epoch_deg_sum += outputs.deg_rmse
                epoch_steps += 1
            if epoch_steps > 0:
                mean_deg = epoch_deg_sum / epoch_steps
                if mean_deg < best_metric:
                    best_metric = mean_deg
                    save_checkpoint(best_path, trainer.modules(), epoch)
                    logging.info(
                        "Saved best Stage-2.5 checkpoint to %s (degRMSE=%.4f)",
                        best_path.as_posix(),
                        mean_deg,
>>>>>>> Stashed changes
                    )
                global_step += 1
        return trainer
    finally:
        logger.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RSS DoA Stage-1/Stage-2.5 학습")
    parser.add_argument("--stage", choices=["1", "2.5"], default="1")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--stage1_file", type=str, default="stage1_rel.csv")
    parser.add_argument("--stage25_file", type=str, default="stage25_rel.csv")
    parser.add_argument(
        "--input_scale",
        type=str,
        default="relative_db",
        choices=["relative_db", "absolute_dbm"],
        help="Combined power scale used by datasets",
    )
    parser.add_argument("--mix_warmup_steps", type=int, default=500)
    parser.add_argument("--mix_ramp_steps", type=int, default=1500)
<<<<<<< Updated upstream
    parser.add_argument("--use_m3", action="store_true")
=======
    parser.add_argument("--enable_m3", action="store_true")
    parser.add_argument(
        "--m3_preset",
        type=str,
        choices=["phase1", "phase2", "phase3", "none"],
        default="none",
        help="Preset bundle for M3 fine-tuning hyperparameters",
    )
    parser.add_argument(
        "--m3_gate_mode",
        type=str,
        default=None,
        choices=["none", "kappa", "inv_kappa", "mcdrop"],
        help="Gate mode for residual calibrator",
    )
    parser.add_argument("--m3_delta_max_deg", "--m3_delta_cap_deg", type=float, default=None)
    parser.add_argument("--m3_delta_warmup_deg", type=float, default=None)
    parser.add_argument("--m3_warmup_frac", type=float, default=None)
    parser.add_argument("--m3_output_gain", type=float, default=None)
    parser.add_argument("--m3_gain_start", type=float, default=None)
    parser.add_argument("--m3_gain_end", type=float, default=None)
    parser.add_argument("--m3_gain_ramp_steps", type=int, default=None)
    parser.add_argument("--m3_lambda_resid", "--m3_resid_reg_w", type=float, default=None)
    parser.add_argument("--m3_lambda_gate_entropy", "--m3_entropy_w", type=float, default=None)
    parser.add_argument("--m3_lambda_keep_target", type=float, default=None)
    parser.add_argument("--m3_gate_keep_threshold", "--m3_gate_threshold", type=float, default=None)
    parser.add_argument("--m3_gate_tau", "--m3_gate_temp", type=float, default=None)
    parser.add_argument("--m3_detach_m2", dest="m3_detach_m2", action="store_true", default=None)
    parser.add_argument("--no_m3_detach_m2", dest="m3_detach_m2", action="store_false")
    parser.set_defaults(m3_detach_m2=None)
    parser.add_argument("--m3_detach_warmup_epochs", type=int, default=None)
    parser.add_argument("--m3_keep_warmup_epochs", "--m3_warmup_epochs", type=int, default=None)
    parser.add_argument("--m3_target_keep_start", type=float, default=None)
    parser.add_argument("--m3_target_keep_end", type=float, default=None)
    parser.add_argument("--m3_quantile_keep", type=float, default=None)
    parser.add_argument("--m3_freeze_m2", "--freeze_m2", action="store_true", default=None)
    parser.add_argument("--m3_apply_eval_only", action="store_true")
    parser.add_argument("--phi_gate_disable", action="store_true")
    parser.add_argument("--phi_gate_threshold", type=float, default=None)
    parser.add_argument("--phi_gate_quantile", type=float, default=None)
    parser.add_argument("--phi_gate_min_keep", type=float, default=None)
>>>>>>> Stashed changes
    parser.add_argument("--use_coral", action="store_true")
    parser.add_argument("--use_dann", action="store_true")
    parser.add_argument("--use_cdan", action="store_true")
    parser.add_argument("--gradnorm", action="store_true")
    parser.add_argument("--uncertainty", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--logdir", type=str, default="./runs/demo")
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Console log level (DEBUG, INFO, WARNING, ...)",
    )
    parser.add_argument(
        "--no_file_log",
        action="store_true",
        help="Disable writing detailed logs to <logdir>/train.log",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _setup_logging(log_dir: Path, level: str, disable_file: bool) -> None:
    level_name = level.upper()
    level_value = getattr(logging, level_name, logging.INFO)
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    log_path: Optional[Path] = None
    if not disable_file:
        log_path = log_dir / "train.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=level_value,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.info("Console log level set to %s", level_name)
    if log_path is not None:
        logging.info("File logging enabled at %s", log_path.as_posix())


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cfg = Config()
    cfg.train.stage = args.stage
    cfg.train.use_m3 = args.use_m3
    cfg.train.use_coral = args.use_coral
    cfg.train.use_dann = args.use_dann
    cfg.train.use_cdan = args.use_cdan
    cfg.train.gradnorm = args.gradnorm
    cfg.train.uncertainty_weighting = args.uncertainty
    requested_epochs = max(1, int(args.epochs))
    cfg.train.epochs = requested_epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.log_dir = args.logdir
<<<<<<< Updated upstream
=======
    if args.m3_preset != "none":
        _apply_m3_preset(cfg, args.m3_preset)
        cfg.train.use_m3 = True
    if args.m3_gate_mode is not None:
        cfg.train.m3_gate_mode = args.m3_gate_mode
    if args.m3_delta_max_deg is not None:
        cfg.train.m3_delta_max_deg = args.m3_delta_max_deg
    if args.m3_delta_warmup_deg is not None:
        cfg.train.m3_delta_warmup_deg = args.m3_delta_warmup_deg
    if args.m3_warmup_frac is not None:
        cfg.train.m3_warmup_frac = args.m3_warmup_frac
    if args.m3_detach_m2 is not None:
        cfg.train.m3_detach_m2 = args.m3_detach_m2
    if args.m3_detach_warmup_epochs is not None:
        cfg.train.m3_detach_warmup_epochs = args.m3_detach_warmup_epochs
    if args.m3_freeze_m2 is not None:
        cfg.train.m3_freeze_m2 = args.m3_freeze_m2
    cfg.train.m3_freeze_m2 = cfg.train.m3_freeze_m2 or args.phase == "finetune_m3"
    cfg.train.m3_apply_eval_only = cfg.train.m3_apply_eval_only or args.m3_apply_eval_only
    if args.m3_output_gain is not None:
        cfg.train.m3_output_gain = args.m3_output_gain
    if args.m3_gain_start is not None:
        cfg.train.m3_gain_start = args.m3_gain_start
    if args.m3_gain_end is not None:
        cfg.train.m3_gain_end = args.m3_gain_end
    if args.m3_gain_ramp_steps is not None:
        cfg.train.m3_gain_ramp_steps = max(0, args.m3_gain_ramp_steps)
    if args.m3_lambda_resid is not None:
        cfg.train.m3_lambda_resid = args.m3_lambda_resid
    if args.m3_lambda_gate_entropy is not None:
        cfg.train.m3_lambda_gate_entropy = args.m3_lambda_gate_entropy
    if args.m3_lambda_keep_target is not None:
        cfg.train.m3_lambda_keep_target = args.m3_lambda_keep_target
    if args.m3_gate_keep_threshold is not None:
        cfg.train.m3_gate_keep_threshold = args.m3_gate_keep_threshold
    if args.m3_gate_tau is not None:
        cfg.train.m3_gate_tau = args.m3_gate_tau
    if args.m3_keep_warmup_epochs is not None:
        cfg.train.m3_keep_warmup_epochs = args.m3_keep_warmup_epochs
    if args.m3_target_keep_start is not None:
        cfg.train.m3_target_keep_start = args.m3_target_keep_start
    if args.m3_target_keep_end is not None:
        cfg.train.m3_target_keep_end = args.m3_target_keep_end
    if args.m3_quantile_keep is not None:
        cfg.train.m3_quantile_keep = args.m3_quantile_keep
    if args.phi_gate_disable:
        cfg.train.phi_gate_enabled = False
    if args.phi_gate_threshold is not None:
        cfg.train.phi_gate_threshold = args.phi_gate_threshold
    if args.phi_gate_quantile is not None:
        cfg.train.phi_gate_quantile = args.phi_gate_quantile
    if args.phi_gate_min_keep is not None:
        cfg.train.phi_gate_min_keep = max(0.0, min(1.0, args.phi_gate_min_keep))
    if cfg.train.epochs != requested_epochs:
        logging.warning(
            "TrainConfig epochs mutated from %d to %d after CLI overrides; forcing requested value",
            cfg.train.epochs,
            requested_epochs,
        )
        cfg.train.epochs = requested_epochs
>>>>>>> Stashed changes
    cfg.data.input_scale = args.input_scale
    cfg.data.mix_warmup_steps = args.mix_warmup_steps
    cfg.data.mix_ramp_steps = args.mix_ramp_steps
    data_root = Path(args.data_root)
    if not data_root.exists():
        logging.warning("data_root %s does not exist. Synthetic data will be used.", data_root)
    else:
        logging.info("Using data root: %s", data_root)
    log_dir = Path(args.logdir)
    log_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(log_dir, args.log_level, args.no_file_log)
    logging.info(
        "Resolved training epochs: cfg=%d (requested --epochs %d)",
        cfg.train.epochs,
        requested_epochs,
    )
    stage1_file = args.stage1_file
    stage25_file = args.stage25_file
    if args.stage == "1":
<<<<<<< Updated upstream
        logging.info("Starting Stage-1 training for %d epochs", args.epochs)
        _stage1_loop(cfg, args.epochs, log_dir, data_root, stage1_file, stage25_file)
=======
        logging.info("Starting Stage-1 training for %d epochs", cfg.train.epochs)
        _stage1_loop(
            cfg,
            cfg.train.epochs,
            log_dir,
            data_root,
            stage1_file,
            stage25_file,
            load_path=load_path,
        )
>>>>>>> Stashed changes
    else:
        logging.info("Preparing Stage-1 teacher for Stage-2.5 training")
        teacher_logdir = log_dir / "stage1_teacher"
        teacher = _stage1_loop(cfg, 1, teacher_logdir, data_root, stage1_file, stage25_file)
        scaler_path = getattr(teacher, "scaler_path", teacher_logdir / cfg.data.scaler_dir / "stage1" / "z5d_scaler.pkl")
        logging.info("Starting Stage-2.5 training for %d epochs", cfg.train.epochs)
        _stage25_loop(
            cfg,
            cfg.train.epochs,
            log_dir,
            teacher,
            data_root,
            stage1_file,
            stage25_file,
            scaler_path,
        )


if __name__ == "__main__":
    main()
