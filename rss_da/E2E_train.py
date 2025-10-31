"""엔드투엔드 학습 스크립트."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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


def _build_dataloader(
    stage: str,
    batch_size: int,
    data_root: Path,
    stage1_file: str,
    stage25_file: str,
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
    loader = _build_dataloader("1", cfg.train.batch_size, data_root, stage1_file, stage25_file)
    logger = Logger(log_dir)
    global_step = 0
    try:
        for epoch in range(epochs):
            logging.info("[Stage-1] Starting epoch %d/%d", epoch + 1, epochs)
            for batch in loader:
                outputs = trainer.train_step(batch)
                metrics = {
                    "loss_total": outputs.loss_total,
                    "loss_sup0": outputs.loss_sup0,
                    "loss_sup1": outputs.loss_sup1,
                    "loss_mix": outputs.loss_mix,
                    "loss_phys": outputs.loss_phys,
                    "deg_rmse": outputs.deg_rmse,
                    "kappa_mean": outputs.kappa_mean,
                }
                if global_step % cfg.train.log_interval == 0:
                    _log_metrics(logger, global_step, "train", metrics)
                    logging.info(
                        "[Stage-1][step=%d] loss=%.4f sup0=%.4f sup1=%.4f mix=%.4f phys=%.4f degRMSE=%.3f",
                        global_step,
                        metrics["loss_total"],
                        metrics["loss_sup0"],
                        metrics["loss_sup1"],
                        metrics["loss_mix"],
                        metrics["loss_phys"],
                        metrics["deg_rmse"],
                    )
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
) -> Stage25Trainer:
    trainer = Stage25Trainer(cfg, teacher_modules=teacher.modules())
    logging.info("Stage-2.5 trainer initialized on device: %s", trainer.device)
    loader = _build_dataloader("2.5", cfg.train.batch_size, data_root, stage1_file, stage25_file)
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
                    "loss_mix": outputs.loss_mix,
                    "loss_phys": outputs.loss_phys,
                    "loss_align": outputs.loss_align,
                    "deg_rmse": outputs.deg_rmse,
                    "kappa_mean": outputs.kappa_mean,
                }
                if global_step % cfg.train.log_interval == 0:
                    _log_metrics(logger, global_step, "train", metrics)
                    logging.info(
                        "[Stage-2.5][step=%d] loss=%.4f sup=%.4f kd=%.4f mix=%.4f align=%.4f degRMSE=%.3f",
                        global_step,
                        metrics["loss_total"],
                        metrics["loss_sup"],
                        metrics["loss_kd"],
                        metrics["loss_mix"],
                        metrics["loss_align"],
                        metrics["deg_rmse"],
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
    parser.add_argument("--use_m3", action="store_true")
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
    cfg.train.epochs = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.log_dir = args.logdir
    data_root = Path(args.data_root)
    if not data_root.exists():
        logging.warning("data_root %s does not exist. Synthetic data will be used.", data_root)
    else:
        logging.info("Using data root: %s", data_root)
    log_dir = Path(args.logdir)
    log_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(log_dir, args.log_level, args.no_file_log)
    stage1_file = args.stage1_file
    stage25_file = args.stage25_file
    if args.stage == "1":
        logging.info("Starting Stage-1 training for %d epochs", args.epochs)
        _stage1_loop(cfg, args.epochs, log_dir, data_root, stage1_file, stage25_file)
    else:
        logging.info("Preparing Stage-1 teacher for Stage-2.5 training")
        teacher = _stage1_loop(cfg, 1, log_dir / "stage1_teacher", data_root, stage1_file, stage25_file)
        logging.info("Starting Stage-2.5 training for %d epochs", args.epochs)
        _stage25_loop(cfg, args.epochs, log_dir, teacher, data_root, stage1_file, stage25_file)


if __name__ == "__main__":
    main()
