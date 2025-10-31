"""엔드투엔드 학습 스크립트."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

from torch.utils.data import DataLoader

from rss_da.config import Config
from rss_da.data.dataset import RssDoADataset, Sample, collate_samples
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


def _build_dataloader(stage: str, batch_size: int) -> DataLoader:
    samples = _create_samples(stage, max(batch_size * 4, 128))
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


def _stage1_loop(cfg: Config, epochs: int, log_dir: Path) -> Stage1Trainer:
    trainer = Stage1Trainer(cfg)
    loader = _build_dataloader("1", cfg.train.batch_size)
    logger = Logger(log_dir)
    global_step = 0
    try:
        for epoch in range(epochs):
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
                global_step += 1
        return trainer
    finally:
        logger.close()


def _stage25_loop(cfg: Config, epochs: int, log_dir: Path, teacher: Stage1Trainer) -> Stage25Trainer:
    trainer = Stage25Trainer(cfg, teacher_modules=teacher.modules())
    loader = _build_dataloader("2.5", cfg.train.batch_size)
    logger = Logger(log_dir)
    global_step = 0
    try:
        for epoch in range(epochs):
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
                global_step += 1
        return trainer
    finally:
        logger.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RSS DoA Stage-1/Stage-2.5 학습")
    parser.add_argument("--stage", choices=["1", "2.5"], default="1")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--use_m3", action="store_true")
    parser.add_argument("--use_coral", action="store_true")
    parser.add_argument("--use_dann", action="store_true")
    parser.add_argument("--use_cdan", action="store_true")
    parser.add_argument("--gradnorm", action="store_true")
    parser.add_argument("--uncertainty", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--logdir", type=str, default="./runs/demo")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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
    log_dir = Path(args.logdir)
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "1":
        _stage1_loop(cfg, args.epochs, log_dir)
    else:
        teacher = _stage1_loop(cfg, 1, log_dir / "stage1_teacher")
        _stage25_loop(cfg, args.epochs, log_dir, teacher)


if __name__ == "__main__":
    main()
