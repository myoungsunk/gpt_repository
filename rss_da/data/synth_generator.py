"""합성 RSS 데이터 생성기."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .dataset import Sample


@dataclass
class SynthConfig:
    num_samples: int = 256
    tx_power_dbm: float = 0.0
    pathloss_exponent: float = 2.0
    reference_distance_m: float = 1.0
    reference_loss_db: float = 40.0
    noise_std_db: float = 2.0
    antenna_gain_main_db: float = 3.0
    antenna_gain_side_db: float = -3.0


def _rad_to_deg(theta_rad: np.ndarray) -> np.ndarray:
    return np.rad2deg(theta_rad)


def _ldpl_loss(distance_m: np.ndarray, config: SynthConfig) -> np.ndarray:
    with np.errstate(divide="ignore"):
        loss = config.reference_loss_db + 10 * config.pathloss_exponent * np.log10(
            np.maximum(distance_m / config.reference_distance_m, 1e-3)
        )
    return loss


def _antenna_pattern(theta_rad: np.ndarray, config: SynthConfig) -> Tuple[np.ndarray, np.ndarray]:
    theta_deg = _rad_to_deg(theta_rad)
    main_gain = config.antenna_gain_main_db - 0.1 * (theta_deg**2) / 90.0
    side_gain = np.full_like(main_gain, config.antenna_gain_side_db)
    return main_gain, side_gain


def generate_synth_samples(config: SynthConfig, *, seed: int = 0) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, size=(config.num_samples, 2))  # rad
    distances = rng.uniform(1.0, 50.0, size=(config.num_samples, 2))  # m

    main_gain, side_gain = _antenna_pattern(theta[:, 0], config)
    main_gain2, side_gain2 = _antenna_pattern(theta[:, 1], config)

    pathloss1 = _ldpl_loss(distances[:, 0], config)
    pathloss2 = _ldpl_loss(distances[:, 1], config)

    rss1 = config.tx_power_dbm + main_gain - pathloss1
    rss2 = config.tx_power_dbm + main_gain2 - pathloss2
    rss3 = config.tx_power_dbm + side_gain - pathloss1
    rss4 = config.tx_power_dbm + side_gain2 - pathloss2

    noise = rng.normal(0.0, config.noise_std_db, size=(config.num_samples, 4))
    four_rss = np.stack([rss1, rss2, rss3, rss4], axis=-1) + noise
    four_rss = four_rss.astype(np.float32)

    c_meas = np.stack([
        10 * np.log10(np.sum(10 ** (four_rss[:, :2] / 10), axis=-1)),
        10 * np.log10(np.sum(10 ** (four_rss[:, 2:] / 10), axis=-1)),
    ], axis=-1).astype(np.float32)

    delta = rss1 - rss2
    sum_db = rss1 + rss2
    log_ratio = np.log1p(np.abs(delta))
    z5d = np.stack([
        (rss1 + rss3) / 2,
        (rss2 + rss4) / 2,
        delta,
        sum_db,
        log_ratio,
    ], axis=-1).astype(np.float32)

    data = {
        "z5d": z5d,
        "c_meas": c_meas,
        "theta_gt": theta.astype(np.float32),
        "four_rss": four_rss,
    }
    return theta, data


def build_samples(data: Dict[str, np.ndarray]) -> Tuple[Sample, ...]:
    samples = []
    num = data["z5d"].shape[0]
    for i in range(num):
        samples.append(
            Sample(
                z5d=data["z5d"][i].tolist(),
                c_meas=data["c_meas"][i].tolist(),
                theta_gt=data["theta_gt"][i].tolist(),
                four_rss=data["four_rss"][i].tolist(),
                mask_4rss_is_gt=1.0,
            )
        )
    return tuple(samples)


def save_dataset(path: Path, data: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: array.tolist() for key, array in data.items()}
    path.write_text(json.dumps(payload))


def load_dataset(path: Path) -> Dict[str, np.ndarray]:
    payload = json.loads(path.read_text())
    return {key: np.asarray(value, dtype=np.float32) for key, value in payload.items()}


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic RSS-DoA samples.")
    parser.add_argument("--out", type=Path, required=True, help="저장할 JSON 경로")
    parser.add_argument("--num-samples", type=int, default=256, help="생성할 샘플 수")
    parser.add_argument("--seed", type=int, default=0, help="난수 시드")
    parser.add_argument("--tx-power", type=float, default=0.0, help="송신 전력(dBm)")
    parser.add_argument("--pathloss-exp", type=float, default=2.0, help="경로손실 지수")
    parser.add_argument("--noise-std", type=float, default=2.0, help="RSS 노이즈 표준편차(dB)")
    return parser


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()
    config = SynthConfig(
        num_samples=args.num_samples,
        tx_power_dbm=args.tx_power,
        pathloss_exponent=args.pathloss_exp,
        noise_std_db=args.noise_std,
    )
    _, data = generate_synth_samples(config, seed=args.seed)
    save_dataset(args.out, data)


if __name__ == "__main__":
    main()


__all__ = [
    "SynthConfig",
    "generate_synth_samples",
    "build_samples",
    "save_dataset",
    "load_dataset",
]
