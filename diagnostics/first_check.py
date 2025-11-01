"""빠른 점검용 스크립트."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from rss_da.data.dataset import RssDoADataset, collate_samples, load_standardized_csv

REQUIRED_STAGE1 = {
    "c1_rel_db",
    "c2_rel_db",
    "delta_rel_db",
    "sum_rel_db",
    "log_ratio",
    "rss_a_p1_rel_db",
    "rss_b_p1_rel_db",
    "rss_a_p2_rel_db",
    "rss_b_p2_rel_db",
}
REQUIRED_STAGE25 = {
    "c1_rel_db",
    "c2_rel_db",
    "delta_rel_db",
    "sum_rel_db",
    "log_ratio",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 데이터셋 점검")
    parser.add_argument("csv", type=str, help="prepare_datasets_v2.py 결과 CSV 경로")
    parser.add_argument("stage", choices=["1", "2.5"], help="스테이지")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    required = REQUIRED_STAGE1 if args.stage == "1" else REQUIRED_STAGE25
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    samples = load_standardized_csv(csv_path, stage=args.stage)
    dataset = RssDoADataset(list(samples), stage=args.stage, training=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_samples)
    batch = next(iter(loader))
    z = batch["z5d"]
    named = batch.get("z5d_named")
    cols = ["c1_rel_db", "c2_rel_db", "delta_rel_db", "sum_rel_db", "log_ratio"]
    if named is not None:
        for col in cols:
            tensor = named[col]
            print(
                f"{col:15s} mean={tensor.mean().item():8.3f} std={tensor.std().item():8.3f} "
                f"min={tensor.min().item():8.3f} max={tensor.max().item():8.3f}"
            )
    else:
        print(
            "z5d stats (all features):",
            z.mean().item(),
            z.std().item(),
            z.min().item(),
            z.max().item(),
        )
    if batch["four_rss"].numel() > 0:
        print("four_rss shape:", tuple(batch["four_rss"].shape))
    else:
        print("four_rss absent (Stage-2.5)")
    c_rel = batch["c_meas_rel"]
    if c_rel.numel() > 0:
        print("c_meas_rel stats:", c_rel.mean().item(), c_rel.std().item())
    print(f"Loaded {len(dataset)} samples from {csv_path}")


if __name__ == "__main__":
    main()
