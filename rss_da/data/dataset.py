"""공용 RSS DoA Dataset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ArrayLike = Sequence[float]
PathLike = Union[str, Path]


@dataclass
class Sample:
    """데이터 샘플 스키마."""

    z5d: ArrayLike  # 길이 5, 상대 dB 기반
    c_meas: ArrayLike  # 길이 2, 상대 dB
    theta_gt: ArrayLike  # 길이 2, rad
    four_rss: Optional[ArrayLike] = None  # 길이 4, Stage-1에서만 사용 (상대 dB)
    c_meas_rel: Optional[ArrayLike] = None  # 길이 2, 상대 dB(중복 보관용)
    c_meas_abs: Optional[ArrayLike] = None  # 길이 2, 절대 dBm (있다면)
    z5d_named: Optional[Dict[str, float]] = None  # 컬럼별 특성 저장
    mask_4rss_is_gt: float = 0.0  # {0,1}


class RssDoADataset(Dataset):
    """Stage-1/2.5 공용 Dataset."""

    def __init__(
        self,
        samples: List[Sample],
        stage: str = "1",
        modality_dropout_p: float = 0.0,
        training: bool = True,
        rng: Optional[np.random.RandomState] = None,
    ) -> None:
        self.samples = samples
        self.stage = stage
        self.modality_dropout_p = modality_dropout_p
        self.training = training
        self.rng = rng or np.random.RandomState(0)
        self._z5d_mean: Optional[np.ndarray] = None
        self._z5d_std: Optional[np.ndarray] = None
        if stage not in {"1", "2.5"}:
            raise ValueError(f"Unsupported stage: {stage}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        z5d = np.asarray(sample.z5d, dtype=np.float32)  # (5,), float32
        c_meas = np.asarray(sample.c_meas, dtype=np.float32)  # (2,), float32 (상대 dB)
        theta_gt = np.asarray(sample.theta_gt, dtype=np.float32)  # (2,), float32
        mask = float(sample.mask_4rss_is_gt)
        four_rss = None if sample.four_rss is None else np.asarray(sample.four_rss, dtype=np.float32)
        c_meas_abs = None if sample.c_meas_abs is None else np.asarray(sample.c_meas_abs, dtype=np.float32)
        z5d_named = sample.z5d_named

        if self.training and self.modality_dropout_p > 0.0:
            if self.rng.rand() < self.modality_dropout_p:
                # z5d 일부 요소 마스킹
                dropout_mask = self.rng.rand(*z5d.shape) < 0.5
                z5d = z5d.copy()
                z5d[dropout_mask] = 0.0
            if four_rss is not None and self.rng.rand() < self.modality_dropout_p:
                four_rss = np.full((4,), -120.0, dtype=np.float32)
                mask = 0.0

        if self._z5d_mean is not None and self._z5d_std is not None:
            z5d = (z5d - self._z5d_mean) / self._z5d_std

        result: Dict[str, Any] = {
            "z5d": torch.from_numpy(z5d),  # torch.FloatTensor[B=?,5]
            "c_meas": torch.from_numpy(c_meas),  # torch.FloatTensor[2]
            "theta_gt": torch.from_numpy(theta_gt),  # torch.FloatTensor[2]
            "mask_4rss_is_gt": torch.tensor([mask], dtype=torch.float32),  # torch.FloatTensor[1]
        }
        if sample.c_meas_rel is not None:
            c_meas_rel = np.asarray(sample.c_meas_rel, dtype=np.float32)
            result["c_meas_rel"] = torch.from_numpy(c_meas_rel)  # torch.FloatTensor[2]
        else:
            result["c_meas_rel"] = torch.from_numpy(c_meas)
        if c_meas_abs is not None:
            result["c_meas_abs"] = torch.from_numpy(c_meas_abs)  # torch.FloatTensor[2]
        if four_rss is not None:
            result["four_rss"] = torch.from_numpy(four_rss)  # torch.FloatTensor[4]
        else:
            result["four_rss"] = torch.empty(0, dtype=torch.float32)  # torch.FloatTensor[0]
        if z5d_named is not None:
            result["z5d_named"] = {
                key: torch.tensor(float(value), dtype=torch.float32)
                for key, value in z5d_named.items()
            }
        return result

    def set_standardization(self, mean: np.ndarray, std: np.ndarray) -> None:
        """z5d 표준화 파라미터 설정."""

        std_safe = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        self._z5d_mean = mean.astype(np.float32)
        self._z5d_std = std_safe


def collate_samples(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """배치 단위로 텐서를 스택한다."""

    z5d = torch.stack([item["z5d"] for item in batch], dim=0)  # torch.FloatTensor[B,5]
    c_meas = torch.stack([item["c_meas"] for item in batch], dim=0)  # torch.FloatTensor[B,2]
    theta_gt = torch.stack([item["theta_gt"] for item in batch], dim=0)  # torch.FloatTensor[B,2]
    mask = torch.stack([item["mask_4rss_is_gt"] for item in batch], dim=0)  # torch.FloatTensor[B,1]
    four_rss_list = [item["four_rss"] for item in batch]
    if all(tensor.numel() == 0 for tensor in four_rss_list):
        four_rss = torch.empty((z5d.shape[0], 0), dtype=torch.float32)  # torch.FloatTensor[B,0]
    else:
        four_rss = torch.stack([
            tensor if tensor.numel() > 0 else torch.full((4,), -120.0, dtype=torch.float32)
            for tensor in four_rss_list
        ], dim=0)  # torch.FloatTensor[B,4]
    batch_out: Dict[str, torch.Tensor] = {
        "z5d": z5d,
        "c_meas": c_meas,
        "theta_gt": theta_gt,
        "four_rss": four_rss,
        "mask_4rss_is_gt": mask,
    }
    if "c_meas_rel" in batch[0]:
        c_meas_rel = torch.stack([item["c_meas_rel"] for item in batch], dim=0)  # torch.FloatTensor[B,2]
        batch_out["c_meas_rel"] = c_meas_rel
    else:
        batch_out["c_meas_rel"] = torch.empty((z5d.shape[0], 0), dtype=torch.float32)
    if "c_meas_abs" in batch[0]:
        c_meas_abs = torch.stack([item["c_meas_abs"] for item in batch], dim=0)  # torch.FloatTensor[B,2]
        batch_out["c_meas_abs"] = c_meas_abs
    if "z5d_named" in batch[0]:
        keys = batch[0]["z5d_named"].keys()
        batch_out["z5d_named"] = {
            key: torch.stack([item["z5d_named"][key] for item in batch], dim=0)
            for key in keys
        }
    return batch_out


__all__ = ["Sample", "RssDoADataset", "collate_samples"]


def load_standardized_csv(
    path: PathLike,
    stage: str,
    prefer_absolute: bool = True,
) -> List[Sample]:
    """prepare_datasets_v2.py 결과 CSV를 Sample 리스트로 변환한다."""

    df = pd.read_csv(path)
    stage = str(stage)
    if stage not in {"1", "2.5"}:
        raise ValueError(f"Unsupported stage: {stage}")

    samples: List[Sample] = []
    rss_cols = [
        "rss_a_p1_rel_db",
        "rss_b_p1_rel_db",
        "rss_a_p2_rel_db",
        "rss_b_p2_rel_db",
    ]

    for _, row in df.iterrows():
        c1_rel = float(row["c1_rel_db"])
        c2_rel = float(row["c2_rel_db"])
        delta_rel = float(row["delta_rel_db"])
        sum_rel = float(row["sum_rel_db"])
        log_ratio = float(row["log_ratio"])
        has_abs = (
            prefer_absolute
            and pd.notna(row.get("c1_dbm"))
            and pd.notna(row.get("c2_dbm"))
        )
        c_meas_abs: Optional[List[float]] = None
        if has_abs:
            c_meas_abs = [float(row["c1_dbm"]), float(row["c2_dbm"])]
        z5d_values = [c1_rel, c2_rel, delta_rel, sum_rel, log_ratio]
        z5d_named = {
            "c1_rel_db": c1_rel,
            "c2_rel_db": c2_rel,
            "delta_rel_db": delta_rel,
            "sum_rel_db": sum_rel,
            "log_ratio": log_ratio,
        }

        mask = float(row.get("mask_4rss_is_gt", 0.0))
        theta = [float(row["theta1_rad"]), float(row["theta2_rad"])]

        four_rss_values: Optional[List[float]]
        if stage == "1":
            rss_raw = [row.get(col, np.nan) for col in rss_cols]
            if all(pd.isna(value) for value in rss_raw):
                four_rss_values = None
            else:
                four_array = np.asarray(rss_raw, dtype=np.float32)
                four_rss_values = four_array.tolist()
        else:
            four_rss_values = None

        samples.append(
            Sample(
                z5d=z5d_values,
                c_meas=[c1_rel, c2_rel],
                c_meas_rel=[c1_rel, c2_rel],
                c_meas_abs=c_meas_abs,
                theta_gt=theta,
                four_rss=four_rss_values,
                z5d_named=z5d_named,
                mask_4rss_is_gt=mask,
            )
        )

    return samples


__all__.append("load_standardized_csv")
