"""공용 RSS DoA Dataset."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

ArrayLike = Sequence[float]


@dataclass
class Sample:
    """데이터 샘플 스키마."""

    z5d: ArrayLike  # 길이 5
    c_meas: ArrayLike  # 길이 2
    theta_gt: ArrayLike  # 길이 2, rad
    four_rss: Optional[ArrayLike] = None  # 길이 4, Stage-1에서만 사용
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
        if stage not in {"1", "2.5"}:
            raise ValueError(f"Unsupported stage: {stage}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        z5d = np.asarray(sample.z5d, dtype=np.float32)  # (5,), float32
        c_meas = np.asarray(sample.c_meas, dtype=np.float32)  # (2,), float32
        theta_gt = np.asarray(sample.theta_gt, dtype=np.float32)  # (2,), float32
        mask = float(sample.mask_4rss_is_gt)
        four_rss = None if sample.four_rss is None else np.asarray(sample.four_rss, dtype=np.float32)

        if self.training and self.modality_dropout_p > 0.0:
            if self.rng.rand() < self.modality_dropout_p:
                # z5d 일부 요소 마스킹
                dropout_mask = self.rng.rand(*z5d.shape) < 0.5
                z5d = z5d.copy()
                z5d[dropout_mask] = 0.0
            if four_rss is not None and self.rng.rand() < self.modality_dropout_p:
                four_rss = np.full((4,), -120.0, dtype=np.float32)
                mask = 0.0

        result: Dict[str, Any] = {
            "z5d": torch.from_numpy(z5d),  # torch.FloatTensor[B=?,5]
            "c_meas": torch.from_numpy(c_meas),  # torch.FloatTensor[2]
            "theta_gt": torch.from_numpy(theta_gt),  # torch.FloatTensor[2]
            "mask_4rss_is_gt": torch.tensor([mask], dtype=torch.float32),  # torch.FloatTensor[1]
        }
        if four_rss is not None:
            result["four_rss"] = torch.from_numpy(four_rss)  # torch.FloatTensor[4]
        else:
            result["four_rss"] = torch.empty(0, dtype=torch.float32)  # torch.FloatTensor[0]
        return result


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
    return {
        "z5d": z5d,
        "c_meas": c_meas,
        "theta_gt": theta_gt,
        "four_rss": four_rss,
        "mask_4rss_is_gt": mask,
    }


__all__ = ["Sample", "RssDoADataset", "collate_samples"]