"""로깅 유틸."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

from torch.utils.tensorboard import SummaryWriter


class Logger:
    """TensorBoard와 CSV를 동시에 기록한다."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir.as_posix())
        self.csv_path = self.log_dir / "metrics.csv"
        self._csv_file = self.csv_path.open("w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(["step", "metric", "value"])

    def add_scalars(self, step: int, metrics: Dict[str, float]) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
            self._csv_writer.writerow([step, key, value])
        self._csv_file.flush()

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
        self._csv_file.close()


__all__ = ["Logger"]
