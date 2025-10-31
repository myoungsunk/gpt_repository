"""프로젝트 전역 설정 모듈."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


def _default_loss_weights() -> List[float]:
    return [1.0, 0.5, 1.0, 0.2, 0.3]


def _default_data_roots() -> List[str]:
    return ["./data"]


@dataclass
class TrainConfig:
    """학습 관련 하이퍼파라미터."""

    stage: str = "1"  # {"1", "2.5"}
    latent_dim: int = 128
    phi_dim: int = 16
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    loss_weights: List[float] = field(default_factory=_default_loss_weights)
    # w_sup, w_kd, w_mix, w_align, w_phys
    tau: float = 2.0  # KD temperature
    ema_decay: float = 0.99
    dropout_p: float = 0.1
    epochs: int = 50
    grad_clip: Optional[float] = 5.0
    use_m3: bool = False
    use_coral: bool = False
    use_dann: bool = False
    use_cdan: bool = False
    gradnorm: bool = True
    uncertainty_weighting: bool = False
    log_interval: int = 10
    eval_interval: int = 1
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    seed: int = 42


@dataclass
class DataConfig:
    """데이터 관련 설정."""

    roots: List[str] = field(default_factory=_default_data_roots)
    stage: str = "1"
    modality_dropout_p: float = 0.0
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    shuffle: bool = True


@dataclass
class PhysicsConfig:
    """물리 제약 관련 설정."""

    tx_power_dbm: float = 0.0
    pathloss_exponent: float = 2.0
    shadow_sigma_db: float = 4.0
    bias_db: float = 0.0
    min_distance_m: float = 1.0
    max_distance_m: float = 100.0


@dataclass
class Config:
    """프로젝트 최상위 설정 객체."""

    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)


__all__ = ["Config", "TrainConfig", "DataConfig", "PhysicsConfig"]
