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
    phase: str = "pretrain_m2"  # {"pretrain_m2", "finetune_m3"}
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
    grad_clip: Optional[float] = 1.0
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
    m2_head_lr_scale: float = 2.0
    m3_gate_mode: str = "kappa"  # {"none", "kappa", "inv_kappa", "mcdrop"}
    m3_delta_max_deg: float = 10.0
    m3_delta_warmup_deg: float = 2.0
    m3_warmup_frac: float = 0.1
    m3_detach_m2: bool = True
    m3_detach_warmup_epochs: int = 3
    m3_freeze_m2: bool = False
    m3_apply_eval_only: bool = False
    m3_output_gain: float = 1.0
    m3_gain_start: float = 0.0
    m3_gain_end: float = 1.0
    m3_gain_ramp_steps: int = 0
    m3_lambda_resid: float = 5e-2
    m3_lambda_gate_entropy: float = 1e-3
    m3_lambda_keep_target: float = 0.0
    m3_gate_keep_threshold: float = 0.5
    m3_target_keep_start: float = 0.2
    m3_target_keep_end: float = 0.6
    m3_keep_warmup_epochs: int = 5
    m3_gate_tau: float = 1.0
    m3_quantile_keep: Optional[float] = None


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
    input_scale: str = "relative_db"  # {"relative_db", "absolute_dbm"}
    mix_warmup_steps: int = 500
    mix_ramp_steps: int = 1500
    mix_weight_max: float = 0.25
    mix_variance_floor: float = 1.0
    scaler_dir: str = "meta"


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
