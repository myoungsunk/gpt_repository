"""모델 공통 블록."""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


def get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """활성화 함수 팩토리."""

    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class MLPBlock(nn.Module):
    """Linear + Norm + Activation + Dropout 블록."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: str = "gelu",
        dropout_p: float = 0.0,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim) if use_layer_norm else nn.Identity()
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout_p)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """전방 계산.

        Args:
            x: torch.FloatTensor[B,in_dim]
        Returns:
            torch.FloatTensor[B,out_dim]
        """

        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


def init_last_bias_zero(module: nn.Module) -> None:
    """마지막 Linear bias를 0으로 초기화."""

    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)


__all__ = ["MLPBlock", "init_last_bias_zero", "get_activation"]
