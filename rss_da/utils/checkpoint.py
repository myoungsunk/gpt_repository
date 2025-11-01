"""체크포인트 유틸."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch


def save_checkpoint(path: Path, modules: Dict[str, torch.nn.Module], step: int) -> None:
    """모듈 state_dict를 저장한다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"{name}_state": module.state_dict() for name, module in modules.items()}
    payload["step"] = step
    torch.save(payload, path)


def load_checkpoint(path: Path, modules: Dict[str, torch.nn.Module]) -> int:
    """모듈 state_dict를 로드하고 step 반환."""

    payload = torch.load(path, map_location="cpu")
    step = int(payload.get("step", 0))
    for name, module in modules.items():
        key = f"{name}_state"
        if key in payload:
            module.load_state_dict(payload[key])
    return step


__all__ = ["save_checkpoint", "load_checkpoint"]
