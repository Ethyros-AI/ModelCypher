from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TrainingStatus(str, Enum):
    pending = "pending"
    running = "running"
    paused = "paused"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


@dataclass(frozen=True)
class LoRAConfig:
    rank: int
    alpha: float
    dropout: float
    targets: list[str]
    layers: Optional[int] = None

    @property
    def scale(self) -> float:
        return self.alpha / float(self.rank) if self.rank > 0 else 1.0


@dataclass(frozen=True)
class TrainingConfig:
    model_id: str
    dataset_path: str
    learning_rate: float
    batch_size: int
    epochs: int
    sequence_length: int
    grad_accum: Optional[int] = None
    warmup_steps: Optional[int] = None
    weight_decay: Optional[float] = None
    gradient_clip: Optional[float] = None
    resume_from: Optional[str] = None
    lora: Optional[LoRAConfig] = None
    out_dir: Optional[str] = None
    seed: Optional[int] = None
    deterministic: bool = False


@dataclass(frozen=True)
class PreflightResult:
    predicted_batch_size: int
    estimated_vram_bytes: int
    available_vram_bytes: int
    can_proceed: bool


@dataclass(frozen=True)
class TrainingProgress:
    step: int
    total_steps: int
    epoch: int
    total_epochs: int
    loss: float
    learning_rate: float
    tokens_per_second: Optional[float] = None
    eta_seconds: Optional[int] = None
