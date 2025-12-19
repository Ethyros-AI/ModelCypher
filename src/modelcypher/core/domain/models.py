from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .training import TrainingConfig, TrainingStatus


@dataclass(frozen=True)
class ModelInfo:
    id: str
    alias: str
    architecture: str
    format: str
    path: str
    size_bytes: int
    parameter_count: Optional[int]
    is_default_chat: bool
    created_at: datetime


@dataclass(frozen=True)
class DatasetInfo:
    id: str
    name: str
    path: str
    size_bytes: int
    example_count: int
    created_at: datetime


@dataclass(frozen=True)
class CheckpointRecord:
    job_id: str
    step: int
    loss: float
    timestamp: datetime
    file_path: str


@dataclass(frozen=True)
class TrainingJob:
    job_id: str
    status: TrainingStatus
    model_id: str
    dataset_path: str
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_step: int = 0
    total_steps: int = 0
    current_epoch: int = 0
    total_epochs: int = 0
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    config: Optional[TrainingConfig] = None
    checkpoints: Optional[list[CheckpointRecord]] = None
    loss_history: Optional[list[dict]] = None


@dataclass(frozen=True)
class EvaluationResult:
    id: str
    model_path: str
    model_name: str
    dataset_path: str
    dataset_name: str
    average_loss: float
    perplexity: float
    sample_count: int
    timestamp: datetime
    config: dict
    sample_results: list[dict]


@dataclass(frozen=True)
class CompareCheckpointResult:
    checkpoint_path: str
    model_name: str
    base_model_name: Optional[str]
    response: str
    status: str
    metrics: dict


@dataclass(frozen=True)
class CompareSession:
    id: str
    created_at: datetime
    prompt: str
    config: dict
    checkpoints: list[CompareCheckpointResult]
    notes: Optional[str] = None
    tags: Optional[list[str]] = None
