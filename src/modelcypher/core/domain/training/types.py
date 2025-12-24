from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from datetime import datetime

class ComputePrecision(str, Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


class TrainingStatus(str, Enum):
    pending = "pending"
    running = "running"
    paused = "paused"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


@dataclass
class PreflightResult:
    predicted_batch_size: int
    estimated_vram_bytes: int
    available_vram_bytes: int
    can_proceed: bool

@dataclass
class Hyperparameters:
    batch_size: int = 4
    learning_rate: float = 3e-5
    epochs: int = 3
    sequence_length: int = 1024
    gradient_accumulation_steps: int = 1
    # For Unsloth-like optimizations
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    compute_precision: ComputePrecision = ComputePrecision.FLOAT16
    warmup_steps: int = 10
    weight_decay: float = 0.01
    seed: int = 42
    deterministic: bool = True
    
    # MLX specifics
    optimizer_type: str = "adamw" # adamw, adam, sgd
    
@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
@dataclass
class TrainingConfig:
    model_id: str
    dataset_path: str
    output_path: str
    hyperparameters: Hyperparameters
    lora_config: LoRAConfig | None = None
    resume_from_checkpoint_path: str | None = None
    
@dataclass
class TrainingProgress:
    job_id: str
    epoch: int
    step: int
    total_steps: int
    loss: float
    learning_rate: float
    tokens_per_second: float | None = None
    estimated_time_remaining: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    
@dataclass
class CheckpointMetadata:
    version: int
    step: int
    total_steps: int
    train_config: TrainingConfig # Stores the config used to create this checkpoint
    loss_history: list[float]
    timestamp: datetime
    checksum: str
    weights_file: str
    optimizer_file: str | None = None
