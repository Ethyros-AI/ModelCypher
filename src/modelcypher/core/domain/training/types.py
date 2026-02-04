# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


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


class FineTuneType(str, Enum):
    """Fine-tuning method type."""

    LORA = "lora"
    DORA = "dora"  # Weight-decomposed LoRA


@dataclass
class PreflightResult:
    predicted_batch_size: int
    estimated_vram_bytes: int
    available_vram_bytes: int
    can_proceed: bool


@dataclass
class Hyperparameters:
    batch_size: int
    learning_rate: float
    epochs: int
    sequence_length: int
    gradient_accumulation_steps: int
    # For Unsloth-like optimizations
    gradient_checkpointing: bool
    mixed_precision: bool
    compute_precision: ComputePrecision
    warmup_steps: int
    weight_decay: float
    seed: int
    deterministic: bool

    # MLX specifics
    optimizer_type: str  # geometric (geometry-derived, no magic hyperparameters)


@dataclass
class LoRASettings:
    """Settings for LoRA adapters."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    fine_tune_type: FineTuneType = FineTuneType.LORA
    num_layers: int | None = None  # None = all layers

    @property
    def scale(self) -> float:
        """LoRA scaling factor: alpha / rank."""
        return self.alpha / max(self.rank, 1)

    @classmethod
    def default(cls) -> "LoRASettings":
        return cls()

    @classmethod
    def for_mistral(cls) -> "LoRASettings":
        """Preset for Mistral-style models."""
        return cls(
            rank=16,
            alpha=32.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

    @classmethod
    def for_llama(cls) -> "LoRASettings":
        """Preset for Llama-style models."""
        return cls(
            rank=8,
            alpha=16.0,
            target_modules=["q_proj", "v_proj"],
        )

    @classmethod
    def for_qwen(cls) -> "LoRASettings":
        """Preset for Qwen-style models (gate in MLP).

        DEPRECATED: Prefer geometry-derived configuration.
        """
        return cls(
            rank=16,
            alpha=32.0,
            target_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"],
        )


@dataclass
class TrainingSpec:
    model_id: str
    dataset_path: str
    output_path: str
    hyperparameters: Hyperparameters
    lora_config: LoRASettings | None = None
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
    train_config: TrainingSpec  # Stores the training spec used to create this checkpoint
    loss_history: list[float]
    timestamp: datetime
    checksum: str
    weights_file: str
    optimizer_file: str | None = None
