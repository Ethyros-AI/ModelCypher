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

"""Checkpoint data models for training persistence.

Contains metadata structures for checkpoints, optimizer state,
and crash recovery information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class CheckpointErrorKind(str, Enum):
    """Kind of checkpoint error."""

    INSUFFICIENT_DISK_SPACE = "insufficient_disk_space"
    WRITE_FAILED = "write_failed"
    NO_VALID_CHECKPOINTS = "no_valid_checkpoints"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    MISSING_FILE = "missing_file"


@dataclass(frozen=True)
class OptimizerStateMetadata:
    """Metadata describing an optimizer state persisted alongside a checkpoint."""

    type_name: str
    """Name of the optimizer type (e.g., 'AdamW')."""

    state_file: str
    """Filename of the optimizer state file."""

    checksum: str
    """SHA256 checksum of the state file."""

    scalar_hyperparameters: dict[str, float] = field(default_factory=dict)
    """Scalar hyperparameters (learning_rate, weight_decay, etc.)."""

    vector_hyperparameters: dict[str, list[float]] = field(default_factory=dict)
    """Vector hyperparameters (betas, etc.)."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type_name": self.type_name,
            "state_file": self.state_file,
            "checksum": self.checksum,
            "scalar_hyperparameters": self.scalar_hyperparameters,
            "vector_hyperparameters": self.vector_hyperparameters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizerStateMetadata:
        """Create from dictionary."""
        return cls(
            type_name=data.get("type_name", "unknown"),
            state_file=data.get("state_file", ""),
            checksum=data.get("checksum", ""),
            scalar_hyperparameters=data.get("scalar_hyperparameters", {}),
            vector_hyperparameters=data.get("vector_hyperparameters", {}),
        )


@dataclass(frozen=True)
class FineTunedModelMetadata:
    """Metadata describing fine-tuned model context for evaluation/export."""

    base_model_id: str
    """ID of the base model that was fine-tuned."""

    tokenizer_strategy: str
    """Tokenizer strategy used."""

    lora_config: dict[str, Any] | None = None
    """LoRA configuration if applicable."""

    quantization_config: dict[str, Any] | None = None
    """Quantization configuration if applicable."""

    hyperparameters: dict[str, Any] | None = None
    """Training hyperparameters."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "base_model_id": self.base_model_id,
            "tokenizer_strategy": self.tokenizer_strategy,
        }
        if self.lora_config is not None:
            result["lora_config"] = self.lora_config
        if self.quantization_config is not None:
            result["quantization_config"] = self.quantization_config
        if self.hyperparameters is not None:
            result["hyperparameters"] = self.hyperparameters
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FineTunedModelMetadata:
        """Create from dictionary."""
        return cls(
            base_model_id=data.get("base_model_id", ""),
            tokenizer_strategy=data.get("tokenizer_strategy", "default"),
            lora_config=data.get("lora_config"),
            quantization_config=data.get("quantization_config"),
            hyperparameters=data.get("hyperparameters"),
        )


@dataclass(frozen=True)
class ModelArchitectureConfig:
    """Defines transformer architecture dimensions for parameter counting and memory estimation."""

    model_type: str
    """Type of model architecture (llama, mistral, qwen2, phi, gemma, etc.)."""

    vocabulary_size: int
    """Size of the vocabulary (number of unique tokens)."""

    hidden_size: int
    """Hidden dimension size (embedding dimension)."""

    num_layers: int
    """Number of transformer layers."""

    num_heads: int
    """Number of attention heads."""

    memory_overrides: dict[str, Any] | None = None
    """Optional overrides that influence memory estimation heuristics."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "model_type": self.model_type,
            "vocabulary_size": self.vocabulary_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
        }
        if self.memory_overrides is not None:
            result["memory_overrides"] = self.memory_overrides
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelArchitectureConfig:
        """Create from dictionary."""
        return cls(
            model_type=data.get("model_type", "simple_transformer"),
            vocabulary_size=data.get("vocabulary_size", 32000),
            hidden_size=data.get("hidden_size", 4096),
            num_layers=data.get("num_layers", 32),
            num_heads=data.get("num_heads", 32),
            memory_overrides=data.get("memory_overrides"),
        )


@dataclass
class CheckpointMetadataV2:
    """Metadata for a training checkpoint (version 2).

    Contains all information needed to resume training from this point,
    plus integrity validation (checksum) and a schema version for evolution.
    """

    version: int
    """Checkpoint format version (increment when schema changes)."""

    step: int
    """Training step when this checkpoint was saved."""

    total_steps: int
    """Total steps in the training job."""

    timestamp: datetime
    """When this checkpoint was saved."""

    checksum: str
    """SHA256 checksum of weights file (for corruption detection)."""

    weights_file: str
    """Filename of the weights file (e.g., 'checkpoint-1000.safetensors')."""

    loss_history: list[float] = field(default_factory=list)
    """Training loss history (for progress tracking)."""

    model_config: ModelArchitectureConfig | None = None
    """Model architecture configuration (required for evaluation)."""

    optimizer_state: OptimizerStateMetadata | None = None
    """Optional optimizer state metadata if optimizer parameters were persisted."""

    fine_tuned_model: FineTunedModelMetadata | None = None
    """Metadata describing the fine-tuned model context."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "version": self.version,
            "step": self.step,
            "total_steps": self.total_steps,
            "timestamp": self.timestamp.isoformat(),
            "checksum": self.checksum,
            "weights_file": self.weights_file,
            "loss_history": self.loss_history,
        }
        if self.model_config is not None:
            result["model_config"] = self.model_config.to_dict()
        if self.optimizer_state is not None:
            result["optimizer_state"] = self.optimizer_state.to_dict()
        if self.fine_tuned_model is not None:
            result["fine_tuned_model"] = self.fine_tuned_model.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointMetadataV2:
        """Create from dictionary."""
        model_config = None
        if "model_config" in data and data["model_config"]:
            model_config = ModelArchitectureConfig.from_dict(data["model_config"])

        optimizer_state = None
        if "optimizer_state" in data and data["optimizer_state"]:
            optimizer_state = OptimizerStateMetadata.from_dict(data["optimizer_state"])

        fine_tuned_model = None
        if "fine_tuned_model" in data and data["fine_tuned_model"]:
            fine_tuned_model = FineTunedModelMetadata.from_dict(data["fine_tuned_model"])

        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        return cls(
            version=data.get("version", 2),
            step=data.get("step", 0),
            total_steps=data.get("total_steps", 0),
            timestamp=timestamp,
            checksum=data.get("checksum", ""),
            weights_file=data.get("weights_file", ""),
            loss_history=data.get("loss_history", []),
            model_config=model_config,
            optimizer_state=optimizer_state,
            fine_tuned_model=fine_tuned_model,
        )


@dataclass(frozen=True)
class RecoveryInfo:
    """Information about a recovered checkpoint after crash."""

    checkpoint: CheckpointMetadataV2
    """The validated checkpoint metadata."""

    checkpoints_dir: Path
    """Directory containing checkpoints."""

    output_dir: Path
    """Training output directory."""
