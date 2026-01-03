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

"""Tests for checkpoint models module."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from modelcypher.core.domain.training.checkpoint_models import (
    CheckpointErrorKind,
    CheckpointMetadataV2,
    FineTunedModelMetadata,
    ModelArchitectureSpec,
    OptimizerStateMetadata,
    RecoveryInfo,
)


class TestCheckpointErrorKind:
    """Tests for CheckpointErrorKind enum."""

    def test_insufficient_disk_space(self):
        """Test INSUFFICIENT_DISK_SPACE value."""
        assert CheckpointErrorKind.INSUFFICIENT_DISK_SPACE.value == "insufficient_disk_space"

    def test_write_failed(self):
        """Test WRITE_FAILED value."""
        assert CheckpointErrorKind.WRITE_FAILED.value == "write_failed"

    def test_no_valid_checkpoints(self):
        """Test NO_VALID_CHECKPOINTS value."""
        assert CheckpointErrorKind.NO_VALID_CHECKPOINTS.value == "no_valid_checkpoints"

    def test_checksum_mismatch(self):
        """Test CHECKSUM_MISMATCH value."""
        assert CheckpointErrorKind.CHECKSUM_MISMATCH.value == "checksum_mismatch"

    def test_missing_file(self):
        """Test MISSING_FILE value."""
        assert CheckpointErrorKind.MISSING_FILE.value == "missing_file"

    def test_is_string_enum(self):
        """Test CheckpointErrorKind is a string enum."""
        assert isinstance(CheckpointErrorKind.WRITE_FAILED, str)


class TestOptimizerStateMetadata:
    """Tests for OptimizerStateMetadata dataclass."""

    def test_required_fields(self):
        """Test required fields."""
        meta = OptimizerStateMetadata(
            type_name="AdamW",
            state_file="optimizer.safetensors",
            checksum="abc123def456",
        )

        assert meta.type_name == "AdamW"
        assert meta.state_file == "optimizer.safetensors"
        assert meta.checksum == "abc123def456"

    def test_default_hyperparameters(self):
        """Test default hyperparameters are empty dicts."""
        meta = OptimizerStateMetadata(
            type_name="SGD",
            state_file="opt.safetensors",
            checksum="xyz",
        )

        assert meta.scalar_hyperparameters == {}
        assert meta.vector_hyperparameters == {}

    def test_custom_hyperparameters(self):
        """Test custom hyperparameters."""
        meta = OptimizerStateMetadata(
            type_name="AdamW",
            state_file="opt.safetensors",
            checksum="abc",
            scalar_hyperparameters={"learning_rate": 3e-5, "weight_decay": 0.01},
            vector_hyperparameters={"betas": [0.9, 0.999]},
        )

        assert meta.scalar_hyperparameters["learning_rate"] == 3e-5
        assert meta.vector_hyperparameters["betas"] == [0.9, 0.999]

    def test_is_frozen(self):
        """Test dataclass is frozen (immutable)."""
        meta = OptimizerStateMetadata(
            type_name="AdamW",
            state_file="opt.safetensors",
            checksum="abc",
        )

        try:
            meta.type_name = "SGD"  # type: ignore
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass  # Expected

    def test_to_dict(self):
        """Test to_dict serialization."""
        meta = OptimizerStateMetadata(
            type_name="AdamW",
            state_file="opt.safetensors",
            checksum="abc123",
            scalar_hyperparameters={"lr": 1e-4},
            vector_hyperparameters={"betas": [0.9, 0.999]},
        )

        result = meta.to_dict()

        assert result["type_name"] == "AdamW"
        assert result["state_file"] == "opt.safetensors"
        assert result["checksum"] == "abc123"
        assert result["scalar_hyperparameters"] == {"lr": 1e-4}
        assert result["vector_hyperparameters"] == {"betas": [0.9, 0.999]}

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "type_name": "AdamW",
            "state_file": "optimizer.safetensors",
            "checksum": "xyz789",
            "scalar_hyperparameters": {"weight_decay": 0.01},
            "vector_hyperparameters": {},
        }

        meta = OptimizerStateMetadata.from_dict(data)

        assert meta.type_name == "AdamW"
        assert meta.state_file == "optimizer.safetensors"
        assert meta.checksum == "xyz789"
        assert meta.scalar_hyperparameters == {"weight_decay": 0.01}

    def test_from_dict_missing_fields(self):
        """Test from_dict handles missing fields with defaults."""
        data = {}

        meta = OptimizerStateMetadata.from_dict(data)

        assert meta.type_name == "unknown"
        assert meta.state_file == ""
        assert meta.checksum == ""

    def test_roundtrip(self):
        """Test to_dict/from_dict roundtrip."""
        original = OptimizerStateMetadata(
            type_name="AdamW",
            state_file="opt.safetensors",
            checksum="abc",
            scalar_hyperparameters={"lr": 1e-4},
            vector_hyperparameters={"betas": [0.9, 0.999]},
        )

        restored = OptimizerStateMetadata.from_dict(original.to_dict())

        assert restored.type_name == original.type_name
        assert restored.state_file == original.state_file
        assert restored.checksum == original.checksum
        assert restored.scalar_hyperparameters == original.scalar_hyperparameters
        assert restored.vector_hyperparameters == original.vector_hyperparameters


class TestFineTunedModelMetadata:
    """Tests for FineTunedModelMetadata dataclass."""

    def test_required_fields(self):
        """Test required fields."""
        meta = FineTunedModelMetadata(
            base_model_id="meta-llama/Llama-2-7b",
            tokenizer_strategy="llama",
        )

        assert meta.base_model_id == "meta-llama/Llama-2-7b"
        assert meta.tokenizer_strategy == "llama"

    def test_optional_fields_default_none(self):
        """Test optional fields default to None."""
        meta = FineTunedModelMetadata(
            base_model_id="test",
            tokenizer_strategy="default",
        )

        assert meta.lora_config is None
        assert meta.quantization_config is None
        assert meta.hyperparameters is None

    def test_optional_fields_set(self):
        """Test optional fields can be set."""
        meta = FineTunedModelMetadata(
            base_model_id="test",
            tokenizer_strategy="default",
            lora_config={"rank": 8, "alpha": 16},
            quantization_config={"bits": 4},
            hyperparameters={"batch_size": 4},
        )

        assert meta.lora_config == {"rank": 8, "alpha": 16}
        assert meta.quantization_config == {"bits": 4}
        assert meta.hyperparameters == {"batch_size": 4}

    def test_to_dict_minimal(self):
        """Test to_dict with only required fields."""
        meta = FineTunedModelMetadata(
            base_model_id="test-model",
            tokenizer_strategy="default",
        )

        result = meta.to_dict()

        assert result == {
            "base_model_id": "test-model",
            "tokenizer_strategy": "default",
        }

    def test_to_dict_full(self):
        """Test to_dict with all fields."""
        meta = FineTunedModelMetadata(
            base_model_id="test-model",
            tokenizer_strategy="llama",
            lora_config={"rank": 8},
            quantization_config={"bits": 4},
            hyperparameters={"lr": 1e-4},
        )

        result = meta.to_dict()

        assert result["base_model_id"] == "test-model"
        assert result["lora_config"] == {"rank": 8}
        assert result["quantization_config"] == {"bits": 4}
        assert result["hyperparameters"] == {"lr": 1e-4}

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "base_model_id": "llama-7b",
            "tokenizer_strategy": "llama",
            "lora_config": {"rank": 16},
        }

        meta = FineTunedModelMetadata.from_dict(data)

        assert meta.base_model_id == "llama-7b"
        assert meta.tokenizer_strategy == "llama"
        assert meta.lora_config == {"rank": 16}

    def test_from_dict_missing_fields(self):
        """Test from_dict handles missing fields."""
        data = {}

        meta = FineTunedModelMetadata.from_dict(data)

        assert meta.base_model_id == ""
        assert meta.tokenizer_strategy == "default"


class TestModelArchitectureSpec:
    """Tests for ModelArchitectureSpec dataclass."""

    def test_required_fields(self):
        """Test required fields."""
        config = ModelArchitectureSpec(
            model_type="llama",
            vocabulary_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
        )

        assert config.model_type == "llama"
        assert config.vocabulary_size == 32000
        assert config.hidden_size == 4096
        assert config.num_layers == 32
        assert config.num_heads == 32

    def test_memory_overrides_default(self):
        """Test memory_overrides defaults to None."""
        config = ModelArchitectureSpec(
            model_type="llama",
            vocabulary_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
        )

        assert config.memory_overrides is None

    def test_memory_overrides_set(self):
        """Test memory_overrides can be set."""
        config = ModelArchitectureSpec(
            model_type="llama",
            vocabulary_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            memory_overrides={"activation_checkpointing": True},
        )

        assert config.memory_overrides == {"activation_checkpointing": True}

    def test_to_dict_minimal(self):
        """Test to_dict without memory_overrides."""
        config = ModelArchitectureSpec(
            model_type="mistral",
            vocabulary_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
        )

        result = config.to_dict()

        assert result == {
            "model_type": "mistral",
            "vocabulary_size": 32000,
            "hidden_size": 4096,
            "num_layers": 32,
            "num_heads": 32,
        }

    def test_to_dict_with_overrides(self):
        """Test to_dict with memory_overrides."""
        config = ModelArchitectureSpec(
            model_type="llama",
            vocabulary_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            memory_overrides={"key": "value"},
        )

        result = config.to_dict()

        assert result["memory_overrides"] == {"key": "value"}

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "model_type": "qwen2",
            "vocabulary_size": 152064,
            "hidden_size": 3584,
            "num_layers": 28,
            "num_heads": 28,
        }

        config = ModelArchitectureSpec.from_dict(data)

        assert config.model_type == "qwen2"
        assert config.vocabulary_size == 152064
        assert config.hidden_size == 3584

    def test_from_dict_defaults(self):
        """Test from_dict uses defaults for missing fields."""
        data = {}

        config = ModelArchitectureSpec.from_dict(data)

        assert config.model_type == "simple_transformer"
        assert config.vocabulary_size == 32000
        assert config.hidden_size == 4096
        assert config.num_layers == 32
        assert config.num_heads == 32


class TestCheckpointMetadataV2:
    """Tests for CheckpointMetadataV2 dataclass."""

    def test_required_fields(self):
        """Test required fields."""
        now = datetime.now()
        meta = CheckpointMetadataV2(
            version=2,
            step=1000,
            total_steps=5000,
            timestamp=now,
            checksum="sha256hash",
            weights_file="checkpoint-1000.safetensors",
        )

        assert meta.version == 2
        assert meta.step == 1000
        assert meta.total_steps == 5000
        assert meta.timestamp == now
        assert meta.checksum == "sha256hash"
        assert meta.weights_file == "checkpoint-1000.safetensors"

    def test_default_loss_history(self):
        """Test loss_history defaults to empty list."""
        meta = CheckpointMetadataV2(
            version=2,
            step=0,
            total_steps=100,
            timestamp=datetime.now(),
            checksum="",
            weights_file="",
        )

        assert meta.loss_history == []

    def test_optional_nested_configs(self):
        """Test optional nested configuration fields."""
        model_config = ModelArchitectureSpec(
            model_type="llama",
            vocabulary_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
        )
        optimizer_state = OptimizerStateMetadata(
            type_name="AdamW",
            state_file="opt.safetensors",
            checksum="abc",
        )
        fine_tuned = FineTunedModelMetadata(
            base_model_id="llama-7b",
            tokenizer_strategy="llama",
        )

        meta = CheckpointMetadataV2(
            version=2,
            step=1000,
            total_steps=5000,
            timestamp=datetime.now(),
            checksum="hash",
            weights_file="weights.safetensors",
            model_config=model_config,
            optimizer_state=optimizer_state,
            fine_tuned_model=fine_tuned,
        )

        assert meta.model_config == model_config
        assert meta.optimizer_state == optimizer_state
        assert meta.fine_tuned_model == fine_tuned

    def test_to_dict_minimal(self):
        """Test to_dict with minimal fields."""
        now = datetime(2025, 1, 15, 10, 30, 0)
        meta = CheckpointMetadataV2(
            version=2,
            step=500,
            total_steps=1000,
            timestamp=now,
            checksum="abc123",
            weights_file="checkpoint.safetensors",
            loss_history=[1.0, 0.5, 0.3],
        )

        result = meta.to_dict()

        assert result["version"] == 2
        assert result["step"] == 500
        assert result["total_steps"] == 1000
        assert result["timestamp"] == "2025-01-15T10:30:00"
        assert result["checksum"] == "abc123"
        assert result["weights_file"] == "checkpoint.safetensors"
        assert result["loss_history"] == [1.0, 0.5, 0.3]

    def test_to_dict_with_nested(self):
        """Test to_dict includes nested configs."""
        model_config = ModelArchitectureSpec(
            model_type="llama",
            vocabulary_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
        )

        meta = CheckpointMetadataV2(
            version=2,
            step=500,
            total_steps=1000,
            timestamp=datetime.now(),
            checksum="abc",
            weights_file="weights.safetensors",
            model_config=model_config,
        )

        result = meta.to_dict()

        assert "model_config" in result
        assert result["model_config"]["model_type"] == "llama"

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "version": 2,
            "step": 1000,
            "total_steps": 5000,
            "timestamp": "2025-01-15T12:00:00",
            "checksum": "hash123",
            "weights_file": "checkpoint.safetensors",
            "loss_history": [1.0, 0.8, 0.6],
        }

        meta = CheckpointMetadataV2.from_dict(data)

        assert meta.version == 2
        assert meta.step == 1000
        assert meta.total_steps == 5000
        assert meta.timestamp == datetime(2025, 1, 15, 12, 0, 0)
        assert meta.checksum == "hash123"
        assert meta.loss_history == [1.0, 0.8, 0.6]

    def test_from_dict_with_nested(self):
        """Test from_dict with nested configurations."""
        data = {
            "version": 2,
            "step": 1000,
            "total_steps": 5000,
            "timestamp": "2025-01-15T12:00:00",
            "checksum": "hash",
            "weights_file": "weights.safetensors",
            "model_config": {
                "model_type": "mistral",
                "vocabulary_size": 32000,
                "hidden_size": 4096,
                "num_layers": 32,
                "num_heads": 32,
            },
            "optimizer_state": {
                "type_name": "AdamW",
                "state_file": "opt.safetensors",
                "checksum": "opt_hash",
            },
        }

        meta = CheckpointMetadataV2.from_dict(data)

        assert meta.model_config is not None
        assert meta.model_config.model_type == "mistral"
        assert meta.optimizer_state is not None
        assert meta.optimizer_state.type_name == "AdamW"

    def test_from_dict_defaults(self):
        """Test from_dict uses defaults for missing fields."""
        data = {}

        meta = CheckpointMetadataV2.from_dict(data)

        assert meta.version == 2
        assert meta.step == 0
        assert meta.total_steps == 0
        assert meta.checksum == ""
        assert meta.weights_file == ""
        assert meta.loss_history == []
        assert meta.model_config is None
        assert meta.optimizer_state is None

    def test_roundtrip(self):
        """Test to_dict/from_dict roundtrip."""
        model_config = ModelArchitectureSpec(
            model_type="llama",
            vocabulary_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
        )
        original = CheckpointMetadataV2(
            version=2,
            step=1000,
            total_steps=5000,
            timestamp=datetime(2025, 1, 15, 12, 0, 0),
            checksum="hash123",
            weights_file="checkpoint.safetensors",
            loss_history=[1.0, 0.8, 0.6],
            model_config=model_config,
        )

        restored = CheckpointMetadataV2.from_dict(original.to_dict())

        assert restored.version == original.version
        assert restored.step == original.step
        assert restored.timestamp == original.timestamp
        assert restored.loss_history == original.loss_history
        assert restored.model_config is not None
        assert restored.model_config.model_type == original.model_config.model_type


class TestRecoveryInfo:
    """Tests for RecoveryInfo dataclass."""

    def test_fields(self):
        """Test all fields are accessible."""
        checkpoint = CheckpointMetadataV2(
            version=2,
            step=1000,
            total_steps=5000,
            timestamp=datetime.now(),
            checksum="hash",
            weights_file="checkpoint.safetensors",
        )

        recovery = RecoveryInfo(
            checkpoint=checkpoint,
            checkpoints_dir=Path("/training/checkpoints"),
            output_dir=Path("/training/output"),
        )

        assert recovery.checkpoint == checkpoint
        assert recovery.checkpoints_dir == Path("/training/checkpoints")
        assert recovery.output_dir == Path("/training/output")

    def test_is_frozen(self):
        """Test dataclass is frozen (immutable)."""
        checkpoint = CheckpointMetadataV2(
            version=2,
            step=0,
            total_steps=100,
            timestamp=datetime.now(),
            checksum="",
            weights_file="",
        )

        recovery = RecoveryInfo(
            checkpoint=checkpoint,
            checkpoints_dir=Path("/a"),
            output_dir=Path("/b"),
        )

        try:
            recovery.output_dir = Path("/c")  # type: ignore
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass  # Expected
