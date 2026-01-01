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

"""Tests for training configuration and checkpoint models.

These tests verify configuration dataclasses, serialization,
checkpoint validation, and error handling without requiring GPU hardware.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from modelcypher.core.domain.training.checkpoint_models import (
    CheckpointErrorKind,
    CheckpointMetadataV2,
    FineTunedModelMetadata,
    ModelArchitectureConfig,
    OptimizerStateMetadata,
    RecoveryInfo,
)
from modelcypher.core.domain.training.checkpoint_validation import CheckpointValidation
from modelcypher.core.domain.training.types import (
    CheckpointMetadata,
    ComputePrecision,
    Hyperparameters,
    LoRAConfig,
    PreflightResult,
    TrainingConfig,
    TrainingProgress,
    TrainingStatus,
)


class TestComputePrecision:
    """Tests for ComputePrecision enum."""

    def test_enum_values(self):
        """All precision types should have string values."""
        assert ComputePrecision.FLOAT32.value == "float32"
        assert ComputePrecision.FLOAT16.value == "float16"
        assert ComputePrecision.BFLOAT16.value == "bfloat16"

    def test_enum_string_conversion(self):
        """Precision can be compared to strings."""
        assert ComputePrecision.FLOAT32 == "float32"
        assert ComputePrecision.FLOAT16 == "float16"


class TestTrainingStatus:
    """Tests for TrainingStatus enum."""

    def test_all_statuses(self):
        """All training statuses should be defined."""
        statuses = [s.value for s in TrainingStatus]
        assert "pending" in statuses
        assert "running" in statuses
        assert "paused" in statuses
        assert "completed" in statuses
        assert "failed" in statuses
        assert "cancelled" in statuses


class TestHyperparameters:
    """Tests for Hyperparameters dataclass."""

    def test_requires_explicit_values(self):
        """Hyperparameters require explicit values (no implicit defaults)."""
        with pytest.raises(TypeError):
            Hyperparameters()

    def test_custom_values(self):
        """Custom hyperparameters should be preserved."""
        hp = Hyperparameters(
            batch_size=8,
            learning_rate=1e-4,
            epochs=5,
            sequence_length=2048,
            gradient_accumulation_steps=4,
            gradient_checkpointing=False,
            mixed_precision=False,
            compute_precision=ComputePrecision.BFLOAT16,
            warmup_steps=100,
            weight_decay=0.1,
            seed=123,
            deterministic=False,
            optimizer_type="adam",
        )

        assert hp.batch_size == 8
        assert hp.learning_rate == 1e-4
        assert hp.epochs == 5
        assert hp.compute_precision == ComputePrecision.BFLOAT16
        assert hp.optimizer_type == "adam"


class TestLoRAConfig:
    """Tests for LoRAConfig dataclass."""

    def test_requires_explicit_values(self):
        """LoRAConfig requires explicit values (no implicit defaults)."""
        with pytest.raises(TypeError):
            LoRAConfig()

    def test_custom_values(self):
        """Custom LoRA config should be preserved."""
        cfg = LoRAConfig(
            rank=16,
            alpha=32.0,
            dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

        assert cfg.rank == 16
        assert cfg.alpha == 32.0
        assert cfg.dropout == 0.1
        assert len(cfg.target_modules) == 4


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_minimal_config(self):
        """Training config with minimal required fields."""
        hp = Hyperparameters(
            batch_size=4,
            learning_rate=3e-5,
            epochs=3,
            sequence_length=1024,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            mixed_precision=True,
            compute_precision=ComputePrecision.FLOAT16,
            warmup_steps=10,
            weight_decay=0.01,
            seed=42,
            deterministic=True,
            optimizer_type="adamw",
        )
        cfg = TrainingConfig(
            model_id="meta-llama/Llama-2-7b",
            dataset_path="/path/to/dataset",
            output_path="/path/to/output",
            hyperparameters=hp,
        )

        assert cfg.model_id == "meta-llama/Llama-2-7b"
        assert cfg.lora_config is None
        assert cfg.resume_from_checkpoint_path is None

    def test_full_config(self):
        """Training config with all optional fields."""
        hp = Hyperparameters(
            batch_size=4,
            learning_rate=3e-5,
            epochs=10,
            sequence_length=1024,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            mixed_precision=True,
            compute_precision=ComputePrecision.FLOAT16,
            warmup_steps=10,
            weight_decay=0.01,
            seed=42,
            deterministic=True,
            optimizer_type="adamw",
        )
        lora = LoRAConfig(
            rank=32,
            alpha=16.0,
            dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )
        cfg = TrainingConfig(
            model_id="meta-llama/Llama-2-7b",
            dataset_path="/path/to/dataset",
            output_path="/path/to/output",
            hyperparameters=hp,
            lora_config=lora,
            resume_from_checkpoint_path="/path/to/checkpoint",
        )

        assert cfg.lora_config is not None
        assert cfg.lora_config.rank == 32
        assert cfg.resume_from_checkpoint_path == "/path/to/checkpoint"


class TestTrainingProgress:
    """Tests for TrainingProgress dataclass."""

    def test_minimal_progress(self):
        """Progress with minimal fields."""
        prog = TrainingProgress(
            job_id="job-123",
            epoch=1,
            step=100,
            total_steps=1000,
            loss=2.5,
            learning_rate=3e-5,
        )

        assert prog.job_id == "job-123"
        assert prog.epoch == 1
        assert prog.step == 100
        assert prog.total_steps == 1000
        assert prog.loss == 2.5
        assert prog.tokens_per_second is None
        assert prog.estimated_time_remaining is None
        assert prog.metrics == {}

    def test_full_progress(self):
        """Progress with all fields."""
        prog = TrainingProgress(
            job_id="job-456",
            epoch=3,
            step=500,
            total_steps=1000,
            loss=1.5,
            learning_rate=1e-5,
            tokens_per_second=1500.0,
            estimated_time_remaining=3600.0,
            metrics={"eval_loss": 1.6, "accuracy": 0.85},
        )

        assert prog.tokens_per_second == 1500.0
        assert prog.estimated_time_remaining == 3600.0
        assert prog.metrics["accuracy"] == 0.85


class TestPreflightResult:
    """Tests for PreflightResult dataclass."""

    def test_can_proceed(self):
        """Preflight with sufficient resources."""
        result = PreflightResult(
            predicted_batch_size=4,
            estimated_vram_bytes=8 * 1024**3,  # 8GB
            available_vram_bytes=16 * 1024**3,  # 16GB
            can_proceed=True,
        )

        assert result.can_proceed is True
        assert result.available_vram_bytes > result.estimated_vram_bytes

    def test_cannot_proceed(self):
        """Preflight with insufficient resources."""
        result = PreflightResult(
            predicted_batch_size=1,
            estimated_vram_bytes=32 * 1024**3,  # 32GB
            available_vram_bytes=8 * 1024**3,  # 8GB
            can_proceed=False,
        )

        assert result.can_proceed is False


class TestCheckpointErrorKind:
    """Tests for CheckpointErrorKind enum."""

    def test_all_error_kinds(self):
        """All error kinds should be defined."""
        kinds = [k.value for k in CheckpointErrorKind]
        assert "insufficient_disk_space" in kinds
        assert "write_failed" in kinds
        assert "no_valid_checkpoints" in kinds
        assert "checksum_mismatch" in kinds
        assert "missing_file" in kinds


class TestOptimizerStateMetadata:
    """Tests for OptimizerStateMetadata dataclass."""

    def test_minimal_state(self):
        """Optimizer state with minimal fields."""
        state = OptimizerStateMetadata(
            type_name="AdamW",
            state_file="optimizer.safetensors",
            checksum="abc123",
        )

        assert state.type_name == "AdamW"
        assert state.scalar_hyperparameters == {}
        assert state.vector_hyperparameters == {}

    def test_to_dict_from_dict_roundtrip(self):
        """Serialization and deserialization should be consistent."""
        state = OptimizerStateMetadata(
            type_name="AdamW",
            state_file="optimizer.safetensors",
            checksum="abc123def456",
            scalar_hyperparameters={"learning_rate": 3e-5, "weight_decay": 0.01},
            vector_hyperparameters={"betas": [0.9, 0.999]},
        )

        data = state.to_dict()
        restored = OptimizerStateMetadata.from_dict(data)

        assert restored.type_name == state.type_name
        assert restored.state_file == state.state_file
        assert restored.checksum == state.checksum
        assert restored.scalar_hyperparameters == state.scalar_hyperparameters
        assert restored.vector_hyperparameters == state.vector_hyperparameters


class TestModelArchitectureConfig:
    """Tests for ModelArchitectureConfig dataclass."""

    def test_llama_config(self):
        """Typical LLaMA architecture config."""
        cfg = ModelArchitectureConfig(
            model_type="llama",
            vocabulary_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
        )

        assert cfg.model_type == "llama"
        assert cfg.vocabulary_size == 32000
        assert cfg.hidden_size == 4096
        assert cfg.memory_overrides is None

    def test_to_dict_from_dict_roundtrip(self):
        """Serialization and deserialization should be consistent."""
        cfg = ModelArchitectureConfig(
            model_type="mistral",
            vocabulary_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            memory_overrides={"kv_cache_size": 1024},
        )

        data = cfg.to_dict()
        restored = ModelArchitectureConfig.from_dict(data)

        assert restored.model_type == cfg.model_type
        assert restored.hidden_size == cfg.hidden_size
        assert restored.memory_overrides == cfg.memory_overrides


class TestFineTunedModelMetadata:
    """Tests for FineTunedModelMetadata dataclass."""

    def test_minimal_metadata(self):
        """Fine-tuned model with minimal fields."""
        meta = FineTunedModelMetadata(
            base_model_id="meta-llama/Llama-2-7b",
            tokenizer_strategy="default",
        )

        assert meta.base_model_id == "meta-llama/Llama-2-7b"
        assert meta.lora_config is None
        assert meta.quantization_config is None
        assert meta.hyperparameters is None

    def test_to_dict_from_dict_roundtrip(self):
        """Serialization and deserialization should be consistent."""
        meta = FineTunedModelMetadata(
            base_model_id="meta-llama/Llama-2-7b",
            tokenizer_strategy="chat",
            lora_config={"rank": 8, "alpha": 16},
            quantization_config={"bits": 4, "group_size": 128},
            hyperparameters={"learning_rate": 3e-5},
        )

        data = meta.to_dict()
        restored = FineTunedModelMetadata.from_dict(data)

        assert restored.base_model_id == meta.base_model_id
        assert restored.lora_config == meta.lora_config
        assert restored.quantization_config == meta.quantization_config


class TestCheckpointMetadataV2:
    """Tests for CheckpointMetadataV2 dataclass."""

    def test_minimal_checkpoint(self):
        """Checkpoint with minimal required fields."""
        now = datetime.now()
        ckpt = CheckpointMetadataV2(
            version=2,
            step=1000,
            total_steps=5000,
            timestamp=now,
            checksum="abc123",
            weights_file="checkpoint-1000.safetensors",
        )

        assert ckpt.version == 2
        assert ckpt.step == 1000
        assert ckpt.loss_history == []
        assert ckpt.model_config is None
        assert ckpt.optimizer_state is None

    def test_to_dict_from_dict_roundtrip(self):
        """Serialization and deserialization should be consistent."""
        now = datetime.now()
        model_cfg = ModelArchitectureConfig(
            model_type="llama",
            vocabulary_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
        )
        opt_state = OptimizerStateMetadata(
            type_name="AdamW",
            state_file="optimizer.safetensors",
            checksum="opt_checksum",
        )

        ckpt = CheckpointMetadataV2(
            version=2,
            step=1000,
            total_steps=5000,
            timestamp=now,
            checksum="abc123def456",
            weights_file="checkpoint-1000.safetensors",
            loss_history=[2.5, 2.0, 1.5],
            model_config=model_cfg,
            optimizer_state=opt_state,
        )

        data = ckpt.to_dict()
        restored = CheckpointMetadataV2.from_dict(data)

        assert restored.version == ckpt.version
        assert restored.step == ckpt.step
        assert restored.total_steps == ckpt.total_steps
        assert restored.checksum == ckpt.checksum
        assert restored.weights_file == ckpt.weights_file
        assert restored.loss_history == ckpt.loss_history
        assert restored.model_config is not None
        assert restored.model_config.model_type == model_cfg.model_type
        assert restored.optimizer_state is not None
        assert restored.optimizer_state.type_name == opt_state.type_name


class TestCheckpointValidation:
    """Tests for CheckpointValidation class."""

    def test_calculate_checksum(self):
        """Checksum calculation should be deterministic."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content for checksum")
            f.flush()
            path = Path(f.name)

        try:
            checksum1 = CheckpointValidation.calculate_checksum(path)
            checksum2 = CheckpointValidation.calculate_checksum(path)

            assert checksum1 == checksum2
            assert len(checksum1) == 64  # SHA256 hex length
        finally:
            path.unlink()

    def test_validate_checkpoint_missing_weights(self):
        """Validation should fail if weights file is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)

            metadata = CheckpointMetadataV2(
                version=2,
                step=100,
                total_steps=1000,
                timestamp=datetime.now(),
                checksum="expected_checksum",
                weights_file="missing.safetensors",
            )

            result = CheckpointValidation.validate_checkpoint(metadata, directory)
            assert result is False

    def test_validate_checkpoint_checksum_mismatch(self):
        """Validation should fail if checksum doesn't match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)

            # Create weights file
            weights_file = directory / "checkpoint.safetensors"
            weights_file.write_bytes(b"checkpoint weights data")

            metadata = CheckpointMetadataV2(
                version=2,
                step=100,
                total_steps=1000,
                timestamp=datetime.now(),
                checksum="wrong_checksum",
                weights_file="checkpoint.safetensors",
            )

            result = CheckpointValidation.validate_checkpoint(metadata, directory)
            assert result is False

    def test_validate_checkpoint_success(self):
        """Validation should pass with correct checksum."""
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)

            # Create weights file
            weights_file = directory / "checkpoint.safetensors"
            content = b"checkpoint weights data"
            weights_file.write_bytes(content)

            # Calculate actual checksum
            checksum = CheckpointValidation.calculate_checksum(weights_file)

            metadata = CheckpointMetadataV2(
                version=2,
                step=100,
                total_steps=1000,
                timestamp=datetime.now(),
                checksum=checksum,
                weights_file="checkpoint.safetensors",
            )

            result = CheckpointValidation.validate_checkpoint(metadata, directory)
            assert result is True

    def test_validate_checkpoint_with_optimizer_state(self):
        """Validation should check optimizer state if present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)

            # Create weights file
            weights_file = directory / "checkpoint.safetensors"
            weights_content = b"checkpoint weights data"
            weights_file.write_bytes(weights_content)
            weights_checksum = CheckpointValidation.calculate_checksum(weights_file)

            # Create optimizer file
            optimizer_file = directory / "optimizer.safetensors"
            optimizer_content = b"optimizer state data"
            optimizer_file.write_bytes(optimizer_content)
            optimizer_checksum = CheckpointValidation.calculate_checksum(optimizer_file)

            opt_state = OptimizerStateMetadata(
                type_name="AdamW",
                state_file="optimizer.safetensors",
                checksum=optimizer_checksum,
            )

            metadata = CheckpointMetadataV2(
                version=2,
                step=100,
                total_steps=1000,
                timestamp=datetime.now(),
                checksum=weights_checksum,
                weights_file="checkpoint.safetensors",
                optimizer_state=opt_state,
            )

            result = CheckpointValidation.validate_checkpoint(metadata, directory)
            assert result is True


class TestRecoveryInfo:
    """Tests for RecoveryInfo dataclass."""

    def test_recovery_info_creation(self):
        """RecoveryInfo should hold checkpoint and paths."""
        ckpt = CheckpointMetadataV2(
            version=2,
            step=500,
            total_steps=1000,
            timestamp=datetime.now(),
            checksum="abc",
            weights_file="checkpoint-500.safetensors",
        )

        info = RecoveryInfo(
            checkpoint=ckpt,
            checkpoints_dir=Path("/train/checkpoints"),
            output_dir=Path("/train/output"),
        )

        assert info.checkpoint.step == 500
        assert info.checkpoints_dir == Path("/train/checkpoints")
        assert info.output_dir == Path("/train/output")
