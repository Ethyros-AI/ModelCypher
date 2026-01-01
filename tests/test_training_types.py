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

"""Tests for training types module."""

from __future__ import annotations

from datetime import datetime

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

    def test_float32_value(self):
        """Test FLOAT32 value."""
        assert ComputePrecision.FLOAT32.value == "float32"

    def test_float16_value(self):
        """Test FLOAT16 value."""
        assert ComputePrecision.FLOAT16.value == "float16"

    def test_bfloat16_value(self):
        """Test BFLOAT16 value."""
        assert ComputePrecision.BFLOAT16.value == "bfloat16"

    def test_is_string_enum(self):
        """Test ComputePrecision is a string enum."""
        assert isinstance(ComputePrecision.FLOAT32, str)
        assert ComputePrecision.FLOAT32 == "float32"


class TestTrainingStatus:
    """Tests for TrainingStatus enum."""

    def test_pending_value(self):
        """Test pending status."""
        assert TrainingStatus.pending.value == "pending"

    def test_running_value(self):
        """Test running status."""
        assert TrainingStatus.running.value == "running"

    def test_paused_value(self):
        """Test paused status."""
        assert TrainingStatus.paused.value == "paused"

    def test_completed_value(self):
        """Test completed status."""
        assert TrainingStatus.completed.value == "completed"

    def test_failed_value(self):
        """Test failed status."""
        assert TrainingStatus.failed.value == "failed"

    def test_cancelled_value(self):
        """Test cancelled status."""
        assert TrainingStatus.cancelled.value == "cancelled"

    def test_is_string_enum(self):
        """Test TrainingStatus is a string enum."""
        assert isinstance(TrainingStatus.running, str)
        assert TrainingStatus.running == "running"


class TestPreflightResult:
    """Tests for PreflightResult dataclass."""

    def test_fields(self):
        """Test all fields are accessible."""
        result = PreflightResult(
            predicted_batch_size=4,
            estimated_vram_bytes=8_000_000_000,
            available_vram_bytes=16_000_000_000,
            can_proceed=True,
        )

        assert result.predicted_batch_size == 4
        assert result.estimated_vram_bytes == 8_000_000_000
        assert result.available_vram_bytes == 16_000_000_000
        assert result.can_proceed is True

    def test_can_proceed_false(self):
        """Test can_proceed=False case."""
        result = PreflightResult(
            predicted_batch_size=1,
            estimated_vram_bytes=32_000_000_000,
            available_vram_bytes=16_000_000_000,
            can_proceed=False,
        )

        assert result.can_proceed is False


class TestHyperparameters:
    """Tests for Hyperparameters dataclass."""

    def test_requires_explicit_values(self):
        """Hyperparameters require explicit values (no implicit defaults)."""
        try:
            Hyperparameters()
        except TypeError:
            assert True
        else:
            assert False, "Hyperparameters should require explicit values."

    def test_custom_values(self):
        """Test custom values override defaults."""
        hp = Hyperparameters(
            batch_size=8,
            learning_rate=1e-4,
            epochs=10,
            sequence_length=2048,
            gradient_accumulation_steps=2,
            gradient_checkpointing=False,
            mixed_precision=False,
            compute_precision=ComputePrecision.BFLOAT16,
            warmup_steps=50,
            weight_decay=0.1,
            seed=123,
            deterministic=False,
            optimizer_type="adamw",
        )

        assert hp.batch_size == 8
        assert hp.learning_rate == 1e-4
        assert hp.epochs == 10
        assert hp.sequence_length == 2048
        assert hp.compute_precision == ComputePrecision.BFLOAT16


class TestLoRAConfig:
    """Tests for LoRAConfig dataclass."""

    def test_requires_explicit_values(self):
        """LoRAConfig requires explicit values (no implicit defaults)."""
        try:
            LoRAConfig()
        except TypeError:
            assert True
        else:
            assert False, "LoRAConfig should require explicit values."

    def test_custom_values(self):
        """Test custom values override defaults."""
        config = LoRAConfig(
            rank=16,
            alpha=32.0,
            dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

        assert config.rank == 16
        assert config.alpha == 32.0
        assert config.dropout == 0.1
        assert config.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]

    def test_target_modules_is_mutable_default(self):
        """Test that target_modules is not shared across configs."""
        config1 = LoRAConfig(
            rank=8,
            alpha=16.0,
            dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )
        config2 = LoRAConfig(
            rank=8,
            alpha=16.0,
            dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )

        # Modify config1's list
        config1.target_modules.append("o_proj")

        # config2 should be unaffected
        assert config2.target_modules == ["q_proj", "v_proj"]


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_required_fields(self):
        """Test required fields."""
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
        config = TrainingConfig(
            model_id="meta-llama/Llama-2-7b",
            dataset_path="/data/train.jsonl",
            output_path="/output/model",
            hyperparameters=hp,
        )

        assert config.model_id == "meta-llama/Llama-2-7b"
        assert config.dataset_path == "/data/train.jsonl"
        assert config.output_path == "/output/model"
        assert config.hyperparameters == hp

    def test_optional_lora_config(self):
        """Test optional lora_config field."""
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
        lora = LoRAConfig(
            rank=8,
            alpha=16.0,
            dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )
        config = TrainingConfig(
            model_id="test-model",
            dataset_path="/data/train.jsonl",
            output_path="/output",
            hyperparameters=hp,
            lora_config=lora,
        )

        assert config.lora_config == lora

    def test_optional_resume_checkpoint(self):
        """Test optional resume_from_checkpoint_path field."""
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
        config = TrainingConfig(
            model_id="test-model",
            dataset_path="/data/train.jsonl",
            output_path="/output",
            hyperparameters=hp,
            resume_from_checkpoint_path="/checkpoints/step-1000",
        )

        assert config.resume_from_checkpoint_path == "/checkpoints/step-1000"

    def test_defaults_are_none(self):
        """Test optional fields default to None."""
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
        config = TrainingConfig(
            model_id="test-model",
            dataset_path="/data/train.jsonl",
            output_path="/output",
            hyperparameters=hp,
        )

        assert config.lora_config is None
        assert config.resume_from_checkpoint_path is None


class TestTrainingProgress:
    """Tests for TrainingProgress dataclass."""

    def test_required_fields(self):
        """Test required fields."""
        progress = TrainingProgress(
            job_id="job-123",
            epoch=1,
            step=100,
            total_steps=1000,
            loss=0.5,
            learning_rate=3e-5,
        )

        assert progress.job_id == "job-123"
        assert progress.epoch == 1
        assert progress.step == 100
        assert progress.total_steps == 1000
        assert progress.loss == 0.5
        assert progress.learning_rate == 3e-5

    def test_optional_fields(self):
        """Test optional fields."""
        progress = TrainingProgress(
            job_id="job-123",
            epoch=2,
            step=500,
            total_steps=1000,
            loss=0.3,
            learning_rate=1e-5,
            tokens_per_second=1500.0,
            estimated_time_remaining=3600.0,
            metrics={"accuracy": 0.95, "perplexity": 2.5},
        )

        assert progress.tokens_per_second == 1500.0
        assert progress.estimated_time_remaining == 3600.0
        assert progress.metrics == {"accuracy": 0.95, "perplexity": 2.5}

    def test_defaults(self):
        """Test default values for optional fields."""
        progress = TrainingProgress(
            job_id="job-123",
            epoch=1,
            step=0,
            total_steps=100,
            loss=1.0,
            learning_rate=3e-5,
        )

        assert progress.tokens_per_second is None
        assert progress.estimated_time_remaining is None
        assert progress.metrics == {}


class TestCheckpointMetadata:
    """Tests for CheckpointMetadata dataclass."""

    def test_required_fields(self):
        """Test required fields."""
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
        train_config = TrainingConfig(
            model_id="test-model",
            dataset_path="/data/train.jsonl",
            output_path="/output",
            hyperparameters=hp,
        )
        now = datetime.now()

        metadata = CheckpointMetadata(
            version=1,
            step=500,
            total_steps=1000,
            train_config=train_config,
            loss_history=[1.0, 0.8, 0.6, 0.5],
            timestamp=now,
            checksum="abc123",
            weights_file="checkpoint-500.safetensors",
        )

        assert metadata.version == 1
        assert metadata.step == 500
        assert metadata.total_steps == 1000
        assert metadata.train_config == train_config
        assert metadata.loss_history == [1.0, 0.8, 0.6, 0.5]
        assert metadata.timestamp == now
        assert metadata.checksum == "abc123"
        assert metadata.weights_file == "checkpoint-500.safetensors"

    def test_optional_optimizer_file(self):
        """Test optional optimizer_file field."""
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
        train_config = TrainingConfig(
            model_id="test-model",
            dataset_path="/data/train.jsonl",
            output_path="/output",
            hyperparameters=hp,
        )

        metadata = CheckpointMetadata(
            version=1,
            step=500,
            total_steps=1000,
            train_config=train_config,
            loss_history=[],
            timestamp=datetime.now(),
            checksum="abc123",
            weights_file="checkpoint.safetensors",
            optimizer_file="optimizer-500.safetensors",
        )

        assert metadata.optimizer_file == "optimizer-500.safetensors"

    def test_optimizer_file_default_none(self):
        """Test optimizer_file defaults to None."""
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
        train_config = TrainingConfig(
            model_id="test-model",
            dataset_path="/data/train.jsonl",
            output_path="/output",
            hyperparameters=hp,
        )

        metadata = CheckpointMetadata(
            version=1,
            step=0,
            total_steps=100,
            train_config=train_config,
            loss_history=[],
            timestamp=datetime.now(),
            checksum="",
            weights_file="",
        )

        assert metadata.optimizer_file is None
