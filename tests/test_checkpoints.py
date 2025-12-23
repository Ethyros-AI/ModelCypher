"""Tests for CheckpointManager.

Tests the training checkpoint persistence system including:
- Atomic checkpoint writes
- SHA-256 checksum validation
- Optimizer state serialization
- Best checkpoint alias
- Retention-based pruning
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import mlx.core as mx
import pytest

from modelcypher.core.domain.training.checkpoints import (
    CheckpointManager,
    CheckpointError,
    InsufficientDiskSpaceError,
    MIN_DISK_SPACE_BYTES,
)
from modelcypher.core.domain.training.types import (
    CheckpointMetadata,
    TrainingConfig,
    Hyperparameters,
    LoRAConfig,
    ComputePrecision,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_weights() -> Dict[str, mx.array]:
    """Sample model weights for testing."""
    return {
        "model.layers.0.self_attn.q_proj.weight": mx.random.normal((64, 64)),
        "model.layers.0.self_attn.v_proj.weight": mx.random.normal((64, 64)),
        "model.embed_tokens.weight": mx.random.normal((1000, 64)),
    }


@pytest.fixture
def sample_config() -> TrainingConfig:
    """Sample training config for testing."""
    return TrainingConfig(
        model_id="test-model",
        dataset_path="/path/to/data",
        output_path="/path/to/output",
        hyperparameters=Hyperparameters(
            batch_size=4,
            learning_rate=3e-5,
            epochs=3,
            sequence_length=512,
        ),
        lora_config=LoRAConfig(
            rank=8,
            alpha=16.0,
            dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        ),
    )


@pytest.fixture
def sample_optimizer_state() -> Dict[str, Any]:
    """Sample optimizer state for testing."""
    return {
        "state": {
            "model.layers.0.self_attn.q_proj.weight": {
                "m": mx.zeros((64, 64)),
                "v": mx.zeros((64, 64)),
            },
        },
        "step": 100,
    }


class TestCheckpointManagerInit:
    """Tests for CheckpointManager initialization."""

    def test_default_max_checkpoints(self):
        """Default max checkpoints is 3."""
        manager = CheckpointManager()
        assert manager.max_checkpoints == 3

    def test_custom_max_checkpoints(self):
        """Custom max checkpoints is respected."""
        manager = CheckpointManager(max_checkpoints=5)
        assert manager.max_checkpoints == 5

    def test_initial_best_loss(self):
        """Initial best loss is infinity."""
        manager = CheckpointManager()
        assert manager._best_loss == float("inf")
        assert manager._best_step == -1


class TestSaveCheckpoint:
    """Tests for save_checkpoint method."""

    @pytest.mark.asyncio
    async def test_saves_weights_file(self, temp_dir, sample_weights, sample_config):
        """Checkpoint creates weights file."""
        manager = CheckpointManager()

        await manager.save_checkpoint(
            model_weights=sample_weights,
            optimizer_state=None,
            step=100,
            total_steps=1000,
            loss_history=[1.0, 0.5, 0.3],
            config=sample_config,
            output_dir=temp_dir,
        )

        weights_path = Path(temp_dir) / "checkpoints" / "checkpoint-100.safetensors"
        assert weights_path.exists()

    @pytest.mark.asyncio
    async def test_saves_metadata_file(self, temp_dir, sample_weights, sample_config):
        """Checkpoint creates metadata JSON file."""
        manager = CheckpointManager()

        await manager.save_checkpoint(
            model_weights=sample_weights,
            optimizer_state=None,
            step=100,
            total_steps=1000,
            loss_history=[1.0, 0.5],
            config=sample_config,
            output_dir=temp_dir,
        )

        metadata_path = Path(temp_dir) / "checkpoints" / "checkpoint-100.json"
        assert metadata_path.exists()

        with open(metadata_path) as f:
            data = json.load(f)

        assert data["step"] == 100
        assert data["total_steps"] == 1000
        assert data["version"] == 2

    @pytest.mark.asyncio
    async def test_saves_optimizer_state(
        self, temp_dir, sample_weights, sample_config, sample_optimizer_state
    ):
        """Checkpoint saves optimizer state when provided."""
        manager = CheckpointManager()

        await manager.save_checkpoint(
            model_weights=sample_weights,
            optimizer_state=sample_optimizer_state,
            step=100,
            total_steps=1000,
            loss_history=[1.0],
            config=sample_config,
            output_dir=temp_dir,
        )

        optimizer_path = Path(temp_dir) / "checkpoints" / "optimizer-100.safetensors"
        assert optimizer_path.exists()

    @pytest.mark.asyncio
    async def test_returns_metadata(self, temp_dir, sample_weights, sample_config):
        """save_checkpoint returns CheckpointMetadata."""
        manager = CheckpointManager()

        metadata = await manager.save_checkpoint(
            model_weights=sample_weights,
            optimizer_state=None,
            step=100,
            total_steps=1000,
            loss_history=[1.0, 0.5],
            config=sample_config,
            output_dir=temp_dir,
        )

        assert isinstance(metadata, CheckpointMetadata)
        assert metadata.step == 100
        assert metadata.total_steps == 1000
        assert metadata.version == 2
        assert len(metadata.checksum) == 64  # SHA-256 hex

    @pytest.mark.asyncio
    async def test_sanitizes_loss_history(self, temp_dir, sample_weights, sample_config):
        """NaN and Inf values are filtered from loss history."""
        manager = CheckpointManager()

        loss_with_nan = [1.0, float("nan"), 0.5, float("inf"), 0.3]

        metadata = await manager.save_checkpoint(
            model_weights=sample_weights,
            optimizer_state=None,
            step=100,
            total_steps=1000,
            loss_history=loss_with_nan,
            config=sample_config,
            output_dir=temp_dir,
        )

        assert len(metadata.loss_history) == 3
        assert 1.0 in metadata.loss_history
        assert 0.5 in metadata.loss_history
        assert 0.3 in metadata.loss_history

    @pytest.mark.asyncio
    async def test_atomic_write_cleans_temp_dir(
        self, temp_dir, sample_weights, sample_config
    ):
        """Temp directory is cleaned up after save."""
        manager = CheckpointManager()

        await manager.save_checkpoint(
            model_weights=sample_weights,
            optimizer_state=None,
            step=100,
            total_steps=1000,
            loss_history=[1.0],
            config=sample_config,
            output_dir=temp_dir,
        )

        tmp_dir = Path(temp_dir) / "checkpoints" / ".tmp"
        # Either doesn't exist or is empty
        if tmp_dir.exists():
            assert len(list(tmp_dir.iterdir())) == 0


class TestBestCheckpointAlias:
    """Tests for best checkpoint alias functionality."""

    @pytest.mark.asyncio
    async def test_updates_best_alias(self, temp_dir, sample_weights, sample_config):
        """Best alias is created for lowest loss."""
        manager = CheckpointManager()

        await manager.save_checkpoint(
            model_weights=sample_weights,
            optimizer_state=None,
            step=100,
            total_steps=1000,
            loss_history=[1.0],
            config=sample_config,
            output_dir=temp_dir,
        )

        best_path = Path(temp_dir) / "checkpoints" / "checkpoint-best.json"
        assert best_path.exists()

        with open(best_path) as f:
            data = json.load(f)

        assert data["step"] == 100
        assert data["loss"] == 1.0

    @pytest.mark.asyncio
    async def test_best_alias_updates_on_lower_loss(
        self, temp_dir, sample_weights, sample_config
    ):
        """Best alias updates when new checkpoint has lower loss."""
        manager = CheckpointManager()

        # First checkpoint with higher loss
        await manager.save_checkpoint(
            model_weights=sample_weights,
            optimizer_state=None,
            step=100,
            total_steps=1000,
            loss_history=[1.0],
            config=sample_config,
            output_dir=temp_dir,
        )

        # Second checkpoint with lower loss
        await manager.save_checkpoint(
            model_weights=sample_weights,
            optimizer_state=None,
            step=200,
            total_steps=1000,
            loss_history=[0.5],
            config=sample_config,
            output_dir=temp_dir,
        )

        best_path = Path(temp_dir) / "checkpoints" / "checkpoint-best.json"
        with open(best_path) as f:
            data = json.load(f)

        assert data["step"] == 200
        assert data["loss"] == 0.5

    @pytest.mark.asyncio
    async def test_best_alias_not_updated_on_higher_loss(
        self, temp_dir, sample_weights, sample_config
    ):
        """Best alias does not update when new checkpoint has higher loss."""
        manager = CheckpointManager()

        await manager.save_checkpoint(
            model_weights=sample_weights,
            optimizer_state=None,
            step=100,
            total_steps=1000,
            loss_history=[0.5],
            config=sample_config,
            output_dir=temp_dir,
        )

        await manager.save_checkpoint(
            model_weights=sample_weights,
            optimizer_state=None,
            step=200,
            total_steps=1000,
            loss_history=[1.0],  # Higher loss
            config=sample_config,
            output_dir=temp_dir,
        )

        best_path = Path(temp_dir) / "checkpoints" / "checkpoint-best.json"
        with open(best_path) as f:
            data = json.load(f)

        assert data["step"] == 100  # Still best


class TestCheckpointPruning:
    """Tests for checkpoint retention/pruning."""

    @pytest.mark.asyncio
    async def test_prunes_old_checkpoints(self, temp_dir, sample_weights, sample_config):
        """Old checkpoints are pruned when exceeding max."""
        manager = CheckpointManager(max_checkpoints=2)

        # Create 4 checkpoints
        for step in [100, 200, 300, 400]:
            await manager.save_checkpoint(
                model_weights=sample_weights,
                optimizer_state=None,
                step=step,
                total_steps=1000,
                loss_history=[1.0 / step],  # Lower loss = better
                config=sample_config,
                output_dir=temp_dir,
            )

        checkpoints_dir = Path(temp_dir) / "checkpoints"
        checkpoint_files = list(checkpoints_dir.glob("checkpoint-*.json"))

        # Should keep best + 2 most recent (may overlap)
        # At minimum, only 2 regular checkpoints plus best alias
        regular_checkpoints = [
            f for f in checkpoint_files if f.name != "checkpoint-best.json"
        ]
        assert len(regular_checkpoints) <= 3  # max_checkpoints + best

    @pytest.mark.asyncio
    async def test_best_checkpoint_not_pruned(
        self, temp_dir, sample_weights, sample_config
    ):
        """Best checkpoint is never pruned."""
        manager = CheckpointManager(max_checkpoints=1)

        # Create checkpoint with best loss
        await manager.save_checkpoint(
            model_weights=sample_weights,
            optimizer_state=None,
            step=100,
            total_steps=1000,
            loss_history=[0.1],  # Best loss
            config=sample_config,
            output_dir=temp_dir,
        )

        # Create more checkpoints with worse loss
        for step in [200, 300, 400]:
            await manager.save_checkpoint(
                model_weights=sample_weights,
                optimizer_state=None,
                step=step,
                total_steps=1000,
                loss_history=[1.0],  # Worse loss
                config=sample_config,
                output_dir=temp_dir,
            )

        # Best checkpoint should still exist
        best_path = Path(temp_dir) / "checkpoints" / "checkpoint-100.safetensors"
        assert best_path.exists()


class TestLoadCheckpoint:
    """Tests for loading checkpoints."""

    @pytest.mark.asyncio
    async def test_load_latest_checkpoint(
        self, temp_dir, sample_weights, sample_config
    ):
        """load_latest_checkpoint returns most recent checkpoint."""
        manager = CheckpointManager()

        await manager.save_checkpoint(
            model_weights=sample_weights,
            optimizer_state=None,
            step=100,
            total_steps=1000,
            loss_history=[1.0],
            config=sample_config,
            output_dir=temp_dir,
        )

        await manager.save_checkpoint(
            model_weights=sample_weights,
            optimizer_state=None,
            step=200,
            total_steps=1000,
            loss_history=[0.5],
            config=sample_config,
            output_dir=temp_dir,
        )

        metadata = await manager.load_latest_checkpoint(temp_dir)

        # Should load best (lowest loss = step 200)
        assert metadata is not None
        assert metadata.step == 200

    @pytest.mark.asyncio
    async def test_load_latest_returns_none_if_no_checkpoints(self, temp_dir):
        """load_latest_checkpoint returns None if no checkpoints exist."""
        manager = CheckpointManager()
        metadata = await manager.load_latest_checkpoint(temp_dir)
        assert metadata is None

    @pytest.mark.asyncio
    async def test_load_checkpoint_metadata(
        self, temp_dir, sample_weights, sample_config
    ):
        """load_checkpoint_metadata loads specific step."""
        manager = CheckpointManager()

        await manager.save_checkpoint(
            model_weights=sample_weights,
            optimizer_state=None,
            step=100,
            total_steps=1000,
            loss_history=[1.0],
            config=sample_config,
            output_dir=temp_dir,
        )

        checkpoints_dir = Path(temp_dir) / "checkpoints"
        metadata = await manager.load_checkpoint_metadata(str(checkpoints_dir), 100)

        assert metadata.step == 100
        assert metadata.total_steps == 1000
        assert metadata.train_config is not None
        assert metadata.train_config.model_id == "test-model"

    @pytest.mark.asyncio
    async def test_load_weights(self, temp_dir, sample_weights, sample_config):
        """load_weights returns saved model weights."""
        manager = CheckpointManager()

        await manager.save_checkpoint(
            model_weights=sample_weights,
            optimizer_state=None,
            step=100,
            total_steps=1000,
            loss_history=[1.0],
            config=sample_config,
            output_dir=temp_dir,
        )

        checkpoints_dir = Path(temp_dir) / "checkpoints"
        loaded = await manager.load_weights(str(checkpoints_dir), 100)

        # Weights are flattened, so check first key exists
        assert len(loaded) > 0
        # Verify a known key exists
        assert any("q_proj" in k for k in loaded.keys())

    @pytest.mark.asyncio
    async def test_load_optimizer_state(
        self, temp_dir, sample_weights, sample_config, sample_optimizer_state
    ):
        """load_optimizer_state returns saved optimizer state."""
        manager = CheckpointManager()

        await manager.save_checkpoint(
            model_weights=sample_weights,
            optimizer_state=sample_optimizer_state,
            step=100,
            total_steps=1000,
            loss_history=[1.0],
            config=sample_config,
            output_dir=temp_dir,
        )

        checkpoints_dir = Path(temp_dir) / "checkpoints"
        loaded = await manager.load_optimizer_state(str(checkpoints_dir), 100)

        assert loaded is not None
        assert len(loaded) > 0

    @pytest.mark.asyncio
    async def test_load_optimizer_state_returns_none_if_missing(
        self, temp_dir, sample_weights, sample_config
    ):
        """load_optimizer_state returns None if no optimizer was saved."""
        manager = CheckpointManager()

        await manager.save_checkpoint(
            model_weights=sample_weights,
            optimizer_state=None,  # No optimizer
            step=100,
            total_steps=1000,
            loss_history=[1.0],
            config=sample_config,
            output_dir=temp_dir,
        )

        checkpoints_dir = Path(temp_dir) / "checkpoints"
        loaded = await manager.load_optimizer_state(str(checkpoints_dir), 100)

        assert loaded is None


class TestChecksumValidation:
    """Tests for checksum computation."""

    @pytest.mark.asyncio
    async def test_checksum_is_sha256_hex(self, temp_dir, sample_weights, sample_config):
        """Checksum is 64-character hex string (SHA-256)."""
        manager = CheckpointManager()

        metadata = await manager.save_checkpoint(
            model_weights=sample_weights,
            optimizer_state=None,
            step=100,
            total_steps=1000,
            loss_history=[1.0],
            config=sample_config,
            output_dir=temp_dir,
        )

        assert len(metadata.checksum) == 64
        assert all(c in "0123456789abcdef" for c in metadata.checksum)

    @pytest.mark.asyncio
    async def test_checksum_consistent(self, temp_dir, sample_weights, sample_config):
        """Same weights produce same checksum."""
        manager = CheckpointManager()

        # Force MLX to evaluate deterministically
        weights = {k: mx.zeros((4, 4)) for k in ["layer.weight"]}

        meta1 = await manager.save_checkpoint(
            model_weights=weights,
            optimizer_state=None,
            step=100,
            total_steps=1000,
            loss_history=[1.0],
            config=sample_config,
            output_dir=temp_dir,
        )

        # Different step, same weights
        meta2 = await manager.save_checkpoint(
            model_weights=weights,
            optimizer_state=None,
            step=200,
            total_steps=1000,
            loss_history=[1.0],
            config=sample_config,
            output_dir=temp_dir,
        )

        assert meta1.checksum == meta2.checksum


class TestFlattenWeights:
    """Tests for weight flattening."""

    def test_flattens_nested_dict(self):
        """Nested weight dicts are flattened."""
        manager = CheckpointManager()

        nested = {
            "model": {
                "layers": {
                    "0": {"weight": mx.zeros((2, 2))},
                }
            }
        }

        flat = manager._flatten_weights(nested)

        assert "model.layers.0.weight" in flat
        assert isinstance(flat["model.layers.0.weight"], mx.array)

    def test_flat_dict_unchanged(self):
        """Already flat dict is preserved."""
        manager = CheckpointManager()

        flat_input = {"layer.weight": mx.zeros((2, 2))}
        flat = manager._flatten_weights(flat_input)

        assert "layer.weight" in flat


class TestExtractOptimizerArrays:
    """Tests for optimizer array extraction."""

    def test_extracts_arrays_from_nested_state(self):
        """MLX arrays are extracted from nested optimizer state."""
        manager = CheckpointManager()

        state = {
            "state": {
                "layer1": {"m": mx.zeros((2, 2)), "v": mx.ones((2, 2))},
            },
            "step": 100,
        }

        arrays = manager._extract_optimizer_arrays(state)

        assert "state.layer1.m" in arrays
        assert "state.layer1.v" in arrays
        assert isinstance(arrays["state.layer1.m"], mx.array)

    def test_extracts_arrays_from_list(self):
        """MLX arrays in lists are extracted with indices."""
        manager = CheckpointManager()

        state = {
            "params": [mx.zeros((2, 2)), mx.ones((2, 2))],
        }

        arrays = manager._extract_optimizer_arrays(state)

        assert "params.0" in arrays
        assert "params.1" in arrays


class TestDeserializeConfig:
    """Tests for config deserialization."""

    def test_deserializes_full_config(self):
        """Complete config is deserialized correctly."""
        manager = CheckpointManager()

        data = {
            "model_id": "test-model",
            "dataset_path": "/data",
            "output_path": "/output",
            "hyperparameters": {
                "batch_size": 8,
                "learning_rate": 1e-4,
                "epochs": 5,
                "sequence_length": 1024,
            },
            "lora_config": {
                "rank": 16,
                "alpha": 32.0,
                "dropout": 0.1,
                "target_modules": ["q_proj", "k_proj"],
            },
        }

        config = manager._deserialize_config(data)

        assert config is not None
        assert config.model_id == "test-model"
        assert config.hyperparameters.batch_size == 8
        assert config.lora_config.rank == 16

    def test_deserialize_none_returns_none(self):
        """None input returns None."""
        manager = CheckpointManager()
        assert manager._deserialize_config(None) is None

    def test_deserialize_uses_defaults(self):
        """Missing fields use defaults."""
        manager = CheckpointManager()

        data = {"model_id": "test"}  # Minimal data

        config = manager._deserialize_config(data)

        assert config is not None
        assert config.hyperparameters.batch_size == 4  # Default


class TestEdgeCases:
    """Edge case tests."""

    @pytest.mark.asyncio
    async def test_empty_loss_history(self, temp_dir, sample_weights, sample_config):
        """Empty loss history doesn't crash."""
        manager = CheckpointManager()

        metadata = await manager.save_checkpoint(
            model_weights=sample_weights,
            optimizer_state=None,
            step=100,
            total_steps=1000,
            loss_history=[],
            config=sample_config,
            output_dir=temp_dir,
        )

        assert metadata.loss_history == []

    @pytest.mark.asyncio
    async def test_empty_weights(self, temp_dir, sample_config):
        """Empty weights dict is handled."""
        manager = CheckpointManager()

        metadata = await manager.save_checkpoint(
            model_weights={},
            optimizer_state=None,
            step=100,
            total_steps=1000,
            loss_history=[1.0],
            config=sample_config,
            output_dir=temp_dir,
        )

        assert metadata is not None

    @pytest.mark.asyncio
    async def test_step_zero(self, temp_dir, sample_weights, sample_config):
        """Step 0 is valid."""
        manager = CheckpointManager()

        metadata = await manager.save_checkpoint(
            model_weights=sample_weights,
            optimizer_state=None,
            step=0,
            total_steps=1000,
            loss_history=[1.0],
            config=sample_config,
            output_dir=temp_dir,
        )

        assert metadata.step == 0
        assert (Path(temp_dir) / "checkpoints" / "checkpoint-0.safetensors").exists()

    @pytest.mark.asyncio
    async def test_overwrites_existing_step(
        self, temp_dir, sample_weights, sample_config
    ):
        """Saving to same step overwrites."""
        manager = CheckpointManager()

        await manager.save_checkpoint(
            model_weights=sample_weights,
            optimizer_state=None,
            step=100,
            total_steps=1000,
            loss_history=[1.0],
            config=sample_config,
            output_dir=temp_dir,
        )

        # Save again with different loss
        await manager.save_checkpoint(
            model_weights=sample_weights,
            optimizer_state=None,
            step=100,
            total_steps=1000,
            loss_history=[0.5],
            config=sample_config,
            output_dir=temp_dir,
        )

        # Should only have one checkpoint-100
        checkpoints_dir = Path(temp_dir) / "checkpoints"
        step_100_files = list(checkpoints_dir.glob("checkpoint-100.*"))
        assert len(step_100_files) == 2  # .json and .safetensors


class TestSerializeMetadata:
    """Tests for metadata serialization."""

    def test_serializes_datetime(self):
        """datetime objects are serialized to ISO format."""
        manager = CheckpointManager()

        metadata = CheckpointMetadata(
            version=2,
            step=100,
            total_steps=1000,
            train_config=TrainingConfig(
                model_id="test",
                dataset_path="/data",
                output_path="/out",
                hyperparameters=Hyperparameters(),
            ),
            loss_history=[1.0],
            timestamp=datetime(2024, 1, 15, 12, 30, 45),
            checksum="abc123",
            weights_file="weights.safetensors",
        )

        json_str = manager._serialize_metadata(metadata)
        data = json.loads(json_str)

        assert "2024-01-15T12:30:45" in data["timestamp"]

    def test_serializes_enums(self):
        """Enum values are serialized by value."""
        manager = CheckpointManager()

        metadata = CheckpointMetadata(
            version=2,
            step=100,
            total_steps=1000,
            train_config=TrainingConfig(
                model_id="test",
                dataset_path="/data",
                output_path="/out",
                hyperparameters=Hyperparameters(
                    compute_precision=ComputePrecision.FLOAT16
                ),
            ),
            loss_history=[1.0],
            timestamp=datetime.now(),
            checksum="abc123",
            weights_file="weights.safetensors",
        )

        json_str = manager._serialize_metadata(metadata)
        data = json.loads(json_str)

        # Check that compute_precision is serialized as value, not enum
        hp = data.get("train_config", {}).get("hyperparameters", {})
        if "compute_precision" in hp:
            assert hp["compute_precision"] == "float16"
