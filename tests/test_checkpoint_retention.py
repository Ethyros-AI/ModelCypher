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

"""Tests for checkpoint retention policy (pruning old checkpoints)."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from modelcypher.core.domain.training.checkpoint_models import (
    CheckpointMetadataV2,
    OptimizerStateMetadata,
)
from modelcypher.core.domain.training.checkpoint_retention import CheckpointRetention


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def _make_metadata(step: int, total_steps: int = 1000) -> CheckpointMetadataV2:
    """Create a CheckpointMetadataV2 for testing."""
    return CheckpointMetadataV2(
        version=2,
        step=step,
        total_steps=total_steps,
        timestamp=datetime.now(),
        checksum=f"checksum-{step}",
        weights_file=f"checkpoint-{step}.safetensors",
    )


def _write_checkpoint_files(directory: Path, metadata: CheckpointMetadataV2) -> None:
    """Write checkpoint metadata and dummy weights file."""
    metadata_file = directory / f"checkpoint-{metadata.step}.json"
    weights_file = directory / metadata.weights_file

    with open(metadata_file, "w") as f:
        json.dump(metadata.to_dict(), f)

    weights_file.write_bytes(b"dummy weights data")


class TestCheckpointRetentionInit:
    """Tests for CheckpointRetention initialization."""

    def test_default_max_checkpoints(self):
        retention = CheckpointRetention()
        assert retention.max_checkpoints is None

    def test_custom_max_checkpoints(self):
        retention = CheckpointRetention(max_checkpoints=5)
        assert retention.max_checkpoints == 5

    def test_max_checkpoints_minimum_enforced(self):
        # Should enforce minimum of 1
        retention = CheckpointRetention(max_checkpoints=0)
        assert retention.max_checkpoints == 1

    def test_max_checkpoints_negative_becomes_1(self):
        retention = CheckpointRetention(max_checkpoints=-5)
        assert retention.max_checkpoints == 1

    def test_confirm_prune_default_false(self):
        retention = CheckpointRetention()
        assert retention._confirm_prune is False

    def test_confirm_prune_enabled(self):
        retention = CheckpointRetention(confirm_prune=True)
        assert retention._confirm_prune is True


class TestListCheckpoints:
    """Tests for list_checkpoints() method."""

    @pytest.mark.asyncio
    async def test_empty_directory(self, temp_dir):
        retention = CheckpointRetention()
        checkpoints = await retention.list_checkpoints(temp_dir)
        assert checkpoints == []

    @pytest.mark.asyncio
    async def test_nonexistent_directory(self):
        retention = CheckpointRetention()
        nonexistent = Path("/nonexistent/directory")
        checkpoints = await retention.list_checkpoints(nonexistent)
        assert checkpoints == []

    @pytest.mark.asyncio
    async def test_lists_checkpoint_metadata(self, temp_dir):
        retention = CheckpointRetention()

        # Create some checkpoint files
        for step in [100, 200, 300]:
            metadata = _make_metadata(step)
            _write_checkpoint_files(temp_dir, metadata)

        checkpoints = await retention.list_checkpoints(temp_dir)
        assert len(checkpoints) == 3
        steps = {c.step for c in checkpoints}
        assert steps == {100, 200, 300}

    @pytest.mark.asyncio
    async def test_skips_checkpoint_best(self, temp_dir):
        retention = CheckpointRetention()

        # Create a regular checkpoint
        metadata = _make_metadata(100)
        _write_checkpoint_files(temp_dir, metadata)

        # Create checkpoint-best.json (should be skipped)
        best_file = temp_dir / "checkpoint-best.json"
        best_file.write_text("{}")

        checkpoints = await retention.list_checkpoints(temp_dir)
        assert len(checkpoints) == 1
        assert checkpoints[0].step == 100

    @pytest.mark.asyncio
    async def test_skips_non_checkpoint_json(self, temp_dir):
        retention = CheckpointRetention()

        # Create a regular checkpoint
        metadata = _make_metadata(100)
        _write_checkpoint_files(temp_dir, metadata)

        # Create non-checkpoint JSON files
        (temp_dir / "config.json").write_text("{}")
        (temp_dir / "training_args.json").write_text("{}")

        checkpoints = await retention.list_checkpoints(temp_dir)
        assert len(checkpoints) == 1

    @pytest.mark.asyncio
    async def test_handles_corrupted_metadata(self, temp_dir):
        retention = CheckpointRetention()

        # Create valid checkpoint
        metadata = _make_metadata(100)
        _write_checkpoint_files(temp_dir, metadata)

        # Create corrupted checkpoint metadata
        corrupted = temp_dir / "checkpoint-200.json"
        corrupted.write_text("not valid json{{{")

        checkpoints = await retention.list_checkpoints(temp_dir)
        # Should only return the valid checkpoint
        assert len(checkpoints) == 1
        assert checkpoints[0].step == 100


class TestPruneOldCheckpoints:
    """Tests for prune_old_checkpoints() method."""

    @pytest.mark.asyncio
    async def test_no_pruning_when_under_limit(self, temp_dir):
        retention = CheckpointRetention(max_checkpoints=5)

        # Create 3 checkpoints (under limit of 5)
        for step in [100, 200, 300]:
            metadata = _make_metadata(step)
            _write_checkpoint_files(temp_dir, metadata)

        deleted = await retention.prune_old_checkpoints(temp_dir)
        assert deleted == 0

        # All checkpoints should still exist
        checkpoints = await retention.list_checkpoints(temp_dir)
        assert len(checkpoints) == 3

    @pytest.mark.asyncio
    async def test_no_pruning_when_at_limit(self, temp_dir):
        retention = CheckpointRetention(max_checkpoints=3)

        # Create exactly 3 checkpoints
        for step in [100, 200, 300]:
            metadata = _make_metadata(step)
            _write_checkpoint_files(temp_dir, metadata)

        deleted = await retention.prune_old_checkpoints(temp_dir)
        assert deleted == 0

    @pytest.mark.asyncio
    async def test_prunes_oldest_checkpoints(self, temp_dir):
        retention = CheckpointRetention(max_checkpoints=2)

        # Create 4 checkpoints
        for step in [100, 200, 300, 400]:
            metadata = _make_metadata(step)
            _write_checkpoint_files(temp_dir, metadata)

        deleted = await retention.prune_old_checkpoints(temp_dir)
        assert deleted == 2  # Should delete 100 and 200

        checkpoints = await retention.list_checkpoints(temp_dir)
        assert len(checkpoints) == 2
        steps = {c.step for c in checkpoints}
        assert steps == {300, 400}  # Kept newest

    @pytest.mark.asyncio
    async def test_deletes_weights_and_metadata_files(self, temp_dir):
        retention = CheckpointRetention(max_checkpoints=1)

        # Create 2 checkpoints
        for step in [100, 200]:
            metadata = _make_metadata(step)
            _write_checkpoint_files(temp_dir, metadata)

        await retention.prune_old_checkpoints(temp_dir)

        # Step 100 files should be deleted
        assert not (temp_dir / "checkpoint-100.json").exists()
        assert not (temp_dir / "checkpoint-100.safetensors").exists()

        # Step 200 files should still exist
        assert (temp_dir / "checkpoint-200.json").exists()
        assert (temp_dir / "checkpoint-200.safetensors").exists()

    @pytest.mark.asyncio
    async def test_custom_delete_function(self, temp_dir):
        deleted_checkpoints = []

        def custom_delete(metadata: CheckpointMetadataV2, directory: Path) -> None:
            deleted_checkpoints.append(metadata.step)

        retention = CheckpointRetention(max_checkpoints=1)

        for step in [100, 200, 300]:
            metadata = _make_metadata(step)
            _write_checkpoint_files(temp_dir, metadata)

        deleted = await retention.prune_old_checkpoints(temp_dir, delete_fn=custom_delete)
        assert deleted == 2
        assert set(deleted_checkpoints) == {100, 200}

    @pytest.mark.asyncio
    async def test_confirm_prune_calls_callback(self, temp_dir):
        callback_args = []

        def on_prune(dir_path: str, keep_count: int, delete_count: int) -> None:
            callback_args.append((dir_path, keep_count, delete_count))

        retention = CheckpointRetention(
            max_checkpoints=2,
            confirm_prune=True,
            on_prune_requested=on_prune,
        )

        for step in [100, 200, 300, 400]:
            metadata = _make_metadata(step)
            _write_checkpoint_files(temp_dir, metadata)

        # Should not actually delete, just call callback
        deleted = await retention.prune_old_checkpoints(temp_dir)
        assert deleted == 0

        # Callback should have been called
        assert len(callback_args) == 1
        dir_path, keep_count, delete_count = callback_args[0]
        assert dir_path == str(temp_dir)
        assert keep_count == 2
        assert delete_count == 2

        # All checkpoints should still exist
        checkpoints = await retention.list_checkpoints(temp_dir)
        assert len(checkpoints) == 4

    @pytest.mark.asyncio
    async def test_empty_directory_returns_zero(self, temp_dir):
        retention = CheckpointRetention(max_checkpoints=2)
        deleted = await retention.prune_old_checkpoints(temp_dir)
        assert deleted == 0

    @pytest.mark.asyncio
    async def test_handles_delete_failure(self, temp_dir):
        failure_count = [0]

        def failing_delete(metadata: CheckpointMetadataV2, directory: Path) -> None:
            failure_count[0] += 1
            raise RuntimeError("Delete failed")

        retention = CheckpointRetention(max_checkpoints=1)

        for step in [100, 200]:
            metadata = _make_metadata(step)
            _write_checkpoint_files(temp_dir, metadata)

        # Should handle failure gracefully
        deleted = await retention.prune_old_checkpoints(temp_dir, delete_fn=failing_delete)
        assert deleted == 0  # None succeeded
        assert failure_count[0] == 1  # Tried to delete one


class TestDefaultDelete:
    """Tests for _default_delete() method."""

    def test_deletes_weights_and_metadata(self, temp_dir):
        retention = CheckpointRetention()
        metadata = _make_metadata(100)
        _write_checkpoint_files(temp_dir, metadata)

        retention._default_delete(metadata, temp_dir)

        assert not (temp_dir / "checkpoint-100.json").exists()
        assert not (temp_dir / "checkpoint-100.safetensors").exists()

    def test_handles_missing_files(self, temp_dir):
        retention = CheckpointRetention()
        metadata = _make_metadata(100)
        # Don't create files - just try to delete

        # Should not raise
        retention._default_delete(metadata, temp_dir)

    def test_deletes_optimizer_state_file(self, temp_dir):
        retention = CheckpointRetention()

        optimizer_state = OptimizerStateMetadata(
            type_name="AdamW",
            state_file="optimizer-100.safetensors",
            checksum="opt-checksum",
        )
        metadata = CheckpointMetadataV2(
            version=2,
            step=100,
            total_steps=1000,
            timestamp=datetime.now(),
            checksum="checksum-100",
            weights_file="checkpoint-100.safetensors",
            optimizer_state=optimizer_state,
        )

        # Create all files
        _write_checkpoint_files(temp_dir, metadata)
        optimizer_file = temp_dir / "optimizer-100.safetensors"
        optimizer_file.write_bytes(b"optimizer state")

        retention._default_delete(metadata, temp_dir)

        assert not (temp_dir / "checkpoint-100.json").exists()
        assert not (temp_dir / "checkpoint-100.safetensors").exists()
        assert not optimizer_file.exists()


class TestSortingBehavior:
    """Tests for checkpoint sorting during pruning."""

    @pytest.mark.asyncio
    async def test_sorts_by_step_descending(self, temp_dir):
        retention = CheckpointRetention(max_checkpoints=2)

        # Create checkpoints in non-sequential order
        for step in [300, 100, 400, 200]:
            metadata = _make_metadata(step)
            _write_checkpoint_files(temp_dir, metadata)

        await retention.prune_old_checkpoints(temp_dir)

        # Should keep 400 and 300 (highest steps)
        checkpoints = await retention.list_checkpoints(temp_dir)
        steps = sorted([c.step for c in checkpoints], reverse=True)
        assert steps == [400, 300]

    @pytest.mark.asyncio
    async def test_keeps_newest_by_step_not_timestamp(self, temp_dir):
        retention = CheckpointRetention(max_checkpoints=1)

        # Create with mixed timestamps
        for step in [100, 200]:
            metadata = _make_metadata(step)
            _write_checkpoint_files(temp_dir, metadata)

        await retention.prune_old_checkpoints(temp_dir)

        # Step 200 should be kept (highest step)
        checkpoints = await retention.list_checkpoints(temp_dir)
        assert len(checkpoints) == 1
        assert checkpoints[0].step == 200
