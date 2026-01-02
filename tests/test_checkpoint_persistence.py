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

"""Tests for checkpoint persistence (atomic writes, disk space checks)."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest import mock

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.training.checkpoint_models import (
    CheckpointErrorKind,
    CheckpointMetadataV2,
)
from modelcypher.core.domain.training.checkpoint_persistence import (
    CheckpointPersistence,
)
from modelcypher.core.domain.training.exceptions import CheckpointError


@pytest.fixture
def persistence():
    """Create a CheckpointPersistence instance."""
    return CheckpointPersistence()


def _bytes_per_parameter() -> int:
    backend = get_default_backend()
    arr = backend.array([1.0])
    return int(backend.to_numpy(arr).dtype.itemsize)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestEstimateCheckpointSize:
    """Tests for estimate_checkpoint_size()."""

    def test_small_model(self, persistence):
        bytes_per_param = _bytes_per_parameter()
        size = persistence.estimate_checkpoint_size(1_000_000, bytes_per_param)
        assert size == 1_000_000 * bytes_per_param

    def test_large_model(self, persistence):
        bytes_per_param = _bytes_per_parameter()
        size = persistence.estimate_checkpoint_size(7_000_000_000, bytes_per_param)
        expected = 7_000_000_000 * bytes_per_param
        assert size == expected

    def test_zero_parameters(self, persistence):
        size = persistence.estimate_checkpoint_size(0, _bytes_per_parameter())
        assert size == 0

    def test_formula_is_bytes_per_param_plus_metadata(self, persistence):
        params = 1234567
        bytes_per_param = _bytes_per_parameter()
        metadata_bytes = 1024
        expected = params * bytes_per_param + metadata_bytes
        assert (
            persistence.estimate_checkpoint_size(params, bytes_per_param, metadata_bytes)
            == expected
        )


class TestEnsureSufficientSpace:
    """Tests for ensure_sufficient_space()."""

    def test_sufficient_space_no_error(self, persistence, temp_dir):
        # Request a small amount of space (should always have enough in temp dir)
        persistence.ensure_sufficient_space(1024, temp_dir, auto_prune=False)
        # Should not raise

    def test_insufficient_space_raises(self, persistence, temp_dir):
        # Mock disk_usage to return very little free space
        mock_usage = mock.Mock()
        mock_usage.free = 1000  # Only 1KB free
        with mock.patch("shutil.disk_usage", return_value=mock_usage):
            with pytest.raises(CheckpointError) as exc_info:
                persistence.ensure_sufficient_space(
                    1_000_000, temp_dir, auto_prune=False
                )
            assert "Insufficient disk space" in str(exc_info.value)

    def test_oserror_silently_passes(self, persistence, temp_dir):
        # If we can't check disk space, proceed anyway
        with mock.patch("shutil.disk_usage", side_effect=OSError("Can't check")):
            # Should not raise
            persistence.ensure_sufficient_space(1_000_000_000, temp_dir)


class TestAtomicWrite:
    """Tests for atomic_write()."""

    def test_writes_content_to_file(self, persistence, temp_dir):
        dest = temp_dir / "test_file.bin"
        content = b"Hello, world!"
        persistence.atomic_write(content, dest)
        assert dest.read_bytes() == content

    def test_no_temp_file_remains_on_success(self, persistence, temp_dir):
        dest = temp_dir / "test_file.bin"
        persistence.atomic_write(b"content", dest)
        # Temp file should not exist
        temp_file = dest.with_suffix(".bin.tmp")
        assert not temp_file.exists()

    def test_creates_parent_directory_if_exists(self, persistence, temp_dir):
        dest = temp_dir / "subdir" / "test_file.bin"
        dest.parent.mkdir(parents=True)
        persistence.atomic_write(b"content", dest)
        assert dest.read_bytes() == b"content"

    def test_overwrites_existing_file(self, persistence, temp_dir):
        dest = temp_dir / "test_file.bin"
        dest.write_bytes(b"old content")
        persistence.atomic_write(b"new content", dest)
        assert dest.read_bytes() == b"new content"


class TestAtomicWriteJson:
    """Tests for atomic_write_json()."""

    def test_writes_json_content(self, persistence, temp_dir):
        dest = temp_dir / "config.json"
        data = {"key": "value", "number": 42}
        persistence.atomic_write_json(data, dest)

        with open(dest) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_json_is_indented(self, persistence, temp_dir):
        dest = temp_dir / "config.json"
        data = {"key": "value"}
        persistence.atomic_write_json(data, dest, indent=4)

        content = dest.read_text()
        # Should have newlines for indentation
        assert "\n" in content

    def test_handles_datetime_serialization(self, persistence, temp_dir):
        dest = temp_dir / "config.json"
        data = {"timestamp": datetime.now()}
        # Should not raise (uses default=str)
        persistence.atomic_write_json(data, dest)
        assert dest.exists()


class TestSyncToDisk:
    """Tests for sync_to_disk()."""

    def test_sync_existing_file(self, persistence, temp_dir):
        test_file = temp_dir / "sync_test.txt"
        test_file.write_text("content")
        # Should not raise
        persistence.sync_to_disk(test_file)

    def test_sync_nonexistent_file_raises(self, persistence, temp_dir):
        nonexistent = temp_dir / "does_not_exist.txt"
        with pytest.raises(CheckpointError) as exc_info:
            persistence.sync_to_disk(nonexistent)
        assert "Failed to sync" in str(exc_info.value)


class TestDeleteCheckpoint:
    """Tests for delete_checkpoint()."""

    def _make_metadata(self, step: int = 100) -> CheckpointMetadataV2:
        """Create a CheckpointMetadataV2 for testing."""
        return CheckpointMetadataV2(
            version=2,
            step=step,
            total_steps=1000,
            timestamp=datetime.now(),
            checksum="abc123",
            weights_file=f"checkpoint-{step}.safetensors",
        )

    def test_deletes_weights_and_metadata(self, persistence, temp_dir):
        # Create dummy checkpoint files
        metadata = self._make_metadata(100)
        weights_file = temp_dir / "checkpoint-100.safetensors"
        metadata_file = temp_dir / "checkpoint-100.json"
        weights_file.write_bytes(b"weights")
        metadata_file.write_text("{}")

        persistence.delete_checkpoint(metadata, temp_dir)

        assert not weights_file.exists()
        assert not metadata_file.exists()

    def test_handles_missing_files(self, persistence, temp_dir):
        # Create metadata for non-existent files
        metadata = self._make_metadata(100)
        # Should not raise
        persistence.delete_checkpoint(metadata, temp_dir)
