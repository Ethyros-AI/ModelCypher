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

"""Tests for checkpoint integrity validation (SHA256 checksums)."""

import hashlib
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from modelcypher.core.domain.training.checkpoint_models import (
    CheckpointMetadataV2,
    OptimizerStateMetadata,
)
from modelcypher.core.domain.training.checkpoint_validation import CheckpointValidation


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def _sha256(data: bytes) -> str:
    """Calculate SHA256 hex digest for test data."""
    return hashlib.sha256(data).hexdigest()


def _make_metadata(
    step: int,
    checksum: str,
    weights_file: str | None = None,
    optimizer_state: OptimizerStateMetadata | None = None,
) -> CheckpointMetadataV2:
    """Create a CheckpointMetadataV2 for testing."""
    return CheckpointMetadataV2(
        version=2,
        step=step,
        total_steps=1000,
        timestamp=datetime.now(),
        checksum=checksum,
        weights_file=weights_file or f"checkpoint-{step}.safetensors",
        optimizer_state=optimizer_state,
    )


class TestChunkSize:
    """Tests for CHUNK_SIZE constant."""

    def test_chunk_size_is_16mb(self):
        assert CheckpointValidation.CHUNK_SIZE == 16 * 1024 * 1024

    def test_chunk_size_in_bytes(self):
        # 16MB = 16,777,216 bytes
        assert CheckpointValidation.CHUNK_SIZE == 16777216


class TestCalculateChecksum:
    """Tests for calculate_checksum() method."""

    def test_calculates_sha256(self, temp_dir):
        test_file = temp_dir / "test.bin"
        content = b"Hello, world!"
        test_file.write_bytes(content)

        checksum = CheckpointValidation.calculate_checksum(test_file)
        expected = _sha256(content)

        assert checksum == expected

    def test_empty_file_checksum(self, temp_dir):
        test_file = temp_dir / "empty.bin"
        test_file.write_bytes(b"")

        checksum = CheckpointValidation.calculate_checksum(test_file)
        expected = _sha256(b"")

        assert checksum == expected

    def test_large_file_checksum(self, temp_dir):
        # Create a file larger than CHUNK_SIZE (16MB)
        # Use 1MB for faster test
        test_file = temp_dir / "large.bin"
        content = b"x" * (1024 * 1024)  # 1MB
        test_file.write_bytes(content)

        checksum = CheckpointValidation.calculate_checksum(test_file)
        expected = _sha256(content)

        assert checksum == expected

    def test_raises_for_missing_file(self, temp_dir):
        missing_file = temp_dir / "does_not_exist.bin"

        with pytest.raises(FileNotFoundError):
            CheckpointValidation.calculate_checksum(missing_file)

    def test_returns_hex_string(self, temp_dir):
        test_file = temp_dir / "test.bin"
        test_file.write_bytes(b"test")

        checksum = CheckpointValidation.calculate_checksum(test_file)

        # SHA256 hex digest is 64 characters
        assert len(checksum) == 64
        # All characters should be hex
        assert all(c in "0123456789abcdef" for c in checksum)

    def test_deterministic(self, temp_dir):
        test_file = temp_dir / "test.bin"
        test_file.write_bytes(b"deterministic content")

        checksum1 = CheckpointValidation.calculate_checksum(test_file)
        checksum2 = CheckpointValidation.calculate_checksum(test_file)

        assert checksum1 == checksum2


class TestCalculateChecksumAsync:
    """Tests for calculate_checksum_async() method."""

    @pytest.mark.asyncio
    async def test_async_matches_sync(self, temp_dir):
        test_file = temp_dir / "test.bin"
        content = b"Async test content"
        test_file.write_bytes(content)

        sync_checksum = CheckpointValidation.calculate_checksum(test_file)
        async_checksum = await CheckpointValidation.calculate_checksum_async(test_file)

        assert async_checksum == sync_checksum

    @pytest.mark.asyncio
    async def test_async_raises_for_missing_file(self, temp_dir):
        missing_file = temp_dir / "does_not_exist.bin"

        with pytest.raises(FileNotFoundError):
            await CheckpointValidation.calculate_checksum_async(missing_file)


class TestValidateCheckpoint:
    """Tests for validate_checkpoint() method."""

    def test_valid_checkpoint_returns_true(self, temp_dir):
        # Create weights file
        weights_content = b"model weights data"
        weights_file = temp_dir / "checkpoint-100.safetensors"
        weights_file.write_bytes(weights_content)

        # Create metadata with correct checksum
        metadata = _make_metadata(
            step=100,
            checksum=_sha256(weights_content),
        )

        result = CheckpointValidation.validate_checkpoint(metadata, temp_dir)
        assert result is True

    def test_missing_weights_file_returns_false(self, temp_dir):
        metadata = _make_metadata(
            step=100,
            checksum="abc123",
        )

        result = CheckpointValidation.validate_checkpoint(metadata, temp_dir)
        assert result is False

    def test_checksum_mismatch_returns_false(self, temp_dir):
        # Create weights file
        weights_file = temp_dir / "checkpoint-100.safetensors"
        weights_file.write_bytes(b"actual content")

        # Create metadata with wrong checksum
        metadata = _make_metadata(
            step=100,
            checksum="wrong_checksum_value",
        )

        result = CheckpointValidation.validate_checkpoint(metadata, temp_dir)
        assert result is False

    def test_valid_with_optimizer_state(self, temp_dir):
        # Create weights file
        weights_content = b"model weights"
        weights_file = temp_dir / "checkpoint-100.safetensors"
        weights_file.write_bytes(weights_content)

        # Create optimizer state file
        optimizer_content = b"optimizer state data"
        optimizer_file = temp_dir / "optimizer-100.safetensors"
        optimizer_file.write_bytes(optimizer_content)

        optimizer_state = OptimizerStateMetadata(
            type_name="AdamW",
            state_file="optimizer-100.safetensors",
            checksum=_sha256(optimizer_content),
        )

        metadata = _make_metadata(
            step=100,
            checksum=_sha256(weights_content),
            optimizer_state=optimizer_state,
        )

        result = CheckpointValidation.validate_checkpoint(metadata, temp_dir)
        assert result is True

    def test_missing_optimizer_state_file_returns_false(self, temp_dir):
        # Create only weights file
        weights_content = b"model weights"
        weights_file = temp_dir / "checkpoint-100.safetensors"
        weights_file.write_bytes(weights_content)

        optimizer_state = OptimizerStateMetadata(
            type_name="AdamW",
            state_file="optimizer-100.safetensors",
            checksum="some_checksum",
        )

        metadata = _make_metadata(
            step=100,
            checksum=_sha256(weights_content),
            optimizer_state=optimizer_state,
        )

        result = CheckpointValidation.validate_checkpoint(metadata, temp_dir)
        assert result is False

    def test_optimizer_state_checksum_mismatch_returns_false(self, temp_dir):
        # Create weights file
        weights_content = b"model weights"
        weights_file = temp_dir / "checkpoint-100.safetensors"
        weights_file.write_bytes(weights_content)

        # Create optimizer state file with different content
        optimizer_file = temp_dir / "optimizer-100.safetensors"
        optimizer_file.write_bytes(b"actual optimizer content")

        optimizer_state = OptimizerStateMetadata(
            type_name="AdamW",
            state_file="optimizer-100.safetensors",
            checksum="wrong_optimizer_checksum",
        )

        metadata = _make_metadata(
            step=100,
            checksum=_sha256(weights_content),
            optimizer_state=optimizer_state,
        )

        result = CheckpointValidation.validate_checkpoint(metadata, temp_dir)
        assert result is False


class TestValidateCheckpointAsync:
    """Tests for validate_checkpoint_async() method."""

    @pytest.mark.asyncio
    async def test_async_valid_checkpoint(self, temp_dir):
        weights_content = b"async model weights"
        weights_file = temp_dir / "checkpoint-100.safetensors"
        weights_file.write_bytes(weights_content)

        metadata = _make_metadata(
            step=100,
            checksum=_sha256(weights_content),
        )

        result = await CheckpointValidation.validate_checkpoint_async(metadata, temp_dir)
        assert result is True

    @pytest.mark.asyncio
    async def test_async_invalid_checkpoint(self, temp_dir):
        weights_file = temp_dir / "checkpoint-100.safetensors"
        weights_file.write_bytes(b"content")

        metadata = _make_metadata(
            step=100,
            checksum="wrong_checksum",
        )

        result = await CheckpointValidation.validate_checkpoint_async(metadata, temp_dir)
        assert result is False


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_validates_correct_weights_file_name(self, temp_dir):
        # Create weights with custom name
        custom_name = "custom-weights.safetensors"
        weights_content = b"weights"
        (temp_dir / custom_name).write_bytes(weights_content)

        metadata = _make_metadata(
            step=100,
            checksum=_sha256(weights_content),
            weights_file=custom_name,
        )

        result = CheckpointValidation.validate_checkpoint(metadata, temp_dir)
        assert result is True

    def test_different_file_same_checksum_impossible(self, temp_dir):
        # Two different files will have different checksums
        file1 = temp_dir / "file1.bin"
        file2 = temp_dir / "file2.bin"
        file1.write_bytes(b"content A")
        file2.write_bytes(b"content B")

        checksum1 = CheckpointValidation.calculate_checksum(file1)
        checksum2 = CheckpointValidation.calculate_checksum(file2)

        assert checksum1 != checksum2

    def test_binary_content_checksum(self, temp_dir):
        # Test with binary content (not just text)
        test_file = temp_dir / "binary.bin"
        binary_content = bytes(range(256))  # All possible byte values
        test_file.write_bytes(binary_content)

        checksum = CheckpointValidation.calculate_checksum(test_file)
        expected = _sha256(binary_content)

        assert checksum == expected
