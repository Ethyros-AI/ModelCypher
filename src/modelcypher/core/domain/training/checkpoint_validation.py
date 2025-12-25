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

"""Checkpoint integrity validator using SHA256 checksums.

Validates weights and optimizer state files for corruption detection.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from modelcypher.core.domain.training.checkpoint_models import CheckpointMetadataV2

logger = logging.getLogger(__name__)


class CheckpointValidation:
    """Checkpoint integrity validator using SHA256 checksums for weights and optimizer state."""

    # Chunk size for reading files (16MB)
    CHUNK_SIZE = 16 * 1024 * 1024

    @staticmethod
    def calculate_checksum(path: Path) -> str:
        """Calculate SHA256 checksum of a file using chunked reads.

        Performance: Reads 16MB chunks instead of byte-at-a-time for 1000x+ speedup on large files.

        Args:
            path: File path to checksum.

        Returns:
            Hexadecimal SHA256 checksum string.

        Raises:
            FileNotFoundError: If file doesn't exist.
            IOError: If file can't be read.
        """
        sha256 = hashlib.sha256()

        with open(path, "rb") as f:
            while True:
                chunk = f.read(CheckpointValidation.CHUNK_SIZE)
                if not chunk:
                    break
                sha256.update(chunk)

        return sha256.hexdigest()

    @staticmethod
    async def calculate_checksum_async(path: Path) -> str:
        """Calculate SHA256 checksum asynchronously.

        Args:
            path: File path to checksum.

        Returns:
            Hexadecimal SHA256 checksum string.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, CheckpointValidation.calculate_checksum, path)

    @staticmethod
    def validate_checkpoint(metadata: CheckpointMetadataV2, directory: Path) -> bool:
        """Validate checkpoint integrity (checksum verification).

        Verifies:
        - Weights file exists
        - Weights checksum matches metadata
        - Optimizer state file exists (if present in metadata)
        - Optimizer state checksum matches (if present)

        Args:
            metadata: Checkpoint metadata containing expected checksums.
            directory: Directory containing checkpoint files.

        Returns:
            True if all integrity checks pass, False otherwise.
        """
        weights_file = directory / metadata.weights_file

        if not weights_file.exists():
            logger.warning(f"Checkpoint weights file missing: {metadata.weights_file}")
            return False

        try:
            actual_checksum = CheckpointValidation.calculate_checksum(weights_file)
        except (IOError, OSError) as e:
            logger.warning(f"Failed to read weights file: {e}")
            return False

        if actual_checksum != metadata.checksum:
            logger.warning(
                f"Checkpoint checksum mismatch: expected={metadata.checksum[:8]}, "
                f"actual={actual_checksum[:8]}"
            )
            return False

        # Validate optimizer state if present
        if metadata.optimizer_state is not None:
            optimizer_file = directory / metadata.optimizer_state.state_file

            if not optimizer_file.exists():
                logger.warning(
                    f"Optimizer state file missing: {metadata.optimizer_state.state_file}"
                )
                return False

            try:
                optimizer_checksum = CheckpointValidation.calculate_checksum(optimizer_file)
            except (IOError, OSError) as e:
                logger.warning(f"Failed to read optimizer state file: {e}")
                return False

            if optimizer_checksum != metadata.optimizer_state.checksum:
                logger.warning(
                    f"Optimizer checksum mismatch: expected={metadata.optimizer_state.checksum[:8]}, "
                    f"actual={optimizer_checksum[:8]}"
                )
                return False

        return True

    @staticmethod
    async def validate_checkpoint_async(metadata: CheckpointMetadataV2, directory: Path) -> bool:
        """Validate checkpoint integrity asynchronously.

        Args:
            metadata: Checkpoint metadata containing expected checksums.
            directory: Directory containing checkpoint files.

        Returns:
            True if all integrity checks pass, False otherwise.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, CheckpointValidation.validate_checkpoint, metadata, directory
        )
