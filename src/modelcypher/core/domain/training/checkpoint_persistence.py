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

"""Atomic checkpoint writer using temp-file/rename, fsync, and disk-space checks.

Provides crash-safe persistence for training checkpoints.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from modelcypher.core.domain.training.checkpoint_models import (
    CheckpointErrorKind,
    CheckpointMetadataV2,
)
from modelcypher.core.domain.training.checkpoint_retention import CheckpointRetention
from modelcypher.core.domain.training.exceptions import CheckpointError

logger = logging.getLogger(__name__)

# Minimum required disk space in bytes (500MB)
MIN_DISK_SPACE_BYTES = 500 * 1024 * 1024


class CheckpointPersistence:
    """Atomic checkpoint writer using temp-file/rename, fsync, and disk-space checks.

    Provides crash-safe persistence for training checkpoints.
    """

    def __init__(
        self,
        retention: CheckpointRetention | None = None,
    ):
        """Initialize persistence manager.

        Args:
            retention: Optional retention policy for pruning old checkpoints.
        """
        self._retention = retention or CheckpointRetention()

    def estimate_checkpoint_size(self, parameter_count: int) -> int:
        """Estimate checkpoint size in bytes (for disk space checks).

        Args:
            parameter_count: Number of model parameters.

        Returns:
            Estimated size in bytes (with 10% overhead for metadata).
        """
        # 4 bytes per float32 parameter
        base_bytes = parameter_count * 4
        # 10% overhead for metadata and safetensors format
        return int(base_bytes * 1.1)

    def ensure_sufficient_space(
        self,
        required: int,
        directory: Path,
        auto_prune: bool = True,
    ) -> None:
        """Ensure sufficient disk space is available.

        If insufficient space is detected and auto_prune is enabled,
        attempts to free space by deleting the oldest checkpoint.

        Args:
            required: Required bytes.
            directory: Directory to check.
            auto_prune: Whether to attempt pruning to free space.

        Raises:
            CheckpointError: If space cannot be freed.
        """
        try:
            usage = shutil.disk_usage(directory)
            free_space = usage.free
        except OSError:
            # Can't check disk space, proceed anyway
            return

        if free_space >= required:
            return

        required_mb = required / 1_000_000
        available_mb = free_space / 1_000_000

        logger.error(
            f"Insufficient disk space: required={required_mb:.1f}MB, available={available_mb:.1f}MB"
        )

        if auto_prune:
            # Try to free space by pruning
            import asyncio

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Try synchronous pruning
            checkpoints = []
            if directory.exists():
                for path in directory.iterdir():
                    if path.suffix == ".json" and path.name.startswith("checkpoint-"):
                        if path.name == "checkpoint-best.json":
                            continue
                        try:
                            with open(path, "r") as f:
                                data = json.load(f)
                            checkpoints.append(CheckpointMetadataV2.from_dict(data))
                        except Exception:
                            pass

            if checkpoints:
                # Delete oldest
                checkpoints.sort(key=lambda c: c.step)
                oldest = checkpoints[0]
                logger.warning(
                    f"Attempting to free space by deleting oldest checkpoint: step={oldest.step}"
                )
                self.delete_checkpoint(oldest, directory)

                # Recheck space
                try:
                    usage = shutil.disk_usage(directory)
                    if usage.free >= required:
                        return
                    available_mb = usage.free / 1_000_000
                except OSError:
                    pass

        raise CheckpointError(
            CheckpointErrorKind.INSUFFICIENT_DISK_SPACE,
            f"Insufficient disk space: required={required_mb:.1f}MB, "
            f"available={available_mb:.1f}MB",
        )

    def sync_to_disk(self, path: Path) -> None:
        """Sync file to disk (fsync) for durability.

        Args:
            path: File to sync.

        Raises:
            CheckpointError: If sync fails.
        """
        try:
            fd = os.open(str(path), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError as e:
            raise CheckpointError(
                CheckpointErrorKind.WRITE_FAILED,
                f"Failed to sync path: {path} - {e}",
            )

    def sync_directory(self, path: Path) -> None:
        """Sync directory to disk (fsync) to ensure directory operations are durable.

        Args:
            path: Directory to sync.

        Raises:
            CheckpointError: If sync fails.
        """
        self.sync_to_disk(path)

    def delete_checkpoint(self, metadata: CheckpointMetadataV2, directory: Path) -> None:
        """Delete a checkpoint (weights + metadata + optimizer state).

        Args:
            metadata: Checkpoint metadata.
            directory: Directory containing checkpoint files.
        """
        weights_file = directory / metadata.weights_file
        metadata_file = directory / f"checkpoint-{metadata.step}.json"

        # Delete optimizer state if present
        if metadata.optimizer_state is not None:
            optimizer_file = directory / metadata.optimizer_state.state_file
            optimizer_file.unlink(missing_ok=True)

        weights_file.unlink(missing_ok=True)
        metadata_file.unlink(missing_ok=True)

        logger.debug(f"Deleted checkpoint: step={metadata.step}")

    async def delete_checkpoint_async(
        self, metadata: CheckpointMetadataV2, directory: Path
    ) -> None:
        """Delete a checkpoint asynchronously.

        Args:
            metadata: Checkpoint metadata.
            directory: Directory containing checkpoint files.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.delete_checkpoint, metadata, directory)

    def atomic_write(
        self,
        content: bytes,
        destination: Path,
        temp_suffix: str = ".tmp",
    ) -> None:
        """Write content atomically using temp file and rename.

        Args:
            content: Content to write.
            destination: Final destination path.
            temp_suffix: Suffix for temporary file.

        Raises:
            CheckpointError: If write fails.
        """
        temp_path = destination.with_suffix(destination.suffix + temp_suffix)

        try:
            # Write to temp file
            with open(temp_path, "wb") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename
            temp_path.rename(destination)

            # Sync parent directory
            self.sync_directory(destination.parent)

        except Exception as e:
            # Clean up temp file on failure
            temp_path.unlink(missing_ok=True)
            raise CheckpointError(
                CheckpointErrorKind.WRITE_FAILED,
                f"Failed to write {destination}: {e}",
            )

    def atomic_write_json(
        self,
        data: dict[str, Any],
        destination: Path,
        indent: int = 2,
    ) -> None:
        """Write JSON content atomically.

        Args:
            data: JSON data to write.
            destination: Final destination path.
            indent: JSON indentation.

        Raises:
            CheckpointError: If write fails.
        """
        content = json.dumps(data, indent=indent, default=str).encode("utf-8")
        self.atomic_write(content, destination)
