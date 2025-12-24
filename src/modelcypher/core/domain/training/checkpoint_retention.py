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

"""Checkpoint retention policy enforcement.

Keeps N most recent checkpoints to prevent unbounded disk growth.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

from modelcypher.core.domain.training.checkpoint_models import CheckpointMetadataV2

logger = logging.getLogger(__name__)


class CheckpointRetention:
    """Enforces checkpoint retention (keeps N most recent, optional confirmation).

    Prevents unbounded disk growth by pruning old checkpoints.
    """

    def __init__(
        self,
        max_checkpoints: int = 3,
        confirm_prune: bool = False,
        on_prune_requested: Callable[[str, int, int], None] | None = None,
    ):
        """Create a checkpoint retention manager.

        Args:
            max_checkpoints: Maximum number of checkpoints to retain (minimum: 1).
            confirm_prune: If True, notify instead of auto-deleting.
            on_prune_requested: Callback when prune confirmation is requested.
                Args: (checkpoints_dir, keep_count, pending_delete_count)
        """
        self._max_checkpoints = max(1, max_checkpoints)
        self._confirm_prune = confirm_prune
        self._on_prune_requested = on_prune_requested

    @property
    def max_checkpoints(self) -> int:
        """Maximum number of checkpoints to retain."""
        return self._max_checkpoints

    async def prune_old_checkpoints(
        self,
        directory: Path,
        delete_fn: Callable[[CheckpointMetadataV2, Path], None] | None = None,
    ) -> int:
        """Prune old checkpoints (keeps N most recent).

        Retention behavior:
        - Sorts checkpoints by step (newest first)
        - Keeps `max_checkpoints` most recent
        - Deletes older checkpoints
        - If confirmation is enabled, calls callback instead of deleting

        Args:
            directory: Directory containing checkpoints.
            delete_fn: Function to delete a checkpoint. If None, uses default deletion.

        Returns:
            Number of checkpoints deleted.
        """
        checkpoints = await self._list_checkpoints(directory)
        checkpoints.sort(key=lambda c: c.step, reverse=True)

        if len(checkpoints) <= self._max_checkpoints:
            return 0

        to_delete = checkpoints[self._max_checkpoints:]

        if self._confirm_prune and self._on_prune_requested:
            self._on_prune_requested(
                str(directory),
                self._max_checkpoints,
                len(to_delete),
            )
            logger.info(
                f"Prune confirmation requested: keep={self._max_checkpoints}, "
                f"pending={len(to_delete)}"
            )
            return 0

        # Auto-prune without confirmation
        deleted = 0
        for checkpoint in to_delete:
            try:
                if delete_fn:
                    delete_fn(checkpoint, directory)
                else:
                    self._default_delete(checkpoint, directory)
                logger.debug(f"Pruned old checkpoint: step={checkpoint.step}")
                deleted += 1
            except Exception as e:
                logger.warning(f"Failed to delete checkpoint step={checkpoint.step}: {e}")

        return deleted

    async def list_checkpoints(self, directory: Path) -> list[CheckpointMetadataV2]:
        """List all checkpoints in a directory.

        Args:
            directory: Directory to scan.

        Returns:
            List of checkpoint metadata.
        """
        return await self._list_checkpoints(directory)

    async def _list_checkpoints(self, directory: Path) -> list[CheckpointMetadataV2]:
        """List all checkpoints in a directory.

        Args:
            directory: Directory to scan.

        Returns:
            List of checkpoint metadata.
        """
        checkpoints: list[CheckpointMetadataV2] = []
        failures: list[str] = []

        if not directory.exists():
            return checkpoints

        for path in directory.iterdir():
            if path.suffix == ".json" and path.name.startswith("checkpoint-"):
                if path.name == "checkpoint-best.json":
                    continue

                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                    checkpoints.append(CheckpointMetadataV2.from_dict(data))
                except Exception:
                    failures.append(path.name)

        if failures:
            for failure in failures:
                logger.warning(f"Failed to read checkpoint metadata: {failure}")

        return checkpoints

    def _default_delete(
        self, metadata: CheckpointMetadataV2, directory: Path
    ) -> None:
        """Default checkpoint deletion.

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
