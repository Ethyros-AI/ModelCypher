"""Crash detection and checkpoint recovery.

Uses active markers and validation to restore latest valid checkpoint after failures.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from modelcypher.core.domain.training.checkpoint_models import (
    CheckpointError,
    CheckpointErrorKind,
    CheckpointMetadataV2,
    RecoveryInfo,
)
from modelcypher.core.domain.training.checkpoint_validation import CheckpointValidation

logger = logging.getLogger(__name__)


class CheckpointRecovery:
    """Crash detection and checkpoint recovery using active markers and validation."""

    def __init__(self, temp_dir: Optional[Path] = None):
        """Initialize recovery manager.

        Args:
            temp_dir: Directory for crash markers. Defaults to system temp.
        """
        self._temp_dir = temp_dir or Path(tempfile.gettempdir())
        self._seen_markers: set[str] = set()

    def _get_crash_marker_path(self, output_dir: Path) -> Path:
        """Get crash marker path for output directory."""
        return self._temp_dir / f".training-active-{output_dir.name}"

    def _get_progress_marker_path(self, output_dir: Path) -> Path:
        """Get progress marker path for output directory."""
        return self._temp_dir / f".training-progress-{output_dir.name}"

    async def recover_from_crash_if_needed(
        self,
        output_dir: Path,
    ) -> Optional[RecoveryInfo]:
        """Check for crash recovery and return the latest valid checkpoint.

        This method is called on app launch to detect if training was interrupted
        unexpectedly. It validates all checkpoints and returns the most recent
        one that passes integrity checks.

        Args:
            output_dir: The training output directory to check.

        Returns:
            Recovery information if crash detected, None otherwise.

        Raises:
            CheckpointError: If crash detected but no valid checkpoints found.
        """
        crash_marker = self._get_crash_marker_path(output_dir)

        # Check if crash marker exists
        if not crash_marker.exists():
            marker_name = crash_marker.name
            if marker_name not in self._seen_markers:
                self._seen_markers.add(marker_name)
                logger.debug(f"No crash marker found - normal launch. marker: {marker_name}")
            return None

        logger.warning("Crash detected! Attempting recovery...")

        # Find all checkpoints and validate them (newest → oldest)
        checkpoints_dir = output_dir / "checkpoints"
        if not checkpoints_dir.is_dir():
            self._remove_crash_marker(crash_marker)
            raise CheckpointError(
                CheckpointErrorKind.NO_VALID_CHECKPOINTS,
                "No checkpoints directory found",
            )

        checkpoints = await self._list_checkpoints(checkpoints_dir)
        checkpoints.sort(key=lambda c: c.step, reverse=True)

        if not checkpoints:
            self._remove_crash_marker(crash_marker)
            raise CheckpointError(
                CheckpointErrorKind.NO_VALID_CHECKPOINTS,
                "No checkpoints found in directory",
            )

        # Validate checkpoints from newest to oldest
        for checkpoint in checkpoints:
            try:
                is_valid = await CheckpointValidation.validate_checkpoint_async(
                    checkpoint, checkpoints_dir
                )
                if is_valid:
                    logger.info(f"Valid checkpoint found: step={checkpoint.step}")

                    # Clean up crash marker after successful recovery
                    self._remove_crash_marker(crash_marker)

                    return RecoveryInfo(
                        checkpoint=checkpoint,
                        checkpoints_dir=checkpoints_dir,
                        output_dir=output_dir,
                    )
                else:
                    logger.warning(
                        f"Checkpoint validation failed: step={checkpoint.step}, trying next..."
                    )
            except Exception as e:
                logger.warning(
                    f"Checkpoint read failed: step={checkpoint.step}, error={e}"
                )
                continue

        # No valid checkpoints found - total loss
        self._remove_crash_marker(crash_marker)
        raise CheckpointError(
            CheckpointErrorKind.NO_VALID_CHECKPOINTS,
            "All checkpoints corrupted or unreadable",
        )

    async def mark_training_active(self, output_dir: Path) -> None:
        """Mark training as active (creates crash marker).

        Call this at the start of training. If the app crashes, the marker
        will remain and trigger recovery on next launch.

        Args:
            output_dir: The training output directory.
        """
        marker = self._get_crash_marker_path(output_dir)
        marker.parent.mkdir(parents=True, exist_ok=True)

        content = {
            "started": datetime.now().isoformat(),
            "output_dir": str(output_dir),
        }

        with open(marker, "w") as f:
            json.dump(content, f, indent=2)

        logger.debug(f"Training marked as active: {marker.name}")

    async def mark_training_inactive(self, output_dir: Path) -> None:
        """Mark training as inactive (removes crash marker).

        Call this on normal training completion or cancellation.

        Args:
            output_dir: The training output directory.
        """
        marker = self._get_crash_marker_path(output_dir)
        self._remove_crash_marker(marker)
        logger.debug("Training marked as inactive")

    async def update_progress_marker(
        self, step: int, total_steps: int, output_dir: Path
    ) -> None:
        """Update progress marker during training (for crash recovery).

        This lightweight marker tracks current progress without blocking
        the training loop. It's used to show users how much work was lost
        in a crash.

        Args:
            step: Current training step.
            total_steps: Total steps in training job.
            output_dir: The training output directory.
        """
        marker = self._get_progress_marker_path(output_dir)

        content = {
            "step": step,
            "total": total_steps,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            with open(marker, "w") as f:
                json.dump(content, f)
        except Exception as e:
            # Log failure but don't crash training loop
            logger.warning(f"Failed to update progress marker: {e}")

    def _remove_crash_marker(self, marker: Path) -> None:
        """Remove a crash marker file with proper error handling.

        Args:
            marker: Path of the crash marker file to remove.
        """
        try:
            if marker.exists():
                marker.unlink()
                logger.debug(f"Removed crash marker: {marker.name}")
        except Exception as e:
            logger.error(
                f"Failed to remove crash marker at {marker}: {e}. "
                "Manual cleanup may be required."
            )

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
