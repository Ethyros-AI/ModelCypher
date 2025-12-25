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

"""
CUDA Checkpoint Manager (PyTorch Backend).

This is the PyTorch/CUDA implementation. For other backends:
- MLX/macOS: see checkpoints_mlx.py
- JAX/TPU: see checkpoints_jax.py

Use _platform.get_checkpoint_manager() for automatic platform selection.

Implementation based on safetensors 0.5.x API (2025):
- safetensors.torch.save_file for tensor serialization
- safetensors.torch.load_file for tensor loading with device placement
- Atomic writes with temp files for crash safety
- SHA-256 checksums for integrity verification

References:
- https://huggingface.co/docs/safetensors/en/api/torch
- https://huggingface.co/docs/safetensors/torch_shared_tensors
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .exceptions import CheckpointError
from .types import CheckpointMetadata, TrainingConfig

logger = logging.getLogger(__name__)

# Minimum required disk space in bytes (500MB)
MIN_DISK_SPACE_BYTES = 500 * 1024 * 1024


class InsufficientDiskSpaceErrorCUDA(CheckpointError):
    """Raised when there's not enough disk space for checkpoint."""

    pass


class CheckpointManagerCUDA:
    """
    CUDA Checkpoint Manager (PyTorch backend).

    Features (matching MLX parity):
    - Atomic checkpoint writes with temp files
    - SHA-256 checksum validation
    - Optimizer state preservation
    - Retention-based pruning (keeps N most recent)
    - Best checkpoint tracking

    Uses safetensors format for:
    - Fast, memory-efficient loading
    - Cross-platform compatibility
    - Zero-copy when possible
    """

    def __init__(self, max_checkpoints: int = 3) -> None:
        self.max_checkpoints = max_checkpoints
        self._best_loss: float = float("inf")
        self._best_step: int = -1

    async def save_checkpoint(
        self,
        model_weights: dict[str, torch.Tensor],
        optimizer_state: dict[str, Any] | None,
        step: int,
        total_steps: int,
        loss_history: list[float],
        config: TrainingConfig,
        output_dir: str,
    ) -> CheckpointMetadata:
        """
        Save a checkpoint atomically with full state preservation.

        Args:
            model_weights: Model state dict (tensors should be on CPU)
            optimizer_state: Optimizer state dict
            step: Current training step
            total_steps: Total training steps
            loss_history: Loss values recorded so far
            config: Training configuration
            output_dir: Output directory for checkpoints

        Returns:
            CheckpointMetadata for the saved checkpoint
        """
        checkpoints_dir = Path(output_dir) / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Check disk space
        try:
            stat = shutil.disk_usage(checkpoints_dir)
            if stat.free < MIN_DISK_SPACE_BYTES:
                raise InsufficientDiskSpaceErrorCUDA(
                    f"Insufficient disk space: {stat.free / 1e6:.1f}MB available, "
                    f"need {MIN_DISK_SPACE_BYTES / 1e6:.1f}MB"
                )
        except OSError as e:
            logger.warning("Could not check disk space: %s", e)

        step_dir = checkpoints_dir / f"step_{step:06d}"

        # Atomic write: save to temp dir first, then rename
        temp_dir = checkpoints_dir / f".tmp_step_{step:06d}"
        try:
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Save model weights using safetensors
            weights_path = temp_dir / "model.safetensors"

            # Ensure all tensors are contiguous and on CPU
            contiguous_weights = {}
            for k, v in model_weights.items():
                if isinstance(v, torch.Tensor):
                    v = v.contiguous().cpu() if not v.is_contiguous() else v.cpu()
                    contiguous_weights[k] = v

            save_file(contiguous_weights, str(weights_path))

            # Compute checksum
            weights_checksum = self._compute_checksum(weights_path)

            # Save optimizer state (use torch.save for complex nested state)
            optimizer_checksum = None
            if optimizer_state is not None:
                optimizer_path = temp_dir / "optimizer.pt"
                torch.save(optimizer_state, str(optimizer_path))
                optimizer_checksum = self._compute_checksum(optimizer_path)

            # Create metadata
            current_loss = loss_history[-1] if loss_history else 0.0
            is_best = current_loss < self._best_loss
            if is_best:
                self._best_loss = current_loss
                self._best_step = step

            metadata = CheckpointMetadata(
                step=step,
                total_steps=total_steps,
                timestamp=datetime.now().isoformat(),
                loss_history=loss_history.copy(),
                hyperparameters=config.hyperparameters,
                weights_checksum=weights_checksum,
                optimizer_checksum=optimizer_checksum,
                is_best=is_best,
            )

            # Save metadata
            metadata_path = temp_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata.to_dict(), f, indent=2)

            # Atomic rename
            if step_dir.exists():
                shutil.rmtree(step_dir)
            temp_dir.rename(step_dir)

            logger.info(
                "Saved checkpoint at step %d (loss=%.4f, best=%s)", step, current_loss, is_best
            )

            # Update best symlink
            if is_best:
                best_link = checkpoints_dir / "best"
                if best_link.is_symlink():
                    best_link.unlink()
                best_link.symlink_to(step_dir.name)

            # Prune old checkpoints
            await self._prune_old_checkpoints(checkpoints_dir)

            return metadata

        except Exception as e:
            # Cleanup on failure
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise CheckpointError(f"Failed to save checkpoint: {e}") from e

    async def load_latest_checkpoint(self, output_dir: str) -> CheckpointMetadata | None:
        """Load metadata for the latest checkpoint."""
        checkpoints_dir = Path(output_dir) / "checkpoints"
        if not checkpoints_dir.exists():
            return None

        # Find latest step directory
        step_dirs = sorted(
            [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
            key=lambda d: int(d.name.split("_")[1]),
            reverse=True,
        )

        if not step_dirs:
            return None

        latest_dir = step_dirs[0]
        metadata_path = latest_dir / "metadata.json"

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path) as f:
                data = json.load(f)
            return CheckpointMetadata.from_dict(data)
        except Exception as e:
            logger.warning("Failed to load checkpoint metadata: %s", e)
            return None

    async def load_checkpoint_metadata(self, checkpoints_dir: str, step: int) -> CheckpointMetadata:
        """Load checkpoint metadata for a specific step."""
        step_dir = Path(checkpoints_dir) / f"step_{step:06d}"
        metadata_path = step_dir / "metadata.json"

        if not metadata_path.exists():
            raise CheckpointError(f"No checkpoint found at step {step}")

        with open(metadata_path) as f:
            data = json.load(f)
        return CheckpointMetadata.from_dict(data)

    async def load_weights(
        self, checkpoints_dir: str, step: int, device: str = "cuda:0"
    ) -> dict[str, torch.Tensor]:
        """
        Load model weights from checkpoint.

        Args:
            checkpoints_dir: Directory containing checkpoints
            step: Step number to load
            device: Target device for loaded weights

        Returns:
            Model state dict with weights on specified device
        """
        step_dir = Path(checkpoints_dir) / f"step_{step:06d}"
        weights_path = step_dir / "model.safetensors"

        if not weights_path.exists():
            raise CheckpointError(f"No weights found at step {step}")

        # Verify checksum
        metadata = await self.load_checkpoint_metadata(checkpoints_dir, step)
        if metadata.weights_checksum:
            actual_checksum = self._compute_checksum(weights_path)
            if actual_checksum != metadata.weights_checksum:
                raise CheckpointError(
                    f"Checksum mismatch for weights at step {step}. Checkpoint may be corrupted."
                )

        # Load with device placement
        weights = load_file(str(weights_path), device=device)
        logger.info("Loaded weights from step %d to %s", step, device)
        return weights

    async def load_optimizer_state(self, checkpoints_dir: str, step: int) -> dict[str, Any] | None:
        """Load optimizer state from checkpoint if it exists."""
        step_dir = Path(checkpoints_dir) / f"step_{step:06d}"
        optimizer_path = step_dir / "optimizer.pt"

        if not optimizer_path.exists():
            return None

        # Verify checksum
        metadata = await self.load_checkpoint_metadata(checkpoints_dir, step)
        if metadata.optimizer_checksum:
            actual_checksum = self._compute_checksum(optimizer_path)
            if actual_checksum != metadata.optimizer_checksum:
                raise CheckpointError(
                    f"Checksum mismatch for optimizer at step {step}. Checkpoint may be corrupted."
                )

        state = torch.load(str(optimizer_path), map_location="cpu", weights_only=False)
        logger.info("Loaded optimizer state from step %d", step)
        return state

    def _compute_checksum(self, path: Path) -> str:
        """Compute SHA-256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    async def _prune_old_checkpoints(self, checkpoints_dir: Path) -> None:
        """Remove old checkpoints, keeping max_checkpoints most recent."""
        step_dirs = sorted(
            [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
            key=lambda d: int(d.name.split("_")[1]),
            reverse=True,
        )

        # Keep best checkpoint
        best_link = checkpoints_dir / "best"
        best_target = None
        if best_link.is_symlink():
            best_target = best_link.resolve().name

        # Remove old checkpoints (but keep best)
        for step_dir in step_dirs[self.max_checkpoints :]:
            if step_dir.name != best_target:
                logger.info("Pruning old checkpoint: %s", step_dir.name)
                shutil.rmtree(step_dir)


__all__ = [
    "CheckpointManagerCUDA",
    "InsufficientDiskSpaceErrorCUDA",
    "MIN_DISK_SPACE_BYTES",
]
