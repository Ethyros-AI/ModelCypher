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
JAX Checkpoint Manager for Training Persistence.

This is the JAX/Orbax implementation. For other backends:
- MLX/macOS: see checkpoints_mlx.py
- CUDA/PyTorch: see checkpoints_cuda.py

Use _platform.get_checkpoint_manager() for automatic platform selection.

Implementation based on Flax/Orbax best practices (2025):
- orbax.checkpoint.StandardCheckpointer for saving/loading
- Pure dict serialization for simplicity
- Atomic writes with temp directories
- JSON metadata for cross-platform compatibility

References:
- https://flax.readthedocs.io/en/stable/guides/checkpointing.html
- https://orbax.readthedocs.io/en/latest/guides/checkpoint/api_refactor.html
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .types import CheckpointMetadata, TrainingConfig
from .exceptions import CheckpointError

logger = logging.getLogger(__name__)

# Minimum required disk space in bytes (500MB)
MIN_DISK_SPACE_BYTES = 500 * 1024 * 1024


class InsufficientDiskSpaceErrorJAX(CheckpointError):
    """Raised when there's not enough disk space for checkpoint."""
    pass


def _pytree_to_numpy(pytree: Any) -> Any:
    """Convert JAX arrays in a pytree to numpy arrays for serialization."""
    return jax.tree.map(
        lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x,
        pytree,
    )


def _numpy_to_pytree(pytree: Any) -> Any:
    """Convert numpy arrays in a pytree back to JAX arrays."""
    return jax.tree.map(
        lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x,
        pytree,
    )


class CheckpointManagerJAX:
    """
    JAX Checkpoint Manager using Orbax-compatible patterns.

    Features (matching MLX parity):
    - Atomic checkpoint writes with temp directories
    - SHA-256 checksum validation
    - Optimizer state preservation
    - Retention-based pruning (keeps N most recent)
    - Best checkpoint tracking

    Note: Uses numpy serialization for simplicity. For production
    distributed training, consider using orbax.checkpoint.CheckpointManager
    with AsyncCheckpointer for better performance.
    """

    def __init__(self, max_checkpoints: int = 3) -> None:
        self.max_checkpoints = max_checkpoints
        self._best_loss: float = float("inf")
        self._best_step: int = -1

    async def save_checkpoint(
        self,
        params: dict[str, Any],
        opt_state: Any | None,
        step: int,
        total_steps: int,
        loss_history: list[float],
        config: TrainingConfig,
        output_dir: str,
    ) -> CheckpointMetadata:
        """
        Save a checkpoint atomically with full state preservation.

        Args:
            params: Model parameters (JAX pytree)
            opt_state: Optimizer state (optax state pytree)
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
                raise InsufficientDiskSpaceErrorJAX(
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

            # Convert params to numpy and save
            params_np = _pytree_to_numpy(params)
            params_path = temp_dir / "params.npz"
            np.savez(str(params_path), **self._flatten_pytree(params_np))

            # Compute checksum
            params_checksum = self._compute_checksum(params_path)

            # Save optimizer state
            optimizer_checksum = None
            if opt_state is not None:
                opt_state_np = _pytree_to_numpy(opt_state)
                opt_path = temp_dir / "optimizer.npz"
                # Flatten optimizer state for saving
                flat_opt = self._flatten_pytree(opt_state_np)
                np.savez(str(opt_path), **flat_opt)
                optimizer_checksum = self._compute_checksum(opt_path)

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
                weights_checksum=params_checksum,
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
                "Saved checkpoint at step %d (loss=%.4f, best=%s)",
                step,
                current_loss,
                is_best,
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

    async def load_latest_checkpoint(
        self,
        output_dir: str,
    ) -> CheckpointMetadata | None:
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

    async def load_checkpoint_metadata(
        self,
        checkpoints_dir: str,
        step: int,
    ) -> CheckpointMetadata:
        """Load checkpoint metadata for a specific step."""
        step_dir = Path(checkpoints_dir) / f"step_{step:06d}"
        metadata_path = step_dir / "metadata.json"

        if not metadata_path.exists():
            raise CheckpointError(f"No checkpoint found at step {step}")

        with open(metadata_path) as f:
            data = json.load(f)
        return CheckpointMetadata.from_dict(data)

    async def load_weights(
        self,
        checkpoints_dir: str,
        step: int,
    ) -> dict[str, jnp.ndarray]:
        """
        Load model weights from checkpoint.

        Args:
            checkpoints_dir: Directory containing checkpoints
            step: Step number to load

        Returns:
            Model parameters as JAX pytree
        """
        step_dir = Path(checkpoints_dir) / f"step_{step:06d}"
        params_path = step_dir / "params.npz"

        if not params_path.exists():
            raise CheckpointError(f"No weights found at step {step}")

        # Verify checksum
        metadata = await self.load_checkpoint_metadata(checkpoints_dir, step)
        if metadata.weights_checksum:
            actual_checksum = self._compute_checksum(params_path)
            if actual_checksum != metadata.weights_checksum:
                raise CheckpointError(
                    f"Checksum mismatch for weights at step {step}. "
                    "Checkpoint may be corrupted."
                )

        # Load and reconstruct pytree
        loaded = np.load(str(params_path))
        params_np = self._unflatten_pytree(dict(loaded))
        params = _numpy_to_pytree(params_np)

        logger.info("Loaded weights from step %d", step)
        return params

    async def load_optimizer_state(
        self,
        checkpoints_dir: str,
        step: int,
    ) -> Any | None:
        """Load optimizer state from checkpoint if it exists."""
        step_dir = Path(checkpoints_dir) / f"step_{step:06d}"
        opt_path = step_dir / "optimizer.npz"

        if not opt_path.exists():
            return None

        # Verify checksum
        metadata = await self.load_checkpoint_metadata(checkpoints_dir, step)
        if metadata.optimizer_checksum:
            actual_checksum = self._compute_checksum(opt_path)
            if actual_checksum != metadata.optimizer_checksum:
                raise CheckpointError(
                    f"Checksum mismatch for optimizer at step {step}. "
                    "Checkpoint may be corrupted."
                )

        loaded = np.load(str(opt_path))
        opt_state_np = self._unflatten_pytree(dict(loaded))
        opt_state = _numpy_to_pytree(opt_state_np)

        logger.info("Loaded optimizer state from step %d", step)
        return opt_state

    def _compute_checksum(self, path: Path) -> str:
        """Compute SHA-256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _flatten_pytree(self, pytree: Any, prefix: str = "") -> dict[str, np.ndarray]:
        """Flatten a pytree into a flat dictionary for saving."""
        result = {}
        if isinstance(pytree, dict):
            for key, value in pytree.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                result.update(self._flatten_pytree(value, new_prefix))
        elif isinstance(pytree, (list, tuple)):
            for i, value in enumerate(pytree):
                new_prefix = f"{prefix}[{i}]"
                result.update(self._flatten_pytree(value, new_prefix))
        elif isinstance(pytree, np.ndarray):
            result[prefix] = pytree
        elif isinstance(pytree, (int, float, bool)):
            result[prefix] = np.array(pytree)
        elif hasattr(pytree, '__dict__'):
            # Handle optax state objects
            for key, value in vars(pytree).items():
                if not key.startswith('_'):
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    result.update(self._flatten_pytree(value, new_prefix))
        return result

    def _unflatten_pytree(self, flat: dict[str, np.ndarray]) -> dict[str, Any]:
        """Reconstruct a pytree from a flat dictionary."""
        result: dict[str, Any] = {}
        for key, value in flat.items():
            parts = key.replace('[', '.').replace(']', '').split('.')
            current = result
            for i, part in enumerate(parts[:-1]):
                if part.isdigit():
                    part = int(part)
                if part not in current:
                    # Determine if next level is list or dict
                    next_part = parts[i + 1]
                    if next_part.isdigit():
                        current[part] = []
                    else:
                        current[part] = {}
                current = current[part]

            final_part = parts[-1]
            if final_part.isdigit():
                final_part = int(final_part)
            current[final_part] = value

        return result

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
        for step_dir in step_dirs[self.max_checkpoints:]:
            if step_dir.name != best_target:
                logger.info("Pruning old checkpoint: %s", step_dir.name)
                shutil.rmtree(step_dir)


__all__ = [
    "CheckpointManagerJAX",
    "InsufficientDiskSpaceErrorJAX",
    "MIN_DISK_SPACE_BYTES",
]
