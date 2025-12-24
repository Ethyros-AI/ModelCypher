"""
Checkpoint Manager for Training Persistence (JAX Backend).

This module provides a JAX implementation of the checkpoint manager.
Currently a stub - implement when JAX support is needed.

For other backends:
- MLX/macOS: see checkpoints_mlx.py
- CUDA/PyTorch: see checkpoints_cuda.py

Use _platform.get_checkpoint_manager() for automatic platform selection.

Implementation Notes:
- Replace mx.save_safetensors with flax.serialization or orbax
- Replace mx.load with corresponding JAX checkpoint loading
- Handle JAX pytree structures for nested weights
- Consider using orbax.checkpoint for distributed checkpointing
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from datetime import datetime

from .types import CheckpointMetadata, TrainingConfig
from .exceptions import CheckpointError


class InsufficientDiskSpaceErrorJAX(CheckpointError):
    """Raised when there's not enough disk space for checkpoint."""
    pass


class CheckpointManagerJAX:
    """
    Manages atomic writing and loading of training checkpoints (JAX version).

    This is a stub implementation. When JAX support is needed, implement:
    1. Orbax checkpoint integration
    2. Flax serialization support
    3. TPU-optimized checkpoint saving
    4. Pytree handling for nested structures

    See checkpoints_mlx.py for the full MLX implementation to mirror.

    Research Basis:
    - Orbax: https://orbax.readthedocs.io/
    - Flax Checkpointing: https://flax.readthedocs.io/en/latest/guides/training_techniques/use_checkpointing.html
    """

    def __init__(self, max_checkpoints: int = 3):
        self.max_checkpoints = max_checkpoints
        self._best_loss: float = float('inf')
        self._best_step: int = -1

    async def save_checkpoint(
        self,
        model_weights: Dict[str, Any],
        optimizer_state: Optional[Dict[str, Any]],
        step: int,
        total_steps: int,
        loss_history: List[float],
        config: TrainingConfig,
        output_dir: str
    ) -> CheckpointMetadata:
        """
        Saves a checkpoint atomically with full state preservation.

        Raises:
            NotImplementedError: This is a stub.
        """
        raise NotImplementedError(
            "JAX checkpoint manager not yet implemented. "
            "See checkpoints_mlx.py for the MLX implementation to port. "
            "Key differences:\n"
            "  - Use orbax.checkpoint.CheckpointManager\n"
            "  - Use flax.serialization for pytrees\n"
            "  - Handle JAX arrays instead of mx.array\n"
            "  - Consider async checkpointing for large models"
        )

    async def load_latest_checkpoint(self, output_dir: str) -> Optional[CheckpointMetadata]:
        """Load metadata for the latest checkpoint."""
        raise NotImplementedError("JAX checkpoint loading not yet implemented")

    async def load_checkpoint_metadata(self, checkpoints_dir: str, step: int) -> CheckpointMetadata:
        """Load checkpoint metadata for a specific step."""
        raise NotImplementedError("JAX checkpoint metadata loading not yet implemented")

    async def load_weights(self, checkpoints_dir: str, step: int) -> Dict[str, Any]:
        """Load model weights from checkpoint."""
        raise NotImplementedError("JAX weight loading not yet implemented")

    async def load_optimizer_state(self, checkpoints_dir: str, step: int) -> Optional[Dict[str, Any]]:
        """Load optimizer state from checkpoint if it exists."""
        raise NotImplementedError("JAX optimizer state loading not yet implemented")


__all__ = [
    "CheckpointManagerJAX",
    "InsufficientDiskSpaceErrorJAX",
]
