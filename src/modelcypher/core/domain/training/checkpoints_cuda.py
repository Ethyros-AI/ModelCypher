"""
CUDA Checkpoint Manager Stub.

This module provides a PyTorch/CUDA implementation of checkpoint management.
Currently a stub - implement when CUDA support is needed.

See checkpoints.py for the MLX implementation that this mirrors.

Implementation Notes:
- Replace mx.save_safetensors with torch.save or safetensors.torch
- Replace mx.load with torch.load or safetensors.torch.load_file
- Handle device mapping for checkpoint loading
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional

from .types import CheckpointMetadata, TrainingConfig
from .exceptions import CheckpointError


# Minimum required disk space in bytes (500MB)
MIN_DISK_SPACE_BYTES = 500 * 1024 * 1024


class InsufficientDiskSpaceErrorCUDA(CheckpointError):
    """Raised when there's not enough disk space for checkpoint."""
    pass


class CheckpointManagerCUDA:
    """
    CUDA Checkpoint Manager (PyTorch backend).

    This is a stub implementation. When CUDA support is needed, implement:
    1. torch.save for model/optimizer state
    2. safetensors.torch for safetensors format (recommended)
    3. Device mapping for cross-device checkpoint loading

    See checkpoints.py for the full MLX implementation to mirror.
    """

    def __init__(self, max_checkpoints: int = 3) -> None:
        self.max_checkpoints = max_checkpoints
        self._best_loss: float = float("inf")
        self._best_step: int = -1

    async def save_checkpoint(
        self,
        model_weights: Dict[str, Any],  # torch tensors
        optimizer_state: Optional[Dict[str, Any]],
        step: int,
        total_steps: int,
        loss_history: List[float],
        config: TrainingConfig,
        output_dir: str,
    ) -> CheckpointMetadata:
        """
        Save a checkpoint atomically with full state preservation.

        Args:
            model_weights: Model state dict (torch tensors)
            optimizer_state: Optimizer state dict
            step: Current training step
            total_steps: Total training steps
            loss_history: Loss values recorded so far
            config: Training configuration
            output_dir: Output directory for checkpoints

        Returns:
            CheckpointMetadata for the saved checkpoint

        Raises:
            NotImplementedError: This is a stub.
        """
        raise NotImplementedError(
            "CUDA checkpoint saving not yet implemented. "
            "See checkpoints.py for the MLX implementation to port. "
            "Key differences:\n"
            "  - Replace mx.save_safetensors with torch.save\n"
            "  - Or use: from safetensors.torch import save_file\n"
            "  - Move tensors to CPU before saving: tensor.cpu()"
        )

    async def load_latest_checkpoint(
        self, output_dir: str
    ) -> Optional[CheckpointMetadata]:
        """Load metadata for the latest checkpoint."""
        raise NotImplementedError("CUDA checkpoint loading not yet implemented")

    async def load_checkpoint_metadata(
        self, checkpoints_dir: str, step: int
    ) -> CheckpointMetadata:
        """Load checkpoint metadata for a specific step."""
        raise NotImplementedError("CUDA checkpoint metadata loading not yet implemented")

    async def load_weights(
        self, checkpoints_dir: str, step: int, device: str = "cuda:0"
    ) -> Dict[str, Any]:
        """
        Load model weights from checkpoint.

        Args:
            checkpoints_dir: Directory containing checkpoints
            step: Step number to load
            device: Target device for loaded weights

        Returns:
            Model state dict with weights on specified device

        Raises:
            NotImplementedError: This is a stub.
        """
        raise NotImplementedError(
            "CUDA weight loading not yet implemented. "
            "Implementation hint:\n"
            "  import torch\n"
            "  weights = torch.load(path, map_location=device)\n"
            "  # Or for safetensors:\n"
            "  from safetensors.torch import load_file\n"
            "  weights = load_file(path, device=device)"
        )

    async def load_optimizer_state(
        self, checkpoints_dir: str, step: int
    ) -> Optional[Dict[str, Any]]:
        """Load optimizer state from checkpoint if it exists."""
        raise NotImplementedError("CUDA optimizer state loading not yet implemented")


__all__ = [
    "CheckpointManagerCUDA",
    "InsufficientDiskSpaceErrorCUDA",
]
