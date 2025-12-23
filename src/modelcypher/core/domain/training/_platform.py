"""
Training Platform Selector.

This module provides lazy importing of platform-specific training implementations.
On macOS, MLX implementations are used. On Linux with CUDA, PyTorch/CUDA
implementations will be used (when available).

Usage in code that needs platform-specific training:

    from modelcypher.core.domain.training._platform import (
        get_training_engine,
        get_checkpoint_manager,
        get_lora_module,
    )

    engine = get_training_engine()
    manager = get_checkpoint_manager()
"""

from __future__ import annotations

import platform
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .checkpoints import CheckpointManager
    from .engine import TrainingEngine
    from .evaluation import EvaluationEngine


def _is_mlx_available() -> bool:
    """Check if MLX is available (macOS with Apple Silicon)."""
    if platform.system() != "Darwin":
        return False
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


def _is_cuda_available() -> bool:
    """Check if CUDA is available (Linux with NVIDIA GPU)."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_training_platform() -> str:
    """Get the current training platform identifier.

    Returns:
        'mlx' on macOS with Apple Silicon
        'cuda' on Linux with NVIDIA GPU
        'cpu' otherwise
    """
    if _is_mlx_available():
        return "mlx"
    if _is_cuda_available():
        return "cuda"
    return "cpu"


def get_training_engine() -> "TrainingEngine":
    """Get the training engine for the current platform.

    Returns:
        TrainingEngine instance appropriate for the platform.

    Raises:
        NotImplementedError: If no supported training platform is available.
    """
    platform_name = get_training_platform()

    if platform_name == "mlx":
        from .engine import TrainingEngine
        return TrainingEngine()
    elif platform_name == "cuda":
        from .engine_cuda import TrainingEngineCUDA
        return TrainingEngineCUDA()
    else:
        raise NotImplementedError(
            f"No training engine available for platform: {platform_name}. "
            "Install MLX on macOS or PyTorch with CUDA on Linux."
        )


def get_checkpoint_manager(max_checkpoints: int = 3) -> "CheckpointManager":
    """Get the checkpoint manager for the current platform.

    Args:
        max_checkpoints: Maximum number of checkpoints to retain.

    Returns:
        CheckpointManager instance appropriate for the platform.
    """
    platform_name = get_training_platform()

    if platform_name == "mlx":
        from .checkpoints import CheckpointManager
        return CheckpointManager(max_checkpoints=max_checkpoints)
    elif platform_name == "cuda":
        from .checkpoints_cuda import CheckpointManagerCUDA
        return CheckpointManagerCUDA(max_checkpoints=max_checkpoints)
    else:
        raise NotImplementedError(
            f"No checkpoint manager available for platform: {platform_name}."
        )


def get_evaluation_engine() -> "EvaluationEngine":
    """Get the evaluation engine for the current platform.

    Returns:
        EvaluationEngine instance appropriate for the platform.
    """
    platform_name = get_training_platform()

    if platform_name == "mlx":
        from .evaluation import EvaluationEngine
        return EvaluationEngine()
    elif platform_name == "cuda":
        from .evaluation_cuda import EvaluationEngineCUDA
        return EvaluationEngineCUDA()
    else:
        raise NotImplementedError(
            f"No evaluation engine available for platform: {platform_name}."
        )


__all__ = [
    "get_training_platform",
    "get_training_engine",
    "get_checkpoint_manager",
    "get_evaluation_engine",
]
