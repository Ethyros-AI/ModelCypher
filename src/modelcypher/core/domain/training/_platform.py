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
Training Platform Selector.

This module provides lazy importing of platform-specific training implementations.
On macOS, MLX implementations are used. On Linux with CUDA, PyTorch/CUDA
implementations will be used. On Linux with TPU/GPU, JAX implementations
will be used (when available).

Usage in code that needs platform-specific training:

    from modelcypher.core.domain.training._platform import (
        get_training_engine,
        get_checkpoint_manager,
        get_lora_module,
    )

    engine = get_training_engine()
    manager = get_checkpoint_manager()

Platform-specific implementations:
- MLX (macOS/Apple Silicon): *_mlx.py files
- CUDA (Linux/NVIDIA GPU): *_cuda.py files
- JAX (Linux/TPU/GPU): *_jax.py files
"""

from __future__ import annotations

import platform
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .checkpoints_mlx import CheckpointManager
    from .engine_mlx import TrainingEngine
    from .evaluation_mlx import EvaluationEngine


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


def _is_jax_available() -> bool:
    """Check if JAX is available (Linux/TPU/GPU)."""
    try:
        import jax  # noqa: F401

        return True
    except ImportError:
        return False


def get_training_platform() -> str:
    """Get the current training platform identifier.

    Returns:
        'mlx' on macOS with Apple Silicon
        'cuda' on Linux with NVIDIA GPU
        'jax' on Linux with JAX (TPU/GPU)
        'cpu' otherwise
    """
    if _is_mlx_available():
        return "mlx"
    if _is_cuda_available():
        return "cuda"
    if _is_jax_available():
        return "jax"
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
        from .engine_mlx import TrainingEngine

        return TrainingEngine()
    elif platform_name == "cuda":
        from .engine_cuda import TrainingEngineCUDA

        return TrainingEngineCUDA()
    elif platform_name == "jax":
        from .engine_jax import TrainingEngineJAX

        return TrainingEngineJAX()
    else:
        raise NotImplementedError(
            f"No training engine available for platform: {platform_name}. "
            "Install MLX on macOS, PyTorch with CUDA on Linux, or JAX for TPU/GPU."
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
        from .checkpoints_mlx import CheckpointManager

        return CheckpointManager(max_checkpoints=max_checkpoints)
    elif platform_name == "cuda":
        from .checkpoints_cuda import CheckpointManagerCUDA

        return CheckpointManagerCUDA(max_checkpoints=max_checkpoints)
    elif platform_name == "jax":
        from .checkpoints_jax import CheckpointManagerJAX

        return CheckpointManagerJAX(max_checkpoints=max_checkpoints)
    else:
        raise NotImplementedError(f"No checkpoint manager available for platform: {platform_name}.")


def get_evaluation_engine() -> "EvaluationEngine":
    """Get the evaluation engine for the current platform.

    Returns:
        EvaluationEngine instance appropriate for the platform.
    """
    platform_name = get_training_platform()

    if platform_name == "mlx":
        from .evaluation_mlx import EvaluationEngine

        return EvaluationEngine()
    elif platform_name == "cuda":
        from .evaluation_cuda import EvaluationEngineCUDA

        return EvaluationEngineCUDA()
    elif platform_name == "jax":
        from .evaluation_jax import EvaluationEngineJAX

        return EvaluationEngineJAX()
    else:
        raise NotImplementedError(f"No evaluation engine available for platform: {platform_name}.")


def get_lora_config_class() -> type:
    """Get the LoRAConfig class for the current platform.

    Returns:
        LoRAConfig class appropriate for the platform.
    """
    platform_name = get_training_platform()

    if platform_name == "mlx":
        from .lora_mlx import LoRAConfig

        return LoRAConfig
    elif platform_name == "cuda":
        from .lora_cuda import LoRAConfigCUDA

        return LoRAConfigCUDA
    elif platform_name == "jax":
        from .lora_jax import LoRAConfigJAX

        return LoRAConfigJAX
    else:
        raise NotImplementedError(f"No LoRA support available for platform: {platform_name}.")


def get_loss_landscape_computer() -> Any:
    """Get the loss landscape computer for the current platform.

    Returns:
        LossLandscapeComputer instance appropriate for the platform.
    """
    platform_name = get_training_platform()

    if platform_name == "mlx":
        from .loss_landscape_mlx import LossLandscapeComputer

        return LossLandscapeComputer()
    elif platform_name == "cuda":
        from .loss_landscape_cuda import LossLandscapeComputerCUDA

        return LossLandscapeComputerCUDA()
    elif platform_name == "jax":
        from .loss_landscape_jax import LossLandscapeComputerJAX

        return LossLandscapeComputerJAX()
    else:
        raise NotImplementedError(
            f"No loss landscape computer available for platform: {platform_name}."
        )


__all__ = [
    "get_training_platform",
    "get_training_engine",
    "get_checkpoint_manager",
    "get_evaluation_engine",
    "get_lora_config_class",
    "get_loss_landscape_computer",
]
