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

"""Factory for creating training engine implementations.

This factory handles platform detection and returns the appropriate
training engine for the current environment. Moved from domain to
infrastructure to respect hexagonal architecture boundaries.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


def _get_training_platform() -> str:
    """Get the current training platform identifier.

    Returns:
        'mlx' on macOS with Apple Silicon
        'cuda' on Linux with NVIDIA GPU
        'jax' on Linux with JAX (TPU/GPU)
        'cpu' otherwise
    """
    env_backend = os.environ.get("MC_BACKEND", "").lower()
    if not env_backend:
        env_backend = os.environ.get("MODELCYPHER_BACKEND", "").lower()
    if env_backend in ("mlx", "cuda", "jax"):
        return env_backend

    # Check MLX availability
    if sys.platform == "darwin":
        if os.environ.get("MC_DISABLE_MLX", "").lower() not in ("1", "true", "yes"):
            from modelcypher.backends.mlx_probe import probe_mlx_available

            if probe_mlx_available(explicit=False):
                return "mlx"

    # Check CUDA
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    # Check JAX
    try:
        import jax  # noqa: F401

        return "jax"
    except ImportError:
        pass

    return "cpu"


def get_training_engine() -> Any:
    """Get the training engine for the current platform.

    Returns:
        TrainingEngine instance appropriate for the platform.

    Raises:
        NotImplementedError: If no supported training platform is available.
    """
    platform_name = _get_training_platform()

    if platform_name == "mlx":
        from modelcypher.adapters.training.mlx.engine import TrainingEngine

        return TrainingEngine()
    elif platform_name == "cuda":
        from modelcypher.adapters.training.cuda.engine import TrainingEngineCUDA

        return TrainingEngineCUDA()
    elif platform_name == "jax":
        from modelcypher.adapters.training.jax.engine import TrainingEngineJAX

        return TrainingEngineJAX()
    else:
        raise RuntimeError(
            f"No training engine available for platform: {platform_name}. "
            "Install MLX on macOS, PyTorch with CUDA on Linux, or JAX for TPU/GPU."
        )


def get_checkpoint_manager(max_checkpoints: int = 3) -> Any:
    """Get the checkpoint manager for the current platform.

    Args:
        max_checkpoints: Maximum number of checkpoints to retain.

    Returns:
        CheckpointManager instance appropriate for the platform.
    """
    platform_name = _get_training_platform()

    if platform_name == "mlx":
        from modelcypher.adapters.training.mlx.checkpoints import CheckpointManager

        return CheckpointManager(max_checkpoints=max_checkpoints)
    elif platform_name == "cuda":
        from modelcypher.adapters.training.cuda.checkpoints import (
            CheckpointManagerCUDA,
        )

        return CheckpointManagerCUDA(max_checkpoints=max_checkpoints)
    elif platform_name == "jax":
        from modelcypher.adapters.training.jax.checkpoints import (
            CheckpointManagerJAX,
        )

        return CheckpointManagerJAX(max_checkpoints=max_checkpoints)
    else:
        raise RuntimeError(f"No checkpoint manager available for platform: {platform_name}.")


def get_evaluation_engine() -> Any:
    """Get the evaluation engine for the current platform.

    Returns:
        EvaluationEngine instance appropriate for the platform.
    """
    platform_name = _get_training_platform()

    if platform_name == "mlx":
        from modelcypher.adapters.training.mlx.evaluation import EvaluationEngine

        return EvaluationEngine()
    elif platform_name == "cuda":
        from modelcypher.adapters.training.cuda.evaluation import (
            EvaluationEngineCUDA,
        )

        return EvaluationEngineCUDA()
    elif platform_name == "jax":
        from modelcypher.adapters.training.jax.evaluation import EvaluationEngineJAX

        return EvaluationEngineJAX()
    else:
        raise RuntimeError(f"No evaluation engine available for platform: {platform_name}.")


def get_lora_config_class() -> type:
    """Get the LoRAConfig class for the current platform.

    Returns:
        LoRAConfig class appropriate for the platform.
    """
    platform_name = _get_training_platform()

    if platform_name == "mlx":
        from modelcypher.adapters.training.mlx.lora import LoRAConfig

        return LoRAConfig
    elif platform_name == "cuda":
        from modelcypher.adapters.training.cuda.lora import LoRAConfigCUDA

        return LoRAConfigCUDA
    elif platform_name == "jax":
        from modelcypher.adapters.training.jax.lora import LoRAConfigJAX

        return LoRAConfigJAX
    else:
        raise RuntimeError(f"No LoRA support available for platform: {platform_name}.")


def get_loss_landscape_computer() -> Any:
    """Get the loss landscape computer for the current platform.

    Returns:
        LossLandscapeComputer instance appropriate for the platform.
    """
    platform_name = _get_training_platform()

    if platform_name == "mlx":
        from modelcypher.adapters.training.mlx.loss_landscape import (
            LossLandscapeComputer,
        )

        return LossLandscapeComputer()
    elif platform_name == "cuda":
        from modelcypher.adapters.training.cuda.loss_landscape import (
            LossLandscapeComputerCUDA,
        )

        return LossLandscapeComputerCUDA()
    elif platform_name == "jax":
        from modelcypher.adapters.training.jax.loss_landscape import (
            LossLandscapeComputerJAX,
        )

        return LossLandscapeComputerJAX()
    else:
        raise RuntimeError(f"No loss landscape computer available for platform: {platform_name}.")


__all__ = [
    "get_training_engine",
    "get_checkpoint_manager",
    "get_evaluation_engine",
    "get_lora_config_class",
    "get_loss_landscape_computer",
]
