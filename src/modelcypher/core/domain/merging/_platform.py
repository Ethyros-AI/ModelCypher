"""
Merging Platform Selector.

This module provides lazy importing of platform-specific merging implementations.
On macOS, MLX implementations are used. On Linux with CUDA, PyTorch/CUDA
implementations will be used. On Linux with TPU/GPU, JAX implementations
will be used (when available).

Usage:
    from modelcypher.core.domain.merging._platform import (
        get_merging_platform,
        get_lora_adapter_merger,
    )

    platform = get_merging_platform()
    merger = get_lora_adapter_merger()

Platform-specific implementations:
- MLX (macOS/Apple Silicon): *_mlx.py files
- CUDA (Linux/NVIDIA GPU): *_cuda.py files
- JAX (Linux/TPU/GPU): *_jax.py files
"""

from __future__ import annotations

import platform
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .lora_adapter_merger_mlx import LoRAAdapterMerger


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


def get_merging_platform() -> str:
    """Get the current merging platform identifier.

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


def get_lora_adapter_merger_class() -> type:
    """Get the LoRAAdapterMerger class for the current platform.

    Returns:
        LoRAAdapterMerger class appropriate for the platform.

    Raises:
        NotImplementedError: If no supported platform is available.
    """
    platform_name = get_merging_platform()

    if platform_name == "mlx":
        from .lora_adapter_merger_mlx import LoRAAdapterMerger
        return LoRAAdapterMerger
    elif platform_name == "cuda":
        from .lora_adapter_merger_cuda import LoRAAdapterMergerCUDA
        return LoRAAdapterMergerCUDA
    elif platform_name == "jax":
        from .lora_adapter_merger_jax import LoRAAdapterMergerJAX
        return LoRAAdapterMergerJAX
    else:
        raise NotImplementedError(
            f"No LoRA adapter merger available for platform: {platform_name}. "
            "Install MLX on macOS, PyTorch with CUDA on Linux, or JAX for TPU/GPU."
        )


def get_lora_merge_strategy_enum() -> type:
    """Get the Strategy enum for the current platform.

    Returns:
        Strategy enum appropriate for the platform.
    """
    platform_name = get_merging_platform()

    if platform_name == "mlx":
        from .lora_adapter_merger_mlx import Strategy
        return Strategy
    elif platform_name == "cuda":
        from .lora_adapter_merger_cuda import StrategyCUDA
        return StrategyCUDA
    elif platform_name == "jax":
        from .lora_adapter_merger_jax import StrategyJAX
        return StrategyJAX
    else:
        raise NotImplementedError(
            f"No merge strategy available for platform: {platform_name}."
        )


def get_lora_merge_config_class() -> type:
    """Get the Config class for the current platform.

    Returns:
        Config class appropriate for the platform.
    """
    platform_name = get_merging_platform()

    if platform_name == "mlx":
        from .lora_adapter_merger_mlx import Config
        return Config
    elif platform_name == "cuda":
        from .lora_adapter_merger_cuda import ConfigCUDA
        return ConfigCUDA
    elif platform_name == "jax":
        from .lora_adapter_merger_jax import ConfigJAX
        return ConfigJAX
    else:
        raise NotImplementedError(
            f"No merge config available for platform: {platform_name}."
        )


__all__ = [
    "get_merging_platform",
    "get_lora_adapter_merger_class",
    "get_lora_merge_strategy_enum",
    "get_lora_merge_config_class",
]
