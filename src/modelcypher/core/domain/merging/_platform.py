"""
Merging Platform Selector.

The LoRA adapter merger now uses a unified geometric implementation that
delegates to the Backend abstraction for compute. No platform-specific
merger classes needed—one correct geometric merge works everywhere.

Usage:
    from modelcypher.core.domain.merging._platform import (
        get_merging_platform,
        get_lora_adapter_merger_class,
    )

    platform = get_merging_platform()
    Merger = get_lora_adapter_merger_class()
    report = Merger.merge(adapters, output)
"""

from __future__ import annotations

import platform as sys_platform


def _is_mlx_available() -> bool:
    """Check if MLX is available (macOS with Apple Silicon)."""
    if sys_platform.system() != "Darwin":
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
    """Get the unified LoRAAdapterMerger class.

    The merger uses geometric alignment (Procrustes + permutation re-basin)
    and delegates to the Backend abstraction for compute operations.
    Works on all platforms.

    Returns:
        LoRAAdapterMerger class.
    """
    from .lora_adapter_merger import LoRAAdapterMerger
    return LoRAAdapterMerger


__all__ = [
    "get_merging_platform",
    "get_lora_adapter_merger_class",
]
