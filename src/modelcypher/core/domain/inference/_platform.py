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
Inference Platform Selector.

This module provides lazy importing of platform-specific inference implementations.
On macOS, MLX implementations are used. On Linux with CUDA, PyTorch/CUDA
implementations will be used. On Linux with TPU/GPU, JAX implementations
will be used (when available).

Usage:
    from modelcypher.core.domain.inference._platform import (
        get_inference_platform,
        get_dual_path_generator,
    )

    platform = get_inference_platform()
    generator_class = get_dual_path_generator_class()

Platform-specific implementations:
- MLX (macOS/Apple Silicon): *_mlx.py files
- CUDA (Linux/NVIDIA GPU): *_cuda.py files
- JAX (Linux/TPU/GPU): *_jax.py files
"""

from __future__ import annotations

import platform
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .dual_path_mlx import DualPathGenerator


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


def get_inference_platform() -> str:
    """Get the current inference platform identifier.

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


def get_dual_path_generator_class() -> type:
    """Get the DualPathGenerator class for the current platform.

    Returns:
        DualPathGenerator class appropriate for the platform.

    Raises:
        NotImplementedError: If no supported platform is available.
    """
    platform_name = get_inference_platform()

    if platform_name == "mlx":
        from .dual_path_mlx import DualPathGenerator
        return DualPathGenerator
    elif platform_name == "cuda":
        from .dual_path_cuda import DualPathGeneratorCUDA
        return DualPathGeneratorCUDA
    elif platform_name == "jax":
        from .dual_path_jax import DualPathGeneratorJAX
        return DualPathGeneratorJAX
    else:
        raise NotImplementedError(
            f"No dual-path generator available for platform: {platform_name}. "
            "Install MLX on macOS, PyTorch with CUDA on Linux, or JAX for TPU/GPU."
        )


def get_dual_path_config_class() -> type:
    """Get the DualPathGeneratorConfiguration class for the current platform.

    Returns:
        DualPathGeneratorConfiguration class appropriate for the platform.
    """
    platform_name = get_inference_platform()

    if platform_name == "mlx":
        from .dual_path_mlx import DualPathGeneratorConfiguration
        return DualPathGeneratorConfiguration
    elif platform_name == "cuda":
        from .dual_path_cuda import DualPathGeneratorConfigurationCUDA
        return DualPathGeneratorConfigurationCUDA
    elif platform_name == "jax":
        from .dual_path_jax import DualPathGeneratorConfigurationJAX
        return DualPathGeneratorConfigurationJAX
    else:
        raise NotImplementedError(
            f"No dual-path config available for platform: {platform_name}."
        )


def get_security_scan_metrics_class() -> type:
    """Get the SecurityScanMetrics class for the current platform.

    Returns:
        SecurityScanMetrics class appropriate for the platform.
    """
    platform_name = get_inference_platform()

    if platform_name == "mlx":
        from .dual_path_mlx import SecurityScanMetrics
        return SecurityScanMetrics
    elif platform_name == "cuda":
        from .dual_path_cuda import SecurityScanMetricsCUDA
        return SecurityScanMetricsCUDA
    elif platform_name == "jax":
        from .dual_path_jax import SecurityScanMetricsJAX
        return SecurityScanMetricsJAX
    else:
        raise NotImplementedError(
            f"No security scan metrics available for platform: {platform_name}."
        )


__all__ = [
    "get_inference_platform",
    "get_dual_path_generator_class",
    "get_dual_path_config_class",
    "get_security_scan_metrics_class",
]
