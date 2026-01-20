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

"""Factory for creating dual-path inference implementations.

This factory handles platform detection and returns the appropriate
dual-path generator for the current environment. Moved from domain to
infrastructure to respect hexagonal architecture boundaries.
"""

from __future__ import annotations

import os
import sys


def _get_inference_platform() -> str:
    """Get the current inference platform identifier.

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


def get_dual_path_generator_class() -> type:
    """Get the DualPathGenerator class for the current platform.

    Returns:
        DualPathGenerator class appropriate for the platform.

    Raises:
        NotImplementedError: If no supported platform is available.
    """
    platform_name = _get_inference_platform()

    if platform_name == "mlx":
        from modelcypher.infrastructure.dual_path_mlx import DualPathGenerator

        return DualPathGenerator
    elif platform_name == "cuda":
        from modelcypher.infrastructure.dual_path_cuda import DualPathGeneratorCUDA

        return DualPathGeneratorCUDA
    elif platform_name == "jax":
        from modelcypher.infrastructure.dual_path_jax import DualPathGeneratorJAX

        return DualPathGeneratorJAX
    else:
        raise NotImplementedError(
            f"No dual-path generator available for platform: {platform_name}. "
            "Install MLX on macOS, PyTorch with CUDA on Linux, or JAX for TPU/GPU."
        )


def get_security_scan_metrics_class() -> type:
    """Get the SecurityScanMetrics class for the current platform.

    Returns:
        SecurityScanMetrics class appropriate for the platform.
    """
    platform_name = _get_inference_platform()

    if platform_name == "mlx":
        from modelcypher.infrastructure.dual_path_mlx import SecurityScanMetrics

        return SecurityScanMetrics
    elif platform_name == "cuda":
        from modelcypher.infrastructure.dual_path_cuda import SecurityScanMetricsCUDA

        return SecurityScanMetricsCUDA
    elif platform_name == "jax":
        from modelcypher.infrastructure.dual_path_jax import SecurityScanMetricsJAX

        return SecurityScanMetricsJAX
    else:
        raise NotImplementedError(
            f"No security scan metrics available for platform: {platform_name}."
        )


__all__ = [
    "get_dual_path_generator_class",
    "get_security_scan_metrics_class",
]
