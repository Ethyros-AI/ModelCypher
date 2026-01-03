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

"""Activation Provider Factory.

This module provides the factory function for creating ActivationProvider instances.
It lives in the infrastructure layer because it imports concrete adapter implementations.

Following hexagonal architecture:
- Ports define abstract protocols (ActivationProvider)
- Adapters implement those protocols (MLXActivationProvider, etc.)
- Infrastructure wires them together (this file)
"""

from __future__ import annotations

import sys

from modelcypher.ports.activation_provider import ActivationProvider


def get_activation_provider() -> ActivationProvider:
    """Get the appropriate activation provider for the current platform.

    Auto-selects:
    - MLXActivationProvider on macOS (Metal GPU)
    - CUDAActivationProvider on Linux with CUDA
    - JAXActivationProvider on Linux/TPU without CUDA

    Returns:
        An ActivationProvider instance for the current platform.

    Raises:
        RuntimeError: If no suitable backend is available.
    """
    # macOS: Use MLX (Metal GPU)
    if sys.platform == "darwin":
        try:
            from modelcypher.adapters.mlx_activation_provider import MLXActivationProvider

            return MLXActivationProvider()
        except ImportError as e:
            raise RuntimeError(
                "MLX not available on macOS. Install with: pip install mlx mlx-lm"
            ) from e

    # Linux/other: Try CUDA first, then JAX
    try:
        import torch

        if torch.cuda.is_available():
            from modelcypher.adapters.cuda_activation_provider import CUDAActivationProvider

            return CUDAActivationProvider()
    except ImportError:
        pass

    # Try JAX (works on CPU, TPU, and GPU)
    try:
        from modelcypher.adapters.jax_activation_provider import JAXActivationProvider

        provider = JAXActivationProvider()
        if provider.available:
            return provider
    except ImportError:
        pass

    # No suitable backend found
    raise RuntimeError(
        "No activation provider available. Install one of:\n"
        "  - macOS: pip install mlx mlx-lm\n"
        "  - CUDA: pip install torch\n"
        "  - JAX/TPU: pip install jax jaxlib"
    )


__all__ = ["get_activation_provider"]
