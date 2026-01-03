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

"""Factory for creating InferenceEngine implementations.

This factory handles platform detection and returns the appropriate
inference engine for the current environment. It lives in infrastructure
(not ports) to properly separate factory logic from service logic.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.inference import HiddenStateEngine


def get_inference_engine() -> "HiddenStateEngine":
    """Get the appropriate inference engine for the current platform.

    Returns:
        HiddenStateEngine implementation for the current backend.

    Platform selection:
        - macOS (Darwin): LocalInferenceEngine (MLX)
        - Linux + CUDA available: CUDAInferenceEngine
        - Linux + JAX available: JAXInferenceEngine
        - Fallback: CUDAInferenceEngine (requires PyTorch)

    Raises:
        RuntimeError: If no suitable backend is available.
    """
    if sys.platform == "darwin":
        try:
            from modelcypher.adapters.local_inference import LocalInferenceEngine

            return LocalInferenceEngine()
        except ImportError as exc:
            raise RuntimeError(
                "MLX not available on macOS. Install with: pip install mlx mlx-lm"
            ) from exc

    # Linux: try CUDA first, then JAX
    try:
        import torch

        if torch.cuda.is_available():
            from modelcypher.adapters.cuda_inference import CUDAInferenceEngine

            return CUDAInferenceEngine()
    except ImportError:
        pass

    try:
        from modelcypher.adapters.jax_inference import JAXInferenceEngine

        engine = JAXInferenceEngine()
        if engine.available:
            return engine
    except ImportError:
        pass

    # Fall back to CUDA engine (which can use CPU if needed)
    try:
        from modelcypher.adapters.cuda_inference import CUDAInferenceEngine

        return CUDAInferenceEngine()
    except ImportError:
        pass

    raise RuntimeError(
        "No inference engine available. Install one of:\n"
        "  - macOS: pip install mlx mlx-lm\n"
        "  - Linux/CUDA: pip install torch transformers\n"
        "  - Linux/TPU: pip install jax jaxlib transformers flax"
    )


__all__ = ["get_inference_engine"]
