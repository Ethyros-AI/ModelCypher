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

"""Factory for creating ModelProbePort implementations.

This factory handles platform detection and returns the appropriate
model probe for the current environment. It lives in infrastructure
(not use_cases) to properly separate factory logic from service logic.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.model_probe import ModelProbePort


def get_model_probe() -> "ModelProbePort":
    """Get the appropriate model probe for the current platform.

    Returns:
        ModelProbePort implementation for the current backend.

    Platform selection:
        - macOS (Darwin): MLXModelProbe (when MLX runtime available)
        - macOS (Darwin) fallback: SafeTensorsModelProbe (reads safetensors headers)
        - Linux + CUDA available: CUDAModelProbe
        - Linux + JAX available: JAXModelProbe
        - Fallback: SafeTensorsModelProbe (reads safetensors headers)

    Raises:
        RuntimeError: If required dependencies are missing for the chosen probe.
    """
    if sys.platform == "darwin":
        from modelcypher.backends.mlx_probe import probe_mlx_available

        if not probe_mlx_available():
            from modelcypher.backends.safetensors_model_probe import SafeTensorsModelProbe

            return SafeTensorsModelProbe()
        try:
            from modelcypher.backends.mlx_model_probe import MLXModelProbe

            return MLXModelProbe()
        except ImportError as exc:
            raise RuntimeError(
                "MLX not available on macOS. Install with: pip install mlx"
            ) from exc

    # Linux: try CUDA first, then JAX
    try:
        from modelcypher.backends.cuda_model_probe import CUDAModelProbe

        probe = CUDAModelProbe()
        if probe.available:
            return probe
    except ImportError:
        pass

    try:
        from modelcypher.backends.jax_model_probe import JAXModelProbe

        probe = JAXModelProbe()
        if probe.available:
            return probe
    except ImportError:
        pass

    from modelcypher.backends.safetensors_model_probe import SafeTensorsModelProbe

    return SafeTensorsModelProbe()


__all__ = ["get_model_probe"]
