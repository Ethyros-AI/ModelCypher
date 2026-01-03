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
Inference domain - model inference utilities and types.

Platform-Specific Implementations:
- MLX (macOS): *_mlx.py files
- CUDA (Linux): *_cuda.py files
- JAX (TPU/GPU): *_jax.py files

For platform-specific generators, use infrastructure factories:
    from modelcypher.infrastructure.dual_path_factory import get_dual_path_generator_class

For orchestration (CheckpointComparisonCoordinator), use:
    from modelcypher.core.use_cases.inference import CheckpointComparisonCoordinator
"""

from __future__ import annotations

from .activation_stream import ActivationFrame, ActivationStream
from .adapter_pool import *  # noqa: F401,F403
from .entropy_dynamics import *  # noqa: F401,F403
from .types import *  # noqa: F401,F403

__all__ = [
    # Activation streaming for real-time visualization
    "ActivationFrame",
    "ActivationStream",
]
