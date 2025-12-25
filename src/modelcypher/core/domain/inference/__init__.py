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
- Use _platform module for automatic selection
"""

from __future__ import annotations

# Platform selection (auto-detects MLX on macOS, CUDA on Linux, JAX on TPU)
from ._platform import (
    get_dual_path_config_class,
    get_dual_path_generator_class,
    get_inference_platform,
    get_security_scan_metrics_class,
)
from .adapter_pool import *  # noqa: F401,F403
from .comparison import *  # noqa: F401,F403
from .dual_path_mlx import *  # noqa: F401,F403
from .entropy_dynamics import *  # noqa: F401,F403
from .types import *  # noqa: F401,F403
