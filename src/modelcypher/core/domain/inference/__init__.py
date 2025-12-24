"""
Inference domain - model inference utilities and types.

Platform-Specific Implementations:
- MLX (macOS): *_mlx.py files
- CUDA (Linux): *_cuda.py files
- JAX (TPU/GPU): *_jax.py files
- Use _platform module for automatic selection
"""
from __future__ import annotations

from .adapter_pool import *  # noqa: F401,F403
from .comparison import *  # noqa: F401,F403
from .dual_path_mlx import *  # noqa: F401,F403
from .entropy_dynamics import *  # noqa: F401,F403
from .types import *  # noqa: F401,F403

# Platform selection (auto-detects MLX on macOS, CUDA on Linux, JAX on TPU)
from ._platform import (
    get_inference_platform,
    get_dual_path_generator_class,
    get_dual_path_config_class,
    get_security_scan_metrics_class,
)
