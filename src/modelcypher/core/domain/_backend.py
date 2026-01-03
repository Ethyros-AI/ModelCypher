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

"""Default backend manager for domain classes.

This module provides a way for domain classes to access a compute backend
without directly importing MLX or other platform-specific implementations.

Supported backends:
    - mlx: Apple MLX for macOS (default on Darwin)
    - cuda: PyTorch CUDA for NVIDIA GPUs
    - jax: JAX for TPU/GPU/CPU (Google/Anthropic infrastructure)
    - numpy: NumPy CPU backend (no GPU required)

Usage in domain classes:

    from modelcypher.core.domain._backend import get_default_backend

    class SomeAnalyzer:
        def __init__(self, backend: Backend | None = None) -> None:
            self._backend = backend or get_default_backend()

To select a specific backend:

    from modelcypher.core.domain._backend import get_backend, set_default_backend
    set_default_backend(get_backend("jax"))
"""

from __future__ import annotations

import os
import platform
import sys
import importlib.util
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

BackendType = Literal["mlx", "jax", "cuda", "numpy"]

_default_backend: Backend | None = None
_mlx_probe_result: bool | None = None
_mlx_probe_error: str | None = None


def probe_mlx_available(*, explicit: bool = False) -> bool:
    """Check whether MLX is available on this system.

    Avoids importing MLX at probe time to keep initialization fast and
    prevent crash-prone subprocess probes. This verifies platform support
    and package presence only.
    """
    global _mlx_probe_result, _mlx_probe_error
    if _mlx_probe_result is not None:
        return _mlx_probe_result
    if os.environ.get("MC_DISABLE_MLX", "").lower() in ("1", "true", "yes"):
        _mlx_probe_result = False
        _mlx_probe_error = "MLX disabled via MC_DISABLE_MLX"
        return False
    if sys.platform != "darwin":
        _mlx_probe_result = False
        _mlx_probe_error = "MLX requires macOS"
        return False
    if platform.machine() not in ("arm64", "aarch64"):
        _mlx_probe_result = False
        _mlx_probe_error = "MLX requires Apple Silicon"
        return False

    if importlib.util.find_spec("mlx.core") is None:
        _mlx_probe_result = False
        _mlx_probe_error = "MLX not installed"
        return False

    _mlx_probe_result = True
    _mlx_probe_error = None
    return True


def get_backend(backend_type: BackendType) -> Backend:
    """Get a specific backend by type.

    Args:
        backend_type: One of "mlx", "jax", "cuda", "numpy"

    Returns:
        The requested backend instance.

    Raises:
        ImportError: If the backend's dependencies are not installed.
        ValueError: If the backend type is not recognized.
    """
    if backend_type == "mlx":
        if not probe_mlx_available(explicit=True):
            detail = _mlx_probe_error or "MLX probe failed"
            raise RuntimeError(
                "MLX backend requested but failed to initialize. "
                f"{detail}. Set MC_DISABLE_MLX=1 to force fallback."
            )
        from modelcypher.backends.mlx_backend import MLXBackend

        return MLXBackend()
    elif backend_type == "jax":
        from modelcypher.backends.jax_backend import JAXBackend

        return JAXBackend()
    elif backend_type == "cuda":
        from modelcypher.backends.cuda_backend import CUDABackend

        return CUDABackend()
    elif backend_type == "numpy":
        from modelcypher.backends.numpy_backend import NumpyBackend

        return NumpyBackend()
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def _detect_default_backend_type() -> BackendType:
    """Detect the best available backend for the current platform.

    Priority:
        1. MC_BACKEND (or MODELCYPHER_BACKEND) environment variable
        2. MLX on macOS (Darwin)
        3. JAX if available

    Raises:
        RuntimeError: If no GPU backend is available.
    """
    # Check environment variable override
    env_backend = os.environ.get("MC_BACKEND", "").lower()
    if not env_backend:
        env_backend = os.environ.get("MODELCYPHER_BACKEND", "").lower()
    if env_backend in ("mlx", "jax", "cuda", "numpy"):
        if env_backend == "mlx":
            if not probe_mlx_available(explicit=True):
                detail = _mlx_probe_error or "MLX probe failed"
                raise RuntimeError(
                    "MC_BACKEND=mlx requested but MLX failed to initialize. "
                    f"{detail}. Set MC_DISABLE_MLX=1 to force fallback."
                )
        return env_backend  # type: ignore
    disable_mlx = os.environ.get("MC_DISABLE_MLX", "").lower() in ("1", "true", "yes")

    # macOS: prefer MLX (probe in subprocess to avoid in-process abort)
    if sys.platform == "darwin" and not disable_mlx:
        if probe_mlx_available(explicit=False):
            return "mlx"

    # Try CUDA (NVIDIA GPU)
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    # Try JAX (works on TPU, GPU, CPU)
    try:
        import jax  # noqa: F401

        return "jax"
    except ImportError:
        pass

    return "numpy"


def get_default_backend() -> Backend:
    """Get the default compute backend, auto-detecting if needed.

    Returns:
        The current default backend instance.

    Note:
        On first call, this detects the best available backend.
        Use MC_BACKEND (or MODELCYPHER_BACKEND) environment variable to override:
            MC_BACKEND=jax python script.py
        Or call set_default_backend() programmatically.
    """
    global _default_backend
    if _default_backend is None:
        backend_type = _detect_default_backend_type()
        _default_backend = get_backend(backend_type)
    return _default_backend


def set_default_backend(backend: Backend) -> None:
    """Set the default compute backend.

    Args:
        backend: The backend instance to use as default.

    Note:
        Call this before any domain classes are instantiated if you
        want to override the default MLXBackend (e.g., for testing).
    """
    global _default_backend
    _default_backend = backend


def reset_default_backend() -> None:
    """Reset the default backend to None.

    The next call to get_default_backend() will re-initialize MLXBackend.
    Useful for testing to ensure clean state.
    """
    global _default_backend
    _default_backend = None
