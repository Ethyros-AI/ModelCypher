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

from __future__ import annotations

import os
import sys
from typing import Literal

from modelcypher.ports.backend import Backend

BackendType = Literal["mlx", "jax", "cuda"]


def _try_mlx_available() -> tuple[bool, str | None]:
    """Check if MLX is available via Backend."""
    try:
        from modelcypher.backends.mlx_backend import MLXBackend
        info = MLXBackend().get_system_info()
        return info.get("available", False), None
    except Exception as e:
        return False, str(e)


def _try_cuda_available() -> bool:
    """Check if CUDA is available via Backend."""
    try:
        from modelcypher.backends.cuda_backend import CUDABackend
        info = CUDABackend().get_system_info()
        return info.get("available", False)
    except Exception:
        return False


def _try_jax_available() -> bool:
    """Check if JAX is available via Backend."""
    try:
        from modelcypher.backends.jax_backend import JAXBackend
        info = JAXBackend().get_system_info()
        return info.get("available", False)
    except Exception:
        return False


def detect_default_backend_type() -> BackendType:
    """Detect the preferred backend for the current platform.

    Priority:
        1. MC_BACKEND (or MODELCYPHER_BACKEND) environment variable
        2. MLX on macOS (Apple Silicon)
        3. CUDA if available
        4. JAX if available

    Returns:
        The backend type string to use.

    Raises:
        RuntimeError: If an explicitly requested backend is unavailable.
    """
    # Check environment variable override
    env_backend = os.environ.get("MC_BACKEND", "").lower()
    if not env_backend:
        env_backend = os.environ.get("MODELCYPHER_BACKEND", "").lower()

    if env_backend in ("mlx", "jax", "cuda"):
        if env_backend == "mlx":
            available, error = _try_mlx_available()
            if not available:
                raise RuntimeError(
                    f"MC_BACKEND=mlx requested but MLX failed to initialize. {error or 'Unknown error'}."
                )
        return env_backend  # type: ignore[return-value]

    # Auto-detect best available backend
    mlx_available, mlx_error = _try_mlx_available()

    if sys.platform == "darwin":
        if mlx_available:
            return "mlx"
        message = "MLX backend unavailable on macOS."
        if mlx_error:
            message = f"{message} {mlx_error}."
        raise RuntimeError(message)

    if mlx_available:
        return "mlx"

    if _try_cuda_available():
        return "cuda"

    if _try_jax_available():
        return "jax"

    message = "No GPU backend available. ModelCypher requires GPU acceleration."
    if mlx_error:
        message = f"{message} MLX probe error: {mlx_error}."
    raise RuntimeError(f"{message} Install MLX (macOS), CUDA (NVIDIA), or JAX (TPU/GPU).")


def get_backend(backend_type: BackendType) -> Backend:
    """Get a specific backend by type.

    Args:
        backend_type: One of "mlx", "jax", "cuda"

    Returns:
        The requested backend instance.

    Raises:
        ImportError: If the backend's dependencies are not installed.
        ValueError: If the backend type is not recognized.
        RuntimeError: If the backend failed to initialize.
    """
    if backend_type == "mlx":
        available, error = _try_mlx_available()
        if not available:
            raise RuntimeError(
                f"MLX backend requested but failed to initialize. {error or 'Unknown error'}."
            )
        from modelcypher.backends.mlx_backend import MLXBackend
        return MLXBackend()
    elif backend_type == "jax":
        from modelcypher.backends.jax_backend import JAXBackend
        return JAXBackend()
    elif backend_type == "cuda":
        from modelcypher.backends.cuda_backend import CUDABackend
        return CUDABackend()
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def initialize_default_backend() -> Backend:
    """Initialize the default backend based on platform detection.

    Returns:
        The initialized backend instance.
    """
    from modelcypher.core.domain._backend import get_default_backend, set_default_backend

    try:
        return get_default_backend()
    except RuntimeError:
        pass

    backend_type = detect_default_backend_type()
    backend = get_backend(backend_type)
    set_default_backend(backend)
    return backend


def default_backend() -> Backend:
    """Get the default backend."""
    from modelcypher.core.domain._backend import get_default_backend
    return get_default_backend()


__all__ = [
    "Backend",
    "BackendType",
    "default_backend",
    "detect_default_backend_type",
    "get_backend",
    "initialize_default_backend",
]


def __getattr__(name: str):
    """Lazy import backends to avoid import errors when dependencies missing."""
    if name == "MLXBackend":
        from modelcypher.backends.mlx_backend import MLXBackend
        return MLXBackend
    if name == "JAXBackend":
        from modelcypher.backends.jax_backend import JAXBackend
        return JAXBackend
    if name == "CUDABackend":
        from modelcypher.backends.cuda_backend import CUDABackend
        return CUDABackend
    if name == "LazyBackend":
        from modelcypher.backends.lazy_backend import LazyBackend
        return LazyBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
