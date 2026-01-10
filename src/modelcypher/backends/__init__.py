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
from modelcypher.backends.lazy_backend import LazyBackend

BackendType = Literal["mlx", "jax", "cuda"]


def detect_default_backend_type() -> BackendType:
    """Detect the preferred backend for the current platform.

    Priority:
        1. MC_BACKEND (or MODELCYPHER_BACKEND) environment variable
        2. MLX on macOS (Apple Silicon)
        3. CUDA if available
        4. JAX if available
        5. (No automatic CPU fallback)

    Returns:
        The backend type string to use.

    Raises:
        RuntimeError: If an explicitly requested backend is unavailable.
    """
    from modelcypher.core.domain._backend import (
        probe_mlx_available,
        get_mlx_probe_error,
    )

    # Check environment variable override
    env_backend = os.environ.get("MC_BACKEND", "").lower()
    if not env_backend:
        env_backend = os.environ.get("MODELCYPHER_BACKEND", "").lower()

    if env_backend in ("mlx", "jax", "cuda"):
        if env_backend == "mlx":
            if not probe_mlx_available(explicit=True):
                detail = get_mlx_probe_error() or "MLX probe failed"
                raise RuntimeError(
                    "MC_BACKEND=mlx requested but MLX failed to initialize. "
                    f"{detail}."
                )
        return env_backend  # type: ignore[return-value]

    # Auto-detect best available backend
    if sys.platform == "darwin":
        if probe_mlx_available():
            return "mlx"
        detail = get_mlx_probe_error()
        message = "MLX backend unavailable on macOS."
        if detail:
            message = f"{message} {detail}."
        raise RuntimeError(message)

    if probe_mlx_available():
        return "mlx"

    # Try CUDA
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    # Try JAX
    try:
        import jax  # noqa: F401

        return "jax"
    except ImportError:
        pass

    detail = get_mlx_probe_error()
    message = "No GPU backend available. ModelCypher requires GPU acceleration."
    if detail:
        message = f"{message} MLX probe error: {detail}."
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
    from modelcypher.core.domain._backend import (
        probe_mlx_available,
        get_mlx_probe_error,
    )

    if backend_type == "mlx":
        if not probe_mlx_available(explicit=True):
            detail = get_mlx_probe_error() or "MLX probe failed"
            raise RuntimeError(
                "MLX backend requested but failed to initialize. "
                f"{detail}."
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

    This is the main entry point for applications to set up the backend.
    Must be called before any domain code that uses get_default_backend().

    Returns:
        The initialized backend instance.

    Example:
        from modelcypher.backends import initialize_default_backend
        initialize_default_backend()  # Now domain code can use get_default_backend()
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
    """Get the default backend.

    Note:
        Raises RuntimeError if initialize_default_backend() hasn't been called.
    """
    from modelcypher.core.domain._backend import get_default_backend

    return get_default_backend()


__all__ = [
    "Backend",
    "BackendType",
    "default_backend",
    "detect_default_backend_type",
    "get_backend",
    "initialize_default_backend",
    "LazyBackend",
    "MLXBackend",
    "JAXBackend",
    "CUDABackend",
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
