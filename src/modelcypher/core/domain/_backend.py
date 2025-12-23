"""Default backend manager for domain classes.

This module provides a way for domain classes to access a compute backend
without directly importing MLX or other platform-specific implementations.

Supported backends:
    - mlx: Apple MLX for macOS (default on Darwin)
    - jax: JAX for TPU/GPU/CPU (Google/Anthropic infrastructure)
    - cuda: PyTorch CUDA for NVIDIA GPUs (stub, not yet implemented)

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
import sys
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

BackendType = Literal["mlx", "jax", "cuda", "numpy"]

_default_backend: Backend | None = None


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
        from modelcypher.backends.mlx_backend import MLXBackend
        return MLXBackend()
    elif backend_type == "jax":
        from modelcypher.backends.jax_backend import JAXBackend
        return JAXBackend()
    elif backend_type == "cuda":
        from modelcypher.backends.cuda_backend import CUDABackend
        return CUDABackend()
    elif backend_type == "numpy":
        # NumpyBackend is primarily for testing; import from test fixtures
        from tests.conftest import NumpyBackend
        return NumpyBackend()
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def _detect_default_backend_type() -> BackendType:
    """Detect the best available backend for the current platform.

    Priority:
        1. MC_BACKEND environment variable
        2. MLX on macOS (Darwin)
        3. JAX if available
        4. NumPy fallback
    """
    # Check environment variable override
    env_backend = os.environ.get("MC_BACKEND", "").lower()
    if env_backend in ("mlx", "jax", "cuda", "numpy"):
        return env_backend  # type: ignore

    # macOS: prefer MLX
    if sys.platform == "darwin":
        try:
            import mlx.core  # noqa: F401
            return "mlx"
        except ImportError:
            pass

    # Try JAX (works on TPU, GPU, CPU)
    try:
        import jax  # noqa: F401
        return "jax"
    except ImportError:
        pass

    # Fallback to numpy
    return "numpy"


def get_default_backend() -> Backend:
    """Get the default compute backend, auto-detecting if needed.

    Returns:
        The current default backend instance.

    Note:
        On first call, this detects the best available backend.
        Use MC_BACKEND environment variable to override:
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
