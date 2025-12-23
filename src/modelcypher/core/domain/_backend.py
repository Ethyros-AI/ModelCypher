"""Default backend manager for domain classes.

This module provides a way for domain classes to access a compute backend
without directly importing MLX or other platform-specific implementations.

Usage in domain classes:

    from modelcypher.core.domain._backend import get_default_backend

    class SomeAnalyzer:
        def __init__(self, backend: Backend | None = None) -> None:
            self._backend = backend or get_default_backend()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

_default_backend: Backend | None = None


def get_default_backend() -> Backend:
    """Get the default compute backend, initializing MLXBackend if needed.

    Returns:
        The current default backend instance.

    Note:
        On first call, this lazily imports and instantiates MLXBackend.
        Use set_default_backend() to override with a different backend
        (e.g., NumpyBackend for testing).
    """
    global _default_backend
    if _default_backend is None:
        from modelcypher.backends.mlx_backend import MLXBackend

        _default_backend = MLXBackend()
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
