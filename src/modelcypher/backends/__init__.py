from __future__ import annotations

from modelcypher.ports.backend import Backend


def default_backend() -> Backend:
    """Get the default backend (delegates to _backend module)."""
    from modelcypher.core.domain._backend import get_default_backend
    return get_default_backend()


__all__ = [
    "Backend",
    "default_backend",
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
