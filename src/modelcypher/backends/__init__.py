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

from typing import Literal

from modelcypher.ports.backend import Backend
from modelcypher.backends.lazy_backend import LazyBackend

BackendType = Literal["mlx", "jax", "cuda", "numpy"]


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
    from modelcypher.core.domain._backend import probe_mlx_available, _mlx_probe_error

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


def default_backend() -> Backend:
    """Get the default backend (delegates to _backend module)."""
    from modelcypher.core.domain._backend import get_default_backend

    return get_default_backend()


__all__ = [
    "Backend",
    "BackendType",
    "default_backend",
    "get_backend",
    "LazyBackend",
    "MLXBackend",
    "JAXBackend",
    "CUDABackend",
    "NumpyBackend",
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
    if name == "NumpyBackend":
        from modelcypher.backends.numpy_backend import NumpyBackend

        return NumpyBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
