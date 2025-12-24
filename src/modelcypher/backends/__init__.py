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
