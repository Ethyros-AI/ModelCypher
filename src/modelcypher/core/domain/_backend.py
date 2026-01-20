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

"""Default backend accessor for domain classes.

This module provides domain classes access to a compute backend without
importing any outer layer code. The backend MUST be set by entry points
before domain code uses it.

Usage in domain classes:

    from modelcypher.core.domain._backend import get_default_backend

    class SomeAnalyzer:
        def __init__(self, backend: Backend | None = None) -> None:
            self._backend = backend or get_default_backend()

Entry points MUST initialize the backend:

    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()  # Detects and sets the default backend

Or set explicitly:

    from modelcypher.backends import get_backend
    from modelcypher.core.domain._backend import set_default_backend
    set_default_backend(get_backend("mlx"))

IMPORTANT: This module follows hexagonal architecture - it does NOT import
from backends/. Entry points are responsible for backend initialization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

_default_backend: "Backend | None" = None


def get_default_backend() -> "Backend":
    """Get the default compute backend.

    Returns:
        The default backend instance.

    Raises:
        RuntimeError: If no backend has been set. Entry points must call
            initialize_default_backend() or set_default_backend() first.
    """
    if _default_backend is None:
        raise RuntimeError(
            "No default backend has been set. "
            "Entry points must call initialize_default_backend() from "
            "modelcypher.backends before using domain code, or inject a "
            "backend explicitly via set_default_backend()."
        )
    return _default_backend


def set_default_backend(backend: "Backend") -> None:
    """Set the default compute backend.

    Args:
        backend: The backend instance to use as default.

    Note:
        Must be called by entry points before any domain classes are used.
    """
    global _default_backend
    _default_backend = backend


def reset_default_backend() -> None:
    """Reset the default backend to None.

    Useful for testing to ensure clean state between tests.
    """
    global _default_backend
    _default_backend = None


def is_backend_initialized() -> bool:
    """Check if a default backend has been set.

    Returns:
        True if set_default_backend() has been called, False otherwise.
    """
    return _default_backend is not None


# Backward compatibility re-exports
# These probe functions are now in backends.mlx_probe but some code may
# import them from here. We re-export them lazily to avoid circular imports.
def probe_mlx_available(*, explicit: bool = False) -> bool:
    """Check whether MLX is available. See backends.mlx_probe for details."""
    from modelcypher.backends.mlx_probe import probe_mlx_available as _probe

    return _probe(explicit=explicit)


def get_mlx_probe_error() -> str | None:
    """Get MLX probe error. See backends.mlx_probe for details."""
    from modelcypher.backends.mlx_probe import get_mlx_probe_error as _get_error

    return _get_error()
