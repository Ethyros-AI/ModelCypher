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
importing any outer layer code. The backend MUST be initialized by the
application entry point before any domain code runs.

Usage in domain classes:

    from modelcypher.core.domain._backend import get_default_backend

    class SomeAnalyzer:
        def __init__(self, backend: Backend | None = None) -> None:
            self._backend = backend or get_default_backend()

Entry points must initialize the backend:

    from modelcypher.backends import get_backend
    from modelcypher.core.domain._backend import set_default_backend

    set_default_backend(get_backend("mlx"))
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

_default_backend: "Backend | None" = None
_mlx_probe_result: bool | None = None
_mlx_probe_error: str | None = None


def probe_mlx_available(*, explicit: bool = False) -> bool:
    """Check whether MLX is available on this system.

    This is pure platform detection with no outer layer imports.
    Verifies platform support and package presence only.

    Args:
        explicit: If True, this is an explicit user request (for error messages).

    Returns:
        True if MLX is available, False otherwise.
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


def get_mlx_probe_error() -> str | None:
    """Get the error message from the last MLX probe, if any."""
    return _mlx_probe_error


def get_default_backend() -> "Backend":
    """Get the default compute backend.

    Returns:
        The current default backend instance.

    Raises:
        RuntimeError: If no backend has been set. Entry points must call
            set_default_backend() before any domain code runs.
    """
    if _default_backend is None:
        raise RuntimeError(
            "No default backend has been set. "
            "Entry points must call set_default_backend() before using domain code. "
            "Example: set_default_backend(get_backend('mlx'))"
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
