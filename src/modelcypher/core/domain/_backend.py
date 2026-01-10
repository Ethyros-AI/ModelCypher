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
importing any outer layer code. The backend auto-initializes on first
access, or can be explicitly set by entry points for full control.

Usage in domain classes:

    from modelcypher.core.domain._backend import get_default_backend

    class SomeAnalyzer:
        def __init__(self, backend: Backend | None = None) -> None:
            self._backend = backend or get_default_backend()

Entry points may initialize the backend explicitly:

    from modelcypher.backends import get_backend
    from modelcypher.core.domain._backend import set_default_backend

    set_default_backend(get_backend("mlx"))
"""

from __future__ import annotations

import importlib.util
import os
import platform
import subprocess
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

    runtime_check = os.environ.get("MC_MLX_RUNTIME_CHECK", "1").lower() in ("1", "true", "yes")
    if runtime_check:
        if _is_sandboxed_environment():
            allow_probe = (os.environ.get("MC_ALLOW_MLX_RUNTIME_PROBE_IN_SANDBOX") or "").lower()
            if allow_probe not in ("1", "true", "yes"):
                _mlx_probe_result = True
                _mlx_probe_error = None
                return True

        ok, err = _probe_mlx_runtime()
        if not ok:
            _mlx_probe_result = False
            _mlx_probe_error = err or "MLX runtime probe failed"
            return False

    _mlx_probe_result = True
    _mlx_probe_error = None
    return True


def get_mlx_probe_error() -> str | None:
    """Get the error message from the last MLX probe, if any."""
    return _mlx_probe_error


def get_default_backend() -> "Backend":
    """Get the default compute backend.

    Auto-detects and initializes the backend on first access so callers
    do not need to set MC_BACKEND on single-backend machines.
    """
    global _default_backend
    if _default_backend is None:
        from modelcypher.backends import detect_default_backend_type, get_backend

        backend_type = detect_default_backend_type()
        _default_backend = get_backend(backend_type)
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


def _is_sandboxed_environment() -> bool:
    """Check if running in a sandboxed environment (VSCode extension, etc.)."""
    # VSCode/Claude Code extension indicators
    if os.environ.get("VSCODE_PID") or os.environ.get("VSCODE_CWD"):
        return True
    term_program = (os.environ.get("TERM_PROGRAM") or "").strip().lower()
    if term_program in {"vscode", "visual studio code"}:
        return True
    return False


def _probe_mlx_runtime() -> tuple[bool, str | None]:
    """Probe MLX runtime initialization in a subprocess to avoid hard crashes.

    Uses environment variables to suppress crash dialogs on macOS.
    """
    if _is_sandboxed_environment():
        allow_probe = (os.environ.get("MC_ALLOW_MLX_RUNTIME_PROBE_IN_SANDBOX") or "").lower()
        if allow_probe not in ("1", "true", "yes"):
            return (
                False,
                "MLX runtime probe disabled in VSCode/Claude Code sandbox to avoid crash reports. "
                "Run from Terminal.app to use MLX, or set MC_ALLOW_MLX_RUNTIME_PROBE_IN_SANDBOX=1 to force the probe.",
            )

    code = "import mlx.core as mx; mx.random.key(0); mx.zeros((1,))"

    # Environment variables to suppress crash reporting/dialogs
    probe_env = os.environ.copy()
    probe_env.update({
        # Suppress Apple crash reporter dialog
        "LLVM_DISABLE_CRASH_REPORT": "1",
        # Disable os_log activity tracing
        "OS_ACTIVITY_MODE": "disable",
        # Prevent core dumps
        "MALLOC_CHECK_": "0",
    })

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
            env=probe_env,
            timeout=30,  # Prevent hanging
        )
    except subprocess.TimeoutExpired:
        return False, "MLX runtime probe timed out (30s)"
    except Exception as exc:
        return False, f"MLX runtime probe failed: {exc}"

    if result.returncode == 0:
        return True, None

    # Check for sandbox-related failures
    is_sandboxed = _is_sandboxed_environment()

    # Parse error details
    detail = (result.stderr or result.stdout).strip()

    # Detect crash failures (SIGABRT=-6, SIGKILL=-9)
    if result.returncode in (-6, -9) or (not detail and result.returncode != 0):
        if is_sandboxed:
            detail = (
                "MLX failed to load in this sandboxed environment (VSCode/Claude Code). "
                "This can occur due to GPU access restrictions or code signing enforcement. "
                "Workarounds:\n"
                "  1. Run ModelCypher from Terminal.app directly\n"
                "  2. If using dev MLX, try a signed release version"
            )
        else:
            detail = (
                f"MLX crashed during initialization (exit code {result.returncode}). "
                "This may indicate a Metal driver issue, GPU access restriction, or "
                "code signing problem. "
                "Set MC_DISABLE_MLX=1 to skip MLX."
            )

    if not detail:
        detail = f"MLX runtime probe exited with code {result.returncode}"

    return False, detail
