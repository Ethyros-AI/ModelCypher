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

"""MLX platform probing and availability detection.

This module handles platform detection for MLX without importing MLX itself.
It verifies platform support, package presence, and runtime initialization.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import subprocess
import sys

_mlx_probe_result: bool | None = None
_mlx_probe_error: str | None = None


def probe_mlx_available(*, explicit: bool = False) -> bool:
    """Check whether MLX is available on this system.

    Verifies platform support, package presence, and runtime initialization.

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

    # Runtime probe - verify MLX actually works
    runtime_check = os.environ.get("MC_MLX_RUNTIME_CHECK", "1").lower() in (
        "1",
        "true",
        "yes",
    )
    if runtime_check:
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


def reset_mlx_probe() -> None:
    """Reset the MLX probe cache. Useful for testing."""
    global _mlx_probe_result, _mlx_probe_error
    _mlx_probe_result = None
    _mlx_probe_error = None


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
    code = "import mlx.core as mx; mx.random.key(0); mx.zeros((1,))"

    # Environment variables to suppress crash reporting/dialogs
    probe_env = os.environ.copy()
    probe_env.update(
        {
            # Suppress Apple crash reporter dialog
            "LLVM_DISABLE_CRASH_REPORT": "1",
            # Disable os_log activity tracing
            "OS_ACTIVITY_MODE": "disable",
            # Prevent core dumps
            "MALLOC_CHECK_": "0",
        }
    )

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
