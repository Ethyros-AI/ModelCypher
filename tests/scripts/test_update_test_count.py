# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for deterministic README test-count collection."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import update_test_count


def test_collect_test_count_ignores_backend_selection(monkeypatch: Any) -> None:
    captured_env: dict[str, str] = {}

    def fake_run(*_args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured_env.update(kwargs["env"])
        return subprocess.CompletedProcess(
            args=kwargs,
            returncode=0,
            stdout="123 tests collected in 0.01s\n",
        )

    monkeypatch.setenv("MC_DISABLE_MLX", "1")
    monkeypatch.setenv("MC_BACKEND", "jax")
    monkeypatch.setattr(update_test_count.subprocess, "run", fake_run)

    assert update_test_count.collect_test_count() == 123
    assert "MC_DISABLE_MLX" not in captured_env
    assert "MC_BACKEND" not in captured_env
