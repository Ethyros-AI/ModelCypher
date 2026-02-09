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

"""Reasoning geometry validation orchestration.

Promotes the cross-model reasoning geometry validation experiment to a stable
use-case entrypoint callable from the CLI.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReasoningGeometryValidationRequest:
    """Configuration for running reasoning geometry validation."""

    models: tuple[str, ...] = ()
    benchmarks: tuple[str, ...] = ("gsm8k", "arithmetic")
    samples: int = 500
    max_tokens: int = 256
    seed: int = 42
    batch_size: int = 50
    output_dir: Path = Path("results/reasoning_geometry_validation")


@dataclass(frozen=True)
class ReasoningGeometryValidationResult:
    """Paths and process metadata from a validation run."""

    command: tuple[str, ...]
    output_dir: Path
    report_path: Path
    results_path: Path

    def to_dict(self) -> dict[str, str | list[str]]:
        return {
            "command": list(self.command),
            "outputDir": str(self.output_dir),
            "reportPath": str(self.report_path),
            "resultsPath": str(self.results_path),
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _validation_script_path() -> Path:
    return _repo_root() / "scripts" / "reasoning_geometry_validation.py"


def run_reasoning_geometry_validation(
    request: ReasoningGeometryValidationRequest,
) -> ReasoningGeometryValidationResult:
    """Run the cross-model reasoning geometry validation workflow."""
    script_path = _validation_script_path()
    if not script_path.exists():
        raise FileNotFoundError(
            f"Reasoning geometry validation script not found: {script_path}"
        )

    output_dir = request.output_dir.expanduser().resolve()
    command: list[str] = [
        sys.executable,
        str(script_path),
        "--output",
        str(output_dir),
        "--samples",
        str(request.samples),
        "--max-tokens",
        str(request.max_tokens),
        "--seed",
        str(request.seed),
        "--batch-size",
        str(request.batch_size),
    ]

    if request.models:
        command.extend(["--models", *request.models])
    if request.benchmarks:
        command.extend(["--benchmark", *request.benchmarks])

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            "Reasoning geometry validation failed "
            f"(exit={completed.returncode}): {detail}"
        )

    return ReasoningGeometryValidationResult(
        command=tuple(command),
        output_dir=output_dir,
        report_path=output_dir / "VALIDATION_REPORT.md",
        results_path=output_dir / "analysis" / "per_model_results.json",
    )

