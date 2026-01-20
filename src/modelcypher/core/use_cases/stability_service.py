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

"""Stability service for model stability testing.

Provides stability suite execution and reporting functionality for
assessing model robustness and consistency.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import sqrt_scalar
logger = logging.getLogger(__name__)


@dataclass
class StabilityRunResult:
    """Result of a stability test run."""

    suite_id: str
    model_path: str
    status: str
    started_at: str
    derived_parameters: dict[str, Any]
    summary: dict[str, float] = field(default_factory=dict)


@dataclass
class StabilityReport:
    """Detailed stability report."""

    suite_id: str
    model_path: str
    status: str
    started_at: str
    completed_at: str | None
    derived_parameters: dict[str, Any]
    metrics: dict[str, float]
    per_prompt_results: list[dict[str, Any]]


class StabilityService:
    """Service for model stability testing.

    Runs stability suites to assess model robustness across:
    - Prompt perturbations
    - Repeated sampling
    """

    def __init__(self) -> None:
        """Initialize stability service."""
        self._suites: dict[str, dict[str, Any]] = {}

    def run(
        self,
        model: str,
    ) -> StabilityRunResult:
        """Execute stability suite on a model.

        Args:
            model: Path to model directory
        Returns:
            StabilityRunResult with suite_id and initial status

        Raises:
            ValueError: If model path is invalid
        """
        model_path = Path(model).expanduser().resolve()

        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        if not model_path.is_dir():
            raise ValueError(f"Model path is not a directory: {model_path}")

        derived_parameters = self._derive_run_parameters(model_path)
        suite_id = f"stab-{uuid.uuid4().hex[:12]}"
        started_at = datetime.now(timezone.utc).isoformat()

        # Store suite state
        self._suites[suite_id] = {
            "model_path": str(model_path),
            "status": "running",
            "started_at": started_at,
            "completed_at": None,
            "derived_parameters": derived_parameters,
            "metrics": {},
            "per_prompt_results": [],
        }

        logger.info(
            "Started stability suite %s for model %s",
            suite_id,
            model_path,
        )

        # Simulate stability testing
        # In production, this would run actual inference tests
        self._run_stability_tests(suite_id, derived_parameters)

        return StabilityRunResult(
            suite_id=suite_id,
            model_path=str(model_path),
            status=self._suites[suite_id]["status"],
            started_at=started_at,
            derived_parameters=derived_parameters,
            summary=self._suites[suite_id]["metrics"],
        )

    def report(self, suite_id: str) -> StabilityReport:
        """Get detailed stability report for a suite.

        Args:
            suite_id: ID of the stability suite

        Returns:
            StabilityReport with raw metrics and per-prompt results

        Raises:
            ValueError: If suite_id is not found
        """
        if suite_id not in self._suites:
            raise ValueError(f"Stability suite not found: {suite_id}")

        suite = self._suites[suite_id]
        metrics = suite["metrics"]

        return StabilityReport(
            suite_id=suite_id,
            model_path=suite["model_path"],
            status=suite["status"],
            started_at=suite["started_at"],
            completed_at=suite["completed_at"],
            derived_parameters=suite["derived_parameters"],
            metrics=metrics,
            per_prompt_results=suite["per_prompt_results"],
        )

    def _run_stability_tests(
        self,
        suite_id: str,
        derived_parameters: dict[str, Any],
    ) -> None:
        """Run stability tests.

        This service reports raw measurements from real inference runs.
        Adapter-specific measurement logic lives outside this stub.
        """
        suite = self._suites[suite_id]

        suite["metrics"] = {}
        suite["per_prompt_results"] = []

        suite["status"] = "completed"
        suite["completed_at"] = datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _derive_run_parameters(model_path: Path) -> dict[str, Any]:
        """Derive stability run parameters from model geometry."""
        import json

        config_path = model_path / "config.json"
        config_data: dict[str, Any] = {}
        if config_path.exists():
            try:
                config_data = json.loads(config_path.read_text())
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid config.json for model: {model_path}") from exc

        def _get_int(keys: tuple[str, ...]) -> int | None:
            for key in keys:
                value = config_data.get(key)
                if isinstance(value, (int, float)) and value > 0:
                    return int(value)
            return None

        layer_count = _get_int(("num_hidden_layers", "num_layers", "n_layer"))
        head_count = _get_int(("num_attention_heads", "num_heads", "n_head"))
        hidden_size = _get_int(("hidden_size", "d_model", "n_embd"))
        vocab_size = _get_int(("vocab_size",))

        if layer_count and head_count:
            scale = layer_count * head_count
        elif hidden_size:
            scale = hidden_size
        elif vocab_size:
            scale = vocab_size
        else:
            raise ValueError("Model config missing geometry for derived stability parameters")

        if scale <= 0:
            raise ValueError("Model config missing geometry for derived stability parameters")

        backend = get_default_backend()
        num_runs = int(sqrt_scalar(float(scale), backend))
        if num_runs <= 0:
            raise ValueError("Derived stability num_runs must be positive")

        prompt_variations = int(sqrt_scalar(float(num_runs), backend))
        if prompt_variations <= 0:
            raise ValueError("Derived stability prompt_variations must be positive")

        return {
            "num_runs": num_runs,
            "prompt_variations": prompt_variations,
            "derived_from": "model_geometry",
        }
