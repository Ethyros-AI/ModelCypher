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

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import sqrt_scalar
from modelcypher.ports.inference import HiddenStateEngine
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

    def __init__(
        self,
        backend: Backend,
        inference_engine: HiddenStateEngine | None = None,
    ) -> None:
        """Initialize stability service."""
        self._suites: dict[str, dict[str, Any]] = {}
        self._backend = backend
        self._inference_engine = inference_engine

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

        derived_parameters = self._derive_run_parameters(model_path, self._backend)
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
        """
        suite = self._suites[suite_id]
        if self._inference_engine is None:
            raise RuntimeError("Inference engine required for stability tests")

        model_path = suite["model_path"]
        num_runs = int(derived_parameters.get("num_runs", 0))
        prompt_variations = int(derived_parameters.get("prompt_variations", 0))

        prompts = self._load_prompts(prompt_variations)
        backend = self._backend

        per_prompt_results: list[dict[str, Any]] = []
        for prompt in prompts:
            responses: list[str] = []
            token_counts: list[int] = []
            response_lengths: list[int] = []

            for _ in range(num_runs):
                result = self._inference_engine.infer(model_path, prompt)
                response = result.get("response", "")
                responses.append(response)
                token_count = result.get("tokenCount")
                token_counts.append(int(token_count) if isinstance(token_count, int) else 0)
                response_lengths.append(len(response))

            unique_responses = {}
            for response in responses:
                unique_responses[response] = unique_responses.get(response, 0) + 1
            max_count = max(unique_responses.values()) if unique_responses else 0
            consensus_fraction = max_count / max(num_runs, 1)

            token_arr = backend.array(token_counts if token_counts else [0])
            length_arr = backend.array(response_lengths if response_lengths else [0])
            mean_tokens = backend.mean(token_arr)
            var_tokens = backend.var(token_arr)
            mean_len = backend.mean(length_arr)
            var_len = backend.var(length_arr)
            backend.eval(mean_tokens, var_tokens, mean_len, var_len)

            per_prompt_results.append(
                {
                    "prompt": prompt,
                    "numRuns": num_runs,
                    "uniqueResponseCount": len(unique_responses),
                    "consensusFraction": float(consensus_fraction),
                    "meanTokenCount": float(backend.to_scalar(mean_tokens)),
                    "tokenCountVariance": float(backend.to_scalar(var_tokens)),
                    "meanResponseLength": float(backend.to_scalar(mean_len)),
                    "responseLengthVariance": float(backend.to_scalar(var_len)),
                }
            )

        consensus_arr = backend.array(
            [p["consensusFraction"] for p in per_prompt_results] or [0.0]
        )
        unique_arr = backend.array(
            [p["uniqueResponseCount"] for p in per_prompt_results] or [0.0]
        )
        mean_tokens_arr = backend.array(
            [p["meanTokenCount"] for p in per_prompt_results] or [0.0]
        )
        mean_lengths_arr = backend.array(
            [p["meanResponseLength"] for p in per_prompt_results] or [0.0]
        )
        backend.eval(consensus_arr, unique_arr, mean_tokens_arr, mean_lengths_arr)

        mean_consensus = backend.mean(consensus_arr)
        mean_unique = backend.mean(unique_arr)
        mean_tokens = backend.mean(mean_tokens_arr)
        mean_lengths = backend.mean(mean_lengths_arr)
        backend.eval(mean_consensus, mean_unique, mean_tokens, mean_lengths)

        metrics = {
            "promptCount": len(per_prompt_results),
            "numRuns": num_runs,
            "meanConsensusFraction": float(backend.to_scalar(mean_consensus)),
            "meanUniqueResponseCount": float(backend.to_scalar(mean_unique)),
            "meanTokenCount": float(backend.to_scalar(mean_tokens)),
            "meanResponseLength": float(backend.to_scalar(mean_lengths)),
        }

        suite["metrics"] = metrics
        suite["per_prompt_results"] = per_prompt_results

        suite["status"] = "completed"
        suite["completed_at"] = datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _load_prompts(prompt_variations: int) -> list[str]:
        data_root = Path(__file__).resolve().parents[4] / "data" / "eval_prompts"
        suite_path = data_root / "stuffed_model_tests.jsonl"
        if not suite_path.exists():
            raise ValueError("Stability prompts not found")

        prompts: list[str] = []
        for line in suite_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            prompt = payload.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                prompts.append(prompt.strip())
        if prompt_variations > 0:
            return prompts[:prompt_variations]
        return prompts

    @staticmethod
    def _derive_run_parameters(model_path: Path, backend: Backend) -> dict[str, Any]:
        """Derive stability run parameters from model geometry."""
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
