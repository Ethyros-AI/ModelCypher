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

"""Evaluation execution engine for semantic evaluation scenarios.

Orchestrates scenario-based model evaluation with support for:
- Real entropy calculation from logits (when provided)
- Custom semantic scoring functions
- Concept activation evaluation
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from ..entropy.entropy_math import EntropyMath

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    LOSS = "loss"
    PERPLEXITY = "perplexity"
    ACCURACY = "accuracy"


@dataclass
class EvaluationConfig:
    dataset_path: str
    metrics: list[MetricType]
    batch_size: int = 1
    max_samples: int | None = None


@dataclass
class EvaluationScenario:
    name: str
    description: str
    prompts: list[str]
    target_concepts: list[str]  # Concepts expected to activate


@dataclass
class PromptResult:
    """Result for a single prompt evaluation."""

    prompt: str
    output: str
    entropy: float | None
    score: float | None


@dataclass
class ScenarioResult:
    scenario_name: str
    avg_entropy: float | None
    avg_score: float | None
    details: dict[str, Any]
    prompt_results: list[PromptResult] = field(default_factory=list)


# Type aliases for callback functions
InferenceFn = Callable[[str], str]
ScoringFn = Callable[[str, list[str]], float]
EntropyFn = Callable[[str], float]


class EvaluationExecutionEngine:
    """
    Orchestrates semantic evaluation scenarios.

    Uses EntropyMath to aggregate entropy statistics
    and supports pluggable inference, scoring, and entropy callbacks.

    Example:
        ```python
        engine = EvaluationExecutionEngine()

        # With real entropy from logits
        result = await engine.run_scenario(
            scenario=scenario,
            inference_fn=model.generate,
            scoring_fn=semantic_scorer,
            entropy_fn=lambda p: model.get_last_entropy(p),
        )
        ```
    """

    def __init__(self, config: EvaluationConfig | None = None):
        self.config = config or EvaluationConfig(dataset_path="", metrics=[])

    async def run_scenario(
        self,
        scenario: EvaluationScenario,
        inference_fn: InferenceFn,
        scoring_fn: ScoringFn | None = None,
        entropy_fn: EntropyFn | None = None,
    ) -> ScenarioResult:
        """
        Executes an evaluation scenario.

        Args:
            scenario: The evaluation scenario with prompts and target concepts.
            inference_fn: Function that generates model output from a prompt.
            scoring_fn: Optional function that scores output against target concepts.
                        If None, uses a simple heuristic (output exists = 1.0).
            entropy_fn: Optional function that returns entropy for a prompt.
                        If None, uses DEFAULT_ENTROPY for backward compatibility.

        Returns:
            ScenarioResult with aggregate metrics and per-prompt details.
        """
        logger.info(f"Running Scenario: {scenario.name}")

        entropies: list[float] = []
        scores: list[float] = []
        prompt_results: list[PromptResult] = []

        for prompt in scenario.prompts:
            # 1. Inference
            output = inference_fn(prompt)

            # 2. Entropy calculation
            if entropy_fn is not None:
                try:
                    entropy = entropy_fn(prompt)
                except Exception as e:
                    logger.warning(f"Entropy calculation failed for prompt: {e}")
                    entropy = None
            else:
                entropy = None

            if entropy is not None:
                entropies.append(entropy)

            # 3. Semantic scoring
            if scoring_fn is not None:
                try:
                    score = scoring_fn(output, scenario.target_concepts)
                except Exception as e:
                    logger.warning(f"Scoring failed for output: {e}")
                    score = None
            else:
                score = None

            if score is not None:
                scores.append(score)
            prompt_results.append(
                PromptResult(
                    prompt=prompt,
                    output=output,
                    entropy=entropy,
                    score=score,
                )
            )

        # Aggregate statistics using EntropyMath
        if entropies:
            stats = EntropyMath.calculate_trajectory_stats(entropies)
            avg_entropy = stats.mean_entropy
        else:
            avg_entropy = None

        avg_score = sum(scores) / len(scores) if scores else None

        return ScenarioResult(
            scenario_name=scenario.name,
            avg_entropy=avg_entropy,
            avg_score=avg_score,
            details={
                "prompts_count": len(scenario.prompts),
                "used_real_entropy": entropy_fn is not None,
                "used_custom_scoring": scoring_fn is not None,
                "entropy_sample_count": len(entropies),
                "score_sample_count": len(scores),
            },
            prompt_results=prompt_results,
        )

    async def run_scenarios(
        self,
        scenarios: list[EvaluationScenario],
        inference_fn: InferenceFn,
        scoring_fn: ScoringFn | None = None,
        entropy_fn: EntropyFn | None = None,
    ) -> list[ScenarioResult]:
        """
        Run multiple scenarios sequentially.

        Args:
            scenarios: List of scenarios to evaluate.
            inference_fn: Function that generates model output.
            scoring_fn: Optional scoring function.
            entropy_fn: Optional entropy function.

        Returns:
            List of ScenarioResults, one per scenario.
        """
        results = []
        for scenario in scenarios:
            result = await self.run_scenario(
                scenario=scenario,
                inference_fn=inference_fn,
                scoring_fn=scoring_fn,
                entropy_fn=entropy_fn,
            )
            results.append(result)
        return results
