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

"""Tests for EvaluationExecutionEngine."""

from typing import List

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.evaluation.engine import (
    EvaluationConfig,
    EvaluationExecutionEngine,
    EvaluationScenario,
    MetricType,
    PromptResult,
    ScenarioResult,
)
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon


def _eps(*values: float) -> float:
    backend = get_default_backend()
    return division_epsilon(backend, backend.array(list(values) or [1.0]))


@pytest.fixture
def basic_scenario() -> EvaluationScenario:
    """A basic test scenario."""
    return EvaluationScenario(
        name="test_scenario",
        description="A test scenario",
        prompts=["Hello", "World", "Test"],
        target_concepts=["greeting", "response"],
    )


@pytest.fixture
def engine() -> EvaluationExecutionEngine:
    """Default evaluation engine."""
    return EvaluationExecutionEngine()


class TestEvaluationConfig:
    """Tests for EvaluationConfig."""

    def test_default_values(self):
        """Default configuration values are preserved."""
        config = EvaluationConfig(dataset_path="", metrics=[])
        assert config.batch_size == 1
        assert config.max_samples is None

    def test_custom_values(self):
        """Custom configuration values are preserved."""
        config = EvaluationConfig(
            dataset_path="/data",
            metrics=[MetricType.LOSS, MetricType.PERPLEXITY],
            batch_size=4,
            max_samples=100,
        )
        assert config.dataset_path == "/data"
        assert config.batch_size == 4
        assert config.max_samples == 100


class TestEvaluationExecutionEngine:
    """Tests for EvaluationExecutionEngine."""

    @pytest.mark.asyncio
    async def test_run_scenario_basic(self, engine, basic_scenario):
        """Basic scenario execution with default callbacks."""

        def inference_fn(prompt: str) -> str:
            return f"Response to: {prompt}"

        result = await engine.run_scenario(
            scenario=basic_scenario,
            inference_fn=inference_fn,
        )

        assert isinstance(result, ScenarioResult)
        assert result.scenario_name == "test_scenario"
        assert result.avg_entropy is None
        assert result.avg_score is None
        assert result.details["used_real_entropy"] is False
        assert result.details["used_custom_scoring"] is False
        assert result.details["entropy_sample_count"] == 0
        assert result.details["score_sample_count"] == 0
        for prompt_result in result.prompt_results:
            assert prompt_result.entropy is None
            assert prompt_result.score is None

    @pytest.mark.asyncio
    async def test_run_scenario_with_scoring_fn(self, engine, basic_scenario):
        """Scenario with custom scoring function."""

        def inference_fn(prompt: str) -> str:
            return f"Response: {prompt}"

        def scoring_fn(output: str, concepts: List[str]) -> float:
            # Score based on output length
            return min(1.0, len(output) / 50.0)

        result = await engine.run_scenario(
            scenario=basic_scenario,
            inference_fn=inference_fn,
            scoring_fn=scoring_fn,
        )

        assert result.details["used_custom_scoring"] is True
        expected_scores = [
            scoring_fn(inference_fn(prompt), basic_scenario.target_concepts)
            for prompt in basic_scenario.prompts
        ]
        expected_avg = sum(expected_scores) / len(expected_scores)
        assert result.avg_score is not None
        assert abs(result.avg_score - expected_avg) <= _eps(result.avg_score, expected_avg)
        assert result.avg_entropy is None
        assert result.details["score_sample_count"] == len(expected_scores)
        assert result.details["entropy_sample_count"] == 0

    @pytest.mark.asyncio
    async def test_run_scenario_with_entropy_fn(self, engine, basic_scenario):
        """Scenario with custom entropy function."""

        def inference_fn(prompt: str) -> str:
            return f"Response: {prompt}"

        def entropy_fn(prompt: str) -> float:
            # Return different entropy based on prompt length
            return len(prompt) * 0.5

        result = await engine.run_scenario(
            scenario=basic_scenario,
            inference_fn=inference_fn,
            entropy_fn=entropy_fn,
        )

        assert result.details["used_real_entropy"] is True
        expected_entropies = [entropy_fn(prompt) for prompt in basic_scenario.prompts]
        expected_avg = sum(expected_entropies) / len(expected_entropies)
        assert result.avg_entropy is not None
        assert abs(result.avg_entropy - expected_avg) <= _eps(
            result.avg_entropy, expected_avg
        )
        assert result.avg_score is None
        assert result.details["entropy_sample_count"] == len(expected_entropies)
        assert result.details["score_sample_count"] == 0

    @pytest.mark.asyncio
    async def test_run_scenario_empty_output_scores_zero(self, engine, basic_scenario):
        """Empty outputs with no scoring_fn leave scores unset."""

        def inference_fn(prompt: str) -> str:
            return ""  # Empty output

        result = await engine.run_scenario(
            scenario=basic_scenario,
            inference_fn=inference_fn,
        )

        assert result.avg_score is None

    @pytest.mark.asyncio
    async def test_run_scenario_whitespace_output_scores_zero(self, engine, basic_scenario):
        """Whitespace-only outputs with no scoring_fn leave scores unset."""

        def inference_fn(prompt: str) -> str:
            return "   \n\t  "  # Whitespace only

        result = await engine.run_scenario(
            scenario=basic_scenario,
            inference_fn=inference_fn,
        )

        assert result.avg_score is None

    @pytest.mark.asyncio
    async def test_run_scenario_prompt_results_populated(self, engine, basic_scenario):
        """Per-prompt results are captured."""

        def inference_fn(prompt: str) -> str:
            return f"Output for {prompt}"

        result = await engine.run_scenario(
            scenario=basic_scenario,
            inference_fn=inference_fn,
        )

        assert len(result.prompt_results) == 3
        assert all(isinstance(pr, PromptResult) for pr in result.prompt_results)
        assert result.prompt_results[0].prompt == "Hello"
        assert result.prompt_results[0].output == "Output for Hello"
        assert result.prompt_results[0].entropy is None
        assert result.prompt_results[0].score is None

    @pytest.mark.asyncio
    async def test_run_scenario_entropy_fn_exception_handled(self, engine, basic_scenario):
        """Entropy function exceptions leave entropy unset."""

        def inference_fn(prompt: str) -> str:
            return "output"

        def entropy_fn(prompt: str) -> float:
            raise ValueError("Entropy calculation failed")

        result = await engine.run_scenario(
            scenario=basic_scenario,
            inference_fn=inference_fn,
            entropy_fn=entropy_fn,
        )

        assert result.avg_entropy is None
        assert result.details["entropy_sample_count"] == 0

    @pytest.mark.asyncio
    async def test_run_scenario_scoring_fn_exception_handled(self, engine, basic_scenario):
        """Scoring function exceptions leave score unset."""

        def inference_fn(prompt: str) -> str:
            return "output"

        def scoring_fn(output: str, concepts: List[str]) -> float:
            raise ValueError("Scoring failed")

        result = await engine.run_scenario(
            scenario=basic_scenario,
            inference_fn=inference_fn,
            scoring_fn=scoring_fn,
        )

        assert result.avg_score is None
        assert result.details["score_sample_count"] == 0

    @pytest.mark.asyncio
    async def test_run_scenario_empty_prompts(self, engine):
        """Empty prompt list handles gracefully."""
        scenario = EvaluationScenario(
            name="empty",
            description="Empty scenario",
            prompts=[],
            target_concepts=[],
        )

        def inference_fn(prompt: str) -> str:
            return "output"

        result = await engine.run_scenario(
            scenario=scenario,
            inference_fn=inference_fn,
        )

        assert result.avg_entropy is None
        assert result.avg_score is None
        assert result.prompt_results == []

    @pytest.mark.asyncio
    async def test_run_scenarios_multiple(self, engine):
        """Run multiple scenarios sequentially."""
        scenario1 = EvaluationScenario(
            name="scenario1",
            description="First",
            prompts=["A", "B"],
            target_concepts=[],
        )
        scenario2 = EvaluationScenario(
            name="scenario2",
            description="Second",
            prompts=["C", "D", "E"],
            target_concepts=[],
        )

        def inference_fn(prompt: str) -> str:
            return f"Response: {prompt}"

        results = await engine.run_scenarios(
            scenarios=[scenario1, scenario2],
            inference_fn=inference_fn,
        )

        assert len(results) == 2
        assert results[0].scenario_name == "scenario1"
        assert results[1].scenario_name == "scenario2"
        assert len(results[0].prompt_results) == 2
        assert len(results[1].prompt_results) == 3


class TestMetricType:
    """Tests for MetricType enum."""

    def test_metric_types_exist(self):
        """All expected metric types exist."""
        assert MetricType.LOSS == "loss"
        assert MetricType.PERPLEXITY == "perplexity"
        assert MetricType.ACCURACY == "accuracy"
