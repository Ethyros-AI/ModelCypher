"""Tests for EvaluationExecutionEngine."""

import pytest
from typing import List

from modelcypher.core.domain.evaluation.engine import (
    EvaluationConfig,
    EvaluationExecutionEngine,
    EvaluationScenario,
    MetricType,
    PromptResult,
    ScenarioResult,
)


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


@pytest.fixture
def configured_engine() -> EvaluationExecutionEngine:
    """Engine with custom thresholds."""
    config = EvaluationConfig(
        dataset_path="",
        metrics=[MetricType.ACCURACY],
        entropy_threshold=3.0,
        score_threshold=0.7,
    )
    return EvaluationExecutionEngine(config)


class TestEvaluationConfig:
    """Tests for EvaluationConfig."""

    def test_default_thresholds(self):
        """Default thresholds are sensible."""
        config = EvaluationConfig(dataset_path="", metrics=[])
        assert config.entropy_threshold == 5.0
        assert config.score_threshold == 0.5

    def test_custom_thresholds(self):
        """Custom thresholds are preserved."""
        config = EvaluationConfig(
            dataset_path="/data",
            metrics=[MetricType.LOSS, MetricType.PERPLEXITY],
            entropy_threshold=2.5,
            score_threshold=0.8,
        )
        assert config.dataset_path == "/data"
        assert config.entropy_threshold == 2.5
        assert config.score_threshold == 0.8


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
        assert result.avg_entropy == 2.0  # DEFAULT_ENTROPY
        assert result.avg_score == 1.0  # All outputs exist
        assert result.passed is True

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
        assert 0.0 < result.avg_score < 1.0  # Length-based score

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
        # "Hello"=5, "World"=5, "Test"=4 -> avg=(2.5+2.5+2.0)/3 = 2.33...
        assert result.avg_entropy > 0.0
        assert result.avg_entropy != 2.0  # Not default

    @pytest.mark.asyncio
    async def test_run_scenario_empty_output_scores_zero(self, engine, basic_scenario):
        """Empty outputs score 0.0 with default scoring."""
        def inference_fn(prompt: str) -> str:
            return ""  # Empty output

        result = await engine.run_scenario(
            scenario=basic_scenario,
            inference_fn=inference_fn,
        )

        assert result.avg_score == 0.0

    @pytest.mark.asyncio
    async def test_run_scenario_whitespace_output_scores_zero(self, engine, basic_scenario):
        """Whitespace-only outputs score 0.0 with default scoring."""
        def inference_fn(prompt: str) -> str:
            return "   \n\t  "  # Whitespace only

        result = await engine.run_scenario(
            scenario=basic_scenario,
            inference_fn=inference_fn,
        )

        assert result.avg_score == 0.0

    @pytest.mark.asyncio
    async def test_run_scenario_pass_fail_logic(self, configured_engine, basic_scenario):
        """Pass/fail uses configured thresholds."""
        # High entropy, low score -> should fail
        def inference_fn(prompt: str) -> str:
            return ""

        def entropy_fn(prompt: str) -> float:
            return 10.0  # High entropy

        result = await configured_engine.run_scenario(
            scenario=basic_scenario,
            inference_fn=inference_fn,
            entropy_fn=entropy_fn,
        )

        assert result.passed is False
        assert result.details["entropy_threshold"] == 3.0
        assert result.details["score_threshold"] == 0.7

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

    @pytest.mark.asyncio
    async def test_run_scenario_entropy_fn_exception_handled(self, engine, basic_scenario):
        """Entropy function exceptions fall back to default."""
        def inference_fn(prompt: str) -> str:
            return "output"

        def entropy_fn(prompt: str) -> float:
            raise ValueError("Entropy calculation failed")

        result = await engine.run_scenario(
            scenario=basic_scenario,
            inference_fn=inference_fn,
            entropy_fn=entropy_fn,
        )

        # Should fall back to default entropy
        assert result.avg_entropy == 2.0

    @pytest.mark.asyncio
    async def test_run_scenario_scoring_fn_exception_handled(self, engine, basic_scenario):
        """Scoring function exceptions score 0.0."""
        def inference_fn(prompt: str) -> str:
            return "output"

        def scoring_fn(output: str, concepts: List[str]) -> float:
            raise ValueError("Scoring failed")

        result = await engine.run_scenario(
            scenario=basic_scenario,
            inference_fn=inference_fn,
            scoring_fn=scoring_fn,
        )

        # Should score 0.0 on error
        assert result.avg_score == 0.0

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

        assert result.avg_entropy == 0.0
        assert result.avg_score == 0.0
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
