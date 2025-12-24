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

"""Agent evaluation service for assessing agent performance.

Provides agent evaluation execution and results retrieval functionality
for measuring agent capabilities on structured tasks.
Also provides action scoring and semantic drift assessment.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from modelcypher.core.domain.agents.agent_eval_suite_engine import (
    AgentAction,
    AgentActionKind,
    AgentEvalCase,
    AgentEvalCaseCategory,
    AgentEvalRisk,
    AgentEvalScoringEngine,
    EvalCaseConstraints,
    Expected,
    ExpectedOption,
)
from modelcypher.core.domain.agents.semantic_prime_drift import (
    DriftVerdict,
    SemanticPrimeDriftConfig,
    SemanticPrimeDriftDetector,
)
from modelcypher.core.domain.agents.semantic_prime_atlas import SemanticPrimeAtlas

logger = logging.getLogger(__name__)


@dataclass
class AgentEvalConfig:
    """Configuration for agent evaluation."""

    model_path: str
    eval_suite: str = "default"
    max_turns: int = 10
    timeout_seconds: int = 300
    tools_enabled: bool = True
    seed: int | None = None


@dataclass
class AgentEvalRunResult:
    """Result of an agent evaluation run."""

    eval_id: str
    model_path: str
    eval_suite: str
    status: str
    started_at: str
    config: dict[str, Any]
    summary: dict[str, float] = field(default_factory=dict)


@dataclass
class AgentEvalResults:
    """Detailed agent evaluation results."""

    eval_id: str
    model_path: str
    eval_suite: str
    status: str
    started_at: str
    completed_at: str | None
    config: dict[str, Any]
    metrics: dict[str, float]
    task_results: list[dict[str, Any]]
    interpretation: str
    overall_score: float


class AgentEvalService:
    """Service for agent evaluation.

    Evaluates agent performance on structured tasks including:
    - Tool use accuracy
    - Task completion rate
    - Response quality
    - Multi-turn coherence
    """

    def __init__(self) -> None:
        """Initialize agent eval service."""
        self._evaluations: dict[str, dict[str, Any]] = {}

    def run(self, config: AgentEvalConfig) -> AgentEvalRunResult:
        """Execute agent evaluation.

        Args:
            config: Agent evaluation configuration

        Returns:
            AgentEvalRunResult with eval_id and initial status

        Raises:
            ValueError: If model path is invalid
        """
        model_path = Path(config.model_path).expanduser().resolve()

        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        if not model_path.is_dir():
            raise ValueError(f"Model path is not a directory: {model_path}")

        eval_id = f"aeval-{uuid.uuid4().hex[:12]}"
        started_at = datetime.now(timezone.utc).isoformat()

        config_dict = {
            "model_path": str(model_path),
            "eval_suite": config.eval_suite,
            "max_turns": config.max_turns,
            "timeout_seconds": config.timeout_seconds,
            "tools_enabled": config.tools_enabled,
            "seed": config.seed,
        }

        # Store evaluation state
        self._evaluations[eval_id] = {
            "model_path": str(model_path),
            "eval_suite": config.eval_suite,
            "status": "running",
            "started_at": started_at,
            "completed_at": None,
            "config": config_dict,
            "metrics": {},
            "task_results": [],
        }

        logger.info(
            "Started agent evaluation %s for model %s with suite %s",
            eval_id,
            model_path,
            config.eval_suite,
        )

        # Run evaluation
        self._run_evaluation(eval_id, config)

        return AgentEvalRunResult(
            eval_id=eval_id,
            model_path=str(model_path),
            eval_suite=config.eval_suite,
            status=self._evaluations[eval_id]["status"],
            started_at=started_at,
            config=config_dict,
            summary=self._evaluations[eval_id]["metrics"],
        )

    def results(self, eval_id: str) -> AgentEvalResults:
        """Get detailed results for an agent evaluation.

        Args:
            eval_id: ID of the evaluation

        Returns:
            AgentEvalResults with detailed metrics and task results

        Raises:
            ValueError: If eval_id is not found
        """
        if eval_id not in self._evaluations:
            raise ValueError(f"Agent evaluation not found: {eval_id}")

        evaluation = self._evaluations[eval_id]
        metrics = evaluation["metrics"]

        # Calculate overall score
        overall_score = (
            metrics.get("task_completion_rate", 0.0) * 0.4
            + metrics.get("tool_accuracy", 0.0) * 0.3
            + metrics.get("response_quality", 0.0) * 0.3
        )

        # Generate interpretation
        if overall_score >= 0.9:
            interpretation = "Agent demonstrates excellent performance across all metrics."
        elif overall_score >= 0.7:
            interpretation = "Agent shows good performance with room for improvement."
        elif overall_score >= 0.5:
            interpretation = "Agent shows moderate performance. Consider additional training."
        else:
            interpretation = "Agent shows poor performance. Significant improvements needed."

        return AgentEvalResults(
            eval_id=eval_id,
            model_path=evaluation["model_path"],
            eval_suite=evaluation["eval_suite"],
            status=evaluation["status"],
            started_at=evaluation["started_at"],
            completed_at=evaluation["completed_at"],
            config=evaluation["config"],
            metrics=metrics,
            task_results=evaluation["task_results"],
            interpretation=interpretation,
            overall_score=overall_score,
        )

    def _run_evaluation(
        self,
        eval_id: str,
        config: AgentEvalConfig,
    ) -> None:
        """Run agent evaluation (simulated).

        In production, this would:
        1. Load the model
        2. Run through evaluation tasks
        3. Measure tool use, completion, quality
        4. Compute aggregate metrics
        """
        evaluation = self._evaluations[eval_id]

        # Simulated metrics
        evaluation["metrics"] = {
            "task_completion_rate": 0.82,
            "tool_accuracy": 0.88,
            "response_quality": 0.79,
            "multi_turn_coherence": 0.85,
            "average_turns": 4.2,
            "timeout_rate": 0.05,
            "tasks_evaluated": 20,
        }

        # Simulated task results
        evaluation["task_results"] = [
            {
                "task_id": f"task-{i}",
                "task_type": ["tool_use", "reasoning", "coding"][i % 3],
                "completed": i % 5 != 0,
                "turns": 3 + (i % 4),
                "score": 0.7 + (i % 3) * 0.1,
                "tool_calls": i % 3,
                "correct_tool_calls": i % 3 if i % 5 != 0 else 0,
            }
            for i in range(20)
        ]

        evaluation["status"] = "completed"
        evaluation["completed_at"] = datetime.now(timezone.utc).isoformat()

    def score_action(
        self,
        output: str,
        eval_case_id: str = "adhoc",
        prompt: str = "",
        expected_kinds: list[str] | None = None,
        expected_tools: list[str] | None = None,
        expected_text_patterns: list[str] | None = None,
        constraints_max_turns: int = 10,
        constraints_require_tool: bool = False,
        constraints_allow_delegation: bool = True,
        category: str = "functional",
        risk: str = "low",
    ) -> dict[str, Any]:
        """Score an agent output for action quality.

        Args:
            output: The agent's output text to score
            eval_case_id: Identifier for this evaluation case
            prompt: The prompt that generated the output
            expected_kinds: List of allowed action kinds (text, tool_call, delegation)
            expected_tools: List of expected tool names if tool_call is expected
            expected_text_patterns: List of regex patterns to match in text output
            constraints_max_turns: Maximum turns allowed
            constraints_require_tool: Whether a tool call is required
            constraints_allow_delegation: Whether delegation is allowed
            category: Evaluation category (functional, safety, robustness, efficiency)
            risk: Risk level (low, medium, high, critical)

        Returns:
            Dict with scoring results including parsed actions, scores, and assessment
        """
        # Map string category to enum
        category_map = {
            "functional": AgentEvalCaseCategory.functional,
            "safety": AgentEvalCaseCategory.safety,
            "robustness": AgentEvalCaseCategory.robustness,
            "efficiency": AgentEvalCaseCategory.efficiency,
        }
        risk_map = {
            "low": AgentEvalRisk.low,
            "medium": AgentEvalRisk.medium,
            "high": AgentEvalRisk.high,
            "critical": AgentEvalRisk.critical,
        }

        # Build expected structure
        expected_options: list[ExpectedOption] = []
        if expected_kinds:
            for kind_str in expected_kinds:
                kind = AgentActionKind[kind_str] if kind_str in AgentActionKind.__members__ else AgentActionKind.text
                expected_options.append(ExpectedOption(
                    action_kind=kind,
                    tool_name=expected_tools[0] if expected_tools else None,
                    text_pattern=expected_text_patterns[0] if expected_text_patterns else None,
                ))
        else:
            # Default: allow text output
            expected_options.append(ExpectedOption(action_kind=AgentActionKind.text))

        expected = Expected(options=expected_options)
        constraints = EvalCaseConstraints(
            max_turns=constraints_max_turns,
            require_tool=constraints_require_tool,
            allow_delegation=constraints_allow_delegation,
        )

        eval_case = AgentEvalCase(
            case_id=eval_case_id,
            prompt=prompt,
            expected=expected,
            constraints=constraints,
            category=category_map.get(category, AgentEvalCaseCategory.functional),
            risk=risk_map.get(risk, AgentEvalRisk.low),
        )

        # Determine allowed action kinds and tools
        allowed_kinds = set(expected_kinds) if expected_kinds else {AgentActionKind.text.name}
        allowed_tools = set(expected_tools) if expected_tools else set()

        # Score the output
        scored = AgentEvalScoringEngine.score(
            eval_case=eval_case,
            output=output,
            allowed_action_kinds=allowed_kinds,
            allowed_tools=allowed_tools,
        )

        return {
            "case_id": eval_case_id,
            "parsed_action": {
                "kind": scored.parsed_action.kind.name,
                "tool_call": {
                    "name": scored.parsed_action.tool_call.name,
                    "arguments": scored.parsed_action.tool_call.arguments,
                } if scored.parsed_action.tool_call else None,
                "text": scored.parsed_action.text,
            },
            "expectation_matched": scored.expectation_matched,
            "constraint_violations": scored.constraint_violations,
            "is_overrefusal": scored.is_overrefusal,
            "is_unsafe_completion": scored.is_unsafe_completion,
            "scores": {
                "functional": scored.scores.functional,
                "safety": scored.scores.safety,
                "constraint": scored.scores.constraint,
            },
        }

    def assess_drift(
        self,
        baseline_text: str,
        observed_text: str,
        threshold: float = 0.65,
    ) -> dict[str, Any]:
        """Assess semantic drift between baseline and observed text.

        Uses semantic prime decomposition to measure how much an agent's
        response has drifted from expected baseline behavior.

        Args:
            baseline_text: The expected/baseline text
            observed_text: The observed/actual text to compare
            threshold: Similarity threshold below which drift is flagged

        Returns:
            Dict with drift assessment including similarity, verdict, and details
        """
        config = SemanticPrimeDriftConfig(
            similarity_threshold=threshold,
            alert_on_major_drift=True,
        )
        detector = SemanticPrimeDriftDetector(config=config)
        atlas = SemanticPrimeAtlas()

        # Decompose both texts into semantic primes
        baseline_primes = atlas.decompose(baseline_text)
        observed_primes = atlas.decompose(observed_text)

        # Calculate similarity using the detector
        result = detector.assess(
            baseline_primes=baseline_primes,
            observed_primes=observed_primes,
        )

        return {
            "similarity": result.similarity,
            "verdict": result.verdict.name,
            "is_drifted": result.verdict != DriftVerdict.stable,
            "baseline_primes": baseline_primes,
            "observed_primes": observed_primes,
            "threshold": threshold,
            "delta": abs(1.0 - result.similarity),
        }
