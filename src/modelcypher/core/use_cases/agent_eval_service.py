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
    AgentActionKind,
    AgentEvalCase,
    AgentEvalCaseCategory,
    AgentEvalCaseProfile,
    AgentEvalScoringEngine,
    EvalCaseConstraints,
    Expected,
    ExpectedOption,
    ExpectedToolSpec,
)
# Semantic drift detection removed - probes now loaded from JSON

logger = logging.getLogger(__name__)


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

    def run(
        self,
        model_path: str,
        eval_suite: str = "default",
        max_turns: int = 10,
        timeout_seconds: int = 300,
        tools_enabled: bool = True,
        seed: int | None = None,
    ) -> AgentEvalRunResult:
        """Execute agent evaluation.

        Args:
            model_path: Path to model directory.
            eval_suite: Evaluation suite name.
            max_turns: Maximum conversation turns.
            timeout_seconds: Timeout in seconds.
            tools_enabled: Whether tool use is enabled.
            seed: Optional random seed.

        Returns:
            AgentEvalRunResult with eval_id and initial status

        Raises:
            ValueError: If model path is invalid
        """
        model_path = Path(model_path).expanduser().resolve()

        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        if not model_path.is_dir():
            raise ValueError(f"Model path is not a directory: {model_path}")

        eval_id = f"aeval-{uuid.uuid4().hex[:12]}"
        started_at = datetime.now(timezone.utc).isoformat()

        config_dict = {
            "model_path": str(model_path),
            "eval_suite": eval_suite,
            "max_turns": max_turns,
            "timeout_seconds": timeout_seconds,
            "tools_enabled": tools_enabled,
            "seed": seed,
        }

        # Store evaluation state
        self._evaluations[eval_id] = {
            "model_path": str(model_path),
            "eval_suite": eval_suite,
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
            eval_suite,
        )

        # Run evaluation
        self._run_evaluation(eval_id)

        return AgentEvalRunResult(
            eval_id=eval_id,
            model_path=str(model_path),
            eval_suite=eval_suite,
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
            overall_score=overall_score,
        )

    def _run_evaluation(
        self,
        eval_id: str,
    ) -> None:
        """Run agent evaluation.

        Real evaluation results must come from measured task runs.
        This stub records empty metrics until wired to an evaluator.
        """
        evaluation = self._evaluations[eval_id]

        evaluation["metrics"] = {}
        evaluation["task_results"] = []

        evaluation["status"] = "completed"
        evaluation["completed_at"] = datetime.now(timezone.utc).isoformat()

    def score_action(
        self,
        output: str,
        eval_case_id: str = "adhoc",
        prompt: str = "",
        expected_kinds: list[str] | None = None,
        expected_tools: list[str] | None = None,
        max_steps: int | None = None,
        category: str = "tool_call",
        profile: str = "open",
    ) -> dict[str, Any]:
        """Score an agent output for action quality.

        Args:
            output: The agent's output text to score
            eval_case_id: Identifier for this evaluation case
            prompt: The prompt that generated the output
            expected_kinds: List of expected action kinds (by value, e.g. "tool_call")
            expected_tools: List of expected tool names if tool_call is expected
            max_steps: Optional maximum step constraint
            category: Evaluation category (tool_call, constraint, regression, routing, other)
            profile: Case profile (open, restricted, ambiguous)

        Returns:
            Dict with scoring results including parsed actions and raw scores
        """
        category_map = {
            "tool_call": AgentEvalCaseCategory.TOOL_CALL,
            "constraint": AgentEvalCaseCategory.CONSTRAINT,
            "regression": AgentEvalCaseCategory.REGRESSION,
            "routing": AgentEvalCaseCategory.ROUTING,
            "other": AgentEvalCaseCategory.OTHER,
        }
        profile_map = {
            "open": AgentEvalCaseProfile.OPEN,
            "restricted": AgentEvalCaseProfile.RESTRICTED,
            "ambiguous": AgentEvalCaseProfile.AMBIGUOUS,
        }

        def parse_kind(kind_value: str) -> AgentActionKind:
            for kind in AgentActionKind:
                if kind.value == kind_value:
                    return kind
            return AgentActionKind.RESPOND

        expected_options: list[ExpectedOption] = []
        if expected_kinds:
            for kind_value in expected_kinds:
                kind = parse_kind(kind_value)
                if kind == AgentActionKind.TOOL_CALL and expected_tools:
                    for tool_name in expected_tools:
                        expected_options.append(
                            ExpectedOption(
                                kind=kind,
                                tool=ExpectedToolSpec(name=tool_name),
                            )
                        )
                else:
                    expected_options.append(ExpectedOption(kind=kind))
        else:
            expected_options.append(ExpectedOption(kind=AgentActionKind.RESPOND))

        expected = Expected(any_of=tuple(expected_options))
        constraints = EvalCaseConstraints(
            allowed_action_kinds=tuple({parse_kind(k) for k in expected_kinds})
            if expected_kinds
            else None,
            allowed_tools=tuple(expected_tools) if expected_tools else None,
            max_steps=max_steps,
        )

        messages = ({"role": "user", "content": prompt},) if prompt else ()
        eval_case = AgentEvalCase(
            case_id=eval_case_id,
            category=category_map.get(category, AgentEvalCaseCategory.OTHER),
            profile=profile_map.get(profile, AgentEvalCaseProfile.AMBIGUOUS),
            tags=(),
            messages=messages,
            constraints=constraints,
            expected=expected,
        )

        scored = AgentEvalScoringEngine.score(
            eval_case=eval_case,
            output=output,
            allowed_action_kinds=constraints.allowed_action_kinds,
            allowed_tools=constraints.allowed_tools,
        )

        action_payload = None
        if scored.action:
            action_payload = {"kind": scored.action.kind.value}
            if scored.action.tool:
                action_payload["tool"] = {
                    "name": scored.action.tool.name,
                    "arguments": scored.action.tool.arguments,
                }

        return {
            "case_id": eval_case_id,
            "action": action_payload,
            "scores": scored.scores,
            "error_taxonomy": list(scored.error_taxonomy),
        }

    def assess_drift(
        self,
        baseline_text: str,
        observed_text: str,
        threshold: float = 0.65,
    ) -> dict[str, Any]:
        """Assess semantic drift between baseline and observed text.

        Note: Semantic prime drift detection has been removed.
        Probes are now loaded from JSON. Use probe-based comparison instead.

        Args:
            baseline_text: The expected/baseline text
            observed_text: The observed/actual text to compare
            threshold: Similarity threshold for comparison (returned as reference)

        Returns:
            Dict indicating feature is deprecated
        """
        return {
            "cosine_similarity": None,
            "threshold": threshold,
            "note": "semantic_drift_deprecated",
        }
