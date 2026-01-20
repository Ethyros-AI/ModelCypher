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

import json
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from modelcypher.ports.inference import HiddenStateEngine
from modelcypher.core.domain.agents.agent_eval_suite_engine import (
    AgentActionKind,
    AgentEvalCase,
    AgentEvalCaseCategory,
    AgentEvalCaseProfile,
    AgentEvalScoringEngine,
    CaseResult,
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

    def __init__(self, inference_engine: HiddenStateEngine | None = None) -> None:
        """Initialize agent eval service."""
        self._evaluations: dict[str, dict[str, Any]] = {}
        self._inference_engine = inference_engine

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
        overall_score = metrics.get(
            "actionAllowedRate", metrics.get("schemaValidRate", 0.0)
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

        Executes the evaluation suite using the inference engine and scores outputs.
        """
        evaluation = self._evaluations[eval_id]
        if self._inference_engine is None:
            raise RuntimeError("Inference engine required for agent evaluation")

        model_path = evaluation["model_path"]
        eval_suite = evaluation["eval_suite"]
        config = evaluation["config"]
        max_turns = config.get("max_turns")
        timeout_seconds = config.get("timeout_seconds")
        tools_enabled = config.get("tools_enabled", True)
        seed = config.get("seed")

        cases = self._load_eval_cases(eval_suite)
        if seed is not None:
            random.Random(seed).shuffle(cases)
        if isinstance(max_turns, int) and max_turns > 0:
            cases = cases[:max_turns]

        start_time = time.time()
        results: list[CaseResult] = []

        for case in cases:
            if timeout_seconds and time.time() - start_time > timeout_seconds:
                break

            prompt = self._render_messages(case.messages)
            case_start = time.time()
            try:
                response = self._inference_engine.infer(model_path, prompt)
                output = response.get("response", "")
                allowed_kinds = self._allowed_action_kinds(case, tools_enabled)
                allowed_tools = () if not tools_enabled else None

                scored = AgentEvalScoringEngine.score(
                    eval_case=case,
                    output=output,
                    allowed_action_kinds=allowed_kinds,
                    allowed_tools=allowed_tools,
                )
                duration = response.get("totalDuration")
                latency_ms = int((duration if duration is not None else time.time() - case_start) * 1000)
                tokens_generated = response.get("tokenCount")

                results.append(
                    CaseResult(
                        case_id=case.case_id,
                        category=case.category,
                        profile=case.profile,
                        tags=case.tags,
                        action=scored.action,
                        scores=scored.scores,
                        error_taxonomy=scored.error_taxonomy,
                        latency_ms=latency_ms,
                        tokens_generated=tokens_generated if isinstance(tokens_generated, int) else None,
                    )
                )
            except Exception as exc:
                logger.warning("Agent eval case %s failed: %s", case.case_id, exc)
                results.append(
                    CaseResult(
                        case_id=case.case_id,
                        category=case.category,
                        profile=case.profile,
                        tags=case.tags,
                        scores={
                            "parseable_action": 0.0,
                            "schema_valid": 0.0,
                            "action_allowed": 0.0,
                            "expected_kind": 0.0,
                        },
                        error_taxonomy=("inference_error",),
                    )
                )

        aggregate, _ = AgentEvalScoringEngine.aggregate(results)
        evaluation["metrics"] = aggregate.to_dict()
        evaluation["task_results"] = [result.to_dict() for result in results]

        evaluation["status"] = "completed"
        evaluation["completed_at"] = datetime.now(timezone.utc).isoformat()

    def _load_eval_cases(self, eval_suite: str) -> list[AgentEvalCase]:
        suite_path = Path(eval_suite).expanduser()
        if suite_path.exists():
            return self._load_cases_from_path(suite_path)

        data_root = Path(__file__).resolve().parents[4] / "data" / "eval_prompts"
        default_map = {"default": "stuffed_model_tests.jsonl"}
        candidate = default_map.get(eval_suite, f"{eval_suite}.jsonl")
        path = data_root / candidate
        if not path.exists():
            alt = data_root / f"{eval_suite}.json"
            if alt.exists():
                path = alt
            else:
                raise ValueError(f"Evaluation suite not found: {eval_suite}")
        return self._load_cases_from_path(path)

    def _load_cases_from_path(self, path: Path) -> list[AgentEvalCase]:
        if path.suffix.lower() == ".jsonl":
            items: list[dict[str, Any]] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
            return self._parse_case_items(items)

        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            items = payload.get("cases") or payload.get("tests") or []
        elif isinstance(payload, list):
            items = payload
        else:
            items = []
        return self._parse_case_items(items)

    def _parse_case_items(self, items: list[dict[str, Any]]) -> list[AgentEvalCase]:
        cases: list[AgentEvalCase] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            case = self._parse_eval_case(item, idx)
            if case is not None:
                cases.append(case)
        return cases

    def _parse_eval_case(self, item: dict[str, Any], idx: int) -> AgentEvalCase | None:
        case_id = str(item.get("id") or item.get("name") or f"case-{idx}")
        category = self._parse_category(item.get("category"))
        profile = self._parse_profile(item.get("profile"))
        tags = tuple(item.get("tags") or ())

        messages = item.get("messages")
        if not messages and "prompt" in item:
            messages = [{"role": "user", "content": item.get("prompt", "")}]
        if not messages:
            return None

        parsed_messages: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            parsed_messages.append({"role": role, "content": content})

        constraints = self._parse_constraints(item.get("constraints"))
        expected = self._parse_expected(item.get("expected"))

        return AgentEvalCase(
            case_id=case_id,
            category=category,
            profile=profile,
            tags=tags,
            messages=tuple(parsed_messages),
            constraints=constraints,
            expected=expected,
        )

    @staticmethod
    def _parse_category(value: Any) -> AgentEvalCaseCategory:
        if isinstance(value, str):
            for cat in AgentEvalCaseCategory:
                if cat.value == value:
                    return cat
        return AgentEvalCaseCategory.OTHER

    @staticmethod
    def _parse_profile(value: Any) -> AgentEvalCaseProfile:
        if isinstance(value, str):
            for profile in AgentEvalCaseProfile:
                if profile.value == value:
                    return profile
        return AgentEvalCaseProfile.OPEN

    def _parse_constraints(self, payload: Any) -> EvalCaseConstraints | None:
        if not isinstance(payload, dict):
            return None
        allowed_kinds = payload.get("allowed_action_kinds") or payload.get("allowedActionKinds")
        allowed_tools = payload.get("allowed_tools") or payload.get("allowedTools")
        max_steps = payload.get("max_steps") or payload.get("maxSteps")

        parsed_kinds: tuple[AgentActionKind, ...] | None = None
        if isinstance(allowed_kinds, list):
            kinds: list[AgentActionKind] = []
            for kind in allowed_kinds:
                parsed = self._parse_action_kind(kind)
                if parsed:
                    kinds.append(parsed)
            parsed_kinds = tuple(kinds) if kinds else None

        parsed_tools = tuple(str(t) for t in allowed_tools) if isinstance(allowed_tools, list) else None

        return EvalCaseConstraints(
            allowed_action_kinds=parsed_kinds,
            allowed_tools=parsed_tools,
            max_steps=int(max_steps) if isinstance(max_steps, (int, float)) else None,
        )

    def _parse_expected(self, payload: Any) -> Expected | None:
        if payload is None:
            return None

        options_payload = payload.get("any_of") if isinstance(payload, dict) else payload
        if not isinstance(options_payload, list):
            return None

        options: list[ExpectedOption] = []
        for option in options_payload:
            if not isinstance(option, dict):
                continue
            kind = self._parse_action_kind(option.get("kind"))
            if kind is None:
                continue
            tool_payload = option.get("tool")
            tool_spec = None
            if isinstance(tool_payload, dict):
                tool_name = tool_payload.get("name")
                if tool_name:
                    tool_spec = ExpectedToolSpec(
                        name=str(tool_name),
                        arguments=tool_payload.get("arguments"),
                    )
            options.append(ExpectedOption(kind=kind, tool=tool_spec))

        return Expected(any_of=tuple(options)) if options else None

    @staticmethod
    def _parse_action_kind(value: Any) -> AgentActionKind | None:
        if isinstance(value, str):
            for kind in AgentActionKind:
                if kind.value == value:
                    return kind
        return None

    @staticmethod
    def _render_messages(messages: tuple[dict[str, str], ...]) -> str:
        return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)

    @staticmethod
    def _allowed_action_kinds(
        case: AgentEvalCase, tools_enabled: bool
    ) -> tuple[AgentActionKind, ...] | None:
        if tools_enabled:
            return case.constraints.allowed_action_kinds if case.constraints else None
        if case.constraints and case.constraints.allowed_action_kinds:
            return tuple(
                kind
                for kind in case.constraints.allowed_action_kinds
                if kind != AgentActionKind.TOOL_CALL
            )
        return tuple(kind for kind in AgentActionKind if kind != AgentActionKind.TOOL_CALL)

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
