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

"""Agent MCP tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from .common import (
    MUTATING_ANNOTATIONS,
    READ_ONLY_ANNOTATIONS,
    ServiceContext,
    require_existing_directory,
)

if TYPE_CHECKING:
    pass


def register_agent_tools(ctx: ServiceContext) -> None:
    """Register agent-related MCP tools."""
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    if "mc_agent_eval_run" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_agent_eval_run(
            model: str,
            evalSuite: str = "default",
            maxTurns: int = 10,
            timeout: int = 300,
            seed: int | None = None,
        ) -> dict:
            """Execute agent evaluation."""
            from modelcypher.core.use_cases.agent_eval_service import AgentEvalService

            model_path = require_existing_directory(model)
            service = AgentEvalService(inference_engine=ctx.inference_engine)
            result = service.run(
                model_path=model_path,
                eval_suite=evalSuite,
                max_turns=maxTurns,
                timeout_seconds=timeout,
                seed=seed,
            )
            return {
                "_schema": "mc.agent_eval.run.v1",
                "evalId": result.eval_id,
                "modelPath": result.model_path,
                "evalSuite": result.eval_suite,
                "status": result.status,
                "startedAt": result.started_at,
                "config": result.config,
                "summary": result.summary,
            }

    if "mc_agent_eval_results" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_agent_eval_results(evalId: str) -> dict:
            """Get agent evaluation results."""
            from modelcypher.core.use_cases.agent_eval_service import AgentEvalService

            service = AgentEvalService(inference_engine=ctx.inference_engine)
            result = service.results(evalId)
            return {
                "_schema": "mc.agent_eval.results.v1",
                "evalId": result.eval_id,
                "modelPath": result.model_path,
                "evalSuite": result.eval_suite,
                "status": result.status,
                "startedAt": result.started_at,
                "completedAt": result.completed_at,
                "config": result.config,
                "metrics": result.metrics,
                "taskResults": result.task_results,
                "overallScore": result.overall_score,
            }

    # Phase 2: New agent tools
    if "mc_agent_trace_import" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_agent_trace_import(
            filePath: str,
            sanitize: bool = True,
            maxValueLength: int = 1000,
        ) -> dict:
            """Import agent traces from Monocle/OpenTelemetry JSON format."""
            from modelcypher.core.domain.agents import (
                MonocleTraceImporter,
                TraceImportError,
            )
            from modelcypher.core.domain.agents.agent_trace_value import ImportOptions

            file_path = Path(filePath).expanduser().resolve()
            if not file_path.exists():
                raise ValueError(f"Trace file not found: {file_path}")
            data = file_path.read_bytes()
            value_options = ImportOptions(
                sanitize_pii=sanitize,
                max_string_length=maxValueLength,
            )
            try:
                result = MonocleTraceImporter.import_file(
                    data=data,
                    file_name=file_path.name,
                    value_options=value_options,
                )
            except TraceImportError as exc:
                raise ValueError(f"Trace import failed: {exc}")
            traces_payload = []
            for trace in result.traces[:10]:
                traces_payload.append(
                    {
                        "id": str(trace.id),
                        "kind": trace.kind.value,
                        "status": trace.status.value,
                        "startedAt": trace.started_at.isoformat() if trace.started_at else None,
                        "completedAt": trace.completed_at.isoformat()
                        if trace.completed_at
                        else None,
                        "baseModelId": trace.base_model_id,
                        "spanCount": len(trace.spans),
                    }
                )
            return {
                "_schema": "mc.agent.trace_import.v1",
                "filePath": str(file_path),
                "tracesImported": len(result.traces),
                "warnings": result.warnings,
                "traces": traces_payload,
            }

    if "mc_agent_trace_analyze" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_agent_trace_analyze(filePath: str) -> dict:
            """Analyze agent traces for patterns and compliance."""
            from modelcypher.core.domain.agents import (
                AgentTraceAnalytics,
                MonocleTraceImporter,
                TraceImportError,
            )
            from modelcypher.core.domain.agents.agent_trace_value import ImportOptions

            file_path = Path(filePath).expanduser().resolve()
            if not file_path.exists():
                raise ValueError(f"Trace file not found: {file_path}")
            data = file_path.read_bytes()
            try:
                import_result = MonocleTraceImporter.import_file(
                    data=data,
                    file_name=file_path.name,
                    value_options=ImportOptions.safe_default(),
                )
            except TraceImportError as exc:
                raise ValueError(f"Trace import failed: {exc}")
            if not import_result.traces:
                raise ValueError("No traces found in file")

            # Use the static factory to compute analytics
            analytics = AgentTraceAnalytics.from_traces(
                traces=import_result.traces,
                requested_count=len(import_result.traces),
            )

            # Compute total spans across all traces
            total_spans = sum(len(t.spans) for t in import_result.traces)

            # Compliance info from the ActionCompliance dataclass
            compliance = analytics.action_compliance
            compliance_rate = 0.0
            if compliance.decoded_actions > 0:
                compliance_rate = compliance.valid_actions / compliance.decoded_actions

            return {
                "_schema": "mc.agent.trace_analyze.v1",
                "filePath": str(file_path),
                "traceCount": len(import_result.traces),
                "totalSpans": total_spans,
                "computedAt": analytics.computed_at.isoformat(),
                "timeRange": {
                    "oldest": analytics.oldest_started_at.isoformat()
                    if analytics.oldest_started_at
                    else None,
                    "newest": analytics.newest_started_at.isoformat()
                    if analytics.newest_started_at
                    else None,
                },
                "kinds": {k.value: v for k, v in analytics.kinds.items()},
                "statuses": {k.value: v for k, v in analytics.statuses.items()},
                "interventionCount": analytics.intervention_count,
                "actionCompliance": {
                    "decodedActions": compliance.decoded_actions,
                    "validActions": compliance.valid_actions,
                    "invalidActions": compliance.invalid_actions,
                    "unvalidatedActions": compliance.unvalidated_actions,
                    "complianceRate": compliance_rate,
                    "topErrors": [
                        {"message": e.message, "count": e.count}
                        for e in compliance.top_errors
                    ],
                },
                "entropyByCompliance": {
                    "validAction": {
                        "count": analytics.entropy_by_compliance.valid_action.count,
                        "average": analytics.entropy_by_compliance.valid_action.average,
                    },
                    "invalidAction": {
                        "count": analytics.entropy_by_compliance.invalid_action.count,
                        "average": analytics.entropy_by_compliance.invalid_action.average,
                    },
                },
                "issues": analytics.issues,
            }

    if "mc_agent_validate_action" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_agent_validate_action(
            action: str,
        ) -> dict:
            """Validate an agent action for safety and compliance.

            The action should be a JSON object conforming to the AgentActionEnvelope schema:
            {
                "_schema": "tc.agent.action.v1",
                "_version": 1,
                "kind": "tool_call" | "respond" | "ask_clarification" | "refuse" | "defer",
                "tool": {"name": "...", "arguments": {...}},  // for kind=tool_call
                "response": {"text": "..."},  // for kind=respond
                ...
            }
            """
            from modelcypher.core.domain.agents import (
                AgentActionEnvelope,
                AgentActionValidator,
            )

            try:
                action_data = json.loads(action)
                if not isinstance(action_data, dict):
                    raise ValueError("Action must be a JSON object")
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid action format: {exc}")

            # Parse the action envelope
            envelope = AgentActionEnvelope.from_dict(action_data)
            if envelope is None:
                # Missing schema or invalid format
                return {
                    "_schema": "mc.agent.validate_action.v1",
                    "valid": False,
                    "kind": action_data.get("kind"),
                    "errors": [
                        "Action must have _schema='tc.agent.action.v1' and _version=1"
                    ],
                    "warnings": [],
                }

            # Validate the envelope
            result = AgentActionValidator.validate(envelope)

            return {
                "_schema": "mc.agent.validate_action.v1",
                "valid": result.is_valid,
                "kind": envelope.kind.value,
                "errors": result.errors,
                "warnings": result.warnings,
            }
