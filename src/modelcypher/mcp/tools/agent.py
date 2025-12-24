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
    READ_ONLY_ANNOTATIONS,
    MUTATING_ANNOTATIONS,
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
            from modelcypher.core.use_cases.agent_eval_service import (
                AgentEvalConfig,
                AgentEvalService,
            )
            model_path = require_existing_directory(model)
            config = AgentEvalConfig(
                model_path=model_path,
                eval_suite=evalSuite,
                max_turns=maxTurns,
                timeout_seconds=timeout,
                seed=seed,
            )
            service = AgentEvalService()
            result = service.run(config)
            return {
                "_schema": "mc.agent_eval.run.v1",
                "evalId": result.eval_id,
                "modelPath": result.model_path,
                "evalSuite": result.eval_suite,
                "status": result.status,
                "startedAt": result.started_at,
                "config": result.config,
                "summary": result.summary,
                "nextActions": [
                    f"mc_agent_eval_results with evalId={result.eval_id}",
                ],
            }

    if "mc_agent_eval_results" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_agent_eval_results(evalId: str) -> dict:
            """Get agent evaluation results."""
            from modelcypher.core.use_cases.agent_eval_service import AgentEvalService
            service = AgentEvalService()
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
                "interpretation": result.interpretation,
                "overallScore": result.overall_score,
                "nextActions": [
                    "mc_agent_eval_run to run another evaluation",
                ],
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
                traces_payload.append({
                    "id": str(trace.id),
                    "kind": trace.kind.value,
                    "status": trace.status.value,
                    "startedAt": trace.started_at.isoformat() if trace.started_at else None,
                    "completedAt": trace.completed_at.isoformat() if trace.completed_at else None,
                    "baseModelId": trace.base_model_id,
                    "spanCount": len(trace.spans),
                })
            return {
                "_schema": "mc.agent.trace_import.v1",
                "filePath": str(file_path),
                "tracesImported": len(result.traces),
                "warnings": result.warnings,
                "traces": traces_payload,
                "nextActions": [
                    "mc_agent_trace_analyze to analyze imported traces",
                    "mc_agent_validate_action to validate specific actions",
                ],
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
            analytics = AgentTraceAnalytics()
            for trace in import_result.traces:
                analytics.add_trace(trace)
            summary = analytics.summary()
            return {
                "_schema": "mc.agent.trace_analyze.v1",
                "filePath": str(file_path),
                "traceCount": len(import_result.traces),
                "totalSpans": summary.total_spans,
                "messageCounts": {
                    "user": summary.message_counts.user,
                    "assistant": summary.message_counts.assistant,
                    "system": summary.message_counts.system,
                    "tool": summary.message_counts.tool,
                },
                "actionCompliance": {
                    "totalActions": summary.action_compliance.total_actions,
                    "compliantActions": summary.action_compliance.compliant_actions,
                    "complianceRate": summary.action_compliance.compliance_rate,
                },
                "entropyBuckets": {
                    "low": summary.entropy_buckets.low,
                    "medium": summary.entropy_buckets.medium,
                    "high": summary.entropy_buckets.high,
                },
                "averageSpanDurationMs": summary.average_span_duration_ms,
                "uniqueModels": list(summary.unique_models),
                "nextActions": [
                    "mc_agent_validate_action to validate specific actions",
                    "mc_entropy_analyze for entropy pattern analysis",
                ],
            }

    if "mc_agent_validate_action" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_agent_validate_action(
            action: str,
            strict: bool = False,
        ) -> dict:
            """Validate an agent action for safety and compliance."""
            from modelcypher.core.domain.agents import AgentActionValidator
            try:
                action_data = json.loads(action)
                if not isinstance(action_data, dict):
                    raise ValueError("Action must be a JSON object")
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid action format: {exc}")
            validator = AgentActionValidator()
            kind_str = action_data.get("kind", "response")
            content = action_data.get("content", "")
            tool_name = action_data.get("tool")
            tool_input = action_data.get("input", {})
            result = validator.validate(
                kind=kind_str,
                content=content,
                tool_name=tool_name,
                tool_input=tool_input,
                strict=strict,
            )
            return {
                "_schema": "mc.agent.validate_action.v1",
                "valid": result.is_valid,
                "kind": kind_str,
                "errors": result.errors,
                "warnings": result.warnings,
                "sanitizedContent": result.sanitized_content,
                "riskLevel": result.risk_level.value if result.risk_level else None,
                "blockedPatterns": result.blocked_patterns,
                "nextActions": [
                    "mc_agent_trace_analyze for full trace analysis",
                    "mc_safety_circuit_breaker if risk detected",
                ],
            }
