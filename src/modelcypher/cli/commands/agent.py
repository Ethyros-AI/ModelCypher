"""Agent trace CLI commands.

Provides commands for trace import, analysis, and action validation.

Commands:
    mc agent trace-import --file <path>
    mc agent trace-analyze --file <path>
    mc agent validate-action --action <json>
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("trace-import")
def agent_trace_import(
    ctx: typer.Context,
    file: str = typer.Option(..., "--file", help="Path to trace file (JSON)"),
    sanitize: bool = typer.Option(True, "--sanitize/--no-sanitize", help="Sanitize sensitive data"),
    max_value_length: int = typer.Option(1000, "--max-value-length", help="Max attribute value length"),
) -> None:
    """Import agent traces from Monocle/OpenTelemetry JSON format.

    Parses trace files and converts to ModelCypher's AgentTrace format.
    Supports both array of spans and object with 'spans' key.

    Examples:
        mc agent trace-import --file ./traces.json
        mc agent trace-import --file ./monocle_trace.json --no-sanitize
    """
    context = _context(ctx)

    from modelcypher.core.domain.agents import (
        MonocleTraceImporter,
        TraceImportError,
    )
    from modelcypher.core.domain.agents.agent_trace_value import ImportOptions

    file_path = Path(file)
    if not file_path.exists():
        error = ErrorDetail(
            code="MC-1060",
            title="Trace file not found",
            detail=f"Trace file does not exist: {file}",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    try:
        data = file_path.read_bytes()
    except OSError as exc:
        error = ErrorDetail(
            code="MC-1061",
            title="Failed to read trace file",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # If sanitize is False, use no_previews; otherwise use safe_default with custom length
    if sanitize:
        value_options = ImportOptions(max_string_preview_length=max_value_length)
    else:
        value_options = ImportOptions.no_previews()

    try:
        result = MonocleTraceImporter.import_file(
            data=data,
            file_name=file_path.name,
            value_options=value_options,
        )
    except TraceImportError as exc:
        error = ErrorDetail(
            code="MC-1062",
            title="Trace import failed",
            detail=str(exc),
            hint="Ensure file is valid JSON with spans array",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Build payload
    traces_payload = []
    for trace in result.traces:
        traces_payload.append({
            "id": str(trace.id),
            "kind": trace.kind.value,
            "status": trace.status.value,
            "startedAt": trace.started_at.isoformat() if trace.started_at else None,
            "completedAt": trace.completed_at.isoformat() if trace.completed_at else None,
            "baseModelId": trace.base_model_id,
            "spanCount": len(trace.spans),
            "source": {
                "system": trace.source.system if trace.source else None,
                "details": trace.source.details if trace.source else None,
                "workflowName": trace.source.workflow_name if trace.source else None,
            } if trace.source else None,
        })

    payload = {
        "filePath": str(file_path),
        "tracesImported": len(result.traces),
        "warnings": result.warnings,
        "traces": traces_payload[:10],  # Limit for output
    }

    if context.output_format == "text":
        lines = [
            "TRACE IMPORT RESULT",
            f"File: {file_path}",
            f"Traces Imported: {len(result.traces)}",
        ]
        if result.warnings:
            lines.append(f"Warnings: {len(result.warnings)}")
        lines.append("")
        lines.append("Traces:")
        for trace in result.traces[:5]:
            lines.append(f"  {trace.id}")
            lines.append(f"    Kind: {trace.kind.value}")
            lines.append(f"    Status: {trace.status.value}")
            lines.append(f"    Spans: {len(trace.spans)}")
        if len(result.traces) > 5:
            lines.append(f"  ... and {len(result.traces) - 5} more")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("trace-analyze")
def agent_trace_analyze(
    ctx: typer.Context,
    file: str = typer.Option(..., "--file", help="Path to trace file (JSON)"),
) -> None:
    """Analyze agent traces for patterns and compliance.

    Imports traces and computes analytics including:
    - Message counts by role
    - Action compliance rates
    - Entropy distribution buckets
    - Span timing analysis

    Examples:
        mc agent trace-analyze --file ./traces.json
    """
    context = _context(ctx)

    from modelcypher.core.domain.agents import (
        AgentTraceAnalytics,
        MonocleTraceImporter,
        TraceImportError,
    )
    from modelcypher.core.domain.agents.agent_trace_value import ImportOptions

    file_path = Path(file)
    if not file_path.exists():
        error = ErrorDetail(
            code="MC-1063",
            title="Trace file not found",
            detail=f"Trace file does not exist: {file}",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    try:
        data = file_path.read_bytes()
    except OSError as exc:
        error = ErrorDetail(
            code="MC-1064",
            title="Failed to read trace file",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    try:
        import_result = MonocleTraceImporter.import_file(
            data=data,
            file_name=file_path.name,
            value_options=ImportOptions.safe_default(),
        )
    except TraceImportError as exc:
        error = ErrorDetail(
            code="MC-1065",
            title="Trace import failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    if not import_result.traces:
        error = ErrorDetail(
            code="MC-1066",
            title="No traces found",
            detail="File contains no valid traces",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Analyze traces
    analytics = AgentTraceAnalytics()
    for trace in import_result.traces:
        analytics.add_trace(trace)

    summary = analytics.summary()

    payload = {
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
    }

    if context.output_format == "text":
        lines = [
            "TRACE ANALYSIS RESULT",
            f"File: {file_path}",
            f"Traces: {len(import_result.traces)}",
            f"Total Spans: {summary.total_spans}",
            "",
            "Message Counts:",
            f"  User: {summary.message_counts.user}",
            f"  Assistant: {summary.message_counts.assistant}",
            f"  System: {summary.message_counts.system}",
            f"  Tool: {summary.message_counts.tool}",
            "",
            "Action Compliance:",
            f"  Total Actions: {summary.action_compliance.total_actions}",
            f"  Compliant: {summary.action_compliance.compliant_actions}",
            f"  Rate: {summary.action_compliance.compliance_rate:.1%}",
            "",
            "Entropy Distribution:",
            f"  Low: {summary.entropy_buckets.low}",
            f"  Medium: {summary.entropy_buckets.medium}",
            f"  High: {summary.entropy_buckets.high}",
        ]
        if summary.unique_models:
            lines.append("")
            lines.append("Models Used:")
            for model in summary.unique_models:
                lines.append(f"  - {model}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("validate-action")
def agent_validate_action(
    ctx: typer.Context,
    action: str = typer.Argument(..., help="JSON action object"),
    strict: bool = typer.Option(False, "--strict", help="Use strict validation"),
) -> None:
    """Validate an agent action for safety and compliance.

    Checks action structure and content against safety rules.
    Input format: {"kind": "tool_call", "tool": "...", "input": {...}}

    Examples:
        mc agent validate-action '{"kind": "response", "content": "Hello"}'
        mc agent validate-action '{"kind": "tool_call", "tool": "search", "input": {"q": "test"}}'
    """
    context = _context(ctx)

    from modelcypher.core.domain.agents import (
        ActionKind,
        ActionResponse,
        ActionToolCall,
        AgentActionEnvelope,
        AgentActionValidator,
    )

    try:
        action_data = json.loads(action)
        if not isinstance(action_data, dict):
            raise ValueError("Action must be a JSON object")
    except json.JSONDecodeError as exc:
        error = ErrorDetail(
            code="MC-1067",
            title="Invalid action format",
            detail=str(exc),
            hint="Provide action as valid JSON object",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Extract action components and build envelope
    kind_str = action_data.get("kind", "response")
    content = action_data.get("content", "")
    tool_name = action_data.get("tool")
    tool_input = action_data.get("input", {})

    # Map simplified kind strings to ActionKind
    kind_map = {
        "response": ActionKind.RESPOND,
        "respond": ActionKind.RESPOND,
        "tool_call": ActionKind.TOOL_CALL,
        "tool": ActionKind.TOOL_CALL,
        "refuse": ActionKind.REFUSE,
        "refusal": ActionKind.REFUSE,
        "clarification": ActionKind.ASK_CLARIFICATION,
        "ask_clarification": ActionKind.ASK_CLARIFICATION,
        "defer": ActionKind.DEFERRAL,
        "deferral": ActionKind.DEFERRAL,
    }

    kind = kind_map.get(kind_str, ActionKind.RESPOND)

    # Build envelope
    tool = None
    response = None

    if kind == ActionKind.TOOL_CALL and tool_name:
        tool = ActionToolCall(name=tool_name, arguments=tool_input)
    elif kind == ActionKind.RESPOND:
        response = ActionResponse(text=content)

    envelope = AgentActionEnvelope.create(
        kind=kind,
        tool=tool,
        response=response,
    )

    # Validate
    result = AgentActionValidator.validate(envelope)

    payload = {
        "valid": result.is_valid,
        "kind": kind_str,
        "errors": result.errors,
        "warnings": result.warnings,
    }

    if context.output_format == "text":
        status = "VALID" if result.is_valid else "INVALID"
        lines = [
            "ACTION VALIDATION RESULT",
            f"Status: {status}",
            f"Kind: {kind_str}",
        ]
        if result.errors:
            lines.append("")
            lines.append("Errors:")
            for err in result.errors:
                lines.append(f"  - {err}")
        if result.warnings:
            lines.append("")
            lines.append("Warnings:")
            for warn in result.warnings:
                lines.append(f"  - {warn}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
