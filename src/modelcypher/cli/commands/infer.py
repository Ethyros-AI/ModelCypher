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

"""Inference CLI commands."""

from __future__ import annotations

import typer

from modelcypher.adapters.local_inference import LocalInferenceEngine
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("run")
def infer_run(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Model identifier or path"),
    prompt: str = typer.Option(..., "--prompt", help="Input prompt"),
    adapter: str | None = typer.Option(None, "--adapter", help="Path to adapter directory"),
    security_scan: bool = typer.Option(
        False, "--security-scan", help="Perform dual-path security analysis"
    ),
    max_tokens: int = typer.Option(512, "--max-tokens", help="Max tokens per response"),
    temperature: float = typer.Option(0.7, "--temperature", help="Sampling temperature"),
    top_p: float = typer.Option(0.95, "--top-p", help="Top-p sampling"),
) -> None:
    """Execute inference with optional adapter and security scanning."""
    context = _context(ctx)
    engine = LocalInferenceEngine()

    try:
        result = engine.run(
            model=model,
            prompt=prompt,
            adapter=adapter,
            security_scan=security_scan,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1015",
            title="Inference failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
    except RuntimeError as exc:
        error = ErrorDetail(
            code="MC-1017",
            title="Inference locked",
            detail=str(exc),
            hint="Wait for training to complete or cancel it",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "model": result.model,
        "prompt": result.prompt,
        "response": result.response,
        "tokenCount": result.token_count,
        "tokensPerSecond": result.tokens_per_second,
        "timeToFirstToken": result.time_to_first_token,
        "totalDuration": result.total_duration,
        "stopReason": result.stop_reason,
        "adapter": result.adapter,
    }

    if result.security:
        payload["security"] = {
            "hasSecurityFlags": result.security.has_security_flags,
            "anomalyCount": result.security.anomaly_count,
            "maxAnomalyScore": result.security.max_anomaly_score,
            "avgDelta": result.security.avg_delta,
            "disagreementRate": result.security.disagreement_rate,
            "circuitBreakerTripped": result.security.circuit_breaker_tripped,
            "circuitBreakerTripIndex": result.security.circuit_breaker_trip_index,
        }

    if context.output_format == "text":
        lines = [
            "INFERENCE RESULT",
            f"Model: {result.model}",
            f"Prompt: {result.prompt[:50]}...",
            f"Response: {result.response[:100]}...",
            f"Tokens: {result.token_count} ({result.tokens_per_second:.1f} tok/s)",
            f"Duration: {result.total_duration:.2f}s",
        ]
        if result.adapter:
            lines.append(f"Adapter: {result.adapter}")
        if result.security:
            lines.append(f"Security flags: {result.security.has_security_flags}, score: {result.security.max_anomaly_score:.3f}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("suite")
def infer_suite(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Model identifier or path"),
    suite_file: str = typer.Option(..., "--suite", help="Path to suite file (.txt, .json, .jsonl)"),
    adapter: str | None = typer.Option(None, "--adapter", help="Path to adapter directory"),
    security_scan: bool = typer.Option(False, "--security-scan", help="Perform security analysis"),
    max_tokens: int = typer.Option(512, "--max-tokens", help="Default max tokens"),
    temperature: float = typer.Option(0.7, "--temperature", help="Default temperature"),
) -> None:
    """Execute batched inference over a suite of prompts."""
    context = _context(ctx)
    engine = LocalInferenceEngine()

    try:
        result = engine.suite(
            model=model,
            suite_file=suite_file,
            adapter=adapter,
            security_scan=security_scan,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1016",
            title="Inference suite failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    cases_payload = []
    for case in result.cases:
        case_dict = {
            "name": case.name,
            "prompt": case.prompt,
            "response": case.response,
            "tokenCount": case.token_count,
            "duration": case.duration,
            "passed": case.passed,
            "expected": case.expected,
        }
        if case.error:
            case_dict["error"] = case.error
        cases_payload.append(case_dict)

    payload = {
        "model": result.model,
        "adapter": result.adapter,
        "suite": result.suite,
        "totalCases": result.total_cases,
        "passed": result.passed,
        "failed": result.failed,
        "totalDuration": result.total_duration,
        "summary": result.summary,
        "cases": cases_payload[:10],
    }

    if context.output_format == "text":
        lines = [
            "INFERENCE SUITE RESULTS",
            f"Model: {result.model}",
            f"Suite: {result.suite}",
            f"Cases: {result.total_cases} ({result.passed} passed, {result.failed} failed)",
        ]
        if result.summary.get("pass_rate") is not None:
            lines.append(f"Pass Rate: {result.summary.get('pass_rate', 0) * 100:.1f}%")
        lines.extend([
            f"Duration: {result.total_duration:.2f}s",
            "",
            "Case Results:",
        ])
        for case in result.cases:
            if case.passed is not None:
                status = "+" if case.passed else "x"
            else:
                status = "o"
            lines.append(f"  {status} {case.name}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
