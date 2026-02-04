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

from modelcypher.cli.composition import get_inference_engine
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.cli.prompt_input import resolve_prompt_input
from modelcypher.cli.input_validation import validate_model_path
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("run")
def infer_run(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Model identifier or path"),
    prompt: str | None = typer.Option(
        None,
        "--prompt",
        help="Input prompt (use --prompt-file or --prompt-stdin for multi-line)",
    ),
    prompt_file: str | None = typer.Option(
        None, "--prompt-file", help="Read prompt from a UTF-8 text file"
    ),
    prompt_stdin: bool = typer.Option(
        False, "--prompt-stdin", help="Read prompt from stdin (multi-line)"
    ),
    adapter: str | None = typer.Option(None, "--adapter", help="Path to adapter directory"),
    security_scan: bool = typer.Option(
        False, "--security-scan", help="Perform dual-path security analysis"
    ),
    entropy_aware: bool = typer.Option(
        False, "--entropy-aware", help="Enable real-time entropy monitoring"
    ),
    uncertainty_mode: str = typer.Option(
        "human_in_loop",
        "--uncertainty-mode",
        help="Uncertainty response mode: butler, autonomous, human_in_loop",
    ),
    entropy_threshold: float = typer.Option(
        0.7, "--entropy-threshold", help="Normalized entropy threshold (0-1)"
    ),
    eigenscore_threshold: float = typer.Option(
        0.6, "--eigenscore-threshold", help="EigenScore threshold (0-1)"
    ),
    agent: str | None = typer.Option(
        None,
        "--agent",
        help="Agent ID for LoRA memory. Enables learning from sparse regions.",
    ),
) -> None:
    """Execute inference with optional adapter and security scanning."""
    context = _context(ctx)

    # Validate model path early for clear error messages
    validate_model_path(model, context=context)

    prompt_text = resolve_prompt_input(
        prompt=prompt,
        prompt_file=prompt_file,
        prompt_stdin=prompt_stdin,
        context=context,
    )

    engine = get_inference_engine()

    # Use entropy-aware inference if requested
    if entropy_aware:
        try:
            result = engine.run_with_entropy(
                model=model,
                prompt=prompt_text,
                adapter=adapter,
                uncertainty_mode=uncertainty_mode,
                entropy_threshold=entropy_threshold,
                eigenscore_threshold=eigenscore_threshold,
                agent_id=agent,
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

        # Format entropy-aware result
        sparsity_count = len(result.entropy_summary.sparsity_events)
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
            "agent": agent,
            "agentLoraLoaded": result.agent_lora_loaded,
            "uncertaintyMode": result.uncertainty_mode,
            "entropy": {
                "meanEntropy": result.entropy_summary.mean_entropy,
                "maxEntropy": result.entropy_summary.max_entropy,
                "meanEigenscore": result.entropy_summary.mean_eigenscore,
                "maxEigenscore": result.entropy_summary.max_eigenscore,
                "uncertaintyEvents": result.entropy_summary.uncertainty_events,
                "abstentionTriggered": result.entropy_summary.abstention_triggered,
                "sparsityEventsQueued": sparsity_count,
            },
        }

        if context.output_format == "text":
            # Color-code based on uncertainty
            entropy_color = ""
            if result.entropy_summary.max_entropy > 0.8:
                entropy_color = " [HIGH]"
            elif result.entropy_summary.max_entropy > 0.5:
                entropy_color = " [MODERATE]"

            lines = [
                "ENTROPY-AWARE INFERENCE",
                f"Model: {result.model}",
                f"Mode: {result.uncertainty_mode}",
                f"Prompt: {result.prompt[:50]}...",
                f"Response: {result.response[:100]}...",
                f"Tokens: {result.token_count} ({result.tokens_per_second:.1f} tok/s)",
                f"Duration: {result.total_duration:.2f}s",
                f"Stop reason: {result.stop_reason}",
                "",
                "ENTROPY METRICS:",
                f"  Mean entropy: {result.entropy_summary.mean_entropy:.3f}{entropy_color}",
                f"  Max entropy: {result.entropy_summary.max_entropy:.3f}",
                f"  Mean EigenScore: {result.entropy_summary.mean_eigenscore:.3f}",
                f"  Max EigenScore: {result.entropy_summary.max_eigenscore:.3f}",
                f"  Uncertainty events: {result.entropy_summary.uncertainty_events}",
                f"  Abstention triggered: {result.entropy_summary.abstention_triggered}",
            ]
            if result.adapter:
                adapter_note = " (auto-loaded from agent)" if result.agent_lora_loaded else ""
                lines.insert(3, f"Adapter: {result.adapter}{adapter_note}")
            if agent:
                lines.append("")
                lines.append("LORA MEMORY:")
                lines.append(f"  Agent: {agent}")
                if result.agent_lora_loaded:
                    lines.append("  LoRA loaded: Yes (using learned knowledge)")
                if sparsity_count > 0:
                    lines.append(f"  Sparsity events saved: {sparsity_count}")
                elif not result.agent_lora_loaded:
                    lines.append("  LoRA loaded: No (no trained weights yet)")
            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)
        return

    # Standard inference path
    try:
        result = engine.run(
            model=model,
            prompt=prompt_text,
            adapter=adapter,
            security_scan=security_scan,
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
            "anomalyCount": result.security.anomaly_count,
            "maxAnomalyScore": result.security.max_anomaly_score,
            "avgDelta": result.security.avg_delta,
            "disagreementRate": result.security.disagreement_rate,
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
            lines.append(
                "Security metrics: "
                f"anomalies={result.security.anomaly_count}, "
                f"max_score={result.security.max_anomaly_score:.3f}"
            )
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
) -> None:
    """Execute batched inference over a suite of prompts."""
    context = _context(ctx)

    # Validate inputs early for clear error messages
    validate_model_path(model, context=context)
    from modelcypher.cli.input_validation import validate_file_exists
    validate_file_exists(suite_file, description="Suite file", context=context)

    engine = get_inference_engine()

    try:
        result = engine.suite(
            model=model,
            suite_file=suite_file,
            adapter=adapter,
            security_scan=security_scan,
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
