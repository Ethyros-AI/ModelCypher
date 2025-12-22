"""Thermodynamic analysis CLI commands.

Provides commands for thermodynamic analysis of model training,
including entropy measurement, path integration, and unsafe pattern detection.

Commands:
    mc thermo analyze <job_id>
    mc thermo path --checkpoint <path> ...
    mc thermo entropy <job_id>
    mc thermo measure <prompt> --model <path>
    mc thermo detect <prompt> --model <path>
    mc thermo detect-batch <prompts_file> --model <path>
"""

from __future__ import annotations

from typing import Optional

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("analyze")
def thermo_analyze(
    ctx: typer.Context,
    job_id: str = typer.Argument(..., help="Job ID to analyze"),
) -> None:
    """Thermodynamic analysis of training."""
    context = _context(ctx)
    from modelcypher.core.use_cases.thermo_service import ThermoService

    service = ThermoService()
    result = service.analyze(job_id)

    payload = {
        "jobId": result.job_id,
        "entropy": result.entropy,
        "temperature": result.temperature,
        "freeEnergy": result.free_energy,
        "interpretation": result.interpretation,
    }

    if context.output_format == "text":
        lines = [
            "THERMO ANALYSIS",
            f"Job: {result.job_id}",
            f"Entropy: {result.entropy:.4f}",
            f"Temperature: {result.temperature:.4f}",
            f"Free Energy: {result.free_energy:.4f}",
            "",
            result.interpretation,
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("path")
def thermo_path(
    ctx: typer.Context,
    checkpoints: list[str] = typer.Option(..., "--checkpoint", help="Checkpoint paths"),
) -> None:
    """Path integration analysis between checkpoints."""
    context = _context(ctx)
    from modelcypher.core.use_cases.thermo_service import ThermoService

    service = ThermoService()
    try:
        result = service.path(checkpoints)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1008",
            title="Path analysis failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "checkpoints": result.checkpoints,
        "pathLength": result.path_length,
        "curvature": result.curvature,
        "interpretation": result.interpretation,
    }

    write_output(payload, context.output_format, context.pretty)


@app.command("entropy")
def thermo_entropy(
    ctx: typer.Context,
    job_id: str = typer.Argument(..., help="Job ID"),
) -> None:
    """Entropy metrics over training."""
    context = _context(ctx)
    from modelcypher.core.use_cases.thermo_service import ThermoService

    service = ThermoService()
    result = service.entropy(job_id)

    payload = {
        "jobId": result.job_id,
        "entropyHistory": result.entropy_history,
        "finalEntropy": result.final_entropy,
        "entropyTrend": result.entropy_trend,
    }

    write_output(payload, context.output_format, context.pretty)


@app.command("measure")
def thermo_measure(
    ctx: typer.Context,
    prompt: str = typer.Argument(..., help="Prompt to measure"),
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    modifiers: Optional[list[str]] = typer.Option(None, "--modifier", help="Modifier names to use"),
) -> None:
    """Measure entropy across linguistic modifiers for a prompt."""
    context = _context(ctx)
    from modelcypher.core.use_cases.thermo_service import ThermoService

    service = ThermoService()
    result = service.measure(prompt, model, modifiers)

    payload = {
        "basePrompt": result.base_prompt,
        "measurements": [
            {
                "modifier": m.modifier,
                "meanEntropy": m.mean_entropy,
                "deltaH": m.delta_h,
                "ridgeCrossed": m.ridge_crossed,
                "behavioralOutcome": m.behavioral_outcome,
            }
            for m in result.measurements
        ],
        "statistics": {
            "meanEntropy": result.statistics.mean_entropy,
            "stdEntropy": result.statistics.std_entropy,
            "minEntropy": result.statistics.min_entropy,
            "maxEntropy": result.statistics.max_entropy,
            "meanDeltaH": result.statistics.mean_delta_h,
            "intensityCorrelation": result.statistics.intensity_correlation,
        },
        "timestamp": result.timestamp.isoformat(),
    }

    if context.output_format == "text":
        lines = [
            "THERMO MEASURE",
            f"Prompt: {result.base_prompt[:50]}{'...' if len(result.base_prompt) > 50 else ''}",
            "",
            "Measurements:",
        ]
        for m in result.measurements:
            delta_str = f"{m.delta_h:.4f}" if m.delta_h is not None else "N/A"
            lines.append(f"  {m.modifier}: entropy={m.mean_entropy:.4f}, delta_h={delta_str}, outcome={m.behavioral_outcome}")
        lines.append("")
        lines.append(f"Mean Entropy: {result.statistics.mean_entropy:.4f}")
        lines.append(f"Std Entropy: {result.statistics.std_entropy:.4f}")
        if result.statistics.intensity_correlation is not None:
            lines.append(f"Intensity Correlation: {result.statistics.intensity_correlation:.4f}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("detect")
def thermo_detect(
    ctx: typer.Context,
    prompt: str = typer.Argument(..., help="Prompt to analyze"),
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    preset: str = typer.Option("default", "--preset", help="Preset: default, strict, sensitive, quick"),
) -> None:
    """Detect unsafe prompt patterns via entropy differential."""
    context = _context(ctx)
    from modelcypher.core.use_cases.thermo_service import ThermoService

    service = ThermoService()
    result = service.detect(prompt, model, preset)

    payload = {
        "prompt": result.prompt,
        "classification": result.classification,
        "riskLevel": result.risk_level,
        "confidence": result.confidence,
        "baselineEntropy": result.baseline_entropy,
        "intensityEntropy": result.intensity_entropy,
        "deltaH": result.delta_h,
        "processingTime": result.processing_time,
    }

    if context.output_format == "text":
        risk_labels = {0: "NONE", 1: "LOW", 2: "MEDIUM", 3: "HIGH"}
        lines = [
            "THERMO DETECT",
            f"Prompt: {result.prompt[:50]}{'...' if len(result.prompt) > 50 else ''}",
            "",
            f"Classification: {result.classification.upper()}",
            f"Risk Level: {result.risk_level} ({risk_labels.get(result.risk_level, 'UNKNOWN')})",
            f"Confidence: {result.confidence:.2%}",
            "",
            f"Baseline Entropy: {result.baseline_entropy:.4f}",
            f"Intensity Entropy: {result.intensity_entropy:.4f}",
            f"Delta H: {result.delta_h:.4f}",
            f"Processing Time: {result.processing_time:.3f}s",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("detect-batch")
def thermo_detect_batch(
    ctx: typer.Context,
    prompts_file: str = typer.Argument(..., help="Path to prompts file (JSON array or newline-separated)"),
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    preset: str = typer.Option("default", "--preset", help="Preset: default, strict, sensitive, quick"),
) -> None:
    """Batch detect unsafe patterns across multiple prompts."""
    context = _context(ctx)
    from modelcypher.core.use_cases.thermo_service import ThermoService

    service = ThermoService()

    try:
        results = service.detect_batch(prompts_file, model, preset)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1010",
            title="Batch detection failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "promptsFile": prompts_file,
        "totalPrompts": len(results),
        "results": [
            {
                "prompt": r.prompt,
                "classification": r.classification,
                "riskLevel": r.risk_level,
                "confidence": r.confidence,
                "deltaH": r.delta_h,
            }
            for r in results
        ],
        "summary": {
            "safe": sum(1 for r in results if r.classification == "safe"),
            "unsafe": sum(1 for r in results if r.classification == "unsafe"),
            "ambiguous": sum(1 for r in results if r.classification == "ambiguous"),
        },
    }

    if context.output_format == "text":
        lines = [
            "THERMO DETECT BATCH",
            f"File: {prompts_file}",
            f"Total Prompts: {len(results)}",
            "",
            "Summary:",
            f"  Safe: {payload['summary']['safe']}",
            f"  Unsafe: {payload['summary']['unsafe']}",
            f"  Ambiguous: {payload['summary']['ambiguous']}",
            "",
            "Results:",
        ]
        for i, r in enumerate(results[:10]):  # Show first 10
            prompt_preview = r.prompt[:30] + "..." if len(r.prompt) > 30 else r.prompt
            lines.append(f"  {i+1}. [{r.classification.upper()}] {prompt_preview}")
        if len(results) > 10:
            lines.append(f"  ... and {len(results) - 10} more")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
