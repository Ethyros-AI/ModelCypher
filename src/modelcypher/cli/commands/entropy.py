"""Entropy analysis CLI commands.

Provides commands for entropy pattern analysis, distress detection,
and baseline verification for model adapters.

Commands:
    mc entropy analyze <samples>
    mc entropy detect-distress <samples>
    mc entropy verify-baseline --mean ... --std-dev ... --max ... --min ... --observed ...
"""

from __future__ import annotations

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.core.use_cases.entropy_probe_service import EntropyProbeService
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("analyze")
def entropy_analyze(
    ctx: typer.Context,
    samples: str = typer.Argument(
        ...,
        help="JSON array of [entropy, variance] pairs, e.g. '[[3.5, 0.2], [3.6, 0.1]]'"
    ),
) -> None:
    """Analyze entropy/variance samples for patterns and trends."""
    context = _context(ctx)
    import json as json_lib

    service = EntropyProbeService()

    try:
        sample_list = json_lib.loads(samples)
        if not isinstance(sample_list, list):
            raise ValueError("Samples must be a JSON array")
        parsed_samples = [(float(s[0]), float(s[1])) for s in sample_list]
    except (json_lib.JSONDecodeError, IndexError, TypeError, ValueError) as exc:
        error = ErrorDetail(
            code="MC-1050",
            title="Invalid samples format",
            detail=str(exc),
            hint="Provide samples as JSON array of [entropy, variance] pairs",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    pattern = service.analyze_pattern(parsed_samples)
    payload = service.pattern_payload(pattern)

    if context.output_format == "text":
        lines = [
            "ENTROPY PATTERN ANALYSIS",
            f"Trend: {pattern.trend.value}",
            f"Trend Slope: {pattern.trend_slope:.4f}",
            f"Volatility: {pattern.volatility:.4f}",
            f"Entropy Mean: {pattern.entropy_mean:.4f}",
            f"Entropy StdDev: {pattern.entropy_std_dev:.4f}",
            f"Variance Mean: {pattern.variance_mean:.4f}",
            f"Entropy-Variance Correlation: {pattern.entropy_variance_correlation:.4f}",
            f"Sustained High Count: {pattern.sustained_high_count}",
            f"Peak Entropy: {pattern.peak_entropy:.4f}",
            f"Min Entropy: {pattern.min_entropy:.4f}",
            f"Sample Count: {pattern.sample_count}",
            f"Concerning: {'YES' if pattern.is_concerning else 'NO'}",
        ]
        if pattern.anomaly_indices:
            lines.append(f"Anomaly Indices: {list(pattern.anomaly_indices)}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("detect-distress")
def entropy_detect_distress(
    ctx: typer.Context,
    samples: str = typer.Argument(
        ...,
        help="JSON array of [entropy, variance] pairs, e.g. '[[3.5, 0.2], [3.6, 0.1]]'"
    ),
) -> None:
    """Detect distress patterns in entropy/variance samples."""
    context = _context(ctx)
    import json as json_lib

    service = EntropyProbeService()

    try:
        sample_list = json_lib.loads(samples)
        if not isinstance(sample_list, list):
            raise ValueError("Samples must be a JSON array")
        parsed_samples = [(float(s[0]), float(s[1])) for s in sample_list]
    except (json_lib.JSONDecodeError, IndexError, TypeError, ValueError) as exc:
        error = ErrorDetail(
            code="MC-1051",
            title="Invalid samples format",
            detail=str(exc),
            hint="Provide samples as JSON array of [entropy, variance] pairs",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    distress = service.detect_distress(parsed_samples)
    payload = service.distress_payload(distress)

    if context.output_format == "text":
        if distress is None:
            write_output("No distress detected", context.output_format, context.pretty)
            return
        lines = [
            "DISTRESS DETECTION RESULT",
            f"Detected: YES",
            f"Confidence: {distress.confidence:.2%}",
            f"Sustained High Count: {distress.sustained_high_count}",
            f"Average Entropy: {distress.average_entropy:.4f}",
            f"Average Variance: {distress.average_variance:.4f}",
            f"Correlation: {distress.correlation:.4f}",
            f"Indicators: {', '.join(distress.indicators)}",
            f"Recommended Action: {distress.recommended_action.value}",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("verify-baseline")
def entropy_verify_baseline(
    ctx: typer.Context,
    declared_mean: float = typer.Option(..., "--mean", help="Declared delta mean"),
    declared_std_dev: float = typer.Option(..., "--std-dev", help="Declared delta standard deviation"),
    declared_max: float = typer.Option(..., "--max", help="Declared maximum delta"),
    declared_min: float = typer.Option(..., "--min", help="Declared minimum delta"),
    observed_deltas: str = typer.Option(
        ..., "--observed",
        help="JSON array of observed delta values, e.g. '[0.1, 0.15, 0.12]'"
    ),
    base_model_id: str = typer.Option("unknown", "--base-model", help="Base model identifier"),
    adapter_path: str = typer.Option("unknown", "--adapter", help="Path to adapter"),
    tier: str = typer.Option("default", "--tier", help="Verification tier: quick, default, thorough"),
) -> None:
    """Verify observed entropy deltas against declared baseline."""
    context = _context(ctx)
    import json as json_lib

    service = EntropyProbeService()

    try:
        deltas = json_lib.loads(observed_deltas)
        if not isinstance(deltas, list):
            raise ValueError("Observed deltas must be a JSON array")
        parsed_deltas = [float(d) for d in deltas]
    except (json_lib.JSONDecodeError, TypeError, ValueError) as exc:
        error = ErrorDetail(
            code="MC-1052",
            title="Invalid deltas format",
            detail=str(exc),
            hint="Provide observed deltas as JSON array of numbers",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    result = service.verify_baseline(
        declared_mean=declared_mean,
        declared_std_dev=declared_std_dev,
        declared_max=declared_max,
        declared_min=declared_min,
        observed_deltas=parsed_deltas,
        base_model_id=base_model_id,
        adapter_path=adapter_path,
        tier=tier,
    )
    payload = service.verification_payload(result)

    if context.output_format == "text":
        write_output(result.summary, context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
