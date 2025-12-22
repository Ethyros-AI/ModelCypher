"""Geometry semantic primes CLI commands.

Provides commands for:
- List semantic prime anchors
- Probe model for prime activation patterns
- Compare prime alignment between models

Commands:
    mc geometry primes list
    mc geometry primes probe <model-path>
    mc geometry primes compare --model-a <path> --model-b <path>
"""

from __future__ import annotations

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.core.use_cases.geometry_primes_service import GeometryPrimesService
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("list")
def geometry_primes_list(ctx: typer.Context) -> None:
    """List all semantic prime anchors.

    Examples:
        mc geometry primes list
    """
    context = _context(ctx)
    service = GeometryPrimesService()
    primes = service.list_primes()

    payload = {
        "primes": [
            {
                "id": p.id,
                "name": p.name,
                "category": p.category,
                "exponents": p.exponents,
            }
            for p in primes
        ],
        "count": len(primes),
    }

    if context.output_format == "text":
        lines = ["SEMANTIC PRIMES", f"Total: {len(primes)}", ""]
        for p in primes:
            lines.append(f"  {p.id}: {p.name} ({p.category})")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("probe")
def geometry_primes_probe(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model directory"),
) -> None:
    """Probe model for prime activation patterns.

    Examples:
        mc geometry primes probe ./model
    """
    context = _context(ctx)
    service = GeometryPrimesService()

    try:
        activations = service.probe(model_path)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1003",
            title="Prime probe failed",
            detail=str(exc),
            hint="Ensure the path points to a valid model directory",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "modelPath": model_path,
        "activations": [
            {
                "primeId": a.prime_id,
                "activationStrength": a.activation_strength,
                "layerActivations": a.layer_activations,
            }
            for a in activations
        ],
        "count": len(activations),
    }

    if context.output_format == "text":
        lines = ["PRIME ACTIVATIONS", f"Model: {model_path}", ""]
        for a in activations[:20]:  # Limit output
            lines.append(f"  {a.prime_id}: {a.activation_strength:.3f}")
        if len(activations) > 20:
            lines.append(f"  ... and {len(activations) - 20} more")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("compare")
def geometry_primes_compare(
    ctx: typer.Context,
    model_a: str = typer.Option(..., "--model-a", help="Path to first model"),
    model_b: str = typer.Option(..., "--model-b", help="Path to second model"),
) -> None:
    """Compare prime alignment between two models.

    Examples:
        mc geometry primes compare --model-a ./model1 --model-b ./model2
    """
    context = _context(ctx)
    service = GeometryPrimesService()

    try:
        result = service.compare(model_a, model_b)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1004",
            title="Prime comparison failed",
            detail=str(exc),
            hint="Ensure both paths point to valid model directories",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "modelA": model_a,
        "modelB": model_b,
        "alignmentScore": result.alignment_score,
        "divergentPrimes": result.divergent_primes,
        "convergentPrimes": result.convergent_primes,
        "interpretation": result.interpretation,
    }

    if context.output_format == "text":
        lines = [
            "PRIME COMPARISON",
            f"Model A: {model_a}",
            f"Model B: {model_b}",
            f"Alignment Score: {result.alignment_score:.3f}",
            "",
            result.interpretation,
        ]
        if result.divergent_primes:
            lines.append(f"\nDivergent Primes: {', '.join(result.divergent_primes[:10])}")
        if result.convergent_primes:
            lines.append(f"Convergent Primes: {', '.join(result.convergent_primes[:10])}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
