"""Geometry refusal direction CLI commands.

Provides commands for detecting and analyzing refusal directions in model
representations using contrastive prompt pairs.

Commands:
    mc geometry refusal pairs
    mc geometry refusal detect <harmful_file> <harmless_file>
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.core.use_cases.geometry_sparse_service import GeometrySparseService
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("pairs")
def geometry_refusal_pairs(ctx: typer.Context) -> None:
    """List standard contrastive prompt pairs for refusal direction."""
    context = _context(ctx)
    service = GeometrySparseService()
    pairs = service.get_contrastive_pairs()
    payload = service.contrastive_pairs_payload(pairs)

    if context.output_format == "text":
        lines = [
            "CONTRASTIVE PROMPT PAIRS",
            f"Total: {payload['count']}",
            "",
        ]
        for i, p in enumerate(payload["pairs"], 1):
            lines.append(f"{i}. Harmful: {p['harmful'][:60]}...")
            lines.append(f"   Harmless: {p['harmless'][:60]}...")
            lines.append("")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("detect")
def geometry_refusal_detect(
    ctx: typer.Context,
    harmful_file: str = typer.Argument(..., help="Path to harmful activations JSON"),
    harmless_file: str = typer.Argument(..., help="Path to harmless activations JSON"),
    layer_index: int = typer.Option(..., "--layer", help="Layer index"),
    model_id: str = typer.Option("unknown", "--model-id", help="Model identifier"),
    normalize: bool = typer.Option(True, "--normalize/--no-normalize", is_flag=True, flag_value=True, help="Normalize direction"),
) -> None:
    """
    Detect refusal direction from contrastive activations.

    Input files should contain JSON arrays of activation vectors:
    [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
    """
    context = _context(ctx)
    service = GeometrySparseService()

    harmful = json.loads(Path(harmful_file).read_text())
    harmless = json.loads(Path(harmless_file).read_text())

    direction = service.detect_refusal_direction(
        harmful_activations=harmful,
        harmless_activations=harmless,
        layer_index=layer_index,
        model_id=model_id,
        normalize=normalize,
    )

    if direction is None:
        write_error(
            ErrorDetail(
                code="MC-4010",
                message="Failed to compute refusal direction",
                detail="Insufficient data or activation difference below threshold",
            ),
            context.output_format,
        )
        raise typer.Exit(1)

    payload = service.refusal_direction_payload(direction)
    payload["nextActions"] = [
        "mc geometry refusal pairs to see contrastive prompts",
        "mc geometry safety circuit-breaker for safety assessment",
    ]

    if context.output_format == "text":
        lines = [
            "REFUSAL DIRECTION",
            f"Model: {direction.model_id}",
            f"Layer: {direction.layer_index}",
            f"Hidden Size: {direction.hidden_size}",
            f"Strength: {direction.strength:.4f}",
            f"Explained Variance: {direction.explained_variance:.2%}",
            f"Computed At: {direction.computed_at.isoformat()}",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
