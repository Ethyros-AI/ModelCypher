"""Geometry transport-guided merge CLI commands.

Provides commands for transport-guided model weight merging using
Gromov-Wasserstein optimal transport.

Commands:
    mc geometry transport merge --source <file> --target <file> --plan <file>
    mc geometry transport synthesize --source-act <file> --target-act <file> ...
"""

from __future__ import annotations

import json
from pathlib import Path


import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.core.use_cases.geometry_transport_service import GeometryTransportService, MergeConfig
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("merge")
def geometry_transport_merge(
    ctx: typer.Context,
    source_file: Path = typer.Option(..., "--source", "-s", help="JSON file with source weights [N x D]"),
    target_file: Path = typer.Option(..., "--target", "-t", help="JSON file with target weights [M x D]"),
    plan_file: Path = typer.Option(..., "--plan", "-p", help="JSON file with transport plan [N x M]"),
    coupling_threshold: float = typer.Option(0.001, "--threshold", help="Minimum coupling to consider"),
    normalize: bool = typer.Option(True, "--normalize/--no-normalize", is_flag=True, flag_value=True, help="Normalize transport plan rows"),
    blend_alpha: float = typer.Option(0.5, "--alpha", "-a", help="Blend factor with target (0 = transport-only)"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output file for merged weights"),
):
    """
    Merge weights using a transport plan.

    Uses the transport plan pi[i,j] to guide weighted averaging:
    W_merged[j,:] = sum_i pi[i,j] * W_source[i,:]
    """
    context = _context(ctx)
    service = GeometryTransportService()

    source = json.loads(Path(source_file).read_text())
    target = json.loads(Path(target_file).read_text())
    plan = json.loads(Path(plan_file).read_text())

    merged = service.synthesize_weights(
        source_weights=source,
        target_weights=target,
        transport_plan=plan,
        coupling_threshold=coupling_threshold,
        normalize_rows=normalize,
        blend_alpha=blend_alpha,
    )

    if merged is None:
        write_error(
            ErrorDetail(
                code="MC-4012",
                message="Failed to merge weights",
                detail="Invalid dimensions or empty input",
            ),
            context.output_format,
        )
        raise typer.Exit(1)

    if output:
        Path(output).write_text(json.dumps(merged))

    payload = {
        "mergedShape": [len(merged), len(merged[0]) if merged else 0],
        "blendAlpha": blend_alpha,
        "couplingThreshold": coupling_threshold,
        "outputFile": str(output) if output else None,
        "nextActions": [
            "mc geometry gromov-wasserstein to compute transport plan",
            "mc model merge for full model merging",
        ],
    }

    if context.output_format == "text":
        lines = [
            "TRANSPORT-GUIDED MERGE",
            f"Source: {len(source)} x {len(source[0]) if source else 0}",
            f"Target: {len(target)} x {len(target[0]) if target else 0}",
            f"Merged: {len(merged)} x {len(merged[0]) if merged else 0}",
            f"Blend Alpha: {blend_alpha}",
        ]
        if output:
            lines.append(f"Output: {output}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("synthesize")
def geometry_transport_synthesize(
    ctx: typer.Context,
    source_act_file: Path = typer.Option(..., "--source-act", help="JSON file with source activations"),
    target_act_file: Path = typer.Option(..., "--target-act", help="JSON file with target activations"),
    source_weights_file: Path = typer.Option(..., "--source-weights", help="JSON file with source weights"),
    target_weights_file: Path = typer.Option(..., "--target-weights", help="JSON file with target weights"),
    coupling_threshold: float = typer.Option(0.001, "--threshold", help="Minimum coupling to consider"),
    blend_alpha: float = typer.Option(0.5, "--alpha", "-a", help="Blend factor with target"),
    gw_epsilon: float = typer.Option(0.05, "--epsilon", "-e", help="GW entropic regularization"),
    gw_iterations: int = typer.Option(50, "--iterations", "-i", help="Max GW iterations"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output file for merged weights"),
):
    """
    Compute GW transport plan and synthesize merged weights.

    Computes pairwise distances from activations, solves for optimal
    transport using Gromov-Wasserstein, then applies transport-guided merging.
    """
    context = _context(ctx)
    service = GeometryTransportService()

    source_act = json.loads(Path(source_act_file).read_text())
    target_act = json.loads(Path(target_act_file).read_text())
    source_weights = json.loads(Path(source_weights_file).read_text())
    target_weights = json.loads(Path(target_weights_file).read_text())

    config = MergeConfig(
        coupling_threshold=coupling_threshold,
        blend_alpha=blend_alpha,
        gw_epsilon=gw_epsilon,
        gw_max_iterations=gw_iterations,
    )

    result = service.synthesize_with_gw(
        source_activations=source_act,
        target_activations=target_act,
        source_weights=source_weights,
        target_weights=target_weights,
        config=config,
    )

    if result is None:
        write_error(
            ErrorDetail(
                code="MC-4013",
                message="Failed to synthesize with GW",
                detail="Insufficient samples or dimension mismatch",
            ),
            context.output_format,
        )
        raise typer.Exit(1)

    if output:
        Path(output).write_text(json.dumps(result.merged_weights))

    payload = service.merge_result_payload(result)
    payload["outputFile"] = str(output) if output else None
    payload["nextActions"] = [
        "mc geometry intrinsic-dimension to analyze merged space",
        "mc model merge for full model merging",
    ]

    if context.output_format == "text":
        lines = [
            "TRANSPORT-GUIDED SYNTHESIS",
            f"GW Distance: {result.gw_distance:.4f}",
            f"Marginal Error: {result.marginal_error:.4f}",
            f"Effective Rank: {result.effective_rank}",
            f"Converged: {'Yes' if result.converged else 'No'} ({result.iterations} iterations)",
            f"Merged Shape: {len(result.merged_weights)} x {len(result.merged_weights[0]) if result.merged_weights else 0}",
        ]
        if output:
            lines.append(f"Output: {output}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
