"""Geometry sparse region CLI commands.

Provides commands for analyzing sparse regions in model representations
for targeted LoRA injection.

Commands:
    mc geometry sparse domains
    mc geometry sparse locate <domain_stats_file> <baseline_stats_file>
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output
from modelcypher.core.use_cases.geometry_sparse_service import GeometrySparseService

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("domains")
def geometry_sparse_domains(ctx: typer.Context) -> None:
    """List all built-in sparse region domains."""
    context = _context(ctx)
    service = GeometrySparseService()
    domains = service.list_domains()
    payload = service.domains_payload(domains)

    if context.output_format == "text":
        lines = [
            "SPARSE REGION DOMAINS",
            f"Total: {payload['count']}",
            "",
        ]
        for d in payload["domains"]:
            range_str = ""
            if d["expectedLayerRange"]:
                range_str = f" (layers {d['expectedLayerRange'][0]:.0%}-{d['expectedLayerRange'][1]:.0%})"
            lines.append(f"  {d['name']}: {d['description']}{range_str}")
            lines.append(f"    Category: {d['category']}, Probes: {d['probeCount']}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("locate")
def geometry_sparse_locate(
    ctx: typer.Context,
    domain_stats_file: str = typer.Argument(..., help="Path to domain layer stats JSON"),
    baseline_stats_file: str = typer.Argument(..., help="Path to baseline layer stats JSON"),
    domain_name: str = typer.Option("unknown", "--domain", help="Domain name"),
    base_rank: int = typer.Option(16, "--rank", help="Base LoRA rank"),
    sparsity_threshold: float = typer.Option(0.3, "--threshold", help="Sparsity threshold"),
) -> None:
    """
    Locate sparse regions for LoRA injection.

    Input files should contain JSON arrays of layer stats:
    [{"layer_index": 0, "mean_activation": 0.5, ...}, ...]
    """
    context = _context(ctx)
    service = GeometrySparseService()

    domain_stats = json.loads(Path(domain_stats_file).read_text())
    baseline_stats = json.loads(Path(baseline_stats_file).read_text())

    result = service.locate_sparse_regions(
        domain_stats=domain_stats,
        baseline_stats=baseline_stats,
        domain_name=domain_name,
        base_rank=base_rank,
        sparsity_threshold=sparsity_threshold,
    )

    payload = service.analysis_payload(result)
    payload["nextActions"] = [
        "mc geometry sparse domains to see available domain definitions",
        "mc geometry adapter sparsity for DARE analysis",
    ]

    if context.output_format == "text":
        lines = [
            "SPARSE REGION ANALYSIS",
            f"Domain: {result.domain}",
            f"Sparse Layers: {len(result.sparse_layers)} {result.sparse_layers}",
            f"Skip Layers: {len(result.skip_layers)} {result.skip_layers}",
            "",
            "LORA RECOMMENDATION",
            f"  Quality: {result.recommendation.quality.value.upper()}",
            f"  Overall Rank: {result.recommendation.overall_rank}",
            f"  Alpha: {result.recommendation.alpha}",
            f"  Preservation: {result.recommendation.estimated_preservation:.0%}",
            "",
            result.recommendation.rationale,
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
