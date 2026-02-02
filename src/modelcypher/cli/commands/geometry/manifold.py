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

"""Geometry manifold analysis CLI commands.

Provides commands for clustering and analyzing manifold structure
of model representations.

Commands:
    mc geometry manifold cluster --points <file>
    mc geometry manifold dimension --points <file>
    mc geometry manifold query --point <file> --regions <file>
    mc geometry manifold token-cka --source <file> --target <file>
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output
from modelcypher.core.use_cases.geometry_persona_service import GeometryPersonaService

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("cluster")
def geometry_manifold_cluster(
    ctx: typer.Context,
    points_file: Path = typer.Option(..., "--points", "-p", help="JSON file with manifold points"),
    compute_dimension: bool = typer.Option(
        True,
        "--dimension/--no-dimension",
        help="Compute intrinsic dimension",
    ),
):
    """
    Cluster manifold points into regions using DBSCAN.

    DBSCAN epsilon is derived from the geometry of the point cloud.
    No user parameters.

    Points should have entropy and gate features from thermo measurements.
    """
    context = _context(ctx)
    service = GeometryPersonaService()

    points = json.loads(Path(points_file).read_text())
    result = service.cluster_points(
        points=points,
        # epsilon uses service defaults (derived from geometry)
        compute_dimension=compute_dimension,
    )

    payload = service.clustering_payload(result)

    if context.output_format == "text":
        lines = [
            "MANIFOLD CLUSTERING",
            f"Regions: {len(result.regions)}",
            f"Noise Points: {len(result.noise_points)}",
            f"New Clusters: {result.new_clusters_formed}",
            "",
        ]
        for region in result.regions:
            lines.append(f"  Region {str(region.id)[:8]}:")
            lines.append(f"    Type: {region.region_type.value}")
            lines.append(f"    Members: {region.member_count}")
            if region.intrinsic_dimension is not None:
                lines.append(f"    Dimension: {region.intrinsic_dimension:.2f}")
            lines.append(f"    Dominant Gates: {', '.join(region.dominant_gates)}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("dimension")
def geometry_manifold_dimension(
    ctx: typer.Context,
    points_file: Path = typer.Option(..., "--points", "-p", help="JSON file with point vectors"),
):
    """
    Estimate intrinsic dimension of a point cloud using TwoNN.

    All parameters are derived from data - no configuration needed.
    """
    context = _context(ctx)
    service = GeometryPersonaService()

    points = json.loads(Path(points_file).read_text())
    result = service.estimate_dimension(points=points)

    payload = service.dimension_payload(result)

    if context.output_format == "text":
        lines = [
            "INTRINSIC DIMENSION ESTIMATE",
            f"Dimension: {result.intrinsic_dimension:.2f}",
        ]
        if result.ci95_lower is not None and result.ci95_upper is not None:
            lines.append(f"95% CI: [{result.ci95_lower:.2f}, {result.ci95_upper:.2f}]")
        lines.append(f"Samples: {result.sample_count} ({result.usable_count} usable)")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("query")
def geometry_manifold_query(
    ctx: typer.Context,
    point_file: Path = typer.Option(..., "--point", "-p", help="JSON file with point to query"),
    regions_file: Path = typer.Option(..., "--regions", "-r", help="JSON file with regions"),
):
    """
    Query which region a point belongs to.

    Distance threshold is derived from the region geometry. No user parameters.
    """
    context = _context(ctx)
    service = GeometryPersonaService()

    point = json.loads(Path(point_file).read_text())
    regions = json.loads(Path(regions_file).read_text())

    result = service.query_region(
        point=point,
        regions=regions,
        # epsilon uses service default (derived from region geometry)
    )

    payload = service.region_query_payload(result)

    if context.output_format == "text":
        lines = [
            "REGION QUERY RESULT",
            f"Suggested Type: {result.suggested_character.value}",
            f"Within Region: {'Yes' if result.is_within_region else 'No'}",
            f"Distance: {result.distance:.4f}",
            f"Confidence: {result.confidence:.2%}",
        ]
        if result.nearest_region:
            lines.append(
                f"Nearest Region: {str(result.nearest_region.id)[:8]} ({result.nearest_region.region_type.value})"
            )
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("token-cka")
def geometry_token_cka(
    ctx: typer.Context,
    source_file: Path = typer.Option(..., "--source", "-s", help="JSONL file with source activations"),
    target_file: Path = typer.Option(..., "--target", "-t", help="JSONL file with target activations"),
    alignment: str = typer.Option("truncate", "--alignment", "-a", help="Alignment method: truncate, pad, dtw"),
):
    """
    Compute CKA at token-level with text boundary awareness.

    Reads two JSONL files with activations and computes CKA both
    aggregated across all tokens and per-text.

    Input files should contain JSONL records with:
    - "activations": List of activation vectors for each token
    - "text": (optional) Original text for reference

    Implements token-level analysis from arXiv:2601.21571v1.

    Examples:
        mc geometry manifold token-cka --source model_a.jsonl --target model_b.jsonl
        mc geometry manifold token-cka -s source.jsonl -t target.jsonl --alignment dtw
    """
    context = _context(ctx)

    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain.geometry.cka import compute_token_cka

    backend = initialize_default_backend()

    # Load source activations
    source_acts = []
    source_lengths = []
    with open(source_file, "r") as f:
        for line in f:
            record = json.loads(line)
            acts = record.get("activations", [])
            source_acts.extend(acts)
            source_lengths.append(len(acts))

    # Load target activations
    target_acts = []
    target_lengths = []
    with open(target_file, "r") as f:
        for line in f:
            record = json.loads(line)
            acts = record.get("activations", [])
            target_acts.extend(acts)
            target_lengths.append(len(acts))

    if not source_acts or not target_acts:
        write_output("Error: Empty activation files", context.output_format, context.pretty)
        return

    if len(source_lengths) != len(target_lengths):
        write_output(
            f"Error: Number of texts must match: {len(source_lengths)} vs {len(target_lengths)}",
            context.output_format,
            context.pretty,
        )
        return

    source_arr = backend.array(source_acts)
    target_arr = backend.array(target_acts)

    result = compute_token_cka(
        activations_x=source_arr,
        activations_y=target_arr,
        text_lengths_x=source_lengths,
        text_lengths_y=target_lengths,
        backend=backend,
        alignment=alignment,
    )

    payload = {
        "aggregateCka": result.aggregate_cka,
        "meanTextCka": result.mean_text_cka,
        "minTextCka": result.min_text_cka,
        "maxTextCka": result.max_text_cka,
        "textCount": len(result.per_text_cka),
        "perTextCka": result.per_text_cka[:10],  # First 10 for brevity
        "alignment": alignment,
    }

    if context.output_format == "text":
        lines = [
            "TOKEN-LEVEL CKA ANALYSIS",
            f"Alignment: {alignment}",
            "",
            f"Aggregate CKA: {result.aggregate_cka:.4f}",
            f"Mean per-text CKA: {result.mean_text_cka:.4f}",
            f"Min per-text CKA: {result.min_text_cka:.4f}",
            f"Max per-text CKA: {result.max_text_cka:.4f}",
            f"Texts analyzed: {len(result.per_text_cka)}",
        ]
        if len(result.per_text_cka) <= 10:
            lines.append("")
            lines.append("Per-text CKA:")
            for i, cka in enumerate(result.per_text_cka):
                lines.append(f"  Text {i}: {cka:.4f}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
