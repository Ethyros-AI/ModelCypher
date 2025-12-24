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
    epsilon: float = typer.Option(0.3, "--epsilon", "-e", help="DBSCAN epsilon (distance threshold)"),
    min_points: int = typer.Option(5, "--min-points", "-m", help="Minimum points per cluster"),
    compute_dimension: bool = typer.Option(True, "--dimension/--no-dimension", is_flag=True, flag_value=True, help="Compute intrinsic dimension"),
):
    """
    Cluster manifold points into regions using DBSCAN.

    Points should have entropy and gate features from thermo measurements.
    """
    context = _context(ctx)
    service = GeometryPersonaService()

    points = json.loads(Path(points_file).read_text())
    result = service.cluster_points(
        points=points,
        epsilon=epsilon,
        min_points=min_points,
        compute_dimension=compute_dimension,
    )

    payload = service.clustering_payload(result)
    payload["nextActions"] = [
        "mc geometry manifold dimension to estimate dimensionality",
        "mc geometry manifold query to classify new points",
    ]

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
    bootstrap: int = typer.Option(0, "--bootstrap", "-b", help="Bootstrap samples (0 = none)"),
    regression: bool = typer.Option(True, "--regression/--no-regression", is_flag=True, flag_value=True, help="Use regression-based estimation"),
):
    """
    Estimate intrinsic dimension of a point cloud using TwoNN.
    """
    context = _context(ctx)
    service = GeometryPersonaService()

    points = json.loads(Path(points_file).read_text())
    result = service.estimate_dimension(
        points=points,
        bootstrap_samples=bootstrap,
        use_regression=regression,
    )

    payload = service.dimension_payload(result)
    payload["nextActions"] = [
        "mc geometry intrinsic-dimension for alternative estimation",
        "mc geometry manifold cluster to find regions",
    ]

    if context.output_format == "text":
        lines = [
            "INTRINSIC DIMENSION ESTIMATE",
            f"Dimension: {result.intrinsic_dimension:.2f}",
        ]
        if result.ci95_lower is not None and result.ci95_upper is not None:
            lines.append(f"95% CI: [{result.ci95_lower:.2f}, {result.ci95_upper:.2f}]")
        lines.append(f"Samples: {result.sample_count} ({result.usable_count} usable)")
        lines.append(f"Method: {'Regression' if result.uses_regression else 'MLE'}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("query")
def geometry_manifold_query(
    ctx: typer.Context,
    point_file: Path = typer.Option(..., "--point", "-p", help="JSON file with point to query"),
    regions_file: Path = typer.Option(..., "--regions", "-r", help="JSON file with regions"),
    epsilon: float = typer.Option(0.3, "--epsilon", "-e", help="Distance threshold"),
):
    """
    Query which region a point belongs to.
    """
    context = _context(ctx)
    service = GeometryPersonaService()

    point = json.loads(Path(point_file).read_text())
    regions = json.loads(Path(regions_file).read_text())

    result = service.query_region(
        point=point,
        regions=regions,
        epsilon=epsilon,
    )

    payload = service.region_query_payload(result)
    payload["nextActions"] = [
        "mc geometry manifold cluster to update clusters",
        "mc thermo measure to get point features",
    ]

    if context.output_format == "text":
        lines = [
            "REGION QUERY RESULT",
            f"Suggested Type: {result.suggested_type.value}",
            f"Within Region: {'Yes' if result.is_within_region else 'No'}",
            f"Distance: {result.distance:.4f}",
            f"Confidence: {result.confidence:.2%}",
        ]
        if result.nearest_region:
            lines.append(f"Nearest Region: {str(result.nearest_region.id)[:8]} ({result.nearest_region.region_type.value})")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
