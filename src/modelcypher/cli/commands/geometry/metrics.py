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

"""Geometry metrics CLI commands.

Provides commands for geometric analysis of model representations
with no user-configurable parameters. Inputs are point clouds only.

Commands:
    mc geometry metrics gromov-wasserstein <source_file> <target_file>
    mc geometry metrics intrinsic-dimension <points_file>
    mc geometry metrics topological-fingerprint <points_file>
    mc geometry metrics spectral-signature <points_file>
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output
from modelcypher.core.use_cases.geometry_metrics_service import GeometryMetricsService

app = typer.Typer(no_args_is_help=True)

def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("gromov-wasserstein")
def geometry_metrics_gromov_wasserstein(
    ctx: typer.Context,
    source_file: str = typer.Argument(
        ..., help="Path to source point cloud (JSON array of arrays)"
    ),
    target_file: str = typer.Argument(
        ..., help="Path to target point cloud (JSON array of arrays)"
    ),
) -> None:
    """Compute Gromov-Wasserstein distance between two point clouds."""
    source_points = json.loads(Path(source_file).read_text())
    target_points = json.loads(Path(target_file).read_text())

    service = GeometryMetricsService()
    result = service.compute_gromov_wasserstein(
        source_points=source_points,
        target_points=target_points,
    )

    context = _context(ctx)
    payload = service.gromov_wasserstein_payload(result)
    payload["_schema"] = "mc.geometry.gromov_wasserstein.v1"
    write_output(payload, context.output_format, context.pretty)


@app.command("intrinsic-dimension")
def geometry_metrics_intrinsic_dimension(
    ctx: typer.Context,
    points_file: str = typer.Argument(
        ..., help="Path to point cloud (JSON array of arrays or activations dict)"
    ),
) -> None:
    """Estimate intrinsic dimension of a point cloud using TwoNN."""
    raw_points = json.loads(Path(points_file).read_text())
    if isinstance(raw_points, dict):
        points = [raw_points[key] for key in sorted(raw_points.keys())]
    else:
        points = raw_points

    service = GeometryMetricsService()
    result = service.estimate_intrinsic_dimension(points=points)

    context = _context(ctx)
    payload = service.intrinsic_dimension_payload(result)
    payload["_schema"] = "mc.geometry.intrinsic_dimension.v1"
    write_output(payload, context.output_format, context.pretty)


@app.command("topological-fingerprint")
def geometry_metrics_topological_fingerprint(
    ctx: typer.Context,
    points_file: str = typer.Argument(..., help="Path to point cloud (JSON array of arrays)"),
) -> None:
    """Compute topological fingerprint using persistent homology."""
    points = json.loads(Path(points_file).read_text())

    service = GeometryMetricsService()
    result = service.compute_topological_fingerprint(points=points)

    context = _context(ctx)
    payload = service.topological_fingerprint_payload(result)
    payload["_schema"] = "mc.geometry.topological_fingerprint.v1"
    write_output(payload, context.output_format, context.pretty)


@app.command("spectral-signature")
def geometry_metrics_spectral_signature(
    ctx: typer.Context,
    points_file: str = typer.Argument(..., help="Path to point cloud (JSON array of arrays)"),
) -> None:
    """Compute spectral signature of a point cloud."""
    points = json.loads(Path(points_file).read_text())

    service = GeometryMetricsService()
    result = service.compute_spectral_signature(points=points)

    context = _context(ctx)
    payload = service.spectral_signature_payload(result)
    payload["_schema"] = "mc.geometry.spectral_signature.v1"
    write_output(payload, context.output_format, context.pretty)
