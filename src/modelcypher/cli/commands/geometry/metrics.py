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

Provides commands for geometric analysis of model representations,
including Gromov-Wasserstein distance, intrinsic dimension estimation,
and topological fingerprinting.

Commands:
    mc geometry metrics gromov-wasserstein <source_file> <target_file>
    mc geometry metrics intrinsic-dimension <points_file>
    mc geometry metrics topological-fingerprint <points_file>
    mc geometry metrics spectral-signature <points_file>
    mc geometry metrics dimension-constraint <points_file> --pad-dim <n>
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
    epsilon: float = typer.Option(0.05, "--epsilon", help="Entropic regularization parameter"),
    max_iterations: int = typer.Option(50, "--max-iterations", help="Maximum outer iterations"),
) -> None:
    """
    Compute Gromov-Wasserstein distance between two point clouds.

    Measures structural similarity of representation spaces without requiring
    point-to-point correspondence. Smaller values indicate closer structure
    under this metric.

    Input files should contain JSON arrays of point arrays, e.g.:
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], ...]
    """
    context = _context(ctx)

    # Load point clouds
    source_points = json.loads(Path(source_file).read_text())
    target_points = json.loads(Path(target_file).read_text())

    service = GeometryMetricsService()
    result = service.compute_gromov_wasserstein(
        source_points=source_points,
        target_points=target_points,
        epsilon=epsilon,
        max_iterations=max_iterations,
    )

    payload = service.gromov_wasserstein_payload(result)

    if context.output_format == "text":
        lines = [
            "GROMOV-WASSERSTEIN DISTANCE",
            "",
            f"Distance: {result.distance:.6f}",
            f"Normalized Distance: {result.normalized_distance:.4f}",
            f"Alignment Score: {result.alignment_score:.4f}",
            f"Converged: {'Yes' if result.converged else 'No'}",
            f"Iterations: {result.iterations}",
            f"Coupling Shape: {result.coupling_shape[0]} x {result.coupling_shape[1]}",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("intrinsic-dimension")
def geometry_metrics_intrinsic_dimension(
    ctx: typer.Context,
    points_file: str = typer.Argument(
        ..., help="Path to point cloud (JSON array of arrays or activations dict)"
    ),
    use_regression: bool = typer.Option(
        True,
        "--use-regression/--no-use-regression",
        is_flag=True,
        flag_value=True,
        help="Use regression method vs maximum likelihood",
    ),
    bootstrap_samples: int = typer.Option(
        200, "--bootstrap", help="Number of bootstrap samples for confidence intervals"
    ),
) -> None:
    """
    Estimate intrinsic dimension of a point cloud using TwoNN.

    Reveals effective degrees of freedom in representation space.
    Lower values indicate fewer effective degrees of freedom for the sample.

    Input file should contain JSON array of point arrays.
    """
    context = _context(ctx)

    points = json.loads(Path(points_file).read_text())

    service = GeometryMetricsService()
    result = service.estimate_intrinsic_dimension(
        points=points,
        use_regression=use_regression,
        bootstrap_samples=bootstrap_samples,
    )

    payload = service.intrinsic_dimension_payload(result)

    if context.output_format == "text":
        lines = [
            "INTRINSIC DIMENSION ESTIMATION",
            "",
            f"Dimension: {result.dimension:.2f}",
            f"95% CI: [{result.confidence_lower:.2f}, {result.confidence_upper:.2f}]",
            f"Sample Count: {result.sample_count}",
            f"Method: {result.method}",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("topological-fingerprint")
def geometry_metrics_topological_fingerprint(
    ctx: typer.Context,
    points_file: str = typer.Argument(..., help="Path to point cloud (JSON array of arrays)"),
    max_dimension: int = typer.Option(
        1, "--max-dim", help="Maximum homology dimension (0=components, 1=loops)"
    ),
    num_steps: int = typer.Option(50, "--steps", help="Number of filtration steps"),
) -> None:
    """
    Compute topological fingerprint using persistent homology.

    Reveals the shape of the representation manifold:
    - Betti-0: Connected components (clusters)
    - Betti-1: Loops/holes (cyclic structure)
    - Persistence: Feature stability

    Input file should contain JSON array of point arrays.
    """
    context = _context(ctx)

    points = json.loads(Path(points_file).read_text())

    service = GeometryMetricsService()
    result = service.compute_topological_fingerprint(
        points=points,
        max_dimension=max_dimension,
        num_steps=num_steps,
    )

    payload = service.topological_fingerprint_payload(result)

    if context.output_format == "text":
        lines = [
            "TOPOLOGICAL FINGERPRINT",
            "",
            f"Betti-0 (Components): {result.betti_0}",
            f"Betti-1 (Loops): {result.betti_1}",
            f"Persistence Entropy: {result.persistence_entropy:.4f}",
            f"Total Persistence: {result.total_persistence:.4f}",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("spectral-signature")
def geometry_metrics_spectral_signature(
    ctx: typer.Context,
    points_file: str = typer.Argument(..., help="Path to point cloud (JSON array of arrays)"),
    k_neighbors: int | None = typer.Option(
        None, "--k-neighbors", help="k for geodesic k-NN graph construction"
    ),
    kernel_bandwidth: float | None = typer.Option(
        None, "--kernel-bandwidth", help="Gaussian kernel bandwidth (sigma)"
    ),
    normalized: bool = typer.Option(
        True,
        "--normalized/--unnormalized",
        help="Use normalized Laplacian for spectral signature",
    ),
    heat_times: list[float] | None = typer.Option(
        None, "--heat-time", help="Heat trace time (repeatable)"
    ),
    max_eigenvalues: int | None = typer.Option(
        None, "--max-eigenvalues", help="Maximum eigenvalues to include in output"
    ),
) -> None:
    """
    Compute spectral signature of a point cloud.

    Builds a k-NN graph (local geodesic edges), constructs a Laplacian,
    and reports eigenvalues and heat trace as raw spectral measurements.
    """
    context = _context(ctx)

    points = json.loads(Path(points_file).read_text())

    service = GeometryMetricsService()
    result = service.compute_spectral_signature(
        points=points,
        k_neighbors=k_neighbors,
        kernel_bandwidth=kernel_bandwidth,
        normalized_laplacian=normalized,
        heat_times=heat_times,
    )

    payload = service.spectral_signature_payload(result, max_eigenvalues=max_eigenvalues)
    payload["_schema"] = "mc.geometry.spectral_signature.v1"

    if context.output_format == "text":
        lines = [
            "SPECTRAL SIGNATURE",
            "",
            f"Nodes: {result.node_count}",
            f"Edges: {result.edge_count}",
            f"k-Neighbors: {result.k_neighbors}",
            f"Kernel Bandwidth: {result.kernel_bandwidth:.6f}",
            f"Normalized Laplacian: {'Yes' if result.normalized_laplacian else 'No'}",
            f"Connected: {'Yes' if result.connected else 'No'}",
            f"Component Count: {result.component_count}",
            f"Algebraic Connectivity: {result.algebraic_connectivity:.6f}",
            f"Spectral Entropy: {result.spectral_entropy:.6f}",
            "Heat Trace:",
        ]
        for t, value in zip(payload["heatTimes"], payload["heatTrace"]):
            lines.append(f"  t={t}: {value:.6f}")
        eigenvalues = payload["eigenvalues"]
        lines.append(
            f"Eigenvalues (total={payload['eigenvalueCount']}, shown={len(eigenvalues)}): {eigenvalues}"
        )
        if payload.get("eigenvaluesTruncated"):
            lines.append("Eigenvalues truncated for output.")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("dimension-constraint")
def geometry_metrics_dimension_constraint(
    ctx: typer.Context,
    points_file: str = typer.Argument(..., help="Path to point cloud (JSON array of arrays)"),
    pad_dim: int = typer.Option(..., "--pad-dim", help="Target padded dimension"),
    k_neighbors: int | None = typer.Option(
        None, "--k-neighbors", help="k for geodesic k-NN graph construction"
    ),
    heat_times: list[float] | None = typer.Option(
        None, "--heat-time", help="Heat trace time (repeatable)"
    ),
) -> None:
    """
    Measure invariance under zero-padding dimension constraints.

    Compares geometry in the base dimension to the same points padded with
    zero coordinates (e.g., 2D -> 3D -> 4D).

    Accepts either a JSON array of point vectors or a JSON dict of activation
    vectors (values are treated as points in sorted key order).
    """
    context = _context(ctx)

    raw_points = json.loads(Path(points_file).read_text())
    if isinstance(raw_points, dict):
        points = [raw_points[key] for key in sorted(raw_points.keys())]
    else:
        points = raw_points
    if not points:
        raise typer.BadParameter("Point cloud is empty.")

    base_dim = len(points[0])
    for row in points:
        if len(row) != base_dim:
            raise typer.BadParameter("All points must share the same dimension.")
    if pad_dim < base_dim:
        raise typer.BadParameter("pad-dim must be >= base dimension.")

    service = GeometryMetricsService()
    result = service.compute_dimension_constraint_invariance(
        points=points,
        padded_dimension=pad_dim,
        k_neighbors=k_neighbors,
        heat_times=heat_times,
    )
    payload = service.dimension_constraint_invariance_payload(result)
    payload["_schema"] = "mc.geometry.dimension_constraint_invariance.v1"

    if context.output_format == "text":
        lines = [
            "DIMENSION-CONSTRAINT INVARIANCE",
            "",
            f"Base Dimension: {result.base_dimension}",
            f"Padded Dimension: {result.padded_dimension}",
            f"Sample Count: {result.sample_count}",
            f"k-Neighbors: {result.k_neighbors}",
            f"Gram CKA: {result.gram_cka:.6f}",
            "",
            "Geodesic Distance Diff:",
            f"  Mean |Δ|: {result.geodesic_mean_abs_diff:.6e}",
            f"  Max |Δ|: {result.geodesic_max_abs_diff:.6e}",
            "",
            "Spectral Signature Diff:",
            f"  Eigen Mean |Δ|: {result.spectral_eigen_mean_abs_diff:.6e}",
            f"  Eigen Max |Δ|: {result.spectral_eigen_max_abs_diff:.6e}",
            f"  Spectral Entropy Base: {result.spectral_entropy_base:.6f}",
            f"  Spectral Entropy Padded: {result.spectral_entropy_padded:.6f}",
            "Heat Trace:",
        ]
        for t, base_val, pad_val in zip(
            result.heat_times, result.heat_trace_base, result.heat_trace_padded
        ):
            lines.append(f"  t={t}: base={base_val:.6f} padded={pad_val:.6f}")
        lines.extend(
            [
                "",
                "Topology:",
                f"  Betti Base: {result.betti_numbers_base}",
                f"  Betti Padded: {result.betti_numbers_padded}",
                f"  Components Base: {result.component_count_base}",
                f"  Components Padded: {result.component_count_padded}",
                f"  Cycles Base: {result.cycle_count_base}",
                f"  Cycles Padded: {result.cycle_count_padded}",
                f"  Persistence Entropy Base: {result.persistence_entropy_base:.6f}",
                f"  Persistence Entropy Padded: {result.persistence_entropy_padded:.6f}",
                f"  Max Persistence Base: {result.max_persistence_base:.6f}",
                f"  Max Persistence Padded: {result.max_persistence_padded:.6f}",
            ]
        )
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
