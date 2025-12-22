"""Geometry metrics CLI commands.

Provides commands for geometric analysis of model representations,
including Gromov-Wasserstein distance, intrinsic dimension estimation,
and topological fingerprinting.

Commands:
    mc geometry metrics gromov-wasserstein <source_file> <target_file>
    mc geometry metrics intrinsic-dimension <points_file>
    mc geometry metrics topological-fingerprint <points_file>
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
    source_file: str = typer.Argument(..., help="Path to source point cloud (JSON array of arrays)"),
    target_file: str = typer.Argument(..., help="Path to target point cloud (JSON array of arrays)"),
    epsilon: float = typer.Option(0.05, "--epsilon", help="Entropic regularization parameter"),
    max_iterations: int = typer.Option(50, "--max-iterations", help="Maximum outer iterations"),
) -> None:
    """
    Compute Gromov-Wasserstein distance between two point clouds.

    Measures structural similarity of representation spaces without requiring
    point-to-point correspondence. Lower distance = more similar structure.

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
            f"Compatibility Score: {result.compatibility_score:.4f}",
            f"Converged: {'Yes' if result.converged else 'No'}",
            f"Iterations: {result.iterations}",
            f"Coupling Shape: {result.coupling_shape[0]} x {result.coupling_shape[1]}",
            "",
            "Interpretation:",
            result.interpretation,
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("intrinsic-dimension")
def geometry_metrics_intrinsic_dimension(
    ctx: typer.Context,
    points_file: str = typer.Argument(..., help="Path to point cloud (JSON array of arrays)"),
    use_regression: bool = typer.Option(True, "--use-regression/--no-use-regression", help="Use regression method vs maximum likelihood"),
    bootstrap_samples: int = typer.Option(200, "--bootstrap", help="Number of bootstrap samples for confidence intervals"),
) -> None:
    """
    Estimate intrinsic dimension of a point cloud using TwoNN.

    Reveals effective degrees of freedom in representation space.
    Low dimension = compressed/structured, high dimension = rich/complex.

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
            "",
            "Interpretation:",
            result.interpretation,
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("topological-fingerprint")
def geometry_metrics_topological_fingerprint(
    ctx: typer.Context,
    points_file: str = typer.Argument(..., help="Path to point cloud (JSON array of arrays)"),
    max_dimension: int = typer.Option(1, "--max-dim", help="Maximum homology dimension (0=components, 1=loops)"),
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
            "",
            "Interpretation:",
            result.interpretation,
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
