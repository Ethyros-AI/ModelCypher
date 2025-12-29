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

"""Number Theory Geometry CLI commands.

Explores whether prime number distribution has hidden geometric structure
visible through high-dimensional analysis techniques.

Mathematical Motivation:
    1. The zeros of the Riemann zeta function behave like eigenvalues of
       random Hermitian matrices (Montgomery's pair correlation conjecture).

    2. If primes have geometric structure, it should be detectable via:
       - Eigenvalue distribution of Gram matrices
       - Intrinsic dimension (TwoNN)
       - Topological fingerprinting (persistent homology)
       - Manifold curvature (Ollivier-Ricci)

    3. This provides a "pure signal" test case for geometric analysis
       before applying to neural network representations.

Commands:
    mc geometry number-theory spectral --n-primes 10000
    mc geometry number-theory topology --n-primes 5000
    mc geometry number-theory full-analysis --n-primes 10000
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output
from modelcypher.core.domain._backend import get_default_backend

app = typer.Typer(no_args_is_help=True)
logger = logging.getLogger(__name__)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("spectral")
def spectral_analysis(
    ctx: typer.Context,
    n_primes: int = typer.Option(1000, help="Number of primes to analyze"),
    embedding_dim: int = typer.Option(20, help="Time-delay embedding dimension"),
    delay: int = typer.Option(1, help="Time delay for embedding"),
    output_file: str = typer.Option(None, "--output-file", "-o", help="Save raw data to JSON"),
    seed: int = typer.Option(42, help="Random seed for baseline"),
) -> None:
    """Analyze spectral geometry of prime number gaps.

    Embeds prime gaps into high-dimensional space via time-delay embedding,
    then compares eigenvalue distribution to random baseline.

    If primes have hidden geometric structure, the eigenvalue spectrum
    should differ systematically from random.

    Examples:
        mc geometry number-theory spectral
        mc geometry number-theory spectral --n-primes 10000
        mc geometry number-theory spectral --embedding-dim 50 --output-file results.json
    """
    context = _context(ctx)

    from modelcypher.core.domain.geometry.prime_geometry import (
        analyze_prime_geometry,
        format_result,
    )

    typer.echo(f"Analyzing spectral geometry of {n_primes} primes...")
    typer.echo(f"Embedding: dim={embedding_dim}, delay={delay}")

    backend = get_default_backend()
    result = analyze_prime_geometry(
        n_primes=n_primes,
        embedding_dim=embedding_dim,
        delay=delay,
        backend=backend,
        seed=seed,
    )

    # Build JSON payload
    payload = {
        "_schema": "mc.geometry.number_theory.spectral.v1",
        "n_primes": result.prime_count,
        "embedding_dim": result.embedding_dim,
        "prime_spectrum": {
            "participation_ratio": result.prime_eigenvalues.participation_ratio,
            "spectral_entropy": result.prime_eigenvalues.spectral_entropy,
            "condition_number": result.prime_eigenvalues.condition_number,
            "top_k_ratio": result.prime_eigenvalues.top_k_ratio,
        },
        "random_spectrum": {
            "participation_ratio": result.random_eigenvalues.participation_ratio,
            "spectral_entropy": result.random_eigenvalues.spectral_entropy,
            "condition_number": result.random_eigenvalues.condition_number,
            "top_k_ratio": result.random_eigenvalues.top_k_ratio,
        },
        "comparison": {
            "wasserstein_distance": result.comparison.wasserstein_distance,
            "ks_statistic": result.comparison.ks_statistic,
            "participation_ratio_diff": result.comparison.participation_ratio_diff,
            "spectral_entropy_diff": result.comparison.spectral_entropy_diff,
        },
        "intrinsic_dimension": {
            "prime_gaps": result.prime_intrinsic_dim,
            "random_gaps": result.random_intrinsic_dim,
        },
        "cross_representation_cka": result.gap_to_position_cka,
    }

    # Save raw eigenvalues if requested
    if output_file:
        raw_data = {
            **payload,
            "prime_eigenvalues": backend.to_list(result.prime_eigenvalues.eigenvalues),
            "random_eigenvalues": backend.to_list(result.random_eigenvalues.eigenvalues),
        }
        Path(output_file).write_text(json.dumps(raw_data, indent=2))
        typer.echo(f"Saved raw data to {output_file}")

    if context.output_format == "text":
        write_output(format_result(result), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("topology")
def topology_analysis(
    ctx: typer.Context,
    n_primes: int = typer.Option(500, help="Number of primes (keep small for topology)"),
    embedding_dim: int = typer.Option(10, help="Time-delay embedding dimension"),
    max_dimension: int = typer.Option(2, help="Maximum homology dimension (0, 1, or 2)"),
) -> None:
    """Compute topological fingerprint of prime gap distribution.

    Uses persistent homology to detect topological features (connected
    components, loops, voids) in the prime gap embedding.

    Note: Topology computation is O(n^3) for cycle detection.
    Keep n_primes under 1000 for reasonable runtime.

    Examples:
        mc geometry number-theory topology
        mc geometry number-theory topology --n-primes 1000 --max-dimension 1
    """
    context = _context(ctx)

    from modelcypher.core.domain.geometry.prime_geometry import (
        generate_primes,
        generate_random_gaps,
        time_delay_embedding,
    )
    from modelcypher.core.domain.geometry.topological_fingerprint import (
        TopologicalFingerprinter,
    )

    typer.echo(f"Computing topology for {n_primes} primes...")

    backend = get_default_backend()

    # Generate primes and embed
    primes = generate_primes(n_primes, backend)
    prime_embedded = time_delay_embedding(primes.gaps, embedding_dim, backend=backend)

    # Random baseline
    mean_gap = float(backend.mean(primes.gaps))
    random_gaps = generate_random_gaps(primes.gap_count, mean_gap, backend)
    random_embedded = time_delay_embedding(random_gaps, embedding_dim, backend=backend)

    # Compute topology
    fingerprinter = TopologicalFingerprinter(backend)

    typer.echo("Computing prime gap topology...")
    prime_topo = fingerprinter.compute_fingerprint(prime_embedded, max_dimension=max_dimension)

    typer.echo("Computing random gap topology...")
    random_topo = fingerprinter.compute_fingerprint(random_embedded, max_dimension=max_dimension)

    # Compare
    comparison = fingerprinter.compare_fingerprints(prime_topo, random_topo)

    payload = {
        "_schema": "mc.geometry.number_theory.topology.v1",
        "n_primes": n_primes,
        "embedding_dim": embedding_dim,
        "max_dimension": max_dimension,
        "prime_topology": {
            "betti_0": prime_topo.betti_numbers[0] if len(prime_topo.betti_numbers) > 0 else 0,
            "betti_1": prime_topo.betti_numbers[1] if len(prime_topo.betti_numbers) > 1 else 0,
            "betti_2": prime_topo.betti_numbers[2] if len(prime_topo.betti_numbers) > 2 else 0,
            "n_features": len(prime_topo.persistence_pairs),
        },
        "random_topology": {
            "betti_0": random_topo.betti_numbers[0] if len(random_topo.betti_numbers) > 0 else 0,
            "betti_1": random_topo.betti_numbers[1] if len(random_topo.betti_numbers) > 1 else 0,
            "betti_2": random_topo.betti_numbers[2] if len(random_topo.betti_numbers) > 2 else 0,
            "n_features": len(random_topo.persistence_pairs),
        },
        "comparison": {
            "bottleneck_distance": comparison.bottleneck_distance,
            "wasserstein_distance": comparison.wasserstein_distance,
            "betti_difference": comparison.betti_difference,
            "similarity_score": comparison.similarity_score,
        },
    }

    if context.output_format == "text":
        lines = [
            "=" * 60,
            "TOPOLOGICAL FINGERPRINT ANALYSIS",
            "=" * 60,
            "",
            f"Primes analyzed: {n_primes}",
            f"Embedding dimension: {embedding_dim}",
            f"Max homology dimension: {max_dimension}",
            "",
            "--- BETTI NUMBERS ---",
            "",
            f"{'Dimension':<12} | {'Primes':<10} | {'Random':<10}",
            "-" * 40,
            f"{'β₀ (components)':<12} | {payload['prime_topology']['betti_0']:<10} | {payload['random_topology']['betti_0']:<10}",
            f"{'β₁ (loops)':<12} | {payload['prime_topology']['betti_1']:<10} | {payload['random_topology']['betti_1']:<10}",
            f"{'β₂ (voids)':<12} | {payload['prime_topology']['betti_2']:<10} | {payload['random_topology']['betti_2']:<10}",
            "",
            "--- COMPARISON ---",
            "",
            f"Bottleneck distance:  {comparison.bottleneck_distance:.4f}",
            f"Wasserstein distance: {comparison.wasserstein_distance:.4f}",
            f"Betti difference:     {comparison.betti_difference:.4f}",
            f"Similarity score:     {comparison.similarity_score:.4f}",
            "",
            "=" * 60,
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("curvature")
def curvature_analysis(
    ctx: typer.Context,
    n_primes: int = typer.Option(500, help="Number of primes"),
    embedding_dim: int = typer.Option(10, help="Time-delay embedding dimension"),
    k_neighbors: int = typer.Option(10, help="k for k-NN graph"),
) -> None:
    """Measure manifold curvature of prime gap distribution.

    Uses Ollivier-Ricci curvature to detect whether the prime gap
    manifold is positively curved (spherical), negatively curved
    (hyperbolic), or flat.

    Examples:
        mc geometry number-theory curvature
        mc geometry number-theory curvature --n-primes 1000
    """
    context = _context(ctx)

    from modelcypher.core.domain.geometry.manifold_curvature import OllivierRicci
    from modelcypher.core.domain.geometry.prime_geometry import (
        generate_primes,
        generate_random_gaps,
        time_delay_embedding,
    )

    typer.echo(f"Computing curvature for {n_primes} primes...")

    backend = get_default_backend()

    # Generate and embed
    primes = generate_primes(n_primes, backend)
    prime_embedded = time_delay_embedding(primes.gaps, embedding_dim, backend=backend)

    mean_gap = float(backend.mean(primes.gaps))
    random_gaps = generate_random_gaps(primes.gap_count, mean_gap, backend)
    random_embedded = time_delay_embedding(random_gaps, embedding_dim, backend=backend)

    # Compute curvature
    ricci = OllivierRicci(backend)

    typer.echo("Computing prime gap curvature...")
    prime_curv = ricci.compute(prime_embedded, k=k_neighbors)

    typer.echo("Computing random gap curvature...")
    random_curv = ricci.compute(random_embedded, k=k_neighbors)

    payload = {
        "_schema": "mc.geometry.number_theory.curvature.v1",
        "n_primes": n_primes,
        "embedding_dim": embedding_dim,
        "k_neighbors": k_neighbors,
        "prime_curvature": {
            "mean": prime_curv.mean_curvature,
            "std": prime_curv.std_curvature,
            "min": prime_curv.min_curvature,
            "max": prime_curv.max_curvature,
            "health": prime_curv.health.value,
        },
        "random_curvature": {
            "mean": random_curv.mean_curvature,
            "std": random_curv.std_curvature,
            "min": random_curv.min_curvature,
            "max": random_curv.max_curvature,
            "health": random_curv.health.value,
        },
    }

    if context.output_format == "text":
        lines = [
            "=" * 60,
            "MANIFOLD CURVATURE ANALYSIS",
            "=" * 60,
            "",
            f"Primes analyzed: {n_primes}",
            f"Embedding dimension: {embedding_dim}",
            f"k-NN neighbors: {k_neighbors}",
            "",
            "--- OLLIVIER-RICCI CURVATURE ---",
            "",
            f"{'Metric':<12} | {'Primes':<15} | {'Random':<15}",
            "-" * 50,
            f"{'Mean':<12} | {prime_curv.mean_curvature:>15.4f} | {random_curv.mean_curvature:>15.4f}",
            f"{'Std':<12} | {prime_curv.std_curvature:>15.4f} | {random_curv.std_curvature:>15.4f}",
            f"{'Min':<12} | {prime_curv.min_curvature:>15.4f} | {random_curv.min_curvature:>15.4f}",
            f"{'Max':<12} | {prime_curv.max_curvature:>15.4f} | {random_curv.max_curvature:>15.4f}",
            f"{'Health':<12} | {prime_curv.health.value:>15} | {random_curv.health.value:>15}",
            "",
            "Curvature interpretation:",
            "  Negative (< 0): Hyperbolic (spreading, information-rich)",
            "  Near zero: Flat (Euclidean-like)",
            "  Positive (> 0): Spherical (clustering, constrained)",
            "",
            "=" * 60,
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("sweep")
def parameter_sweep(
    ctx: typer.Context,
    max_primes: int = typer.Option(10000, help="Maximum number of primes"),
    steps: int = typer.Option(10, help="Number of steps in sweep"),
    output_file: str = typer.Option("prime_sweep.json", "-o", help="Output file"),
) -> None:
    """Sweep across prime counts to find scale-dependent structure.

    Tests whether prime structure becomes more or less visible at
    different scales (number of primes analyzed).

    Examples:
        mc geometry number-theory sweep --max-primes 50000 --steps 20
    """
    context = _context(ctx)

    from modelcypher.core.domain.geometry.prime_geometry import analyze_prime_geometry

    backend = get_default_backend()

    # Generate logarithmically-spaced prime counts
    import math

    prime_counts = [
        int(100 * (max_primes / 100) ** (i / (steps - 1))) for i in range(steps)
    ]
    prime_counts = sorted(set(prime_counts))  # Remove duplicates

    results = []
    for n in prime_counts:
        typer.echo(f"Analyzing {n} primes...")
        try:
            result = analyze_prime_geometry(n_primes=n, backend=backend)
            results.append(
                {
                    "n_primes": n,
                    "ks_statistic": result.comparison.ks_statistic,
                    "wasserstein": result.comparison.wasserstein_distance,
                    "prime_participation": result.prime_eigenvalues.participation_ratio,
                    "random_participation": result.random_eigenvalues.participation_ratio,
                    "prime_id": result.prime_intrinsic_dim,
                    "random_id": result.random_intrinsic_dim,
                }
            )
        except Exception as e:
            typer.echo(f"  Failed: {e}", err=True)

    # Save results
    Path(output_file).write_text(json.dumps(results, indent=2))
    typer.echo(f"Saved sweep results to {output_file}")

    # Summary
    if results:
        max_ks = max(r["ks_statistic"] for r in results)
        max_ks_n = next(r["n_primes"] for r in results if r["ks_statistic"] == max_ks)

        payload = {
            "_schema": "mc.geometry.number_theory.sweep.v1",
            "steps": len(results),
            "max_primes": max_primes,
            "peak_ks_statistic": max_ks,
            "peak_ks_n_primes": max_ks_n,
            "output_file": output_file,
        }

        if context.output_format == "text":
            lines = [
                "=" * 60,
                "PRIME GEOMETRY SWEEP COMPLETE",
                "=" * 60,
                "",
                f"Steps completed: {len(results)}",
                f"Max primes: {max_primes}",
                "",
                f"Peak KS statistic: {max_ks:.4f} at n={max_ks_n}",
                "",
                f"Full results saved to: {output_file}",
                "",
                "=" * 60,
            ]
            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)
