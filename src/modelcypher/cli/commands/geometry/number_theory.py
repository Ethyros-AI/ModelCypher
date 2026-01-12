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
from modelcypher.core.support.array_utils import array_to_list
from modelcypher.core.domain._backend import get_default_backend

app = typer.Typer(no_args_is_help=True)
logger = logging.getLogger(__name__)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("spectral")
def spectral_analysis(
    ctx: typer.Context,
    n_primes: int = typer.Option(1000, help="Number of primes to analyze"),
    embedding_dim: int = typer.Option(None, help="Time-delay embedding dimension (auto-derived via Takens' theorem if not specified)"),
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
    dim_str = str(embedding_dim) if embedding_dim is not None else "auto (Takens' theorem)"
    typer.echo(f"Embedding: dim={dim_str}, delay={delay}")

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
            "prime_eigenvalues": array_to_list(
                backend, result.prime_eigenvalues.eigenvalues
            ),
            "random_eigenvalues": array_to_list(
                backend, result.random_eigenvalues.eigenvalues
            ),
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
    embedding_dim: int = typer.Option(None, help="Time-delay embedding dimension (auto-derived via Takens' theorem if not specified)"),
) -> None:
    """Compute topological fingerprint of prime gap distribution.

    Uses persistent homology to detect topological features (connected
    components, loops) in the prime gap embedding.

    Note: Topology computation is O(n^3) for cycle detection.
    Keep n_primes under 1000 for reasonable runtime.

    Examples:
        mc geometry number-theory topology
        mc geometry number-theory topology --n-primes 1000
    """
    context = _context(ctx)

    from modelcypher.core.domain.geometry.prime_geometry import (
        _derive_embedding_dim,
        generate_primes,
        generate_random_gaps,
        time_delay_embedding,
    )
    from modelcypher.core.domain.geometry.topological_fingerprint import (
        BackendTopologicalFingerprint,
    )

    typer.echo(f"Computing topology for {n_primes} primes...")

    backend = get_default_backend()

    # Generate primes and embed
    primes = generate_primes(n_primes, backend)

    # Auto-derive embedding dimension if not specified
    if embedding_dim is None:
        embedding_dim = _derive_embedding_dim(primes.gaps, 1, backend)
        typer.echo(f"Auto-derived embedding dimension: {embedding_dim} (Takens' theorem)")

    prime_embedded = time_delay_embedding(primes.gaps, embedding_dim, backend=backend)

    # Random baseline
    mean_gap = float(backend.mean(primes.gaps))
    random_gaps = generate_random_gaps(primes.gap_count, mean_gap, backend)
    random_embedded = time_delay_embedding(random_gaps, embedding_dim, backend=backend)

    # Compute topology
    fingerprinter = BackendTopologicalFingerprint(backend)

    typer.echo("Computing prime gap topology...")
    prime_topo = fingerprinter.compute(backend.tolist(prime_embedded))

    typer.echo("Computing random gap topology...")
    random_topo = fingerprinter.compute(backend.tolist(random_embedded))

    # Compare
    comparison = fingerprinter.compare(prime_topo, random_topo)

    payload = {
        "_schema": "mc.geometry.number_theory.topology.v1",
        "n_primes": n_primes,
        "embedding_dim": embedding_dim,
        "prime_topology": {
            "betti_0": prime_topo.betti_numbers[0] if len(prime_topo.betti_numbers) > 0 else 0,
            "betti_1": prime_topo.betti_numbers[1] if len(prime_topo.betti_numbers) > 1 else 0,
            "n_features": len(prime_topo.diagram.points),
        },
        "random_topology": {
            "betti_0": random_topo.betti_numbers[0] if len(random_topo.betti_numbers) > 0 else 0,
            "betti_1": random_topo.betti_numbers[1] if len(random_topo.betti_numbers) > 1 else 0,
            "n_features": len(random_topo.diagram.points),
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
            "",
            "--- BETTI NUMBERS ---",
            "",
            f"{'Dimension':<12} | {'Primes':<10} | {'Random':<10}",
            "-" * 40,
            f"{'β₀ (components)':<12} | {payload['prime_topology']['betti_0']:<10} | {payload['random_topology']['betti_0']:<10}",
            f"{'β₁ (loops)':<12} | {payload['prime_topology']['betti_1']:<10} | {payload['random_topology']['betti_1']:<10}",
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
    embedding_dim: int = typer.Option(None, help="Time-delay embedding dimension (auto-derived via Takens' theorem if not specified)"),
) -> None:
    """Measure manifold curvature of prime gap distribution.

    Uses Ollivier-Ricci curvature to detect whether the prime gap
    manifold is positively curved (spherical), negatively curved
    (hyperbolic), or flat.

    k for k-NN is derived from the data's intrinsic dimension.

    Examples:
        mc geometry number-theory curvature
        mc geometry number-theory curvature --n-primes 1000
    """
    context = _context(ctx)

    from modelcypher.core.domain.geometry.ollivier_ricci import OllivierRicciCurvature
    from modelcypher.core.domain.geometry.prime_geometry import (
        _derive_embedding_dim,
        generate_primes,
        generate_random_gaps,
        time_delay_embedding,
    )

    typer.echo(f"Computing curvature for {n_primes} primes...")

    backend = get_default_backend()

    # Generate and embed
    primes = generate_primes(n_primes, backend)

    # Auto-derive embedding dimension if not specified
    if embedding_dim is None:
        embedding_dim = _derive_embedding_dim(primes.gaps, 1, backend)
        typer.echo(f"Auto-derived embedding dimension: {embedding_dim} (Takens' theorem)")

    prime_embedded = time_delay_embedding(primes.gaps, embedding_dim, backend=backend)

    mean_gap = float(backend.mean(primes.gaps))
    random_gaps = generate_random_gaps(primes.gap_count, mean_gap, backend)
    random_embedded = time_delay_embedding(random_gaps, embedding_dim, backend=backend)

    # Compute curvature - k derived from default config
    ricci = OllivierRicciCurvature(backend)

    typer.echo("Computing prime gap curvature...")
    prime_curv = ricci.compute(prime_embedded)  # k from config

    typer.echo("Computing random gap curvature...")
    random_curv = ricci.compute(random_embedded)  # k from config

    payload = {
        "_schema": "mc.geometry.number_theory.curvature.v1",
        "n_primes": n_primes,
        "embedding_dim": embedding_dim,
        "k_neighbors": ricci.config.k_neighbors,  # Report actual k used
        "prime_curvature": {
            "mean": prime_curv.mean_curvature,
            "std": prime_curv.std_curvature,
            "min": prime_curv.min_curvature,
            "max": prime_curv.max_curvature,
        },
        "random_curvature": {
            "mean": random_curv.mean_curvature,
            "std": random_curv.std_curvature,
            "min": random_curv.min_curvature,
            "max": random_curv.max_curvature,
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
            f"k-NN neighbors: {ricci.config.k_neighbors} (derived)",
            "",
            "--- OLLIVIER-RICCI CURVATURE ---",
            "",
            f"{'Metric':<12} | {'Primes':<15} | {'Random':<15}",
            "-" * 50,
            f"{'Mean':<12} | {prime_curv.mean_curvature:>15.4f} | {random_curv.mean_curvature:>15.4f}",
            f"{'Std':<12} | {prime_curv.std_curvature:>15.4f} | {random_curv.std_curvature:>15.4f}",
            f"{'Min':<12} | {prime_curv.min_curvature:>15.4f} | {random_curv.min_curvature:>15.4f}",
            f"{'Max':<12} | {prime_curv.max_curvature:>15.4f} | {random_curv.max_curvature:>15.4f}",
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


@app.command("full-analysis")
def full_analysis(
    ctx: typer.Context,
    n_primes: int = typer.Option(1000, help="Number of primes to analyze"),
    embedding_dim: int = typer.Option(None, help="Time-delay embedding dimension (auto-derived via Takens' theorem if not specified)"),
    n_bootstrap: int = typer.Option(None, help="Bootstrap samples for confidence intervals (auto-derived from sqrt(n_samples) if not specified)"),
    output_file: str = typer.Option(None, "--output-file", "-o", help="Save results to JSON"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Run comprehensive prime geometry analysis with multiple baselines.

    Tests primes against multiple random baselines (exponential, uniform,
    shuffled) and runs formal hypothesis tests with effect sizes and
    confidence intervals.

    This is the main command for rigorous scientific analysis.

    Examples:
        mc geometry number-theory full-analysis
        mc geometry number-theory full-analysis --n-primes 10000
        mc geometry number-theory full-analysis --output-file results.json
    """
    context = _context(ctx)

    from modelcypher.core.domain.geometry.prime_geometry import (
        BaselineType,
        format_comprehensive_result,
        run_comprehensive_analysis,
    )

    typer.echo(f"Running comprehensive analysis on {n_primes} primes...")
    dim_str = str(embedding_dim) if embedding_dim is not None else "auto (Takens' theorem)"
    boot_str = str(n_bootstrap) if n_bootstrap is not None else "auto (sqrt formula)"
    typer.echo(f"Embedding dimension: {dim_str}")
    typer.echo(f"Bootstrap samples: {boot_str}")
    typer.echo("")

    backend = get_default_backend()

    result = run_comprehensive_analysis(
        n_primes=n_primes,
        embedding_dim=embedding_dim,
        delay=1,
        baselines=[BaselineType.EXPONENTIAL, BaselineType.UNIFORM, BaselineType.SHUFFLED],
        n_bootstrap=n_bootstrap,
        backend=backend,
        seed=seed,
    )

    # Build JSON payload
    payload = {
        "_schema": "mc.geometry.number_theory.full_analysis.v1",
        "experiment_id": result.experiment_id,
        "timestamp": result.timestamp,
        "n_primes": result.n_primes,
        "max_prime": result.max_prime,
        "embedding_dim": result.embedding_dim,
        "prime_metrics": {},
        "baseline_metrics": {},
        "hypothesis_tests": {},
        "summary": result.summary,
    }

    # Prime metrics
    if "prime_time_delay" in result.embedding_results:
        ev = result.embedding_results["prime_time_delay"]
        payload["prime_metrics"]["time_delay"] = {
            "participation_ratio": ev.participation_ratio,
            "spectral_entropy": ev.spectral_entropy,
            "condition_number": ev.condition_number,
            "top_k_ratio": ev.top_k_ratio,
        }

    # Baseline metrics
    for name, ev in result.baseline_results.items():
        payload["baseline_metrics"][name] = {
            "participation_ratio": ev.participation_ratio,
            "spectral_entropy": ev.spectral_entropy,
            "condition_number": ev.condition_number,
            "top_k_ratio": ev.top_k_ratio,
        }

    # Hypothesis tests
    for name, test in result.hypothesis_tests.items():
        payload["hypothesis_tests"][name] = {
            "passed": test.passed,
            "p_value": test.p_value,
            "effect_size": test.effect_size.d,
            "prime_value": test.prime_value,
            "baseline_value": test.baseline_value,
        }

    # Save if requested
    if output_file:
        Path(output_file).write_text(json.dumps(payload, indent=2))
        typer.echo(f"Saved results to {output_file}")

    if context.output_format == "text":
        write_output(format_comprehensive_result(result), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("scale-study")
def scale_study(
    ctx: typer.Context,
    max_primes: int = typer.Option(10000, help="Maximum number of primes"),
    output_file: str = typer.Option(None, "--output-file", "-o", help="Save results to JSON"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Run scale analysis to test H7 (scale invariance).

    Tests whether prime structure is consistent or strengthens as we
    analyze larger ranges of primes. This validates that any detected
    structure is not a small-sample artifact.

    Scales tested: 100, 500, 1000, 5000, 10000 (up to max_primes)

    Examples:
        mc geometry number-theory scale-study
        mc geometry number-theory scale-study --max-primes 50000
    """
    context = _context(ctx)

    from modelcypher.core.domain.geometry.prime_geometry import run_scale_sweep

    typer.echo(f"Running scale study up to {max_primes} primes...")
    typer.echo("")

    backend = get_default_backend()

    # Define scales up to max_primes
    all_scales = [100, 500, 1000, 5000, 10000, 50000, 100000]
    scales = [s for s in all_scales if s <= max_primes]

    result = run_scale_sweep(
        scales=scales,
        embedding_dim=None,  # Auto-derived per-scale via Takens' theorem
        delay=1,
        backend=backend,
        seed=seed,
    )

    # Build payload
    payload = {
        "_schema": "mc.geometry.number_theory.scale_study.v1",
        "scales": result.scales,
        "n_successful": len(result.results),
        "effect_size_trend": result.effect_size_trend,
        "p_value_trend": result.p_value_trend,
        "scale_invariance_passed": result.scale_invariance_passed,
        "scale_results": [],
    }

    for analysis in result.results:
        scale_data = {
            "n_primes": analysis.n_primes,
            "prime_participation_ratio": analysis.summary.get("prime_participation_ratio", 0.0),
            "h1_pass_rate": analysis.summary.get("h1_pass_rate", 0.0),
        }
        payload["scale_results"].append(scale_data)

    # Save if requested
    if output_file:
        Path(output_file).write_text(json.dumps(payload, indent=2))
        typer.echo(f"Saved results to {output_file}")

    if context.output_format == "text":
        lines = [
            "=" * 70,
            "SCALE STUDY: PRIME GEOMETRY ACROSS SCALES",
            "=" * 70,
            "",
            f"Scales tested: {result.scales}",
            f"Successful runs: {len(result.results)}",
            "",
            "-" * 70,
            "RESULTS BY SCALE",
            "-" * 70,
            "",
            f"{'Scale':<12} | {'Participation':<15} | {'H1 Pass Rate':<15}",
            "-" * 50,
        ]

        for analysis in result.results:
            pr = analysis.summary.get("prime_participation_ratio", 0.0)
            h1_rate = analysis.summary.get("h1_pass_rate", 0.0)
            lines.append(f"{analysis.n_primes:<12} | {pr:<15.3f} | {h1_rate:<15.3f}")

        lines.extend([
            "",
            "-" * 70,
            "EFFECT SIZE TREND",
            "-" * 70,
            "",
        ])

        for i, (scale, effect) in enumerate(zip(result.scales, result.effect_size_trend)):
            lines.append(f"  n={scale}: Cohen's d = {effect:.3f}")

        lines.extend([
            "",
            "-" * 70,
            "H7: SCALE INVARIANCE",
            "-" * 70,
            "",
            f"  Status: {'✓ PASSED' if result.scale_invariance_passed else '✗ FAILED'}",
            "",
            "=" * 70,
        ])

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("perturbation")
def perturbation_study(
    ctx: typer.Context,
    n_primes: int = typer.Option(1000, help="Number of primes to analyze"),
    noise_levels: str = typer.Option("0.0,0.1,0.2,0.5,1.0", help="Comma-separated noise levels"),
    output_file: str = typer.Option(None, "--output-file", "-o", help="Save results to JSON"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Test H8: Perturbation robustness of prime geometry.

    Adds varying levels of noise to prime gaps and measures how much
    the geometric structure degrades. If prime structure is robust,
    it should maintain its spectral properties under moderate noise.

    Noise levels are expressed as fractions of the mean gap size.

    Examples:
        mc geometry number-theory perturbation
        mc geometry number-theory perturbation --noise-levels 0.0,0.1,0.3,0.5
    """
    context = _context(ctx)

    from modelcypher.core.domain.geometry.prime_geometry import run_perturbation_study

    # Parse noise levels
    levels = [float(x.strip()) for x in noise_levels.split(",")]

    typer.echo(f"Running perturbation study on {n_primes} primes...")
    typer.echo(f"Noise levels: {levels}")
    typer.echo("")

    backend = get_default_backend()

    results = run_perturbation_study(
        n_primes=n_primes,
        noise_levels=levels,
        embedding_dim=None,  # Auto-derived via Takens' theorem
        backend=backend,
        seed=seed,
    )

    # Build payload
    payload = {
        "_schema": "mc.geometry.number_theory.perturbation.v1",
        "n_primes": n_primes,
        "noise_levels": levels,
        "results": [],
    }

    for r in results:
        payload["results"].append({
            "noise_level": r.noise_level,
            "original_participation_ratio": r.original_participation_ratio,
            "perturbed_participation_ratio": r.perturbed_participation_ratio,
            "stability_score": r.stability_score,
        })

    # Raw measurements only - user interprets the stability curve
    # No "passed/failed" interpretation, no arbitrary thresholds

    # Save if requested
    if output_file:
        Path(output_file).write_text(json.dumps(payload, indent=2))
        typer.echo(f"Saved results to {output_file}")

    if context.output_format == "text":
        lines = [
            "=" * 70,
            "PERTURBATION STUDY: PRIME GEOMETRY ROBUSTNESS",
            "=" * 70,
            "",
            f"Primes analyzed: {n_primes}",
            "",
            "-" * 70,
            "STABILITY BY NOISE LEVEL",
            "-" * 70,
            "",
            f"{'Noise Level':<15} | {'Original PR':<15} | {'Perturbed PR':<15} | {'Stability':<10}",
            "-" * 65,
        ]

        for r in results:
            lines.append(
                f"{r.noise_level:<15.2f} | {r.original_participation_ratio:<15.3f} | "
                f"{r.perturbed_participation_ratio:<15.3f} | {r.stability_score:<10.3f}"
            )

        lines.extend([
            "",
            "=" * 70,
        ])

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("hypothesis-summary")
def hypothesis_summary(
    ctx: typer.Context,
    results_file: str = typer.Argument(..., help="JSON file with analysis results"),
) -> None:
    """Summarize hypothesis test results from a saved analysis.

    Takes a JSON results file from full-analysis and presents a summary
    of which hypotheses passed or failed.

    Examples:
        mc geometry number-theory hypothesis-summary results.json
    """
    context = _context(ctx)

    # Load results
    data = json.loads(Path(results_file).read_text())

    # Extract hypothesis tests
    tests = data.get("hypothesis_tests", {})

    payload = {
        "_schema": "mc.geometry.number_theory.hypothesis_summary.v1",
        "source_file": results_file,
        "n_tests": len(tests),
        "n_passed": sum(1 for t in tests.values() if t.get("passed", False)),
        "tests": tests,
    }

    if context.output_format == "text":
        lines = [
            "=" * 70,
            "HYPOTHESIS TEST SUMMARY",
            "=" * 70,
            "",
            f"Source: {results_file}",
            f"Tests: {payload['n_tests']} total, {payload['n_passed']} passed",
            "",
            "-" * 70,
            f"{'Hypothesis':<25} | {'Passed':<7} | {'Effect Size':<12} | {'p-value':<10}",
            "-" * 70,
        ]

        for name, test in tests.items():
            status = str(bool(test.get("passed", False))).lower()
            effect = test.get("effect_size", 0.0)
            p_val = test.get("p_value", 1.0)
            lines.append(f"{name:<25} | {status:<10} | {effect:<12.3f} | {p_val:<10.4f}")

        lines.extend([
            "",
            "-" * 70,
            "LEGEND",
            "-" * 70,
            "  H1: Spectral concentration (participation_ratio < baseline)",
            "  H2: Lower spectral entropy",
            "",
            "=" * 70,
        ])

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
