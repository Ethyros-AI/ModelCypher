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

"""Prime geometry analysis workflows."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    _promote_precision,
    machine_epsilon,
    power_iteration_eigh,
)

from .prime_geometry_baselines import generate_baseline, generate_random_gaps
from .prime_geometry_embeddings import generate_primes, time_delay_embedding
from .prime_geometry_spectral import analyze_eigenvalues, compare_distributions, compute_gram_matrix
from .prime_geometry_stats import _derive_bootstrap_count, run_hypothesis_test
from .prime_geometry_types import (
    BaselineType,
    ComprehensiveResult,
    PerturbationResult,
    PrimeGeometryResult,
    ScaleSweepResult,
)
from .prime_geometry_utils import _array_to_list, _randint_list

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


def derive_embedding_dim(sequence: "Array", delay: int, backend: "Backend") -> int:
    """Derive embedding dimension from sequence using Takens' theorem.

    Takens' embedding theorem states that for a d-dimensional attractor,
    an embedding dimension of 2*d + 1 is sufficient to reconstruct the
    attractor topology.

    Algorithm:
    1. Create minimal embedding (dim=2) to get preliminary Gram matrix
    2. Compute effective dimensionality: d_eff = (Σλ)² / Σλ²
    3. Apply Takens: embedding_dim = ceil(2*d_eff + 1)

    Returns:
        Derived embedding dimension, minimum 3, maximum limited by sequence length.
    """
    import math

    n = int(backend.shape(sequence)[0])

    # Preliminary embedding with minimal dimension
    prelim_dim = 2
    n_windows = n - (prelim_dim - 1) * delay
    if n_windows < 3:
        return 3  # Minimum meaningful embedding

    # Create preliminary embedding
    prelim_embedded = time_delay_embedding(sequence, prelim_dim, delay, backend)
    prelim_gram = compute_gram_matrix(prelim_embedded, backend)

    # Compute eigenvalues for effective dimensionality
    n_gram = int(prelim_gram.shape[0])
    eigenvalues, _ = power_iteration_eigh(backend, prelim_gram, k=n_gram)

    # Filter positive eigenvalues
    eps = machine_epsilon(backend, eigenvalues)
    pos_mask = eigenvalues > eps
    pos_count_arr = backend.sum(backend.astype(pos_mask, "int32"))
    backend.eval(pos_count_arr)
    pos_count = int(backend.to_scalar(pos_count_arr))

    if pos_count < 1:
        return 3

    pos_ev = eigenvalues[:pos_count]

    # Effective dimensionality: d_eff = (Σλ)² / Σλ²
    sum_ev_arr = backend.sum(pos_ev)
    sum_ev_sq_arr = backend.sum(pos_ev * pos_ev)
    backend.eval(sum_ev_arr, sum_ev_sq_arr)
    sum_ev = float(backend.to_scalar(sum_ev_arr))
    sum_ev_sq = float(backend.to_scalar(sum_ev_sq_arr))

    if sum_ev_sq < eps:
        d_eff = 1.0
    else:
        d_eff = (sum_ev * sum_ev) / sum_ev_sq

    # Takens' theorem: embedding_dim = ceil(2*d_eff + 1)
    embedding_dim = int(math.ceil(2.0 * d_eff + 1.0))

    # Clamp to reasonable range based on sequence length
    max_dim = max(3, n // 10)  # At most 10% of sequence length
    embedding_dim = max(3, min(embedding_dim, max_dim))

    return embedding_dim


def analyze_prime_geometry(
    n_primes: int = 1000,
    embedding_dim: int | None = None,
    delay: int = 1,
    backend: "Backend | None" = None,
    seed: int = 42,
) -> PrimeGeometryResult:
    """Perform complete geometric analysis of prime number distribution.

    This is the main entry point for the prime geometry experiment.

    Args:
        n_primes: Number of primes to analyze.
        embedding_dim: Dimension of time-delay embedding. If None, auto-derived
            using Takens' theorem: dim = ceil(2*d_eff + 1) where d_eff is the
            effective dimensionality from a preliminary analysis.
        delay: Time delay for embedding.
        backend: Compute backend.
        seed: Random seed for baseline comparison.

    Returns:
        PrimeGeometryResult with complete analysis.
    """
    backend = backend or get_default_backend()

    logger.info(f"Generating {n_primes} primes...")
    primes = generate_primes(n_primes, backend)

    logger.info(f"Prime gaps: {primes.gap_count}, max prime: {primes.max_prime}")

    # Auto-derive embedding dimension if not specified
    if embedding_dim is None:
        embedding_dim = derive_embedding_dim(primes.gaps, delay, backend)
        logger.info(f"Auto-derived embedding dimension: {embedding_dim} (Takens' theorem)")

    # Compute mean gap for random baseline
    gaps_float = _promote_precision(primes.gaps, backend)
    mean_gap_arr = backend.mean(gaps_float)
    backend.eval(mean_gap_arr)
    mean_gap = float(backend.to_scalar(mean_gap_arr))
    logger.info(f"Mean prime gap: {mean_gap:.2f}")

    # Time-delay embedding of prime gaps
    logger.info(f"Creating time-delay embedding (dim={embedding_dim}, delay={delay})...")
    prime_embedded = time_delay_embedding(primes.gaps, embedding_dim, delay, backend)
    n_windows = int(backend.shape(prime_embedded)[0])
    logger.info(f"Embedded shape: {n_windows} x {embedding_dim}")

    # Generate random baseline
    random_gaps = generate_random_gaps(primes.gap_count, mean_gap, backend, seed)
    random_embedded = time_delay_embedding(random_gaps, embedding_dim, delay, backend)

    # Compute Gram matrices
    logger.info("Computing Gram matrices...")
    prime_gram = compute_gram_matrix(prime_embedded, backend)
    random_gram = compute_gram_matrix(random_embedded, backend)

    # Analyze eigenvalue distributions
    logger.info("Analyzing eigenvalue distributions...")
    prime_ev = analyze_eigenvalues(prime_gram, backend)
    random_ev = analyze_eigenvalues(random_gram, backend)

    # Compare distributions
    comparison = compare_distributions(prime_ev, random_ev, "prime_gaps", "random_gaps", backend)

    # Intrinsic dimension via TwoNN
    logger.info("Computing intrinsic dimensions...")
    from modelcypher.core.domain.geometry.intrinsic_dimension import (
        IntrinsicDimension,
    )

    id_computer = IntrinsicDimension(backend)

    # Promote to float for ID computation
    prime_float = _promote_precision(prime_embedded, backend)
    random_float = _promote_precision(random_embedded, backend)

    try:
        prime_id = id_computer.compute(prime_float)
        prime_intrinsic_dim = prime_id.intrinsic_dimension
    except Exception as e:
        logger.warning(f"Prime ID estimation failed: {e}")
        prime_intrinsic_dim = float("nan")

    try:
        random_id = id_computer.compute(random_float)
        random_intrinsic_dim = random_id.intrinsic_dimension
    except Exception as e:
        logger.warning(f"Random ID estimation failed: {e}")
        random_intrinsic_dim = float("nan")

    # CKA between gap and position embeddings
    logger.info("Computing CKA between representations...")
    from modelcypher.core.domain.geometry.cka import compute_cka

    # Create position embedding (prime positions, not gaps)
    # Use primes[1:] to match gap count - slice using backend, convert to float
    primes_for_embed = primes.primes[1:]  # Skip first prime
    n_pos = int(backend.shape(primes_for_embed)[0])
    if n_pos >= embedding_dim:
        prime_pos = primes_for_embed[: primes.gap_count]
        pos_embedded = time_delay_embedding(prime_pos, embedding_dim, delay, backend)

        # Ensure same number of windows
        min_windows = min(
            int(backend.shape(prime_embedded)[0]),
            int(backend.shape(pos_embedded)[0]),
        )
        prime_for_cka = _promote_precision(prime_embedded[:min_windows], backend)
        pos_for_cka = _promote_precision(pos_embedded[:min_windows], backend)

        try:
            cka_result = compute_cka(prime_for_cka, pos_for_cka, backend)
            gap_to_position_cka = cka_result.cka
        except Exception as e:
            logger.warning(f"CKA computation failed: {e}")
            gap_to_position_cka = float("nan")
    else:
        gap_to_position_cka = float("nan")

    logger.info("Analysis complete.")

    return PrimeGeometryResult(
        prime_count=n_primes,
        embedding_dim=embedding_dim,
        prime_eigenvalues=prime_ev,
        random_eigenvalues=random_ev,
        comparison=comparison,
        prime_intrinsic_dim=prime_intrinsic_dim,
        random_intrinsic_dim=random_intrinsic_dim,
        gap_to_position_cka=gap_to_position_cka,
        prime_gram=prime_gram,
        random_gram=random_gram,
    )


def format_result(result: PrimeGeometryResult) -> str:
    """Format analysis result for display.

    Args:
        result: Analysis result to format.

    Returns:
        Formatted string for terminal output.
    """
    lines = [
        "=" * 60,
        "PRIME NUMBER GEOMETRY ANALYSIS",
        "=" * 60,
        "",
        f"Primes analyzed: {result.prime_count}",
        f"Embedding dimension: {result.embedding_dim}",
        "",
        "--- EIGENVALUE DISTRIBUTION ---",
        "",
        "Metric                    | Primes    | Random    | Diff",
        "-" * 60,
        f"Participation ratio       | {result.prime_eigenvalues.participation_ratio:9.3f} | {result.random_eigenvalues.participation_ratio:9.3f} | {result.comparison.participation_ratio_diff:+.3f}",
        f"Spectral entropy          | {result.prime_eigenvalues.spectral_entropy:9.3f} | {result.random_eigenvalues.spectral_entropy:9.3f} | {result.comparison.spectral_entropy_diff:+.3f}",
        f"Condition number          | {result.prime_eigenvalues.condition_number:9.1f} | {result.random_eigenvalues.condition_number:9.1f} |",
        f"Top-10 eigenvalue ratio   | {result.prime_eigenvalues.top_k_ratio:9.3f} | {result.random_eigenvalues.top_k_ratio:9.3f} |",
        "",
        "--- DISTRIBUTION COMPARISON ---",
        "",
        f"Wasserstein distance:     {result.comparison.wasserstein_distance:.4f}",
        f"Kolmogorov-Smirnov stat:  {result.comparison.ks_statistic:.4f}",
        "",
        "--- INTRINSIC DIMENSION ---",
        "",
        f"Prime gaps:               {result.prime_intrinsic_dim:.2f}",
        f"Random gaps:              {result.random_intrinsic_dim:.2f}",
        "",
        "--- CROSS-REPRESENTATION ---",
        "",
        f"CKA (gaps vs positions):  {result.gap_to_position_cka:.4f}",
        "",
        "=" * 60,
    ]

    # Raw measurements reported above; no arbitrary interpretation thresholds
    # Caller compares KS statistic and intrinsic dimensions against their requirements

    return "\n".join(lines)


def run_comprehensive_analysis(
    n_primes: int = 1000,
    embedding_dim: int | None = None,
    delay: int = 1,
    baselines: list[BaselineType] | None = None,
    n_bootstrap: int | None = None,
    backend: "Backend | None" = None,
    seed: int = 42,
) -> ComprehensiveResult:
    """Run comprehensive prime geometry analysis with multiple baselines.

    Args:
        n_primes: Number of primes to analyze.
        embedding_dim: Dimension of time-delay embedding. If None, auto-derived
            using Takens' theorem: dim = ceil(2*d_eff + 1).
        delay: Time delay for embedding.
        baselines: List of baseline types to test against.
        n_bootstrap: Number of bootstrap samples for CIs. If None, auto-derived
            from ceil(sqrt(n_samples)).
        backend: Compute backend.
        seed: Random seed.

    Returns:
        ComprehensiveResult with all analyses.
    """
    backend = backend or get_default_backend()

    if baselines is None:
        baselines = [
            BaselineType.EXPONENTIAL,
            BaselineType.UNIFORM,
            BaselineType.SHUFFLED,
        ]

    # Generate primes
    logger.info(f"Generating {n_primes} primes for comprehensive analysis...")
    primes = generate_primes(n_primes, backend)

    # Auto-derive embedding dimension if not specified
    if embedding_dim is None:
        embedding_dim = derive_embedding_dim(primes.gaps, delay, backend)
        logger.info(f"Auto-derived embedding dimension: {embedding_dim} (Takens' theorem)")

    # Auto-derive bootstrap count if not specified
    if n_bootstrap is None:
        n_bootstrap = _derive_bootstrap_count(primes.gap_count, backend)
        logger.info(f"Auto-derived bootstrap count: {n_bootstrap} (sqrt formula)")

    result = ComprehensiveResult(
        n_primes=n_primes,
        embedding_dim=embedding_dim,
    )

    result.max_prime = primes.max_prime
    mean_gap_arr = backend.mean(_promote_precision(primes.gaps, backend))
    backend.eval(mean_gap_arr)
    mean_gap = float(backend.to_scalar(mean_gap_arr))

    # Prime embeddings and eigenvalue analysis
    prime_embedded = time_delay_embedding(primes.gaps, embedding_dim, delay, backend)
    prime_gram = compute_gram_matrix(prime_embedded, backend)
    prime_ev = analyze_eigenvalues(prime_gram, backend)
    result.embedding_results["prime_time_delay"] = prime_ev

    # Collect bootstrap samples for primes
    prime_participation_samples = []
    gaps_list = _array_to_list(backend, primes.gaps)
    n_subsample = int(primes.gap_count * 0.8)
    for _ in range(n_bootstrap):
        # Subsample and re-analyze
        indices = _randint_list(backend, 0, primes.gap_count, n_subsample)
        subsample = backend.array([gaps_list[idx] for idx in indices])

        if n_subsample >= embedding_dim + 1:
            sub_embedded = time_delay_embedding(subsample, embedding_dim, delay, backend)
            sub_gram = compute_gram_matrix(sub_embedded, backend)
            sub_ev = analyze_eigenvalues(sub_gram, backend)
            prime_participation_samples.append(sub_ev.participation_ratio)

    # Analyze each baseline
    for baseline_type in baselines:
        logger.info(f"Analyzing baseline: {baseline_type.value}")

        baseline_gaps = generate_baseline(
            baseline_type,
            primes.gap_count,
            mean_gap,
            prime_gaps=primes.gaps,
            backend=backend,
            seed=seed,
        )

        baseline_embedded = time_delay_embedding(baseline_gaps, embedding_dim, delay, backend)
        baseline_gram = compute_gram_matrix(baseline_embedded, backend)
        baseline_ev = analyze_eigenvalues(baseline_gram, backend)

        result.baseline_results[baseline_type.value] = baseline_ev

        # Compare to primes
        comparison = compare_distributions(
            prime_ev, baseline_ev, "primes", baseline_type.value, backend
        )
        result.comparisons[f"primes_vs_{baseline_type.value}"] = comparison

    # Run hypothesis tests
    # H1: Spectral Concentration
    for baseline_type in baselines:
        baseline_ev = result.baseline_results[baseline_type.value]
        h1 = run_hypothesis_test(
            f"H1_{baseline_type.value}",
            f"Spectral concentration vs {baseline_type.value}",
            prime_ev.participation_ratio,
            baseline_ev.participation_ratio,
            prime_samples=prime_participation_samples if prime_participation_samples else None,
            one_sided=True,
            backend=backend,
        )
        result.hypothesis_tests[f"H1_{baseline_type.value}"] = h1

    # H2: Lower Spectral Entropy
    for baseline_type in baselines:
        baseline_ev = result.baseline_results[baseline_type.value]
        h2 = run_hypothesis_test(
            f"H2_{baseline_type.value}",
            f"Lower spectral entropy vs {baseline_type.value}",
            prime_ev.spectral_entropy,
            baseline_ev.spectral_entropy,
            one_sided=True,
            backend=backend,
        )
        result.hypothesis_tests[f"H2_{baseline_type.value}"] = h2

    # Summary statistics
    result.summary["prime_participation_ratio"] = prime_ev.participation_ratio
    result.summary["prime_spectral_entropy"] = prime_ev.spectral_entropy
    result.summary["n_baselines_tested"] = float(len(baselines))
    # Count only determinable tests (passed is not None)
    h1_tests = [v for k, v in result.hypothesis_tests.items() if k.startswith("H1")]
    determinable = [t for t in h1_tests if t.passed is not None]
    if determinable:
        result.summary["h1_pass_rate"] = sum(1 for t in determinable if t.passed) / len(determinable)
    else:
        result.summary["h1_pass_rate"] = float("nan")  # No determinable tests

    logger.info("Comprehensive analysis complete.")
    return result


def run_scale_sweep(
    scales: list[int] | None = None,
    embedding_dim: int | None = None,
    delay: int = 1,
    backend: "Backend | None" = None,
    seed: int = 42,
) -> ScaleSweepResult:
    """Run analysis across multiple scales to test scale invariance.

    Args:
        scales: List of n_primes values to test.
        embedding_dim: Dimension of time-delay embedding. If None, auto-derived
            per-scale using Takens' theorem.
        delay: Time delay for embedding.
        backend: Compute backend.
        seed: Random seed.

    Returns:
        ScaleSweepResult with all scale analyses.
    """
    backend = backend or get_default_backend()

    if scales is None:
        scales = [100, 500, 1000, 5000, 10000]

    result = ScaleSweepResult(scales=scales)

    for n_primes in scales:
        logger.info(f"Scale sweep: n_primes = {n_primes}")

        try:
            # Pass None for embedding_dim and n_bootstrap to auto-derive per-scale
            analysis = run_comprehensive_analysis(
                n_primes=n_primes,
                embedding_dim=embedding_dim,  # Auto-derived if None
                delay=delay,
                baselines=[BaselineType.EXPONENTIAL],  # Just one for speed
                n_bootstrap=None,  # Auto-derived from sqrt(n_samples)
                backend=backend,
                seed=seed,
            )
            result.results.append(analysis)

            # Extract effect sizes and p-values
            h1_key = "H1_exponential"
            if h1_key in analysis.hypothesis_tests:
                h1 = analysis.hypothesis_tests[h1_key]
                result.effect_size_trend.append(h1.effect_size.d)
                result.p_value_trend.append(h1.p_value)

        except Exception as e:
            logger.warning(f"Scale {n_primes} failed: {e}")
            continue

    # Evaluate scale invariance (H7)
    # Effect should be stable or increase with scale
    if len(result.effect_size_trend) >= 3:
        # Check if effect sizes are consistently negative (primes more concentrated)
        negative_effects = sum(1 for e in result.effect_size_trend if e < 0)
        result.scale_invariance_passed = negative_effects >= len(result.effect_size_trend) * 0.8

    return result


def run_perturbation_study(
    n_primes: int = 1000,
    noise_levels: list[float] | None = None,
    embedding_dim: int | None = None,
    backend: "Backend | None" = None,
    seed: int = 42,
) -> list[PerturbationResult]:
    """Test robustness of prime geometry to perturbations.

    Args:
        n_primes: Number of primes to analyze.
        noise_levels: List of noise levels (as fraction of mean gap).
        embedding_dim: Dimension of time-delay embedding. If None, auto-derived
            using Takens' theorem.
        backend: Compute backend.
        seed: Random seed.

    Returns:
        List of PerturbationResult for each noise level.
    """
    backend = backend or get_default_backend()

    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.2, 0.5, 1.0]

    results = []

    # Generate primes and compute baseline
    primes = generate_primes(n_primes, backend)

    # Auto-derive embedding dimension if not specified
    if embedding_dim is None:
        embedding_dim = derive_embedding_dim(primes.gaps, 1, backend)
        logger.info(f"Auto-derived embedding dimension: {embedding_dim} (Takens' theorem)")
    mean_gap_arr = backend.mean(_promote_precision(primes.gaps, backend))
    backend.eval(mean_gap_arr)
    mean_gap = float(backend.to_scalar(mean_gap_arr))

    prime_embedded = time_delay_embedding(primes.gaps, embedding_dim, 1, backend)
    prime_gram = compute_gram_matrix(prime_embedded, backend)
    original_ev = analyze_eigenvalues(prime_gram, backend)
    original_pr = original_ev.participation_ratio

    gaps_arr = _promote_precision(primes.gaps, backend)
    n_gaps = int(backend.shape(gaps_arr)[0])
    noise_dtype = gaps_arr.dtype if hasattr(gaps_arr, "dtype") else None

    for noise_level in noise_levels:
        logger.info(f"Perturbation study: noise_level = {noise_level}")

        if noise_level == 0.0:
            perturbed_pr = original_pr
        else:
            noise = backend.random_normal(shape=(n_gaps,), dtype=noise_dtype)
            noise = noise * noise_level * mean_gap
            perturbed_arr = backend.maximum(
                gaps_arr + noise,
                backend.full((n_gaps,), 2.0, dtype=noise_dtype),
            )
            backend.eval(perturbed_arr)
            perturbed_embedded = time_delay_embedding(perturbed_arr, embedding_dim, 1, backend)
            perturbed_gram = compute_gram_matrix(perturbed_embedded, backend)
            perturbed_ev = analyze_eigenvalues(perturbed_gram, backend)
            perturbed_pr = perturbed_ev.participation_ratio

        # Compute stability score
        if original_pr > 0:
            relative_change = abs(perturbed_pr - original_pr) / original_pr
            stability = 1.0 - min(relative_change, 1.0)
        else:
            stability = 0.0

        results.append(
            PerturbationResult(
                noise_level=noise_level,
                original_participation_ratio=original_pr,
                perturbed_participation_ratio=perturbed_pr,
                stability_score=stability,
            )
        )

    return results


def format_comprehensive_result(result: ComprehensiveResult) -> str:
    """Format comprehensive result for display.

    Args:
        result: Comprehensive analysis result.

    Returns:
        Formatted string for terminal output.
    """
    lines = [
        "=" * 70,
        "COMPREHENSIVE PRIME GEOMETRY ANALYSIS",
        "=" * 70,
        "",
        f"Experiment ID: {result.experiment_id}",
        f"Timestamp: {result.timestamp}",
        f"Primes analyzed: {result.n_primes}",
        f"Max prime: {result.max_prime}",
        f"Embedding dimension: {result.embedding_dim}",
        "",
        "--- EMBEDDING RESULTS ---",
        "",
    ]

    for key, ev in result.embedding_results.items():
        lines.append(f"{key}: PR={ev.participation_ratio:.3f}, Entropy={ev.spectral_entropy:.3f}")

    lines.extend(
        [
            "",
            "--- BASELINE RESULTS ---",
            "",
        ]
    )

    for key, ev in result.baseline_results.items():
        lines.append(f"{key}: PR={ev.participation_ratio:.3f}, Entropy={ev.spectral_entropy:.3f}")

    lines.extend(
        [
            "",
            "--- HYPOTHESIS TESTS ---",
            "",
        ]
    )

    for key, test in result.hypothesis_tests.items():
        passed = "PASS" if test.passed else "FAIL" if test.passed is False else "N/A"
        p_val = f"{test.p_value:.4f}" if test.p_value is not None else "N/A"
        lines.append(f"{key}: {passed}, p={p_val}, effect={test.effect_size.d:.3f}")

    lines.extend(
        [
            "",
            "--- SUMMARY ---",
            "",
        ]
    )

    for key, value in result.summary.items():
        lines.append(f"{key}: {value}")

    lines.append("=" * 70)

    return "\n".join(lines)
