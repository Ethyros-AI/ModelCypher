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

"""
Prime Number Geometry Analysis.

Explores the hypothesis that prime number distribution has hidden geometric
structure visible through high-dimensional analysis techniques.

Mathematical Motivation:
    1. The zeros of the Riemann zeta function behave like eigenvalues of
       random Hermitian matrices (Montgomery's pair correlation conjecture).

    2. Prime distribution is encoded in the spectrum of an unknown operator.
       If we can find the right embedding, the eigenvalue statistics should
       reveal this structure.

    3. Concept relationships in neural networks are invariant across models.
       Primes provide a "pure signal" - number-theoretic structure with no
       training noise - to test if our geometric tools can detect invariants.

Approach:
    - Embed prime gaps/positions into high-dimensional space via time-delay
    - Compute Gram matrices (relational structure independent of coordinates)
    - Analyze eigenvalue distributions
    - Compare to random baselines
    - Use intrinsic dimension, topological fingerprinting, and curvature

References:
    - Montgomery (1973): Pair correlation of zeros of the zeta function
    - Berry & Keating (1999): The Riemann zeros and eigenvalue asymptotics
    - Facco et al. (2017): TwoNN intrinsic dimension estimation
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrimeSequence:
    """A sequence of prime numbers with derived properties."""

    primes: "Array"  # The prime numbers [n_primes]
    gaps: "Array"  # Prime gaps: p[i+1] - p[i] [n_primes - 1]
    count: int
    max_prime: int

    @property
    def gap_count(self) -> int:
        return self.count - 1


@dataclass(frozen=True)
class EigenvalueDistribution:
    """Eigenvalue distribution of a Gram matrix."""

    eigenvalues: "Array"  # Sorted eigenvalues (descending)
    participation_ratio: float  # Effective rank: (sum(λ))^2 / sum(λ^2)
    spectral_entropy: float  # -sum(p * log(p)) where p = λ/sum(λ)
    condition_number: float  # λ_max / λ_min (for positive eigenvalues)
    top_k_ratio: float  # sum(top 10 eigenvalues) / sum(all eigenvalues)


@dataclass(frozen=True)
class SpectralComparison:
    """Comparison of two eigenvalue distributions."""

    source_label: str
    target_label: str
    participation_ratio_diff: float
    spectral_entropy_diff: float
    wasserstein_distance: float  # W1 distance between normalized spectra
    ks_statistic: float  # Kolmogorov-Smirnov statistic


@dataclass(frozen=True)
class PrimeGeometryResult:
    """Complete analysis of prime number geometry."""

    # Source data
    prime_count: int
    embedding_dim: int

    # Eigenvalue analysis
    prime_eigenvalues: EigenvalueDistribution
    random_eigenvalues: EigenvalueDistribution
    comparison: SpectralComparison

    # Intrinsic dimension
    prime_intrinsic_dim: float
    random_intrinsic_dim: float

    # CKA between different representations
    gap_to_position_cka: float  # CKA between gap and position embeddings

    # Raw data for further analysis
    prime_gram: "Array"
    random_gram: "Array"


def generate_primes(n: int, backend: "Backend | None" = None) -> PrimeSequence:
    """Generate the first n prime numbers using Sieve of Eratosthenes.

    Args:
        n: Number of primes to generate.
        backend: Compute backend (defaults to system default).

    Returns:
        PrimeSequence with primes and gaps.
    """
    backend = backend or get_default_backend()

    if n < 1:
        raise ValueError("n must be at least 1")

    # Upper bound for nth prime: p_n < n * (ln(n) + ln(ln(n))) for n >= 6
    if n < 6:
        limit = 15
    else:
        ln_n = math.log(n)
        ln_ln_n = math.log(ln_n)
        limit = int(n * (ln_n + ln_ln_n)) + 100

    # Sieve of Eratosthenes
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False

    # Collect primes
    primes_list = [i for i, p in enumerate(is_prime) if p][:n]

    if len(primes_list) < n:
        # Recursively increase limit if needed
        return generate_primes(n, backend)

    primes = backend.array(primes_list)

    # Compute gaps
    gaps_list = [primes_list[i + 1] - primes_list[i] for i in range(len(primes_list) - 1)]
    gaps = backend.array(gaps_list)

    return PrimeSequence(
        primes=primes,
        gaps=gaps,
        count=n,
        max_prime=primes_list[-1],
    )


def time_delay_embedding(
    sequence: "Array",
    embedding_dim: int,
    delay: int = 1,
    backend: "Backend | None" = None,
) -> "Array":
    """Create time-delay (Takens) embedding of a sequence.

    Transforms a 1D sequence into a matrix where each row is a sliding
    window of `embedding_dim` consecutive values.

    Args:
        sequence: 1D array of values [n].
        embedding_dim: Dimension of each embedded vector.
        delay: Time delay between consecutive dimensions (default 1).
        backend: Compute backend.

    Returns:
        Embedded matrix [n_windows, embedding_dim].

    Note:
        Time-delay embedding preserves the topology of the underlying
        dynamical system (Takens' theorem). If prime gaps have structure,
        it should be visible in this embedding.
    """
    backend = backend or get_default_backend()

    n = int(backend.shape(sequence)[0])
    n_windows = n - (embedding_dim - 1) * delay

    if n_windows < 1:
        raise ValueError(
            f"Sequence length {n} too short for embedding_dim={embedding_dim}, delay={delay}"
        )

    # Build embedding matrix
    rows = []
    # Convert sequence to list once outside loop for efficiency
    seq_list = backend.to_numpy(sequence).tolist()

    for i in range(n_windows):
        indices = [i + j * delay for j in range(embedding_dim)]
        window = backend.array([seq_list[idx] for idx in indices])
        rows.append(window)

    return backend.stack(rows, axis=0)


def compute_gram_matrix(X: "Array", backend: "Backend | None" = None) -> "Array":
    """Compute the Gram matrix K = X @ X^T.

    The Gram matrix captures relational geometry independent of feature
    dimension. Entry K[i,j] = <x_i, x_j> measures similarity between
    points i and j.

    Args:
        X: Data matrix [n_samples, n_features].
        backend: Compute backend.

    Returns:
        Gram matrix [n_samples, n_samples].
    """
    backend = backend or get_default_backend()
    # Ensure float dtype for matmul
    X_float = backend.array(backend.to_numpy(X).astype("float32"))
    return backend.matmul(X_float, backend.transpose(X_float))


def analyze_eigenvalues(
    gram: "Array",
    backend: "Backend | None" = None,
) -> EigenvalueDistribution:
    """Analyze the eigenvalue distribution of a Gram matrix.

    Args:
        gram: Symmetric Gram matrix [n, n].
        backend: Compute backend.

    Returns:
        EigenvalueDistribution with spectral metrics.
    """
    backend = backend or get_default_backend()

    # Compute eigenvalues (Gram is symmetric positive semi-definite)
    eigenvalues, _ = backend.eigh(gram)

    # Sort descending
    eigenvalues = backend.sort(eigenvalues)
    n = int(backend.shape(eigenvalues)[0])
    # Reverse for descending order
    ev_list = backend.to_numpy(eigenvalues)
    eigenvalues = backend.array(ev_list[::-1])

    # Filter positive eigenvalues for stability
    eps = 1e-10
    ev_list = backend.to_numpy(eigenvalues)
    pos_eigenvalues = [e for e in ev_list if e > eps]

    if len(pos_eigenvalues) < 2:
        # Degenerate case
        return EigenvalueDistribution(
            eigenvalues=eigenvalues,
            participation_ratio=1.0,
            spectral_entropy=0.0,
            condition_number=1.0,
            top_k_ratio=1.0,
        )

    pos_ev = backend.array(pos_eigenvalues)

    # Participation ratio: (sum(λ))^2 / sum(λ^2)
    # Measures effective number of significant eigenvalues
    sum_ev = float(backend.sum(pos_ev))
    sum_ev_sq = float(backend.sum(pos_ev * pos_ev))
    participation_ratio = (sum_ev * sum_ev) / sum_ev_sq if sum_ev_sq > eps else 1.0

    # Spectral entropy: -sum(p * log(p)) where p = λ/sum(λ)
    # Measures how spread out the spectrum is
    p = pos_ev / sum_ev
    p_list = backend.to_numpy(p)
    log_p = [math.log(pi) if pi > eps else 0.0 for pi in p_list]
    spectral_entropy = -sum(pi * lpi for pi, lpi in zip(p_list, log_p))

    # Condition number
    condition_number = pos_eigenvalues[0] / pos_eigenvalues[-1]

    # Top-k ratio (top 10 or all if fewer)
    k = min(10, len(pos_eigenvalues))
    top_k_sum = sum(pos_eigenvalues[:k])
    top_k_ratio = top_k_sum / sum_ev

    return EigenvalueDistribution(
        eigenvalues=eigenvalues,
        participation_ratio=participation_ratio,
        spectral_entropy=spectral_entropy,
        condition_number=condition_number,
        top_k_ratio=top_k_ratio,
    )


def compare_distributions(
    dist1: EigenvalueDistribution,
    dist2: EigenvalueDistribution,
    label1: str,
    label2: str,
    backend: "Backend | None" = None,
) -> SpectralComparison:
    """Compare two eigenvalue distributions.

    Args:
        dist1, dist2: Eigenvalue distributions to compare.
        label1, label2: Labels for the distributions.
        backend: Compute backend.

    Returns:
        SpectralComparison with distance metrics.
    """
    backend = backend or get_default_backend()

    # Normalize eigenvalues to probability distributions
    ev1 = backend.to_numpy(dist1.eigenvalues)
    ev2 = backend.to_numpy(dist2.eigenvalues)

    ev1_pos = [e for e in ev1 if e > 1e-10]
    ev2_pos = [e for e in ev2 if e > 1e-10]

    sum1 = sum(ev1_pos)
    sum2 = sum(ev2_pos)

    p1 = [e / sum1 for e in ev1_pos] if sum1 > 0 else ev1_pos
    p2 = [e / sum2 for e in ev2_pos] if sum2 > 0 else ev2_pos

    # Pad to same length for comparison
    max_len = max(len(p1), len(p2))
    p1 = p1 + [0.0] * (max_len - len(p1))
    p2 = p2 + [0.0] * (max_len - len(p2))

    # Wasserstein-1 distance (Earth Mover's Distance for 1D)
    # W1 = integral |F1(x) - F2(x)| dx where F is the CDF
    cdf1 = [sum(p1[: i + 1]) for i in range(len(p1))]
    cdf2 = [sum(p2[: i + 1]) for i in range(len(p2))]
    wasserstein = sum(abs(c1 - c2) for c1, c2 in zip(cdf1, cdf2)) / len(cdf1)

    # Kolmogorov-Smirnov statistic: max |F1(x) - F2(x)|
    ks_stat = max(abs(c1 - c2) for c1, c2 in zip(cdf1, cdf2))

    return SpectralComparison(
        source_label=label1,
        target_label=label2,
        participation_ratio_diff=dist1.participation_ratio - dist2.participation_ratio,
        spectral_entropy_diff=dist1.spectral_entropy - dist2.spectral_entropy,
        wasserstein_distance=wasserstein,
        ks_statistic=ks_stat,
    )


def generate_random_gaps(
    n: int,
    mean_gap: float,
    backend: "Backend | None" = None,
    seed: int = 42,
) -> "Array":
    """Generate random gaps with similar statistics to prime gaps.

    Creates a baseline for comparison: if primes have structure beyond
    their local statistics, it should differ from this random baseline.

    Args:
        n: Number of gaps to generate.
        mean_gap: Mean gap size (should match prime gaps for fair comparison).
        backend: Compute backend.
        seed: Random seed for reproducibility.

    Returns:
        Random gap sequence [n].
    """
    backend = backend or get_default_backend()
    backend.random_seed(seed)

    # Use exponential distribution (gaps between Poisson events)
    # This matches the theoretical model of "random" primes
    # E[gap] = mean_gap, so rate = 1/mean_gap
    uniform = backend.random_uniform(low=0.0, high=1.0, shape=(n,))
    # Inverse CDF of exponential: -mean * ln(1 - u)
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    uniform_safe = backend.maximum(uniform, backend.full((n,), eps))
    one_minus_u = backend.full((n,), 1.0) - uniform_safe
    one_minus_u = backend.maximum(one_minus_u, backend.full((n,), eps))

    gaps = -mean_gap * backend.log(one_minus_u)

    # Round to integers (gaps are integers) and ensure >= 2 (min prime gap)
    gaps_list = [max(2.0, round(float(g))) for g in backend.to_numpy(gaps)]
    return backend.array(gaps_list)


def analyze_prime_geometry(
    n_primes: int = 1000,
    embedding_dim: int = 20,
    delay: int = 1,
    backend: "Backend | None" = None,
    seed: int = 42,
) -> PrimeGeometryResult:
    """Perform complete geometric analysis of prime number distribution.

    This is the main entry point for the prime geometry experiment.

    Args:
        n_primes: Number of primes to analyze.
        embedding_dim: Dimension of time-delay embedding.
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

    # Compute mean gap for random baseline
    mean_gap = float(backend.mean(primes.gaps))
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

    # Convert to float for ID computation
    prime_float = backend.array(backend.to_numpy(prime_embedded).astype("float32"))
    random_float = backend.array(backend.to_numpy(random_embedded).astype("float32"))

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
    # Use primes[1:] to match gap count
    primes_for_embed = primes.primes
    prime_pos_list = backend.to_numpy(primes_for_embed)[1:]  # Skip first prime
    if len(prime_pos_list) >= embedding_dim:
        prime_pos = backend.array(prime_pos_list[: primes.gap_count].astype("float32"))
        pos_embedded = time_delay_embedding(prime_pos, embedding_dim, delay, backend)

        # Ensure same number of windows
        min_windows = min(
            int(backend.shape(prime_embedded)[0]),
            int(backend.shape(pos_embedded)[0]),
        )
        # Convert to float for CKA
        prime_for_cka = backend.array(
            backend.to_numpy(prime_embedded[:min_windows]).astype("float32")
        )
        pos_for_cka = backend.array(
            backend.to_numpy(pos_embedded[:min_windows]).astype("float32")
        )

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

    # Interpretation hints (relative, not vibes)
    if result.comparison.ks_statistic > 0.1:
        lines.append("")
        lines.append("NOTE: KS statistic > 0.1 suggests prime spectrum differs from random.")

    if abs(result.prime_intrinsic_dim - result.random_intrinsic_dim) > 1.0:
        lines.append("")
        lines.append("NOTE: Intrinsic dimension differs by >1, suggesting structural difference.")

    return "\n".join(lines)
