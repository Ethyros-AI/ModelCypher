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
    - Use multiple embedding strategies: time-delay, residue classes, digit patterns
    - Compute Gram matrices (relational structure independent of coordinates)
    - Analyze eigenvalue distributions
    - Compare to multiple baselines: exponential, uniform, Poisson, Cramér model
    - Use intrinsic dimension, topological fingerprinting, and curvature
    - Apply statistical testing with bootstrap CIs and effect sizes

Hypotheses (H1-H8):
    H1: Spectral Concentration - participation_ratio(primes) < participation_ratio(random)
    H2: Lower Spectral Entropy - spectral_entropy(primes) < spectral_entropy(random)
    H3: Distinct Intrinsic Dimension - |ID(primes) - ID(random)| > 1.0
    H4: Topological Distinctiveness - betti_diff > 0 OR bottleneck/scale > 0.1
    H5: Curvature Signature - mean_ricci differs significantly
    H6: Cross-Representation Coherence - CKA(prime embeds) > CKA(random embeds)
    H7: Scale Invariance - Effect sizes stable/increase with n
    H8: Perturbation Robustness - Primes more stable under noise

References:
    - Montgomery (1973): Pair correlation of zeros of the zeta function
    - Berry & Keating (1999): The Riemann zeros and eigenvalue asymptotics
    - Facco et al. (2017): TwoNN intrinsic dimension estimation
    - Cramér (1936): Prime number theorem, probabilistic model
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    exp_scalar,
    log_scalar,
    machine_epsilon,
    pi_value,
    sqrt_scalar,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


def _array_to_list(backend: "Backend", array: "Array") -> list[float]:
    """Convert 1D array to Python list using native tolist() - O(1) vs O(n)."""
    flat = backend.reshape(array, (-1,))
    return backend.tolist(flat)


def _uniform_list(backend: "Backend", count: int) -> list[float]:
    """Draw uniform [0,1) samples via backend and return as Python list."""
    if count <= 0:
        return []
    vals = backend.random_uniform(low=0.0, high=1.0, shape=(count,))
    backend.eval(vals)
    return [float(x) for x in backend.tolist(vals)]


def _randint_list(backend: "Backend", low: int, high: int, count: int) -> list[int]:
    """Draw integer samples via backend and return as Python list."""
    if count <= 0:
        return []
    vals = backend.random_randint(low, high, shape=(count,))
    backend.eval(vals)
    return [int(x) for x in backend.tolist(vals)]


def _uniform_sampler(backend: "Backend", batch_size: int = 1024):
    """Return a callable that yields uniform samples from a buffered pool."""
    pool: list[float] = []
    idx = 0

    def next_uniform() -> float:
        nonlocal pool, idx
        if idx >= len(pool):
            pool = _uniform_list(backend, batch_size)
            idx = 0
        val = float(pool[idx])
        idx += 1
        return val

    return next_uniform


class EmbeddingType(Enum):
    """Types of embeddings for prime sequence analysis."""

    TIME_DELAY = "time_delay"
    RESIDUE = "residue"
    DIGIT = "digit"
    POSITION = "position"


class BaselineType(Enum):
    """Types of random baselines for comparison."""

    EXPONENTIAL = "exponential"  # Gaps between Poisson events
    UNIFORM = "uniform"  # Uniform distribution
    POISSON = "poisson"  # Poisson-distributed gaps
    CRAMER = "cramer"  # Cramér probabilistic model
    SHUFFLED = "shuffled"  # Shuffled prime gaps


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


@dataclass(frozen=True)
class ConfidenceInterval:
    """95% confidence interval from bootstrap sampling."""

    lower: float
    upper: float
    mean: float
    std: float
    n_bootstrap: int


@dataclass(frozen=True)
class EffectSize:
    """Cohen's d effect size."""

    d: float  # Cohen's d: (mean1 - mean2) / pooled_std

    @staticmethod
    def from_cohens_d(d: float) -> "EffectSize":
        """Create EffectSize from Cohen's d value."""
        return EffectSize(d=d)


@dataclass(frozen=True)
class HypothesisTest:
    """Result of a single hypothesis test."""

    hypothesis_id: str  # H1-H8
    description: str
    passed: bool | None  # None when samples unavailable for statistical determination
    p_value: float | None  # None when samples unavailable for statistical test
    effect_size: EffectSize
    prime_value: float
    baseline_value: float
    confidence_interval: ConfidenceInterval | None = None


@dataclass
class ComprehensiveResult:
    """Complete results from comprehensive prime geometry analysis."""

    # Metadata
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Scale info
    n_primes: int = 0
    max_prime: int = 0
    embedding_dim: int = 20

    # Results by embedding type
    embedding_results: dict[str, EigenvalueDistribution] = field(default_factory=dict)

    # Results by baseline type
    baseline_results: dict[str, EigenvalueDistribution] = field(default_factory=dict)

    # Pairwise comparisons
    comparisons: dict[str, SpectralComparison] = field(default_factory=dict)

    # Hypothesis tests
    hypothesis_tests: dict[str, HypothesisTest] = field(default_factory=dict)

    # Summary statistics
    summary: dict[str, float] = field(default_factory=dict)


@dataclass
class ScaleSweepResult:
    """Results from testing across multiple scales."""

    scales: list[int] = field(default_factory=list)  # n_primes values tested
    results: list[ComprehensiveResult] = field(default_factory=list)

    # Trend analysis
    effect_size_trend: list[float] = field(default_factory=list)
    p_value_trend: list[float] = field(default_factory=list)
    scale_invariance_passed: bool = False


@dataclass
class PerturbationResult:
    """Results from perturbation robustness testing."""

    noise_level: float
    original_participation_ratio: float
    perturbed_participation_ratio: float
    stability_score: float  # 1 - relative change


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
        ln_n = log_scalar(float(n), backend)
        ln_ln_n = log_scalar(ln_n, backend)
        limit = int(n * (ln_n + ln_ln_n)) + 100

    # Sieve of Eratosthenes
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(sqrt_scalar(float(limit), backend)) + 1):
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

    # Build embedding matrix with vectorized indexing
    starts = backend.arange(n_windows)
    offsets = backend.arange(0, embedding_dim * delay, delay)
    starts_2d = backend.reshape(starts, (-1, 1))
    offsets_2d = backend.reshape(offsets, (1, -1))
    indices = starts_2d + offsets_2d
    return backend.take(sequence, indices, axis=0)


def residue_embedding(
    primes: "Array",
    moduli: list[int] | None = None,
    backend: "Backend | None" = None,
) -> "Array":
    """Create residue class embedding of primes.

    Embeds each prime as a vector of its residues modulo various moduli.
    Uses primorials (2, 6, 30, 210) by default since primes distribute
    non-uniformly across residue classes of primorials.

    Args:
        primes: Array of prime numbers [n_primes].
        moduli: List of moduli for residue computation.
                Default: [2, 6, 30, 210] (primorial sequence).
        backend: Compute backend.

    Returns:
        Residue embedding matrix [n_primes, len(moduli)].

    Note:
        For p > 2, p ≡ 1 or 5 (mod 6) - primes only hit 2 of 6 residue classes.
        This non-uniform distribution encodes prime structure.
    """
    backend = backend or get_default_backend()

    if moduli is None:
        moduli = [2, 6, 30, 210]  # Primorials: 2, 2*3, 2*3*5, 2*3*5*7

    primes_list = _array_to_list(backend, primes)
    rows = []

    for p in primes_list:
        residues = [float(int(p) % m) for m in moduli]
        rows.append(backend.array(residues))

    return backend.stack(rows, axis=0)


def digit_embedding(
    sequence: "Array",
    base: int = 10,
    max_digits: int = 10,
    backend: "Backend | None" = None,
) -> "Array":
    """Create digit pattern embedding of a sequence.

    Embeds each number as a vector of its digits in the specified base.
    Useful for detecting digit-based patterns (e.g., Benford's law).

    Args:
        sequence: Array of numbers [n].
        base: Number base for digit representation (default 10).
        max_digits: Maximum number of digits to consider.
        backend: Compute backend.

    Returns:
        Digit embedding matrix [n, max_digits].

    Note:
        Numbers are padded with leading zeros to ensure uniform dimension.
        The digit sequence is from most significant to least significant.
    """
    backend = backend or get_default_backend()

    seq_list = _array_to_list(backend, sequence)
    rows = []

    for num in seq_list:
        n = int(num)
        digits = []

        if n == 0:
            digits = [0.0] * max_digits
        else:
            while n > 0 and len(digits) < max_digits:
                digits.append(float(n % base))
                n //= base
            # Pad with zeros
            while len(digits) < max_digits:
                digits.append(0.0)
            # Reverse to get MSB first
            digits = digits[::-1]

        rows.append(backend.array(digits))

    return backend.stack(rows, axis=0)


def binary_digit_embedding(
    sequence: "Array",
    max_bits: int = 20,
    backend: "Backend | None" = None,
) -> "Array":
    """Create binary representation embedding.

    Args:
        sequence: Array of numbers [n].
        max_bits: Maximum number of bits to consider.
        backend: Compute backend.

    Returns:
        Binary embedding matrix [n, max_bits].
    """
    return digit_embedding(sequence, base=2, max_digits=max_bits, backend=backend)


def shuffled_gaps(
    gaps: "Array",
    backend: "Backend | None" = None,
    seed: int = 42,
) -> "Array":
    """Create shuffled version of gaps as a baseline.

    Preserves the marginal distribution of gaps but destroys
    sequential structure. If primes have structure beyond their
    gap distribution, shuffled gaps should differ.

    Args:
        gaps: Array of prime gaps [n].
        backend: Compute backend.
        seed: Random seed for reproducibility.

    Returns:
        Shuffled gap sequence [n].
    """
    backend = backend or get_default_backend()
    backend.random_seed(seed)

    gaps_list = _array_to_list(backend, gaps)
    n = len(gaps_list)

    # Fisher-Yates shuffle using backend random
    indices = list(range(n))
    if n <= 1:
        return backend.array(gaps_list)

    rand_vals = backend.random_uniform(low=0.0, high=1.0, shape=(n - 1,))
    backend.eval(rand_vals)
    rand_list = backend.tolist(rand_vals)
    rand_idx = 0
    for i in range(n - 1, 0, -1):
        # Generate random index from 0 to i
        u_val = float(rand_list[rand_idx])
        rand_idx += 1
        j = int(u_val * (i + 1))
        j = min(j, i)  # Safety clamp
        indices[i], indices[j] = indices[j], indices[i]

    shuffled = [gaps_list[idx] for idx in indices]
    return backend.array(shuffled)


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
    # Delegate to canonical implementation
    from modelcypher.core.domain.geometry.backend_matrix_utils import BackendMatrixUtils

    backend = backend or get_default_backend()
    X = backend.astype(X, "float32")
    utils = BackendMatrixUtils(backend)
    return utils.compute_gram_matrix(X, kernel="linear")


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
    n_eig = int(backend.shape(eigenvalues)[0])
    # Reverse for descending order
    reverse_idx = backend.arange(n_eig - 1, -1, -1)
    backend.eval(reverse_idx)
    eigenvalues = backend.take(eigenvalues, reverse_idx, axis=0)
    backend.eval(eigenvalues)

    # Filter positive eigenvalues for stability
    eps = machine_epsilon(backend, eigenvalues)
    pos_mask = eigenvalues > eps
    pos_count_arr = backend.sum(backend.astype(pos_mask, "int32"))
    backend.eval(pos_mask, pos_count_arr)
    pos_count = int(backend.to_scalar(pos_count_arr))

    if pos_count < 2:
        # Degenerate case
        return EigenvalueDistribution(
            eigenvalues=eigenvalues,
            participation_ratio=1.0,
            spectral_entropy=0.0,
            condition_number=1.0,
            top_k_ratio=1.0,
        )

    pos_ev = eigenvalues[:pos_count]

    # Participation ratio: (sum(λ))^2 / sum(λ^2)
    # Measures effective number of significant eigenvalues
    sum_ev_arr = backend.sum(pos_ev)
    sum_ev_sq_arr = backend.sum(pos_ev * pos_ev)
    backend.eval(sum_ev_arr, sum_ev_sq_arr)
    sum_ev = float(backend.to_scalar(sum_ev_arr))
    sum_ev_sq = float(backend.to_scalar(sum_ev_sq_arr))
    participation_ratio = (sum_ev * sum_ev) / sum_ev_sq if sum_ev_sq > eps else 1.0

    # Spectral entropy: -sum(p * log(p)) where p = λ/sum(λ)
    # Measures how spread out the spectrum is
    p = pos_ev / sum_ev
    log_p = backend.where(p > eps, backend.log(p), backend.zeros_like(p))
    entropy_arr = -backend.sum(p * log_p)
    backend.eval(entropy_arr)
    spectral_entropy = float(backend.to_scalar(entropy_arr))

    # Condition number
    first_ev = backend.take(pos_ev, backend.array([0]), axis=0)
    last_ev = backend.take(pos_ev, backend.array([pos_count - 1]), axis=0)
    backend.eval(first_ev, last_ev)
    first_ev_val = float(backend.to_scalar(first_ev))
    last_ev_val = float(backend.to_scalar(last_ev))
    condition_number = first_ev_val / last_ev_val

    # Top-k ratio (top 10 or all if fewer)
    k = min(10, pos_count)
    top_k_sum_arr = backend.sum(pos_ev[:k])
    backend.eval(top_k_sum_arr)
    top_k_sum = float(backend.to_scalar(top_k_sum_arr))
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
    ev1 = _array_to_list(backend, dist1.eigenvalues)
    ev2 = _array_to_list(backend, dist2.eigenvalues)

    eps = machine_epsilon(backend, dist1.eigenvalues)
    ev1_pos = [e for e in ev1 if e > eps]
    ev2_pos = [e for e in ev2 if e > eps]

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
    eps = division_epsilon(backend, uniform)
    uniform_safe = backend.maximum(uniform, backend.full((n,), eps))
    one_minus_u = backend.full((n,), 1.0) - uniform_safe
    one_minus_u = backend.maximum(one_minus_u, backend.full((n,), eps))

    gaps = -mean_gap * backend.log(one_minus_u)

    # Round to integers (gaps are integers) and ensure >= 2 (min prime gap)
    rounded = backend.floor(gaps + 0.5)
    gaps_clamped = backend.maximum(rounded, backend.full((n,), 2.0))
    return gaps_clamped


def generate_uniform_gaps(
    n: int,
    min_gap: float,
    max_gap: float,
    backend: "Backend | None" = None,
    seed: int = 42,
) -> "Array":
    """Generate uniformly distributed gaps.

    Args:
        n: Number of gaps to generate.
        min_gap: Minimum gap value.
        max_gap: Maximum gap value.
        backend: Compute backend.
        seed: Random seed for reproducibility.

    Returns:
        Uniform gap sequence [n].
    """
    backend = backend or get_default_backend()
    backend.random_seed(seed)

    uniform = backend.random_uniform(low=min_gap, high=max_gap, shape=(n,))
    rounded = backend.floor(uniform + 0.5)
    gaps_clamped = backend.maximum(rounded, backend.full((n,), 2.0))
    return gaps_clamped


def generate_poisson_gaps(
    n: int,
    rate: float,
    backend: "Backend | None" = None,
    seed: int = 42,
) -> "Array":
    """Generate Poisson-distributed gaps (counts, not inter-arrival times).

    Uses the Poisson distribution directly for gap counts, which is
    different from exponential inter-arrival times.

    Args:
        n: Number of gaps to generate.
        rate: Poisson rate parameter (lambda).
        backend: Compute backend.
        seed: Random seed for reproducibility.

    Returns:
        Poisson gap sequence [n].
    """
    backend = backend or get_default_backend()
    backend.random_seed(seed)

    # Generate Poisson samples using inverse transform
    # For Poisson, we use the iterative method
    gaps_list = []
    L = exp_scalar(-rate, backend)
    next_uniform = _uniform_sampler(backend, batch_size=2048)
    for _ in range(n):
        # Generate a single Poisson sample
        k = 0
        p = 1.0

        while p > L:
            k += 1
            p *= next_uniform()

        gaps_list.append(max(2.0, float(k)))

    return backend.array(gaps_list)


def generate_cramer_model(
    n_values: int,
    backend: "Backend | None" = None,
    seed: int = 42,
) -> tuple["Array", "Array"]:
    """Generate pseudo-primes using Cramér's probabilistic model.

    In Cramér's model, each integer m is "prime" with probability 1/ln(m).
    This captures the average density of primes but not their fine structure.

    Args:
        n_values: Number of pseudo-primes to generate.
        backend: Compute backend.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (pseudo_primes array, gaps array).
    """
    backend = backend or get_default_backend()
    backend.random_seed(seed)

    pseudo_primes = [2]  # Start with 2
    current = 3
    next_uniform = _uniform_sampler(backend, batch_size=2048)

    while len(pseudo_primes) < n_values:
        # P(m is "prime") = 1/ln(m)
        prob = 1.0 / log_scalar(float(current), backend) if current > 1 else 1.0
        u_val = next_uniform()

        if u_val < prob:
            pseudo_primes.append(current)

        current += 1

        # Safety: don't run forever
        if current > n_values * 100:
            break

    # Compute gaps
    gaps = [pseudo_primes[i + 1] - pseudo_primes[i] for i in range(len(pseudo_primes) - 1)]

    return backend.array(pseudo_primes), backend.array(gaps)


def generate_baseline(
    baseline_type: BaselineType,
    n: int,
    mean_gap: float,
    prime_gaps: "Array | None" = None,
    backend: "Backend | None" = None,
    seed: int = 42,
) -> "Array":
    """Generate a baseline gap sequence of the specified type.

    Args:
        baseline_type: Type of baseline to generate.
        n: Number of gaps to generate.
        mean_gap: Mean gap size for calibration.
        prime_gaps: Original prime gaps (for shuffled baseline).
        backend: Compute backend.
        seed: Random seed for reproducibility.

    Returns:
        Baseline gap sequence [n].
    """
    backend = backend or get_default_backend()

    if baseline_type == BaselineType.EXPONENTIAL:
        return generate_random_gaps(n, mean_gap, backend, seed)

    elif baseline_type == BaselineType.UNIFORM:
        # Use mean ± 50% as range
        return generate_uniform_gaps(n, mean_gap * 0.5, mean_gap * 1.5, backend, seed)

    elif baseline_type == BaselineType.POISSON:
        return generate_poisson_gaps(n, mean_gap, backend, seed)

    elif baseline_type == BaselineType.CRAMER:
        # For Cramér model, we need enough pseudo-primes
        _, gaps = generate_cramer_model(n + 1, backend, seed)
        # Trim to requested size
        gaps_list = _array_to_list(backend, gaps)[:n]
        return backend.array(gaps_list)

    elif baseline_type == BaselineType.SHUFFLED:
        if prime_gaps is None:
            raise ValueError("prime_gaps required for shuffled baseline")
        return shuffled_gaps(prime_gaps, backend, seed)

    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")


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
    mean_gap_arr = backend.mean(primes.gaps)
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

    # Convert to float for ID computation by multiplying by 1.0
    prime_float = prime_embedded * 1.0
    random_float = random_embedded * 1.0

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
        prime_pos = primes_for_embed[: primes.gap_count] * 1.0  # Convert to float
        pos_embedded = time_delay_embedding(prime_pos, embedding_dim, delay, backend)

        # Ensure same number of windows
        min_windows = min(
            int(backend.shape(prime_embedded)[0]),
            int(backend.shape(pos_embedded)[0]),
        )
        # Convert to float for CKA by multiplying by 1.0
        prime_for_cka = prime_embedded[:min_windows] * 1.0
        pos_for_cka = pos_embedded[:min_windows] * 1.0

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


# =============================================================================
# Statistical Testing Utilities
# =============================================================================


def bootstrap_confidence_interval(
    values: list[float],
    n_bootstrap: int = 200,
    confidence: float = 0.95,
    backend: "Backend | None" = None,
) -> ConfidenceInterval:
    """Compute bootstrap confidence interval for a statistic.

    Args:
        values: List of observed values.
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level (default 0.95 for 95% CI).
        backend: Compute backend.

    Returns:
        ConfidenceInterval with lower, upper bounds and statistics.
    """
    backend = backend or get_default_backend()

    n = len(values)
    if n < 2:
        mean_val = values[0] if values else 0.0
        return ConfidenceInterval(
            lower=mean_val,
            upper=mean_val,
            mean=mean_val,
            std=0.0,
            n_bootstrap=0,
        )

    # Generate bootstrap samples
    bootstrap_means = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = _randint_list(backend, 0, n, n)

        sample = [values[i] for i in indices]
        bootstrap_means.append(sum(sample) / len(sample))

    # Sort for percentiles
    bootstrap_means.sort()

    alpha = 1 - confidence
    lower_idx = int(alpha / 2 * n_bootstrap)
    upper_idx = int((1 - alpha / 2) * n_bootstrap) - 1

    mean_val = sum(values) / len(values)
    std_val = sqrt_scalar(sum((v - mean_val) ** 2 for v in values) / (len(values) - 1), backend)

    return ConfidenceInterval(
        lower=bootstrap_means[lower_idx],
        upper=bootstrap_means[upper_idx],
        mean=mean_val,
        std=std_val,
        n_bootstrap=n_bootstrap,
    )


def compute_cohens_d(
    values1: list[float],
    values2: list[float],
    backend: "Backend | None" = None,
) -> EffectSize:
    """Compute Cohen's d effect size between two groups.

    Args:
        values1: First group of values.
        values2: Second group of values.

    Returns:
    EffectSize with Cohen's d.
    """
    backend = backend or get_default_backend()
    n1, n2 = len(values1), len(values2)

    if n1 < 2 or n2 < 2:
        return EffectSize(d=0.0)

    mean1 = sum(values1) / n1
    mean2 = sum(values2) / n2

    var1 = sum((v - mean1) ** 2 for v in values1) / (n1 - 1)
    var2 = sum((v - mean2) ** 2 for v in values2) / (n2 - 1)

    # Pooled standard deviation
    pooled_std = sqrt_scalar(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2), backend)

    eps = machine_epsilon(backend, backend.array([0.0]))
    if pooled_std < eps:
        d = 0.0
    else:
        d = (mean1 - mean2) / pooled_std

    return EffectSize.from_cohens_d(d)


def permutation_test(
    values1: list[float],
    values2: list[float],
    n_permutations: int = 1000,
    backend: "Backend | None" = None,
) -> float:
    """Compute p-value via permutation test.

    Tests the null hypothesis that the two groups come from the same
    distribution, specifically testing if the difference in means is
    significant.

    Args:
        values1: First group of values.
        values2: Second group of values.
        n_permutations: Number of permutations.
        backend: Compute backend.

    Returns:
        Two-tailed p-value.
    """
    backend = backend or get_default_backend()

    observed_diff = abs(sum(values1) / len(values1) - sum(values2) / len(values2))
    combined = values1 + values2
    n1 = len(values1)
    n_total = len(combined)

    count_extreme = 0

    for _ in range(n_permutations):
        # Shuffle combined data
        shuffled = combined.copy()
        rand_vals = _uniform_list(backend, n_total - 1)
        rand_idx = 0
        for i in range(n_total - 1, 0, -1):
            u_val = rand_vals[rand_idx]
            rand_idx += 1
            j = int(u_val * (i + 1))
            j = min(j, i)
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]

        # Split and compute difference
        perm_mean1 = sum(shuffled[:n1]) / n1
        perm_mean2 = sum(shuffled[n1:]) / (n_total - n1)
        perm_diff = abs(perm_mean1 - perm_mean2)

        if perm_diff >= observed_diff:
            count_extreme += 1

    return (count_extreme + 1) / (n_permutations + 1)


def run_hypothesis_test(
    hypothesis_id: str,
    description: str,
    prime_value: float,
    baseline_value: float,
    prime_samples: list[float] | None = None,
    baseline_samples: list[float] | None = None,
    one_sided: bool = True,
    backend: "Backend | None" = None,
) -> HypothesisTest:
    """Run a single hypothesis test.

    Args:
        hypothesis_id: Identifier (H1-H8).
        description: Human-readable description.
        prime_value: Observed value for primes.
        baseline_value: Observed value for baseline.
        prime_samples: Bootstrap samples for primes (if available).
        baseline_samples: Bootstrap samples for baseline (if available).
        one_sided: If True, test if prime < baseline (for concentration metrics).
        backend: Compute backend.

    Returns:
        HypothesisTest with results.
    """
    backend = backend or get_default_backend()

    # Compute effect size
    if prime_samples and baseline_samples:
        effect = compute_cohens_d(prime_samples, baseline_samples, backend=backend)
        p_value = permutation_test(prime_samples, baseline_samples, backend=backend)
        ci = bootstrap_confidence_interval(
            [p - b for p, b in zip(prime_samples, baseline_samples)],
            backend=backend,
        )
    else:
        # Single-value comparison - no samples means no p-value
        diff = prime_value - baseline_value
        eps = division_epsilon(backend, backend.array([baseline_value]))
        effect = EffectSize.from_cohens_d(diff / (abs(baseline_value) + eps))
        p_value = None  # Cannot compute without samples
        ci = None

    # Determine pass/fail
    # With samples: require statistical significance (p < 0.05)
    # Without samples: cannot determine statistically (passed = None)
    passed: bool | None
    if p_value is not None:
        if one_sided:
            passed = prime_value < baseline_value and p_value < 0.05
        else:
            passed = prime_value != baseline_value and p_value < 0.05
    else:
        # No samples = no statistical determination possible
        # Effect size is still reported; consumer decides interpretation
        passed = None

    return HypothesisTest(
        hypothesis_id=hypothesis_id,
        description=description,
        passed=passed,
        p_value=p_value,
        effect_size=effect,
        prime_value=prime_value,
        baseline_value=baseline_value,
        confidence_interval=ci,
    )


# =============================================================================
# Comprehensive Experiment Runners
# =============================================================================


def run_comprehensive_analysis(
    n_primes: int = 1000,
    embedding_dim: int = 20,
    delay: int = 1,
    baselines: list[BaselineType] | None = None,
    n_bootstrap: int = 50,
    backend: "Backend | None" = None,
    seed: int = 42,
) -> ComprehensiveResult:
    """Run comprehensive prime geometry analysis with multiple baselines.

    Args:
        n_primes: Number of primes to analyze.
        embedding_dim: Dimension of time-delay embedding.
        delay: Time delay for embedding.
        baselines: List of baseline types to test against.
        n_bootstrap: Number of bootstrap samples for CIs.
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

    result = ComprehensiveResult(
        n_primes=n_primes,
        embedding_dim=embedding_dim,
    )

    # Generate primes
    logger.info(f"Generating {n_primes} primes for comprehensive analysis...")
    primes = generate_primes(n_primes, backend)
    result.max_prime = primes.max_prime
    mean_gap_arr = backend.mean(primes.gaps)
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
    embedding_dim: int = 20,
    delay: int = 1,
    backend: "Backend | None" = None,
    seed: int = 42,
) -> ScaleSweepResult:
    """Run analysis across multiple scales to test scale invariance.

    Args:
        scales: List of n_primes values to test.
        embedding_dim: Dimension of time-delay embedding.
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

        # Adjust embedding dim for small scales
        effective_dim = min(embedding_dim, n_primes // 10)
        if effective_dim < 5:
            effective_dim = 5

        try:
            analysis = run_comprehensive_analysis(
                n_primes=n_primes,
                embedding_dim=effective_dim,
                delay=delay,
                baselines=[BaselineType.EXPONENTIAL],  # Just one for speed
                n_bootstrap=20,  # Fewer for speed
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
    embedding_dim: int = 20,
    backend: "Backend | None" = None,
    seed: int = 42,
) -> list[PerturbationResult]:
    """Test robustness of prime geometry to perturbations.

    Args:
        n_primes: Number of primes to analyze.
        noise_levels: List of noise levels (as fraction of mean gap).
        embedding_dim: Dimension of time-delay embedding.
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
    mean_gap_arr = backend.mean(primes.gaps)
    backend.eval(mean_gap_arr)
    mean_gap = float(backend.to_scalar(mean_gap_arr))

    prime_embedded = time_delay_embedding(primes.gaps, embedding_dim, 1, backend)
    prime_gram = compute_gram_matrix(prime_embedded, backend)
    original_ev = analyze_eigenvalues(prime_gram, backend)
    original_pr = original_ev.participation_ratio

    for noise_level in noise_levels:
        logger.info(f"Perturbation study: noise_level = {noise_level}")

        if noise_level == 0.0:
            perturbed_pr = original_pr
        else:
            # Add Gaussian noise scaled to noise_level * mean_gap
            gaps_arr = backend.astype(primes.gaps, "float32")
            n_gaps = int(backend.shape(gaps_arr)[0])
            u1 = backend.random_uniform(low=0.0, high=1.0, shape=(n_gaps,))
            u2 = backend.random_uniform(low=0.0, high=1.0, shape=(n_gaps,))
            u1_eps = division_epsilon(backend, u1)
            u1_safe = backend.maximum(u1, backend.full((n_gaps,), u1_eps))
            two_pi = 2.0 * pi_value(backend)
            z = backend.sqrt(-2.0 * backend.log(u1_safe)) * backend.cos(two_pi * u2)
            noise = z * noise_level * mean_gap
            perturbed_arr = backend.maximum(
                gaps_arr + noise,
                backend.full((n_gaps,), 2.0),
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
        f"Primes: {result.n_primes} (max: {result.max_prime})",
        f"Embedding dimension: {result.embedding_dim}",
        "",
        "-" * 70,
        "SPECTRAL METRICS",
        "-" * 70,
        "",
    ]

    # Prime results
    if "prime_time_delay" in result.embedding_results:
        ev = result.embedding_results["prime_time_delay"]
        lines.append("Prime gaps (time-delay embedding):")
        lines.append(f"  Participation ratio: {ev.participation_ratio:.3f}")
        lines.append(f"  Spectral entropy:    {ev.spectral_entropy:.3f}")
        lines.append(f"  Top-10 ratio:        {ev.top_k_ratio:.3f}")
        lines.append("")

    # Baseline comparisons
    for name, ev in result.baseline_results.items():
        lines.append(f"Baseline [{name}]:")
        lines.append(f"  Participation ratio: {ev.participation_ratio:.3f}")
        lines.append(f"  Spectral entropy:    {ev.spectral_entropy:.3f}")
        lines.append("")

    # Hypothesis tests
    lines.append("-" * 70)
    lines.append("HYPOTHESIS TESTS")
    lines.append("-" * 70)
    lines.append("")

    for name, test in result.hypothesis_tests.items():
        if test.passed is None:
            status = "? INDETERMINATE (no samples)"
        elif test.passed:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
        lines.append(f"{name}: {status}")
        lines.append(f"  {test.description}")
        lines.append(f"  Prime: {test.prime_value:.3f}, Baseline: {test.baseline_value:.3f}")
        lines.append(f"  Effect size: {test.effect_size.d:.3f}")
        p_str = f"{test.p_value:.4f}" if test.p_value is not None else "N/A (no samples)"
        lines.append(f"  p-value: {p_str}")
        lines.append("")

    # Summary
    lines.append("-" * 70)
    lines.append("SUMMARY")
    lines.append("-" * 70)
    for key, value in result.summary.items():
        lines.append(f"  {key}: {value:.3f}")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)
