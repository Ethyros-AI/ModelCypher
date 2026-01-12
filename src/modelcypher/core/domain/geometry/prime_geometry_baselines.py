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

"""Random baseline generators for prime geometry analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    exp_scalar,
    log_scalar,
)

from .prime_geometry_types import BaselineType
from .prime_geometry_utils import _array_to_list, _uniform_sampler

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


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
        mean_gap: Mean gap size (should match prime gaps for like-for-like comparison).
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

    if baseline_type == BaselineType.UNIFORM:
        # Use mean ± 50% as range
        return generate_uniform_gaps(n, mean_gap * 0.5, mean_gap * 1.5, backend, seed)

    if baseline_type == BaselineType.POISSON:
        return generate_poisson_gaps(n, mean_gap, backend, seed)

    if baseline_type == BaselineType.CRAMER:
        # For Cramér model, we need enough pseudo-primes
        _, gaps = generate_cramer_model(n + 1, backend, seed)
        # Trim to requested size
        gaps_list = _array_to_list(backend, gaps)[:n]
        return backend.array(gaps_list)

    if baseline_type == BaselineType.SHUFFLED:
        if prime_gaps is None:
            raise ValueError("prime_gaps required for shuffled baseline")
        return shuffled_gaps(prime_gaps, backend, seed)

    raise ValueError(f"Unknown baseline type: {baseline_type}")
