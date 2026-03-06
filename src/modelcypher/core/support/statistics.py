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

"""Statistics utilities with optional backend acceleration.

This module provides basic statistical functions that can work with either
Python lists or backend arrays.

When working with backend arrays, use the backend-accelerated versions
(mean_array, std_array, percentile_array) to avoid CPU round-trips.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    ceil_scalar,
    floor_scalar,
    sqrt_scalar,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


# =============================================================================
# List-based functions for scalar workflows and tests
# =============================================================================


def mean(values: list[float]) -> float:
    """Compute mean of a list of values.

    For backend arrays, use mean_array() instead.
    """
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def percentile(sorted_values: list[float], p: float) -> float:
    """Compute percentile from pre-sorted list.

    Args:
        sorted_values: Pre-sorted list of values.
        p: Percentile as fraction in [0, 1].

    Returns:
        Interpolated percentile value.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    clamped = max(0.0, min(1.0, p))
    position = clamped * float(len(sorted_values) - 1)
    _b = get_default_backend()
    lower_index = int(floor_scalar(position, _b))
    upper_index = int(ceil_scalar(position, _b))
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    lower_value = float(sorted_values[lower_index])
    upper_value = float(sorted_values[upper_index])
    fraction = position - float(lower_index)
    return lower_value + (upper_value - lower_value) * fraction


def standard_deviation(values: list[float], mean_value: float | None = None) -> float:
    """Compute sample standard deviation of a list.

    For backend arrays, use std_array() instead.

    Args:
        values: List of numeric values.
        mean_value: Pre-computed mean (optional, computed if not provided).

    Returns:
        Sample standard deviation (Bessel's correction: N-1 denominator).
    """
    if len(values) < 2:
        return 0.0
    if mean_value is None:
        mean_value = mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / float(len(values) - 1)
    return sqrt_scalar(max(0.0, variance), get_default_backend())


def standard_deviation_population(values: list[float], mean_value: float | None = None) -> float:
    """Compute population standard deviation of a list.

    Args:
        values: List of numeric values.
        mean_value: Pre-computed mean (optional, computed if not provided).

    Returns:
        Population standard deviation (N denominator).
    """
    if not values:
        return 0.0
    if mean_value is None:
        mean_value = mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / float(len(values))
    return sqrt_scalar(max(0.0, variance), get_default_backend())


# =============================================================================
# Backend-accelerated functions (for GPU arrays)
# =============================================================================


def mean_array(arr: Any, backend: Backend) -> float:
    """Compute mean of a backend array on GPU.

    Args:
        arr: Backend array.
        backend: Backend instance.

    Returns:
        Mean as Python float.
    """
    result = backend.mean(arr)
    backend.eval(result)
    return float(backend.to_scalar(result))


def std_array(arr: Any, backend: Backend, ddof: int = 0) -> float:
    """Compute standard deviation of a backend array on GPU.

    Args:
        arr: Backend array.
        backend: Backend instance.
        ddof: Delta degrees of freedom (0 for population, 1 for sample).

    Returns:
        Standard deviation as Python float.

    Note:
        Most backends compute population std by default (ddof=0).
        For sample std (ddof=1), we adjust manually.
    """
    result = backend.std(arr)
    backend.eval(result)
    std_val = float(backend.to_scalar(result))

    if ddof == 1:
        # Adjust from population to sample std: multiply by sqrt(n/(n-1))
        n = arr.shape[0] if hasattr(arr, 'shape') else len(arr)
        if n > 1:
            std_val *= sqrt_scalar(n / (n - 1), backend)

    return std_val


def var_array(arr: Any, backend: Backend, ddof: int = 0) -> float:
    """Compute variance of a backend array on GPU.

    Args:
        arr: Backend array.
        backend: Backend instance.
        ddof: Delta degrees of freedom (0 for population, 1 for sample).

    Returns:
        Variance as Python float.
    """
    result = backend.var(arr)
    backend.eval(result)
    var_val = float(backend.to_scalar(result))

    if ddof == 1:
        # Adjust from population to sample variance: multiply by n/(n-1)
        n = arr.shape[0] if hasattr(arr, 'shape') else len(arr)
        if n > 1:
            var_val *= n / (n - 1)

    return var_val


def percentile_array(arr: Any, p: float, backend: Backend) -> float:
    """Compute percentile of a backend array using partial sort.

    Uses O(n) partition algorithm for efficiency instead of full O(n log n) sort.

    Args:
        arr: Backend array (1D).
        p: Percentile as fraction in [0, 1].
        backend: Backend instance.

    Returns:
        Percentile value as Python float.
    """
    n = backend.shape(arr)[0]
    if n == 0:
        return 0.0
    if n == 1:
        backend.eval(arr)
        return float(backend.to_scalar(arr))

    clamped = max(0.0, min(1.0, p))
    position = clamped * float(n - 1)
    lower_idx = int(floor_scalar(position, backend))
    upper_idx = int(ceil_scalar(position, backend))

    def _kth_value(k: int) -> float:
        part = backend.argpartition(arr, k)
        prefix = backend.take(part, backend.arange(k + 1), axis=0)
        vals = backend.take(arr, prefix, axis=0)
        kth_val_arr = backend.max(vals)
        backend.eval(kth_val_arr)
        return float(backend.to_scalar(kth_val_arr))

    if lower_idx == upper_idx:
        return _kth_value(lower_idx)

    # Need interpolation between two indices
    # For simplicity, use partial sort on the larger index
    lower_val = _kth_value(lower_idx)
    upper_val = _kth_value(upper_idx)
    fraction = position - float(lower_idx)

    return lower_val + (upper_val - lower_val) * fraction


def softmax_array(arr: Any, backend: Backend, axis: int = -1) -> Any:
    """Compute softmax of a backend array on GPU.

    Args:
        arr: Backend array.
        backend: Backend instance.
        axis: Axis along which to compute softmax.

    Returns:
        Backend array with softmax probabilities.
    """
    return backend.softmax(arr, axis=axis)


# =============================================================================
# Statistical hypothesis testing
# =============================================================================


def permutation_pvalue(observed: float, null_samples: list[float]) -> float:
    """Compute p-value from permutation test.

    Tests whether the observed value is significantly different from
    the null distribution (two-tailed test).

    Args:
        observed: The observed test statistic.
        null_samples: Samples from the null distribution.

    Returns:
        Two-tailed p-value (proportion of null samples at least as extreme).
    """
    if not null_samples:
        return 1.0

    null_mean = mean(null_samples)
    observed_deviation = abs(observed - null_mean)

    # Count null samples at least as extreme as observed
    n_extreme = sum(1 for x in null_samples if abs(x - null_mean) >= observed_deviation)

    # Add 1 to numerator and denominator (Phipson & Smyth 2010, Stat. Appl. Genet.)
    # (accounts for the observed value itself in the permutation distribution)
    return (n_extreme + 1) / (len(null_samples) + 1)


def one_sided_pvalue(observed: float, null_samples: list[float], direction: str = "greater") -> float:
    """Compute one-sided p-value.

    Args:
        observed: The observed test statistic.
        null_samples: Samples from the null distribution.
        direction: "greater" tests if observed > null, "less" tests if observed < null.

    Returns:
        One-sided p-value.
    """
    if not null_samples:
        return 1.0

    if direction == "greater":
        n_extreme = sum(1 for x in null_samples if x >= observed)
    else:
        n_extreme = sum(1 for x in null_samples if x <= observed)

    return (n_extreme + 1) / (len(null_samples) + 1)


def cohens_d(observed: float, null_mean: float, null_std: float) -> float:
    """Compute Cohen's d effect size.

    Measures how many standard deviations the observed value differs
    from the null mean.

    Args:
        observed: The observed value.
        null_mean: Mean of the null distribution.
        null_std: Standard deviation of the null distribution.

    Returns:
        Cohen's d effect size. Conventions: |d| < 0.2 small, 0.2-0.8 medium, > 0.8 large.
    """
    if null_std == 0:
        return float("inf") if observed != null_mean else 0.0
    return (observed - null_mean) / null_std


def bootstrap_ci(
    samples: list[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean.

    Args:
        samples: Data samples.
        confidence: Confidence level (e.g., 0.95 for 95% CI).
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    import random as _random

    if not samples:
        return (0.0, 0.0)

    if seed is not None:
        _random.seed(seed)

    n = len(samples)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        resample = [_random.choice(samples) for _ in range(n)]
        bootstrap_means.append(mean(resample))

    bootstrap_means.sort()

    # Percentile method for CI
    alpha = 1 - confidence
    lower_idx = int((alpha / 2) * n_bootstrap)
    upper_idx = int((1 - alpha / 2) * n_bootstrap) - 1

    lower_idx = max(0, min(lower_idx, n_bootstrap - 1))
    upper_idx = max(0, min(upper_idx, n_bootstrap - 1))

    return (bootstrap_means[lower_idx], bootstrap_means[upper_idx])


def z_score(observed: float, null_mean: float, null_std: float) -> float:
    """Compute z-score (standard score).

    Args:
        observed: The observed value.
        null_mean: Mean of the reference distribution.
        null_std: Standard deviation of the reference distribution.

    Returns:
        Z-score (number of standard deviations from mean).
    """
    if null_std == 0:
        return float("inf") if observed > null_mean else (float("-inf") if observed < null_mean else 0.0)
    return (observed - null_mean) / null_std


# =============================================================================
# Cross-modal alignment metrics (Platonic Representation Hypothesis)
# =============================================================================


def mutual_knn_alignment(
    gram_a: Any,
    gram_b: Any,
    k: int = 10,
    backend: "Backend | None" = None,
) -> float:
    """Compute mutual k-NN alignment between two Gram matrices.

    This metric is used in the Platonic Representation Hypothesis paper
    (Huh et al., 2024) to measure cross-modal alignment. It computes
    the intersection of k-nearest neighbors between two similarity matrices.

    Alignment = (1/n) * sum_i |NN_k(K_a, i) ∩ NN_k(K_b, i)| / k

    Where NN_k(K, i) is the set of k nearest neighbors of point i according
    to Gram matrix K.

    Args:
        gram_a: Gram matrix from model A, shape [n, n].
        gram_b: Gram matrix from model B, shape [n, n].
        k: Number of nearest neighbors to consider.
        backend: Backend instance. If None, uses default.

    Returns:
        Alignment score in [0, 1]. Higher means more similar neighborhood structure.

    References:
        Huh et al. (2024) "The Platonic Representation Hypothesis" arXiv:2405.07987
    """
    if backend is None:
        backend = get_default_backend()

    backend.eval(gram_a, gram_b)

    n = backend.shape(gram_a)[0]
    if n <= k:
        k = max(1, n - 1)

    # Get indices that would sort each row (descending by similarity)
    # For Gram matrix, higher values = more similar
    # argsort gives ascending, so we negate to get descending
    neg_a = backend.multiply(gram_a, backend.array(-1.0))
    neg_b = backend.multiply(gram_b, backend.array(-1.0))

    # Get sorted indices for each row
    sorted_a = backend.argsort(neg_a, axis=1)
    sorted_b = backend.argsort(neg_b, axis=1)
    backend.eval(sorted_a, sorted_b)

    # For each point, compute intersection of top-k neighbors
    # Skip index 0 (self-similarity) and take next k
    total_intersection = 0.0

    for i in range(n):
        # Get top-k neighbors (excluding self)
        row_a = sorted_a[i]
        row_b = sorted_b[i]
        backend.eval(row_a, row_b)

        # Convert to Python lists for set operations
        if hasattr(row_a, 'tolist'):
            neighbors_a = row_a.tolist()
            neighbors_b = row_b.tolist()
        else:
            neighbors_a = list(row_a)
            neighbors_b = list(row_b)

        # Remove self from neighbors and take top k
        neighbors_a = [x for x in neighbors_a if x != i][:k]
        neighbors_b = [x for x in neighbors_b if x != i][:k]

        # Compute intersection
        intersection = len(set(neighbors_a) & set(neighbors_b))
        total_intersection += intersection / k

    return total_intersection / n


def mutual_knn_from_embeddings(
    embeds_a: Any,
    embeds_b: Any,
    k: int = 10,
    backend: "Backend | None" = None,
) -> float:
    """Compute mutual k-NN alignment directly from embedding matrices.

    Convenience function that computes Gram matrices internally.

    Args:
        embeds_a: Embeddings from model A, shape [n, d_a].
        embeds_b: Embeddings from model B, shape [n, d_b].
        k: Number of nearest neighbors to consider.
        backend: Backend instance. If None, uses default.

    Returns:
        Alignment score in [0, 1].
    """
    if backend is None:
        backend = get_default_backend()

    backend.eval(embeds_a, embeds_b)

    # Compute Gram matrices (inner product similarity)
    gram_a = backend.matmul(embeds_a, backend.transpose(embeds_a))
    gram_b = backend.matmul(embeds_b, backend.transpose(embeds_b))
    backend.eval(gram_a, gram_b)

    return mutual_knn_alignment(gram_a, gram_b, k=k, backend=backend)
