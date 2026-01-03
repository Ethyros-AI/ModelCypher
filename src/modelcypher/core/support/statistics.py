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
Python lists (for backward compatibility) or backend arrays (for GPU acceleration).

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
# List-based functions (backward compatible, for small data)
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

    if lower_idx == upper_idx:
        # Exact index - use partition for O(n) complexity
        partitioned = backend.partition(arr, kth=lower_idx)
        backend.eval(partitioned)
        return float(backend.to_scalar(partitioned[lower_idx]))

    # Need interpolation between two indices
    # For simplicity, use partial sort on the larger index
    partitioned = backend.partition(arr, kth=upper_idx)
    backend.eval(partitioned)
    lower_val = float(backend.to_scalar(partitioned[lower_idx]))
    upper_val = float(backend.to_scalar(partitioned[upper_idx]))
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


def entropy_array(probs: Any, backend: Backend, eps: float | None = None) -> float:
    """Compute Shannon entropy of a probability distribution on GPU.

    Args:
        probs: Backend array of probabilities (should sum to 1).
        backend: Backend instance.
        eps: Small constant to avoid log(0). If None, uses machine epsilon.

    Returns:
        Entropy in nats (natural log base).
    """
    if eps is None:
        eps = backend.finfo().eps

    # Clamp probabilities to avoid log(0)
    safe_probs = backend.clip(probs, eps, 1.0)

    # H = -sum(p * log(p))
    log_probs = backend.log(safe_probs)
    entropy_terms = probs * log_probs
    result = -backend.sum(entropy_terms)
    backend.eval(result)

    return float(backend.to_scalar(result))


def entropy_bits_array(probs: Any, backend: Backend, eps: float | None = None) -> float:
    """Compute Shannon entropy in bits (base-2 log) on GPU.

    Args:
        probs: Backend array of probabilities (should sum to 1).
        backend: Backend instance.
        eps: Small constant to avoid log(0). If None, uses machine epsilon.

    Returns:
        Entropy in bits.
    """
    if eps is None:
        eps = backend.finfo().eps

    # Clamp probabilities to avoid log(0)
    safe_probs = backend.clip(probs, eps, 1.0)

    # H = -sum(p * log2(p))
    log2_probs = backend.log2(safe_probs)
    entropy_terms = probs * log2_probs
    result = -backend.sum(entropy_terms)
    backend.eval(result)

    return float(backend.to_scalar(result))
