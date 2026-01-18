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

"""Statistical testing utilities for prime geometry analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
    sqrt_scalar,
)

from .prime_geometry_types import ConfidenceInterval, EffectSize, HypothesisTest
from .prime_geometry_utils import _randint_list

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


def _derive_bootstrap_count(n_samples: int, backend: "Backend") -> int:
    """Derive number of bootstrap samples from data size.

    Formula: n_bootstrap = ceil(sqrt(n_samples))

    This is mathematically derived from:
    - Bootstrap standard error ~ 1/sqrt(n_bootstrap)
    - We want bootstrap error comparable to sampling error ~ 1/sqrt(n_samples)
    - Therefore n_bootstrap ~ sqrt(n_samples) is the data-derived minimum

    Returns:
        Number of bootstrap samples.
    """
    import math

    if n_samples < 2:
        return 0
    n_boot = int(math.ceil(sqrt_scalar(float(n_samples), backend)))
    return max(1, n_boot)


def bootstrap_confidence_interval(
    values: list[float],
    backend: "Backend | None" = None,
) -> ConfidenceInterval:
    """Compute bootstrap interval bounds for a statistic.

    Args:
        values: List of observed values.
        backend: Compute backend.

    Returns:
        ConfidenceInterval with bounds and statistics.
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
    n_bootstrap = _derive_bootstrap_count(n, backend)

    # Generate bootstrap samples
    bootstrap_means = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = _randint_list(backend, 0, n, n)

        sample = [values[i] for i in indices]
        bootstrap_means.append(sum(sample) / len(sample))

    lower_bound = min(bootstrap_means)
    upper_bound = max(bootstrap_means)

    mean_val = sum(values) / len(values)
    std_val = sqrt_scalar(sum((v - mean_val) ** 2 for v in values) / (len(values) - 1), backend)

    return ConfidenceInterval(
        lower=lower_bound,
        upper=upper_bound,
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
    backend: "Backend | None" = None,
) -> float:
    """Compute exact two-tailed p-value via permutation test.

    Tests the null hypothesis that the two groups come from the same
    distribution, specifically testing if the difference in means is
    significant.

    Args:
        values1: First group of values.
        values2: Second group of values.
        backend: Compute backend.

    Returns:
        Two-tailed p-value.
    """
    backend = backend or get_default_backend()

    if not values1 or not values2:
        return float("nan")

    observed_diff = abs(sum(values1) / len(values1) - sum(values2) / len(values2))
    combined = values1 + values2
    n1 = len(values1)
    n_total = len(combined)
    n2 = n_total - n1
    if n2 == 0:
        return float("nan")

    from itertools import combinations

    total_sum = sum(combined)
    backend = backend or get_default_backend()
    eps = float(machine_epsilon(backend, backend.array(combined)))

    count_extreme = 0
    total = 0
    for combo in combinations(range(n_total), n1):
        sum1 = sum(combined[idx] for idx in combo)
        mean1 = sum1 / n1
        mean2 = (total_sum - sum1) / n2
        perm_diff = abs(mean1 - mean2)
        total += 1
        if perm_diff + eps >= observed_diff:
            count_extreme += 1

    return count_extreme / total


def run_hypothesis_test(
    hypothesis_id: str,
    description: str,
    prime_value: float,
    baseline_value: float,
    prime_samples: list[float] | None = None,
    baseline_samples: list[float] | None = None,
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
        backend: Compute backend.

    Returns:
        HypothesisTest with effect sizes and bootstrap interval bounds.
    """
    backend = backend or get_default_backend()

    # Compute effect size
    if prime_samples and baseline_samples:
        effect = compute_cohens_d(prime_samples, baseline_samples, backend=backend)
        p_value = None
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

    # No pass/fail. Return raw metrics only.
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
