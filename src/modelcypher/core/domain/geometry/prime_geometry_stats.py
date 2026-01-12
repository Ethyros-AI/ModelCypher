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
from .prime_geometry_utils import _randint_list, _uniform_list

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
        Number of bootstrap samples, minimum 10.
    """
    import math

    if n_samples < 1:
        return 10
    n_boot = int(math.ceil(sqrt_scalar(float(n_samples), backend)))
    return max(10, n_boot)


def bootstrap_confidence_interval(
    values: list[float],
    n_bootstrap: int | None = None,
    confidence: float = 0.95,
    backend: "Backend | None" = None,
) -> ConfidenceInterval:
    """Compute bootstrap confidence interval for a statistic.

    Args:
        values: List of observed values.
        n_bootstrap: Number of bootstrap samples. If None, auto-derived from
            ceil(sqrt(n_samples)) based on bootstrap standard error formula.
        confidence: Confidence level (default 0.95 for 95% CI).
        backend: Compute backend.

    Returns:
        ConfidenceInterval with lower, upper bounds and statistics.
    """
    backend = backend or get_default_backend()

    n = len(values)
    if n_bootstrap is None:
        n_bootstrap = _derive_bootstrap_count(n, backend)
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
