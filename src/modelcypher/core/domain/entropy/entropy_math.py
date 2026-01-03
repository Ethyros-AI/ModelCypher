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
Entropy Math - Pure Stateless Statistics for Entropy Trajectories.

This module consolidates all entropy trajectory statistics into one place,
ensuring consistent math across the entire framework.

Notes
-----
Previously duplicated in OptimizationMetricCalculator.calculate_statistics(),
LinguisticCalorimeter._measure_real(), and LinguisticCalorimeter._measure_simulated().
Now unified here with EntropyMath.calculate_trajectory_stats().

Examples
--------
From an entropy trajectory (list of per-token entropy values):

    from modelcypher.core.domain.entropy.entropy_math import EntropyMath

    stats = EntropyMath.calculate_trajectory_stats(trajectory)
    print(f"Mean: {stats.mean_entropy}, Variance: {stats.entropy_variance}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from modelcypher.core.domain._backend import get_default_backend


@dataclass(frozen=True)
class TrajectoryStats:
    """Statistics computed from an entropy trajectory."""

    mean_entropy: float
    entropy_variance: float
    first_token_entropy: float
    last_token_entropy: float
    trajectory_length: int

    @property
    def is_valid(self) -> bool:
        """Returns True if stats were computed from valid data."""
        return self.trajectory_length > 0

    @property
    def entropy_delta(self) -> float:
        """Change from first to last token entropy."""
        if self.trajectory_length < 2:
            return 0.0
        return self.last_token_entropy - self.first_token_entropy

    @property
    def is_cooling(self) -> bool:
        """True if entropy decreased over generation (cooling effect)."""
        return self.entropy_delta < 0  # Any decrease is cooling

    @property
    def is_heating(self) -> bool:
        """True if entropy increased over generation."""
        return self.entropy_delta > 0  # Any increase is heating


class EntropyMath:
    """
    Pure stateless entropy statistics.

    All methods are static - no state, no side effects.
    This is THE canonical source for entropy trajectory math.
    """

    @staticmethod
    def calculate_trajectory_stats(
        trajectory: Sequence[float],
        fallback_entropy: float | None = None,
    ) -> TrajectoryStats:
        """
        Calculate statistics from an entropy trajectory.

        Args:
            trajectory: Sequence of per-token entropy values.
            fallback_entropy: Value to use if trajectory is empty.

        Returns:
            TrajectoryStats with mean, variance, first/last token entropy.

        Math:
            - Mean: sum(H_i) / n
            - Variance: sum((H_i - mean)^2) / (n-1)  [sample variance]
        """
        if not trajectory:
            fallback = fallback_entropy if fallback_entropy is not None else 0.0
            return TrajectoryStats(
                mean_entropy=fallback,
                entropy_variance=0.0,
                first_token_entropy=fallback,
                last_token_entropy=fallback,
                trajectory_length=0,
            )

        n = len(trajectory)
        first_h = trajectory[0]
        last_h = trajectory[-1]
        backend = get_default_backend()
        traj_arr = backend.array(list(trajectory))
        mean_arr = backend.mean(traj_arr)
        if n > 1:
            diff = traj_arr - mean_arr
            variance_arr = backend.sum(diff * diff) / float(n - 1)
        else:
            variance_arr = backend.array([0.0])
        backend.eval(mean_arr, variance_arr)
        mean_h = float(backend.to_scalar(mean_arr))
        variance_h = float(backend.to_scalar(variance_arr))

        return TrajectoryStats(
            mean_entropy=mean_h,
            entropy_variance=variance_h,
            first_token_entropy=first_h,
            last_token_entropy=last_h,
            trajectory_length=n,
        )

    @staticmethod
    def sample_mean(values: Sequence[float]) -> float:
        """Compute sample mean of a sequence."""
        if not values:
            return 0.0
        backend = get_default_backend()
        values_arr = backend.array(list(values))
        mean_arr = backend.mean(values_arr)
        backend.eval(mean_arr)
        return float(backend.to_scalar(mean_arr))

    @staticmethod
    def sample_variance(values: Sequence[float], ddof: int = 1) -> float:
        """
        Compute sample variance.

        Args:
            values: Sequence of values.
            ddof: Delta degrees of freedom (1 for sample variance, 0 for population).

        Returns:
            Sample variance (Bessel-corrected by default).
        """
        if len(values) <= ddof:
            return 0.0
        backend = get_default_backend()
        values_arr = backend.array(list(values))
        mean_arr = backend.mean(values_arr)
        diff = values_arr - mean_arr
        variance_arr = backend.sum(diff * diff) / float(len(values) - ddof)
        backend.eval(variance_arr)
        return float(backend.to_scalar(variance_arr))

    @staticmethod
    def sample_std(values: Sequence[float], ddof: int = 1) -> float:
        """Compute sample standard deviation."""
        from modelcypher.core.domain.geometry.numerical_stability import sqrt_scalar

        backend = get_default_backend()
        variance = EntropyMath.sample_variance(values, ddof)
        return sqrt_scalar(variance, backend)

    @staticmethod
    def compute_delta_h(
        current_entropy: float,
        baseline_entropy: float | None,
    ) -> float | None:
        """
        Compute delta H relative to baseline.

        Args:
            current_entropy: Current measurement's mean entropy.
            baseline_entropy: Baseline mean entropy (or None).

        Returns:
            Delta H (current - baseline) or None if no baseline.
        """
        if baseline_entropy is None:
            return None
        return current_entropy - baseline_entropy

    @staticmethod
    def percentile(values: Sequence[float], p: float) -> float:
        """
        Compute percentile of a sequence.

        Args:
            values: Sequence of values.
            p: Percentile in [0, 100].

        Returns:
            Value at the given percentile.
        """
        if not values:
            return 0.0
        backend = get_default_backend()
        values_arr = backend.array(list(values))
        sorted_arr = backend.sort(values_arr)
        backend.eval(sorted_arr)
        n = int(sorted_arr.shape[0])
        idx = int(n * p / 100.0)
        idx = min(idx, n - 1)
        idx_arr = backend.array([idx])
        selected = backend.take(sorted_arr, idx_arr, axis=0)
        backend.eval(selected)
        return float(backend.to_scalar(selected))

    @staticmethod
    def entropy_percentiles(values: Sequence[float]) -> dict[int, float]:
        """Compute standard percentiles (25, 50, 75, 95)."""
        return {
            25: EntropyMath.percentile(values, 25),
            50: EntropyMath.percentile(values, 50),
            75: EntropyMath.percentile(values, 75),
            95: EntropyMath.percentile(values, 95),
        }
