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

from __future__ import annotations

import math
import random


from dataclasses import dataclass

from modelcypher.core.support import statistics
from modelcypher.core.domain.geometry.exceptions import EstimatorError


@dataclass(frozen=True)
class BootstrapConfiguration:
    resamples: int = 200
    confidence_level: float = 0.95
    seed: int = 42


@dataclass(frozen=True)
class TwoNNConfiguration:
    use_regression: bool = True
    bootstrap: BootstrapConfiguration | None = None


@dataclass(frozen=True)
class ConfidenceInterval:
    level: float
    lower: float
    upper: float
    resamples: int
    seed: int


@dataclass(frozen=True)
class TwoNNEstimate:
    intrinsic_dimension: float
    sample_count: int
    usable_count: int
    uses_regression: bool
    ci: ConfidenceInterval | None


class IntrinsicDimensionEstimator:
    @staticmethod
    def estimate_two_nn(
        points: list[list[float]],
        configuration: TwoNNConfiguration = TwoNNConfiguration(),
    ) -> TwoNNEstimate:
        mu = IntrinsicDimensionEstimator._compute_two_nn_mu(points)
        estimate = IntrinsicDimensionEstimator._estimate_two_nn_from_mu(mu, configuration.use_regression)

        ci = None
        if configuration.bootstrap is not None:
            ci = IntrinsicDimensionEstimator._bootstrap_two_nn(
                mu,
                configuration.use_regression,
                configuration.bootstrap,
            )

        return TwoNNEstimate(
            intrinsic_dimension=estimate,
            sample_count=len(points),
            usable_count=len(mu),
            uses_regression=configuration.use_regression,
            ci=ci,
        )

    @staticmethod
    def _compute_two_nn_mu(points: list[list[float]]) -> list[float]:
        n = len(points)
        if n < 3:
            raise EstimatorError.insufficient_samples(n)
        if not points:
            raise EstimatorError.insufficient_samples(0)

        d = len(points[0])
        if d <= 0:
            raise EstimatorError.invalid_point_dimension(1, 0)

        for point in points:
            if len(point) != d:
                raise EstimatorError.invalid_point_dimension(d, len(point))
            if not all(math.isfinite(value) for value in point):
                raise EstimatorError.non_finite_point_value()

        mu: list[float] = []
        for i in range(n):
            r1_sq = float("inf")
            r2_sq = float("inf")
            for j in range(n):
                if j == i:
                    continue
                dist_sq = IntrinsicDimensionEstimator._squared_euclidean(points[i], points[j])
                if dist_sq < r1_sq:
                    r2_sq = r1_sq
                    r1_sq = dist_sq
                elif dist_sq < r2_sq:
                    r2_sq = dist_sq

            if not math.isfinite(r1_sq) or not math.isfinite(r2_sq):
                continue
            if r1_sq <= 0:
                continue
            r1 = math.sqrt(r1_sq)
            r2 = math.sqrt(r2_sq)
            ratio = r2 / r1
            if not math.isfinite(ratio) or ratio < 1:
                continue
            mu.append(ratio)

        if len(mu) < 3:
            raise EstimatorError.nearest_neighbor_degenerate()
        return mu

    @staticmethod
    def _estimate_two_nn_from_mu(mu: list[float], use_regression: bool) -> float:
        n = len(mu)
        if n < 3:
            raise EstimatorError.insufficient_samples(n)

        log_mu = [math.log(value) for value in mu if value > 0 and math.isfinite(value)]
        log_mu = [value for value in log_mu if math.isfinite(value) and value >= 0]
        if len(log_mu) < 3:
            raise EstimatorError.regression_degenerate()

        if not use_regression:
            mean_value = sum(log_mu) / float(len(log_mu))
            if mean_value <= 0:
                raise EstimatorError.regression_degenerate()
            return 1.0 / mean_value

        mu_sorted = sorted(mu)
        sum_xx = 0.0
        sum_xy = 0.0
        usable = 0
        denom_n = float(len(mu_sorted))
        for i in range(len(mu_sorted) - 1):
            x = math.log(mu_sorted[i])
            if not math.isfinite(x) or x < 0:
                continue
            f = float(i + 1) / denom_n
            y = -math.log(max(1e-12, 1.0 - f))
            if not math.isfinite(y):
                continue
            sum_xx += x * x
            sum_xy += x * y
            usable += 1

        if usable < 3 or sum_xx <= 0:
            raise EstimatorError.regression_degenerate()

        d_hat = sum_xy / sum_xx
        if not math.isfinite(d_hat) or d_hat <= 0:
            raise EstimatorError.regression_degenerate()
        return d_hat

    @staticmethod
    def _bootstrap_two_nn(
        mu: list[float],
        use_regression: bool,
        config: BootstrapConfiguration,
    ) -> ConfidenceInterval | None:
        n = len(mu)
        if n < 3:
            raise EstimatorError.insufficient_samples(n)

        resamples = max(0, config.resamples)
        if resamples <= 0:
            return None

        level = max(0.0, min(1.0, config.confidence_level))
        alpha = (1.0 - level) / 2.0

        rng = random.Random(config.seed)
        estimates: list[float] = []
        for _ in range(resamples):
            sample = [mu[rng.randrange(n)] for _ in range(n)]
            try:
                estimates.append(
                    IntrinsicDimensionEstimator._estimate_two_nn_from_mu(sample, use_regression)
                )
            except EstimatorError:
                continue

        if len(estimates) < 10:
            return None

        sorted_estimates = sorted(estimates)
        lower = statistics.percentile(sorted_estimates, alpha)
        upper = statistics.percentile(sorted_estimates, 1.0 - alpha)
        return ConfidenceInterval(
            level=level,
            lower=lower,
            upper=upper,
            resamples=resamples,
            seed=config.seed,
        )

    @staticmethod
    def _squared_euclidean(a: list[float], b: list[float]) -> float:
        total = 0.0
        for idx in range(len(a)):
            diff = a[idx] - b[idx]
            total += diff * diff
        return total
