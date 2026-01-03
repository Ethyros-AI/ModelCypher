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

from dataclasses import dataclass
from typing import Iterable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    compute_pearson_correlation,
    exp_scalar,
    is_finite,
    log_scalar,
    sqrt_scalar,
)


@dataclass(frozen=True)
class Prediction:
    """Transfer fidelity prediction.

    Attributes
    ----------
    expected_fidelity : float
        Pearson correlation between Gram matrices.
    confidence : float
        Statistical confidence (1 - CI width).
    sample_size : int
        Number of off-diagonal elements compared.
    fisher_z : float
        Fisher z-transformed correlation.
    fisher_z_standard_error : float
        Standard error of Fisher z.
    correlation_ci95 : tuple of float
        95% confidence interval for correlation.
    """

    expected_fidelity: float
    confidence: float
    sample_size: int
    fisher_z: float
    fisher_z_standard_error: float
    correlation_ci95: tuple[float, float]


class TransferFidelityPrediction:
    @staticmethod
    def predict(gram_a: list[float], gram_b: list[float], n: int) -> Prediction | None:
        if len(gram_a) != n * n or len(gram_b) != n * n or n <= 1:
            return None

        vec_a: list[float] = []
        vec_b: list[float] = []

        for i in range(n):
            for j in range(i + 1, n):
                vec_a.append(float(gram_a[i * n + j]))
                vec_b.append(float(gram_b[i * n + j]))

        _b = get_default_backend()
        correlation = compute_pearson_correlation(vec_a, vec_b)
        if not is_finite(correlation, _b):
            return None

        fisher_z = _fisher_z_transform(correlation)
        sample_size = len(vec_a)
        if sample_size <= 3:
            return Prediction(
                expected_fidelity=correlation,
                confidence=0.0,
                sample_size=sample_size,
                fisher_z=fisher_z,
                fisher_z_standard_error=float("nan"),
                correlation_ci95=(float("nan"), float("nan")),
            )

        fisher_z_se = 1.0 / sqrt_scalar(sample_size - 3, _b)
        z_lower = fisher_z - 1.96 * fisher_z_se
        z_upper = fisher_z + 1.96 * fisher_z_se
        r_lower = _inverse_fisher_z(z_lower)
        r_upper = _inverse_fisher_z(z_upper)

        ci_width = r_upper - r_lower
        confidence = max(0.0, min(1.0, 1.0 - ci_width))

        return Prediction(
            expected_fidelity=correlation,
            confidence=confidence,
            sample_size=sample_size,
            fisher_z=fisher_z,
            fisher_z_standard_error=fisher_z_se,
            correlation_ci95=(r_lower, r_upper),
        )

    @staticmethod
    def predict_with_null_distribution(
        gram_a: list[float],
        gram_b: list[float],
        n: int,
        null_samples: Iterable[float],
    ) -> Prediction | None:
        base = TransferFidelityPrediction.predict(gram_a, gram_b, n)
        if base is None:
            return None
        samples = list(null_samples)
        if not samples:
            return base

        observed = base.expected_fidelity
        count_below = sum(1 for val in samples if val < observed)
        null_percentile = count_below / len(samples)

        return Prediction(
            expected_fidelity=base.expected_fidelity,
            confidence=null_percentile,
            sample_size=base.sample_size,
            fisher_z=base.fisher_z,
            fisher_z_standard_error=base.fisher_z_standard_error,
            correlation_ci95=base.correlation_ci95,
        )


def _fisher_z_transform(value: float) -> float:
    r_clamped = max(-0.9999, min(0.9999, value))
    _b = get_default_backend()
    return 0.5 * log_scalar((1.0 + r_clamped) / (1.0 - r_clamped), _b)


def _inverse_fisher_z(value: float) -> float:
    _b = get_default_backend()
    e2z = exp_scalar(2.0 * value, _b)
    return (e2z - 1.0) / (e2z + 1.0)
