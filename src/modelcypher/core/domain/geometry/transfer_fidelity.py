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
    machine_epsilon,
    sqrt_scalar,
)


@dataclass(frozen=True)
class Prediction:
    """Transfer fidelity prediction.

    Attributes
    ----------
    expected_fidelity : float
        Geodesic correlation between Gram matrices.
    confidence : float
        Null distribution percentile when provided; NaN otherwise.
    sample_size : int
        Number of off-diagonal elements compared.
    fisher_z : float
        Fisher z-transformed correlation.
    fisher_z_standard_error : float
        Standard error of Fisher z.
    correlation_ci95 : tuple of float
        Confidence interval not computed without null distribution (NaN bounds).
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

        # Extract upper triangular elements (i < j) using backend indexing.
        _b = get_default_backend()
        gram_a_arr = _b.reshape(_b.array(gram_a), (n, n))
        gram_b_arr = _b.reshape(_b.array(gram_b), (n, n))
        row_idx, col_idx = _b.triu_indices(n, k=1)
        flat_idx = row_idx * n + col_idx
        flat_a = _b.reshape(gram_a_arr, (-1,))
        flat_b = _b.reshape(gram_b_arr, (-1,))
        vec_a_arr = _b.take(flat_a, flat_idx, axis=0)
        vec_b_arr = _b.take(flat_b, flat_idx, axis=0)
        _b.eval(vec_a_arr, vec_b_arr)

        sample_size = int(_b.shape(vec_a_arr)[0])
        if sample_size == 0:
            return None

        vec_a_list = _b.tolist(vec_a_arr)
        vec_b_list = _b.tolist(vec_b_arr)
        correlation = compute_pearson_correlation(
            vec_a_list,
            vec_b_list,
            backend=_b,
        )
        if not is_finite(correlation, _b):
            return None

        fisher_z = _fisher_z_transform(correlation)
        if sample_size <= 3:
            return Prediction(
                expected_fidelity=correlation,
                confidence=float("nan"),
                sample_size=sample_size,
                fisher_z=fisher_z,
                fisher_z_standard_error=float("nan"),
                correlation_ci95=(float("nan"), float("nan")),
            )

        fisher_z_se = 1.0 / sqrt_scalar(sample_size - 3, _b)

        return Prediction(
            expected_fidelity=correlation,
            confidence=float("nan"),
            sample_size=sample_size,
            fisher_z=fisher_z,
            fisher_z_standard_error=fisher_z_se,
            correlation_ci95=(float("nan"), float("nan")),
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
    _b = get_default_backend()
    # Clamp to avoid log(0) at r=±1 - use dtype-derived bound
    eps = float(machine_epsilon(_b, _b.array([1.0])))
    bound = 1.0 - eps
    r_clamped = max(-bound, min(bound, value))
    return 0.5 * log_scalar((1.0 + r_clamped) / (1.0 - r_clamped), _b)


def _inverse_fisher_z(value: float) -> float:
    _b = get_default_backend()
    e2z = exp_scalar(2.0 * value, _b)
    return (e2z - 1.0) / (e2z + 1.0)
