from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import math


@dataclass(frozen=True)
class Prediction:
    expected_fidelity: float
    confidence: float
    sample_size: int
    fisher_z: float
    fisher_z_standard_error: float
    correlation_ci95: tuple[float, float]

    @property
    def qualitative_assessment(self) -> str:
        if self.expected_fidelity > 0.9:
            return "excellent"
        if self.expected_fidelity > 0.7:
            return "good"
        if self.expected_fidelity > 0.5:
            return "moderate"
        if self.expected_fidelity > 0.3:
            return "poor"
        return "very_poor"


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

        correlation = _pearson_correlation(vec_a, vec_b)
        if not math.isfinite(correlation):
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

        fisher_z_se = 1.0 / math.sqrt(sample_size - 3)
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
    return 0.5 * math.log((1.0 + r_clamped) / (1.0 - r_clamped))


def _inverse_fisher_z(value: float) -> float:
    e2z = math.exp(2.0 * value)
    return (e2z - 1.0) / (e2z + 1.0)


def _pearson_correlation(lhs: list[float], rhs: list[float]) -> float:
    if not lhs or len(lhs) != len(rhs):
        return float("nan")
    n = float(len(lhs))
    mean_l = sum(lhs) / n
    mean_r = sum(rhs) / n

    num = 0.0
    denom_l = 0.0
    denom_r = 0.0

    for i in range(len(lhs)):
        diff_l = lhs[i] - mean_l
        diff_r = rhs[i] - mean_r
        num += diff_l * diff_r
        denom_l += diff_l * diff_l
        denom_r += diff_r * diff_r

    denom = math.sqrt(denom_l * denom_r)
    if denom <= 1e-12:
        return float("nan")
    return num / denom
