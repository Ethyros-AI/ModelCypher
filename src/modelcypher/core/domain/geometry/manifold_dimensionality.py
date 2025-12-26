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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain.entropy.entropy_delta_sample import EntropyDeltaSample

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend
from modelcypher.core.domain.geometry.intrinsic_dimension import (
    IntrinsicDimension,
    TwoNNConfiguration,
)
from modelcypher.core.support import statistics


@dataclass(frozen=True)
class EntropyTraceFeatures:
    token_count: int
    mean: float
    std_dev: float
    max: float

    @property
    def feature_vector(self) -> list[float]:
        return [self.max, self.mean, self.std_dev]


@dataclass(frozen=True)
class FeatureStat:
    index: int
    name: str
    mean: float
    std_dev: float


@dataclass(frozen=True)
class PriorTensionSummary:
    token_count: int
    mean_base_surprisal: float | None
    p95_base_surprisal: float | None
    mean_base_approval_probability: float | None
    p05_base_approval_probability: float | None
    mean_normalized_approval: float | None
    p05_normalized_approval: float | None
    top_token_disagreement_rate: float | None


@dataclass(frozen=True)
class IDEstimateSummary:
    intrinsic_dimension: float
    ci95_lower: float | None
    ci95_upper: float | None
    sample_count: int
    usable_count: int
    uses_regression: bool

    @staticmethod
    def from_estimate(estimate) -> "IDEstimateSummary":
        return IDEstimateSummary(
            intrinsic_dimension=estimate.intrinsic_dimension,
            ci95_lower=estimate.ci.lower if estimate.ci else None,
            ci95_upper=estimate.ci.upper if estimate.ci else None,
            sample_count=estimate.sample_count,
            usable_count=estimate.usable_count,
            uses_regression=estimate.uses_regression,
        )


class ManifoldDimensionality:
    @staticmethod
    def entropy_trace_features(entropies: list[float]) -> EntropyTraceFeatures | None:
        cleaned = [
            float(value) for value in entropies if value is not None and math.isfinite(value)
        ]
        if not cleaned:
            return None
        token_count = len(cleaned)
        mean = sum(cleaned) / float(token_count)
        std_dev = statistics.standard_deviation(cleaned, mean)
        max_val = max(cleaned) if cleaned else mean
        return EntropyTraceFeatures(
            token_count=token_count,
            mean=mean,
            std_dev=std_dev,
            max=max_val,
        )

    @staticmethod
    def feature_stats(points: list[list[float]], feature_names: list[str]) -> list[FeatureStat]:
        if not points:
            return []
        d = len(points[0])
        if d <= 0 or any(len(row) != d for row in points):
            return []
        names = feature_names if len(feature_names) == d else [f"feature_{i}" for i in range(d)]

        stats: list[FeatureStat] = []
        for j in range(d):
            values = [
                float(row[j]) for row in points if row[j] is not None and math.isfinite(row[j])
            ]
            if not values:
                continue
            mean_val = sum(values) / float(len(values))
            std_val = statistics.standard_deviation(values, mean_val)
            stats.append(FeatureStat(index=j, name=names[j], mean=mean_val, std_dev=std_val))
        return stats

    @staticmethod
    def summarize_prior_tension(samples: list[EntropyDeltaSample]) -> PriorTensionSummary | None:
        if not samples:
            return None

        surprisal = [float(s.base_surprisal) for s in samples if s.base_surprisal is not None]
        approval_prob = [
            float(s.base_approval_probability)
            for s in samples
            if s.base_approval_probability is not None
        ]
        normalized = [
            float(s.normalized_approval_score)
            for s in samples
            if s.normalized_approval_score is not None
        ]

        disagreement_rate = None
        if samples:
            disagree = sum(1 for s in samples if s.top_token_disagreement)
            disagreement_rate = float(disagree) / float(len(samples))

        def mean(values: list[float]) -> float | None:
            return sum(values) / float(len(values)) if values else None

        def percentile(values: list[float], p: float) -> float | None:
            if not values:
                return None
            sorted_values = sorted(values)
            return statistics.percentile(sorted_values, p)

        return PriorTensionSummary(
            token_count=len(samples),
            mean_base_surprisal=mean(surprisal),
            p95_base_surprisal=percentile(surprisal, 0.95),
            mean_base_approval_probability=mean(approval_prob),
            p05_base_approval_probability=percentile(approval_prob, 0.05),
            mean_normalized_approval=mean(normalized),
            p05_normalized_approval=percentile(normalized, 0.05),
            top_token_disagreement_rate=disagreement_rate,
        )

    @staticmethod
    def estimate_id(
        points: list[list[float]],
        bootstrap_resamples: int | None = None,
        seed: int = 42,
        use_regression: bool = True,
    ) -> IDEstimateSummary:
        # Note: bootstrap_resamples and seed are kept for API compatibility
        # but TwoNNConfiguration doesn't currently support bootstrap CI
        _ = bootstrap_resamples, seed  # Unused, kept for future compatibility
        estimate = IntrinsicDimension.compute_two_nn(
            points,
            configuration=TwoNNConfiguration(
                use_regression=use_regression,
            ),
        )
        return IDEstimateSummary.from_estimate(estimate)


class BackendManifoldDimensionality:
    """GPU-accelerated manifold dimensionality analysis using the Backend protocol.

    This class provides the same functionality as ManifoldDimensionality but uses
    Backend tensor operations for GPU acceleration. Key optimizations:

    - entropy_trace_features: Backend mean/std/max instead of Python loops
    - feature_stats: Vectorized column-wise statistics
    """

    def __init__(self, backend: "Backend"):
        """Initialize with a Backend instance.

        Args:
            backend: Backend instance (MLXBackend, JAXBackend, etc.)
        """
        self.backend = backend
        self._finfo = backend.finfo()

    def entropy_trace_features(self, entropies: list[float]) -> EntropyTraceFeatures | None:
        """Compute entropy trace features using GPU acceleration.

        Args:
            entropies: List of entropy values.

        Returns:
            EntropyTraceFeatures with mean, std_dev, max, or None if empty.
        """
        # Filter valid values
        cleaned = [
            float(value) for value in entropies if value is not None and math.isfinite(value)
        ]
        if not cleaned:
            return None

        token_count = len(cleaned)
        arr = self.backend.array(cleaned)

        # Compute statistics using backend
        mean_val = self.backend.mean(arr)
        self.backend.eval(mean_val)
        mean_float = self._to_scalar(mean_val)

        # Variance and std_dev (sample variance with N-1 denominator)
        diff = arr - mean_val
        sum_sq = self.backend.sum(diff * diff)
        self.backend.eval(sum_sq)
        # Use sample variance (N-1) to match statistics.standard_deviation
        if token_count < 2:
            std_dev = 0.0
        else:
            variance = self._to_scalar(sum_sq) / float(token_count - 1)
            std_dev = math.sqrt(variance)

        # Max
        max_val = self.backend.max(arr)
        self.backend.eval(max_val)
        max_float = self._to_scalar(max_val)

        return EntropyTraceFeatures(
            token_count=token_count,
            mean=mean_float,
            std_dev=std_dev,
            max=max_float,
        )

    def feature_stats(
        self, points: list[list[float]], feature_names: list[str]
    ) -> list[FeatureStat]:
        """Compute per-feature statistics using GPU acceleration.

        Args:
            points: List of feature vectors [n_samples, n_features].
            feature_names: Names for each feature dimension.

        Returns:
            List of FeatureStat for each valid feature.
        """
        if not points:
            return []

        d = len(points[0])
        if d <= 0 or any(len(row) != d for row in points):
            return []

        names = feature_names if len(feature_names) == d else [f"feature_{i}" for i in range(d)]

        # Convert to backend array [n, d]
        # Filter out non-finite values per column
        stats: list[FeatureStat] = []

        for j in range(d):
            values = [
                float(row[j]) for row in points if row[j] is not None and math.isfinite(row[j])
            ]
            if not values:
                continue

            arr = self.backend.array(values)
            n_vals = len(values)
            mean_val = self.backend.mean(arr)
            self.backend.eval(mean_val)
            mean_float = self._to_scalar(mean_val)

            # Sample variance (N-1) to match statistics.standard_deviation
            diff = arr - mean_val
            sum_sq = self.backend.sum(diff * diff)
            self.backend.eval(sum_sq)
            if n_vals < 2:
                std_float = 0.0
            else:
                variance = self._to_scalar(sum_sq) / float(n_vals - 1)
                std_float = math.sqrt(variance)

            stats.append(FeatureStat(index=j, name=names[j], mean=mean_float, std_dev=std_float))

        return stats

    def summarize_prior_tension(
        self, samples: list[EntropyDeltaSample]
    ) -> PriorTensionSummary | None:
        """Summarize prior tension samples using GPU acceleration.

        Args:
            samples: List of EntropyDeltaSample objects.

        Returns:
            PriorTensionSummary with statistics, or None if empty.
        """
        if not samples:
            return None

        surprisal = [float(s.base_surprisal) for s in samples if s.base_surprisal is not None]
        approval_prob = [
            float(s.base_approval_probability)
            for s in samples
            if s.base_approval_probability is not None
        ]
        normalized = [
            float(s.normalized_approval_score)
            for s in samples
            if s.normalized_approval_score is not None
        ]

        # Disagreement rate
        disagreement_rate = None
        if samples:
            disagree = sum(1 for s in samples if s.top_token_disagreement)
            disagreement_rate = float(disagree) / float(len(samples))

        def compute_mean(values: list[float]) -> float | None:
            if not values:
                return None
            arr = self.backend.array(values)
            result = self.backend.mean(arr)
            self.backend.eval(result)
            return self._to_scalar(result)

        def compute_percentile(values: list[float], p: float) -> float | None:
            if not values:
                return None
            # Sort and compute percentile
            arr = self.backend.array(values)
            sorted_arr = self.backend.sort(arr)
            self.backend.eval(sorted_arr)
            sorted_list = sorted_arr.tolist()
            return statistics.percentile(sorted_list, p)

        return PriorTensionSummary(
            token_count=len(samples),
            mean_base_surprisal=compute_mean(surprisal),
            p95_base_surprisal=compute_percentile(surprisal, 0.95),
            mean_base_approval_probability=compute_mean(approval_prob),
            p05_base_approval_probability=compute_percentile(approval_prob, 0.05),
            mean_normalized_approval=compute_mean(normalized),
            p05_normalized_approval=compute_percentile(normalized, 0.05),
            top_token_disagreement_rate=disagreement_rate,
        )

    def estimate_id(
        self,
        points: list[list[float]],
        bootstrap_resamples: int | None = None,
        seed: int = 42,
        use_regression: bool = True,
    ) -> IDEstimateSummary:
        """Estimate intrinsic dimensionality.

        This delegates to IntrinsicDimension which already uses Backend
        operations internally.

        Args:
            points: Data points [n_samples, n_features].
            bootstrap_resamples: Number of bootstrap resamples for CI (unused).
            seed: Random seed for reproducibility (unused).
            use_regression: Whether to use regression-based estimation.

        Returns:
            IDEstimateSummary with intrinsic dimension estimate.
        """
        # Note: bootstrap_resamples and seed are kept for API compatibility
        # but TwoNNConfiguration doesn't currently support bootstrap CI
        _ = bootstrap_resamples, seed  # Unused, kept for future compatibility
        estimate = IntrinsicDimension.compute_two_nn(
            points,
            configuration=TwoNNConfiguration(
                use_regression=use_regression,
            ),
        )
        return IDEstimateSummary.from_estimate(estimate)

    def _to_scalar(self, val: Any) -> float:
        """Convert backend scalar to Python float."""
        if hasattr(val, "item"):
            return float(val.item())
        if hasattr(val, "tolist"):
            result = val.tolist()
            return float(result) if not isinstance(result, list) else float(result[0])
        return float(val)


def get_manifold_dimensionality(
    backend: "Backend | None" = None,
) -> type[ManifoldDimensionality] | BackendManifoldDimensionality:
    """Get the best available manifold dimensionality implementation.

    Args:
        backend: Optional Backend instance. If provided, returns
                 BackendManifoldDimensionality for GPU acceleration.

    Returns:
        ManifoldDimensionality class or BackendManifoldDimensionality instance.

    Example:
        >>> from modelcypher.core.domain._backend import get_default_backend
        >>> backend = get_default_backend()
        >>> md = get_manifold_dimensionality(backend)
        >>> features = md.entropy_trace_features([1.0, 2.0, 3.0])
    """
    if backend is not None:
        return BackendManifoldDimensionality(backend)
    return ManifoldDimensionality
