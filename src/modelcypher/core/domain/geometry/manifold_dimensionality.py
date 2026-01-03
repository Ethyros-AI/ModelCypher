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
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.entropy.entropy_delta_sample import EntropyDeltaSample
from modelcypher.core.domain.geometry.numerical_stability import (
    is_finite,
    sqrt_scalar,
)

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
    mean_base_logit_margin: float | None
    max_base_logit_margin: float | None
    mean_base_token_logit: float | None
    min_base_token_logit: float | None
    mean_base_rank_fraction: float | None
    min_base_rank_fraction: float | None
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
        _b = get_default_backend()
        cleaned = [
            float(value) for value in entropies if value is not None and is_finite(value, _b)
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
        _b = get_default_backend()
        for j in range(d):
            values = [
                float(row[j]) for row in points if row[j] is not None and is_finite(row[j], _b)
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

        logit_margin = [
            float(s.base_logit_margin) for s in samples if s.base_logit_margin is not None
        ]
        token_logit = [
            float(s.base_token_logit) for s in samples if s.base_token_logit is not None
        ]
        rank_fraction = [
            float(s.base_rank_fraction) for s in samples if s.base_rank_fraction is not None
        ]

        disagreement_rate = None
        if samples:
            disagree = sum(1 for s in samples if s.top_token_disagreement)
            disagreement_rate = float(disagree) / float(len(samples))

        def mean(values: list[float]) -> float | None:
            return sum(values) / float(len(values)) if values else None

        return PriorTensionSummary(
            token_count=len(samples),
            mean_base_logit_margin=mean(logit_margin),
            max_base_logit_margin=max(logit_margin) if logit_margin else None,
            mean_base_token_logit=mean(token_logit),
            min_base_token_logit=min(token_logit) if token_logit else None,
            mean_base_rank_fraction=mean(rank_fraction),
            min_base_rank_fraction=min(rank_fraction) if rank_fraction else None,
            top_token_disagreement_rate=disagreement_rate,
        )

    @staticmethod
    def estimate_id(
        points: list[list[float]],
        use_regression: bool = True,
    ) -> IDEstimateSummary:
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
            float(value) for value in entropies if value is not None and is_finite(value, self.backend)
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
            std_dev = sqrt_scalar(variance, self.backend)

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
                float(row[j]) for row in points if row[j] is not None and is_finite(row[j], self.backend)
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
                std_float = sqrt_scalar(variance, self.backend)

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

        logit_margin = [
            float(s.base_logit_margin) for s in samples if s.base_logit_margin is not None
        ]
        token_logit = [
            float(s.base_token_logit) for s in samples if s.base_token_logit is not None
        ]
        rank_fraction = [
            float(s.base_rank_fraction) for s in samples if s.base_rank_fraction is not None
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

        def compute_min(values: list[float]) -> float | None:
            if not values:
                return None
            arr = self.backend.array(values)
            result = self.backend.min(arr)
            self.backend.eval(result)
            return self._to_scalar(result)

        def compute_max(values: list[float]) -> float | None:
            if not values:
                return None
            arr = self.backend.array(values)
            result = self.backend.max(arr)
            self.backend.eval(result)
            return self._to_scalar(result)

        return PriorTensionSummary(
            token_count=len(samples),
            mean_base_logit_margin=compute_mean(logit_margin),
            max_base_logit_margin=compute_max(logit_margin),
            mean_base_token_logit=compute_mean(token_logit),
            min_base_token_logit=compute_min(token_logit),
            mean_base_rank_fraction=compute_mean(rank_fraction),
            min_base_rank_fraction=compute_min(rank_fraction),
            top_token_disagreement_rate=disagreement_rate,
        )

    def estimate_id(
        self,
        points: list[list[float]],
        use_regression: bool = True,
    ) -> IDEstimateSummary:
        """Estimate intrinsic dimensionality.

        This delegates to IntrinsicDimension which already uses Backend
        operations internally.

        Args:
            points: Data points [n_samples, n_features].
            use_regression: Whether to use regression-based estimation.

        Returns:
            IDEstimateSummary with intrinsic dimension estimate.
        """
        estimate = IntrinsicDimension.compute_two_nn(
            points,
            configuration=TwoNNConfiguration(
                use_regression=use_regression,
            ),
        )
        return IDEstimateSummary.from_estimate(estimate)

    def _to_scalar(self, val: Any) -> float:
        """Convert backend scalar to Python float."""
        if hasattr(val, "shape") or hasattr(val, "item") or hasattr(val, "tolist"):
            self.backend.eval(val)
            return float(self.backend.to_scalar(val))
        try:
            return float(val)
        except TypeError:
            if isinstance(val, (list, tuple)) and val:
                return float(val[0])
            return 0.0


def get_manifold_dimensionality(
    backend: "Backend | None" = None,
) -> type[ManifoldDimensionality] | BackendManifoldDimensionality:
    """Get the default manifold dimensionality implementation.

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
