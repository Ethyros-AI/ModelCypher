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

from modelcypher.core.domain.entropy.entropy_delta_sample import EntropyDeltaSample
from modelcypher.core.domain.geometry.intrinsic_dimension import (
    BootstrapConfiguration,
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
        bootstrap = (
            BootstrapConfiguration(resamples=bootstrap_resamples, confidence_level=0.95, seed=seed)
            if bootstrap_resamples is not None and bootstrap_resamples > 0
            else None
        )
        estimate = IntrinsicDimension.compute_two_nn(
            points,
            configuration=TwoNNConfiguration(
                use_regression=use_regression,
                bootstrap=bootstrap,
            ),
        )
        return IDEstimateSummary.from_estimate(estimate)
