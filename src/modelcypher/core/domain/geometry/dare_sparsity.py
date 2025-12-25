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
from datetime import datetime, timezone
from enum import Enum


@dataclass(frozen=True)
class Configuration:
    sparsity_threshold: float = 0.01
    droppable_percentile: float = 0.95
    analysis_layers: set[str] | None = None
    compute_per_layer_metrics: bool = True

    @staticmethod
    def default() -> "Configuration":
        return Configuration()

    @staticmethod
    def aggressive() -> "Configuration":
        return Configuration(sparsity_threshold=0.001, droppable_percentile=0.99)


@dataclass(frozen=True)
class LayerSparsityMetrics:
    layer_name: str
    parameter_count: int
    sparsity: float
    mean_magnitude: float
    max_magnitude: float
    essential_fraction: float
    has_significant_updates: bool


@dataclass(frozen=True)
class MagnitudeStatistics:
    mean: float
    standard_deviation: float
    median: float
    max: float
    min_non_zero: float
    percentile1: float
    percentile5: float
    percentile95: float
    percentile99: float


class QualityAssessment(str, Enum):
    excellent_for_merging = "excellentForMerging"
    good = "good"
    moderate = "moderate"
    dense = "dense"
    concerningly_dense = "concerninglyDense"


@dataclass(frozen=True)
class SparsityAnalysis:
    total_parameters: int
    non_zero_parameters: int
    effective_sparsity: float
    essential_fraction: float
    per_layer_sparsity: dict[str, LayerSparsityMetrics]
    magnitude_stats: MagnitudeStatistics
    recommended_drop_rate: float
    quality_assessment: QualityAssessment
    computed_at: datetime


class DARESparsityAnalyzer:
    @staticmethod
    def analyze(
        delta_weights: dict[str, list[float]], configuration: Configuration = Configuration()
    ) -> SparsityAnalysis:
        filtered = (
            {
                name: values
                for name, values in delta_weights.items()
                if configuration.analysis_layers and name in configuration.analysis_layers
            }
            if configuration.analysis_layers
            else delta_weights
        )

        all_magnitudes: list[float] = []
        per_layer_data: dict[str, tuple[list[float], int]] = {}

        for name, deltas in filtered.items():
            magnitudes = [abs(float(value)) for value in deltas]
            all_magnitudes.extend(magnitudes)
            per_layer_data[name] = (magnitudes, len(deltas))

        if not all_magnitudes:
            return DARESparsityAnalyzer._empty_analysis()

        sorted_magnitudes = sorted(all_magnitudes)
        magnitude_stats = DARESparsityAnalyzer._compute_magnitude_stats(sorted_magnitudes)

        threshold_by_magnitude = magnitude_stats.max * configuration.sparsity_threshold
        percentile_index = int(len(sorted_magnitudes) * configuration.droppable_percentile)
        threshold_by_percentile = sorted_magnitudes[
            min(percentile_index, len(sorted_magnitudes) - 1)
        ]
        drop_threshold = max(threshold_by_magnitude, threshold_by_percentile)

        if magnitude_stats.max == 0:
            droppable_count = len(sorted_magnitudes)
        else:
            droppable_count = sum(1 for value in sorted_magnitudes if value <= drop_threshold)

        total_count = len(sorted_magnitudes)
        effective_sparsity = float(droppable_count) / float(total_count)
        essential_fraction = 1.0 - effective_sparsity

        per_layer_metrics: dict[str, LayerSparsityMetrics] = {}
        if configuration.compute_per_layer_metrics:
            for name, (magnitudes, _) in per_layer_data.items():
                per_layer_metrics[name] = DARESparsityAnalyzer._compute_layer_metrics(
                    layer_name=name,
                    magnitudes=magnitudes,
                    drop_threshold=drop_threshold,
                )

        recommended_drop_rate = DARESparsityAnalyzer._compute_recommended_drop_rate(
            effective_sparsity=effective_sparsity
        )
        quality_assessment = DARESparsityAnalyzer._assess_quality(effective_sparsity)

        return SparsityAnalysis(
            total_parameters=total_count,
            non_zero_parameters=sum(1 for value in sorted_magnitudes if value > 0),
            effective_sparsity=effective_sparsity,
            essential_fraction=essential_fraction,
            per_layer_sparsity=per_layer_metrics,
            magnitude_stats=magnitude_stats,
            recommended_drop_rate=recommended_drop_rate,
            quality_assessment=quality_assessment,
            computed_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def identify_essential_parameters(
        delta_weights: dict[str, list[float]],
        threshold: float,
    ) -> dict[str, set[int]]:
        result: dict[str, set[int]] = {}
        for name, deltas in delta_weights.items():
            essential_indices = {
                idx for idx, value in enumerate(deltas) if abs(float(value)) >= threshold
            }
            result[name] = essential_indices
        return result

    @staticmethod
    def to_metrics_dictionary(analysis: SparsityAnalysis) -> dict[str, float]:
        return {
            "geometry/dare_effective_sparsity": float(analysis.effective_sparsity),
            "geometry/dare_essential_fraction": float(analysis.essential_fraction),
            "geometry/dare_recommended_drop_rate": float(analysis.recommended_drop_rate),
        }

    @staticmethod
    def _empty_analysis() -> SparsityAnalysis:
        return SparsityAnalysis(
            total_parameters=0,
            non_zero_parameters=0,
            effective_sparsity=1.0,
            essential_fraction=0.0,
            per_layer_sparsity={},
            magnitude_stats=MagnitudeStatistics(
                mean=0.0,
                standard_deviation=0.0,
                median=0.0,
                max=0.0,
                min_non_zero=0.0,
                percentile1=0.0,
                percentile5=0.0,
                percentile95=0.0,
                percentile99=0.0,
            ),
            recommended_drop_rate=0.0,
            quality_assessment=QualityAssessment.excellent_for_merging,
            computed_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _compute_magnitude_stats(sorted_values: list[float]) -> MagnitudeStatistics:
        if not sorted_values:
            return MagnitudeStatistics(
                mean=0.0,
                standard_deviation=0.0,
                median=0.0,
                max=0.0,
                min_non_zero=0.0,
                percentile1=0.0,
                percentile5=0.0,
                percentile95=0.0,
                percentile99=0.0,
            )

        count = len(sorted_values)
        mean = sum(sorted_values) / float(count)
        variance_sum = sum((value - mean) ** 2 for value in sorted_values)
        std_dev = (variance_sum / float(count)) ** 0.5
        median = sorted_values[count // 2]
        max_value = sorted_values[-1]
        min_non_zero = next((value for value in sorted_values if value > 0), 0.0)
        p1 = sorted_values[int(count * 0.01)]
        p5 = sorted_values[int(count * 0.05)]
        p95 = sorted_values[min(int(count * 0.95), count - 1)]
        p99 = sorted_values[min(int(count * 0.99), count - 1)]

        return MagnitudeStatistics(
            mean=float(mean),
            standard_deviation=float(std_dev),
            median=float(median),
            max=float(max_value),
            min_non_zero=float(min_non_zero),
            percentile1=float(p1),
            percentile5=float(p5),
            percentile95=float(p95),
            percentile99=float(p99),
        )

    @staticmethod
    def _compute_layer_metrics(
        layer_name: str,
        magnitudes: list[float],
        drop_threshold: float,
    ) -> LayerSparsityMetrics:
        if not magnitudes:
            return LayerSparsityMetrics(
                layer_name=layer_name,
                parameter_count=0,
                sparsity=1.0,
                mean_magnitude=0.0,
                max_magnitude=0.0,
                essential_fraction=0.0,
                has_significant_updates=False,
            )

        max_value = max(magnitudes)
        if max_value == 0:
            droppable = len(magnitudes)
        else:
            droppable = sum(1 for value in magnitudes if value <= drop_threshold)
        sparsity = float(droppable) / float(len(magnitudes))
        mean = sum(magnitudes) / float(len(magnitudes))

        return LayerSparsityMetrics(
            layer_name=layer_name,
            parameter_count=len(magnitudes),
            sparsity=float(sparsity),
            mean_magnitude=float(mean),
            max_magnitude=float(max_value),
            essential_fraction=float(1.0 - sparsity),
            has_significant_updates=sparsity < 0.9,
        )

    @staticmethod
    def _compute_recommended_drop_rate(effective_sparsity: float) -> float:
        if effective_sparsity > 0.95:
            return 0.95
        if effective_sparsity > 0.90:
            return 0.90
        if effective_sparsity > 0.80:
            return 0.85
        if effective_sparsity > 0.50:
            return effective_sparsity * 0.9
        return effective_sparsity * 0.5

    @staticmethod
    def _assess_quality(effective_sparsity: float) -> QualityAssessment:
        if effective_sparsity >= 0.95:
            return QualityAssessment.excellent_for_merging
        if effective_sparsity >= 0.80:
            return QualityAssessment.good
        if effective_sparsity >= 0.50:
            return QualityAssessment.moderate
        if effective_sparsity >= 0.20:
            return QualityAssessment.dense
        return QualityAssessment.concerningly_dense
