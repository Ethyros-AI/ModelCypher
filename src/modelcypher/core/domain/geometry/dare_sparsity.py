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
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


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
    def analyze_with_backend(
        delta_weights: dict[str, Any],
        backend: "Backend | None" = None,
        configuration: Configuration = Configuration(),
    ) -> SparsityAnalysis:
        """GPU-accelerated DARE sparsity analysis via Backend protocol."""
        b = backend or get_default_backend()

        filtered = (
            {
                name: arr
                for name, arr in delta_weights.items()
                if configuration.analysis_layers and name in configuration.analysis_layers
            }
            if configuration.analysis_layers
            else delta_weights
        )

        if not filtered:
            return DARESparsityAnalyzer._empty_analysis()

        per_layer_arrays: dict[str, Any] = {}
        global_max = 0.0
        total_count = 0
        total_sum = 0.0
        total_sum_sq = 0.0
        non_zero_count = 0
        min_non_zero = float("inf")

        for name, arr in filtered.items():
            flat = b.abs(b.reshape(arr, (-1,)))
            b.eval(flat)
            per_layer_arrays[name] = flat

            layer_count = int(flat.shape[0])
            layer_max_arr = b.max(flat)
            layer_mean_arr = b.mean(flat)
            layer_sum_sq_arr = b.sum(flat * flat)
            layer_non_zero_arr = b.sum(flat > 0)
            b.eval(layer_max_arr, layer_mean_arr, layer_sum_sq_arr, layer_non_zero_arr)

            layer_max = float(b.to_numpy(layer_max_arr).item())
            layer_mean = float(b.to_numpy(layer_mean_arr).item())
            layer_non_zero = int(b.to_numpy(layer_non_zero_arr).item())

            if layer_non_zero > 0:
                sorted_layer = b.sort(flat)
                first_nz_idx = layer_count - layer_non_zero
                layer_min_nz = sorted_layer[first_nz_idx : first_nz_idx + 1]
                b.eval(layer_min_nz)
                min_non_zero = min(min_non_zero, float(b.to_numpy(layer_min_nz).item()))

            global_max = max(global_max, layer_max)
            total_count += layer_count
            total_sum += layer_mean * layer_count
            total_sum_sq += float(b.to_numpy(layer_sum_sq_arr).item())
            non_zero_count += layer_non_zero

        if total_count == 0 or global_max == 0:
            return DARESparsityAnalyzer._empty_analysis()

        if min_non_zero == float("inf"):
            min_non_zero = 0.0

        mean_val = total_sum / total_count
        variance = (total_sum_sq / total_count) - (mean_val**2)
        std_dev = variance**0.5 if variance > 0 else 0.0
        threshold_by_magnitude = global_max * configuration.sparsity_threshold

        def count_below(threshold: float) -> int:
            total = 0
            for flat in per_layer_arrays.values():
                cnt = b.sum(flat <= threshold)
                b.eval(cnt)
                total += int(b.to_numpy(cnt).item())
            return total

        def find_percentile(p: float) -> float:
            target = int(p * total_count)
            lo, hi = 0.0, global_max
            for _ in range(40):
                mid = (lo + hi) / 2
                if count_below(mid) < target:
                    lo = mid
                else:
                    hi = mid
            return (lo + hi) / 2

        p1 = find_percentile(0.01)
        p5 = find_percentile(0.05)
        median = find_percentile(0.50)
        p95 = find_percentile(0.95)
        p99 = find_percentile(0.99)
        threshold_by_percentile = find_percentile(configuration.droppable_percentile)
        drop_threshold = max(threshold_by_magnitude, threshold_by_percentile)

        total_droppable = 0
        per_layer_metrics: dict[str, LayerSparsityMetrics] = {}

        for name, flat in per_layer_arrays.items():
            layer_count = int(flat.shape[0])
            layer_max_arr = b.max(flat)
            layer_mean_arr = b.mean(flat)
            b.eval(layer_max_arr, layer_mean_arr)

            layer_max = float(b.to_numpy(layer_max_arr).item())
            layer_mean = float(b.to_numpy(layer_mean_arr).item())

            if layer_max == 0:
                layer_droppable = layer_count
            else:
                droppable_arr = b.sum(flat <= drop_threshold)
                b.eval(droppable_arr)
                layer_droppable = int(b.to_numpy(droppable_arr).item())

            total_droppable += layer_droppable
            layer_sparsity = float(layer_droppable) / float(layer_count) if layer_count > 0 else 1.0

            if configuration.compute_per_layer_metrics:
                per_layer_metrics[name] = LayerSparsityMetrics(
                    layer_name=name,
                    parameter_count=layer_count,
                    sparsity=layer_sparsity,
                    mean_magnitude=layer_mean,
                    max_magnitude=layer_max,
                    essential_fraction=1.0 - layer_sparsity,
                    has_significant_updates=layer_sparsity < 0.9,
                )

        effective_sparsity = float(total_droppable) / float(total_count)

        return SparsityAnalysis(
            total_parameters=total_count,
            non_zero_parameters=non_zero_count,
            effective_sparsity=effective_sparsity,
            essential_fraction=1.0 - effective_sparsity,
            per_layer_sparsity=per_layer_metrics,
            magnitude_stats=MagnitudeStatistics(
                mean=mean_val,
                standard_deviation=std_dev,
                median=median,
                max=global_max,
                min_non_zero=min_non_zero,
                percentile1=p1,
                percentile5=p5,
                percentile95=p95,
                percentile99=p99,
            ),
            recommended_drop_rate=DARESparsityAnalyzer._compute_recommended_drop_rate(effective_sparsity),
            quality_assessment=DARESparsityAnalyzer._assess_quality(effective_sparsity),
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
