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
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    find_magnitude_gap_threshold,
    ulp_scalar,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


# Thresholds derived from data; analyze all layers with per-layer metrics.


@dataclass(frozen=True)
class LayerSparsityMetrics:
    layer_name: str
    parameter_count: int
    sparsity: float
    mean_magnitude: float
    max_magnitude: float
    essential_fraction: float
    has_non_zero_updates: bool


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


@dataclass(frozen=True)
class SparsityAnalysis:
    """DARE sparsity analysis result.

    Higher effective_sparsity values indicate more droppable/sparse weights.
    """

    total_parameters: int
    non_zero_parameters: int
    effective_sparsity: float
    essential_fraction: float
    per_layer_sparsity: dict[str, LayerSparsityMetrics]
    magnitude_stats: MagnitudeStatistics
    computed_at: datetime


class DARESparsityAnalyzer:
    @staticmethod
    def analyze(
        delta_weights: dict[str, list[float]],
    ) -> SparsityAnalysis:
        """Analyze sparsity of delta weights. All layers, all metrics."""
        # Always analyze all layers - no filtering
        filtered = delta_weights

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

        # Derive thresholds from data, not arbitrary constants:
        # 1. Machine epsilon threshold: values below eps * max are numerical noise
        eps = ulp_scalar(1.0, get_default_backend())
        zero_threshold = magnitude_stats.max * eps

        # 2. Spectral gap: find natural break in magnitude distribution
        gap_threshold = find_magnitude_gap_threshold(sorted_magnitudes, eps=eps)

        # Use the larger of the two thresholds
        drop_threshold = max(zero_threshold, gap_threshold)

        backend = get_default_backend()
        mag_arr = backend.array(sorted_magnitudes)
        total_count = len(sorted_magnitudes)

        if magnitude_stats.max == 0:
            droppable_count = total_count
            non_zero_count = 0
        else:
            # Vectorized counting
            droppable_arr = backend.sum(mag_arr <= drop_threshold)
            non_zero_arr = backend.sum(mag_arr > 0)
            backend.eval(droppable_arr, non_zero_arr)
            droppable_count = int(backend.to_scalar(droppable_arr))
            non_zero_count = int(backend.to_scalar(non_zero_arr))

        effective_sparsity = float(droppable_count) / float(total_count)
        essential_fraction = 1.0 - effective_sparsity

        # Always compute per-layer metrics
        per_layer_metrics: dict[str, LayerSparsityMetrics] = {}
        for name, (magnitudes, _) in per_layer_data.items():
            per_layer_metrics[name] = DARESparsityAnalyzer._compute_layer_metrics(
                layer_name=name,
                magnitudes=magnitudes,
                drop_threshold=drop_threshold,
            )

        return SparsityAnalysis(
            total_parameters=total_count,
            non_zero_parameters=non_zero_count,
            effective_sparsity=effective_sparsity,
            essential_fraction=essential_fraction,
            per_layer_sparsity=per_layer_metrics,
            magnitude_stats=magnitude_stats,
            computed_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def analyze_with_backend(
        delta_weights: dict[str, Any],
        backend: "Backend | None" = None,
    ) -> SparsityAnalysis:
        """GPU-accelerated DARE sparsity analysis via Backend protocol."""
        b = backend or get_default_backend()

        # Always analyze all layers - no filtering
        filtered = delta_weights

        if not filtered:
            return DARESparsityAnalyzer._empty_analysis()

        per_layer_arrays: dict[str, Any] = {}
        global_max = 0.0
        total_count = 0
        total_sum = 0.0
        total_sum_sq = 0.0
        non_zero_count = 0
        min_non_zero = float("inf")
        inf_val = float(b.finfo().max)

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

            layer_max = float(b.to_scalar(layer_max_arr))
            layer_mean = float(b.to_scalar(layer_mean_arr))
            layer_non_zero = int(b.to_scalar(layer_non_zero_arr))

            if layer_non_zero > 0:
                masked = b.where(flat > 0, flat, b.full(flat.shape, inf_val))
                layer_min_nz_arr = b.min(masked)
                b.eval(layer_min_nz_arr)
                min_non_zero = min(min_non_zero, float(b.to_scalar(layer_min_nz_arr)))

            global_max = max(global_max, layer_max)
            total_count += layer_count
            total_sum += layer_mean * layer_count
            total_sum_sq += float(b.to_scalar(layer_sum_sq_arr))
            non_zero_count += layer_non_zero

        if total_count == 0 or global_max == 0:
            return DARESparsityAnalyzer._empty_analysis()

        if min_non_zero == float("inf"):
            min_non_zero = 0.0

        mean_val = total_sum / total_count
        variance = (total_sum_sq / total_count) - (mean_val**2)
        std_dev = variance**0.5 if variance > 0 else 0.0

        # Derive thresholds from data:
        # 1. Machine epsilon threshold (values below this are numerical noise)
        eps = b.finfo(next(iter(per_layer_arrays.values())).dtype).eps
        zero_threshold = global_max * eps

        all_magnitudes = b.concatenate(list(per_layer_arrays.values()))
        b.eval(all_magnitudes)
        sorted_mags = b.sort(all_magnitudes)
        b.eval(sorted_mags)

        def find_percentile_sorted(p: float) -> float:
            kth = max(0, min(int(p * total_count), total_count - 1))
            val = b.take(sorted_mags, b.array([kth]), axis=0)
            val = b.squeeze(val)
            b.eval(val)
            return float(b.to_scalar(val))

        p1 = find_percentile_sorted(0.01)
        p5 = find_percentile_sorted(0.05)
        median = find_percentile_sorted(0.50)
        p95 = find_percentile_sorted(0.95)
        p99 = find_percentile_sorted(0.99)

        # 2. Spectral gap: find largest relative jump in sorted magnitudes
        gap_threshold = find_magnitude_gap_threshold(sorted_mags, eps=eps, backend=b)

        drop_threshold = max(zero_threshold, gap_threshold)

        total_droppable = 0
        per_layer_metrics: dict[str, LayerSparsityMetrics] = {}

        for name, flat in per_layer_arrays.items():
            layer_count = int(flat.shape[0])
            layer_max_arr = b.max(flat)
            layer_mean_arr = b.mean(flat)
            b.eval(layer_max_arr, layer_mean_arr)

            layer_max = float(b.to_scalar(layer_max_arr))
            layer_mean = float(b.to_scalar(layer_mean_arr))

            if layer_max == 0:
                layer_droppable = layer_count
            else:
                droppable_arr = b.sum(flat <= drop_threshold)
                b.eval(droppable_arr)
                layer_droppable = int(b.to_scalar(droppable_arr))

            total_droppable += layer_droppable
            layer_sparsity = float(layer_droppable) / float(layer_count) if layer_count > 0 else 1.0

            # Always compute per-layer metrics
            per_layer_metrics[name] = LayerSparsityMetrics(
                layer_name=name,
                parameter_count=layer_count,
                sparsity=layer_sparsity,
                mean_magnitude=layer_mean,
                max_magnitude=layer_max,
                essential_fraction=1.0 - layer_sparsity,
                has_non_zero_updates=layer_sparsity < 1.0,
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
            computed_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def identify_essential_parameters(
        delta_weights: dict[str, list[float]],
    ) -> dict[str, set[int]]:
        magnitudes = [abs(float(value)) for values in delta_weights.values() for value in values]
        if not magnitudes:
            return {name: set() for name in delta_weights}

        eps = ulp_scalar(1.0, get_default_backend())
        sorted_magnitudes = sorted(magnitudes)
        gap_threshold = find_magnitude_gap_threshold(sorted_magnitudes, eps=eps)
        zero_threshold = max(sorted_magnitudes) * eps
        drop_threshold = max(zero_threshold, gap_threshold)

        precision = ulp_scalar(drop_threshold, get_default_backend())
        result: dict[str, set[int]] = {}
        for name, deltas in delta_weights.items():
            essential_indices = {
                idx
                for idx, value in enumerate(deltas)
                if abs(float(value)) + precision >= drop_threshold
            }
            result[name] = essential_indices
        return result

    @staticmethod
    def to_metrics_dictionary(analysis: SparsityAnalysis) -> dict[str, float]:
        return {
            "geometry/dare_effective_sparsity": float(analysis.effective_sparsity),
            "geometry/dare_essential_fraction": float(analysis.essential_fraction),
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

        backend = get_default_backend()
        arr = backend.array(sorted_values)
        count = len(sorted_values)

        # Vectorized mean and std
        mean_arr = backend.mean(arr)
        backend.eval(mean_arr)
        mean = float(backend.to_scalar(mean_arr))

        centered = arr - mean
        variance_arr = backend.mean(centered * centered)
        backend.eval(variance_arr)
        variance = float(backend.to_scalar(variance_arr))
        std_dev = variance**0.5

        # Direct indexing for percentiles (array is already sorted)
        median = sorted_values[count // 2]
        max_value = sorted_values[-1]

        # Find min non-zero using backend masking
        inf_val = float(backend.finfo().max)
        masked = backend.where(arr > 0, arr, backend.full(arr.shape, inf_val))
        min_nz_arr = backend.min(masked)
        backend.eval(min_nz_arr)
        min_nz_val = float(backend.to_scalar(min_nz_arr))
        min_non_zero = 0.0 if min_nz_val == inf_val else min_nz_val

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
                has_non_zero_updates=False,
            )

        backend = get_default_backend()
        arr = backend.array(magnitudes)
        count = len(magnitudes)

        # Vectorized max
        max_arr = backend.max(arr)
        backend.eval(max_arr)
        max_value = float(backend.to_scalar(max_arr))

        if max_value == 0:
            droppable = count
        else:
            # Vectorized count of droppable values
            droppable_arr = backend.sum(arr <= drop_threshold)
            backend.eval(droppable_arr)
            droppable = int(backend.to_scalar(droppable_arr))

        sparsity = float(droppable) / float(count)

        # Vectorized mean
        mean_arr = backend.mean(arr)
        backend.eval(mean_arr)
        mean = float(backend.to_scalar(mean_arr))

        return LayerSparsityMetrics(
            layer_name=layer_name,
            parameter_count=count,
            sparsity=float(sparsity),
            mean_magnitude=float(mean),
            max_magnitude=float(max_value),
            essential_fraction=float(1.0 - sparsity),
            has_non_zero_updates=sparsity < 1.0,
        )
