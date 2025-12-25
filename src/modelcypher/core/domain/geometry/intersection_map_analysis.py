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

import sys
from dataclasses import dataclass

from modelcypher.core.domain.geometry.manifold_stitcher import (
    IntersectionMap,
    Thresholds,
)
from modelcypher.core.support import statistics


@dataclass(frozen=True)
class HistogramBin:
    lower_inclusive: float
    upper_exclusive: float
    count: int


@dataclass(frozen=True)
class OverallStats:
    pair_count: int
    mean_correlation: float
    standard_deviation_correlation: float | None
    min_correlation: float | None
    max_correlation: float | None
    strong_count: int
    moderate_count: int
    weak_count: int
    histogram: list[HistogramBin]


@dataclass(frozen=True)
class LayerStats:
    layer: int
    confidence: float | None
    count: int
    mean_correlation: float
    standard_deviation_correlation: float | None
    min_correlation: float | None
    max_correlation: float | None
    strong_count: int
    moderate_count: int
    weak_count: int


@dataclass(frozen=True)
class Analysis:
    overall_correlation: float
    aligned_dimension_count: int
    total_source_dims: int
    total_target_dims: int
    average_layer_confidence: float | None
    overall_stats: OverallStats
    per_layer: list[LayerStats]


@dataclass(frozen=True)
class DimensionPair:
    layer: int
    source_dim: int
    target_dim: int
    correlation: float


@dataclass(frozen=True)
class MarkdownReportOptions:
    input_label: str | None = None
    top_pairs: int = 25
    histogram_bins: int = 30


class IntersectionMapAnalysis:
    @staticmethod
    def analyze(map_data: IntersectionMap, histogram_bins: int = 30) -> Analysis:
        confidence_by_layer = {item.layer: item for item in map_data.layer_confidences}
        layers = sorted(
            set(map_data.dimension_correlations.keys()).union(confidence_by_layer.keys())
        )

        all_correlations: list[float] = []
        per_layer_stats: list[LayerStats] = []

        for layer in layers:
            correlations = map_data.dimension_correlations.get(layer, [])
            values: list[float] = []
            strong = 0
            moderate = 0
            weak = 0

            for pair in correlations:
                value = float(pair.correlation)
                if not (value == value):
                    continue
                values.append(value)
                if pair.is_strong_correlation:
                    strong += 1
                elif pair.is_moderate_correlation:
                    moderate += 1
                else:
                    weak += 1

            all_correlations.extend(values)
            mean = sum(values) / float(len(values)) if values else 0.0
            stdev = (
                statistics.standard_deviation_population(values, mean) if len(values) > 1 else None
            )

            per_layer_stats.append(
                LayerStats(
                    layer=layer,
                    confidence=confidence_by_layer.get(layer).confidence
                    if layer in confidence_by_layer
                    else None,
                    count=len(values),
                    mean_correlation=mean,
                    standard_deviation_correlation=stdev,
                    min_correlation=min(values) if values else None,
                    max_correlation=max(values) if values else None,
                    strong_count=strong,
                    moderate_count=moderate,
                    weak_count=weak,
                )
            )

        overall_mean = (
            sum(all_correlations) / float(len(all_correlations)) if all_correlations else 0.0
        )
        overall_stdev = (
            statistics.standard_deviation_population(all_correlations, overall_mean)
            if len(all_correlations) > 1
            else None
        )

        overall_strong = sum(item.strong_count for item in per_layer_stats)
        overall_moderate = sum(item.moderate_count for item in per_layer_stats)
        overall_weak = sum(item.weak_count for item in per_layer_stats)

        if all_correlations:
            min_val = min(all_correlations)
            max_val = max(all_correlations)
            hist_lower = min(0.0, min_val)
            hist_upper = max(1.0, max_val)
        else:
            hist_lower = 0.0
            hist_upper = 1.0

        histogram = IntersectionMapAnalysis._histogram(
            values=all_correlations,
            lower=hist_lower,
            upper=hist_upper,
            bins=histogram_bins,
        )

        if map_data.layer_confidences:
            avg_layer_confidence = sum(
                item.confidence for item in map_data.layer_confidences
            ) / float(len(map_data.layer_confidences))
        else:
            avg_layer_confidence = None

        return Analysis(
            overall_correlation=map_data.overall_correlation,
            aligned_dimension_count=map_data.aligned_dimension_count,
            total_source_dims=map_data.total_source_dims,
            total_target_dims=map_data.total_target_dims,
            average_layer_confidence=avg_layer_confidence,
            overall_stats=OverallStats(
                pair_count=len(all_correlations),
                mean_correlation=overall_mean,
                standard_deviation_correlation=overall_stdev,
                min_correlation=min(all_correlations) if all_correlations else None,
                max_correlation=max(all_correlations) if all_correlations else None,
                strong_count=overall_strong,
                moderate_count=overall_moderate,
                weak_count=overall_weak,
                histogram=histogram,
            ),
            per_layer=per_layer_stats,
        )

    @staticmethod
    def top_pairs(map_data: IntersectionMap, limit: int) -> list[DimensionPair]:
        if limit <= 0:
            return []
        out: list[DimensionPair] = []
        for layer, pairs in map_data.dimension_correlations.items():
            for pair in pairs:
                out.append(
                    DimensionPair(
                        layer=layer,
                        source_dim=pair.source_dim,
                        target_dim=pair.target_dim,
                        correlation=pair.correlation,
                    )
                )
        out.sort(
            key=lambda item: (
                -item.correlation,
                item.layer,
                item.source_dim,
                item.target_dim,
            )
        )
        return out[:limit]

    @staticmethod
    def render_markdown_report(
        map_data: IntersectionMap,
        options: MarkdownReportOptions = MarkdownReportOptions(),
    ) -> str:
        analysis = IntersectionMapAnalysis.analyze(map_data, histogram_bins=options.histogram_bins)
        top_pairs = IntersectionMapAnalysis.top_pairs(map_data, limit=options.top_pairs)

        def f3(value: float) -> str:
            return f"{value:.3f}"

        lines: list[str] = []
        lines.append("# Intersection Map Report\n")
        if options.input_label:
            lines.append(f"- Input: `{options.input_label}`\n")
        lines.append(f"- Source: `{map_data.source_model}`\n")
        lines.append(f"- Target: `{map_data.target_model}`\n")
        lines.append(f"- Overall correlation: **{f3(map_data.overall_correlation)}**\n")
        lines.append(
            f"- Aligned dimensions: **{map_data.aligned_dimension_count}** "
            f"(source {map_data.total_source_dims}, target {map_data.total_target_dims})\n"
        )

        lines.append("\n## Overall Statistics\n")
        lines.append(f"- Pairs: **{analysis.overall_stats.pair_count}**\n")
        lines.append(f"- Mean correlation: **{f3(analysis.overall_stats.mean_correlation)}**\n")
        if analysis.overall_stats.standard_deviation_correlation is not None:
            lines.append(
                f"- Std dev (population): **{f3(analysis.overall_stats.standard_deviation_correlation)}**\n"
            )
        if (
            analysis.overall_stats.min_correlation is not None
            and analysis.overall_stats.max_correlation is not None
        ):
            lines.append(
                f"- Min/Max: **{f3(analysis.overall_stats.min_correlation)} / "
                f"{f3(analysis.overall_stats.max_correlation)}**\n"
            )
        lines.append(
            f"- Breakdown: strong={analysis.overall_stats.strong_count} "
            f"(>{Thresholds.strong_correlation}), moderate={analysis.overall_stats.moderate_count} "
            f"(>{Thresholds.moderate_correlation}), weak={analysis.overall_stats.weak_count}\n"
        )

        lines.append("\n## Per-Layer Summary\n")
        lines.append(
            "| Layer | Confidence | Correlations | Mean corr | Strong | Moderate | Weak |\n"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|\n")
        for layer in sorted(analysis.per_layer, key=lambda item: item.layer):
            confidence = f3(layer.confidence) if layer.confidence is not None else "---"
            lines.append(
                f"| {layer.layer} | {confidence} | {layer.count} | "
                f"{f3(layer.mean_correlation)} | {layer.strong_count} | "
                f"{layer.moderate_count} | {layer.weak_count} |\n"
            )

        lines.append("\n## Top Dimension Correspondences\n")
        lines.append("| Rank | Layer | Source dim | Target dim | Corr |\n")
        lines.append("|---:|---:|---:|---:|---:|\n")
        for index, pair in enumerate(top_pairs):
            lines.append(
                f"| {index + 1} | {pair.layer} | {pair.source_dim} | {pair.target_dim} | "
                f"{f3(pair.correlation)} |\n"
            )

        return "".join(lines)

    @staticmethod
    def _histogram(
        values: list[float], lower: float, upper: float, bins: int
    ) -> list[HistogramBin]:
        cleaned = [value for value in values if value == value]
        if not cleaned:
            return []
        bin_count = max(1, bins)
        width = (upper - lower) / float(bin_count)
        if width <= 0:
            return []
        counts = [0 for _ in range(bin_count)]
        for value in cleaned:
            clamped = max(lower, min(upper - sys.float_info.epsilon, value))
            index = int((clamped - lower) / width)
            if 0 <= index < bin_count:
                counts[index] += 1

        bins_out: list[HistogramBin] = []
        for index in range(bin_count):
            lower_bound = lower + float(index) * width
            upper_bound = lower_bound + width
            bins_out.append(
                HistogramBin(
                    lower_inclusive=lower_bound,
                    upper_exclusive=upper_bound,
                    count=counts[index],
                )
            )
        return bins_out
