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

import logging
from dataclasses import dataclass, field
from datetime import datetime

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.dare_sparsity import SparsityAnalysis
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    precision_dtype,
)

logger = logging.getLogger(__name__)


# =============================================================================
# NO CONFIGURATION CLASSES
# =============================================================================
# All parameters are derived from data:
# - sparsity_threshold: derived from maximum gap in sparsity distribution
# - DARE alignment: always computed when dare_analysis is provided
# =============================================================================


@dataclass(frozen=True)
class DAREAlignment:
    high_droppability_layers: list[int]
    overlap_with_sparse: float


@dataclass(frozen=True)
class LayerActivationStats:
    layer_index: int
    mean_activation: float
    max_activation: float
    activation_variance: float
    prompt_count: int


@dataclass(frozen=True)
class AnalysisResult:
    layer_sparsity: dict[int, float]
    sparse_layers: list[int]
    skip_layers: list[int]
    sparsity_threshold: float
    dare_alignment: DAREAlignment | None
    domain: str
    analyzed_at: datetime = field(default_factory=datetime.utcnow)

    def generate_report(self) -> str:
        report_lines = [
            "# Sparse Region Analysis Report",
            "",
            "## Overview",
            f"- Domain: {self.domain}",
            f"- Analyzed: {self.analyzed_at}",
            f"- Total Layers: {len(self.layer_sparsity)}",
            f"- Sparse Layers: {len(self.sparse_layers)}",
            f"- Skip Layers: {len(self.skip_layers)}",
            f"- Sparsity Threshold: {self.sparsity_threshold:.4f}",
            "",
            "## Layer Sparsity",
        ]

        for layer in sorted(self.layer_sparsity):
            sparsity = self.layer_sparsity[layer]
            marker = ""
            if layer in self.sparse_layers:
                marker = " [SPARSE]"
            elif layer in self.skip_layers:
                marker = " [SKIP]"
            report_lines.append(f"- Layer {layer}: {sparsity:.2f}{marker}")

        if self.dare_alignment:
            report_lines.extend(
                [
                    "",
                    "## DARE Alignment",
                    f"- High Droppability Layers: {self.dare_alignment.high_droppability_layers}",
                    f"- Overlap with Sparse: {self.dare_alignment.overlap_with_sparse * 100:.1f}%",
                ]
            )

        return "\n".join(report_lines)


class SparseRegionLocator:
    """Locates sparse regions in neural network layers.

    All parameters are derived from data - no configuration needed.
    """

    @staticmethod
    def _derive_sparsity_threshold(values: list[float]) -> float:
        if len(values) < 2:
            raise ValueError("Need at least two sparsity values to derive a threshold")
        sorted_vals = sorted(values)
        gaps = [sorted_vals[i + 1] - sorted_vals[i] for i in range(len(sorted_vals) - 1)]
        max_gap = max(gaps) if gaps else 0.0
        if max_gap <= 0:
            # No separable gap - use median as fallback threshold
            # This handles the case where all layers have similar sparsity
            return sorted_vals[len(sorted_vals) // 2]
        gap_index = gaps.index(max_gap)
        return (sorted_vals[gap_index] + sorted_vals[gap_index + 1]) / 2.0

    def analyze(
        self,
        domain_stats: list[LayerActivationStats],
        baseline_stats: list[LayerActivationStats],
        domain: str,
        dare_analysis: SparsityAnalysis | None = None,
    ) -> AnalysisResult:
        domain_by_layer = {stat.layer_index: stat for stat in domain_stats}
        baseline_by_layer = {stat.layer_index: stat for stat in baseline_stats}

        layer_sparsity: dict[int, float] = {}
        all_layers = sorted(set(domain_by_layer) | set(baseline_by_layer))
        for layer in all_layers:
            domain_stat = domain_by_layer.get(layer)
            baseline_stat = baseline_by_layer.get(layer)
            if domain_stat is None or baseline_stat is None:
                continue
            if baseline_stat.mean_activation <= 0:
                raise ValueError(
                    "Baseline mean activation must be positive to compute sparsity ratios"
                )
            ratio = domain_stat.mean_activation / baseline_stat.mean_activation
            sparsity = max(0.0, min(1.0, 1.0 - ratio))
            layer_sparsity[layer] = sparsity

        if not layer_sparsity:
            raise ValueError("No comparable layer statistics available for sparsity analysis")

        # Always derive sparsity threshold from data (maximum gap in distribution)
        sparsity_threshold = self._derive_sparsity_threshold(list(layer_sparsity.values()))

        sparse_layers = sorted(
            layer
            for layer, sparsity in layer_sparsity.items()
            if sparsity >= sparsity_threshold
        )
        b = get_default_backend()
        eps = machine_epsilon(b, b.array([1.0], dtype=precision_dtype(b)))
        skip_layers = sorted(
            layer for layer, sparsity in layer_sparsity.items() if sparsity <= eps
        )

        # Always compute DARE alignment when dare_analysis is provided
        dare_alignment = (
            self._compute_dare_alignment(sparse_layers, dare_analysis)
            if dare_analysis is not None
            else None
        )

        logger.info(
            "Sparse region analysis for %s: %s sparse layers, %s skip layers",
            domain,
            len(sparse_layers),
            len(skip_layers),
        )

        return AnalysisResult(
            layer_sparsity=layer_sparsity,
            sparse_layers=sparse_layers,
            skip_layers=skip_layers,
            sparsity_threshold=sparsity_threshold,
            dare_alignment=dare_alignment,
            domain=domain,
        )

    def analyze_from_activations(
        self,
        domain_activations: list[dict[int, float]],
        baseline_activations: list[dict[int, float]],
        domain: str,
        dare_analysis: SparsityAnalysis | None = None,
    ) -> AnalysisResult:
        domain_stats = self._aggregate_activations(domain_activations)
        baseline_stats = self._aggregate_activations(baseline_activations)
        return self.analyze(
            domain_stats=domain_stats,
            baseline_stats=baseline_stats,
            domain=domain,
            dare_analysis=dare_analysis,
        )

    def _aggregate_activations(
        self, activations: list[dict[int, float]]
    ) -> list[LayerActivationStats]:
        if not activations:
            return []

        layer_values: dict[int, list[float]] = {}
        for prompt_activations in activations:
            for layer, value in prompt_activations.items():
                layer_values.setdefault(layer, []).append(float(value))

        stats: list[LayerActivationStats] = []
        for layer, values in layer_values.items():
            mean = sum(values) / float(len(values))
            max_val = max(values) if values else 0.0
            variance = (
                sum((value - mean) ** 2 for value in values) / float(max(1, len(values) - 1))
                if len(values) > 1
                else 0.0
            )
            stats.append(
                LayerActivationStats(
                    layer_index=layer,
                    mean_activation=mean,
                    max_activation=max_val,
                    activation_variance=variance,
                    prompt_count=len(values),
                )
            )
        stats.sort(key=lambda item: item.layer_index)
        return stats

    def _compute_dare_alignment(
        self,
        sparse_layers: list[int],
        dare_analysis: SparsityAnalysis | None,
    ) -> DAREAlignment | None:
        if dare_analysis is None:
            return None

        high_droppability: list[int] = []
        for layer_name, metrics in dare_analysis.per_layer_sparsity.items():
            # Any layer with positive sparsity is droppable; magnitude in metrics
            if metrics.sparsity <= 0:
                continue
            for component in layer_name.split("."):
                if component.isdigit():
                    high_droppability.append(int(component))
                    break

        sparse_set = set(sparse_layers)
        dare_set = set(high_droppability)
        if sparse_set and dare_set:
            overlap = len(sparse_set & dare_set) / float(max(len(sparse_set), len(dare_set)))
        else:
            overlap = 0.0

        return DAREAlignment(
            high_droppability_layers=sorted(set(high_droppability)),
            overlap_with_sparse=overlap,
        )
