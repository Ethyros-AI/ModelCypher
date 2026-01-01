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

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime

from modelcypher.core.domain.geometry.dare_sparsity import SparsityAnalysis

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Configuration:
    base_rank: int
    sparsity_threshold: float | None
    use_dare_alignment: bool
    target_module_types: list[str]


@dataclass(frozen=True)
class LoRAConfigRecommendation:
    """LoRA configuration derived from sparse region analysis.

    Attributes
    ----------
    target_modules : list[str]
        Module names to target for LoRA.
    rank_by_layer : dict[int, int]
        Per-layer rank recommendations.
    skip_layers : list[int]
        Layers to skip.
    overall_rank : int
        Recommended overall rank.
    alpha : int
        LoRA alpha parameter.
    sparse_ratio : float
        Fraction of layers identified as sparse (0-1).
    """

    target_modules: list[str]
    rank_by_layer: dict[int, int]
    skip_layers: list[int]
    overall_rank: int
    alpha: int
    sparse_ratio: float

    def to_peft_config(self) -> dict[str, object]:
        return {
            "r": self.overall_rank,
            "lora_alpha": self.alpha,
            "target_modules": self.target_modules,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }

    def to_json(self) -> str:
        payload = {
            "targetModules": self.target_modules,
            "rankByLayer": self.rank_by_layer,
            "skipLayers": self.skip_layers,
            "overallRank": self.overall_rank,
            "alpha": self.alpha,
            "sparseRatio": self.sparse_ratio,
        }
        return json.dumps(payload, indent=2, sort_keys=True)


@dataclass(frozen=True)
class DAREAlignment:
    high_droppability_layers: list[int]
    overlap_with_sparse: float
    confidence: float


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
    recommendation: LoRAConfigRecommendation
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

        report_lines.extend(
            [
                "",
                "## LoRA Parameters",
                f"- Sparse Ratio: {self.recommendation.sparse_ratio:.1%}",
                f"- Overall Rank: {self.recommendation.overall_rank}",
                f"- Alpha: {self.recommendation.alpha}",
                f"- Target Modules: {', '.join(self.recommendation.target_modules)}",
            ]
        )

        if self.dare_alignment:
            report_lines.extend(
                [
                    "",
                    "## DARE Alignment",
                    f"- High Droppability Layers: {self.dare_alignment.high_droppability_layers}",
                    f"- Overlap with Sparse: {self.dare_alignment.overlap_with_sparse * 100:.1f}%",
                    f"- Confidence: {self.dare_alignment.confidence * 100:.1f}%",
                ]
            )

        return "\n".join(report_lines)


class SparseRegionLocator:
    def __init__(self, configuration: Configuration) -> None:
        self.config = configuration

    @staticmethod
    def _derive_sparsity_threshold(values: list[float]) -> float:
        if len(values) < 2:
            raise ValueError("Need at least two sparsity values to derive a threshold")
        sorted_vals = sorted(values)
        gaps = [sorted_vals[i + 1] - sorted_vals[i] for i in range(len(sorted_vals) - 1)]
        max_gap = max(gaps)
        if max_gap <= 0:
            raise ValueError("Sparsity values have no separable gap for threshold derivation")
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

        if self.config.sparsity_threshold is None:
            sparsity_threshold = self._derive_sparsity_threshold(list(layer_sparsity.values()))
        else:
            sparsity_threshold = self.config.sparsity_threshold

        sparse_layers = sorted(
            layer
            for layer, sparsity in layer_sparsity.items()
            if sparsity >= sparsity_threshold
        )
        min_sparsity = min(layer_sparsity.values())
        eps = sys.float_info.epsilon
        skip_layers = sorted(
            layer for layer, sparsity in layer_sparsity.items() if sparsity <= min_sparsity + eps
        )

        dare_alignment = (
            self._compute_dare_alignment(sparse_layers, dare_analysis)
            if self.config.use_dare_alignment
            else None
        )
        recommendation = self._generate_recommendation(
            layer_sparsity=layer_sparsity,
            sparse_layers=sparse_layers,
            skip_layers=skip_layers,
            dare_alignment=dare_alignment,
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
            recommendation=recommendation,
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
            if metrics.sparsity <= 0.8:
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

        confidence = overlap
        return DAREAlignment(
            high_droppability_layers=sorted(set(high_droppability)),
            overlap_with_sparse=overlap,
            confidence=confidence,
        )

    def _generate_recommendation(
        self,
        layer_sparsity: dict[int, float],
        sparse_layers: list[int],
        skip_layers: list[int],
        dare_alignment: DAREAlignment | None,
    ) -> LoRAConfigRecommendation:
        rank_by_layer: dict[int, int] = {}
        for layer, sparsity in layer_sparsity.items():
            if layer in skip_layers:
                continue
            rank = int(round(float(self.config.base_rank) * (1.0 - sparsity)))
            rank = max(1, rank)
            rank_by_layer[layer] = rank

        ranks = sorted(rank_by_layer.values())
        overall_rank = ranks[len(ranks) // 2] if ranks else self.config.base_rank
        alpha = overall_rank

        sparse_ratio = float(len(sparse_layers)) / float(max(1, len(layer_sparsity)))

        return LoRAConfigRecommendation(
            target_modules=self.config.target_module_types,
            rank_by_layer=rank_by_layer,
            skip_layers=skip_layers,
            overall_rank=overall_rank,
            alpha=alpha,
            sparse_ratio=sparse_ratio,
        )
