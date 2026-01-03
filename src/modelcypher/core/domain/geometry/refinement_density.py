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

"""
Refinement Density Analysis: Per-layer scoring for geometric refinement.

Combines signals from:
- DARE sparsity (what fraction of weights are essential)
- DoRA directional drift (how much the feature space rotated)
- Transition CKA (how aligned layer transitions are between models)

A higher refinement density score indicates the layer is more refined in the
source model relative to the target model. This module reports measurements only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from modelcypher.core.domain.geometry.concept_response_matrix import (
    LayerTransitionResult,
    TransitionExperiment,
)
from modelcypher.core.domain.geometry.dare_sparsity import (
    LayerSparsityMetrics,
    SparsityAnalysis,
)
from modelcypher.core.domain.geometry.dora_decomposition import (
    DecompositionResult,
    MagnitudeDirectionMetrics,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayerRefinementScore:
    """Refinement density score for a single layer."""

    layer_name: str
    layer_index: int
    sparsity_contribution: float | None
    directional_contribution: float | None
    transition_contribution: float | None
    composite_score: float
    component_count: int
    raw_sparsity: float | None = None
    raw_directional_drift: float | None = None
    raw_transition_cka: float | None = None
    raw_state_cka: float | None = None


@dataclass
class RefinementDensityResult:
    """Complete refinement density analysis result."""

    source_model: str
    target_model: str
    computed_at: datetime

    # Per-layer scores
    layer_scores: dict[int, LayerRefinementScore]

    # Aggregate metrics
    mean_composite_score: float
    max_composite_score: float
    std_composite_score: float
    scored_layer_count: int

    # Normalization context (data-derived)
    max_directional_drift: float
    max_transition_advantage: float

    # Component availability
    has_sparsity_data: bool
    has_directional_data: bool
    has_transition_data: bool

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "sourceModel": self.source_model,
            "targetModel": self.target_model,
            "computedAt": self.computed_at.isoformat(),
            "meanCompositeScore": self.mean_composite_score,
            "stdCompositeScore": self.std_composite_score,
            "maxCompositeScore": self.max_composite_score,
            "scoredLayerCount": self.scored_layer_count,
            "normalization": {
                "maxDirectionalDrift": self.max_directional_drift,
                "maxTransitionAdvantage": self.max_transition_advantage,
            },
            "layerScores": {
                str(idx): {
                    "layerName": score.layer_name,
                    "layerIndex": score.layer_index,
                    "compositeScore": score.composite_score,
                    "componentCount": score.component_count,
                    "sparsityContribution": score.sparsity_contribution,
                    "directionalContribution": score.directional_contribution,
                    "transitionContribution": score.transition_contribution,
                    "rawSparsity": score.raw_sparsity,
                    "rawDirectionalDrift": score.raw_directional_drift,
                    "rawTransitionCKA": score.raw_transition_cka,
                    "rawStateCKA": score.raw_state_cka,
                }
                for idx, score in self.layer_scores.items()
            },
            "hasSparsityData": self.has_sparsity_data,
            "hasDirectionalData": self.has_directional_data,
            "hasTransitionData": self.has_transition_data,
        }


class RefinementDensityAnalyzer:
    """Analyzes refinement density across layers using multiple geometric signals."""

    def analyze(
        self,
        source_model: str,
        target_model: str,
        sparsity_analysis: SparsityAnalysis | None = None,
        dora_result: DecompositionResult | None = None,
        transition_experiment: TransitionExperiment | None = None,
        layer_count: int | None = None,
    ) -> RefinementDensityResult:
        """Perform refinement density analysis across all layers."""
        inferred_count = self._infer_layer_count(
            sparsity_analysis, dora_result, transition_experiment
        )
        effective_count = layer_count or inferred_count
        if effective_count == 0:
            logger.warning("No layers found in refinement density inputs.")
            return self._empty_result(source_model, target_model)

        sparsity_by_layer = self._index_sparsity(sparsity_analysis)
        dora_by_layer = self._index_dora(dora_result)
        transition_by_layer = self._index_transition(transition_experiment)

        max_directional_drift = max(
            (metric.directional_drift for metric in dora_by_layer.values()),
            default=0.0,
        )
        max_transition_advantage = max(
            (self._transition_ratio(t) for t in transition_by_layer.values()),
            default=0.0,
        )

        layer_scores: dict[int, LayerRefinementScore] = {}
        for layer_idx in range(effective_count):
            score = self._compute_layer_score(
                layer_idx,
                sparsity_by_layer.get(layer_idx),
                dora_by_layer.get(layer_idx),
                transition_by_layer.get(layer_idx),
                max_directional_drift,
                max_transition_advantage,
            )
            layer_scores[layer_idx] = score

        scored_values = [
            s.composite_score for s in layer_scores.values() if s.component_count > 0
        ]
        if scored_values:
            mean_score = sum(scored_values) / len(scored_values)
            max_score = max(scored_values)
            variance = sum((s - mean_score) ** 2 for s in scored_values) / len(scored_values)
            std_score = variance**0.5
            scored_count = len(scored_values)
        else:
            mean_score = 0.0
            max_score = 0.0
            std_score = 0.0
            scored_count = 0

        return RefinementDensityResult(
            source_model=source_model,
            target_model=target_model,
            computed_at=datetime.now(timezone.utc),
            layer_scores=layer_scores,
            mean_composite_score=mean_score,
            max_composite_score=max_score,
            std_composite_score=std_score,
            scored_layer_count=scored_count,
            max_directional_drift=max_directional_drift,
            max_transition_advantage=max_transition_advantage,
            has_sparsity_data=bool(sparsity_by_layer),
            has_directional_data=bool(dora_by_layer),
            has_transition_data=bool(transition_by_layer),
        )

    def analyze_from_weights(
        self,
        source_model: str,
        target_model: str,
        base_weights: dict[str, any],
        adapted_weights: dict[str, any],
        transition_experiment: TransitionExperiment | None = None,
    ) -> RefinementDensityResult:
        """Convenience method to compute DARE and DoRA internally and analyze."""
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.geometry.dare_sparsity import (
            Configuration as DAREConfig,
        )
        from modelcypher.core.domain.geometry.dare_sparsity import (
            DARESparsityAnalyzer,
        )
        from modelcypher.core.domain.geometry.dora_decomposition import (
            DoRADecomposition,
        )

        b = get_default_backend()

        delta_weights: dict[str, list[float]] = {}
        for name in base_weights:
            if name not in adapted_weights:
                continue
            base = base_weights[name]
            adapted = adapted_weights[name]
            if hasattr(base, "shape") and hasattr(adapted, "shape"):
                if base.shape != adapted.shape:
                    continue
                delta = adapted - base
                if hasattr(delta, "flatten"):
                    delta = delta.flatten()
                if hasattr(delta, "tolist") or hasattr(delta, "shape"):
                    delta_weights[name] = b.tolist(delta)
                else:
                    delta_weights[name] = list(delta)

        sparsity_analysis = DARESparsityAnalyzer.analyze(
            delta_weights, DAREConfig(compute_per_layer_metrics=True)
        )

        base_arr: dict[str, any] = {}
        adapted_arr: dict[str, any] = {}
        for name in base_weights:
            if name not in adapted_weights:
                continue
            base_arr[name] = b.array(base_weights[name])
            adapted_arr[name] = b.array(adapted_weights[name])

        dora = DoRADecomposition(backend=b)
        dora_result = dora.analyze_adapter(base_arr, adapted_arr)

        return self.analyze(
            source_model=source_model,
            target_model=target_model,
            sparsity_analysis=sparsity_analysis,
            dora_result=dora_result,
            transition_experiment=transition_experiment,
        )

    def _compute_layer_score(
        self,
        layer_idx: int,
        sparsity: LayerSparsityMetrics | None,
        dora: MagnitudeDirectionMetrics | None,
        transition: LayerTransitionResult | None,
        max_directional_drift: float,
        max_transition_advantage: float,
    ) -> LayerRefinementScore:
        """Compute refinement score for a single layer."""
        raw_sparsity = None
        sparsity_contrib = None
        if sparsity is not None:
            raw_sparsity = sparsity.sparsity
            sparsity_contrib = max(0.0, min(1.0, 1.0 - sparsity.sparsity))

        raw_drift = None
        directional_contrib = None
        if dora is not None:
            raw_drift = dora.directional_drift
            if max_directional_drift > 0:
                directional_contrib = max(
                    0.0, min(1.0, dora.directional_drift / max_directional_drift)
                )
            else:
                directional_contrib = 0.0

        raw_transition = None
        raw_state = None
        transition_contrib = None
        if transition is not None:
            raw_transition = transition.transition_cka
            raw_state = transition.state_cka
            ratio = self._transition_ratio(transition)
            if max_transition_advantage > 0:
                transition_contrib = max(0.0, min(1.0, ratio / max_transition_advantage))
            else:
                transition_contrib = 0.0

        contributions = [
            c for c in (sparsity_contrib, directional_contrib, transition_contrib) if c is not None
        ]
        component_count = len(contributions)
        if component_count:
            composite = sum(contributions) / component_count
        else:
            composite = 0.0

        layer_name = f"layer_{layer_idx}"
        if sparsity is not None:
            layer_name = sparsity.layer_name
        elif dora is not None:
            layer_name = dora.layer_name

        return LayerRefinementScore(
            layer_name=layer_name,
            layer_index=layer_idx,
            sparsity_contribution=sparsity_contrib,
            directional_contribution=directional_contrib,
            transition_contribution=transition_contrib,
            composite_score=max(0.0, min(1.0, composite)),
            component_count=component_count,
            raw_sparsity=raw_sparsity,
            raw_directional_drift=raw_drift,
            raw_transition_cka=raw_transition,
            raw_state_cka=raw_state,
        )

    @staticmethod
    def _transition_ratio(transition: LayerTransitionResult) -> float:
        if transition.state_cka > 0:
            return transition.transition_cka / transition.state_cka
        return transition.transition_cka

    def _infer_layer_count(
        self,
        sparsity: SparsityAnalysis | None,
        dora: DecompositionResult | None,
        transition: TransitionExperiment | None,
    ) -> int:
        """Infer layer count from available inputs."""
        counts = []

        if sparsity and sparsity.per_layer_sparsity:
            indices = set()
            for key in sparsity.per_layer_sparsity.keys():
                idx = self._extract_layer_index(key)
                if idx is not None:
                    indices.add(idx)
            if indices:
                counts.append(max(indices) + 1)

        if dora and dora.per_layer_metrics:
            indices = set()
            for key in dora.per_layer_metrics.keys():
                idx = self._extract_layer_index(key)
                if idx is not None:
                    indices.add(idx)
            if indices:
                counts.append(max(indices) + 1)

        if transition and transition.transitions:
            max_layer = max(t.to_layer for t in transition.transitions)
            counts.append(max_layer + 1)

        return max(counts) if counts else 0

    def _index_sparsity(self, sparsity: SparsityAnalysis | None) -> dict[int, LayerSparsityMetrics]:
        """Index sparsity metrics by layer index."""
        if sparsity is None or not sparsity.per_layer_sparsity:
            return {}

        result: dict[int, LayerSparsityMetrics] = {}
        for key, metrics in sparsity.per_layer_sparsity.items():
            idx = self._extract_layer_index(key)
            if idx is not None:
                if idx not in result or metrics.essential_fraction > result[idx].essential_fraction:
                    result[idx] = metrics
        return result

    def _index_dora(self, dora: DecompositionResult | None) -> dict[int, MagnitudeDirectionMetrics]:
        """Index DoRA metrics by layer index."""
        if dora is None or not dora.per_layer_metrics:
            return {}

        result: dict[int, MagnitudeDirectionMetrics] = {}
        for key, metrics in dora.per_layer_metrics.items():
            idx = self._extract_layer_index(key)
            if idx is not None:
                if idx not in result or metrics.directional_drift > result[idx].directional_drift:
                    result[idx] = metrics
        return result

    def _index_transition(
        self, transition: TransitionExperiment | None
    ) -> dict[int, LayerTransitionResult]:
        """Index transition results by layer index."""
        if transition is None or not transition.transitions:
            return {}

        return {t.from_layer: t for t in transition.transitions}

    @staticmethod
    def _extract_layer_index(key: str) -> int | None:
        """Extract layer index from a weight key like 'layers.5.mlp.gate_proj.weight'."""
        parts = key.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return None

    def _empty_result(self, source: str, target: str) -> RefinementDensityResult:
        """Return an empty result when no data is available."""
        return RefinementDensityResult(
            source_model=source,
            target_model=target,
            computed_at=datetime.now(timezone.utc),
            layer_scores={},
            mean_composite_score=0.0,
            max_composite_score=0.0,
            std_composite_score=0.0,
            scored_layer_count=0,
            max_directional_drift=0.0,
            max_transition_advantage=0.0,
            has_sparsity_data=False,
            has_directional_data=False,
            has_transition_data=False,
        )


# =============================================================================
# Metric Keys for Training Progress Emission
# =============================================================================


class RefinementMetricKey:
    """Metric keys for geometry tracking."""

    MEAN_COMPOSITE = "geometry/refinement_mean_composite"
    MAX_COMPOSITE = "geometry/refinement_max_composite"
    SCORED_LAYER_COUNT = "geometry/refinement_scored_layer_count"


def to_metrics_dict(result: RefinementDensityResult) -> dict[str, float]:
    """Convert refinement result to metrics dictionary."""
    return {
        RefinementMetricKey.MEAN_COMPOSITE: result.mean_composite_score,
        RefinementMetricKey.MAX_COMPOSITE: result.max_composite_score,
        RefinementMetricKey.SCORED_LAYER_COUNT: float(result.scored_layer_count),
    }
