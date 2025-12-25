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
Refinement Density Analysis: Per-layer scoring for selective model merging.

Combines signals from:
- DARE sparsity (what fraction of weights are essential)
- DoRA directional drift (how much the feature space rotated)
- Transition CKA (how aligned layer transitions are between models)

A high refinement density score indicates the layer is "more refined" in the source
model and should be preferentially selected during merging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

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


class MergeRecommendation(str, Enum):
    """Recommended merge strategy for a layer based on refinement density."""

    HARD_SWAP = "hard_swap"  # Take entirely from source (refinement >> baseline)
    HIGH_ALPHA = "high_alpha"  # Strong preference for source (alpha 0.7-0.9)
    MEDIUM_ALPHA = "medium_alpha"  # Balanced blend (alpha 0.4-0.6)
    LOW_ALPHA = "low_alpha"  # Prefer target (alpha 0.2-0.4)
    SKIP = "skip"  # Leave as target (refinement negligible)


class RefinementLevel(str, Enum):
    """Qualitative refinement level for human interpretation."""

    HIGHLY_REFINED = "highly_refined"
    MODERATELY_REFINED = "moderately_refined"
    LIGHTLY_REFINED = "lightly_refined"
    MINIMAL = "minimal"


@dataclass(frozen=True)
class LayerRefinementScore:
    """Refinement density score for a single layer."""

    layer_name: str
    layer_index: int

    # Component contributions (0.0 to 1.0)
    sparsity_contribution: float  # 1 - sparsity (higher = more essential params)
    directional_contribution: float  # Normalized directional drift
    transition_contribution: float  # Normalized transition advantage

    # Composite score (0.0 to 1.0)
    composite_score: float

    # Interpretation
    refinement_level: RefinementLevel
    merge_recommendation: MergeRecommendation
    recommended_alpha: float

    # Raw metrics for inspection
    raw_sparsity: float | None = None
    raw_directional_drift: float | None = None
    raw_transition_cka: float | None = None
    raw_state_cka: float | None = None

    @property
    def interpretation(self) -> str:
        """Human-readable interpretation of this layer's refinement."""
        if self.refinement_level == RefinementLevel.HIGHLY_REFINED:
            return f"Layer {self.layer_index} is highly refined (score={self.composite_score:.2f}). Recommend hard swap or high alpha."
        elif self.refinement_level == RefinementLevel.MODERATELY_REFINED:
            return f"Layer {self.layer_index} is moderately refined (score={self.composite_score:.2f}). Use balanced blending."
        elif self.refinement_level == RefinementLevel.LIGHTLY_REFINED:
            return f"Layer {self.layer_index} has light refinement (score={self.composite_score:.2f}). Prefer target with some source."
        else:
            return f"Layer {self.layer_index} shows minimal refinement (score={self.composite_score:.2f}). Skip or use low alpha."


@dataclass(frozen=True)
class RefinementDensityConfig:
    """Configuration for refinement density analysis."""

    # Equal component weights
    sparsity_weight: float = 1.0 / 3.0
    directional_weight: float = 1.0 / 3.0
    transition_weight: float = 1.0 / 3.0

    # Thresholds for merge recommendations
    hard_swap_threshold: float = 0.80  # Score >= this → hard swap
    high_alpha_threshold: float = 0.60  # Score >= this → high alpha
    medium_alpha_threshold: float = 0.35  # Score >= this → medium alpha
    low_alpha_threshold: float = 0.15  # Score >= this → low alpha
    # Below low_alpha_threshold → skip

    # Alpha mapping
    hard_swap_alpha: float = 0.05  # Near-complete source takeover
    high_alpha_range: tuple[float, float] = (0.15, 0.30)  # Low alpha = more source
    medium_alpha_range: tuple[float, float] = (0.35, 0.55)
    low_alpha_range: tuple[float, float] = (0.60, 0.80)
    skip_alpha: float = 0.95  # Near-complete target retention

    # Normalization parameters
    max_directional_drift: float = 0.5  # Drift values above this are clipped
    max_transition_advantage: float = 2.0  # Transition ratio above this is clipped

    @staticmethod
    def default() -> "RefinementDensityConfig":
        return RefinementDensityConfig()

    @staticmethod
    def aggressive() -> "RefinementDensityConfig":
        """More aggressive about porting refined layers."""
        return RefinementDensityConfig(
            hard_swap_threshold=0.70,
            high_alpha_threshold=0.50,
            medium_alpha_threshold=0.25,
            low_alpha_threshold=0.10,
        )

    @staticmethod
    def conservative() -> "RefinementDensityConfig":
        """More conservative, prefer blending over hard swaps."""
        return RefinementDensityConfig(
            hard_swap_threshold=0.95,  # Rarely hard swap
            high_alpha_threshold=0.75,
            medium_alpha_threshold=0.50,
            low_alpha_threshold=0.25,
        )


@dataclass
class RefinementDensityResult:
    """Complete refinement density analysis result."""

    source_model: str
    target_model: str
    computed_at: datetime
    config: RefinementDensityConfig

    # Per-layer scores
    layer_scores: dict[int, LayerRefinementScore]

    # Aggregate metrics
    mean_composite_score: float
    max_composite_score: float
    layers_above_hard_swap: int
    layers_above_high_alpha: int
    layers_above_medium_alpha: int

    # Component availability
    has_sparsity_data: bool
    has_directional_data: bool
    has_transition_data: bool

    @property
    def hard_swap_layers(self) -> list[int]:
        """Layer indices recommended for hard swap."""
        return sorted(
            idx
            for idx, score in self.layer_scores.items()
            if score.merge_recommendation == MergeRecommendation.HARD_SWAP
        )

    @property
    def high_alpha_layers(self) -> list[int]:
        """Layer indices recommended for high alpha."""
        return sorted(
            idx
            for idx, score in self.layer_scores.items()
            if score.merge_recommendation == MergeRecommendation.HIGH_ALPHA
        )

    @property
    def alpha_by_layer(self) -> dict[int, float]:
        """Recommended alpha value for each layer."""
        return {idx: score.recommended_alpha for idx, score in self.layer_scores.items()}

    @property
    def interpretation(self) -> str:
        """Human-readable summary of the analysis."""
        total = len(self.layer_scores)
        if total == 0:
            return "No layers analyzed."

        hard_pct = (self.layers_above_hard_swap / total) * 100
        high_pct = (self.layers_above_high_alpha / total) * 100
        med_pct = (self.layers_above_medium_alpha / total) * 100

        lines = [
            f"Refinement Density Analysis: {self.source_model} → {self.target_model}",
            f"Mean Score: {self.mean_composite_score:.3f}, Max: {self.max_composite_score:.3f}",
            f"Hard Swap Candidates: {self.layers_above_hard_swap}/{total} ({hard_pct:.1f}%)",
            f"High Alpha Candidates: {self.layers_above_high_alpha}/{total} ({high_pct:.1f}%)",
            f"Medium+ Alpha Candidates: {self.layers_above_medium_alpha}/{total} ({med_pct:.1f}%)",
        ]

        if self.hard_swap_layers:
            lines.append(f"Recommended Hard Swap Layers: {self.hard_swap_layers}")

        components = []
        if self.has_sparsity_data:
            components.append("DARE")
        if self.has_directional_data:
            components.append("DoRA")
        if self.has_transition_data:
            components.append("Transition-CKA")
        lines.append(f"Data Sources: {', '.join(components) or 'None'}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "sourceModel": self.source_model,
            "targetModel": self.target_model,
            "computedAt": self.computed_at.isoformat(),
            "meanCompositeScore": self.mean_composite_score,
            "maxCompositeScore": self.max_composite_score,
            "layersAboveHardSwap": self.layers_above_hard_swap,
            "layersAboveHighAlpha": self.layers_above_high_alpha,
            "layersAboveMediumAlpha": self.layers_above_medium_alpha,
            "hardSwapLayers": self.hard_swap_layers,
            "highAlphaLayers": self.high_alpha_layers,
            "alphaByLayer": self.alpha_by_layer,
            "layerScores": {
                str(idx): {
                    "layerName": score.layer_name,
                    "layerIndex": score.layer_index,
                    "compositeScore": score.composite_score,
                    "sparsityContribution": score.sparsity_contribution,
                    "directionalContribution": score.directional_contribution,
                    "transitionContribution": score.transition_contribution,
                    "refinementLevel": score.refinement_level.value,
                    "mergeRecommendation": score.merge_recommendation.value,
                    "recommendedAlpha": score.recommended_alpha,
                }
                for idx, score in self.layer_scores.items()
            },
            "hasSparsityData": self.has_sparsity_data,
            "hasDirectionalData": self.has_directional_data,
            "hasTransitionData": self.has_transition_data,
        }


class RefinementDensityAnalyzer:
    """
    Analyzes refinement density across layers to guide selective merging.

    Combines three signals:
    1. DARE sparsity: Low sparsity = many essential parameters = more refined
    2. DoRA drift: High directional drift = learning new features = more refined
    3. Transition CKA: High transition alignment = important capability = worth porting

    Usage:
        analyzer = RefinementDensityAnalyzer()
        result = analyzer.analyze(
            source_model="model_A",
            target_model="model_B",
            sparsity_analysis=dare_result,
            dora_result=dora_result,
            transition_experiment=transition_result,
        )
        print(result.interpretation)
        alphas = result.alpha_by_layer
    """

    def __init__(self, config: RefinementDensityConfig | None = None):
        self.config = config or RefinementDensityConfig.default()

    def analyze(
        self,
        source_model: str,
        target_model: str,
        sparsity_analysis: SparsityAnalysis | None = None,
        dora_result: DecompositionResult | None = None,
        transition_experiment: TransitionExperiment | None = None,
        layer_count: int | None = None,
    ) -> RefinementDensityResult:
        """
        Perform refinement density analysis across all layers.

        Args:
            source_model: Identifier for the source (refined) model
            target_model: Identifier for the target (base) model
            sparsity_analysis: DARE sparsity analysis result
            dora_result: DoRA decomposition result
            transition_experiment: CKA transition experiment result
            layer_count: Override layer count (inferred from inputs if not provided)

        Returns:
            RefinementDensityResult with per-layer scores and recommendations
        """
        # Infer layer count from available data
        inferred_count = self._infer_layer_count(
            sparsity_analysis, dora_result, transition_experiment
        )
        effective_count = layer_count or inferred_count
        if effective_count == 0:
            logger.warning("No layers found in refinement density inputs.")
            return self._empty_result(source_model, target_model)

        # Build per-layer indices
        sparsity_by_layer = self._index_sparsity(sparsity_analysis)
        dora_by_layer = self._index_dora(dora_result)
        transition_by_layer = self._index_transition(transition_experiment)

        # Compute scores for each layer
        layer_scores: dict[int, LayerRefinementScore] = {}
        for layer_idx in range(effective_count):
            score = self._compute_layer_score(
                layer_idx,
                sparsity_by_layer.get(layer_idx),
                dora_by_layer.get(layer_idx),
                transition_by_layer.get(layer_idx),
            )
            layer_scores[layer_idx] = score

        # Aggregate metrics
        composite_scores = [s.composite_score for s in layer_scores.values()]
        mean_score = sum(composite_scores) / len(composite_scores) if composite_scores else 0.0
        max_score = max(composite_scores) if composite_scores else 0.0

        cfg = self.config
        above_hard = sum(1 for s in composite_scores if s >= cfg.hard_swap_threshold)
        above_high = sum(1 for s in composite_scores if s >= cfg.high_alpha_threshold)
        above_med = sum(1 for s in composite_scores if s >= cfg.medium_alpha_threshold)

        return RefinementDensityResult(
            source_model=source_model,
            target_model=target_model,
            computed_at=datetime.now(timezone.utc),
            config=self.config,
            layer_scores=layer_scores,
            mean_composite_score=mean_score,
            max_composite_score=max_score,
            layers_above_hard_swap=above_hard,
            layers_above_high_alpha=above_high,
            layers_above_medium_alpha=above_med,
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
        """
        Convenience method to compute DARE and DoRA internally and analyze.

        Args:
            source_model: Identifier for the source model
            target_model: Identifier for the target model
            base_weights: Base model weights (dict of layer_name → weight array)
            adapted_weights: Adapted model weights
            transition_experiment: Optional CKA transition data

        Returns:
            RefinementDensityResult
        """
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

        # Compute delta weights for DARE
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
                if hasattr(delta, "tolist"):
                    delta_weights[name] = delta.tolist()
                else:
                    delta_weights[name] = list(delta)

        sparsity_analysis = DARESparsityAnalyzer.analyze(
            delta_weights, DAREConfig(compute_per_layer_metrics=True)
        )

        # Compute DoRA decomposition
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
    ) -> LayerRefinementScore:
        """Compute refinement score for a single layer."""
        cfg = self.config

        # Sparsity contribution: 1 - sparsity (more essential = higher score)
        raw_sparsity = None
        if sparsity is not None:
            raw_sparsity = sparsity.sparsity
            sparsity_contrib = 1.0 - sparsity.sparsity
        else:
            sparsity_contrib = 0.5  # Neutral if missing

        # Directional contribution: normalized drift
        raw_drift = None
        if dora is not None:
            raw_drift = dora.directional_drift
            normalized_drift = min(dora.directional_drift / cfg.max_directional_drift, 1.0)
            directional_contrib = normalized_drift
        else:
            directional_contrib = 0.5  # Neutral if missing

        # Transition contribution: transition_cka / state_cka ratio
        raw_transition = None
        raw_state = None
        if transition is not None:
            raw_transition = transition.transition_cka
            raw_state = transition.state_cka
            if transition.state_cka > 0.001:
                ratio = transition.transition_cka / transition.state_cka
                normalized_ratio = min(ratio / cfg.max_transition_advantage, 1.0)
                transition_contrib = normalized_ratio
            else:
                transition_contrib = transition.transition_cka  # Use raw if no state baseline
        else:
            transition_contrib = 0.5  # Neutral if missing

        # Composite score with configured weights
        composite = (
            cfg.sparsity_weight * sparsity_contrib
            + cfg.directional_weight * directional_contrib
            + cfg.transition_weight * transition_contrib
        )
        composite = max(0.0, min(1.0, composite))

        # Determine merge recommendation
        recommendation, alpha = self._score_to_recommendation(composite)

        # Determine refinement level
        level = self._score_to_level(composite)

        # Layer name
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
            composite_score=composite,
            refinement_level=level,
            merge_recommendation=recommendation,
            recommended_alpha=alpha,
            raw_sparsity=raw_sparsity,
            raw_directional_drift=raw_drift,
            raw_transition_cka=raw_transition,
            raw_state_cka=raw_state,
        )

    def _score_to_recommendation(self, score: float) -> tuple[MergeRecommendation, float]:
        """Map composite score to merge recommendation and alpha value."""
        cfg = self.config

        if score >= cfg.hard_swap_threshold:
            return MergeRecommendation.HARD_SWAP, cfg.hard_swap_alpha

        if score >= cfg.high_alpha_threshold:
            # Interpolate within high alpha range
            t = (score - cfg.high_alpha_threshold) / (
                cfg.hard_swap_threshold - cfg.high_alpha_threshold
            )
            alpha = cfg.high_alpha_range[1] - t * (
                cfg.high_alpha_range[1] - cfg.high_alpha_range[0]
            )
            return MergeRecommendation.HIGH_ALPHA, alpha

        if score >= cfg.medium_alpha_threshold:
            t = (score - cfg.medium_alpha_threshold) / (
                cfg.high_alpha_threshold - cfg.medium_alpha_threshold
            )
            alpha = cfg.medium_alpha_range[1] - t * (
                cfg.medium_alpha_range[1] - cfg.medium_alpha_range[0]
            )
            return MergeRecommendation.MEDIUM_ALPHA, alpha

        if score >= cfg.low_alpha_threshold:
            t = (score - cfg.low_alpha_threshold) / (
                cfg.medium_alpha_threshold - cfg.low_alpha_threshold
            )
            alpha = cfg.low_alpha_range[1] - t * (cfg.low_alpha_range[1] - cfg.low_alpha_range[0])
            return MergeRecommendation.LOW_ALPHA, alpha

        return MergeRecommendation.SKIP, cfg.skip_alpha

    def _score_to_level(self, score: float) -> RefinementLevel:
        """Map composite score to qualitative refinement level."""
        if score >= 0.70:
            return RefinementLevel.HIGHLY_REFINED
        if score >= 0.45:
            return RefinementLevel.MODERATELY_REFINED
        if score >= 0.25:
            return RefinementLevel.LIGHTLY_REFINED
        return RefinementLevel.MINIMAL

    def _infer_layer_count(
        self,
        sparsity: SparsityAnalysis | None,
        dora: DecompositionResult | None,
        transition: TransitionExperiment | None,
    ) -> int:
        """Infer layer count from available inputs."""
        counts = []

        if sparsity and sparsity.per_layer_sparsity:
            # Extract layer indices from keys like "layers.0.mlp.gate_proj.weight"
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
                # Take the first metric for each layer (or aggregate if multiple)
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
                # Take highest drift for the layer
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
            config=self.config,
            layer_scores={},
            mean_composite_score=0.0,
            max_composite_score=0.0,
            layers_above_hard_swap=0,
            layers_above_high_alpha=0,
            layers_above_medium_alpha=0,
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
    HARD_SWAP_COUNT = "geometry/refinement_hard_swap_count"
    HIGH_ALPHA_COUNT = "geometry/refinement_high_alpha_count"


def to_metrics_dict(result: RefinementDensityResult) -> dict[str, float]:
    """Convert refinement result to metrics dictionary."""
    return {
        RefinementMetricKey.MEAN_COMPOSITE: result.mean_composite_score,
        RefinementMetricKey.MAX_COMPOSITE: result.max_composite_score,
        RefinementMetricKey.HARD_SWAP_COUNT: float(result.layers_above_hard_swap),
        RefinementMetricKey.HIGH_ALPHA_COUNT: float(result.layers_above_high_alpha),
    }
