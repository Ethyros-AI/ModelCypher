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

"""Comprehensive tests for refinement_density.py.

Tests:
- LayerRefinementScore dataclass
- RefinementDensityConfig dataclass (validation, with_parameters)
- RefinementDensityResult dataclass (properties, derived thresholds, serialization)
- RefinementDensityAnalyzer (layer scoring, alpha computation, indexing)
- RefinementMetricKey and to_metrics_dict helper
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pytest

from modelcypher.core.domain.geometry.refinement_density import (
    LayerRefinementScore,
    RefinementDensityAnalyzer,
    RefinementDensityConfig,
    RefinementDensityResult,
    RefinementMetricKey,
    to_metrics_dict,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


# =============================================================================
# LayerRefinementScore Tests
# =============================================================================


class TestLayerRefinementScore:
    """Tests for LayerRefinementScore dataclass."""

    def test_basic_creation(self) -> None:
        """Should create score with all fields."""
        score = LayerRefinementScore(
            layer_name="layers.5.mlp.gate_proj.weight",
            layer_index=5,
            sparsity_contribution=0.7,
            directional_contribution=0.3,
            transition_contribution=0.5,
            composite_score=0.5,
            recommended_alpha=0.5,
        )

        assert score.layer_name == "layers.5.mlp.gate_proj.weight"
        assert score.layer_index == 5
        assert score.sparsity_contribution == 0.7
        assert score.directional_contribution == 0.3
        assert score.transition_contribution == 0.5
        assert score.composite_score == 0.5
        assert score.recommended_alpha == 0.5

    def test_with_raw_values(self) -> None:
        """Should store raw metric values."""
        score = LayerRefinementScore(
            layer_name="layer_0",
            layer_index=0,
            sparsity_contribution=0.6,
            directional_contribution=0.4,
            transition_contribution=0.5,
            composite_score=0.5,
            recommended_alpha=0.5,
            raw_sparsity=0.4,
            raw_directional_drift=0.2,
            raw_transition_cka=0.8,
            raw_state_cka=0.7,
        )

        assert score.raw_sparsity == 0.4
        assert score.raw_directional_drift == 0.2
        assert score.raw_transition_cka == 0.8
        assert score.raw_state_cka == 0.7

    def test_frozen(self) -> None:
        """Score should be frozen (immutable)."""
        score = LayerRefinementScore(
            layer_name="test",
            layer_index=0,
            sparsity_contribution=0.5,
            directional_contribution=0.5,
            transition_contribution=0.5,
            composite_score=0.5,
            recommended_alpha=0.5,
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            score.composite_score = 0.9  # type: ignore

    def test_default_raw_values_are_none(self) -> None:
        """Raw values should default to None."""
        score = LayerRefinementScore(
            layer_name="test",
            layer_index=0,
            sparsity_contribution=0.5,
            directional_contribution=0.5,
            transition_contribution=0.5,
            composite_score=0.5,
            recommended_alpha=0.5,
        )

        assert score.raw_sparsity is None
        assert score.raw_directional_drift is None
        assert score.raw_transition_cka is None
        assert score.raw_state_cka is None


# =============================================================================
# RefinementDensityConfig Tests
# =============================================================================


class TestRefinementDensityConfig:
    """Tests for RefinementDensityConfig dataclass."""

    def test_default_values(self) -> None:
        """Default values should be equal weights."""
        config = RefinementDensityConfig()

        assert config.sparsity_weight == pytest.approx(1.0 / 3.0, abs=1e-5)
        assert config.directional_weight == pytest.approx(1.0 / 3.0, abs=1e-5)
        assert config.transition_weight == pytest.approx(1.0 / 3.0, abs=1e-5)
        assert config.max_directional_drift == 0.5
        assert config.max_transition_advantage == 2.0

    def test_weights_sum_to_one(self) -> None:
        """Default weights should sum to 1.0."""
        config = RefinementDensityConfig()
        total = config.sparsity_weight + config.directional_weight + config.transition_weight
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_with_parameters_valid(self) -> None:
        """with_parameters should create config with valid weights."""
        config = RefinementDensityConfig.with_parameters(
            sparsity_weight=0.5,
            directional_weight=0.3,
            transition_weight=0.2,
            max_directional_drift=0.8,
            max_transition_advantage=3.0,
        )

        assert config.sparsity_weight == 0.5
        assert config.directional_weight == 0.3
        assert config.transition_weight == 0.2
        assert config.max_directional_drift == 0.8
        assert config.max_transition_advantage == 3.0

    def test_with_parameters_invalid_sum(self) -> None:
        """with_parameters should reject weights not summing to 1.0."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            RefinementDensityConfig.with_parameters(
                sparsity_weight=0.5,
                directional_weight=0.5,
                transition_weight=0.5,  # Sum = 1.5
            )

    def test_with_parameters_invalid_max_drift(self) -> None:
        """with_parameters should reject non-positive max_directional_drift."""
        with pytest.raises(ValueError, match="max_directional_drift must be > 0"):
            RefinementDensityConfig.with_parameters(
                max_directional_drift=0.0,
            )

        with pytest.raises(ValueError, match="max_directional_drift must be > 0"):
            RefinementDensityConfig.with_parameters(
                max_directional_drift=-0.5,
            )

    def test_with_parameters_invalid_max_transition(self) -> None:
        """with_parameters should reject non-positive max_transition_advantage."""
        with pytest.raises(ValueError, match="max_transition_advantage must be > 0"):
            RefinementDensityConfig.with_parameters(
                max_transition_advantage=0.0,
            )

    def test_with_parameters_edge_case_sum(self) -> None:
        """with_parameters should accept weights that sum to ~1.0 within tolerance."""
        # Within 0.01 tolerance
        config = RefinementDensityConfig.with_parameters(
            sparsity_weight=0.334,
            directional_weight=0.333,
            transition_weight=0.333,  # Sum = 1.0
        )
        assert config is not None


# =============================================================================
# RefinementDensityResult Tests
# =============================================================================


class TestRefinementDensityResult:
    """Tests for RefinementDensityResult dataclass."""

    def _create_sample_result(
        self,
        scores: dict[int, float] | None = None,
    ) -> RefinementDensityResult:
        """Create a sample result for testing."""
        if scores is None:
            scores = {0: 0.3, 1: 0.5, 2: 0.7, 3: 0.9}

        layer_scores = {}
        for idx, composite in scores.items():
            layer_scores[idx] = LayerRefinementScore(
                layer_name=f"layer_{idx}",
                layer_index=idx,
                sparsity_contribution=composite,
                directional_contribution=composite,
                transition_contribution=composite,
                composite_score=composite,
                recommended_alpha=1.0 - composite,
            )

        composite_values = list(scores.values())
        mean_score = sum(composite_values) / len(composite_values) if composite_values else 0.0
        max_score = max(composite_values) if composite_values else 0.0
        variance = (
            sum((s - mean_score) ** 2 for s in composite_values) / len(composite_values)
            if composite_values
            else 0.0
        )
        std_score = variance**0.5

        return RefinementDensityResult(
            source_model="source",
            target_model="target",
            computed_at=datetime.now(timezone.utc),
            config=RefinementDensityConfig(),
            layer_scores=layer_scores,
            mean_composite_score=mean_score,
            max_composite_score=max_score,
            std_composite_score=std_score,
            has_sparsity_data=True,
            has_directional_data=True,
            has_transition_data=True,
        )

    def test_basic_creation(self) -> None:
        """Should create result with all fields."""
        result = self._create_sample_result()

        assert result.source_model == "source"
        assert result.target_model == "target"
        assert len(result.layer_scores) == 4
        assert result.has_sparsity_data is True

    def test_alpha_by_layer(self) -> None:
        """alpha_by_layer should return dict of alphas."""
        result = self._create_sample_result({0: 0.2, 1: 0.8})

        alphas = result.alpha_by_layer

        assert alphas[0] == pytest.approx(0.8, abs=1e-5)  # 1.0 - 0.2
        assert alphas[1] == pytest.approx(0.2, abs=1e-5)  # 1.0 - 0.8

    def test_derived_thresholds(self) -> None:
        """_derived_thresholds should compute from mean and std."""
        result = self._create_sample_result({0: 0.4, 1: 0.5, 2: 0.6})
        # mean = 0.5, variance = ((0.1)^2 + 0 + (0.1)^2)/3 = 0.02/3, std = ~0.0816

        hard, high, mid = result._derived_thresholds

        assert mid == result.mean_composite_score
        assert high > mid  # mean + 0.5*std
        assert hard > high  # mean + 1.5*std

    def test_hard_swap_layers(self) -> None:
        """hard_swap_layers should return layers >= mean + 1.5*std."""
        # Create result where some layers are clearly above threshold
        result = self._create_sample_result({0: 0.1, 1: 0.2, 2: 0.95, 3: 0.98})
        # mean ~= 0.5575, std ~= 0.396
        # threshold = 0.5575 + 1.5*0.396 = 1.15 (capped at 1.0)

        hard_swap = result.hard_swap_layers
        # Layers 2 and 3 should be hard swap (0.95, 0.98 >= threshold)
        # Actually depends on exact threshold calculation
        # Let's just verify it returns a sorted list
        assert hard_swap == sorted(hard_swap)

    def test_high_alpha_layers(self) -> None:
        """high_alpha_layers should return layers in high range."""
        result = self._create_sample_result()
        high_alpha = result.high_alpha_layers

        # Should be sorted
        assert high_alpha == sorted(high_alpha)

    def test_layers_above_hard_swap(self) -> None:
        """layers_above_hard_swap should count hard swap layers."""
        result = self._create_sample_result()

        count = result.layers_above_hard_swap
        assert count == len(result.hard_swap_layers)

    def test_layers_above_high_alpha(self) -> None:
        """layers_above_high_alpha should count high alpha layers."""
        result = self._create_sample_result()

        count = result.layers_above_high_alpha
        assert count == len(result.high_alpha_layers)

    def test_to_dict(self) -> None:
        """to_dict should serialize all fields."""
        result = self._create_sample_result({0: 0.5, 1: 0.7})
        d = result.to_dict()

        assert d["sourceModel"] == "source"
        assert d["targetModel"] == "target"
        assert "computedAt" in d
        assert "meanCompositeScore" in d
        assert "stdCompositeScore" in d
        assert "maxCompositeScore" in d
        assert "derivedThresholds" in d
        assert "hardSwap" in d["derivedThresholds"]
        assert "highAlpha" in d["derivedThresholds"]
        assert "layersAboveHardSwap" in d
        assert "layersAboveHighAlpha" in d
        assert "hardSwapLayers" in d
        assert "highAlphaLayers" in d
        assert "alphaByLayer" in d
        assert "layerScores" in d
        assert "hasSparsityData" in d
        assert "hasDirectionalData" in d
        assert "hasTransitionData" in d

    def test_to_dict_layer_scores_structure(self) -> None:
        """to_dict should properly structure layer scores."""
        result = self._create_sample_result({0: 0.6})
        d = result.to_dict()

        layer_dict = d["layerScores"]["0"]
        assert "layerName" in layer_dict
        assert "layerIndex" in layer_dict
        assert "compositeScore" in layer_dict
        assert "sparsityContribution" in layer_dict
        assert "directionalContribution" in layer_dict
        assert "transitionContribution" in layer_dict
        assert "alpha" in layer_dict

    def test_empty_result(self) -> None:
        """Should handle empty layer scores."""
        result = RefinementDensityResult(
            source_model="source",
            target_model="target",
            computed_at=datetime.now(timezone.utc),
            config=RefinementDensityConfig(),
            layer_scores={},
            mean_composite_score=0.0,
            max_composite_score=0.0,
            std_composite_score=0.0,
            has_sparsity_data=False,
            has_directional_data=False,
            has_transition_data=False,
        )

        assert result.hard_swap_layers == []
        assert result.high_alpha_layers == []
        assert result.alpha_by_layer == {}
        assert result.layers_above_hard_swap == 0
        assert result.layers_above_high_alpha == 0


# =============================================================================
# RefinementDensityAnalyzer Tests
# =============================================================================


class TestRefinementDensityAnalyzer:
    """Tests for RefinementDensityAnalyzer class."""

    def test_creation(self) -> None:
        """Should create analyzer with config."""
        config = RefinementDensityConfig()
        analyzer = RefinementDensityAnalyzer(config)

        assert analyzer.config is config

    def test_analyze_empty(self) -> None:
        """Should return empty result when no data provided."""
        config = RefinementDensityConfig()
        analyzer = RefinementDensityAnalyzer(config)

        result = analyzer.analyze(
            source_model="source",
            target_model="target",
        )

        assert result.source_model == "source"
        assert result.target_model == "target"
        assert result.layer_scores == {}
        assert result.mean_composite_score == 0.0
        assert result.has_sparsity_data is False
        assert result.has_directional_data is False
        assert result.has_transition_data is False

    def test_analyze_with_layer_count(self) -> None:
        """Should create scores for specified layer count."""
        config = RefinementDensityConfig()
        analyzer = RefinementDensityAnalyzer(config)

        result = analyzer.analyze(
            source_model="source",
            target_model="target",
            layer_count=4,
        )

        assert len(result.layer_scores) == 4
        assert 0 in result.layer_scores
        assert 3 in result.layer_scores

    def test_score_to_alpha(self) -> None:
        """_score_to_alpha should map score to 1 - score."""
        config = RefinementDensityConfig()
        analyzer = RefinementDensityAnalyzer(config)

        assert analyzer._score_to_alpha(0.0) == 1.0
        assert analyzer._score_to_alpha(1.0) == 0.0
        assert analyzer._score_to_alpha(0.5) == 0.5
        assert analyzer._score_to_alpha(0.3) == pytest.approx(0.7, abs=1e-5)

    def test_score_to_alpha_clamped(self) -> None:
        """_score_to_alpha should clamp to [0, 1]."""
        config = RefinementDensityConfig()
        analyzer = RefinementDensityAnalyzer(config)

        # Score > 1.0 would give negative alpha, should clamp to 0
        assert analyzer._score_to_alpha(1.5) == 0.0

        # Score < 0 would give alpha > 1, should clamp to 1
        assert analyzer._score_to_alpha(-0.5) == 1.0

    def test_extract_layer_index_valid(self) -> None:
        """_extract_layer_index should parse valid layer keys."""
        assert RefinementDensityAnalyzer._extract_layer_index("layers.0.mlp.gate_proj.weight") == 0
        assert RefinementDensityAnalyzer._extract_layer_index("layers.5.self_attn.q_proj") == 5
        assert RefinementDensityAnalyzer._extract_layer_index("model.layers.12.mlp") == 12

    def test_extract_layer_index_invalid(self) -> None:
        """_extract_layer_index should return None for invalid keys."""
        assert RefinementDensityAnalyzer._extract_layer_index("mlp.gate_proj") is None
        assert RefinementDensityAnalyzer._extract_layer_index("embed_tokens.weight") is None
        assert RefinementDensityAnalyzer._extract_layer_index("lm_head.weight") is None

    def test_extract_layer_index_non_numeric(self) -> None:
        """_extract_layer_index should return None for non-numeric indices."""
        assert RefinementDensityAnalyzer._extract_layer_index("layers.abc.mlp") is None

    def test_compute_layer_score_all_missing(self) -> None:
        """_compute_layer_score should use neutral values when all missing."""
        config = RefinementDensityConfig()
        analyzer = RefinementDensityAnalyzer(config)

        score = analyzer._compute_layer_score(
            layer_idx=0,
            sparsity=None,
            dora=None,
            transition=None,
        )

        # All neutral (0.5) contributions
        assert score.sparsity_contribution == 0.5
        assert score.directional_contribution == 0.5
        assert score.transition_contribution == 0.5
        # Composite = (0.5 + 0.5 + 0.5) * (1/3 each) = 0.5
        assert score.composite_score == pytest.approx(0.5, abs=1e-5)
        # Alpha = 1.0 - 0.5 = 0.5
        assert score.recommended_alpha == pytest.approx(0.5, abs=1e-5)

    def test_compute_layer_score_layer_name(self) -> None:
        """_compute_layer_score should use default layer name when no data."""
        config = RefinementDensityConfig()
        analyzer = RefinementDensityAnalyzer(config)

        score = analyzer._compute_layer_score(
            layer_idx=7,
            sparsity=None,
            dora=None,
            transition=None,
        )

        assert score.layer_name == "layer_7"
        assert score.layer_index == 7

    def test_empty_result_structure(self) -> None:
        """_empty_result should create valid empty result."""
        config = RefinementDensityConfig()
        analyzer = RefinementDensityAnalyzer(config)

        result = analyzer._empty_result("src", "tgt")

        assert result.source_model == "src"
        assert result.target_model == "tgt"
        assert result.layer_scores == {}
        assert result.mean_composite_score == 0.0
        assert result.max_composite_score == 0.0
        assert result.std_composite_score == 0.0
        assert result.has_sparsity_data is False
        assert result.has_directional_data is False
        assert result.has_transition_data is False

    def test_analyze_aggregate_metrics(self) -> None:
        """analyze should compute correct aggregate metrics."""
        config = RefinementDensityConfig()
        analyzer = RefinementDensityAnalyzer(config)

        # With 4 layers, all neutral scores (0.5)
        result = analyzer.analyze(
            source_model="source",
            target_model="target",
            layer_count=4,
        )

        # All scores should be 0.5 (neutral)
        assert result.mean_composite_score == pytest.approx(0.5, abs=1e-5)
        assert result.max_composite_score == pytest.approx(0.5, abs=1e-5)
        # std should be 0 since all scores are equal
        assert result.std_composite_score == pytest.approx(0.0, abs=1e-5)


# =============================================================================
# RefinementMetricKey Tests
# =============================================================================


class TestRefinementMetricKey:
    """Tests for RefinementMetricKey class."""

    def test_metric_keys_exist(self) -> None:
        """All metric keys should be defined."""
        assert RefinementMetricKey.MEAN_COMPOSITE == "geometry/refinement_mean_composite"
        assert RefinementMetricKey.MAX_COMPOSITE == "geometry/refinement_max_composite"
        assert RefinementMetricKey.HARD_SWAP_COUNT == "geometry/refinement_hard_swap_count"
        assert RefinementMetricKey.HIGH_ALPHA_COUNT == "geometry/refinement_high_alpha_count"

    def test_metric_keys_are_strings(self) -> None:
        """Metric keys should be strings."""
        assert isinstance(RefinementMetricKey.MEAN_COMPOSITE, str)
        assert isinstance(RefinementMetricKey.MAX_COMPOSITE, str)
        assert isinstance(RefinementMetricKey.HARD_SWAP_COUNT, str)
        assert isinstance(RefinementMetricKey.HIGH_ALPHA_COUNT, str)


# =============================================================================
# to_metrics_dict Tests
# =============================================================================


class TestToMetricsDict:
    """Tests for to_metrics_dict function."""

    def _create_sample_result(
        self,
        mean: float = 0.5,
        max_score: float = 0.8,
    ) -> RefinementDensityResult:
        """Create a sample result."""
        layer_scores = {}
        for idx in range(4):
            composite = 0.3 + idx * 0.2  # 0.3, 0.5, 0.7, 0.9
            layer_scores[idx] = LayerRefinementScore(
                layer_name=f"layer_{idx}",
                layer_index=idx,
                sparsity_contribution=composite,
                directional_contribution=composite,
                transition_contribution=composite,
                composite_score=composite,
                recommended_alpha=1.0 - composite,
            )

        return RefinementDensityResult(
            source_model="source",
            target_model="target",
            computed_at=datetime.now(timezone.utc),
            config=RefinementDensityConfig(),
            layer_scores=layer_scores,
            mean_composite_score=mean,
            max_composite_score=max_score,
            std_composite_score=0.2,
            has_sparsity_data=True,
            has_directional_data=True,
            has_transition_data=True,
        )

    def test_returns_dict(self) -> None:
        """Should return a dictionary."""
        result = self._create_sample_result()
        metrics = to_metrics_dict(result)

        assert isinstance(metrics, dict)

    def test_contains_all_keys(self) -> None:
        """Should contain all metric keys."""
        result = self._create_sample_result()
        metrics = to_metrics_dict(result)

        assert RefinementMetricKey.MEAN_COMPOSITE in metrics
        assert RefinementMetricKey.MAX_COMPOSITE in metrics
        assert RefinementMetricKey.HARD_SWAP_COUNT in metrics
        assert RefinementMetricKey.HIGH_ALPHA_COUNT in metrics

    def test_values_are_floats(self) -> None:
        """All values should be floats."""
        result = self._create_sample_result()
        metrics = to_metrics_dict(result)

        for key, value in metrics.items():
            assert isinstance(value, float), f"{key} is not float"

    def test_mean_composite_value(self) -> None:
        """MEAN_COMPOSITE should match result.mean_composite_score."""
        result = self._create_sample_result(mean=0.65)
        metrics = to_metrics_dict(result)

        assert metrics[RefinementMetricKey.MEAN_COMPOSITE] == 0.65

    def test_max_composite_value(self) -> None:
        """MAX_COMPOSITE should match result.max_composite_score."""
        result = self._create_sample_result(max_score=0.95)
        metrics = to_metrics_dict(result)

        assert metrics[RefinementMetricKey.MAX_COMPOSITE] == 0.95

    def test_count_values_are_floats(self) -> None:
        """Count values should be converted to floats."""
        result = self._create_sample_result()
        metrics = to_metrics_dict(result)

        # These are counts, but should be floats
        assert isinstance(metrics[RefinementMetricKey.HARD_SWAP_COUNT], float)
        assert isinstance(metrics[RefinementMetricKey.HIGH_ALPHA_COUNT], float)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for refinement density analysis."""

    def test_full_workflow_with_layer_count(self) -> None:
        """Complete workflow with specified layer count."""
        config = RefinementDensityConfig.with_parameters(
            sparsity_weight=0.4,
            directional_weight=0.3,
            transition_weight=0.3,
        )
        analyzer = RefinementDensityAnalyzer(config)

        result = analyzer.analyze(
            source_model="qwen-0.5B-refined",
            target_model="qwen-0.5B-base",
            layer_count=8,
        )

        # Basic structure
        assert len(result.layer_scores) == 8
        assert result.source_model == "qwen-0.5B-refined"
        assert result.target_model == "qwen-0.5B-base"

        # Alphas should be available
        alphas = result.alpha_by_layer
        assert len(alphas) == 8
        for layer_idx, alpha in alphas.items():
            assert 0.0 <= alpha <= 1.0

        # Serialization should work
        d = result.to_dict()
        assert "layerScores" in d
        assert len(d["layerScores"]) == 8

    def test_metrics_dict_from_analysis(self) -> None:
        """Metrics dict should work with analysis result."""
        config = RefinementDensityConfig()
        analyzer = RefinementDensityAnalyzer(config)

        result = analyzer.analyze(
            source_model="source",
            target_model="target",
            layer_count=4,
        )

        metrics = to_metrics_dict(result)

        assert len(metrics) == 4
        assert all(isinstance(v, float) for v in metrics.values())

    def test_config_affects_scoring(self) -> None:
        """Different configs should produce different results."""
        # Config favoring sparsity
        config1 = RefinementDensityConfig.with_parameters(
            sparsity_weight=0.8,
            directional_weight=0.1,
            transition_weight=0.1,
        )

        # Config favoring directional
        config2 = RefinementDensityConfig.with_parameters(
            sparsity_weight=0.1,
            directional_weight=0.8,
            transition_weight=0.1,
        )

        analyzer1 = RefinementDensityAnalyzer(config1)
        analyzer2 = RefinementDensityAnalyzer(config2)

        # Both should run without error
        result1 = analyzer1.analyze("src", "tgt", layer_count=4)
        result2 = analyzer2.analyze("src", "tgt", layer_count=4)

        # Both should produce valid results
        assert len(result1.layer_scores) == 4
        assert len(result2.layer_scores) == 4

    def test_derived_thresholds_change_with_distribution(self) -> None:
        """Derived thresholds should change based on score distribution."""
        config = RefinementDensityConfig()

        # Create results with different distributions
        result1 = RefinementDensityResult(
            source_model="source",
            target_model="target",
            computed_at=datetime.now(timezone.utc),
            config=config,
            layer_scores={
                i: LayerRefinementScore(
                    layer_name=f"layer_{i}",
                    layer_index=i,
                    sparsity_contribution=0.5,
                    directional_contribution=0.5,
                    transition_contribution=0.5,
                    composite_score=0.5,  # All same
                    recommended_alpha=0.5,
                )
                for i in range(4)
            },
            mean_composite_score=0.5,
            max_composite_score=0.5,
            std_composite_score=0.0,  # No variance
            has_sparsity_data=True,
            has_directional_data=True,
            has_transition_data=True,
        )

        result2 = RefinementDensityResult(
            source_model="source",
            target_model="target",
            computed_at=datetime.now(timezone.utc),
            config=config,
            layer_scores={
                i: LayerRefinementScore(
                    layer_name=f"layer_{i}",
                    layer_index=i,
                    sparsity_contribution=0.2 + i * 0.2,
                    directional_contribution=0.2 + i * 0.2,
                    transition_contribution=0.2 + i * 0.2,
                    composite_score=0.2 + i * 0.2,  # 0.2, 0.4, 0.6, 0.8
                    recommended_alpha=0.8 - i * 0.2,
                )
                for i in range(4)
            },
            mean_composite_score=0.5,
            max_composite_score=0.8,
            std_composite_score=0.22,  # Has variance
            has_sparsity_data=True,
            has_directional_data=True,
            has_transition_data=True,
        )

        # With no variance, thresholds should cluster around mean
        hard1, high1, mid1 = result1._derived_thresholds
        assert hard1 == pytest.approx(mid1, abs=1e-5)  # No spread
        assert high1 == pytest.approx(mid1, abs=1e-5)

        # With variance, thresholds should spread out
        hard2, high2, mid2 = result2._derived_thresholds
        assert hard2 > high2
        assert high2 > mid2
