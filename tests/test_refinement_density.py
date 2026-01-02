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

"""Tests for refinement_density.py."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from modelcypher.core.domain.geometry.concept_response_matrix import (
    LayerTransitionResult,
    TransitionExperiment,
)
from modelcypher.core.domain.geometry.dora_decomposition import (
    ChangeType,
    DecompositionResult,
    MagnitudeDirectionMetrics,
)
from modelcypher.core.domain.geometry.refinement_density import (
    LayerRefinementScore,
    RefinementDensityAnalyzer,
    RefinementDensityResult,
    RefinementMetricKey,
    to_metrics_dict,
)


class TestLayerRefinementScore:
    def test_basic_creation(self) -> None:
        score = LayerRefinementScore(
            layer_name="layers.0.mlp",
            layer_index=0,
            sparsity_contribution=None,
            directional_contribution=0.5,
            transition_contribution=None,
            composite_score=0.5,
            component_count=1,
        )

        assert score.layer_name == "layers.0.mlp"
        assert score.component_count == 1
        assert score.directional_contribution == 0.5

    def test_frozen(self) -> None:
        score = LayerRefinementScore(
            layer_name="layer_0",
            layer_index=0,
            sparsity_contribution=0.2,
            directional_contribution=None,
            transition_contribution=None,
            composite_score=0.2,
            component_count=1,
        )

        with pytest.raises(Exception):
            score.composite_score = 0.9  # type: ignore


class TestRefinementDensityAnalyzer:
    def test_analyze_no_data(self) -> None:
        analyzer = RefinementDensityAnalyzer()
        result = analyzer.analyze("source", "target", layer_count=2)

        assert result.scored_layer_count == 0
        assert result.mean_composite_score == 0.0
        assert len(result.layer_scores) == 2
        for score in result.layer_scores.values():
            assert score.component_count == 0
            assert score.composite_score == 0.0

    def test_directional_normalization(self) -> None:
        analyzer = RefinementDensityAnalyzer()

        metrics0 = MagnitudeDirectionMetrics(
            layer_name="layers.0.mlp",
            base_magnitude=1.0,
            current_magnitude=1.0,
            magnitude_ratio=1.0,
            direction_cosine=1.0,
            directional_drift=2.0,
            absolute_magnitude_change=0.0,
            relative_magnitude_change=0.0,
        )
        metrics1 = MagnitudeDirectionMetrics(
            layer_name="layers.1.mlp",
            base_magnitude=1.0,
            current_magnitude=1.0,
            magnitude_ratio=1.0,
            direction_cosine=1.0,
            directional_drift=1.0,
            absolute_magnitude_change=0.0,
            relative_magnitude_change=0.0,
        )

        dora_result = DecompositionResult(
            per_layer_metrics={"layers.0.mlp": metrics0, "layers.1.mlp": metrics1},
            overall_magnitude_change=0.0,
            overall_directional_drift=0.0,
            dominant_change_type=ChangeType.BALANCED,
            magnitude_to_direction_ratio=1.0,
            layers_with_significant_direction_change=[],
            layers_with_significant_magnitude_change=[],
        )

        result = analyzer.analyze(
            source_model="source",
            target_model="target",
            dora_result=dora_result,
            layer_count=2,
        )

        assert result.max_directional_drift == pytest.approx(2.0, abs=1e-6)
        assert result.layer_scores[0].directional_contribution == pytest.approx(1.0, abs=1e-6)
        assert result.layer_scores[1].directional_contribution == pytest.approx(0.5, abs=1e-6)
        assert result.layer_scores[0].component_count == 1
        assert result.layer_scores[0].composite_score == pytest.approx(1.0, abs=1e-6)

    def test_transition_normalization(self) -> None:
        analyzer = RefinementDensityAnalyzer()

        transitions = [
            LayerTransitionResult(0, 1, 0.8, 0.4, 1.0, 1.0),
            LayerTransitionResult(1, 2, 0.4, 0.4, 1.0, 1.0),
        ]
        experiment = TransitionExperiment(
            source_model="source",
            target_model="target",
            timestamp=datetime.now(timezone.utc),
            transitions=transitions,
            mean_transition_cka=0.6,
            mean_state_cka=0.4,
            transition_better_than_state=True,
            transition_advantage=1.5,
            anchor_count=4,
            layer_transition_count=len(transitions),
        )

        result = analyzer.analyze(
            source_model="source",
            target_model="target",
            transition_experiment=experiment,
            layer_count=2,
        )

        assert result.max_transition_advantage == pytest.approx(2.0, abs=1e-6)
        assert result.layer_scores[0].transition_contribution == pytest.approx(1.0, abs=1e-6)
        assert result.layer_scores[1].transition_contribution == pytest.approx(0.5, abs=1e-6)

    def test_extract_layer_index(self) -> None:
        assert RefinementDensityAnalyzer._extract_layer_index("layers.0.mlp.gate_proj") == 0
        assert RefinementDensityAnalyzer._extract_layer_index("layers.12.self_attn") == 12
        assert RefinementDensityAnalyzer._extract_layer_index("mlp.gate_proj") is None


class TestRefinementDensityResult:
    def test_to_dict_contains_normalization(self) -> None:
        result = RefinementDensityResult(
            source_model="source",
            target_model="target",
            computed_at=datetime.now(timezone.utc),
            layer_scores={},
            mean_composite_score=0.0,
            max_composite_score=0.0,
            std_composite_score=0.0,
            scored_layer_count=0,
            max_directional_drift=1.2,
            max_transition_advantage=1.8,
            has_sparsity_data=False,
            has_directional_data=False,
            has_transition_data=False,
        )

        payload = result.to_dict()
        assert "normalization" in payload
        assert payload["normalization"]["maxDirectionalDrift"] == 1.2
        assert payload["normalization"]["maxTransitionAdvantage"] == 1.8


class TestMetricsDict:
    def test_to_metrics_dict(self) -> None:
        result = RefinementDensityResult(
            source_model="source",
            target_model="target",
            computed_at=datetime.now(timezone.utc),
            layer_scores={},
            mean_composite_score=0.4,
            max_composite_score=0.7,
            std_composite_score=0.0,
            scored_layer_count=3,
            max_directional_drift=0.0,
            max_transition_advantage=0.0,
            has_sparsity_data=False,
            has_directional_data=False,
            has_transition_data=False,
        )

        metrics = to_metrics_dict(result)
        assert metrics[RefinementMetricKey.MEAN_COMPOSITE] == 0.4
        assert metrics[RefinementMetricKey.MAX_COMPOSITE] == 0.7
        assert metrics[RefinementMetricKey.SCORED_LAYER_COUNT] == 3.0
