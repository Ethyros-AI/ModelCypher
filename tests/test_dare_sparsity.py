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

from datetime import datetime

import pytest

from modelcypher.core.domain.geometry.dare_sparsity import (
    Configuration,
    DARESparsityAnalyzer,
    QualityAssessment,
)


def test_empty_analysis() -> None:
    analysis = DARESparsityAnalyzer.analyze({})

    assert analysis.total_parameters == 0
    assert analysis.non_zero_parameters == 0
    assert analysis.effective_sparsity == 1.0
    assert analysis.essential_fraction == 0.0
    assert analysis.recommended_drop_rate == 0.0
    assert analysis.quality_assessment == QualityAssessment.excellent_for_merging
    assert analysis.per_layer_sparsity == {}
    assert isinstance(analysis.computed_at, datetime)


def test_identify_essential_parameters() -> None:
    deltas = {"layer1": [0.1, -0.05, 0.0], "layer2": [-0.2]}
    essential = DARESparsityAnalyzer.identify_essential_parameters(deltas, threshold=0.1)

    assert essential["layer1"] == {0}
    assert essential["layer2"] == {0}


def test_analysis_with_custom_threshold() -> None:
    deltas = {
        "layer1": [0.0, 0.2, 0.5, 1.0],
        "layer2": [0.05, 0.0],
    }
    config = Configuration(sparsity_threshold=0.2, droppable_percentile=0.5)

    analysis = DARESparsityAnalyzer.analyze(deltas, configuration=config)

    assert analysis.total_parameters == 6
    assert analysis.non_zero_parameters == 4
    assert analysis.effective_sparsity == pytest.approx(4.0 / 6.0)
    assert analysis.essential_fraction == pytest.approx(2.0 / 6.0)
    assert analysis.recommended_drop_rate == pytest.approx(0.6)
    assert analysis.quality_assessment == QualityAssessment.moderate

    layer1 = analysis.per_layer_sparsity["layer1"]
    assert layer1.parameter_count == 4
    assert layer1.sparsity == pytest.approx(0.5)
    assert layer1.mean_magnitude == pytest.approx(0.425)
    assert layer1.max_magnitude == pytest.approx(1.0)
    assert layer1.essential_fraction == pytest.approx(0.5)
    assert layer1.has_significant_updates is True

    layer2 = analysis.per_layer_sparsity["layer2"]
    assert layer2.parameter_count == 2
    assert layer2.sparsity == pytest.approx(1.0)
    assert layer2.mean_magnitude == pytest.approx(0.025)
    assert layer2.max_magnitude == pytest.approx(0.05)
    assert layer2.essential_fraction == pytest.approx(0.0)
    assert layer2.has_significant_updates is False

    stats = analysis.magnitude_stats
    assert stats.max == pytest.approx(1.0)
    assert stats.min_non_zero == pytest.approx(0.05)
    assert stats.median == pytest.approx(0.2)
    assert stats.percentile1 == pytest.approx(0.0)
    assert stats.percentile5 == pytest.approx(0.0)
    assert stats.percentile95 == pytest.approx(1.0)
    assert stats.percentile99 == pytest.approx(1.0)


def test_analysis_layer_filtering() -> None:
    deltas = {
        "layer1": [0.0, 0.2, 0.5, 1.0],
        "layer2": [0.05, 0.0],
    }
    config = Configuration(
        sparsity_threshold=0.2,
        droppable_percentile=0.5,
        analysis_layers={"layer1"},
    )

    analysis = DARESparsityAnalyzer.analyze(deltas, configuration=config)

    assert analysis.total_parameters == 4
    assert analysis.effective_sparsity == pytest.approx(0.75)
    assert analysis.essential_fraction == pytest.approx(0.25)
    assert analysis.recommended_drop_rate == pytest.approx(0.675)
    assert analysis.quality_assessment == QualityAssessment.moderate
    assert set(analysis.per_layer_sparsity.keys()) == {"layer1"}


def test_metrics_dictionary() -> None:
    deltas = {"layer1": [0.0, 1.0], "layer2": [0.1, 0.0]}
    config = Configuration(sparsity_threshold=0.5, droppable_percentile=0.5)
    analysis = DARESparsityAnalyzer.analyze(deltas, configuration=config)
    metrics = DARESparsityAnalyzer.to_metrics_dictionary(analysis)

    assert metrics["geometry/dare_effective_sparsity"] == pytest.approx(analysis.effective_sparsity)
    assert metrics["geometry/dare_essential_fraction"] == pytest.approx(analysis.essential_fraction)
    assert metrics["geometry/dare_recommended_drop_rate"] == pytest.approx(analysis.recommended_drop_rate)
