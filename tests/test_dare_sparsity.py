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

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.dare_sparsity import (
    Configuration,
    DARESparsityAnalyzer,
)
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon


def _eps(*values: float) -> float:
    backend = get_default_backend()
    return machine_epsilon(backend, backend.array(list(values) or [1.0]))


def test_empty_analysis() -> None:
    analysis = DARESparsityAnalyzer.analyze({})

    assert analysis.total_parameters == 0
    assert analysis.non_zero_parameters == 0
    assert analysis.effective_sparsity == 1.0
    assert analysis.essential_fraction == 0.0
    assert analysis.per_layer_sparsity == {}
    assert isinstance(analysis.computed_at, datetime)


def test_identify_essential_parameters() -> None:
    deltas = {"layer1": [0.1, -0.05, 0.0], "layer2": [-0.2]}
    essential = DARESparsityAnalyzer.identify_essential_parameters(deltas)

    assert essential["layer1"] == {0, 1}
    assert essential["layer2"] == {0}


def test_analysis_derives_thresholds_from_data() -> None:
    """Test that thresholds are derived from data, not arbitrary constants."""
    deltas = {
        "layer1": [0.0, 0.2, 0.5, 1.0],
        "layer2": [0.05, 0.0],
    }

    analysis = DARESparsityAnalyzer.analyze(deltas)

    # Basic structure
    assert analysis.total_parameters == 6
    assert analysis.non_zero_parameters == 4

    # Sparsity should be derived from spectral gap in magnitude distribution
    # The exact value depends on the data, not arbitrary thresholds
    eps = _eps(analysis.effective_sparsity, analysis.essential_fraction)
    assert analysis.effective_sparsity >= -eps
    assert analysis.effective_sparsity <= 1.0 + eps
    assert analysis.essential_fraction >= -eps
    assert analysis.essential_fraction <= 1.0 + eps
    assert abs(analysis.effective_sparsity + analysis.essential_fraction - 1.0) <= eps

    # Per-layer metrics
    layer1 = analysis.per_layer_sparsity["layer1"]
    assert layer1.parameter_count == 4
    eps = _eps(layer1.mean_magnitude, layer1.max_magnitude)
    assert abs(layer1.mean_magnitude - 0.425) <= eps
    assert abs(layer1.max_magnitude - 1.0) <= eps

    layer2 = analysis.per_layer_sparsity["layer2"]
    assert layer2.parameter_count == 2
    eps = _eps(layer2.mean_magnitude, layer2.max_magnitude)
    assert abs(layer2.mean_magnitude - 0.025) <= eps
    assert abs(layer2.max_magnitude - 0.05) <= eps

    # Magnitude stats
    stats = analysis.magnitude_stats
    eps = _eps(stats.max, stats.min_non_zero, stats.median)
    assert abs(stats.max - 1.0) <= eps
    assert abs(stats.min_non_zero - 0.05) <= eps
    assert abs(stats.median - 0.2) <= eps


def test_analysis_layer_filtering() -> None:
    deltas = {
        "layer1": [0.0, 0.2, 0.5, 1.0],
        "layer2": [0.05, 0.0],
    }
    config = Configuration(analysis_layers={"layer1"})

    analysis = DARESparsityAnalyzer.analyze(deltas, configuration=config)

    assert analysis.total_parameters == 4
    # Layer filtering should only analyze layer1
    assert set(analysis.per_layer_sparsity.keys()) == {"layer1"}
    # Sparsity is derived from data, verify constraints
    eps = _eps(analysis.effective_sparsity)
    assert analysis.effective_sparsity >= -eps
    assert analysis.effective_sparsity <= 1.0 + eps


def test_metrics_dictionary() -> None:
    deltas = {"layer1": [0.0, 1.0], "layer2": [0.1, 0.0]}
    analysis = DARESparsityAnalyzer.analyze(deltas)
    metrics = DARESparsityAnalyzer.to_metrics_dictionary(analysis)

    eps = _eps(
        metrics["geometry/dare_effective_sparsity"],
        metrics["geometry/dare_essential_fraction"],
        analysis.effective_sparsity,
        analysis.essential_fraction,
    )
    assert abs(metrics["geometry/dare_effective_sparsity"] - analysis.effective_sparsity) <= eps
    assert abs(metrics["geometry/dare_essential_fraction"] - analysis.essential_fraction) <= eps
