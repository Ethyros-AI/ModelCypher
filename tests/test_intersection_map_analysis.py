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

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.core.domain.geometry.intersection_map_analysis import (
    IntersectionMapAnalysis,
    MarkdownReportOptions,
)
from modelcypher.core.domain.geometry.manifold_stitcher import (
    DimensionCorrelation,
    IntersectionMap,
    LayerConfidence,
)


def _eps(*values: float) -> float:
    backend = get_default_backend()
    return machine_epsilon(backend, backend.array(list(values)))


def test_intersection_map_analysis_counts() -> None:
    correlations = {
        0: [
            DimensionCorrelation(source_dim=0, target_dim=0, correlation=0.8),
            DimensionCorrelation(source_dim=1, target_dim=1, correlation=0.3),
        ]
    }
    layer_confidences = [
        LayerConfidence(layer=0, confidence=0.65, correlation_count=2)
    ]
    map_data = IntersectionMap(
        source_model="source",
        target_model="target",
        dimension_correlations=correlations,
        raw_fingerprint_similarity=0.55,
        aligned_dimension_count=2,
        total_source_dims=4,
        total_target_dims=4,
        layer_confidences=layer_confidences,
    )

    analysis = IntersectionMapAnalysis.analyze(map_data)
    assert analysis.overall_stats.pair_count == 2
    assert abs(analysis.overall_stats.mean_correlation - 0.55) < _eps(
        analysis.overall_stats.mean_correlation, 0.55
    )
    eps = _eps(
        analysis.overall_stats.min_correlation,
        analysis.overall_stats.max_correlation,
        analysis.average_layer_confidence,
    )
    assert abs(analysis.overall_stats.min_correlation - 0.3) <= eps
    assert abs(analysis.overall_stats.max_correlation - 0.8) <= eps
    assert abs(analysis.average_layer_confidence - 0.65) <= eps


def test_intersection_map_report() -> None:
    correlations = {0: [DimensionCorrelation(source_dim=0, target_dim=0, correlation=0.9)]}
    map_data = IntersectionMap(
        source_model="source",
        target_model="target",
        dimension_correlations=correlations,
        raw_fingerprint_similarity=0.9,
        aligned_dimension_count=1,
        total_source_dims=2,
        total_target_dims=2,
        layer_confidences=[
            LayerConfidence(layer=0, confidence=0.9, correlation_count=1)
        ],
    )
    report = IntersectionMapAnalysis.render_markdown_report(
        map_data,
        options=MarkdownReportOptions(input_label="test"),
    )
    assert "Intersection Map Report" in report
    assert "Source" in report and "Target" in report
