from __future__ import annotations

from modelcypher.core.domain.geometry.intersection_map_analysis import (
    IntersectionMapAnalysis,
    MarkdownReportOptions,
)
from modelcypher.core.domain.geometry.manifold_stitcher import (
    DimensionCorrelation,
    IntersectionMap,
    LayerConfidence,
)


def test_intersection_map_analysis_counts() -> None:
    correlations = {
        0: [
            DimensionCorrelation(source_dim=0, target_dim=0, correlation=0.8),
            DimensionCorrelation(source_dim=1, target_dim=1, correlation=0.3),
        ]
    }
    layer_confidences = [LayerConfidence(layer=0, strong_correlations=1, moderate_correlations=0, weak_correlations=1)]
    map_data = IntersectionMap(
        source_model="source",
        target_model="target",
        dimension_correlations=correlations,
        overall_correlation=0.55,
        aligned_dimension_count=2,
        total_source_dims=4,
        total_target_dims=4,
        layer_confidences=layer_confidences,
    )

    analysis = IntersectionMapAnalysis.analyze(map_data)
    assert analysis.overall_stats.pair_count == 2
    assert analysis.overall_stats.strong_count == 1
    assert analysis.overall_stats.weak_count == 1


def test_intersection_map_report() -> None:
    correlations = {0: [DimensionCorrelation(source_dim=0, target_dim=0, correlation=0.9)]}
    map_data = IntersectionMap(
        source_model="source",
        target_model="target",
        dimension_correlations=correlations,
        overall_correlation=0.9,
        aligned_dimension_count=1,
        total_source_dims=2,
        total_target_dims=2,
        layer_confidences=[LayerConfidence(layer=0, strong_correlations=1, moderate_correlations=0, weak_correlations=0)],
    )
    report = IntersectionMapAnalysis.render_markdown_report(
        map_data,
        options=MarkdownReportOptions(input_label="test"),
    )
    assert "Intersection Map Report" in report
    assert "Source" in report and "Target" in report
