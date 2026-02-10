from __future__ import annotations

import pytest

from modelcypher.core.domain.geometry.safety_polytope import (
    DiagnosticVector,
    PolytopeBounds,
    SafetyPolytope,
    TransformationType,
    create_diagnostic_vector,
    format_transformation_report,
)


def _bounds() -> PolytopeBounds:
    return PolytopeBounds(
        interference_threshold=0.3,
        importance_threshold=0.3,
        instability_threshold=0.3,
        complexity_threshold=0.3,
        magnitude_threshold=0.5,
        high_instability_threshold=1.0,
        high_interference_threshold=1.0,
    )


def test_diagnostic_vector_properties(any_backend) -> None:
    diag = DiagnosticVector(
        interference_score=0.2,
        importance_score=0.4,
        instability_score=0.1,
        complexity_score=0.7,
    )

    assert diag.vector == [0.2, 0.4, 0.1, 0.7]
    assert diag.max_dimension == "complexity"
    assert diag.magnitude > 0.0


def test_polytope_bounds_from_baseline_metrics_and_validation() -> None:
    bounds = PolytopeBounds.from_baseline_metrics(
        interference_samples=[0.1, 0.2, 0.4, 0.9],
        instability_samples=[0.1, 0.3, 0.5, 1.2],
        complexity_samples=[0.05, 0.2, 0.25, 0.8],
        magnitude_samples=[0.2, 0.4, 0.6, 1.1],
        importance_samples=[0.1, 0.15, 0.2, 0.7],
    )

    assert bounds.high_interference_threshold == 0.9
    assert bounds.high_instability_threshold == 1.2
    assert bounds.interference_threshold >= 0.0
    assert bounds.importance_threshold >= 0.0
    assert bounds.instability_threshold >= 0.0
    assert bounds.complexity_threshold >= 0.0
    assert bounds.magnitude_threshold >= 0.0

    with pytest.raises(ValueError, match="required"):
        PolytopeBounds.from_baseline_metrics(
            interference_samples=[],
            instability_samples=[0.1],
            complexity_samples=[0.1],
            magnitude_samples=[0.1],
        )


def test_analyze_layer_triggers_and_properties() -> None:
    polytope = SafetyPolytope(_bounds())
    diagnostics = DiagnosticVector(
        interference_score=1.0,
        importance_score=0.8,
        instability_score=0.9,
        complexity_score=0.95,
    )

    result = polytope.analyze_layer(diagnostics, layer=7)

    assert result.layer == 7
    assert result.transformation_effort > 0.0
    assert result.needs_spectral_clamping is True

    trigger_dims = {trigger.dimension for trigger in result.triggers}
    assert {"interference", "importance", "instability", "complexity", "magnitude"} <= trigger_dims

    assert TransformationType.GEODESIC_NULL_SPACE in result.transformations
    assert TransformationType.SPECTRAL_CLAMP in result.transformations
    assert TransformationType.TSV_PRUNE in result.transformations
    assert TransformationType.LAYER_SKIP in result.transformations

    for trigger in result.triggers:
        if trigger.dimension != "magnitude":
            assert 0.0 <= trigger.intensity <= 1.0


def test_compute_confidence_negative_and_positive_branches() -> None:
    polytope = SafetyPolytope(_bounds())

    high = DiagnosticVector(1.0, 1.0, 1.0, 1.0)
    low = DiagnosticVector(0.05, 0.05, 0.05, 0.05)

    high_conf = polytope._compute_confidence(high)
    low_conf = polytope._compute_confidence(low)

    assert 0.3 <= high_conf <= 1.0
    assert 0.5 <= low_conf <= 1.0
    assert low_conf > high_conf


def test_analyze_model_pair_classifies_layers() -> None:
    polytope = SafetyPolytope(_bounds())

    profile = polytope.analyze_model_pair(
        {
            0: DiagnosticVector(0.1, 0.1, 0.1, 0.1),
            1: DiagnosticVector(0.33, 0.05, 0.33, 0.05),
            2: DiagnosticVector(0.8, 0.8, 0.8, 0.8),
        }
    )

    assert profile.direct_merge_layers == [0]
    assert profile.light_transform_layers == [1]
    assert profile.heavy_transform_layers == [2]
    assert profile.total_transformation_effort > 0.0
    assert profile.mean_interference > 0.0
    assert profile.mean_importance > 0.0
    assert profile.mean_instability > 0.0
    assert profile.mean_complexity > 0.0


def test_create_diagnostic_vector_and_format_report() -> None:
    diag = create_diagnostic_vector(
        interference=1.3,
        refinement_density=-0.3,
        condition_number=1.0,
        intrinsic_dimension=3,
        hidden_dim=0,
    )

    assert diag.interference_score == 1.0
    assert diag.importance_score == 0.0
    assert diag.instability_score == 0.0
    assert diag.complexity_score == 0.0

    polytope = SafetyPolytope(_bounds())
    profile = polytope.analyze_model_pair({0: diag, 1: DiagnosticVector(0.9, 0.9, 0.9, 0.9)})

    report = format_transformation_report(profile)
    assert "MERGE TRANSFORMATION ANALYSIS" in report
    assert "Layer Classification" in report
    assert "Diagnostic Means" in report
    assert "Transformations Needed" in report
