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
Tests for Safety Polytope.

Validates the unified safety boundary that combines:
- Interference (RiemannianDensity)
- Importance (RefinementDensity)
- Stability (SpectralAnalysis)
- Complexity (IntrinsicDimension)
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.safety_polytope import (
    DiagnosticVector,
    PolytopeBounds,
    SafetyPolytope,
    TransformationType,
    create_diagnostic_vector,
    format_transformation_report,
)


def _test_bounds(
    interference_threshold: float = 0.6,
    importance_threshold: float = 0.7,
    instability_threshold: float = 0.5,
    complexity_threshold: float = 0.8,
    magnitude_threshold: float = 1.2,
    high_instability_threshold: float = 0.9,
    high_interference_threshold: float = 0.9,
) -> PolytopeBounds:
    """Create PolytopeBounds for testing with explicit thresholds."""
    return PolytopeBounds(
        interference_threshold=interference_threshold,
        importance_threshold=importance_threshold,
        instability_threshold=instability_threshold,
        complexity_threshold=complexity_threshold,
        magnitude_threshold=magnitude_threshold,
        high_instability_threshold=high_instability_threshold,
        high_interference_threshold=high_interference_threshold,
    )


class TestDiagnosticVector:
    """Test DiagnosticVector properties."""

    def test_vector_property(self):
        """Vector property should return list."""
        diag = DiagnosticVector(
            interference_score=0.3,
            importance_score=0.5,
            instability_score=0.2,
            complexity_score=0.4,
        )

        vec = diag.vector
        assert isinstance(vec, list)
        assert len(vec) == 4
        expected = [0.3, 0.5, 0.2, 0.4]
        assert vec == expected

    def test_magnitude_property(self):
        """Magnitude should be L2 norm."""
        backend = get_default_backend()
        diag = DiagnosticVector(
            interference_score=0.3,
            importance_score=0.4,
            instability_score=0.0,
            complexity_score=0.0,
        )

        expected = backend.sqrt(0.3**2 + 0.4**2)
        backend.eval(expected)
        assert abs(diag.magnitude - float(backend.to_numpy(expected))) < 1e-6

    def test_max_dimension(self):
        """Should identify highest dimension."""
        diag = DiagnosticVector(
            interference_score=0.3,
            importance_score=0.5,
            instability_score=0.8,  # Highest
            complexity_score=0.4,
        )

        assert diag.max_dimension == "instability"

    def test_zero_vector(self):
        """Zero vector should have zero magnitude."""
        diag = DiagnosticVector(0, 0, 0, 0)
        assert diag.magnitude == 0


class TestSafetyPolytopeBasic:
    """Test basic polytope operations."""

    def test_safe_point_inside_polytope(self):
        """Point inside boundaries needs no transformations."""
        polytope = SafetyPolytope(_test_bounds())

        diag = DiagnosticVector(
            interference_score=0.2,
            importance_score=0.3,
            instability_score=0.1,
            complexity_score=0.2,
        )

        result = polytope.analyze_layer(diag)
        assert len(result.triggers) == 0
        assert len(result.transformations) == 0
        assert result.confidence > 0.5

    def test_point_on_boundary(self):
        """Point exactly on boundary has reduced confidence."""
        bounds = _test_bounds(interference_threshold=0.5)
        polytope = SafetyPolytope(bounds)

        diag = DiagnosticVector(
            interference_score=0.5,  # Exactly at boundary
            importance_score=0.3,
            instability_score=0.1,
            complexity_score=0.2,
        )

        result = polytope.analyze_layer(diag)

        # At boundary, confidence reduced but still returns a valid result
        assert result.confidence < 0.8

    def test_point_outside_triggers_transformation(self):
        """Point beyond threshold triggers transformation."""
        bounds = _test_bounds(interference_threshold=0.5)
        polytope = SafetyPolytope(bounds)

        diag = DiagnosticVector(
            interference_score=0.65,  # Beyond threshold
            importance_score=0.3,
            instability_score=0.1,
            complexity_score=0.2,
        )

        result = polytope.analyze_layer(diag)
        assert len(result.triggers) >= 1
        assert TransformationType.NULL_SPACE_FILTER in result.transformations

    def test_high_instability_triggers_transformation(self):
        """High instability triggers spectral clamping."""
        polytope = SafetyPolytope(_test_bounds())

        diag = DiagnosticVector(
            interference_score=0.3,
            importance_score=0.3,
            instability_score=0.95,  # High instability
            complexity_score=0.2,
        )

        result = polytope.analyze_layer(diag)
        assert TransformationType.SPECTRAL_CLAMP in result.transformations

    def test_recommended_alpha_adjustment(self):
        """Alpha should be reduced for violations."""
        polytope = SafetyPolytope(_test_bounds())

        diag = DiagnosticVector(
            interference_score=0.8,  # High interference
            importance_score=0.3,
            instability_score=0.1,
            complexity_score=0.2,
        )

        result = polytope.analyze_layer(diag, base_alpha=0.5)

        assert result.recommended_alpha is not None
        assert result.recommended_alpha < 0.5  # Reduced due to interference


class TestTransformations:
    """Test transformation recommendations."""

    def test_interference_triggers_null_space(self):
        """High interference triggers null-space filtering."""
        bounds = _test_bounds(interference_threshold=0.5)
        polytope = SafetyPolytope(bounds)

        diag = DiagnosticVector(0.7, 0.3, 0.2, 0.2)
        result = polytope.analyze_layer(diag)

        assert TransformationType.NULL_SPACE_FILTER in result.transformations

    def test_importance_triggers_alpha_reduction(self):
        """High importance triggers alpha reduction."""
        bounds = _test_bounds(importance_threshold=0.5)
        polytope = SafetyPolytope(bounds)

        diag = DiagnosticVector(0.2, 0.8, 0.2, 0.2)
        result = polytope.analyze_layer(diag)

        assert TransformationType.REDUCE_ALPHA in result.transformations

    def test_instability_triggers_spectral_clamp(self):
        """High instability triggers spectral clamping."""
        bounds = _test_bounds(instability_threshold=0.4)
        polytope = SafetyPolytope(bounds)

        diag = DiagnosticVector(0.2, 0.3, 0.6, 0.2)
        result = polytope.analyze_layer(diag)

        assert TransformationType.SPECTRAL_CLAMP in result.transformations

    def test_complexity_triggers_tsv_prune(self):
        """High complexity triggers TSV pruning."""
        bounds = _test_bounds(complexity_threshold=0.5)
        polytope = SafetyPolytope(bounds)

        diag = DiagnosticVector(0.2, 0.3, 0.2, 0.7)
        result = polytope.analyze_layer(diag)

        assert TransformationType.TSV_PRUNE in result.transformations

    def test_extreme_magnitude_triggers_layer_skip(self):
        """Extreme overall magnitude triggers layer skip."""
        bounds = _test_bounds(magnitude_threshold=1.0)
        polytope = SafetyPolytope(bounds)

        # All dimensions high
        diag = DiagnosticVector(0.8, 0.8, 0.8, 0.8)
        result = polytope.analyze_layer(diag)

        assert TransformationType.LAYER_SKIP in result.transformations


class TestModelProfile:
    """Test model-level analysis."""

    def test_all_direct_merge_layers(self):
        """Layers inside bounds need no transformations."""
        polytope = SafetyPolytope(_test_bounds())

        layer_diagnostics = {i: DiagnosticVector(0.2, 0.3, 0.1, 0.2) for i in range(10)}

        profile = polytope.analyze_model_pair(layer_diagnostics)

        assert len(profile.direct_merge_layers) == 10
        assert len(profile.light_transform_layers) == 0
        assert len(profile.all_transformations) == 0

    def test_high_instability_triggers_transformations(self):
        """High instability layer needs spectral clamping."""
        polytope = SafetyPolytope(_test_bounds())

        layer_diagnostics = {}
        for i in range(10):
            if i == 5:
                # High instability layer
                layer_diagnostics[i] = DiagnosticVector(0.2, 0.3, 0.95, 0.2)
            else:
                layer_diagnostics[i] = DiagnosticVector(0.2, 0.3, 0.1, 0.2)

        profile = polytope.analyze_model_pair(layer_diagnostics)

        # Layer 5 needs transformation (spectral clamp for instability)
        assert 5 in profile.light_transform_layers
        assert TransformationType.SPECTRAL_CLAMP in profile.all_transformations

    def test_mean_diagnostics(self):
        """Mean diagnostics should be computed correctly."""
        polytope = SafetyPolytope(_test_bounds())

        layer_diagnostics = {
            0: DiagnosticVector(0.2, 0.4, 0.1, 0.3),
            1: DiagnosticVector(0.4, 0.6, 0.3, 0.5),
        }

        profile = polytope.analyze_model_pair(layer_diagnostics)

        # Mean of [0.2, 0.4] = 0.3, etc.
        assert abs(profile.mean_interference - 0.3) < 1e-6
        assert abs(profile.mean_importance - 0.5) < 1e-6
        assert abs(profile.mean_instability - 0.2) < 1e-6
        assert abs(profile.mean_complexity - 0.4) < 1e-6

    def test_transformations_aggregated(self):
        """Transformations aggregate from all layers."""
        bounds = _test_bounds(interference_threshold=0.5, instability_threshold=0.4)
        polytope = SafetyPolytope(bounds)

        layer_diagnostics = {
            0: DiagnosticVector(0.7, 0.3, 0.2, 0.2),  # High interference
            1: DiagnosticVector(0.2, 0.3, 0.6, 0.2),  # High instability
        }

        profile = polytope.analyze_model_pair(layer_diagnostics)

        assert TransformationType.NULL_SPACE_FILTER in profile.all_transformations
        assert TransformationType.SPECTRAL_CLAMP in profile.all_transformations


class TestCreateDiagnosticVector:
    """Test diagnostic vector creation from raw measurements."""

    def test_normalized_interference(self):
        """Interference should pass through normalized."""
        diag = create_diagnostic_vector(
            interference=0.7,
            refinement_density=0.5,
            condition_number=1.0,
            intrinsic_dimension=100,
            hidden_dim=1000,
        )

        assert diag.interference_score == 0.7

    def test_condition_number_log_scale(self):
        """Condition number should be log-normalized."""
        # κ = 1 → 0
        diag1 = create_diagnostic_vector(0, 0, 1.0, 100, 1000)
        assert diag1.instability_score == 0.0

        # κ = 1000 → 1
        diag2 = create_diagnostic_vector(0, 0, 1000.0, 100, 1000)
        assert diag2.instability_score == 1.0

        # κ = 10 → ~0.33
        diag3 = create_diagnostic_vector(0, 0, 10.0, 100, 1000)
        assert 0.3 < diag3.instability_score < 0.4

    def test_intrinsic_dimension_ratio(self):
        """Complexity should be ratio of intrinsic to hidden dim."""
        diag = create_diagnostic_vector(
            interference=0,
            refinement_density=0,
            condition_number=1.0,
            intrinsic_dimension=500,
            hidden_dim=1000,
        )

        assert diag.complexity_score == 0.5

    def test_clamping(self):
        """Values should be clamped to [0, 1]."""
        diag = create_diagnostic_vector(
            interference=1.5,  # Beyond 1
            refinement_density=-0.5,  # Below 0
            condition_number=10000,  # Way beyond
            intrinsic_dimension=2000,  # Beyond hidden_dim
            hidden_dim=1000,
        )

        assert diag.interference_score == 1.0
        assert diag.importance_score == 0.0
        assert diag.instability_score == 1.0
        assert diag.complexity_score == 1.0


class TestFormatReport:
    """Test report formatting."""

    def test_format_direct_merge_report(self):
        """Direct merge report formats correctly."""
        polytope = SafetyPolytope(_test_bounds())

        layer_diagnostics = {i: DiagnosticVector(0.2, 0.3, 0.1, 0.2) for i in range(5)}

        profile = polytope.analyze_model_pair(layer_diagnostics)
        report = format_transformation_report(profile)

        assert "MERGE TRANSFORMATION ANALYSIS" in report
        assert "Direct Merge:       5 layers" in report

    def test_format_heavy_transform_report(self):
        """Heavy transform report shows transformations needed."""
        polytope = SafetyPolytope(_test_bounds())

        layer_diagnostics = {
            0: DiagnosticVector(0.9, 0.9, 0.95, 0.9),
        }

        profile = polytope.analyze_model_pair(layer_diagnostics)
        report = format_transformation_report(profile)

        assert "Transformations Needed" in report
        assert "Layers Needing Multiple Transformations" in report


class TestPropertyBased:
    """Property-based tests."""

    @given(
        i=st.floats(0.0, 1.0),
        p=st.floats(0.0, 1.0),
        s=st.floats(0.0, 1.0),
        c=st.floats(0.0, 1.0),
    )
    @settings(max_examples=50)
    def test_result_always_available(self, i, p, s, c):
        """Analysis always returns a result with stable fields."""
        polytope = SafetyPolytope(_test_bounds())
        diag = DiagnosticVector(i, p, s, c)
        result = polytope.analyze_layer(diag)

        assert isinstance(result.transformations, list)

    @given(
        i=st.floats(0.0, 1.0),
        p=st.floats(0.0, 1.0),
        s=st.floats(0.0, 1.0),
        c=st.floats(0.0, 1.0),
    )
    @settings(max_examples=50)
    def test_alpha_always_bounded(self, i, p, s, c):
        """Recommended alpha should always be in [0.1, 0.95]."""
        polytope = SafetyPolytope(_test_bounds())
        diag = DiagnosticVector(i, p, s, c)
        result = polytope.analyze_layer(diag, base_alpha=0.5)

        if result.recommended_alpha is not None:
            assert 0.1 <= result.recommended_alpha <= 0.95

    @given(
        i=st.floats(0.0, 1.0),
        p=st.floats(0.0, 1.0),
        s=st.floats(0.0, 1.0),
        c=st.floats(0.0, 1.0),
    )
    @settings(max_examples=50)
    def test_confidence_bounded(self, i, p, s, c):
        """Confidence should be in [0, 1]."""
        polytope = SafetyPolytope(_test_bounds())
        diag = DiagnosticVector(i, p, s, c)
        result = polytope.analyze_layer(diag)

        assert 0.0 <= result.confidence <= 1.0


class TestEdgeCases:
    """Edge case handling."""

    def test_empty_layer_diagnostics(self):
        """Empty diagnostics still works."""
        polytope = SafetyPolytope(_test_bounds())
        profile = polytope.analyze_model_pair({})

        assert len(profile.direct_merge_layers) == 0
        assert len(profile.per_layer) == 0

    def test_single_layer(self):
        """Single layer works."""
        polytope = SafetyPolytope(_test_bounds())
        profile = polytope.analyze_model_pair({0: DiagnosticVector(0.5, 0.5, 0.5, 0.5)})

        assert len(profile.per_layer) == 1

    def test_nan_in_diagnostic(self):
        """NaN handled gracefully."""
        polytope = SafetyPolytope(_test_bounds())

        # This will produce NaN in magnitude
        diag = DiagnosticVector(float("nan"), 0.3, 0.2, 0.2)

        # Should not crash
        result = polytope.analyze_layer(diag)
        assert 0.0 <= result.confidence <= 1.0
