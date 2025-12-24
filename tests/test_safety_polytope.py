"""
Tests for Safety Polytope.

Validates the unified safety boundary that combines:
- Interference (RiemannianDensity)
- Importance (RefinementDensity)
- Stability (SpectralAnalysis)
- Complexity (IntrinsicDimension)
"""
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from modelcypher.core.domain.geometry.safety_polytope import (
    SafetyPolytope,
    SafetyVerdict,
    MitigationType,
    DiagnosticVector,
    PolytopeBounds,
    create_diagnostic_vector,
    format_safety_report,
)


class TestDiagnosticVector:
    """Test DiagnosticVector properties."""

    def test_vector_property(self):
        """Vector property should return numpy array."""
        diag = DiagnosticVector(
            interference_score=0.3,
            importance_score=0.5,
            instability_score=0.2,
            complexity_score=0.4,
        )

        vec = diag.vector
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (4,)
        assert np.allclose(vec, [0.3, 0.5, 0.2, 0.4])

    def test_magnitude_property(self):
        """Magnitude should be L2 norm."""
        diag = DiagnosticVector(
            interference_score=0.3,
            importance_score=0.4,
            instability_score=0.0,
            complexity_score=0.0,
        )

        expected = np.sqrt(0.3**2 + 0.4**2)
        assert abs(diag.magnitude - expected) < 1e-6

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
        """Point well inside boundaries should be SAFE."""
        polytope = SafetyPolytope()

        diag = DiagnosticVector(
            interference_score=0.2,
            importance_score=0.3,
            instability_score=0.1,
            complexity_score=0.2,
        )

        result = polytope.check_layer(diag)

        assert result.verdict == SafetyVerdict.SAFE
        assert len(result.violations) == 0
        assert len(result.mitigations) == 0
        assert result.confidence > 0.5

    def test_point_on_boundary(self):
        """Point exactly on boundary should be SAFE but low confidence."""
        bounds = PolytopeBounds(max_interference=0.5)
        polytope = SafetyPolytope(bounds)

        diag = DiagnosticVector(
            interference_score=0.5,  # Exactly at boundary
            importance_score=0.3,
            instability_score=0.1,
            complexity_score=0.2,
        )

        result = polytope.check_layer(diag)

        # At boundary, still safe but confidence reduced
        assert result.verdict == SafetyVerdict.SAFE
        assert result.confidence < 0.8

    def test_point_outside_triggers_caution(self):
        """Point slightly outside should be CAUTION with mitigation."""
        bounds = PolytopeBounds(max_interference=0.5)
        polytope = SafetyPolytope(bounds)

        diag = DiagnosticVector(
            interference_score=0.65,  # Beyond threshold
            importance_score=0.3,
            instability_score=0.1,
            complexity_score=0.2,
        )

        result = polytope.check_layer(diag)

        assert result.verdict in (SafetyVerdict.CAUTION, SafetyVerdict.UNSAFE)
        assert len(result.violations) >= 1
        assert MitigationType.NULL_SPACE_FILTER in result.mitigations

    def test_critical_instability(self):
        """Very high instability should trigger CRITICAL."""
        polytope = SafetyPolytope()

        diag = DiagnosticVector(
            interference_score=0.3,
            importance_score=0.3,
            instability_score=0.95,  # Beyond critical threshold
            complexity_score=0.2,
        )

        result = polytope.check_layer(diag)

        assert result.verdict == SafetyVerdict.CRITICAL
        assert result.is_critical

    def test_recommended_alpha_adjustment(self):
        """Alpha should be reduced for violations."""
        polytope = SafetyPolytope()

        diag = DiagnosticVector(
            interference_score=0.8,  # High interference
            importance_score=0.3,
            instability_score=0.1,
            complexity_score=0.2,
        )

        result = polytope.check_layer(diag, base_alpha=0.5)

        assert result.recommended_alpha is not None
        assert result.recommended_alpha < 0.5  # Reduced due to interference


class TestMitigations:
    """Test mitigation recommendations."""

    def test_interference_triggers_null_space(self):
        """High interference should recommend null-space filtering."""
        bounds = PolytopeBounds(max_interference=0.5)
        polytope = SafetyPolytope(bounds)

        diag = DiagnosticVector(0.7, 0.3, 0.2, 0.2)
        result = polytope.check_layer(diag)

        assert MitigationType.NULL_SPACE_FILTER in result.mitigations

    def test_importance_triggers_alpha_reduction(self):
        """High importance should recommend alpha reduction."""
        bounds = PolytopeBounds(max_importance_for_blend=0.5)
        polytope = SafetyPolytope(bounds)

        diag = DiagnosticVector(0.2, 0.8, 0.2, 0.2)
        result = polytope.check_layer(diag)

        assert MitigationType.REDUCE_ALPHA in result.mitigations

    def test_instability_triggers_spectral_clamp(self):
        """High instability should recommend spectral clamping."""
        bounds = PolytopeBounds(max_instability=0.4)
        polytope = SafetyPolytope(bounds)

        diag = DiagnosticVector(0.2, 0.3, 0.6, 0.2)
        result = polytope.check_layer(diag)

        assert MitigationType.SPECTRAL_CLAMP in result.mitigations

    def test_complexity_triggers_tsv_prune(self):
        """High complexity should recommend TSV pruning."""
        bounds = PolytopeBounds(max_complexity=0.5)
        polytope = SafetyPolytope(bounds)

        diag = DiagnosticVector(0.2, 0.3, 0.2, 0.7)
        result = polytope.check_layer(diag)

        assert MitigationType.TSV_PRUNE in result.mitigations

    def test_extreme_magnitude_triggers_layer_skip(self):
        """Extreme overall magnitude should recommend layer skip."""
        bounds = PolytopeBounds(max_magnitude=1.0)
        polytope = SafetyPolytope(bounds)

        # All dimensions high
        diag = DiagnosticVector(0.8, 0.8, 0.8, 0.8)
        result = polytope.check_layer(diag)

        assert MitigationType.LAYER_SKIP in result.mitigations


class TestModelProfile:
    """Test model-level analysis."""

    def test_all_safe_layers(self):
        """Model with all safe layers should be SAFE overall."""
        polytope = SafetyPolytope()

        layer_diagnostics = {
            i: DiagnosticVector(0.2, 0.3, 0.1, 0.2)
            for i in range(10)
        }

        profile = polytope.analyze_model_pair(layer_diagnostics)

        assert profile.overall_verdict == SafetyVerdict.SAFE
        assert len(profile.safe_layers) == 10
        assert len(profile.caution_layers) == 0
        assert profile.mergeable

    def test_one_critical_makes_overall_critical(self):
        """Single critical layer should make overall CRITICAL."""
        polytope = SafetyPolytope()

        layer_diagnostics = {}
        for i in range(10):
            if i == 5:
                # One critical layer
                layer_diagnostics[i] = DiagnosticVector(0.2, 0.3, 0.95, 0.2)
            else:
                layer_diagnostics[i] = DiagnosticVector(0.2, 0.3, 0.1, 0.2)

        profile = polytope.analyze_model_pair(layer_diagnostics)

        assert profile.overall_verdict == SafetyVerdict.CRITICAL
        assert 5 in profile.critical_layers
        assert not profile.mergeable

    def test_mean_diagnostics(self):
        """Mean diagnostics should be computed correctly."""
        polytope = SafetyPolytope()

        layer_diagnostics = {
            0: DiagnosticVector(0.2, 0.4, 0.1, 0.3),
            1: DiagnosticVector(0.4, 0.6, 0.3, 0.5),
        }

        profile = polytope.analyze_model_pair(layer_diagnostics)

        assert abs(profile.mean_interference - 0.3) < 1e-6
        assert abs(profile.mean_importance - 0.5) < 1e-6
        assert abs(profile.mean_instability - 0.2) < 1e-6
        assert abs(profile.mean_complexity - 0.4) < 1e-6

    def test_global_mitigations_aggregated(self):
        """Global mitigations should aggregate from all layers."""
        bounds = PolytopeBounds(max_interference=0.5, max_instability=0.4)
        polytope = SafetyPolytope(bounds)

        layer_diagnostics = {
            0: DiagnosticVector(0.7, 0.3, 0.2, 0.2),  # High interference
            1: DiagnosticVector(0.2, 0.3, 0.6, 0.2),  # High instability
        }

        profile = polytope.analyze_model_pair(layer_diagnostics)

        assert MitigationType.NULL_SPACE_FILTER in profile.global_mitigations
        assert MitigationType.SPECTRAL_CLAMP in profile.global_mitigations


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

    def test_format_safe_report(self):
        """Safe report should be properly formatted."""
        polytope = SafetyPolytope()

        layer_diagnostics = {
            i: DiagnosticVector(0.2, 0.3, 0.1, 0.2)
            for i in range(5)
        }

        profile = polytope.analyze_model_pair(layer_diagnostics)
        report = format_safety_report(profile)

        assert "SAFE" in report
        assert "Mergeable: Yes" in report
        assert "Safe:     5 layers" in report

    def test_format_critical_report(self):
        """Critical report should highlight problems."""
        polytope = SafetyPolytope()

        layer_diagnostics = {
            0: DiagnosticVector(0.9, 0.9, 0.95, 0.9),
        }

        profile = polytope.analyze_model_pair(layer_diagnostics)
        report = format_safety_report(profile)

        assert "CRITICAL" in report
        assert "Mergeable: NO" in report
        assert "Problem Layers" in report


class TestPropertyBased:
    """Property-based tests."""

    @given(
        i=st.floats(0.0, 1.0),
        p=st.floats(0.0, 1.0),
        s=st.floats(0.0, 1.0),
        c=st.floats(0.0, 1.0),
    )
    @settings(max_examples=50)
    def test_verdict_always_valid(self, i, p, s, c):
        """Verdict should always be a valid SafetyVerdict."""
        polytope = SafetyPolytope()
        diag = DiagnosticVector(i, p, s, c)
        result = polytope.check_layer(diag)

        assert result.verdict in SafetyVerdict

    @given(
        i=st.floats(0.0, 1.0),
        p=st.floats(0.0, 1.0),
        s=st.floats(0.0, 1.0),
        c=st.floats(0.0, 1.0),
    )
    @settings(max_examples=50)
    def test_alpha_always_bounded(self, i, p, s, c):
        """Recommended alpha should always be in [0.1, 0.95]."""
        polytope = SafetyPolytope()
        diag = DiagnosticVector(i, p, s, c)
        result = polytope.check_layer(diag, base_alpha=0.5)

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
        polytope = SafetyPolytope()
        diag = DiagnosticVector(i, p, s, c)
        result = polytope.check_layer(diag)

        assert 0.0 <= result.confidence <= 1.0


class TestEdgeCases:
    """Edge case handling."""

    def test_empty_layer_diagnostics(self):
        """Empty diagnostics should still work."""
        polytope = SafetyPolytope()
        profile = polytope.analyze_model_pair({})

        assert profile.overall_verdict == SafetyVerdict.SAFE
        assert len(profile.safe_layers) == 0

    def test_single_layer(self):
        """Single layer should work."""
        polytope = SafetyPolytope()
        profile = polytope.analyze_model_pair({
            0: DiagnosticVector(0.5, 0.5, 0.5, 0.5)
        })

        assert len(profile.per_layer) == 1

    def test_nan_in_diagnostic(self):
        """NaN should be handled gracefully."""
        polytope = SafetyPolytope()

        # This will produce NaN in magnitude
        diag = DiagnosticVector(float('nan'), 0.3, 0.2, 0.2)

        # Should not crash
        result = polytope.check_layer(diag)
        # Verdict should still be valid
        assert result.verdict in SafetyVerdict
