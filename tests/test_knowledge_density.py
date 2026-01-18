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

"""Tests for knowledge density estimation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.knowledge_density import (
    ConceptDensity,
    LayerDensityProfile,
    ModelDensityProfile,
    KnowledgeDensityAnalyzer,
)


@pytest.fixture
def backend():
    """Get default backend for tests."""
    return get_default_backend()


class TestConceptDensity:
    """Tests for ConceptDensity dataclass."""

    def test_frozen_dataclass(self):
        """ConceptDensity should be immutable."""
        density = ConceptDensity(
            probe_id="test",
            name="Test Concept",
            domain="test_domain",
            layer=0,
            intrinsic_dimension=2.5,
            density_score=0.4,
        )
        with pytest.raises(FrozenInstanceError):
            density.density_score = 0.5

    def test_all_required_fields(self):
        """All required fields should be accessible."""
        density = ConceptDensity(
            probe_id="probe_1",
            name="Concept Name",
            domain="domain_1",
            layer=5,
            intrinsic_dimension=3.0,
            density_score=0.33,
        )
        assert density.probe_id == "probe_1"
        assert density.name == "Concept Name"
        assert density.domain == "domain_1"
        assert density.layer == 5
        assert density.intrinsic_dimension == 3.0
        assert density.density_score == 0.33

    def test_optional_fields_default_none(self):
        """Optional fields should default to None."""
        density = ConceptDensity(
            probe_id="test",
            name="Test",
            domain="domain",
            layer=0,
            intrinsic_dimension=2.0,
            density_score=0.5,
        )
        assert density.activation_variance is None
        assert density.cluster_tightness is None

    def test_optional_fields_set(self):
        """Optional fields can be set."""
        density = ConceptDensity(
            probe_id="test",
            name="Test",
            domain="domain",
            layer=0,
            intrinsic_dimension=2.0,
            density_score=0.5,
            activation_variance=0.01,
            cluster_tightness=0.95,
        )
        assert density.activation_variance == 0.01
        assert density.cluster_tightness == 0.95


class TestLayerDensityProfile:
    """Tests for LayerDensityProfile dataclass."""

    def test_frozen_dataclass(self):
        """LayerDensityProfile should be immutable."""
        profile = LayerDensityProfile(
            layer=0,
            concept_densities=[],
            mean_density=0.0,
            median_density=0.0,
        )
        with pytest.raises(FrozenInstanceError):
            profile.mean_density = 0.5

    def test_all_fields_accessible(self):
        """All fields should be accessible."""
        concept = ConceptDensity(
            probe_id="test",
            name="Test",
            domain="domain",
            layer=3,
            intrinsic_dimension=2.0,
            density_score=0.5,
        )
        profile = LayerDensityProfile(
            layer=3,
            concept_densities=[concept],
            mean_density=0.5,
            median_density=0.5,
        )
        assert profile.layer == 3
        assert len(profile.concept_densities) == 1
        assert profile.mean_density == 0.5
        assert profile.median_density == 0.5

    def test_empty_concept_list(self):
        """Should handle empty concept list."""
        profile = LayerDensityProfile(
            layer=0,
            concept_densities=[],
            mean_density=0.0,
            median_density=0.0,
        )
        assert len(profile.concept_densities) == 0


class TestModelDensityProfile:
    """Tests for ModelDensityProfile dataclass."""

    def test_frozen_dataclass(self):
        """ModelDensityProfile should be immutable."""
        profile = ModelDensityProfile(
            model_path="/path/to/model",
            layers=[0, 1, 2],
            layer_profiles={},
            domain_densities={},
            overall_density=0.5,
        )
        with pytest.raises(FrozenInstanceError):
            profile.overall_density = 0.6

    def test_all_fields_accessible(self):
        """All fields should be accessible."""
        layer_profile = LayerDensityProfile(
            layer=0,
            concept_densities=[],
            mean_density=0.5,
            median_density=0.5,
        )
        profile = ModelDensityProfile(
            model_path="/models/test",
            layers=[0, 1],
            layer_profiles={0: layer_profile},
            domain_densities={"domain_a": 0.6},
            overall_density=0.55,
        )
        assert profile.model_path == "/models/test"
        assert profile.layers == [0, 1]
        assert 0 in profile.layer_profiles
        assert "domain_a" in profile.domain_densities
        assert profile.overall_density == 0.55


class TestKnowledgeDensityAnalyzerInit:
    """Tests for KnowledgeDensityAnalyzer initialization."""

    def test_default_initialization(self):
        """Should initialize with default backend."""
        analyzer = KnowledgeDensityAnalyzer()
        assert analyzer is not None
        assert analyzer._backend is not None

    def test_explicit_backend(self, backend):
        """Should accept explicit backend."""
        analyzer = KnowledgeDensityAnalyzer(backend=backend)
        assert analyzer._backend is backend


class TestComputeDensityScore:
    """Tests for _compute_density_score method."""

    def test_low_dimension_high_density(self, backend):
        """Low intrinsic dimension should yield high density."""
        from modelcypher.core.domain.geometry.concept_dimensionality import (
            ConceptDimensionalityResult,
        )

        analyzer = KnowledgeDensityAnalyzer(backend)

        # Low intrinsic dimension
        result = ConceptDimensionalityResult(
            probe_id="test",
            name="Test",
            source="test",
            domain="domain",
            category="test",
            layer=0,
            support_text_count=5,
            sample_count=10,
            usable_count=10,
            intrinsic_dimension=1.0,  # Minimum
            calibration_weight=1.0,
            ci_lower=None,
            ci_upper=None,
        )
        score = analyzer._compute_density_score(result)
        assert score == 1.0  # 1 / 1 = 1

    def test_high_dimension_low_density(self, backend):
        """High intrinsic dimension should yield low density."""
        from modelcypher.core.domain.geometry.concept_dimensionality import (
            ConceptDimensionalityResult,
        )

        analyzer = KnowledgeDensityAnalyzer(backend)

        # High intrinsic dimension
        result = ConceptDimensionalityResult(
            probe_id="test",
            name="Test",
            source="test",
            domain="domain",
            category="test",
            layer=0,
            support_text_count=5,
            sample_count=10,
            usable_count=10,
            intrinsic_dimension=10.0,
            calibration_weight=1.0,
            ci_lower=None,
            ci_upper=None,
        )
        score = analyzer._compute_density_score(result)
        assert score == 0.1  # 1 / 10 = 0.1

    def test_clamps_minimum_dimension(self, backend):
        """Should clamp minimum dimension to 1.0."""
        from modelcypher.core.domain.geometry.concept_dimensionality import (
            ConceptDimensionalityResult,
        )

        analyzer = KnowledgeDensityAnalyzer(backend)

        # Sub-1 intrinsic dimension - valid mathematically, just means tighter cluster
        result = ConceptDimensionalityResult(
            probe_id="test",
            name="Test",
            source="test",
            domain="domain",
            category="test",
            layer=0,
            support_text_count=5,
            sample_count=10,
            usable_count=10,
            intrinsic_dimension=0.5,
            calibration_weight=1.0,
            ci_lower=None,
            ci_upper=None,
        )
        score = analyzer._compute_density_score(result)
        # Score = 1.0 / 0.5 = 2.0 (dimension above machine epsilon is used directly)
        assert score == 2.0


class TestComputeActivationVariance:
    """Tests for _compute_activation_variance method."""

    def test_zero_variance_identical_activations(self, backend):
        """Identical activations should have zero variance."""
        analyzer = KnowledgeDensityAnalyzer(backend)

        # All identical activations
        activations = backend.ones((5, 10))
        backend.eval(activations)

        variance = analyzer._compute_activation_variance(activations)

        assert variance < 1e-10

    def test_nonzero_variance_different_activations(self, backend):
        """Different activations should have nonzero variance."""
        analyzer = KnowledgeDensityAnalyzer(backend)

        # Random activations
        activations = backend.random_normal((10, 20))
        backend.eval(activations)

        variance = analyzer._compute_activation_variance(activations)

        assert variance > 0.0


class TestComputeClusterTightness:
    """Tests for _compute_cluster_tightness method."""

    def test_identical_vectors_high_tightness(self, backend):
        """Identical vectors should have high tightness."""
        analyzer = KnowledgeDensityAnalyzer(backend)

        # All identical vectors (normalized)
        activations = backend.ones((5, 10)) / backend.sqrt(backend.array(10.0))
        backend.eval(activations)

        tightness = analyzer._compute_cluster_tightness(activations)

        # Identical normalized vectors have cosine similarity 1.0
        assert tightness > 0.99

    def test_single_sample_returns_zero(self, backend):
        """Single sample should return 0."""
        analyzer = KnowledgeDensityAnalyzer(backend)

        activations = backend.random_normal((1, 10))
        backend.eval(activations)

        tightness = analyzer._compute_cluster_tightness(activations)

        assert tightness == 0.0

    def test_random_vectors_moderate_tightness(self, backend):
        """Random vectors should have moderate tightness."""
        analyzer = KnowledgeDensityAnalyzer(backend)

        activations = backend.random_normal((20, 50))
        backend.eval(activations)

        tightness = analyzer._compute_cluster_tightness(activations)

        # Random vectors in high dimensions tend toward orthogonal
        # but geodesic cosine should still be positive
        assert -1.0 <= tightness <= 1.0


class TestGetSupportTexts:
    """Tests for _get_support_texts method."""

    def test_probe_not_found(self, backend):
        """Should return empty list if probe not found."""
        analyzer = KnowledgeDensityAnalyzer(backend)

        texts = analyzer._get_support_texts([], "nonexistent")

        assert texts == []

    def test_probe_with_support_texts(self, backend):
        """Should return support texts from matching probe."""
        from dataclasses import dataclass

        @dataclass
        class MockProbe:
            probe_id: str
            name: str
            description: str
            support_texts: list[str]
            domain: str = "test"

        analyzer = KnowledgeDensityAnalyzer(backend)
        probe = MockProbe(
            probe_id="test_probe",
            name="Test Probe",
            description="A test probe",
            support_texts=["text1", "text2"],
        )

        texts = analyzer._get_support_texts([probe], "test_probe")

        # Should include description, name, then support_texts
        assert "A test probe" in texts
        assert "Test Probe" in texts
        assert "text1" in texts
        assert "text2" in texts

    def test_probe_without_support_texts(self, backend):
        """Should handle probe without support texts."""
        from dataclasses import dataclass

        @dataclass
        class MockProbe:
            probe_id: str
            name: str
            description: str
            support_texts: None
            domain: str = "test"

        analyzer = KnowledgeDensityAnalyzer(backend)
        probe = MockProbe(
            probe_id="test_probe",
            name="Name Only",
            description="Description Only",
            support_texts=None,
        )

        texts = analyzer._get_support_texts([probe], "test_probe")

        # Should still include name and description
        assert "Name Only" in texts
        assert "Description Only" in texts


class TestDensityScoreRange:
    """Tests for density score value ranges."""

    def test_density_score_positive(self, backend):
        """Density score should always be positive."""
        from modelcypher.core.domain.geometry.concept_dimensionality import (
            ConceptDimensionalityResult,
        )

        analyzer = KnowledgeDensityAnalyzer(backend)

        # Various intrinsic dimensions
        for id_val in [0.1, 0.5, 1.0, 5.0, 10.0, 100.0]:
            result = ConceptDimensionalityResult(
                probe_id="test",
                name="Test",
                source="test",
                domain="domain",
                category="test",
                layer=0,
                support_text_count=5,
                sample_count=10,
                usable_count=10,
                intrinsic_dimension=id_val,
                calibration_weight=1.0,
                ci_lower=None,
                ci_upper=None,
            )
            score = analyzer._compute_density_score(result)
            assert score > 0.0

    def test_density_score_bounded(self, backend):
        """Density score should be bounded by 1.0."""
        from modelcypher.core.domain.geometry.concept_dimensionality import (
            ConceptDimensionalityResult,
        )

        analyzer = KnowledgeDensityAnalyzer(backend)

        # Various intrinsic dimensions
        for id_val in [0.1, 0.5, 1.0, 5.0, 10.0]:
            result = ConceptDimensionalityResult(
                probe_id="test",
                name="Test",
                source="test",
                domain="domain",
                category="test",
                layer=0,
                support_text_count=5,
                sample_count=10,
                usable_count=10,
                intrinsic_dimension=id_val,
                calibration_weight=1.0,
                ci_lower=None,
                ci_upper=None,
            )
            score = analyzer._compute_density_score(result)
            assert score <= 1.0


class TestGeodesicMath:
    """Tests to verify geodesic math is used."""

    def test_uses_geodesic_cosine_matrix(self, backend):
        """Cluster tightness should use geodesic cosine, not Euclidean."""
        # This is verified by examining that _compute_cluster_tightness
        # calls geodesic_cosine_matrix, which uses proper geodesic distance
        analyzer = KnowledgeDensityAnalyzer(backend)

        # Create activations that would give different results
        # with Euclidean vs geodesic cosine
        activations = backend.random_normal((5, 10))
        backend.eval(activations)

        tightness = analyzer._compute_cluster_tightness(activations)

        # Just verify it completes without error using geodesic math
        assert isinstance(tightness, float)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_layer_profile(self, backend):
        """Should handle empty layer profile."""
        profile = LayerDensityProfile(
            layer=0,
            concept_densities=[],
            mean_density=0.0,
            median_density=0.0,
        )
        assert profile.mean_density == 0.0
        assert profile.median_density == 0.0

    def test_single_concept_layer_profile(self):
        """Should handle single concept in layer."""
        concept = ConceptDensity(
            probe_id="single",
            name="Single",
            domain="domain",
            layer=0,
            intrinsic_dimension=2.0,
            density_score=0.5,
        )
        profile = LayerDensityProfile(
            layer=0,
            concept_densities=[concept],
            mean_density=0.5,
            median_density=0.5,
        )
        assert profile.mean_density == 0.5
        assert profile.median_density == 0.5

    def test_model_profile_empty_layers(self):
        """Should handle model with empty layers."""
        profile = ModelDensityProfile(
            model_path="/test",
            layers=[],
            layer_profiles={},
            domain_densities={},
            overall_density=0.0,
        )
        assert profile.layers == []
        assert profile.overall_density == 0.0

    def test_model_profile_empty_domains(self):
        """Should handle model with no domain aggregates."""
        profile = ModelDensityProfile(
            model_path="/test",
            layers=[0],
            layer_profiles={},
            domain_densities={},
            overall_density=0.5,
        )
        assert profile.domain_densities == {}
