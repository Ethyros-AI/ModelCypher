# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for manifold_profile module.

Covers ManifoldPoint, RegionThresholds, ManifoldRegion (including
classify and RegionCharacter enum), ManifoldProfile, RegionQueryResult,
and InterventionSuggestion.

Skips the distance() method which requires a backend.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

import pytest

from modelcypher.core.domain.geometry.manifold_profile import (
    InterventionSuggestion,
    ManifoldPoint,
    ManifoldProfile,
    ManifoldRegion,
    RegionQueryResult,
    RegionThresholds,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_point(**overrides) -> ManifoldPoint:
    """Create a ManifoldPoint with sane defaults."""
    defaults = dict(
        mean_entropy=1.5,
        entropy_variance=0.2,
        first_token_entropy=2.0,
        gate_count=3,
        mean_gate_similarity=0.7,
        dominant_gate_category=0.3,
        entropy_path_correlation=0.5,
        assessment_strength=0.8,
        prompt_hash="abc123",
    )
    defaults.update(overrides)
    return ManifoldPoint(**defaults)


def _make_region(
    region_type: ManifoldRegion.RegionCharacter = ManifoldRegion.RegionCharacter.DENSE,
    intrinsic_dimension: float | None = 3.0,
    **point_overrides,
) -> ManifoldRegion:
    """Create a ManifoldRegion with sane defaults."""
    centroid = _make_point(**point_overrides)
    return ManifoldRegion(
        id=uuid4(),
        region_type=region_type,
        centroid=centroid,
        member_count=5,
        member_ids=[uuid4() for _ in range(5)],
        dominant_gates=["REASON", "CONCLUDE"],
        intrinsic_dimension=intrinsic_dimension,
        radius=0.5,
    )


# ---------------------------------------------------------------------------
# ManifoldPoint
# ---------------------------------------------------------------------------


class TestManifoldPoint:
    """Tests for ManifoldPoint frozen dataclass."""

    def test_instantiation(self):
        p = _make_point()
        assert p.mean_entropy == 1.5
        assert p.entropy_variance == 0.2
        assert p.first_token_entropy == 2.0
        assert p.gate_count == 3
        assert p.mean_gate_similarity == 0.7
        assert p.dominant_gate_category == 0.3
        assert p.entropy_path_correlation == 0.5
        assert p.assessment_strength == 0.8
        assert p.prompt_hash == "abc123"

    def test_frozen(self):
        p = _make_point()
        with pytest.raises(AttributeError):
            p.mean_entropy = 99.0  # type: ignore[misc]

    def test_feature_dimension_class_var(self):
        assert ManifoldPoint.feature_dimension == 8

    def test_feature_names_class_var(self):
        names = ManifoldPoint.feature_names
        assert isinstance(names, list)
        assert len(names) == 8
        assert "meanEntropy" in names
        assert "entropyVariance" in names
        assert "firstTokenEntropy" in names
        assert "gateCount" in names
        assert "meanGateSimilarity" in names
        assert "dominantGateCategory" in names
        assert "entropyPathCorrelation" in names
        assert "assessmentStrength" in names

    def test_feature_vector_length(self):
        p = _make_point()
        fv = p.feature_vector
        assert len(fv) == 8

    def test_feature_vector_values(self):
        p = _make_point(
            mean_entropy=1.0,
            entropy_variance=2.0,
            first_token_entropy=3.0,
            gate_count=4,
            mean_gate_similarity=5.0,
            dominant_gate_category=6.0,
            entropy_path_correlation=7.0,
            assessment_strength=8.0,
        )
        fv = p.feature_vector
        assert fv == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    def test_feature_vector_all_floats(self):
        p = _make_point(gate_count=7)
        fv = p.feature_vector
        for v in fv:
            assert isinstance(v, float)

    def test_default_id_is_uuid(self):
        p = _make_point()
        assert isinstance(p.id, UUID)

    def test_default_timestamp(self):
        p = _make_point()
        assert isinstance(p.timestamp, datetime)

    def test_intervention_level_default_none(self):
        p = _make_point()
        assert p.intervention_level is None

    def test_intervention_level_set(self):
        p = _make_point(intervention_level=3)
        assert p.intervention_level == 3


# ---------------------------------------------------------------------------
# RegionThresholds
# ---------------------------------------------------------------------------


class TestRegionThresholds:
    """Tests for RegionThresholds frozen dataclass and from_data()."""

    def test_instantiation(self):
        t = RegionThresholds(
            low_entropy=0.5,
            high_entropy=2.0,
            low_variance=0.1,
            high_variance=0.5,
            high_coherence=0.8,
            low_coherence=0.3,
        )
        assert t.low_entropy == 0.5
        assert t.high_entropy == 2.0
        assert t.low_variance == 0.1
        assert t.high_variance == 0.5
        assert t.high_coherence == 0.8
        assert t.low_coherence == 0.3

    def test_frozen(self):
        t = RegionThresholds(0.5, 2.0, 0.1, 0.5, 0.8, 0.3)
        with pytest.raises(AttributeError):
            t.low_entropy = 99.0  # type: ignore[misc]

    def test_from_data_basic(self, any_backend, monkeypatch):
        """from_data should compute thresholds from value lists."""
        import modelcypher.core.domain.geometry.manifold_profile as mp_mod

        monkeypatch.setattr(mp_mod, "get_default_backend", lambda: any_backend)

        entropies = [0.1, 0.5, 2.0, 5.0]
        variances = [0.01, 0.1, 0.5, 1.0]
        coherences = [0.1, 0.3, 0.7, 0.9]

        t = RegionThresholds.from_data(entropies, variances, coherences)
        assert isinstance(t, RegionThresholds)
        # Thresholds should fall within the data ranges
        assert t.low_entropy >= min(entropies)
        assert t.high_entropy >= t.low_entropy
        assert t.low_variance >= min(variances)
        assert t.high_variance >= t.low_variance

    def test_from_data_single_value(self, any_backend, monkeypatch):
        """Single-element lists should not crash."""
        import modelcypher.core.domain.geometry.manifold_profile as mp_mod

        monkeypatch.setattr(mp_mod, "get_default_backend", lambda: any_backend)

        t = RegionThresholds.from_data([1.0], [0.5], [0.8])
        assert isinstance(t, RegionThresholds)

    def test_from_data_empty(self, any_backend, monkeypatch):
        """Empty lists should produce 0.0 thresholds."""
        import modelcypher.core.domain.geometry.manifold_profile as mp_mod

        monkeypatch.setattr(mp_mod, "get_default_backend", lambda: any_backend)

        t = RegionThresholds.from_data([], [], [])
        assert t.low_entropy == 0.0
        assert t.high_entropy == 0.0


# ---------------------------------------------------------------------------
# ManifoldRegion.RegionCharacter enum
# ---------------------------------------------------------------------------


class TestRegionCharacter:
    """Tests for the RegionCharacter enum."""

    def test_dense(self):
        assert ManifoldRegion.RegionCharacter.DENSE.value == "dense"

    def test_sparse(self):
        assert ManifoldRegion.RegionCharacter.SPARSE.value == "sparse"

    def test_transitional(self):
        assert ManifoldRegion.RegionCharacter.TRANSITIONAL.value == "transitional"

    def test_all_members(self):
        members = set(ManifoldRegion.RegionCharacter)
        assert len(members) == 3

    def test_is_str_enum(self):
        # RegionCharacter inherits from str
        assert isinstance(ManifoldRegion.RegionCharacter.DENSE, str)


# ---------------------------------------------------------------------------
# ManifoldRegion.classify
# ---------------------------------------------------------------------------


class TestManifoldRegionClassify:
    """Tests for ManifoldRegion.classify() static method."""

    def _thresholds(self) -> RegionThresholds:
        return RegionThresholds(
            low_entropy=1.0,
            high_entropy=3.0,
            low_variance=0.1,
            high_variance=0.5,
            high_coherence=0.8,
            low_coherence=0.3,
        )

    def test_dense_classification(self):
        """Low entropy, low variance, high coherence -> DENSE."""
        thresholds = self._thresholds()
        point = _make_point(
            mean_entropy=0.5,
            entropy_variance=0.05,
            mean_gate_similarity=0.9,
        )
        result = ManifoldRegion.classify(point, thresholds)
        assert result == ManifoldRegion.RegionCharacter.DENSE

    def test_sparse_classification_high_entropy(self):
        """High entropy -> SPARSE."""
        thresholds = self._thresholds()
        point = _make_point(
            mean_entropy=4.0,
            entropy_variance=0.05,
            mean_gate_similarity=0.2,
        )
        result = ManifoldRegion.classify(point, thresholds)
        assert result == ManifoldRegion.RegionCharacter.SPARSE

    def test_transitional_high_variance(self):
        """High variance -> TRANSITIONAL."""
        thresholds = self._thresholds()
        point = _make_point(
            mean_entropy=2.0,
            entropy_variance=0.8,
            mean_gate_similarity=0.5,
        )
        result = ManifoldRegion.classify(point, thresholds)
        assert result == ManifoldRegion.RegionCharacter.TRANSITIONAL

    def test_transitional_mid_entropy_mid_variance(self):
        """Mid entropy with mid variance -> TRANSITIONAL."""
        thresholds = self._thresholds()
        point = _make_point(
            mean_entropy=2.0,
            entropy_variance=0.3,
            mean_gate_similarity=0.5,
        )
        result = ManifoldRegion.classify(point, thresholds)
        assert result == ManifoldRegion.RegionCharacter.TRANSITIONAL

    def test_return_type(self):
        thresholds = self._thresholds()
        point = _make_point()
        result = ManifoldRegion.classify(point, thresholds)
        assert isinstance(result, ManifoldRegion.RegionCharacter)


# ---------------------------------------------------------------------------
# ManifoldRegion properties
# ---------------------------------------------------------------------------


class TestManifoldRegion:
    """Tests for ManifoldRegion dataclass and properties."""

    def test_instantiation(self):
        region = _make_region()
        assert isinstance(region.id, UUID)
        assert region.member_count == 5
        assert len(region.member_ids) == 5
        assert region.dominant_gates == ["REASON", "CONCLUDE"]
        assert region.intrinsic_dimension == 3.0
        assert region.radius == 0.5

    def test_frozen(self):
        region = _make_region()
        with pytest.raises(AttributeError):
            region.radius = 99.0  # type: ignore[misc]

    def test_entropy_property(self):
        region = _make_region(mean_entropy=2.5)
        assert region.entropy == 2.5

    def test_variance_property(self):
        region = _make_region(entropy_variance=0.4)
        assert region.variance == 0.4

    def test_coherence_property(self):
        region = _make_region(mean_gate_similarity=0.85)
        assert region.coherence == 0.85

    def test_none_intrinsic_dimension(self):
        region = _make_region(intrinsic_dimension=None)
        assert region.intrinsic_dimension is None


# ---------------------------------------------------------------------------
# ManifoldProfile
# ---------------------------------------------------------------------------


class TestManifoldProfile:
    """Tests for ManifoldProfile dataclass and compute_statistics."""

    def test_instantiation(self):
        profile = ManifoldProfile(
            id=uuid4(),
            model_id="test-model",
            model_name="Test Model",
        )
        assert profile.model_id == "test-model"
        assert profile.model_name == "Test Model"
        assert profile.regions == []
        assert profile.recent_points == []
        assert profile.total_point_count == 0
        assert profile.version == 1

    def test_not_frozen(self):
        """ManifoldProfile is a regular (mutable) dataclass."""
        profile = ManifoldProfile(
            id=uuid4(),
            model_id="m",
            model_name="n",
        )
        profile.total_point_count = 42
        assert profile.total_point_count == 42

    def test_compute_statistics_empty(self):
        profile = ManifoldProfile(
            id=uuid4(),
            model_id="m",
            model_name="n",
        )
        stats = profile.compute_statistics()
        assert stats.total_points == 0
        assert stats.region_count == 0
        assert stats.dense_region_count == 0
        assert stats.sparse_region_count == 0
        assert stats.transitional_region_count == 0
        assert stats.mean_intrinsic_dimension is None
        assert stats.recent_point_count == 0

    def test_compute_statistics_with_regions(self):
        dense = _make_region(
            region_type=ManifoldRegion.RegionCharacter.DENSE,
            intrinsic_dimension=2.0,
        )
        sparse = _make_region(
            region_type=ManifoldRegion.RegionCharacter.SPARSE,
            intrinsic_dimension=4.0,
        )
        trans = _make_region(
            region_type=ManifoldRegion.RegionCharacter.TRANSITIONAL,
            intrinsic_dimension=3.0,
        )

        profile = ManifoldProfile(
            id=uuid4(),
            model_id="m",
            model_name="n",
            regions=[dense, sparse, trans],
            total_point_count=100,
        )

        stats = profile.compute_statistics()
        assert stats.total_points == 100
        assert stats.region_count == 3
        assert stats.dense_region_count == 1
        assert stats.sparse_region_count == 1
        assert stats.transitional_region_count == 1
        assert stats.mean_intrinsic_dimension == pytest.approx(3.0)
        assert stats.recent_point_count == 0

    def test_compute_statistics_mean_dim_with_none(self):
        """Regions with None intrinsic_dimension should be excluded from mean."""
        r1 = _make_region(
            region_type=ManifoldRegion.RegionCharacter.DENSE,
            intrinsic_dimension=4.0,
        )
        r2 = _make_region(
            region_type=ManifoldRegion.RegionCharacter.DENSE,
            intrinsic_dimension=None,
        )
        r3 = _make_region(
            region_type=ManifoldRegion.RegionCharacter.DENSE,
            intrinsic_dimension=6.0,
        )

        profile = ManifoldProfile(
            id=uuid4(),
            model_id="m",
            model_name="n",
            regions=[r1, r2, r3],
        )

        stats = profile.compute_statistics()
        # Mean of 4.0 and 6.0 (None excluded)
        assert stats.mean_intrinsic_dimension == pytest.approx(5.0)

    def test_compute_statistics_with_recent_points(self):
        p1 = _make_point()
        p2 = _make_point()

        profile = ManifoldProfile(
            id=uuid4(),
            model_id="m",
            model_name="n",
            recent_points=[p1, p2],
        )

        stats = profile.compute_statistics()
        assert stats.recent_point_count == 2

    def test_statistics_frozen(self):
        profile = ManifoldProfile(
            id=uuid4(),
            model_id="m",
            model_name="n",
        )
        stats = profile.compute_statistics()
        with pytest.raises(AttributeError):
            stats.total_points = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RegionQueryResult
# ---------------------------------------------------------------------------


class TestRegionQueryResult:
    """Tests for RegionQueryResult frozen dataclass."""

    def test_instantiation_with_region(self):
        region = _make_region()
        rqr = RegionQueryResult(
            nearest_region=region,
            distance=0.5,
            is_within_region=True,
            suggested_character=ManifoldRegion.RegionCharacter.DENSE,
            confidence=0.9,
        )
        assert rqr.nearest_region is region
        assert rqr.distance == 0.5
        assert rqr.is_within_region is True
        assert rqr.suggested_character == ManifoldRegion.RegionCharacter.DENSE
        assert rqr.confidence == 0.9

    def test_instantiation_with_no_region(self):
        rqr = RegionQueryResult(
            nearest_region=None,
            distance=float("inf"),
            is_within_region=False,
            suggested_character=ManifoldRegion.RegionCharacter.SPARSE,
            confidence=0.0,
        )
        assert rqr.nearest_region is None
        assert rqr.is_within_region is False

    def test_frozen(self):
        rqr = RegionQueryResult(
            nearest_region=None,
            distance=1.0,
            is_within_region=False,
            suggested_character=ManifoldRegion.RegionCharacter.TRANSITIONAL,
            confidence=0.5,
        )
        with pytest.raises(AttributeError):
            rqr.distance = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# InterventionSuggestion
# ---------------------------------------------------------------------------


class TestInterventionSuggestion:
    """Tests for InterventionSuggestion frozen dataclass."""

    def test_instantiation(self):
        s = InterventionSuggestion(
            level=2,
            reason="High entropy detected",
            confidence=0.85,
            based_on_history=True,
            similar_point_count=10,
        )
        assert s.level == 2
        assert s.reason == "High entropy detected"
        assert s.confidence == 0.85
        assert s.based_on_history is True
        assert s.similar_point_count == 10

    def test_frozen(self):
        s = InterventionSuggestion(1, "test", 0.5, False, 0)
        with pytest.raises(AttributeError):
            s.level = 99  # type: ignore[misc]

    def test_no_history_factory(self):
        s = InterventionSuggestion.no_history()
        assert s.level == 0
        assert s.reason == "No historical data available"
        assert s.confidence == 0.0
        assert s.based_on_history is False
        assert s.similar_point_count == 0

    def test_no_history_returns_new_instance(self):
        a = InterventionSuggestion.no_history()
        b = InterventionSuggestion.no_history()
        assert a == b
        # Both have the same values
        assert a.level == b.level
        assert a.reason == b.reason
