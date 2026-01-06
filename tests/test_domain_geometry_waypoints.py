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

"""Comprehensive tests for domain_geometry_waypoints.py.

Tests:
- AtlasDomain enum (unified domain definitions)
- DomainGeometryScore dataclass
- ModelGeometryProfile dataclass (properties, serialization)
- DomainGeometryDelta dataclass
- PreMergeGeometryAudit dataclass
- PostMergeGeometryValidation dataclass
- DomainGeometryWaypointService (strength computation, audit creation)
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.domains import AtlasDomain
from modelcypher.core.domain.geometry.domain_geometry_waypoints import (
    DomainGeometryDelta,
    DomainGeometryScore,
    DomainGeometryWaypointService,
    ModelGeometryProfile,
    PostMergeGeometryValidation,
    PreMergeGeometryAudit,
)
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


def _eps() -> float:
    backend = get_default_backend()
    return machine_epsilon(backend, backend.array([1.0]))


# =============================================================================
# AtlasDomain Enum Tests
# =============================================================================


class TestAtlasDomain:
    """Tests for AtlasDomain enum.

    Note: AtlasDomain is now an alias for AtlasDomain, which contains
    all unified domains. The original 4 domains (SPATIAL, SOCIAL, TEMPORAL, MORAL)
    are preserved, but SOCIAL is now RELATIONAL for consistency with AtlasDomain.
    """

    def test_core_domains_exist(self) -> None:
        """Core geometry domains should be defined."""
        # Original domains (SOCIAL -> RELATIONAL)
        assert AtlasDomain.SPATIAL.value == "spatial"
        assert AtlasDomain.RELATIONAL.value == "relational"  # Was SOCIAL
        assert AtlasDomain.TEMPORAL.value == "temporal"
        assert AtlasDomain.MORAL.value == "moral"

    def test_domain_is_string(self) -> None:
        """AtlasDomain should be a string enum."""
        assert isinstance(AtlasDomain.SPATIAL, str)
        assert AtlasDomain.SPATIAL == "spatial"

    def test_domain_iteration(self) -> None:
        """Should be able to iterate all domains."""
        domains = list(AtlasDomain)
        # Now includes all AtlasDomain values (15 total, including PHYSICAL)
        assert len(domains) == 15
        assert AtlasDomain.SPATIAL in domains
        assert AtlasDomain.RELATIONAL in domains  # Was SOCIAL
        assert AtlasDomain.TEMPORAL in domains
        assert AtlasDomain.MORAL in domains
        # Additional domains from AtlasDomain
        assert AtlasDomain.MATHEMATICAL in domains
        assert AtlasDomain.LOGICAL in domains
        assert AtlasDomain.COMPUTATIONAL in domains
        assert AtlasDomain.PHYSICAL in domains


# =============================================================================
# DomainGeometryScore Tests
# =============================================================================


class TestDomainGeometryScore:
    """Tests for DomainGeometryScore dataclass."""

    def test_basic_creation(self) -> None:
        """Should create score with all fields."""
        score = DomainGeometryScore(
            domain=AtlasDomain.SPATIAL,
            manifold_score=0.85,
            axis_orthogonality=0.92,
            gradient_consistency=0.78,
            anchors_probed=100,
            layer_analyzed=12,
        )

        assert score.domain == AtlasDomain.SPATIAL
        assert score.manifold_score == 0.85
        assert score.axis_orthogonality == 0.92
        assert score.gradient_consistency == 0.78
        assert score.anchors_probed == 100
        assert score.layer_analyzed == 12

    def test_frozen(self) -> None:
        """Score should be frozen (immutable)."""
        score = DomainGeometryScore(
            domain=AtlasDomain.MORAL,
            manifold_score=0.7,
            axis_orthogonality=0.8,
            gradient_consistency=0.6,
            anchors_probed=50,
            layer_analyzed=8,
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            score.manifold_score = 0.9  # type: ignore

    def test_each_domain(self) -> None:
        """Should accept each domain type."""
        for domain in AtlasDomain:
            score = DomainGeometryScore(
                domain=domain,
                manifold_score=0.5,
                axis_orthogonality=0.5,
                gradient_consistency=0.5,
                anchors_probed=10,
                layer_analyzed=0,
            )
            assert score.domain == domain


# =============================================================================
# ModelGeometryProfile Tests
# =============================================================================


class TestModelGeometryProfile:
    """Tests for ModelGeometryProfile dataclass."""

    def _create_sample_profile(
        self,
        scores: dict[AtlasDomain, float] | None = None,
    ) -> ModelGeometryProfile:
        """Create a sample profile for testing."""
        if scores is None:
            scores = {
                AtlasDomain.SPATIAL: 0.9,
                AtlasDomain.RELATIONAL: 0.7,
                AtlasDomain.TEMPORAL: 0.8,
                AtlasDomain.MORAL: 0.6,
            }

        domain_scores = {}
        for domain, manifold_score in scores.items():
            domain_scores[domain] = DomainGeometryScore(
                domain=domain,
                manifold_score=manifold_score,
                axis_orthogonality=0.8,
                gradient_consistency=0.7,
                anchors_probed=50,
                layer_analyzed=12,
            )

        return ModelGeometryProfile(
            model_path="/path/to/model",
            layer=12,
            domain_scores=domain_scores,
            computed_at=datetime(2025, 1, 1, 12, 0, 0),
            total_anchors=200,
        )

    def test_basic_creation(self) -> None:
        """Should create profile with all fields."""
        profile = self._create_sample_profile()

        assert profile.model_path == "/path/to/model"
        assert profile.layer == 12
        assert len(profile.domain_scores) == 4
        assert profile.total_anchors == 200

    def test_mean_manifold_score(self) -> None:
        """mean_manifold_score should compute average."""
        profile = self._create_sample_profile(
            {
                AtlasDomain.SPATIAL: 0.8,
                AtlasDomain.RELATIONAL: 0.6,
            }
        )

        expected_mean = (0.8 + 0.6) / 2
        assert abs(profile.mean_manifold_score - expected_mean) <= _eps()

    def test_mean_manifold_score_empty(self) -> None:
        """mean_manifold_score should return 0 for empty scores."""
        profile = ModelGeometryProfile(
            model_path="/test",
            layer=0,
            domain_scores={},
            computed_at=datetime.now(),
            total_anchors=0,
        )

        assert abs(profile.mean_manifold_score) <= _eps()

    def test_strongest_domain(self) -> None:
        """strongest_domain should return domain with highest score."""
        profile = self._create_sample_profile(
            {
                AtlasDomain.SPATIAL: 0.9,  # Highest
                AtlasDomain.RELATIONAL: 0.5,
                AtlasDomain.TEMPORAL: 0.7,
            }
        )

        assert profile.strongest_domain == AtlasDomain.SPATIAL

    def test_strongest_domain_empty(self) -> None:
        """strongest_domain should return None for empty scores."""
        profile = ModelGeometryProfile(
            model_path="/test",
            layer=0,
            domain_scores={},
            computed_at=datetime.now(),
            total_anchors=0,
        )

        assert profile.strongest_domain is None

    def test_weakest_domain(self) -> None:
        """weakest_domain should return domain with lowest score."""
        profile = self._create_sample_profile(
            {
                AtlasDomain.SPATIAL: 0.9,
                AtlasDomain.RELATIONAL: 0.5,  # Lowest
                AtlasDomain.TEMPORAL: 0.7,
            }
        )

        assert profile.weakest_domain == AtlasDomain.RELATIONAL

    def test_weakest_domain_empty(self) -> None:
        """weakest_domain should return None for empty scores."""
        profile = ModelGeometryProfile(
            model_path="/test",
            layer=0,
            domain_scores={},
            computed_at=datetime.now(),
            total_anchors=0,
        )

        assert profile.weakest_domain is None

    def test_to_dict(self) -> None:
        """to_dict should serialize profile correctly."""
        profile = self._create_sample_profile(
            {
                AtlasDomain.SPATIAL: 0.85,
                AtlasDomain.MORAL: 0.65,
            }
        )

        d = profile.to_dict()

        assert d["modelPath"] == "/path/to/model"
        assert d["layer"] == 12
        assert d["totalAnchors"] == 200
        assert "domainScores" in d
        assert "spatial" in d["domainScores"]
        assert d["domainScores"]["spatial"]["manifoldScore"] == 0.85
        assert d["domainScores"]["moral"]["manifoldScore"] == 0.65
        assert "computedAt" in d
        expected_mean = (0.85 + 0.65) / 2
        assert abs(d["meanManifoldScore"] - expected_mean) <= _eps()

    def test_to_dict_structure(self) -> None:
        """to_dict should have correct structure for each domain score."""
        profile = self._create_sample_profile()
        d = profile.to_dict()

        for domain_key, score_dict in d["domainScores"].items():
            assert "manifoldScore" in score_dict
            assert "axisOrthogonality" in score_dict
            assert "gradientConsistency" in score_dict
            assert "anchorsProbed" in score_dict


# =============================================================================
# DomainGeometryDelta Tests
# =============================================================================


class TestDomainGeometryDelta:
    """Tests for DomainGeometryDelta dataclass."""

    def test_basic_creation(self) -> None:
        """Should create delta with all fields."""
        delta = DomainGeometryDelta(
            domain=AtlasDomain.SPATIAL,
            source_score=0.8,
            target_score=0.6,
            delta=0.2,
        )

        assert delta.domain == AtlasDomain.SPATIAL
        assert delta.source_score == 0.8
        assert delta.target_score == 0.6
        assert delta.delta == 0.2

    def test_delta_calculation(self) -> None:
        """Delta should be absolute difference."""
        # Source higher
        delta1 = DomainGeometryDelta(
            domain=AtlasDomain.RELATIONAL,
            source_score=0.9,
            target_score=0.5,
            delta=0.4,
        )
        expected_delta1 = abs(delta1.source_score - delta1.target_score)
        assert abs(delta1.delta - expected_delta1) <= _eps()

        # Target higher
        delta2 = DomainGeometryDelta(
            domain=AtlasDomain.TEMPORAL,
            source_score=0.3,
            target_score=0.7,
            delta=0.4,
        )
        expected_delta2 = abs(delta2.source_score - delta2.target_score)
        assert abs(delta2.delta - expected_delta2) <= _eps()

    def test_each_domain(self) -> None:
        """Should accept each domain type."""
        for domain in AtlasDomain:
            delta = DomainGeometryDelta(
                domain=domain,
                source_score=0.5,
                target_score=0.5,
                delta=0.0,
            )
            assert delta.domain == domain


# =============================================================================
# PreMergeGeometryAudit Tests
# =============================================================================


class TestPreMergeGeometryAudit:
    """Tests for PreMergeGeometryAudit dataclass."""

    def _create_sample_profile(
        self, path: str, scores: dict[AtlasDomain, float]
    ) -> ModelGeometryProfile:
        """Create a sample profile."""
        domain_scores = {}
        for domain, manifold_score in scores.items():
            domain_scores[domain] = DomainGeometryScore(
                domain=domain,
                manifold_score=manifold_score,
                axis_orthogonality=0.8,
                gradient_consistency=0.7,
                anchors_probed=50,
                layer_analyzed=12,
            )

        return ModelGeometryProfile(
            model_path=path,
            layer=12,
            domain_scores=domain_scores,
            computed_at=datetime.now(),
            total_anchors=100,
        )

    def test_basic_creation(self) -> None:
        """Should create audit with all fields."""
        source = self._create_sample_profile(
            "/source", {AtlasDomain.SPATIAL: 0.8}
        )
        target = self._create_sample_profile(
            "/target", {AtlasDomain.SPATIAL: 0.6}
        )
        deltas = [
            DomainGeometryDelta(
                domain=AtlasDomain.SPATIAL,
                source_score=0.8,
                target_score=0.6,
                delta=0.2,
            )
        ]

        audit = PreMergeGeometryAudit(
            source_profile=source,
            target_profile=target,
            domain_deltas=deltas,
            strength_ratio_variance=0.01,
        )

        assert audit.source_profile.model_path == "/source"
        assert audit.target_profile.model_path == "/target"
        assert len(audit.domain_deltas) == 1
        assert audit.strength_ratio_variance == 0.01

    def test_to_dict(self) -> None:
        """to_dict should serialize audit correctly."""
        source = self._create_sample_profile(
            "/source", {AtlasDomain.SPATIAL: 0.8, AtlasDomain.MORAL: 0.7}
        )
        target = self._create_sample_profile(
            "/target", {AtlasDomain.SPATIAL: 0.6, AtlasDomain.MORAL: 0.9}
        )
        deltas = [
            DomainGeometryDelta(
                domain=AtlasDomain.SPATIAL,
                source_score=0.8,
                target_score=0.6,
                delta=0.2,
            ),
            DomainGeometryDelta(
                domain=AtlasDomain.MORAL,
                source_score=0.7,
                target_score=0.9,
                delta=0.2,
            ),
        ]

        audit = PreMergeGeometryAudit(
            source_profile=source,
            target_profile=target,
            domain_deltas=deltas,
            strength_ratio_variance=0.02,
        )

        d = audit.to_dict()

        assert "sourceProfile" in d
        assert "targetProfile" in d
        assert d["sourceProfile"]["modelPath"] == "/source"
        assert d["targetProfile"]["modelPath"] == "/target"
        assert "domainDeltas" in d
        assert len(d["domainDeltas"]) == 2
        assert d["domainDeltas"][0]["domain"] == "spatial"
        assert d["domainDeltas"][0]["sourceScore"] == 0.8
        assert d["strengthRatioVariance"] == 0.02


# =============================================================================
# PostMergeGeometryValidation Tests
# =============================================================================


class TestPostMergeGeometryValidation:
    """Tests for PostMergeGeometryValidation dataclass."""

    def _create_sample_profile(
        self, path: str, scores: dict[AtlasDomain, float]
    ) -> ModelGeometryProfile:
        """Create a sample profile."""
        domain_scores = {}
        for domain, manifold_score in scores.items():
            domain_scores[domain] = DomainGeometryScore(
                domain=domain,
                manifold_score=manifold_score,
                axis_orthogonality=0.8,
                gradient_consistency=0.7,
                anchors_probed=50,
                layer_analyzed=12,
            )

        return ModelGeometryProfile(
            model_path=path,
            layer=12,
            domain_scores=domain_scores,
            computed_at=datetime.now(),
            total_anchors=100,
        )

    def test_basic_creation(self) -> None:
        """Should create validation with all fields."""
        source = self._create_sample_profile(
            "/source", {AtlasDomain.SPATIAL: 0.8}
        )
        merged = self._create_sample_profile(
            "/merged", {AtlasDomain.SPATIAL: 0.72}
        )

        validation = PostMergeGeometryValidation(
            source_profile=source,
            merged_profile=merged,
            preservation_by_domain={AtlasDomain.SPATIAL: 0.9},  # 0.72/0.8 = 0.9
            overall_preservation=0.9,
        )

        assert validation.source_profile.model_path == "/source"
        assert validation.merged_profile.model_path == "/merged"
        assert validation.preservation_by_domain[AtlasDomain.SPATIAL] == 0.9
        assert validation.overall_preservation == 0.9

    def test_to_dict(self) -> None:
        """to_dict should serialize validation correctly."""
        source = self._create_sample_profile(
            "/source", {AtlasDomain.SPATIAL: 0.8, AtlasDomain.MORAL: 0.7}
        )
        merged = self._create_sample_profile(
            "/merged", {AtlasDomain.SPATIAL: 0.72, AtlasDomain.MORAL: 0.77}
        )

        validation = PostMergeGeometryValidation(
            source_profile=source,
            merged_profile=merged,
            preservation_by_domain={
                AtlasDomain.SPATIAL: 0.9,  # 0.72/0.8
                AtlasDomain.MORAL: 1.1,    # 0.77/0.7
            },
            overall_preservation=1.0,
        )

        d = validation.to_dict()

        assert "sourceProfile" in d
        assert "mergedProfile" in d
        assert d["sourceProfile"]["modelPath"] == "/source"
        assert d["mergedProfile"]["modelPath"] == "/merged"
        assert "preservationByDomain" in d
        assert d["preservationByDomain"]["spatial"] == 0.9
        assert d["preservationByDomain"]["moral"] == 1.1
        assert d["overallPreservation"] == 1.0

    def test_preservation_can_exceed_one(self) -> None:
        """Preservation can be > 1.0 if merge enhances geometry."""
        source = self._create_sample_profile(
            "/source", {AtlasDomain.TEMPORAL: 0.5}
        )
        merged = self._create_sample_profile(
            "/merged", {AtlasDomain.TEMPORAL: 0.7}
        )

        validation = PostMergeGeometryValidation(
            source_profile=source,
            merged_profile=merged,
            preservation_by_domain={AtlasDomain.TEMPORAL: 1.4},  # 0.7/0.5
            overall_preservation=1.4,
        )

        assert validation.preservation_by_domain[AtlasDomain.TEMPORAL] == 1.4


# =============================================================================
# DomainGeometryWaypointService Tests
# =============================================================================


class TestDomainGeometryWaypointService:
    """Tests for DomainGeometryWaypointService class."""

    class MockModelLoader:
        """Mock model loader for testing."""

        def load_model_for_training(self, path: str):
            """Return a mock model and tokenizer."""
            raise NotImplementedError("Test should not call real model loading")

    def test_creation(self, any_backend: "Backend") -> None:
        """Should create service with backend and model loader."""
        loader = self.MockModelLoader()
        service = DomainGeometryWaypointService(
            backend=any_backend,
            model_loader=loader,  # type: ignore
        )

        assert service._backend is any_backend
        assert service._model_loader is loader

    def test_compute_domain_strength_profile_equal_scores(
        self, any_backend: "Backend"
    ) -> None:
        """Strength ratio should be 0.5 when source and target scores are equal."""
        loader = self.MockModelLoader()
        service = DomainGeometryWaypointService(
            backend=any_backend,
            model_loader=loader,  # type: ignore
        )

        deltas = [
            DomainGeometryDelta(
                domain=AtlasDomain.SPATIAL,
                source_score=0.5,
                target_score=0.5,
                delta=0.0,
            )
        ]

        audit = PreMergeGeometryAudit(
            source_profile=ModelGeometryProfile(
                model_path="/source",
                layer=12,
                domain_scores={},
                computed_at=datetime.now(),
                total_anchors=0,
            ),
            target_profile=ModelGeometryProfile(
                model_path="/target",
                layer=12,
                domain_scores={},
                computed_at=datetime.now(),
                total_anchors=0,
            ),
            domain_deltas=deltas,
            strength_ratio_variance=0.0,
        )

        ratios = service.compute_domain_strength_profile(audit)

        expected_ratio = deltas[0].target_score / (
            deltas[0].source_score + deltas[0].target_score
        )
        assert abs(ratios[AtlasDomain.SPATIAL] - expected_ratio) <= _eps()

    def test_compute_domain_strength_profile_target_stronger(
        self, any_backend: "Backend"
    ) -> None:
        """Strength ratio should favor target when target score is higher."""
        loader = self.MockModelLoader()
        service = DomainGeometryWaypointService(
            backend=any_backend,
            model_loader=loader,  # type: ignore
        )

        deltas = [
            DomainGeometryDelta(
                domain=AtlasDomain.MORAL,
                source_score=0.2,
                target_score=0.8,
                delta=0.6,
            )
        ]

        audit = PreMergeGeometryAudit(
            source_profile=ModelGeometryProfile(
                model_path="/source",
                layer=12,
                domain_scores={},
                computed_at=datetime.now(),
                total_anchors=0,
            ),
            target_profile=ModelGeometryProfile(
                model_path="/target",
                layer=12,
                domain_scores={},
                computed_at=datetime.now(),
                total_anchors=0,
            ),
            domain_deltas=deltas,
            strength_ratio_variance=0.0,
        )

        ratios = service.compute_domain_strength_profile(audit)

        expected_ratio = deltas[0].target_score / (
            deltas[0].source_score + deltas[0].target_score
        )
        assert abs(ratios[AtlasDomain.MORAL] - expected_ratio) <= _eps()

    def test_compute_domain_strength_profile_source_stronger(
        self, any_backend: "Backend"
    ) -> None:
        """Strength ratio should favor source when source score is higher."""
        loader = self.MockModelLoader()
        service = DomainGeometryWaypointService(
            backend=any_backend,
            model_loader=loader,  # type: ignore
        )

        deltas = [
            DomainGeometryDelta(
                domain=AtlasDomain.TEMPORAL,
                source_score=0.9,
                target_score=0.1,
                delta=0.8,
            )
        ]

        audit = PreMergeGeometryAudit(
            source_profile=ModelGeometryProfile(
                model_path="/source",
                layer=12,
                domain_scores={},
                computed_at=datetime.now(),
                total_anchors=0,
            ),
            target_profile=ModelGeometryProfile(
                model_path="/target",
                layer=12,
                domain_scores={},
                computed_at=datetime.now(),
                total_anchors=0,
            ),
            domain_deltas=deltas,
            strength_ratio_variance=0.0,
        )

        ratios = service.compute_domain_strength_profile(audit)

        expected_ratio = deltas[0].target_score / (
            deltas[0].source_score + deltas[0].target_score
        )
        assert abs(ratios[AtlasDomain.TEMPORAL] - expected_ratio) <= _eps()

    def test_compute_domain_strength_profile_zero_scores(
        self, any_backend: "Backend"
    ) -> None:
        """Zero scores should yield no ratio entry."""
        loader = self.MockModelLoader()
        service = DomainGeometryWaypointService(
            backend=any_backend,
            model_loader=loader,  # type: ignore
        )

        deltas = [
            DomainGeometryDelta(
                domain=AtlasDomain.RELATIONAL,
                source_score=0.0,
                target_score=0.0,
                delta=0.0,
            )
        ]

        audit = PreMergeGeometryAudit(
            source_profile=ModelGeometryProfile(
                model_path="/source",
                layer=12,
                domain_scores={},
                computed_at=datetime.now(),
                total_anchors=0,
            ),
            target_profile=ModelGeometryProfile(
                model_path="/target",
                layer=12,
                domain_scores={},
                computed_at=datetime.now(),
                total_anchors=0,
            ),
            domain_deltas=deltas,
            strength_ratio_variance=0.0,
        )

        ratios = service.compute_domain_strength_profile(audit)

        assert AtlasDomain.RELATIONAL not in ratios

    def test_compute_domain_strength_profile_multiple_domains(
        self, any_backend: "Backend"
    ) -> None:
        """Should compute ratios for all domains in audit."""
        loader = self.MockModelLoader()
        service = DomainGeometryWaypointService(
            backend=any_backend,
            model_loader=loader,  # type: ignore
        )

        deltas = [
            DomainGeometryDelta(
                domain=AtlasDomain.SPATIAL,
                source_score=0.6,
                target_score=0.4,
                delta=0.2,
            ),
            DomainGeometryDelta(
                domain=AtlasDomain.RELATIONAL,
                source_score=0.3,
                target_score=0.7,
                delta=0.4,
            ),
            DomainGeometryDelta(
                domain=AtlasDomain.TEMPORAL,
                source_score=0.5,
                target_score=0.5,
                delta=0.0,
            ),
            DomainGeometryDelta(
                domain=AtlasDomain.MORAL,
                source_score=0.8,
                target_score=0.2,
                delta=0.6,
            ),
        ]

        audit = PreMergeGeometryAudit(
            source_profile=ModelGeometryProfile(
                model_path="/source",
                layer=12,
                domain_scores={},
                computed_at=datetime.now(),
                total_anchors=0,
            ),
            target_profile=ModelGeometryProfile(
                model_path="/target",
                layer=12,
                domain_scores={},
                computed_at=datetime.now(),
                total_anchors=0,
            ),
            domain_deltas=deltas,
            strength_ratio_variance=0.0,
        )

        ratios = service.compute_domain_strength_profile(audit)

        assert len(ratios) == 4
        expected = {
            delta.domain: delta.target_score / (delta.source_score + delta.target_score)
            for delta in deltas
        }
        for domain, expected_ratio in expected.items():
            assert abs(ratios[domain] - expected_ratio) <= _eps()

    def test_compute_domain_strength_profile_empty_deltas(
        self, any_backend: "Backend"
    ) -> None:
        """Should return empty dict when no deltas."""
        loader = self.MockModelLoader()
        service = DomainGeometryWaypointService(
            backend=any_backend,
            model_loader=loader,  # type: ignore
        )

        audit = PreMergeGeometryAudit(
            source_profile=ModelGeometryProfile(
                model_path="/source",
                layer=12,
                domain_scores={},
                computed_at=datetime.now(),
                total_anchors=0,
            ),
            target_profile=ModelGeometryProfile(
                model_path="/target",
                layer=12,
                domain_scores={},
                computed_at=datetime.now(),
                total_anchors=0,
            ),
            domain_deltas=[],
            strength_ratio_variance=0.0,
        )

        ratios = service.compute_domain_strength_profile(audit)

        assert ratios == {}


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the waypoints module."""

    def _create_profile(
        self, path: str, scores: dict[AtlasDomain, float]
    ) -> ModelGeometryProfile:
        """Create a sample profile."""
        domain_scores = {}
        for domain, manifold_score in scores.items():
            domain_scores[domain] = DomainGeometryScore(
                domain=domain,
                manifold_score=manifold_score,
                axis_orthogonality=0.8,
                gradient_consistency=0.7,
                anchors_probed=50,
                layer_analyzed=12,
            )

        return ModelGeometryProfile(
            model_path=path,
            layer=12,
            domain_scores=domain_scores,
            computed_at=datetime.now(),
            total_anchors=200,
        )

    def test_full_audit_to_dict_workflow(self) -> None:
        """Full workflow from profiles to serialized audit."""
        source = self._create_profile(
            "/source",
            {
                AtlasDomain.SPATIAL: 0.85,
                AtlasDomain.RELATIONAL: 0.70,
                AtlasDomain.TEMPORAL: 0.75,
                AtlasDomain.MORAL: 0.80,
            },
        )
        target = self._create_profile(
            "/target",
            {
                AtlasDomain.SPATIAL: 0.65,
                AtlasDomain.RELATIONAL: 0.90,
                AtlasDomain.TEMPORAL: 0.75,
                AtlasDomain.MORAL: 0.60,
            },
        )

        # Create deltas - only for domains in both profiles
        deltas = []
        for domain in source.domain_scores:
            s = source.domain_scores[domain].manifold_score
            t = target.domain_scores[domain].manifold_score
            deltas.append(
                DomainGeometryDelta(
                    domain=domain,
                    source_score=s,
                    target_score=t,
                    delta=abs(s - t),
                )
            )

        # Create audit
        audit = PreMergeGeometryAudit(
            source_profile=source,
            target_profile=target,
            domain_deltas=deltas,
            strength_ratio_variance=0.015,
        )

        # Serialize
        d = audit.to_dict()

        # Verify structure
        assert d["sourceProfile"]["modelPath"] == "/source"
        assert d["targetProfile"]["modelPath"] == "/target"
        assert len(d["domainDeltas"]) == 4
        assert d["strengthRatioVariance"] == 0.015

    def test_validation_workflow(self) -> None:
        """Full workflow for post-merge validation."""
        source = self._create_profile(
            "/source",
            {
                AtlasDomain.SPATIAL: 0.80,
                AtlasDomain.MORAL: 0.70,
            },
        )
        merged = self._create_profile(
            "/merged",
            {
                AtlasDomain.SPATIAL: 0.72,  # 90% preserved
                AtlasDomain.MORAL: 0.77,   # 110% preserved (enhanced)
            },
        )

        preservation = {
            AtlasDomain.SPATIAL: 0.72 / 0.80,
            AtlasDomain.MORAL: 0.77 / 0.70,
        }
        overall = sum(preservation.values()) / len(preservation)

        validation = PostMergeGeometryValidation(
            source_profile=source,
            merged_profile=merged,
            preservation_by_domain=preservation,
            overall_preservation=overall,
        )

        d = validation.to_dict()

        assert abs(d["overallPreservation"] - overall) <= _eps()
        assert abs(d["preservationByDomain"]["spatial"] - preservation[AtlasDomain.SPATIAL]) <= _eps()
        assert abs(d["preservationByDomain"]["moral"] - preservation[AtlasDomain.MORAL]) <= _eps()
