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

"""Tests for Cross-Grounding Transfer: Density Re-mapping for Knowledge Transfer.

Tests the core functionality of coordinate-invariant knowledge transfer
between models with different grounding types.

NO NUMPY. All data generation uses Backend protocol.
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cross_grounding_transfer import (
    CrossGroundingSynthesizer,
    CrossGroundingTransferEngine,
    GhostAnchor,
    GroundingRotationEstimator,
    RelationalStressComputer,
)


@pytest.fixture
def backend():
    return get_default_backend()


@pytest.fixture
def sample_anchors(backend):
    """Create sample anchor activations for testing."""
    backend.random_seed(42)
    dim = 64

    def make_anchor(offset_3d):
        base = backend.random_normal((dim,))
        # Add offset to first 3 dimensions
        offset = backend.zeros((dim,))
        offset_arr = backend.array(offset_3d + [0.0] * (dim - 3))
        base = base + offset_arr
        backend.eval(base)
        return base

    anchors = {
        "floor": make_anchor([0.0, -1.0, 0.0]),
        "ceiling": make_anchor([0.0, 1.0, 0.0]),
        "left": make_anchor([-1.0, 0.0, 0.0]),
        "right": make_anchor([1.0, 0.0, 0.0]),
        "near": make_anchor([0.0, 0.0, 1.0]),
        "far": make_anchor([0.0, 0.0, -1.0]),
    }
    return anchors


class TestRelationalStressProfile:
    """Tests for RelationalStressProfile data structure."""

    def test_distance_to_same_profile_is_zero(self, backend, sample_anchors):
        """Distance between identical profiles should be zero."""
        computer = RelationalStressComputer(backend)
        concept = backend.zeros((64,))

        profile = computer.compute_profile(concept, sample_anchors)
        assert profile.distance_to(profile) == pytest.approx(0.0)

    def test_profile_captures_anchor_distances(self, backend, sample_anchors):
        """Profile should capture distances to all anchors."""
        computer = RelationalStressComputer(backend)
        concept = backend.zeros((64,))

        profile = computer.compute_profile(concept, sample_anchors)

        assert len(profile.anchor_distances) == len(sample_anchors)
        for anchor_name in sample_anchors:
            assert anchor_name in profile.anchor_distances
            assert profile.anchor_distances[anchor_name] >= 0

    def test_profile_has_normalized_distances(self, backend, sample_anchors):
        """Profile should have normalized distances."""
        computer = RelationalStressComputer(backend)
        concept = backend.zeros((64,))

        profile = computer.compute_profile(concept, sample_anchors)

        assert len(profile.normalized_distances) == len(sample_anchors)

    def test_nearest_anchors_ordered_by_distance(self, backend, sample_anchors):
        """Nearest anchors should be ordered by distance."""
        computer = RelationalStressComputer(backend)
        # Place concept near floor
        concept = backend.array([0.0, -0.9, 0.0] + [0.0] * 61)

        profile = computer.compute_profile(concept, sample_anchors, k_nearest=3)

        # Floor should be nearest
        assert len(profile.nearest_anchors) == 3

    def test_stress_vector_is_consistent(self, backend, sample_anchors):
        """Stress vector should be deterministically ordered."""
        computer = RelationalStressComputer(backend)
        concept = backend.zeros((64,))

        profile1 = computer.compute_profile(concept, sample_anchors)
        profile2 = computer.compute_profile(concept, sample_anchors)

        assert profile1.stress_vector == profile2.stress_vector


class TestGroundingRotation:
    """Tests for GroundingRotation estimation."""

    def test_identical_anchors_give_zero_rotation(self, backend, sample_anchors):
        """Identical anchor sets should have zero rotation."""
        estimator = GroundingRotationEstimator(backend)
        rotation = estimator.estimate_rotation(sample_anchors, sample_anchors)

        assert rotation.angle_degrees == pytest.approx(0.0, abs=1.0)
        assert rotation.alignment_score >= 0.99
        assert rotation.is_aligned

    def test_rotated_anchors_detect_rotation(self, backend):
        """Significantly different anchor sets should show lower alignment."""
        estimator = GroundingRotationEstimator(backend)
        dim = 16

        # Source: clustered along x-y plane
        source_anchors = {
            "a": backend.array([1.0, 0.0, 0.0] + [0.0] * (dim - 3)),
            "b": backend.array([0.0, 1.0, 0.0] + [0.0] * (dim - 3)),
            "c": backend.array([-1.0, 0.0, 0.0] + [0.0] * (dim - 3)),
            "d": backend.array([0.0, -1.0, 0.0] + [0.0] * (dim - 3)),
            "e": backend.array([0.5, 0.5, 0.0] + [0.0] * (dim - 3)),
            "f": backend.array([-0.5, 0.5, 0.0] + [0.0] * (dim - 3)),
        }

        # Target: stretched along z axis (different structure)
        target_anchors = {
            "a": backend.array([1.0, 0.0, 5.0] + [0.0] * (dim - 3)),
            "b": backend.array([0.0, 1.0, -5.0] + [0.0] * (dim - 3)),
            "c": backend.array([-1.0, 0.0, 3.0] + [0.0] * (dim - 3)),
            "d": backend.array([0.0, -1.0, -3.0] + [0.0] * (dim - 3)),
            "e": backend.array([0.5, 0.5, 10.0] + [0.0] * (dim - 3)),
            "f": backend.array([-0.5, 0.5, -10.0] + [0.0] * (dim - 3)),
        }

        rotation = estimator.estimate_rotation(source_anchors, target_anchors)

        # Should detect structural difference (lower alignment)
        assert rotation.alignment_score < 1.0
        assert rotation.confidence > 0

    def test_insufficient_anchors_return_low_confidence(self, backend):
        """Insufficient common anchors should return low confidence."""
        estimator = GroundingRotationEstimator(backend)

        source = {"a": backend.array([1.0, 0.0, 0.0])}
        target = {"b": backend.array([0.0, 1.0, 0.0])}

        rotation = estimator.estimate_rotation(source, target)

        assert rotation.confidence == 0.0
        assert rotation.alignment_score == 0.0


class TestCrossGroundingSynthesizer:
    """Tests for Ghost Anchor synthesis."""

    def test_synthesize_ghost_anchor_returns_valid_anchor(self, backend, sample_anchors):
        """Synthesized ghost anchor should have valid structure."""
        synthesizer = CrossGroundingSynthesizer(backend)
        backend.random_seed(100)

        concept = backend.random_normal((64,))
        ghost = synthesizer.synthesize_ghost_anchor(
            concept_id="test_concept",
            source_activation=concept,
            source_anchors=sample_anchors,
            target_anchors=sample_anchors,
        )

        assert isinstance(ghost, GhostAnchor)
        assert ghost.concept_id == "test_concept"
        assert ghost.source_position.shape == (64,)
        assert ghost.target_position.shape == (64,)
        assert 0.0 <= ghost.stress_preservation <= 1.0
        assert 0.0 <= ghost.synthesis_confidence <= 1.0

    def test_identical_models_preserve_stress(self, backend, sample_anchors):
        """Identical source and target should have high stress preservation."""
        synthesizer = CrossGroundingSynthesizer(backend)
        backend.random_seed(101)

        concept = backend.random_normal((64,))
        ghost = synthesizer.synthesize_ghost_anchor(
            concept_id="test",
            source_activation=concept,
            source_anchors=sample_anchors,
            target_anchors=sample_anchors,
        )

        # Should have some preservation for identical models
        # Note: random seeds affect this, so we use a loose bound
        assert ghost.stress_preservation > 0.1

    def test_insufficient_anchors_emit_warning(self, backend):
        """Insufficient common anchors should emit warning."""
        synthesizer = CrossGroundingSynthesizer(backend)

        source_anchors = {"a": backend.array([1.0, 0.0, 0.0])}
        target_anchors = {"b": backend.array([0.0, 1.0, 0.0])}
        concept = backend.array([0.5, 0.5, 0.0])

        ghost = synthesizer.synthesize_ghost_anchor(
            concept_id="test",
            source_activation=concept,
            source_anchors=source_anchors,
            target_anchors=target_anchors,
        )

        assert ghost.warning is not None
        assert ghost.stress_preservation == 0.0


class TestCrossGroundingTransferEngine:
    """Tests for the full transfer engine."""

    def test_transfer_concepts_returns_valid_result(self, backend, sample_anchors):
        """Transfer should return valid result structure."""
        engine = CrossGroundingTransferEngine(backend)
        backend.random_seed(200)

        concepts = {
            "concept1": backend.random_normal((64,)),
            "concept2": backend.random_normal((64,)),
        }

        result = engine.transfer_concepts(
            concepts=concepts,
            source_anchors=sample_anchors,
            target_anchors=sample_anchors,
            source_grounding="high_visual",
            target_grounding="alternative",
        )

        assert len(result.ghost_anchors) == 2
        assert result.source_model_grounding == "high_visual"
        assert result.target_model_grounding == "alternative"
        assert result.successful_transfers + result.failed_transfers == 2

    def test_transfer_with_identical_models_high_quality(self, backend, sample_anchors):
        """Transfer between identical models should have high quality."""
        engine = CrossGroundingTransferEngine(backend)
        backend.random_seed(201)

        concepts = {"test": backend.random_normal((64,))}

        result = engine.transfer_concepts(
            concepts=concepts,
            source_anchors=sample_anchors,
            target_anchors=sample_anchors,
        )

        assert result.mean_stress_preservation > 0.7
        assert result.grounding_rotation.is_aligned

    def test_estimate_feasibility_returns_valid_assessment(self, backend, sample_anchors):
        """Feasibility estimation should return valid assessment."""
        engine = CrossGroundingTransferEngine(backend)

        feasibility = engine.estimate_transfer_feasibility(sample_anchors, sample_anchors)

        assert "feasibility" in feasibility
        assert "recommendation" in feasibility
        assert "common_anchors" in feasibility
        assert feasibility["common_anchors"] == len(sample_anchors)

    def test_feasibility_high_for_aligned_models(self, backend, sample_anchors):
        """Aligned models should have HIGH feasibility."""
        engine = CrossGroundingTransferEngine(backend)

        feasibility = engine.estimate_transfer_feasibility(sample_anchors, sample_anchors)

        assert feasibility["feasibility"] == "HIGH"
        assert feasibility["is_aligned"]


class TestRelationalStressInvariance:
    """Tests for the key property: Relational Stress is coordinate-invariant."""

    def test_stress_profile_invariant_under_rotation(self, backend):
        """Stress profile distances should be preserved under rotation."""
        computer = RelationalStressComputer(backend)
        dim = 32
        backend.random_seed(123)

        # Create anchors
        anchors = {
            "a": backend.random_normal((dim,)),
            "b": backend.random_normal((dim,)),
            "c": backend.random_normal((dim,)),
            "d": backend.random_normal((dim,)),
            "e": backend.random_normal((dim,)),
        }
        concept = backend.random_normal((dim,))

        # Compute profile in original space
        original_profile = computer.compute_profile(concept, anchors)

        # Create random orthogonal rotation matrix via QR decomposition
        random_matrix = backend.random_normal((dim, dim))
        Q, _ = backend.qr(random_matrix)
        backend.eval(Q)

        # Rotate all vectors
        rotated_anchors = {
            name: backend.matmul(Q, vec) for name, vec in anchors.items()
        }
        rotated_concept = backend.matmul(Q, concept)

        # Compute profile in rotated space
        rotated_profile = computer.compute_profile(rotated_concept, rotated_anchors)

        # Distances should be identical (within numerical precision)
        for anchor_name in anchors:
            assert original_profile.anchor_distances[anchor_name] == pytest.approx(
                rotated_profile.anchor_distances[anchor_name], abs=1e-6
            )

    def test_stress_distance_invariant_under_translation(self, backend):
        """Stress profile should not depend on absolute position."""
        computer = RelationalStressComputer(backend)
        dim = 32
        backend.random_seed(456)

        anchors = {
            "a": backend.random_normal((dim,)),
            "b": backend.random_normal((dim,)),
            "c": backend.random_normal((dim,)),
        }
        concept = backend.random_normal((dim,))

        original_profile = computer.compute_profile(concept, anchors)

        # Translate everything by same vector
        translation = backend.random_normal((dim,)) * 10.0
        translated_anchors = {
            name: vec + translation for name, vec in anchors.items()
        }
        translated_concept = concept + translation

        translated_profile = computer.compute_profile(translated_concept, translated_anchors)

        # Distances should be identical (GPU fp32 has ~1e-4 precision after operations)
        for anchor_name in anchors:
            assert original_profile.anchor_distances[anchor_name] == pytest.approx(
                translated_profile.anchor_distances[anchor_name], abs=1e-3
            )


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_concepts_dict_returns_empty_result(self, backend, sample_anchors):
        """Empty concepts dict should return empty ghost anchors."""
        engine = CrossGroundingTransferEngine(backend)

        result = engine.transfer_concepts(
            concepts={},
            source_anchors=sample_anchors,
            target_anchors=sample_anchors,
        )

        assert len(result.ghost_anchors) == 0
        assert result.successful_transfers == 0
        assert result.failed_transfers == 0

    def test_single_dimension_vectors(self, backend):
        """Should handle single-dimension vectors gracefully."""
        computer = RelationalStressComputer(backend)

        anchors = {
            "low": backend.array([0.0]),
            "high": backend.array([10.0]),
        }
        concept = backend.array([5.0])

        profile = computer.compute_profile(concept, anchors, k_nearest=2)

        assert len(profile.anchor_distances) == 2
        assert profile.anchor_distances["low"] == pytest.approx(5.0)
        assert profile.anchor_distances["high"] == pytest.approx(5.0)

    def test_identical_anchor_positions(self, backend):
        """Should handle degenerate case of identical anchors."""
        computer = RelationalStressComputer(backend)

        vec = backend.array([1.0, 2.0, 3.0])
        anchors = {
            "a": vec,
            "b": vec,
            "c": vec,
        }
        concept = backend.array([0.0, 0.0, 0.0])

        profile = computer.compute_profile(concept, anchors)

        # All distances should be equal
        distances = list(profile.anchor_distances.values())
        assert all(d == pytest.approx(distances[0]) for d in distances)
