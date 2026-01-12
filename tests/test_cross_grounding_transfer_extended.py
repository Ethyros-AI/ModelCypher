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

"""Extended tests for cross-grounding transfer (Ghost Anchor synthesis).

Tests critical APIs:
- RelationalStressComputer.compute_profile(): Coordinate-invariant fingerprints
- GroundingRotationEstimator.estimate_rotation(): Coordinate system alignment
- CrossGroundingSynthesizer.synthesize_ghost_anchor(): Ghost Anchor synthesis
- CrossGroundingTransferEngine.transfer_concepts(): Full transfer pipeline
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cross_grounding_transfer import (
    CrossGroundingSynthesizer,
    CrossGroundingTransferEngine,
    GroundingRotationEstimator,
    RelationalStressComputer,
)
from modelcypher.core.domain.geometry.numerical_stability import all_finite


@pytest.fixture
def backend():
    return get_default_backend()


@pytest.fixture
def sample_anchors(backend):
    """Create sample anchor activations."""
    anchor_names = ["north", "south", "east", "west", "center"]
    anchors = {}
    for i, name in enumerate(anchor_names):
        arr = backend.random_normal((32,))
        backend.eval(arr)
        anchors[name] = arr
    return anchors


class TestRelationalStressComputer:
    """Tests for RelationalStressComputer."""

    def test_compute_profile_basic(self, backend, sample_anchors):
        """Basic profile computation should work."""
        computer = RelationalStressComputer(backend)
        concept = backend.random_normal((32,))
        backend.eval(concept)

        profile = computer.compute_profile(concept, sample_anchors)

        assert profile is not None
        assert len(profile.anchor_distances) == len(sample_anchors)
        assert len(profile.normalized_distances) == len(sample_anchors)
        assert profile.local_density >= 0
        assert profile.activation_magnitude >= 0

    def test_profile_distances_non_negative(self, backend, sample_anchors):
        """All distances should be non-negative."""
        computer = RelationalStressComputer(backend)
        concept = backend.random_normal((32,))
        backend.eval(concept)

        profile = computer.compute_profile(concept, sample_anchors)

        for distance in profile.anchor_distances.values():
            assert distance >= 0

    def test_profile_nearest_anchors(self, backend, sample_anchors):
        """Nearest anchors should be correctly ordered."""
        computer = RelationalStressComputer(backend)
        concept = backend.random_normal((32,))
        backend.eval(concept)

        profile = computer.compute_profile(concept, sample_anchors, k_nearest=3)

        assert len(profile.nearest_anchors) == 3
        # Nearest anchors should be from the anchor set
        for anchor in profile.nearest_anchors:
            assert anchor in sample_anchors

    def test_profile_stress_vector_length(self, backend, sample_anchors):
        """Stress vector should have length equal to number of anchors."""
        computer = RelationalStressComputer(backend)
        concept = backend.random_normal((32,))
        backend.eval(concept)

        profile = computer.compute_profile(concept, sample_anchors)

        assert len(profile.stress_vector) == len(sample_anchors)

    def test_profile_distance_to_self_small(self, backend, sample_anchors):
        """Distance from profile to itself should be near zero."""
        computer = RelationalStressComputer(backend)
        concept = backend.random_normal((32,))
        backend.eval(concept)

        profile = computer.compute_profile(concept, sample_anchors)

        # Distance to itself should be very small
        self_distance = profile.distance_to(profile)
        assert self_distance < 1e-5


class TestGroundingRotationEstimator:
    """Tests for GroundingRotationEstimator."""

    def test_estimate_rotation_identical_anchors(self, backend, sample_anchors):
        """Identical anchors should produce aligned rotation."""
        estimator = GroundingRotationEstimator(backend)

        rotation = estimator.estimate_rotation(sample_anchors, sample_anchors)

        assert rotation.aligned is True
        assert rotation.distance_correlation > 0.99
        assert rotation.angle_degrees < 5.0  # Nearly aligned

    def test_estimate_rotation_insufficient_common(self, backend):
        """Too few common anchors should return low-confidence rotation."""
        estimator = GroundingRotationEstimator(backend)

        source_anchors = {"a": backend.random_normal((32,))}
        target_anchors = {"b": backend.random_normal((32,))}
        backend.eval(source_anchors["a"], target_anchors["b"])

        rotation = estimator.estimate_rotation(source_anchors, target_anchors)

        assert rotation.confidence == 0.0
        assert rotation.aligned is False

    def test_estimate_rotation_confidence_bounded(self, backend, sample_anchors):
        """Rotation confidence should be in [0, 1]."""
        estimator = GroundingRotationEstimator(backend)

        # Create different target anchors
        target_anchors = {
            name: backend.random_normal((32,)) for name in sample_anchors
        }
        for arr in target_anchors.values():
            backend.eval(arr)

        rotation = estimator.estimate_rotation(sample_anchors, target_anchors)

        assert 0.0 <= rotation.confidence <= 1.0

    def test_estimate_rotation_angle_bounded(self, backend, sample_anchors):
        """Rotation angle should be in [0, 180]."""
        estimator = GroundingRotationEstimator(backend)

        target_anchors = {
            name: backend.random_normal((32,)) for name in sample_anchors
        }
        for arr in target_anchors.values():
            backend.eval(arr)

        rotation = estimator.estimate_rotation(sample_anchors, target_anchors)

        assert 0.0 <= rotation.angle_degrees <= 180.0


class TestCrossGroundingSynthesizer:
    """Tests for CrossGroundingSynthesizer."""

    def test_synthesize_ghost_anchor_basic(self, backend, sample_anchors):
        """Basic ghost anchor synthesis should work."""
        synthesizer = CrossGroundingSynthesizer(backend)
        concept = backend.random_normal((32,))
        backend.eval(concept)

        ghost = synthesizer.synthesize_ghost_anchor(
            concept_id="test_concept",
            source_activation=concept,
            source_anchors=sample_anchors,
            target_anchors=sample_anchors,
        )

        assert ghost is not None
        assert ghost.concept_id == "test_concept"
        assert ghost.target_position is not None
        assert 0.0 <= ghost.stress_preservation <= 1.0

    def test_synthesize_ghost_anchor_insufficient_common(self, backend):
        """Insufficient common anchors should return degenerate ghost."""
        synthesizer = CrossGroundingSynthesizer(backend)

        source_anchors = {
            "a": backend.random_normal((32,)),
            "b": backend.random_normal((32,)),
        }
        target_anchors = {
            "c": backend.random_normal((32,)),
            "d": backend.random_normal((32,)),
        }
        concept = backend.random_normal((32,))
        for arr in [*source_anchors.values(), *target_anchors.values(), concept]:
            backend.eval(arr)

        ghost = synthesizer.synthesize_ghost_anchor(
            concept_id="test",
            source_activation=concept,
            source_anchors=source_anchors,
            target_anchors=target_anchors,
        )

        # Should return degenerate ghost with zero preservation
        assert ghost.stress_preservation == 0.0
        assert ghost.synthesis_confidence == 0.0

    def test_synthesize_ghost_anchor_same_model(self, backend, sample_anchors):
        """Same source/target should produce valid preservation."""
        synthesizer = CrossGroundingSynthesizer(backend)
        concept = backend.random_normal((32,))
        backend.eval(concept)

        ghost = synthesizer.synthesize_ghost_anchor(
            concept_id="test_concept",
            source_activation=concept,
            source_anchors=sample_anchors,
            target_anchors=sample_anchors,
        )

        # Same model should have valid preservation (bounded [0, 1])
        # Note: even with same anchors, random data may have low preservation
        # due to the difficulty of multilateration with random anchor positions
        assert 0.0 <= ghost.stress_preservation <= 1.0
        assert ghost.common_anchor_count == len(sample_anchors)


class TestCrossGroundingTransferEngine:
    """Tests for CrossGroundingTransferEngine."""

    def test_transfer_concepts_basic(self, backend, sample_anchors):
        """Basic concept transfer should work."""
        engine = CrossGroundingTransferEngine(backend)

        concepts = {
            "concept_1": backend.random_normal((32,)),
            "concept_2": backend.random_normal((32,)),
        }
        for arr in concepts.values():
            backend.eval(arr)

        result = engine.transfer_concepts(
            concepts=concepts,
            source_anchors=sample_anchors,
            target_anchors=sample_anchors,
        )

        assert result is not None
        assert len(result.ghost_anchors) == 2
        assert 0.0 <= result.mean_stress_preservation <= 1.0

    def test_transfer_concepts_empty(self, backend, sample_anchors):
        """Empty concepts should return empty result."""
        engine = CrossGroundingTransferEngine(backend)

        result = engine.transfer_concepts(
            concepts={},
            source_anchors=sample_anchors,
            target_anchors=sample_anchors,
        )

        assert len(result.ghost_anchors) == 0
        assert result.mean_stress_preservation == 0.0

    def test_estimate_transfer_feasibility(self, backend, sample_anchors):
        """Feasibility estimation should work."""
        engine = CrossGroundingTransferEngine(backend)

        feasibility = engine.estimate_transfer_feasibility(
            source_anchors=sample_anchors,
            target_anchors=sample_anchors,
        )

        assert "common_anchors" in feasibility
        assert "grounding_rotation_degrees" in feasibility
        assert "confidence" in feasibility
        assert feasibility["common_anchors"] == len(sample_anchors)


class TestCrossGroundingMathematicalProperties:
    """Hypothesis-based tests for mathematical invariants."""

    @given(
        n_anchors=st.integers(min_value=5, max_value=10),
        d=st.integers(min_value=16, max_value=32),
    )
    @settings(max_examples=5, deadline=None)
    def test_stress_vector_sorted(self, n_anchors, d):
        """Stress vector should be sorted by anchor name."""
        backend = get_default_backend()
        computer = RelationalStressComputer(backend)

        anchors = {f"anchor_{i:02d}": backend.random_normal((d,)) for i in range(n_anchors)}
        for arr in anchors.values():
            backend.eval(arr)

        concept = backend.random_normal((d,))
        backend.eval(concept)

        profile = computer.compute_profile(concept, anchors)

        # Stress vector should match sorted anchor distances
        expected_order = sorted(anchors.keys())
        expected_vector = tuple(profile.anchor_distances[k] for k in expected_order)

        assert profile.stress_vector == expected_vector

    @given(
        n_anchors=st.integers(min_value=5, max_value=10),
        d=st.integers(min_value=16, max_value=32),
    )
    @settings(max_examples=5, deadline=None)
    def test_ghost_anchor_position_finite(self, n_anchors, d):
        """Ghost anchor position should always be finite."""
        backend = get_default_backend()
        synthesizer = CrossGroundingSynthesizer(backend)

        anchors = {f"anchor_{i}": backend.random_normal((d,)) for i in range(n_anchors)}
        for arr in anchors.values():
            backend.eval(arr)

        concept = backend.random_normal((d,))
        backend.eval(concept)

        ghost = synthesizer.synthesize_ghost_anchor(
            concept_id="test",
            source_activation=concept,
            source_anchors=anchors,
            target_anchors=anchors,
        )

        assert all_finite(ghost.target_position, backend)

    @given(
        n_anchors=st.integers(min_value=5, max_value=10),
        d=st.integers(min_value=16, max_value=32),
    )
    @settings(max_examples=5, deadline=None)
    def test_preservation_bounded(self, n_anchors, d):
        """Stress preservation should be in [0, 1]."""
        backend = get_default_backend()
        synthesizer = CrossGroundingSynthesizer(backend)

        anchors = {f"anchor_{i}": backend.random_normal((d,)) for i in range(n_anchors)}
        for arr in anchors.values():
            backend.eval(arr)

        concept = backend.random_normal((d,))
        backend.eval(concept)

        ghost = synthesizer.synthesize_ghost_anchor(
            concept_id="test",
            source_activation=concept,
            source_anchors=anchors,
            target_anchors=anchors,
        )

        assert 0.0 <= ghost.stress_preservation <= 1.0
