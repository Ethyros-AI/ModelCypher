# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for Representation Consistency analysis module.

Representation consistency measures how stable a model's representations are
for semantically related inputs vs contradictory/unrelated inputs.

For LoRA merging: Validate that LoRA insertion doesn't break semantic
consistency - related inputs should still cluster together, and
contradictory inputs should remain separated.

For safety: Detect when a model confuses related vs unrelated concepts,
which could indicate representation collapse or adversarial drift.
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.core.domain.geometry.representation_consistency import (
    ConsistencyResult,
    RepresentationConsistencyAnalyzer,
)


@pytest.fixture
def backend():
    """Get default backend."""
    return get_default_backend()


@pytest.fixture
def analyzer(backend):
    """Create analyzer instance."""
    return RepresentationConsistencyAnalyzer(backend)


class TestCosineDistance:
    """Tests for cosine_distance method."""

    def test_identical_vectors_zero_distance(self, analyzer, backend):
        """Identical vectors should have distance = 0."""
        vec = backend.array([1.0, 2.0, 3.0, 4.0])

        distance = analyzer.cosine_distance(vec, vec)

        eps = float(machine_epsilon(backend, vec))
        assert abs(distance) < eps * 100, \
            f"Distance between identical vectors should be ~0, got {distance}"

    def test_opposite_vectors_max_distance(self, analyzer, backend):
        """Opposite vectors should have distance = 2 (1 - (-1) = 2)."""
        vec1 = backend.array([1.0, 2.0, 3.0])
        vec2 = backend.array([-1.0, -2.0, -3.0])

        distance = analyzer.cosine_distance(vec1, vec2)

        assert abs(distance - 2.0) < 0.01, \
            f"Distance between opposite vectors should be 2.0, got {distance}"

    def test_orthogonal_vectors_unit_distance(self, analyzer, backend):
        """Orthogonal vectors should have distance = 1."""
        vec1 = backend.array([1.0, 0.0, 0.0])
        vec2 = backend.array([0.0, 1.0, 0.0])

        distance = analyzer.cosine_distance(vec1, vec2)

        assert abs(distance - 1.0) < 0.01, \
            f"Distance between orthogonal vectors should be 1.0, got {distance}"

    def test_distance_in_valid_range(self, analyzer, backend):
        """Cosine distance should always be in [0, 2]."""
        backend.random_seed(42)
        for _ in range(10):
            vec1 = backend.random_normal((64,))
            vec2 = backend.random_normal((64,))

            distance = analyzer.cosine_distance(vec1, vec2)

            assert 0.0 <= distance <= 2.0, \
                f"Cosine distance should be in [0, 2], got {distance}"

    def test_zero_vector_returns_orthogonal(self, analyzer, backend):
        """Zero vector should be treated as orthogonal (distance = 1)."""
        vec = backend.array([1.0, 2.0, 3.0])
        zero = backend.zeros((3,))

        distance = analyzer.cosine_distance(vec, zero)

        assert abs(distance - 1.0) < 0.01, \
            f"Zero vector should be orthogonal (distance=1), got {distance}"


class TestConsistencyCompute:
    """Tests for compute method."""

    def test_identical_related_perfect_consistency(self, analyzer, backend):
        """When related inputs are identical to original, consistency = 1.0."""
        original = backend.array([1.0, 2.0, 3.0, 4.0])
        related = [original, original, original]

        result = analyzer.compute(original, related)

        assert abs(result.implication_consistency - 1.0) < 0.01, \
            f"Identical related should give consistency=1.0, got {result.implication_consistency}"

    def test_orthogonal_related_low_consistency(self, analyzer, backend):
        """When related inputs are orthogonal, consistency should be low."""
        original = backend.array([1.0, 0.0, 0.0, 0.0])
        related = [
            backend.array([0.0, 1.0, 0.0, 0.0]),
            backend.array([0.0, 0.0, 1.0, 0.0]),
            backend.array([0.0, 0.0, 0.0, 1.0]),
        ]

        result = analyzer.compute(original, related)

        # Orthogonal = distance 1 = consistency 0
        assert result.implication_consistency < 0.1, \
            f"Orthogonal related should give low consistency, got {result.implication_consistency}"

    def test_contradictory_increases_score_when_distant(self, analyzer, backend):
        """Distant contradictions should contribute to higher combined score."""
        original = backend.array([1.0, 0.0, 0.0, 0.0])
        related = [
            backend.array([0.9, 0.1, 0.0, 0.0]),  # Close to original
            backend.array([0.8, 0.2, 0.0, 0.0]),
        ]
        contradictory = [
            backend.array([-1.0, 0.0, 0.0, 0.0]),  # Opposite
            backend.array([0.0, -1.0, 0.0, 0.0]),
        ]

        result = analyzer.compute(original, related, contradictory)

        assert result.contradiction_distance > 1.0, \
            f"Opposite contradictions should have high distance, got {result.contradiction_distance}"

    def test_separation_score_meaningful(self, analyzer, backend):
        """Separation should be high when related close and contradictory far."""
        original = backend.array([1.0, 0.0, 0.0, 0.0])

        # Related: very close to original
        related = [
            backend.array([0.99, 0.01, 0.0, 0.0]),
            backend.array([0.98, 0.02, 0.0, 0.0]),
        ]

        # Contradictory: opposite direction
        contradictory = [
            backend.array([-1.0, 0.0, 0.0, 0.0]),
            backend.array([-0.9, -0.1, 0.0, 0.0]),
        ]

        result = analyzer.compute(original, related, contradictory)

        assert result.separation_score > 0.5, \
            f"Good separation should give high score, got {result.separation_score}"

    def test_no_contradictory_still_computes(self, analyzer, backend):
        """Should work without contradictory inputs."""
        original = backend.array([1.0, 2.0, 3.0])
        related = [
            backend.array([1.1, 2.1, 3.1]),
            backend.array([0.9, 1.9, 2.9]),
        ]

        result = analyzer.compute(original, related)

        assert result.n_contradictions == 0
        assert result.implication_consistency > 0
        assert result.separation_score == 0.0  # No contradictions = no separation score

    def test_result_has_all_fields(self, analyzer, backend):
        """Result should have all expected fields populated."""
        original = backend.array([1.0, 2.0, 3.0, 4.0])
        related = [backend.array([1.1, 2.1, 3.1, 4.1])]
        contradictory = [backend.array([-1.0, -2.0, -3.0, -4.0])]

        result = analyzer.compute(original, related, contradictory)

        assert isinstance(result, ConsistencyResult)
        assert 0 <= result.implication_consistency <= 1
        assert 0 <= result.contradiction_distance <= 2
        assert result.consistency_score >= 0
        assert result.separation_score >= 0
        assert result.n_implications == 1
        assert result.n_contradictions == 1
        assert len(result.representation_distances) == 2


class TestHighDimensional:
    """Tests with high-dimensional vectors (more realistic for LLMs)."""

    def test_random_vectors_moderate_consistency(self, analyzer, backend):
        """Random vectors should have moderate consistency (around 0)."""
        backend.random_seed(42)
        original = backend.random_normal((512,))
        related = [backend.random_normal((512,)) for _ in range(5)]

        result = analyzer.compute(original, related)

        # Random vectors: cosine similarity ~ 0, so distance ~ 1, consistency ~ 0
        assert 0 <= result.implication_consistency <= 0.5, \
            f"Random vectors should have low consistency, got {result.implication_consistency}"

    def test_similar_vectors_high_consistency(self, analyzer, backend):
        """Vectors with small perturbations should have high consistency."""
        backend.random_seed(42)
        original = backend.random_normal((512,))

        # Add small noise to create "related" vectors
        backend.random_seed(43)
        noise_scale = 0.05
        related = [
            original + noise_scale * backend.random_normal((512,))
            for _ in range(5)
        ]

        result = analyzer.compute(original, related)

        assert result.implication_consistency > 0.9, \
            f"Similar vectors should have high consistency, got {result.implication_consistency}"

    def test_scaled_vectors_perfect_consistency(self, analyzer, backend):
        """Scaled vectors (same direction) should have perfect consistency."""
        backend.random_seed(42)
        original = backend.random_normal((256,))

        # Scale by different positive factors
        related = [
            original * 0.5,
            original * 2.0,
            original * 0.1,
        ]

        result = analyzer.compute(original, related)

        eps = float(machine_epsilon(backend, original))
        assert abs(result.implication_consistency - 1.0) < eps * 1000, \
            f"Scaled vectors should have perfect consistency, got {result.implication_consistency}"
