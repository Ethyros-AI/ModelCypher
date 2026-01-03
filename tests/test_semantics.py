# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for the semantics module.

Tests ConceptVectorSpace and ActivationGraphProjector functionality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.core.domain.semantics.vector_space import (
    ConceptNode,
    ConceptVectorSpace,
)
from modelcypher.core.domain.semantics.graph import ActivationGraphProjector

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


@pytest.fixture
def backend() -> "Backend":
    """Get the default backend."""
    return get_default_backend()


def _eps(backend: "Backend", *values: float) -> float:
    return machine_epsilon(backend, backend.array(list(values) or [1.0]))


# =============================================================================
# ConceptVectorSpace Tests
# =============================================================================


class TestConceptVectorSpace:
    """Tests for ConceptVectorSpace functionality."""

    def test_creation_with_default_dimension(self, backend: "Backend") -> None:
        """ConceptVectorSpace should create with default dimension."""
        space = ConceptVectorSpace(backend=backend)
        assert space.dimension == 4096
        assert len(space.concepts) == 0

    def test_creation_with_custom_dimension(self, backend: "Backend") -> None:
        """ConceptVectorSpace should accept custom dimension."""
        space = ConceptVectorSpace(dimension=128, backend=backend)
        assert space.dimension == 128

    def test_add_concept_stores_correctly(self, backend: "Backend") -> None:
        """Adding a concept should store it correctly."""
        space = ConceptVectorSpace(dimension=64, backend=backend)

        vector = backend.random_normal((64,))
        backend.eval(vector)

        space.add_concept("test_concept", vector, {"category": "test"})

        assert "test_concept" in space.concepts
        assert space.concepts["test_concept"].id == "test_concept"
        assert space.concepts["test_concept"].metadata == {"category": "test"}

    def test_add_concept_normalizes_vector(self, backend: "Backend") -> None:
        """Vectors should be normalized on insertion."""
        space = ConceptVectorSpace(dimension=64, backend=backend)

        # Create vector with known norm
        vector = backend.ones((64,)) * 10.0  # Norm will be sqrt(64) * 10 = 80
        backend.eval(vector)

        space.add_concept("test", vector)

        # Stored vector should be normalized (norm ≈ 1.0)
        stored = space.concepts["test"].vector
        norm_arr = backend.norm(stored)
        backend.eval(norm_arr)
        norm = float(backend.to_scalar(norm_arr))
        # Use 1e-4 tolerance for normalization precision (64-dim accumulates error)
        assert abs(norm - 1.0) < 1e-4

    def test_add_concept_dimension_mismatch_raises(self, backend: "Backend") -> None:
        """Adding concept with wrong dimension should raise."""
        space = ConceptVectorSpace(dimension=64, backend=backend)

        wrong_dim_vector = backend.random_normal((128,))
        backend.eval(wrong_dim_vector)

        with pytest.raises(ValueError, match="dimension mismatch"):
            space.add_concept("bad", wrong_dim_vector)

    def test_find_nearest_neighbors_empty_space(self, backend: "Backend") -> None:
        """Empty space should return empty neighbor list."""
        space = ConceptVectorSpace(dimension=64, backend=backend)

        query = backend.random_normal((64,))
        backend.eval(query)

        neighbors = space.find_nearest_neighbors(query, k=5)
        assert neighbors == []

    def test_find_nearest_neighbors_returns_correct_order(self, backend: "Backend") -> None:
        """Neighbors should be ordered by similarity (descending)."""
        backend.random_seed(42)
        space = ConceptVectorSpace(dimension=64, backend=backend)

        # Add three concepts
        v1 = backend.random_normal((64,))
        v2 = backend.random_normal((64,))
        v3 = backend.random_normal((64,))
        backend.eval(v1, v2, v3)

        space.add_concept("a", v1)
        space.add_concept("b", v2)
        space.add_concept("c", v3)

        # Query with v1 - should be most similar to "a"
        neighbors = space.find_nearest_neighbors(v1, k=3)

        assert len(neighbors) == 3
        # First neighbor should be "a" (identical after normalization)
        assert neighbors[0][0] == "a"
        # Similarity to self should be high (close to 1.0)
        assert abs(neighbors[0][1] - 1.0) <= _eps(backend, neighbors[0][1])

    def test_find_nearest_neighbors_respects_k(self, backend: "Backend") -> None:
        """Should return at most k neighbors."""
        backend.random_seed(42)
        space = ConceptVectorSpace(dimension=64, backend=backend)

        # Add 10 concepts
        for i in range(10):
            v = backend.random_normal((64,))
            backend.eval(v)
            space.add_concept(f"concept_{i}", v)

        query = backend.random_normal((64,))
        backend.eval(query)

        neighbors = space.find_nearest_neighbors(query, k=3)
        assert len(neighbors) == 3

    def test_arithmetics_positive_sum(self, backend: "Backend") -> None:
        """Vector arithmetic should correctly sum positive concepts."""
        space = ConceptVectorSpace(dimension=64, backend=backend)

        v1 = backend.ones((64,))
        v2 = backend.ones((64,)) * 2
        backend.eval(v1, v2)

        space.add_concept("a", v1)
        space.add_concept("b", v2)

        # Sum of normalized vectors
        result = space.arithmetics(positive=["a", "b"], negative=[])
        backend.eval(result)

        # Result should be non-zero
        norm_arr = backend.norm(result)
        backend.eval(norm_arr)
        norm = float(backend.to_scalar(norm_arr))
        assert norm > 0

    def test_arithmetics_with_negatives(self, backend: "Backend") -> None:
        """Vector arithmetic should correctly subtract negative concepts."""
        space = ConceptVectorSpace(dimension=64, backend=backend)

        v = backend.ones((64,))
        backend.eval(v)

        space.add_concept("a", v)

        # a - a should be approximately zero
        result = space.arithmetics(positive=["a"], negative=["a"])
        backend.eval(result)

        norm_arr = backend.norm(result)
        backend.eval(norm_arr)
        norm = float(backend.to_scalar(norm_arr))
        assert abs(norm - 0.0) < _eps(backend, norm, 0.0)

    def test_arithmetics_missing_concepts_ignored(self, backend: "Backend") -> None:
        """Missing concepts should be silently ignored."""
        space = ConceptVectorSpace(dimension=64, backend=backend)

        v = backend.ones((64,))
        backend.eval(v)
        space.add_concept("a", v)

        # "missing" doesn't exist - should be ignored
        result = space.arithmetics(positive=["a", "missing"], negative=["nonexistent"])
        backend.eval(result)

        # Should still compute with available concepts
        norm_arr = backend.norm(result)
        backend.eval(norm_arr)
        norm = float(backend.to_scalar(norm_arr))
        assert norm > 0


# =============================================================================
# ActivationGraphProjector Tests
# =============================================================================


class TestActivationGraphProjector:
    """Tests for ActivationGraphProjector functionality."""

    def test_empty_graph_creation(self) -> None:
        """Empty graph should have no nodes or edges."""
        projector = ActivationGraphProjector()
        assert len(projector.adjacency) == 0
        assert projector.get_density() == 0.0

    def test_record_single_concept_no_edges(self) -> None:
        """Single concept should not create edges."""
        projector = ActivationGraphProjector()
        projector.record_co_occurrence(["concept_a"])

        # Single concept creates no edges
        assert len(projector.adjacency) == 0

    def test_record_pair_creates_bidirectional_edge(self) -> None:
        """Pair of concepts should create bidirectional edge."""
        projector = ActivationGraphProjector()
        projector.record_co_occurrence(["a", "b"])

        assert "a" in projector.adjacency
        assert "b" in projector.adjacency
        assert projector.adjacency["a"]["b"] == 1.0
        assert projector.adjacency["b"]["a"] == 1.0

    def test_record_multiple_times_accumulates_weight(self) -> None:
        """Repeated co-occurrences should accumulate edge weight."""
        projector = ActivationGraphProjector()

        projector.record_co_occurrence(["a", "b"])
        projector.record_co_occurrence(["a", "b"])
        projector.record_co_occurrence(["a", "b"])

        assert projector.adjacency["a"]["b"] == 3.0
        assert projector.adjacency["b"]["a"] == 3.0

    def test_record_clique_creates_all_pairs(self) -> None:
        """Recording n concepts should create n*(n-1)/2 edges."""
        projector = ActivationGraphProjector()
        projector.record_co_occurrence(["a", "b", "c"])

        # Should have edges: a-b, a-c, b-c
        assert projector.adjacency["a"]["b"] == 1.0
        assert projector.adjacency["a"]["c"] == 1.0
        assert projector.adjacency["b"]["c"] == 1.0

    def test_get_strongest_connections_empty_concept(self) -> None:
        """Unknown concept should return empty connections."""
        projector = ActivationGraphProjector()
        connections = projector.get_strongest_connections("unknown", k=5)
        assert connections == []

    def test_get_strongest_connections_sorted_order(self) -> None:
        """Connections should be sorted by weight (descending)."""
        projector = ActivationGraphProjector()

        # Create edges with different weights
        projector.record_co_occurrence(["a", "b"])
        projector.record_co_occurrence(["a", "b"])  # weight 2
        projector.record_co_occurrence(["a", "c"])  # weight 1
        projector.record_co_occurrence(["a", "d"])
        projector.record_co_occurrence(["a", "d"])
        projector.record_co_occurrence(["a", "d"])  # weight 3

        connections = projector.get_strongest_connections("a", k=3)

        assert len(connections) == 3
        # Should be ordered: d (3), b (2), c (1)
        assert connections[0] == ("d", 3.0)
        assert connections[1] == ("b", 2.0)
        assert connections[2] == ("c", 1.0)

    def test_get_strongest_connections_respects_k(self) -> None:
        """Should return at most k connections."""
        projector = ActivationGraphProjector()

        for i in range(10):
            projector.record_co_occurrence(["center", f"node_{i}"])

        connections = projector.get_strongest_connections("center", k=3)
        assert len(connections) == 3

    def test_density_complete_graph(self) -> None:
        """Complete graph should have density 1.0."""
        projector = ActivationGraphProjector()

        # Complete graph with 3 nodes
        projector.record_co_occurrence(["a", "b", "c"])

        # All 3 edges present: density = 2*3 / (3*2) = 1.0
        density = projector.get_density()
        backend = get_default_backend()
        assert abs(density - 1.0) < _eps(backend, density, 1.0)

    def test_density_partial_graph(self) -> None:
        """Partial graph should have density < 1.0."""
        projector = ActivationGraphProjector()

        # Only 2 edges among 3 nodes
        projector.record_co_occurrence(["a", "b"])
        projector.record_co_occurrence(["b", "c"])

        # 2 edges, 3 nodes: density = 2*2 / (3*2) = 2/3 ≈ 0.667
        density = projector.get_density()
        backend = get_default_backend()
        expected = 2.0 / 3.0
        assert abs(density - expected) < _eps(backend, density, expected)

    def test_density_single_node(self) -> None:
        """Single node graph should have density 0."""
        projector = ActivationGraphProjector()
        projector.record_co_occurrence(["alone"])

        # No edges created, no nodes in adjacency
        density = projector.get_density()
        backend = get_default_backend()
        assert abs(density - 0.0) <= _eps(backend, density, 0.0)
