# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.

"""Tests for Hierarchical Optimal Transport layer matcher.

Tests the SOTA layer matching approach that uses soft coupling
instead of rigid 1-to-1 assignments.
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.hot_layer_matcher import (
    HOTLayerMatcher,
    coupling_to_assignment,
    hot_layer_matching,
)


@pytest.fixture
def backend():
    return get_default_backend()


class TestHOTLayerMatcher:
    """Tests for HOTLayerMatcher."""

    def test_identical_layers_diagonal_coupling(self, backend):
        """Identical layers should produce near-diagonal coupling."""
        b = backend
        n_layers = 4
        n_samples = 30
        d = 16

        # Generate distinct layer activations
        primes = [7, 13, 23, 37]
        source_layers = {}
        for layer in range(n_layers):
            prime = primes[layer]
            acts = b.array([
                [float((i * prime + j * 2) % (prime * 5)) for j in range(d)]
                for i in range(n_samples)
            ])
            b.eval(acts)
            source_layers[layer] = acts

        # Target is identical to source
        target_layers = {k: v for k, v in source_layers.items()}

        # Match
        matcher = HOTLayerMatcher(b)
        result = matcher.match(source_layers, target_layers)

        # Verify we got a result
        assert result.layer_coupling is not None
        assert result.alignment_score >= 0.0

        # Check coupling shape
        assert result.layer_coupling.shape == (n_layers, n_layers)

        # Convert to hard assignment and check diagonal
        assignment = coupling_to_assignment(
            result.layer_coupling,
            result.source_layers,
            result.target_layers,
            b,
        )

        # For identical layers, assignment should be identity
        identity = {i: i for i in range(n_layers)}
        assert assignment == identity, f"Expected identity, got {assignment}"

    def test_different_depths_mass_distribution(self, backend):
        """Different layer counts should distribute mass across layers."""
        b = backend
        n_samples = 30
        d = 16

        # Source has 4 layers
        primes = [7, 13, 23, 37]
        source_layers = {}
        for layer in range(4):
            prime = primes[layer]
            acts = b.array([
                [float((i * prime + j * 2) % (prime * 5)) for j in range(d)]
                for i in range(n_samples)
            ])
            b.eval(acts)
            source_layers[layer] = acts

        # Target has 2 layers (subset of source: layers 0 and 3)
        target_layers = {
            0: source_layers[0],
            1: source_layers[3],
        }

        # Match
        result = hot_layer_matching(source_layers, target_layers, b)

        # Check shape: [4, 2]
        assert result.layer_coupling.shape == (4, 2)

        # Verify marginals approximately sum correctly
        # Row sums should be close to 1/4 each (uniform source marginal)
        row_sums = b.sum(result.layer_coupling, axis=1)
        b.eval(row_sums)
        row_sums_list = b.tolist(row_sums)
        for rs in row_sums_list:
            assert abs(rs - 0.25) < 0.1, f"Row sum {rs} far from uniform 0.25"

        # Column sums should be close to 1/2 each (uniform target marginal)
        col_sums = b.sum(result.layer_coupling, axis=0)
        b.eval(col_sums)
        col_sums_list = b.tolist(col_sums)
        for cs in col_sums_list:
            assert abs(cs - 0.5) < 0.1, f"Col sum {cs} far from uniform 0.5"

    def test_marginal_constraints_satisfied(self, backend):
        """Verify that coupling satisfies marginal constraints."""
        b = backend
        n_layers = 3
        n_samples = 30
        d = 16

        # Generate random-ish activations
        source_layers = {}
        target_layers = {}
        for layer in range(n_layers):
            src = b.array([
                [float((i * 7 + j * 3 + layer * 11) % 29) for j in range(d)]
                for i in range(n_samples)
            ])
            tgt = b.array([
                [float((i * 11 + j * 5 + layer * 13) % 31) for j in range(d)]
                for i in range(n_samples)
            ])
            b.eval(src, tgt)
            source_layers[layer] = src
            target_layers[layer] = tgt

        result = hot_layer_matching(source_layers, target_layers, b)

        # Check row marginals (should sum to 1/n_layers each)
        row_sums = b.sum(result.layer_coupling, axis=1)
        b.eval(row_sums)
        expected_row_sum = 1.0 / n_layers
        for i, rs in enumerate(b.tolist(row_sums)):
            assert abs(rs - expected_row_sum) < 0.05, (
                f"Row {i} sum {rs} != expected {expected_row_sum}"
            )

        # Check column marginals (should sum to 1/n_layers each)
        col_sums = b.sum(result.layer_coupling, axis=0)
        b.eval(col_sums)
        expected_col_sum = 1.0 / n_layers
        for j, cs in enumerate(b.tolist(col_sums)):
            assert abs(cs - expected_col_sum) < 0.05, (
                f"Col {j} sum {cs} != expected {expected_col_sum}"
            )

        # Total mass should be 1
        total = b.sum(result.layer_coupling)
        b.eval(total)
        assert abs(float(b.to_scalar(total)) - 1.0) < 0.01

    def test_coupling_values_in_valid_range(self, backend):
        """All coupling values should be non-negative."""
        b = backend
        n_layers = 3
        n_samples = 30
        d = 16

        source_layers = {}
        target_layers = {}
        for layer in range(n_layers):
            acts = b.array([
                [float(i * d + j + layer * 100) for j in range(d)]
                for i in range(n_samples)
            ])
            b.eval(acts)
            source_layers[layer] = acts
            target_layers[layer] = acts

        result = hot_layer_matching(source_layers, target_layers, b)

        # All values should be >= 0
        min_val = b.min(result.layer_coupling)
        b.eval(min_val)
        assert float(b.to_scalar(min_val)) >= -1e-6, "Coupling has negative values"

    def test_empty_layers_handled(self, backend):
        """Empty activations should return empty result."""
        b = backend

        result = hot_layer_matching({}, {}, b)

        assert result.alignment_score == 0.0
        assert result.source_layers == []
        assert result.target_layers == []

    def test_pairwise_costs_computed(self, backend):
        """Pairwise costs should be computed for all layer pairs."""
        b = backend
        n_src_layers = 3
        n_tgt_layers = 2
        n_samples = 30
        d = 16

        source_layers = {}
        for layer in range(n_src_layers):
            acts = b.array([
                [float(i * d + j + layer * 50) for j in range(d)]
                for i in range(n_samples)
            ])
            b.eval(acts)
            source_layers[layer] = acts

        target_layers = {}
        for layer in range(n_tgt_layers):
            acts = b.array([
                [float(i * d + j + layer * 50) for j in range(d)]
                for i in range(n_samples)
            ])
            b.eval(acts)
            target_layers[layer] = acts

        result = hot_layer_matching(source_layers, target_layers, b)

        # Should have n_src_layers * n_tgt_layers pairwise costs
        expected_pairs = n_src_layers * n_tgt_layers
        assert len(result.pairwise_costs) == expected_pairs

        # All costs should be non-negative
        for (src, tgt), cost in result.pairwise_costs.items():
            assert cost >= 0.0, f"Negative cost for ({src}, {tgt}): {cost}"

    def test_neuron_transport_optional(self, backend):
        """Neuron transport should only be stored when requested."""
        b = backend
        n_layers = 2
        n_samples = 20
        d = 8

        source_layers = {}
        target_layers = {}
        for layer in range(n_layers):
            acts = b.array([
                [float(i * d + j + layer * 10) for j in range(d)]
                for i in range(n_samples)
            ])
            b.eval(acts)
            source_layers[layer] = acts
            target_layers[layer] = acts

        # Without neuron transport
        matcher = HOTLayerMatcher(b)
        result_no_transport = matcher.match(
            source_layers, target_layers, store_neuron_transport=False
        )
        assert result_no_transport.neuron_transport is None

        # With neuron transport
        result_with_transport = matcher.match(
            source_layers, target_layers, store_neuron_transport=True
        )
        assert result_with_transport.neuron_transport is not None
        assert len(result_with_transport.neuron_transport) == n_layers * n_layers


class TestCouplingToAssignment:
    """Tests for coupling_to_assignment helper."""

    def test_extracts_dominant_mapping(self, backend):
        """Should extract the dominant source for each target."""
        b = backend

        # Create a coupling that clearly prefers diagonal
        coupling = b.array([
            [0.4, 0.05, 0.05],  # Source 0 -> Target 0
            [0.05, 0.4, 0.05],  # Source 1 -> Target 1
            [0.05, 0.05, 0.4],  # Source 2 -> Target 2
        ])
        b.eval(coupling)

        source_layers = [0, 1, 2]
        target_layers = [0, 1, 2]

        assignment = coupling_to_assignment(coupling, source_layers, target_layers, b)

        # Should be identity mapping
        assert assignment == {0: 0, 1: 1, 2: 2}

    def test_handles_non_diagonal_coupling(self, backend):
        """Should handle cases where coupling is not diagonal."""
        b = backend

        # Coupling where target 1 maps to source 0
        coupling = b.array([
            [0.3, 0.35],  # Source 0 preferred for both
            [0.2, 0.15],  # Source 1 less preferred
        ])
        b.eval(coupling)

        source_layers = [0, 1]
        target_layers = [0, 1]

        assignment = coupling_to_assignment(coupling, source_layers, target_layers, b)

        # Target 0 -> Source 0 (0.3 > 0.2)
        # Target 1 -> Source 0 (0.35 > 0.15)
        assert assignment[0] == 0
        assert assignment[1] == 0


class TestComparisonWithHungarian:
    """Tests comparing HOT with Hungarian matcher behavior."""

    def test_similar_results_on_clear_structure(self, backend):
        """HOT should find similar correspondences as Hungarian on clear cases."""
        b = backend
        n_layers = 3
        n_samples = 40
        d = 16

        # Create layers with very distinct patterns
        primes = [7, 13, 23]
        source_layers = {}
        for layer in range(n_layers):
            prime = primes[layer]
            acts = b.array([
                [float((i * prime + j * 2) % (prime * 7)) for j in range(d)]
                for i in range(n_samples)
            ])
            b.eval(acts)
            source_layers[layer] = acts

        # Target is same structure, slightly noisy
        target_layers = {}
        for layer in range(n_layers):
            noise = b.array([
                [float((i * 3 + j * 7) % 11) * 0.01 for j in range(d)]
                for i in range(n_samples)
            ])
            b.eval(noise)
            target_layers[layer] = source_layers[layer] + noise
            b.eval(target_layers[layer])

        # HOT matching
        result = hot_layer_matching(source_layers, target_layers, b)

        # Convert to hard assignment
        assignment = coupling_to_assignment(
            result.layer_coupling,
            result.source_layers,
            result.target_layers,
            b,
        )

        # Should find identity mapping for this clear case
        identity = {i: i for i in range(n_layers)}
        assert assignment == identity, f"Expected identity, got {assignment}"

    def test_soft_coupling_captures_ambiguity(self, backend):
        """HOT should spread mass when layers are ambiguous."""
        b = backend
        n_samples = 30
        d = 16

        # Create source with 3 similar layers (same pattern, slight offset)
        base_pattern = b.array([
            [float(i * d + j) for j in range(d)]
            for i in range(n_samples)
        ])
        b.eval(base_pattern)

        source_layers = {
            0: base_pattern,
            1: base_pattern + 0.01,
            2: base_pattern + 0.02,
        }
        for v in source_layers.values():
            b.eval(v)

        # Target has just one layer matching the pattern
        target_layers = {0: base_pattern}
        b.eval(target_layers[0])

        result = hot_layer_matching(source_layers, target_layers, b)

        # Since all source layers are similar, mass should spread
        # (not concentrate on just one source layer)
        coupling_list = b.tolist(result.layer_coupling)

        # All source layers should contribute some mass
        for src_idx in range(3):
            mass_to_target = coupling_list[src_idx][0]
            assert mass_to_target > 0.1, (
                f"Source {src_idx} should contribute mass, got {mass_to_target}"
            )
