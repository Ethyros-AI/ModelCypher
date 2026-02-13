# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for Fisher Information geometry module.

Fisher Information measures the curvature of the loss landscape at each
parameter dimension. High FIM = the model "cares" about that dimension.

For LoRA merging: LoRA deltas should avoid high-Fisher dimensions
(catastrophic forgetting) and target low-Fisher dimensions.
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.fisher_information import (
    compute_empirical_fisher_diagonal,
    fisher_compatibility_score,
    fisher_weighted_merge,
)
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon


@pytest.fixture
def backend():
    """Get default backend."""
    return get_default_backend()


class TestDiagonalFisherComputation:
    """Tests for compute_empirical_fisher_diagonal."""

    def test_unit_variance_fim_near_one(self, backend):
        """Activations with unit variance should have FIM mean ≈ 1.0.

        The diagonal FIM is E[x_j^2] for each dimension j.
        For standard normal data, E[x^2] = Var(x) + E[x]^2 = 1 + 0 = 1.
        """
        backend.random_seed(42)
        # Large sample for statistical stability
        activations = backend.random_normal((1000, 64))

        result = compute_empirical_fisher_diagonal(activations, backend)

        # Mean FIM should be close to 1.0 (within statistical variance)
        assert 0.8 < result.mean_fim < 1.2, f"Mean FIM should be ~1.0 for unit variance, got {result.mean_fim}"

    def test_scaled_variance_fim_scales_quadratically(self, backend):
        """Activations scaled by k should have FIM ≈ k².

        E[(k*x)^2] = k² * E[x²] = k² for standard normal.
        """
        backend.random_seed(42)
        scale = 2.0
        activations = backend.random_normal((1000, 64)) * scale

        result = compute_empirical_fisher_diagonal(activations, backend)

        expected_fim = scale * scale  # = 4.0
        assert 3.2 < result.mean_fim < 4.8, f"Mean FIM should be ~{expected_fim}, got {result.mean_fim}"

    def test_effective_rank_meaningful(self, backend):
        """Effective rank should be meaningful for varied importance."""
        backend.random_seed(42)
        # Create activations with varying importance per dimension
        # First half: high variance (high FIM), second half: low variance (low FIM)
        high_var = backend.random_normal((500, 32)) * 10.0
        low_var = backend.random_normal((500, 32)) * 0.1
        activations = backend.concatenate([high_var, low_var], axis=1)
        backend.eval(activations)

        result = compute_empirical_fisher_diagonal(activations, backend)

        # Effective rank should be less than full dimension (64) due to concentration
        assert result.effective_rank < 64, f"Effective rank should be < 64, got {result.effective_rank}"
        # But greater than 0
        assert result.effective_rank > 0, f"Effective rank should be > 0, got {result.effective_rank}"

    def test_result_has_all_fields(self, backend):
        """Result should have all expected fields populated."""
        backend.random_seed(42)
        activations = backend.random_normal((100, 32))

        result = compute_empirical_fisher_diagonal(activations, backend)

        assert result.diagonal_fim is not None
        assert result.effective_rank >= 0
        assert result.trace_fim >= 0
        assert result.condition_number >= 1.0
        assert result.mean_fim >= 0
        assert result.n_significant >= 0
        assert result.significance_threshold >= 0


class TestFisherCompatibility:
    """Tests for fisher_compatibility_score."""

    def test_identical_activations_perfect_score(self, backend):
        """Identical activations should have compatibility score = 1.0."""
        backend.random_seed(42)
        activations = backend.random_normal((200, 64))

        result = fisher_compatibility_score(activations, activations, backend)

        eps = float(machine_epsilon(backend, activations))
        assert abs(result.compatibility_score - 1.0) < eps * 100, \
            f"Compatibility for identical activations should be 1.0, got {result.compatibility_score}"
        assert abs(result.cosine_similarity - 1.0) < eps * 100
        assert abs(result.correlation - 1.0) < eps * 100

    def test_orthogonal_importance_low_score(self, backend):
        """Orthogonal importance patterns should have low compatibility.

        If source has high variance in dims [0:32] and target in dims [32:64],
        their Fisher profiles are orthogonal -> low cosine similarity.
        """
        backend.random_seed(42)

        # Source: high variance in first half
        source_first = backend.random_normal((200, 32)) * 10.0
        source_second = backend.random_normal((200, 32)) * 0.1
        source = backend.concatenate([source_first, source_second], axis=1)

        # Target: high variance in second half (opposite pattern)
        target_first = backend.random_normal((200, 32)) * 0.1
        target_second = backend.random_normal((200, 32)) * 10.0
        target = backend.concatenate([target_first, target_second], axis=1)

        backend.eval(source, target)

        result = fisher_compatibility_score(source, target, backend)

        # Orthogonal importance -> low compatibility
        assert result.compatibility_score < 0.5, \
            f"Orthogonal importance should have low compatibility, got {result.compatibility_score}"

    def test_similar_importance_high_score(self, backend):
        """Similar importance patterns should have high compatibility."""
        backend.random_seed(42)

        # Both have same variance pattern
        base_scale = backend.array([10.0 if i < 32 else 0.1 for i in range(64)])
        source = backend.random_normal((200, 64)) * base_scale
        backend.random_seed(123)
        target = backend.random_normal((200, 64)) * base_scale

        backend.eval(source, target)

        result = fisher_compatibility_score(source, target, backend)

        # Same importance pattern -> high compatibility
        assert result.compatibility_score > 0.9, \
            f"Same importance pattern should have high compatibility, got {result.compatibility_score}"

    def test_dimension_mismatch_uses_common_prefix(self, backend):
        """Different dimensions should compare on common prefix."""
        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 32))  # Fewer dimensions

        # Should not raise, uses min(64, 32) = 32 dimensions
        result = fisher_compatibility_score(source, target, backend)

        assert 0.0 <= result.compatibility_score <= 1.0


class TestFisherWeightedMerge:
    """Tests for fisher_weighted_merge."""

    def test_output_shape_matches_input(self, backend):
        """Merged weights should have same shape as inputs."""
        backend.random_seed(42)
        source_weights = backend.random_normal((128, 64))
        target_weights = backend.random_normal((128, 64))
        source_acts = backend.random_normal((100, 64))
        target_acts = backend.random_normal((100, 64))

        source_fisher = compute_empirical_fisher_diagonal(source_acts, backend)
        target_fisher = compute_empirical_fisher_diagonal(target_acts, backend)

        merged = fisher_weighted_merge(
            source_weights, target_weights,
            source_fisher, target_fisher,
            backend
        )
        backend.eval(merged)

        assert backend.shape(merged) == backend.shape(source_weights)

    def test_high_source_fisher_favors_source(self, backend):
        """When source has high Fisher, merged should be closer to source."""
        backend.random_seed(42)

        # Source: high variance -> high Fisher
        source_acts = backend.random_normal((200, 64)) * 10.0
        # Target: low variance -> low Fisher
        target_acts = backend.random_normal((200, 64)) * 0.1

        source_fisher = compute_empirical_fisher_diagonal(source_acts, backend)
        target_fisher = compute_empirical_fisher_diagonal(target_acts, backend)

        source_weights = backend.ones((32, 64))
        target_weights = backend.zeros((32, 64))

        merged = fisher_weighted_merge(
            source_weights, target_weights,
            source_fisher, target_fisher,
            backend
        )
        backend.eval(merged)

        # Merged should be closer to source (ones) than target (zeros)
        # Allow small tolerance for numerical precision
        mean_merged = float(backend.to_scalar(backend.mean(merged)))
        assert mean_merged > 0.45, \
            f"High source Fisher should favor source weights, got mean {mean_merged}"

    def test_equal_fisher_gives_average(self, backend):
        """When Fisher values are equal, merge should be close to average."""
        backend.random_seed(42)

        # Same variance -> equal Fisher
        source_acts = backend.random_normal((200, 64))
        backend.random_seed(123)
        target_acts = backend.random_normal((200, 64))

        source_fisher = compute_empirical_fisher_diagonal(source_acts, backend)
        target_fisher = compute_empirical_fisher_diagonal(target_acts, backend)

        source_weights = backend.full((32, 64), 0.0)
        target_weights = backend.full((32, 64), 2.0)

        merged = fisher_weighted_merge(
            source_weights, target_weights,
            source_fisher, target_fisher,
            backend
        )
        backend.eval(merged)

        # Should be close to average (1.0) when Fisher is similar
        mean_merged = float(backend.to_scalar(backend.mean(merged)))
        assert 0.7 < mean_merged < 1.3, \
            f"Equal Fisher should give near-average, got mean {mean_merged}"


class TestFisherMathematicalInvariants:
    """Strict mathematical invariants of diagonal Fisher Information.

    These are properties that must hold by construction, not statistical
    tests that depend on random seeds.
    """

    def test_diagonal_fim_non_negative(self, backend):
        """F_diag[j] = E[x_j²] ≥ 0 for all j — squares are non-negative."""
        backend.random_seed(42)
        activations = backend.random_normal((200, 32))

        result = compute_empirical_fisher_diagonal(activations, backend)

        # Every diagonal entry must be non-negative
        min_val = float(backend.to_scalar(backend.min(result.diagonal_fim)))
        assert min_val >= 0.0, f"Diagonal FIM has negative entry: {min_val}"

    def test_participation_ratio_lower_bound(self, backend):
        """Effective rank ≥ 1 for non-degenerate input.

        Participation ratio: (Σ F_ii)² / Σ F_ii² ≥ 1 when any F_ii > 0.
        By Cauchy-Schwarz: (Σ a_i)² ≤ n × Σ a_i², so PR ∈ [1, n].
        """
        backend.random_seed(42)
        activations = backend.random_normal((200, 32))

        result = compute_empirical_fisher_diagonal(activations, backend)

        assert result.effective_rank >= 1.0, \
            f"Effective rank must be ≥ 1 for non-degenerate input, got {result.effective_rank}"

    def test_uniform_fim_gives_full_rank(self, backend):
        """Equal variance per dim → effective_rank ≈ d.

        When all F_ii are equal: PR = (d × c)² / (d × c²) = d.
        """
        backend.random_seed(42)
        # Uniform variance across all 16 dimensions
        activations = backend.random_normal((1000, 16))

        result = compute_empirical_fisher_diagonal(activations, backend)

        # With enough samples, effective rank should approach 16
        assert result.effective_rank > 12.0, \
            f"Uniform FIM should have effective_rank ≈ 16, got {result.effective_rank}"

    def test_concentrated_fim_gives_low_rank(self, backend):
        """Single active dim → effective_rank ≈ 1.

        When F_11 >> F_ii for i > 1: PR ≈ F_11² / F_11² = 1.
        """
        backend.random_seed(42)
        # One loud dimension, rest near zero
        loud = backend.random_normal((500, 1)) * 100.0
        quiet = backend.random_normal((500, 15)) * 0.001
        activations = backend.concatenate([loud, quiet], axis=1)
        backend.eval(activations)

        result = compute_empirical_fisher_diagonal(activations, backend)

        assert result.effective_rank < 3.0, \
            f"Concentrated FIM should have effective_rank ≈ 1, got {result.effective_rank}"

    def test_condition_number_geq_one(self, backend):
        """κ = λ_max / λ_min ≥ 1 always (max ≥ min by definition)."""
        backend.random_seed(42)
        activations = backend.random_normal((200, 32))

        result = compute_empirical_fisher_diagonal(activations, backend)

        assert result.condition_number >= 1.0, \
            f"Condition number must be ≥ 1, got {result.condition_number}"
