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

"""Tests for sequence-based null-space probe generation.

These tests verify the hybrid vocab + gradient approach for finding
token sequences that activate null-space directions.
"""

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.orthogonal_probe_generator import (
    SequenceProbeResult,
    compute_null_space_basis,
    compute_numerical_rank,
    discretize_embeddings_to_tokens,
    optimize_sequence_for_null_space,
    score_tokens_for_null_space,
)


@pytest.fixture
def backend():
    """Get the default backend."""
    return get_default_backend()


class TestSequenceProbeResult:
    """Tests for SequenceProbeResult dataclass."""

    def test_creation(self):
        """Test that SequenceProbeResult can be created with all fields."""
        result = SequenceProbeResult(
            token_ids=[1, 2, 3, 4],
            text="test sequence",
            null_space_norm=0.5,
            seed_token_id=1,
            gradient_steps=10,
            improvement_ratio=1.5,
        )
        assert result.token_ids == [1, 2, 3, 4]
        assert result.text == "test sequence"
        assert result.null_space_norm == 0.5
        assert result.seed_token_id == 1
        assert result.gradient_steps == 10
        assert result.improvement_ratio == 1.5


class TestDiscretizeEmbeddings:
    """Tests for discretize_embeddings_to_tokens."""

    def test_hard_discretization(self, backend):
        """Test hard discretization finds nearest tokens."""
        b = backend

        # Create a small embedding matrix [5 vocab, 4 dims]
        embed_matrix = b.array([
            [1.0, 0.0, 0.0, 0.0],  # token 0
            [0.0, 1.0, 0.0, 0.0],  # token 1
            [0.0, 0.0, 1.0, 0.0],  # token 2
            [0.0, 0.0, 0.0, 1.0],  # token 3
            [0.5, 0.5, 0.0, 0.0],  # token 4
        ])

        # Test embeddings close to specific tokens
        embeddings = b.array([
            [0.9, 0.1, 0.0, 0.0],  # Should map to token 0
            [0.1, 0.9, 0.0, 0.0],  # Should map to token 1
        ])

        token_ids = discretize_embeddings_to_tokens(
            embeddings=embeddings,
            embedding_matrix=embed_matrix,
            backend=b,
            soft_discretize=False,
        )

        assert len(token_ids) == 2
        assert token_ids[0] == 0
        assert token_ids[1] == 1

    def test_soft_discretization_returns_valid_ids(self, backend):
        """Test soft discretization returns valid token IDs."""
        b = backend

        embed_matrix = b.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])

        embeddings = b.array([
            [0.5, 0.5, 0.0, 0.0],
        ])

        token_ids = discretize_embeddings_to_tokens(
            embeddings=embeddings,
            embedding_matrix=embed_matrix,
            backend=b,
            soft_discretize=True,
            temperature=0.1,
        )

        assert len(token_ids) == 1
        assert 0 <= token_ids[0] < 3


class TestScoreTokensForNullSpace:
    """Tests for score_tokens_for_null_space."""

    def test_scores_higher_for_null_space_aligned(self, backend):
        """Tokens aligned with null space should score higher."""
        b = backend

        # Create activations where token 0 activates null space
        # and token 1 is orthogonal to null space
        activations = b.array([
            [1.0, 0.0, 0.0],  # token 0 - aligned with null basis
            [0.0, 1.0, 0.0],  # token 1 - orthogonal to null basis
        ])

        # Null space basis is first dimension
        U_null = b.array([[1.0], [0.0], [0.0]])

        scores = score_tokens_for_null_space(
            activations_by_token=activations,
            U_null=U_null,
            backend=b,
            normalize=False,
        )
        b.eval(scores)

        scores_list = b.tolist(scores)
        assert scores_list[0] > scores_list[1], "Token aligned with null space should score higher"

    def test_normalized_scores_direction_only(self, backend):
        """Normalized scores should depend on direction, not magnitude."""
        b = backend

        # Same direction, different magnitudes
        activations = b.array([
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],  # Same direction, 2x magnitude
        ])

        U_null = b.array([[1.0], [0.0], [0.0]])

        scores = score_tokens_for_null_space(
            activations_by_token=activations,
            U_null=U_null,
            backend=b,
            normalize=True,  # Normalize to unit vectors
        )
        b.eval(scores)

        scores_list = b.tolist(scores)
        # Should be approximately equal since direction is the same
        assert abs(scores_list[0] - scores_list[1]) < 0.01


class TestComputeNullSpaceBasis:
    """Tests for compute_null_space_basis."""

    def test_null_basis_orthogonal_to_activations(self, backend):
        """Null basis should be orthogonal to activation subspace."""
        b = backend

        # Create activations that span 2D subspace in 4D space
        activations = b.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
        ])
        b.eval(activations)

        rank, hidden_dim = compute_numerical_rank(activations, b)
        assert rank == 2, f"Expected rank 2, got {rank}"
        assert hidden_dim == 4

        U_null = compute_null_space_basis(activations, rank, b)
        assert U_null is not None
        b.eval(U_null)

        null_shape = b.shape(U_null)
        assert null_shape[0] == 4  # hidden_dim
        assert null_shape[1] == 2  # null_rank = hidden_dim - rank

    def test_full_rank_returns_none(self, backend):
        """Full rank activations should return None (no null space)."""
        b = backend

        # Full rank 3x3 matrix
        activations = b.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        b.eval(activations)

        rank, hidden_dim = compute_numerical_rank(activations, b)
        assert rank == 3

        U_null = compute_null_space_basis(activations, rank, b)
        assert U_null is None


class TestComputeNumericalRank:
    """Tests for compute_numerical_rank."""

    def test_rank_of_identity(self, backend):
        """Identity matrix should have full rank."""
        b = backend

        identity = b.eye(5)
        b.eval(identity)

        rank, hidden_dim = compute_numerical_rank(identity, b)
        assert rank == 5
        assert hidden_dim == 5

    def test_rank_of_rank_deficient(self, backend):
        """Rank-deficient matrix should have correct rank."""
        b = backend

        # Rank 2 matrix (rows 2 and 3 are linear combinations)
        matrix = b.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],  # Sum of rows 0 and 1
            [2.0, 0.0, 0.0, 0.0],  # 2x row 0
        ])
        b.eval(matrix)

        rank, hidden_dim = compute_numerical_rank(matrix, b)
        assert rank == 2
        assert hidden_dim == 4

    def test_empty_matrix(self, backend):
        """Empty matrix should have rank 0."""
        b = backend

        empty = b.zeros((0, 4))
        b.eval(empty)

        rank, hidden_dim = compute_numerical_rank(empty, b)
        assert rank == 0
        assert hidden_dim == 4
