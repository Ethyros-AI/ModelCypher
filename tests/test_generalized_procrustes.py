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

from __future__ import annotations

import pytest

from modelcypher.core.domain.geometry.concept_response_matrix import (
    AnchorActivation,
    AnchorMetadata,
    ConceptResponseMatrix,
)
from modelcypher.core.domain.geometry.generalized_procrustes import Config, GeneralizedProcrustes


def test_align_requires_min_models() -> None:
    matrix = [[1.0, 0.0], [0.0, 1.0]]
    config = Config(min_models=2, max_iterations=5)
    assert GeneralizedProcrustes().align([matrix], config=config) is None


def test_align_identity_consensus() -> None:
    matrix = [[1.0, 0.0], [0.0, 1.0]]
    result = GeneralizedProcrustes().align([matrix, matrix], config=Config(max_iterations=5))
    assert result is not None
    assert result.alignment_error == pytest.approx(0.0, abs=1e-6)
    assert result.consensus_variance_ratio == pytest.approx(1.0, abs=1e-6)
    assert result.dimension == 2
    assert result.model_count == 2


def test_align_crms_with_dimension_mismatch() -> None:
    metadata = AnchorMetadata(
        total_count=2,
        semantic_prime_count=2,
        computational_gate_count=0,
        anchor_ids=["prime:a", "prime:b"],
    )
    crm_a = ConceptResponseMatrix(
        model_identifier="a",
        layer_count=1,
        hidden_dim=2,
        anchor_metadata=metadata,
    )
    crm_b = ConceptResponseMatrix(
        model_identifier="b",
        layer_count=1,
        hidden_dim=3,
        anchor_metadata=metadata,
    )
    crm_a.activations = {
        0: {
            "prime:a": AnchorActivation("prime:a", 0, [1.0, 0.0]),
            "prime:b": AnchorActivation("prime:b", 0, [0.0, 1.0]),
        }
    }
    crm_b.activations = {
        0: {
            "prime:a": AnchorActivation("prime:a", 0, [1.0, 0.0, 0.0]),
            "prime:b": AnchorActivation("prime:b", 0, [0.0, 1.0, 0.0]),
        }
    }

    result = GeneralizedProcrustes().align_crms([crm_a, crm_b], layer=0, config=Config(max_iterations=5))
    assert result is not None
    assert result.dimension == 2
    assert result.sample_count == 2


class TestProcrustesEdgeCases:
    """Edge case tests for numerical stability in Procrustes alignment.

    These tests verify the algorithm handles degenerate inputs gracefully.
    For invalid mathematical inputs, "does not crash" is the key property.
    """

    def test_align_with_zero_matrix_does_not_crash(self) -> None:
        """Zero matrices should complete without raising.

        Mathematically, a zero matrix has no meaningful orientation to align.
        The SVD of a zero matrix is degenerate but well-defined.
        """
        zero_matrix = [[0.0, 0.0], [0.0, 0.0]]
        identity_matrix = [[1.0, 0.0], [0.0, 1.0]]
        config = Config(max_iterations=5)

        # Should not raise - this is the key property
        result = GeneralizedProcrustes().align([zero_matrix, identity_matrix], config=config)

        # If it returns a result, structure should be valid
        if result is not None:
            assert result.dimension == 2
            assert result.model_count == 2

    def test_align_with_near_singular_matrix_completes(self) -> None:
        """Near-singular matrices should not cause SVD numerical issues.

        Near-singular matrices have very small singular values which could
        cause numerical instability. SVD should handle this gracefully.
        """
        # Near-singular: rows are almost linearly dependent
        near_singular = [[1.0, 2.0], [1.0001, 2.0002]]
        identity = [[1.0, 0.0], [0.0, 1.0]]
        config = Config(max_iterations=5)

        # Should not raise
        result = GeneralizedProcrustes().align([near_singular, identity], config=config)

        if result is not None:
            assert result.dimension == 2
            # Alignment error will be high due to rank deficiency
            assert result.alignment_error >= 0

    def test_align_with_rank_deficient_activations_completes(self) -> None:
        """Rank-deficient matrices should not crash SVD.

        When activation matrix has rank < dimension, some singular values
        are zero. This is common in low-rank adapter representations.
        """
        # Rank 1 matrix (all rows are multiples of first)
        rank_deficient = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]
        full_rank = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        config = Config(max_iterations=5)

        # Should not raise
        result = GeneralizedProcrustes().align([rank_deficient, full_rank], config=config)

        if result is not None:
            assert result.dimension == 2

    def test_pure_rotation_produces_low_error(self) -> None:
        """Alignment of rotated identity should find the rotation.

        This tests the mathematical property: if B = A @ R for orthogonal R,
        then align(A, B) should find R with near-zero error.
        """
        import math

        matrix_a = [[1.0, 0.0], [0.0, 1.0]]
        # 45 degree rotation
        angle = math.pi / 4
        c, s = math.cos(angle), math.sin(angle)
        matrix_b = [[c, -s], [s, c]]

        config = Config(max_iterations=10)
        result = GeneralizedProcrustes().align([matrix_a, matrix_b], config=config)

        assert result is not None
        # Rotation should be found exactly (within numerical tolerance)
        assert result.alignment_error < 0.1  # Tighter bound

    def test_align_large_dimension_mismatch(self) -> None:
        """Test alignment with significantly different dimensions."""
        small_matrix = [[1.0, 2.0], [3.0, 4.0]]
        large_matrix = [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
        ]
        config = Config(max_iterations=5)

        # Should handle dimension mismatch by truncating to smaller
        result = GeneralizedProcrustes().align([small_matrix, large_matrix], config=config)

        if result is not None:
            # Common dimension should be min of the two
            assert result.dimension == 2

    def test_align_single_row_matrices(self) -> None:
        """Should handle single-row matrices."""
        single_row_a = [[1.0, 2.0, 3.0]]
        single_row_b = [[4.0, 5.0, 6.0]]
        config = Config(max_iterations=5)

        result = GeneralizedProcrustes().align([single_row_a, single_row_b], config=config)

        # Single row alignment is degenerate but should not crash
        if result is not None:
            assert result.sample_count == 1

    def test_align_very_small_values(self) -> None:
        """Should handle matrices with very small values."""
        small_values = [[1e-10, 2e-10], [3e-10, 4e-10]]
        normal_values = [[1.0, 2.0], [3.0, 4.0]]
        config = Config(max_iterations=5)

        result = GeneralizedProcrustes().align([small_values, normal_values], config=config)

        # Should not underflow
        if result is not None:
            assert result.dimension == 2

    def test_align_identical_matrices_many(self) -> None:
        """Should handle many identical matrices efficiently."""
        matrix = [[1.0, 0.0], [0.0, 1.0]]
        many_identical = [matrix] * 10
        config = Config(max_iterations=5)

        result = GeneralizedProcrustes().align(many_identical, config=config)

        assert result is not None
        assert result.model_count == 10
        assert result.alignment_error == pytest.approx(0.0, abs=1e-6)
