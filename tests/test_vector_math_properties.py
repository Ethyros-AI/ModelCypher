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

"""Property-based tests for VectorMath.

Tests mathematical invariants:
- L2 norm ≥ 0 (non-negative)
- Normalized vector has norm 1
- Dot product commutativity: a · b = b · a
- Cosine similarity ∈ [-1, 1]
- Cosine self-similarity = 1
- Cauchy-Schwarz: |a · b| ≤ ||a|| × ||b||
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from modelcypher.core.domain.geometry.vector_math import (
    SparseVectorMath,
    VectorMath,
)

# =============================================================================
# Dense Vector Property Tests
# =============================================================================


class TestL2NormInvariants:
    """Tests for L2 norm mathematical invariants."""

    @given(
        st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=50)
    def test_l2_norm_non_negative(self, vector: list[float]) -> None:
        """L2 norm must be >= 0.

        Mathematical property: ||v|| = sqrt(Σv_i²) ≥ 0.
        """
        # Skip vectors that underflow to zero sum_squares
        sum_sq = sum(v * v for v in vector)
        assume(sum_sq > 1e-300)

        norm = VectorMath.l2_norm(vector)

        assert norm is not None
        assert norm >= 0

    @given(
        st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=50)
    def test_normalized_vector_has_unit_norm(self, vector: list[float]) -> None:
        """Normalized vector should have norm ≈ 1.

        Mathematical property: ||v / ||v||| = 1.
        """
        norm_sq = sum(v * v for v in vector)
        assume(norm_sq > 1e-10)  # Skip near-zero vectors

        normalized = VectorMath.l2_normalized(vector)
        norm_of_normalized = VectorMath.l2_norm(normalized)

        assert norm_of_normalized is not None
        assert norm_of_normalized == pytest.approx(1.0, abs=1e-6)


class TestDotProductInvariants:
    """Tests for dot product mathematical invariants."""

    @given(
        st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=50)
    def test_dot_product_commutativity(self, a: list[float]) -> None:
        """Dot product is commutative: a · b = b · a.

        Mathematical property: Commutativity of multiplication.
        """
        import random

        random.seed(42)  # Deterministic for reproducibility
        b = [random.gauss(0, 10) for _ in range(len(a))]

        ab = VectorMath.dot(a, b)
        ba = VectorMath.dot(b, a)

        if ab is None:
            assert ba is None
        else:
            assert ab == pytest.approx(ba, rel=1e-9)

    @given(
        st.lists(
            st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=50)
    def test_cauchy_schwarz_inequality(self, a: list[float]) -> None:
        """|a · b| ≤ ||a|| × ||b|| (Cauchy-Schwarz).

        Mathematical property: Fundamental inequality in inner product spaces.
        """
        import random

        random.seed(42)
        b = [random.gauss(0, 5) for _ in range(len(a))]

        norm_a = VectorMath.l2_norm(a)
        norm_b = VectorMath.l2_norm(b)
        dot_ab = VectorMath.dot(a, b)

        if norm_a is None or norm_b is None or dot_ab is None:
            return  # Skip degenerate cases

        # |a · b| ≤ ||a|| × ||b||
        assert abs(dot_ab) <= norm_a * norm_b + 1e-6


class TestCosineSimilarityInvariants:
    """Tests for cosine similarity mathematical invariants."""

    @given(
        st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=50)
    def test_cosine_similarity_bounded(self, a: list[float]) -> None:
        """Cosine similarity must be in [-1, 1].

        Mathematical property: cos(θ) ∈ [-1, 1].
        """
        import random

        random.seed(42)
        b = [random.gauss(0, 10) for _ in range(len(a))]

        sim = VectorMath.cosine_similarity(a, b)

        if sim is not None:
            assert -1.0 <= sim <= 1.0

    @given(
        st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=50)
    def test_cosine_self_similarity_is_one(self, a: list[float]) -> None:
        """Cosine similarity of vector with itself = 1.

        Mathematical property: cos(0) = 1.
        """
        norm_sq = sum(v * v for v in a)
        assume(norm_sq > 1e-10)

        sim = VectorMath.cosine_similarity(a, a)

        assert sim is not None
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_cosine_opposite_vectors_is_negative_one(self) -> None:
        """Cosine similarity of opposite vectors = -1.

        Mathematical property: cos(π) = -1.
        """
        a = [1.0, 2.0, 3.0]
        b = [-1.0, -2.0, -3.0]

        sim = VectorMath.cosine_similarity(a, b)

        assert sim is not None
        assert sim == pytest.approx(-1.0, abs=1e-6)

    def test_cosine_orthogonal_vectors_is_zero(self) -> None:
        """Cosine similarity of orthogonal vectors = 0.

        Mathematical property: cos(π/2) = 0.
        """
        a = [1.0, 0.0]
        b = [0.0, 1.0]

        sim = VectorMath.cosine_similarity(a, b)

        assert sim is not None
        assert sim == pytest.approx(0.0, abs=1e-6)


# =============================================================================
# Sparse Vector Property Tests
# =============================================================================


class TestSparseL2NormInvariants:
    """Tests for sparse L2 norm invariants."""

    @pytest.mark.parametrize("seed", range(5))
    def test_sparse_l2_norm_non_negative(self, seed: int) -> None:
        """Sparse L2 norm must be >= 0."""
        import numpy as np

        rng = np.random.default_rng(seed)

        vector = {f"key_{i}": rng.uniform(-10, 10) for i in range(10)}

        norm = SparseVectorMath.l2_norm(vector)

        assert norm is not None
        assert norm >= 0


class TestSparseCosineInvariants:
    """Tests for sparse cosine similarity invariants."""

    @pytest.mark.parametrize("seed", range(5))
    def test_sparse_cosine_bounded(self, seed: int) -> None:
        """Sparse cosine similarity must be in [-1, 1]."""
        import numpy as np

        rng = np.random.default_rng(seed)

        keys = [f"key_{i}" for i in range(10)]
        a = {k: rng.uniform(-10, 10) for k in keys}
        b = {k: rng.uniform(-10, 10) for k in keys}

        sim = SparseVectorMath.cosine_similarity(a, b)

        if sim is not None:
            assert -1.0 <= sim <= 1.0

    def test_sparse_cosine_self_is_one(self) -> None:
        """Sparse cosine of vector with itself = 1."""
        a = {"x": 1.0, "y": 2.0, "z": 3.0}

        sim = SparseVectorMath.cosine_similarity(a, a)

        assert sim is not None
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_sparse_cosine_disjoint_is_zero(self) -> None:
        """Sparse cosine of disjoint vectors = 0."""
        a = {"x": 1.0, "y": 2.0}
        b = {"z": 3.0, "w": 4.0}  # No overlap

        sim = SparseVectorMath.cosine_similarity(a, b)

        assert sim is not None
        assert sim == pytest.approx(0.0, abs=1e-6)
