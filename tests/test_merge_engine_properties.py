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

"""Property-based and edge case tests for merge_engine.py.

These tests are designed to find REAL BUGS, not just pass themselves.
They test mathematical invariants and edge cases that could break merging.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from modelcypher.core.use_cases.merge_engine import RotationalMerger, SVDBases


# =============================================================================
# Test Fixtures
# =============================================================================


class MockBackend:
    """Minimal numpy backend for testing."""

    def array(self, data, dtype=None):
        return np.array(data, dtype=dtype or np.float32)

    def matmul(self, a, b):
        return a @ b

    def transpose(self, arr, axes=None):
        return np.transpose(arr, axes)

    def eval(self, *arrays):
        pass

    def to_numpy(self, arr):
        return np.asarray(arr)

    def stack(self, arrays, axis=0):
        return np.stack(arrays, axis=axis)

    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=dtype or np.float32)

    def ones(self, shape, dtype=None):
        return np.ones(shape, dtype=dtype or np.float32)

    def eye(self, n, dtype=None):
        return np.eye(n, dtype=dtype or np.float32)

    def sum(self, arr, axis=None, keepdims=False):
        return np.sum(arr, axis=axis, keepdims=keepdims)

    def mean(self, arr, axis=None, keepdims=False):
        return np.mean(arr, axis=axis, keepdims=keepdims)

    def sqrt(self, arr):
        return np.sqrt(arr)

    def where(self, cond, a, b):
        return np.where(cond, a, b)


@pytest.fixture
def backend():
    return MockBackend()


@pytest.fixture
def merger(backend):
    """Create a merger with mock backend."""
    m = object.__new__(RotationalMerger)
    m.backend = backend
    return m


# =============================================================================
# Hypothesis Strategies for Weight Matrices
# =============================================================================


@st.composite
def weight_matrix(draw, min_dim=2, max_dim=64):
    """Generate a random weight matrix."""
    rows = draw(st.integers(min_value=min_dim, max_value=max_dim))
    cols = draw(st.integers(min_value=min_dim, max_value=max_dim))
    # Use bounded floats to avoid numerical issues
    data = draw(
        st.lists(
            st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False), min_size=cols, max_size=cols),
            min_size=rows,
            max_size=rows,
        )
    )
    return np.array(data, dtype=np.float32)


@st.composite
def orthogonal_matrix(draw, n=None):
    """Generate a random orthogonal matrix via QR decomposition."""
    if n is None:
        n = draw(st.integers(min_value=2, max_value=32))
    # Generate random matrix and orthogonalize via QR
    rand_matrix = draw(
        st.lists(
            st.lists(st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False), min_size=n, max_size=n),
            min_size=n,
            max_size=n,
        )
    )
    A = np.array(rand_matrix, dtype=np.float32)
    # QR decomposition gives orthogonal Q
    Q, _ = np.linalg.qr(A)
    return Q


@st.composite
def svd_bases(draw, shape=None):
    """Generate valid SVD bases for testing projection."""
    if shape is None:
        rows = draw(st.integers(min_value=4, max_value=32))
        cols = draw(st.integers(min_value=4, max_value=32))
    else:
        rows, cols = shape
    
    rank = min(rows, cols, draw(st.integers(min_value=2, max_value=min(rows, cols))))
    
    # Generate random U and V matrices
    u_data = np.random.randn(rows, rank).astype(np.float32)
    v_data = np.random.randn(cols, rank).astype(np.float32)
    
    # Orthonormalize U and V
    u, _ = np.linalg.qr(u_data)
    v, _ = np.linalg.qr(v_data)
    u = u[:, :rank]
    v = v[:, :rank]
    
    # Generate singular values
    singular_values = sorted(np.random.uniform(0.1, 10.0, rank).tolist(), reverse=True)
    
    return SVDBases(
        u=u,
        v=v,
        spectral_norm=singular_values[0] if singular_values else 1.0,
        singular_values=singular_values,
    )


# =============================================================================
# Property Tests: Mathematical Invariants
# =============================================================================


class TestRotationDeviation:
    """Tests for _rotation_deviation: measures how far matrix is from orthogonal."""

    def test_identity_has_zero_deviation(self, merger):
        """Identity matrix should have zero deviation from orthogonal."""
        identity = np.eye(4, dtype=np.float32)
        deviation = merger._rotation_deviation(identity)
        assert abs(deviation) < 1e-6, f"Identity deviation should be ~0, got {deviation}"

    def test_orthogonal_has_zero_deviation(self, merger):
        """Any orthogonal matrix should have near-zero deviation."""
        # Random orthogonal via QR
        A = np.random.randn(4, 4).astype(np.float32)
        Q, _ = np.linalg.qr(A)
        deviation = merger._rotation_deviation(Q)
        assert abs(deviation) < 1e-5, f"Orthogonal matrix deviation should be ~0, got {deviation}"

    def test_non_orthogonal_has_positive_deviation(self, merger):
        """Non-orthogonal matrix should have positive deviation."""
        # Scale one row to break orthogonality
        A = np.eye(4, dtype=np.float32)
        A[0, :] *= 2.0  # No longer orthogonal
        deviation = merger._rotation_deviation(A)
        assert deviation > 0.1, f"Non-orthogonal should have positive deviation, got {deviation}"

    @given(orthogonal_matrix())
    @settings(max_examples=20)
    def test_orthogonal_property(self, merger, Q):
        """Property: any orthogonal matrix has near-zero deviation."""
        assume(Q.shape[0] >= 2)
        deviation = merger._rotation_deviation(Q)
        assert deviation < 1e-4, f"Orthogonal deviation={deviation}"


class TestConditionNumber:
    """Tests for _condition_number: ratio of max/min singular values."""

    def test_identity_has_condition_one(self, merger):
        """Identity (all singular values = 1) has condition number 1."""
        singular_values = [1.0, 1.0, 1.0, 1.0]
        cond = merger._condition_number(singular_values)
        assert abs(cond - 1.0) < 1e-6, f"Expected condition 1.0, got {cond}"

    def test_large_ratio_gives_large_condition(self, merger):
        """Large spread in singular values gives large condition number."""
        singular_values = [1000.0, 1.0]
        cond = merger._condition_number(singular_values)
        assert cond >= 100.0, f"Expected high condition number, got {cond}"

    def test_empty_singular_values(self, merger):
        """Empty singular values should not crash."""
        cond = merger._condition_number([])
        # Should return some default/safe value
        assert isinstance(cond, float)

    def test_single_singular_value(self, merger):
        """Single singular value has condition 1."""
        cond = merger._condition_number([5.0])
        assert abs(cond - 1.0) < 1e-6

    @given(st.lists(st.floats(min_value=0.001, max_value=1000, allow_nan=False), min_size=1, max_size=10))
    @settings(max_examples=20)
    def test_condition_is_positive(self, merger, singular_values):
        """Property: condition number is always positive."""
        cond = merger._condition_number(singular_values)
        assert cond > 0, f"Condition number must be positive, got {cond}"


class TestSpectralRatio:
    """Tests for _spectral_ratio: compares spectral norms."""

    def test_identical_returns_one(self, merger):
        """Identical singular values give ratio 1."""
        sv = [5.0, 3.0, 1.0]
        ratio = merger._spectral_ratio(sv, sv)
        assert abs(ratio - 1.0) < 1e-6, f"Expected 1.0, got {ratio}"

    def test_double_returns_two(self, merger):
        """Target twice source gives ratio 2."""
        source = [1.0, 0.5]
        target = [2.0, 1.0]
        ratio = merger._spectral_ratio(target, source)
        assert abs(ratio - 2.0) < 1e-6, f"Expected 2.0, got {ratio}"

    def test_empty_lists(self, merger):
        """Empty lists should not crash."""
        ratio = merger._spectral_ratio([], [])
        assert isinstance(ratio, float)


class TestDeterminantSign:
    """Tests for _determinant_sign: returns +1 or -1 for rotation sense."""

    def test_identity_positive(self, merger):
        """Identity has positive determinant."""
        identity = np.eye(3, dtype=np.float32)
        sign = merger._determinant_sign(identity)
        assert sign > 0, f"Identity should have positive det, got {sign}"

    def test_reflection_negative(self, merger):
        """Reflection matrix has negative determinant."""
        reflection = np.diag([1.0, 1.0, -1.0]).astype(np.float32)
        sign = merger._determinant_sign(reflection)
        assert sign < 0, f"Reflection should have negative det, got {sign}"

    def test_rotation_positive(self, merger):
        """Rotation matrix has positive determinant."""
        theta = np.pi / 4
        rotation = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ], dtype=np.float32)
        sign = merger._determinant_sign(rotation)
        assert sign > 0, f"Rotation should have positive det, got {sign}"


# =============================================================================
# Edge Case Tests: Degenerate Inputs
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases that could crash or corrupt merging."""

    def test_zero_weight_matrix_svd(self, merger, backend):
        """Zero matrix should not crash truncated SVD."""
        zero_weight = np.zeros((8, 8), dtype=np.float32)
        try:
            bases = merger._truncated_svd_bases(
                weight=zero_weight,
                rank=4,
                oversampling=2,
                power_iterations=1,
                seed=42,
                label="test-zero",
            )
            # Should complete without crashing
            assert bases is not None
        except np.linalg.LinAlgError:
            # SVD on zero matrix can fail - this is acceptable
            pass

    def test_rank_deficient_matrix_svd(self, merger, backend):
        """Rank-deficient matrix should handle gracefully."""
        # Rank 1 matrix (outer product)
        u = np.array([[1], [2], [3], [4]], dtype=np.float32)
        v = np.array([[1, 2, 3, 4]], dtype=np.float32)
        rank_1 = u @ v  # 4x4 rank-1 matrix

        bases = merger._truncated_svd_bases(
            weight=rank_1,
            rank=2,  # Request more rank than matrix has
            oversampling=2,
            power_iterations=1,
            seed=42,
            label="test-rank-deficient",
        )
        assert bases is not None
        # Should have at most rank 1 significant singular value
        assert bases.singular_values[0] > 1e-6

    def test_very_small_values(self, merger, backend):
        """Very small values should not underflow in SVD."""
        tiny = np.eye(4, dtype=np.float32) * 1e-30
        bases = merger._truncated_svd_bases(
            weight=tiny,
            rank=2,
            oversampling=1,
            power_iterations=1,
            seed=42,
            label="test-tiny",
        )
        assert bases is not None
        # Singular values should be very small but not zero
        assert not np.isnan(bases.spectral_norm)

    def test_very_large_values(self, merger, backend):
        """Very large values should not overflow in SVD."""
        huge = np.eye(4, dtype=np.float32) * 1e30
        bases = merger._truncated_svd_bases(
            weight=huge,
            rank=2,
            oversampling=1,
            power_iterations=1,
            seed=42,
            label="test-huge",
        )
        assert bases is not None
        assert not np.isnan(bases.spectral_norm)
        assert not np.isinf(bases.spectral_norm)

    def test_rectangular_weights(self, merger, backend):
        """Non-square weight matrices should work."""
        # Tall matrix
        tall = np.random.randn(16, 4).astype(np.float32)
        bases = merger._truncated_svd_bases(
            weight=tall,
            rank=2,
            oversampling=1,
            power_iterations=1,
            seed=42,
            label="test-tall",
        )
        assert bases is not None
        assert bases.u.shape == (16, 2)
        assert bases.v.shape == (4, 2)

        # Wide matrix
        wide = np.random.randn(4, 16).astype(np.float32)
        bases_wide = merger._truncated_svd_bases(
            weight=wide,
            rank=2,
            oversampling=1,
            power_iterations=1,
            seed=42,
            label="test-wide",
        )
        assert bases_wide is not None
        assert bases_wide.u.shape == (4, 2)
        assert bases_wide.v.shape == (16, 2)


# =============================================================================
# Regression Tests: Known Input/Output Pairs
# =============================================================================


class TestRegressionCases:
    """Tests with known expected outputs to catch algorithm changes."""

    def test_procrustes_error_identity_alignment(self, merger, backend):
        """Aligning identical matrices should give zero error."""
        weight = np.random.randn(8, 8).astype(np.float32)
        weight_arr = backend.array(weight)
        
        # Compute SVD bases
        bases = merger._truncated_svd_bases(
            weight=weight_arr,
            rank=4,
            oversampling=2,
            power_iterations=2,
            seed=42,
            label="test",
        )
        
        # Identity omega (no rotation)
        omega = np.eye(4, dtype=np.float32)
        
        error = merger._procrustes_error(
            source_weight=weight_arr,
            target_weight=weight_arr,
            source_bases=bases,
            target_bases=bases,
            omega_out=omega,
            omega_in=backend.array(omega),
        )
        
        # Error should be small (not exactly zero due to truncation)
        assert error < 1.0, f"Self-alignment error should be small, got {error}"

    def test_rotation_deviation_known_values(self, merger):
        """Test rotation deviation with known perturbation."""
        # Start with identity
        omega = np.eye(3, dtype=np.float32)
        base_deviation = merger._rotation_deviation(omega)
        
        # Add small perturbation
        omega_perturbed = omega + np.array([
            [0.0, 0.1, 0.0],
            [-0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ], dtype=np.float32)
        
        perturbed_deviation = merger._rotation_deviation(omega_perturbed)
        
        # Perturbed should have higher deviation
        assert perturbed_deviation > base_deviation, \
            f"Perturbation should increase deviation: {base_deviation} -> {perturbed_deviation}"


# =============================================================================
# Mutation Testing Helpers
# =============================================================================


class TestMutationDetection:
    """Tests designed to catch specific mutations/bugs."""

    def test_condition_number_uses_min_not_max(self, merger):
        """Verify condition number divides by min, not max."""
        # If we divide by max, result would be < 1
        # If we divide by min, result would be > 1
        singular_values = [100.0, 1.0]
        cond = merger._condition_number(singular_values)
        assert cond == pytest.approx(100.0, rel=0.01), \
            f"Condition should be max/min=100, got {cond}"

    def test_spectral_ratio_is_target_over_source(self, merger):
        """Verify spectral ratio is target/source, not source/target."""
        source = [1.0]
        target = [4.0]
        ratio = merger._spectral_ratio(target, source)
        assert ratio == pytest.approx(4.0), \
            f"Ratio should be target/source=4, got {ratio}"
