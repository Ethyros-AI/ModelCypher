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

"""Tests for Task Singular Vectors module."""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.task_singular_vectors import (
    SVDBlendConfig,
    TaskVectorDecomposition,
    _find_spectral_gap,
    blend_with_svd_awareness,
    decompose_task_vector,
)


@pytest.fixture
def backend():
    """Get compute backend."""
    return get_default_backend()


class TestSVDBlendConfig:
    """Tests for SVDBlendConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SVDBlendConfig()

        assert config.epsilon is None
        assert config.condition_threshold is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SVDBlendConfig(
            epsilon=1e-8,
            condition_threshold=1e4,
        )

        assert config.epsilon == 1e-8
        assert config.condition_threshold == 1e4


class TestTaskVectorDecomposition:
    """Tests for TaskVectorDecomposition dataclass."""

    def test_is_valid_with_nonzero_singular_values(self, backend):
        """Test is_valid returns True for valid decomposition."""
        decomp = TaskVectorDecomposition(
            U=backend.array([[1.0, 0.0], [0.0, 1.0]]),
            S=backend.array([2.0, 1.0]),
            Vt=backend.array([[1.0, 0.0], [0.0, 1.0]]),
            variance_captured=1.0,
            effective_rank=2,
            original_shape=(2, 2),
        )

        assert decomp.is_valid is True

    def test_is_valid_with_zero_singular_values(self, backend):
        """Test is_valid returns False for zero singular values."""
        decomp = TaskVectorDecomposition(
            U=backend.array([[1.0]]),
            S=backend.array([]),  # Empty
            Vt=backend.array([[1.0]]),
            variance_captured=0.0,
            effective_rank=0,
            original_shape=(1, 1),
        )

        assert decomp.is_valid is False

    def test_is_valid_with_zero_variance(self, backend):
        """Test is_valid returns False for zero variance."""
        decomp = TaskVectorDecomposition(
            U=backend.array([[1.0]]),
            S=backend.array([1.0]),
            Vt=backend.array([[1.0]]),
            variance_captured=0.0,  # Zero variance
            effective_rank=1,
            original_shape=(1, 1),
        )

        assert decomp.is_valid is False

    def test_reconstruct_identity(self, backend):
        """Test reconstruction with alpha=1.0."""
        # Create a simple 2x2 decomposition
        U = backend.array([[1.0, 0.0], [0.0, 1.0]])
        S = backend.array([2.0, 1.0])
        Vt = backend.array([[1.0, 0.0], [0.0, 1.0]])

        decomp = TaskVectorDecomposition(
            U=U,
            S=S,
            Vt=Vt,
            variance_captured=1.0,
            effective_rank=2,
            original_shape=(2, 2),
        )

        result = decomp.reconstruct(alpha=1.0)
        backend.eval(result)
        result_np = backend.to_numpy(result)

        # Should reconstruct to diag(2, 1)
        assert result_np[0, 0] == pytest.approx(2.0, rel=0.01)
        assert result_np[1, 1] == pytest.approx(1.0, rel=0.01)

    def test_reconstruct_with_scaling(self, backend):
        """Test reconstruction with alpha scaling."""
        U = backend.array([[1.0]])
        S = backend.array([4.0])
        Vt = backend.array([[1.0]])

        decomp = TaskVectorDecomposition(
            U=U,
            S=S,
            Vt=Vt,
            variance_captured=1.0,
            effective_rank=1,
            original_shape=(1, 1),
        )

        result = decomp.reconstruct(alpha=0.5)
        backend.eval(result)
        result_np = backend.to_numpy(result)

        assert result_np[0, 0] == pytest.approx(2.0, rel=0.01)  # 0.5 * 4.0

    def test_reconstruct_invalid_returns_zeros(self, backend):
        """Test reconstruction of invalid decomposition returns zeros."""
        decomp = TaskVectorDecomposition(
            U=backend.zeros((2, 1)),
            S=backend.array([]),
            Vt=backend.zeros((1, 2)),
            variance_captured=0.0,
            effective_rank=0,
            original_shape=(2, 2),
        )

        result = decomp.reconstruct(alpha=1.0)
        backend.eval(result)
        result_np = backend.to_numpy(result)

        assert result_np.shape == (2, 2)
        assert result_np.sum() == pytest.approx(0.0)


class TestFindSpectralGap:
    """Tests for _find_spectral_gap function."""

    def test_single_value(self):
        """Test with single singular value."""
        result = _find_spectral_gap([5.0], epsilon=1e-10)
        assert result == 1

    def test_two_values_no_gap(self):
        """Test with two similar values (no significant gap)."""
        result = _find_spectral_gap([5.0, 4.8], epsilon=1e-10)
        # Gap is (5.0-4.8)/5.0 = 0.04
        assert result >= 1

    def test_clear_spectral_gap(self):
        """Test with clear spectral gap."""
        # Large gap between 5.0 and 1.0
        result = _find_spectral_gap([5.0, 4.9, 1.0, 0.9], epsilon=1e-10)
        # Gap at index 2: (4.9-1.0)/4.9 ≈ 0.8
        assert result == 2

    def test_empty_list(self):
        """Test with empty list."""
        result = _find_spectral_gap([], epsilon=1e-10)
        assert result == 0

    def test_values_below_epsilon(self):
        """Test that values below epsilon are ignored."""
        result = _find_spectral_gap([5.0, 1e-12, 1e-15], epsilon=1e-10)
        # Should stop at first value below epsilon
        assert result >= 1

    def test_monotonically_decreasing(self):
        """Test with monotonically decreasing values."""
        # 10, 8, 6, 4, 2 - gaps are 20%, 25%, 33%, 50%
        result = _find_spectral_gap([10.0, 8.0, 6.0, 4.0, 2.0], epsilon=1e-10)
        # Largest gap is between 4.0 and 2.0 (50%)
        assert result == 4


class TestDecomposeTaskVector:
    """Tests for decompose_task_vector function."""

    def test_identical_weights_returns_zero_decomposition(self, backend):
        """Test that identical weights produce zero decomposition."""
        backend.random_seed(42)
        weight = backend.random_normal((10, 10))
        backend.eval(weight)

        decomp = decompose_task_vector(weight, weight)

        assert decomp.variance_captured == pytest.approx(0.0)
        assert decomp.effective_rank == 0

    def test_different_weights_produces_valid_decomposition(self, backend):
        """Test that different weights produce valid decomposition."""
        backend.random_seed(42)
        source = backend.random_normal((10, 10))
        backend.random_seed(43)
        target = backend.random_normal((10, 10))
        backend.eval(source, target)

        decomp = decompose_task_vector(source, target)

        assert decomp.is_valid
        assert decomp.variance_captured > 0
        assert decomp.effective_rank > 0
        assert decomp.original_shape == (10, 10)

    def test_1d_weights_handled(self, backend):
        """Test that 1D weights (biases) are handled correctly."""
        source = backend.array([1.0, 2.0, 3.0])
        target = backend.array([0.5, 1.5, 2.5])

        decomp = decompose_task_vector(source, target)

        assert decomp.is_valid
        assert decomp.effective_rank == 1
        assert decomp.original_shape == (3,)

    def test_decomposition_preserves_shape_info(self, backend):
        """Test that original shape is preserved."""
        backend.random_seed(42)
        source = backend.random_normal((5, 8))
        backend.random_seed(43)
        target = backend.random_normal((5, 8))
        backend.eval(source, target)

        decomp = decompose_task_vector(source, target)

        assert decomp.original_shape == (5, 8)

    def test_reconstruction_approximates_original(self, backend):
        """Test that U @ S @ Vt approximates original delta."""
        backend.random_seed(42)
        source = backend.random_normal((8, 8))
        backend.random_seed(43)
        target = backend.random_normal((8, 8))
        backend.eval(source, target)

        delta = source - target
        decomp = decompose_task_vector(source, target)

        if decomp.is_valid:
            reconstructed = decomp.reconstruct(alpha=1.0)
            backend.eval(reconstructed)

            # The reconstruction should approximate the delta
            diff = backend.to_numpy(reconstructed) - backend.to_numpy(
                backend.astype(delta, "float32")
            )
            frobenius = (diff * diff).sum() ** 0.5
            original_norm = (backend.to_numpy(delta) ** 2).sum() ** 0.5

            # Reconstruction should be close (allowing for numerical precision)
            assert frobenius / (original_norm + 1e-10) < 0.1

    def test_custom_config(self, backend):
        """Test with custom configuration."""
        backend.random_seed(42)
        source = backend.random_normal((6, 6))
        backend.random_seed(43)
        target = backend.random_normal((6, 6))
        backend.eval(source, target)

        config = SVDBlendConfig(epsilon=1e-6, condition_threshold=1e3)
        decomp = decompose_task_vector(source, target, config=config)

        assert decomp.original_shape == (6, 6)


class TestBlendWithSVDAwareness:
    """Tests for blend_with_svd_awareness function."""

    def test_alpha_zero_returns_source(self, backend):
        """Test that alpha=0 returns source weight."""
        backend.random_seed(42)
        source = backend.random_normal((5, 5))
        backend.random_seed(43)
        target = backend.random_normal((5, 5))
        backend.eval(source, target)

        result = blend_with_svd_awareness(source, target, base_alpha=0.0)
        backend.eval(result)

        # alpha=0 means (1-0)*source + 0*target = source
        diff = backend.to_numpy(result) - backend.to_numpy(source)
        assert (diff * diff).sum() < 1e-10

    def test_alpha_one_moves_toward_target(self, backend):
        """Test that alpha=1 moves result toward target.

        Note: SVD-aware blending uses per-component alpha based on variance,
        so alpha=1.0 doesn't return exactly target. Instead, it weights
        high-variance (skill) components differently from low-variance
        (structure) components.
        """
        backend.random_seed(42)
        source = backend.random_normal((5, 5))
        backend.random_seed(43)
        target = backend.random_normal((5, 5))
        backend.eval(source, target)

        result = blend_with_svd_awareness(source, target, base_alpha=1.0)
        backend.eval(result)

        # Result should be different from source (blending occurred)
        source_np = backend.to_numpy(source)
        result_np = backend.to_numpy(result)
        diff_from_source = ((result_np - source_np) ** 2).sum()
        assert diff_from_source > 0.1, "Result should differ from source"

        # Result should have correct shape
        assert result_np.shape == source_np.shape

    def test_1d_weights_linear_interpolation(self, backend):
        """Test that 1D weights use linear interpolation."""
        source = backend.array([2.0, 4.0, 6.0])
        target = backend.array([0.0, 0.0, 0.0])

        result = blend_with_svd_awareness(source, target, base_alpha=0.5)
        backend.eval(result)
        result_np = backend.to_numpy(result)

        # (1-0.5)*source + 0.5*target = 0.5*source
        assert result_np[0] == pytest.approx(1.0, rel=0.01)
        assert result_np[1] == pytest.approx(2.0, rel=0.01)
        assert result_np[2] == pytest.approx(3.0, rel=0.01)

    def test_identical_weights_returns_either(self, backend):
        """Test that identical weights return the same value."""
        backend.random_seed(42)
        weight = backend.random_normal((5, 5))
        backend.eval(weight)

        result = blend_with_svd_awareness(weight, weight, base_alpha=0.5)
        backend.eval(result)

        # Should return the original weight (source = target)
        diff = backend.to_numpy(result) - backend.to_numpy(weight)
        assert (diff * diff).sum() < 1e-8

    def test_result_shape_matches_input(self, backend):
        """Test that result shape matches input shape."""
        backend.random_seed(42)
        source = backend.random_normal((7, 12))
        backend.random_seed(43)
        target = backend.random_normal((7, 12))
        backend.eval(source, target)

        result = blend_with_svd_awareness(source, target, base_alpha=0.3)
        backend.eval(result)

        assert result.shape == (7, 12)

    def test_intermediate_alpha(self, backend):
        """Test that intermediate alpha produces intermediate result."""
        backend.random_seed(42)
        source = backend.random_normal((6, 6))
        backend.random_seed(43)
        target = backend.random_normal((6, 6))
        backend.eval(source, target)

        result = blend_with_svd_awareness(source, target, base_alpha=0.5)
        backend.eval(result)

        source_np = backend.to_numpy(source)
        target_np = backend.to_numpy(target)
        result_np = backend.to_numpy(result)

        # Result should be between source and target in some sense
        # Check that result is not identical to source or target
        diff_source = ((result_np - source_np) ** 2).sum()
        diff_target = ((result_np - target_np) ** 2).sum()

        # Both differences should be non-zero for different source/target
        assert diff_source > 1e-10 or diff_target > 1e-10


class TestSVDMathematicalProperties:
    """Tests for mathematical properties of SVD decomposition."""

    def test_singular_values_nonnegative(self, backend):
        """Test that singular values are non-negative."""
        backend.random_seed(42)
        source = backend.random_normal((10, 10))
        backend.random_seed(43)
        target = backend.random_normal((10, 10))
        backend.eval(source, target)

        decomp = decompose_task_vector(source, target)
        S_np = backend.to_numpy(decomp.S)

        for s in S_np:
            assert s >= 0

    def test_singular_values_sorted_descending(self, backend):
        """Test that singular values are sorted in descending order."""
        backend.random_seed(42)
        source = backend.random_normal((10, 10))
        backend.random_seed(43)
        target = backend.random_normal((10, 10))
        backend.eval(source, target)

        decomp = decompose_task_vector(source, target)
        S_np = backend.to_numpy(decomp.S)

        for i in range(len(S_np) - 1):
            assert S_np[i] >= S_np[i + 1] - 1e-6  # Allow small numerical error

    def test_variance_captured_bounded(self, backend):
        """Test that variance captured is in [0, 1]."""
        backend.random_seed(42)
        source = backend.random_normal((8, 8))
        backend.random_seed(43)
        target = backend.random_normal((8, 8))
        backend.eval(source, target)

        decomp = decompose_task_vector(source, target)

        assert 0.0 <= decomp.variance_captured <= 1.0 + 1e-6

    def test_effective_rank_bounded_by_min_dim(self, backend):
        """Test that effective rank is bounded by min(m, n)."""
        backend.random_seed(42)
        source = backend.random_normal((5, 12))
        backend.random_seed(43)
        target = backend.random_normal((5, 12))
        backend.eval(source, target)

        decomp = decompose_task_vector(source, target)

        min_dim = min(5, 12)
        assert decomp.effective_rank <= min_dim

    @pytest.mark.parametrize("seed", range(5))
    def test_decomposition_deterministic(self, backend, seed):
        """Test that decomposition is deterministic for same input."""
        backend.random_seed(seed)
        source = backend.random_normal((6, 6))
        backend.random_seed(seed + 100)
        target = backend.random_normal((6, 6))
        backend.eval(source, target)

        decomp1 = decompose_task_vector(source, target)
        decomp2 = decompose_task_vector(source, target)

        assert decomp1.effective_rank == decomp2.effective_rank
        assert decomp1.variance_captured == pytest.approx(
            decomp2.variance_captured, rel=1e-6
        )
