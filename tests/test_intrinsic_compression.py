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

"""
Tests for lossless compression to intrinsic dimensionality.

Key invariants proven:
1. Compression is LOSSLESS on the activation manifold (CKA = 1.0)
2. Reconstruction from factorized form is exact on manifold
3. Compression ratio matches theoretical formula
4. Null space directions have zero contribution to output

The math:
    Y = A @ W.T = A @ (W_used + W_null).T = A @ W_used.T + A @ W_null.T = A @ W_used.T + 0

    Since A has zero variance in null directions, the null component contributes nothing.
    Therefore W can be compressed to W_used without any loss of behavior on the manifold.
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.intrinsic_compression import (
    IntrinsicCompressor,
    compress_layer_with_validation,
    compress_weight_to_intrinsic_dim,
    estimate_compression_potential,
    validate_compression_lossless,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    sqrt_scalar,
)


def _sqrt_eps(backend) -> float:
    """Get sqrt(machine_epsilon) - the precision threshold."""
    eps = machine_epsilon(backend, backend.array([1.0]))
    return sqrt_scalar(eps, backend)


def _make_low_rank_activations(b, n_samples: int, in_dim: int, intrinsic_rank: int):
    """Create activations that span only a low-rank subspace.

    Generates A = Z @ B where:
        Z: [n_samples, intrinsic_rank] - random coefficients
        B: [intrinsic_rank, in_dim] - random subspace basis

    The result A has shape [n_samples, in_dim] but rank = intrinsic_rank.
    """
    Z = b.random_normal((n_samples, intrinsic_rank))
    B = b.random_normal((intrinsic_rank, in_dim))
    activations = b.matmul(Z, B)
    b.eval(activations)
    return activations


class TestCompressionIsLossless:
    """Verify that compression to intrinsic dimensionality is lossless on manifold."""

    def test_cka_equals_one_after_compression(self) -> None:
        """CKA between original and compressed outputs should be 1.0.

        This is the KEY TEST: proves the gauge freedom insight is correct.
        Stripping null space directions does not change behavior on the manifold.
        """
        b = get_default_backend()
        b.random_seed(42)

        # Create a weight matrix
        out_dim, in_dim = 128, 256
        W = b.random_normal((out_dim, in_dim))
        b.eval(W)

        # Create activations that span only a SUBSPACE of in_dim
        # This simulates the real case where models use ~20-35% of capacity
        n_samples = 200
        intrinsic_rank = 64  # Only 25% of in_dim is actually used
        activations = _make_low_rank_activations(b, n_samples, in_dim, intrinsic_rank)

        # Compress
        result = compress_weight_to_intrinsic_dim(W, activations, b)

        # Validate CKA
        cka, max_rel_error = validate_compression_lossless(W, result, activations, b)

        # CKA should be 1.0 within numerical precision
        sqrt_eps = _sqrt_eps(b)
        assert cka > 1.0 - sqrt_eps, (
            f"CKA={cka:.10f} should be ~1.0 for lossless compression. "
            f"This indicates the compression is LOSSY."
        )

        # Utilized rank should match the intrinsic rank of activations
        # (may be slightly different due to numerical thresholds)
        assert abs(result.utilized_rank - intrinsic_rank) <= 2, (
            f"Utilized rank {result.utilized_rank} should match "
            f"intrinsic rank {intrinsic_rank}"
        )

    def test_full_rank_activations_no_compression(self) -> None:
        """When activations span full space, compression should be minimal.

        If activations have full rank, there's no null space to strip.
        The compressed representation should be nearly identical to original.
        """
        b = get_default_backend()
        b.random_seed(123)

        out_dim, in_dim = 64, 64
        W = b.random_normal((out_dim, in_dim))
        b.eval(W)

        # Full-rank activations
        n_samples = 200
        activations = b.random_normal((n_samples, in_dim))
        b.eval(activations)

        result = compress_weight_to_intrinsic_dim(W, activations, b)

        # With full-rank activations, utilized_rank should equal in_dim
        assert result.utilized_rank == in_dim, (
            f"Full-rank activations should have utilized_rank={in_dim}, "
            f"got {result.utilized_rank}"
        )

        # Compression ratio should be ~2 (since we store W_left + V_used)
        expected_ratio = 2.0 * in_dim * in_dim / (out_dim * in_dim)
        assert abs(result.compression_ratio - expected_ratio) < 0.01, (
            f"Compression ratio {result.compression_ratio} should be ~{expected_ratio}"
        )

    def test_extreme_low_rank_high_compression(self) -> None:
        """Very low intrinsic rank should give high compression.

        If activations span only 10 dimensions in 256-dim space,
        we should achieve ~25x compression.
        """
        b = get_default_backend()
        b.random_seed(456)

        out_dim, in_dim = 256, 256
        W = b.random_normal((out_dim, in_dim))
        b.eval(W)

        # Very low rank activations
        n_samples = 100
        intrinsic_rank = 10  # Only 10/256 = 4% utilized
        activations = _make_low_rank_activations(b, n_samples, in_dim, intrinsic_rank)

        # Compress
        result = compress_weight_to_intrinsic_dim(W, activations, b)

        # Validate CKA
        cka, _ = validate_compression_lossless(W, result, activations, b)
        sqrt_eps = _sqrt_eps(b)
        assert cka > 1.0 - sqrt_eps, f"CKA={cka:.10f} should be ~1.0"

        # Compression ratio should be approximately r*(out+in)/(out*in)
        # = 10 * 512 / 65536 = 0.078 (12.8x compression)
        assert result.compression_ratio < 0.2, (
            f"Expected high compression, got ratio {result.compression_ratio}"
        )


class TestReconstructionIsExact:
    """Verify that reconstruction from factorized form is exact on manifold."""

    def test_reconstruction_matches_projected_original(self) -> None:
        """W_reconstructed = W_left @ V_used.T should match W projected to used subspace."""
        b = get_default_backend()
        b.random_seed(789)

        out_dim, in_dim = 64, 128
        W = b.random_normal((out_dim, in_dim))
        b.eval(W)

        # Low-rank activations
        n_samples = 100
        intrinsic_rank = 32
        activations = _make_low_rank_activations(b, n_samples, in_dim, intrinsic_rank)

        # Compress
        result = compress_weight_to_intrinsic_dim(W, activations, b)

        # Reconstruct
        W_reconstructed = result.reconstruct(b)
        b.eval(W_reconstructed)

        # W_reconstructed should equal W @ V_used @ V_used.T
        V_used = result.V_used
        W_projected = b.matmul(b.matmul(W, V_used), b.transpose(V_used))
        b.eval(W_projected)

        # Difference should be zero within machine precision
        diff = b.abs(W_reconstructed - W_projected)
        max_diff_arr = b.max(diff)
        b.eval(max_diff_arr)
        max_diff = float(b.to_scalar(max_diff_arr))

        eps = machine_epsilon(b, W)
        assert max_diff < eps * 100, (
            f"Reconstruction differs from projection by {max_diff}, expected < {eps * 100}"
        )

    def test_output_equivalence_on_manifold(self) -> None:
        """Outputs from original and compressed weights should match on manifold.

        Y_orig = A @ W.T
        Y_comp = A @ V_used @ W_left.T

        These should be identical for all A on the activation manifold.
        """
        b = get_default_backend()
        b.random_seed(101112)

        out_dim, in_dim = 64, 128
        W = b.random_normal((out_dim, in_dim))
        b.eval(W)

        # Low-rank activations
        n_samples = 100
        intrinsic_rank = 32
        activations = _make_low_rank_activations(b, n_samples, in_dim, intrinsic_rank)

        # Compress
        result = compress_weight_to_intrinsic_dim(W, activations, b)

        # Compute outputs
        Y_orig = b.matmul(activations, b.transpose(W))
        A_proj = b.matmul(activations, result.V_used)
        Y_comp = b.matmul(A_proj, b.transpose(result.W_left))
        b.eval(Y_orig, Y_comp)

        # Outputs should match
        diff = b.abs(Y_orig - Y_comp)
        max_diff_arr = b.max(diff)
        b.eval(max_diff_arr)
        max_diff = float(b.to_scalar(max_diff_arr))

        sqrt_eps = _sqrt_eps(b)
        W_norm_arr = b.sqrt(b.sum(W * W))
        b.eval(W_norm_arr)
        W_norm = float(b.to_scalar(W_norm_arr))

        # Relative error should be < sqrt(eps)
        rel_error = max_diff / (W_norm + sqrt_eps)
        assert rel_error < sqrt_eps * 10, (
            f"Output difference {rel_error} exceeds sqrt(eps)={sqrt_eps}"
        )


class TestCompressionRatio:
    """Verify compression ratio matches theoretical formula."""

    def test_compression_ratio_formula(self) -> None:
        """Compression ratio = rank * (out + in) / (out * in)."""
        b = get_default_backend()
        b.random_seed(131415)

        out_dim, in_dim = 128, 256
        W = b.random_normal((out_dim, in_dim))
        b.eval(W)

        # Low-rank activations
        n_samples = 200
        intrinsic_rank = 32
        activations = _make_low_rank_activations(b, n_samples, in_dim, intrinsic_rank)

        result = compress_weight_to_intrinsic_dim(W, activations, b)

        # Expected ratio
        r = result.utilized_rank
        expected = r * (out_dim + in_dim) / (out_dim * in_dim)

        assert abs(result.compression_ratio - expected) < 0.01, (
            f"Compression ratio {result.compression_ratio} doesn't match "
            f"expected {expected}"
        )

    def test_higher_compression_with_lower_rank(self) -> None:
        """Lower intrinsic rank should give higher compression."""
        b = get_default_backend()
        b.random_seed(161718)

        out_dim, in_dim = 256, 256
        W = b.random_normal((out_dim, in_dim))
        b.eval(W)
        n_samples = 200

        ratios = []
        for intrinsic_rank in [16, 32, 64, 128]:
            activations = _make_low_rank_activations(b, n_samples, in_dim, intrinsic_rank)
            result = compress_weight_to_intrinsic_dim(W, activations, b)
            ratios.append(result.compression_ratio)

        # Ratios should be monotonically increasing with rank
        for i in range(len(ratios) - 1):
            assert ratios[i] < ratios[i + 1], (
                f"Compression ratios should increase with rank: {ratios}"
            )


class TestNullSpaceContribution:
    """Verify that null space directions contribute nothing to output."""

    def test_null_space_output_is_zero(self) -> None:
        """A @ W_null.T should be zero when A has zero variance in null space.

        This is the fundamental insight: if activations don't use certain
        directions, those directions don't contribute to the output.
        """
        b = get_default_backend()
        b.random_seed(192021)

        out_dim, in_dim = 64, 128
        W = b.random_normal((out_dim, in_dim))
        b.eval(W)

        # Low-rank activations
        n_samples = 100
        intrinsic_rank = 32
        activations = _make_low_rank_activations(b, n_samples, in_dim, intrinsic_rank)

        # Get null space basis
        from modelcypher.core.domain.geometry.orthogonal_probe_generator import (
            compute_variance_null_space,
        )
        variance_result = compute_variance_null_space(activations, b)
        V_null = variance_result.available_basis  # [in_dim, null_rank]
        b.eval(V_null)

        # Compute A @ V_null - should be near zero
        A_null = b.matmul(activations, V_null)
        b.eval(A_null)

        null_norm_arr = b.sqrt(b.sum(A_null * A_null))
        b.eval(null_norm_arr)
        null_norm = float(b.to_scalar(null_norm_arr))

        # Should be zero within machine precision
        sqrt_eps = _sqrt_eps(b)
        assert null_norm < sqrt_eps * n_samples, (
            f"A @ V_null has norm {null_norm}, expected ~0"
        )


class TestIntrinsicCompressorAPI:
    """Test the IntrinsicCompressor class API."""

    def test_compress_layer_validates_by_default(self) -> None:
        """compress_layer should validate CKA = 1.0 by default."""
        b = get_default_backend()
        b.random_seed(222324)

        compressor = IntrinsicCompressor(b)

        out_dim, in_dim = 64, 128
        W = b.random_normal((out_dim, in_dim))
        b.eval(W)

        n_samples = 100
        intrinsic_rank = 32
        activations = _make_low_rank_activations(b, n_samples, in_dim, intrinsic_rank)

        # Should not raise (CKA validation passes)
        result = compressor.compress_layer(W, activations, validate=True)

        assert result.utilized_rank > 0
        assert result.compression_ratio < 1.0

    def test_estimate_compression_potential(self) -> None:
        """estimate_compression_potential should predict compression ratio."""
        b = get_default_backend()
        b.random_seed(252627)

        in_dim = 128
        n_samples = 100
        intrinsic_rank = 32
        activations = _make_low_rank_activations(b, n_samples, in_dim, intrinsic_rank)

        estimate = estimate_compression_potential(activations, b)

        assert estimate["intrinsic_dim"] > 0
        assert estimate["hidden_dim"] == in_dim
        assert 0.0 < estimate["utilized_fraction"] < 1.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_dimension_mismatch_raises(self) -> None:
        """Should raise when activation dim doesn't match weight in_dim."""
        b = get_default_backend()
        b.random_seed(282930)

        W = b.random_normal((64, 128))
        activations = b.random_normal((100, 64))  # Wrong dim
        b.eval(W, activations)

        with pytest.raises(ValueError, match="Dimension mismatch"):
            compress_weight_to_intrinsic_dim(W, activations, b)

    def test_zero_activations_raises(self) -> None:
        """Should raise when activations are zero/constant."""
        b = get_default_backend()
        b.random_seed(313233)

        W = b.random_normal((64, 128))
        activations = b.zeros((100, 128))  # Zero activations
        b.eval(W, activations)

        with pytest.raises(ValueError, match="No utilized dimensions"):
            compress_weight_to_intrinsic_dim(W, activations, b)
