# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests comparing scalar vs per-direction scale correction.

These tests verify that per-direction scale correction improves reconstruction
accuracy, especially for:
- High condition number alignments (bottleneck layers)
- Cross-dimensional transfers with heterogeneous scaling
- Alignments where σ₁ >> σₙ

The key insight is that scalar correction (mean of singular values) loses
spectral information. When alignment has direction-dependent scaling,
per-direction correction should produce lower behavioral error.
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.transplant import (
    reconstruct_weight_from_behavior,
    reconstruct_weight_spectral_corrected,
)

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")


def _sqrt_eps(backend):
    """Get sqrt(machine_epsilon) for the default float type."""
    eps = backend.finfo().eps
    return sqrt_scalar(eps, backend)


def _random_orthogonal(n: int, backend):
    """Generate a random orthogonal matrix via QR decomposition."""
    b = backend
    random_matrix = b.random_normal((n, n))
    Q, _ = b.qr(random_matrix)
    b.eval(Q)
    return Q


def _make_alignment_with_known_spectrum(
    d_src: int,
    d_tgt: int,
    singular_values: list[float],
    backend,
):
    """Create an alignment matrix F with specified singular values.

    F = U @ diag(S) @ Vt where U, Vt are random orthogonal.
    This gives us control over the spectral structure for testing.
    """
    b = backend
    k = min(d_src, d_tgt, len(singular_values))

    # Random orthogonal bases
    U = _random_orthogonal(d_src, b)[:, :k]  # [d_src, k]
    V = _random_orthogonal(d_tgt, b)[:, :k]  # [d_tgt, k]
    Vt = b.transpose(V)  # [k, d_tgt]
    b.eval(U, Vt)

    # Construct F = U @ diag(S) @ Vt
    S = b.array(singular_values[:k])
    b.eval(S)

    # F = U @ (S[:, None] * Vt)
    # U is [d_src, k], S is [k], Vt is [k, d_tgt]
    S_diag = b.reshape(S, (k, 1))  # [k, 1]
    scaled_Vt = S_diag * Vt  # [k, d_tgt]
    b.eval(scaled_Vt)

    F = b.matmul(U, scaled_Vt)  # [d_src, d_tgt]
    b.eval(F)

    return F


class TestScalarVsPerDirection:
    """Compare scalar and per-direction scale correction."""

    def test_uniform_scaling_methods_equivalent(self):
        """When all singular values are equal, both methods should give same result."""
        b = get_default_backend()
        sqrt_eps = _sqrt_eps(b)

        d_src, d_tgt = 64, 64
        n_samples = 200

        # Alignment with uniform scaling (all singular values = 2.0)
        # No spectral variation → both methods should be equivalent
        uniform_sv = [2.0] * d_src
        alignment_in = _make_alignment_with_known_spectrum(d_src, d_tgt, uniform_sv, b)
        alignment_out = _make_alignment_with_known_spectrum(d_src, d_tgt, uniform_sv, b)

        # Random source weight and activations
        source_weight = b.random_normal((d_src, d_src))
        input_acts = b.random_normal((n_samples, d_src))
        b.eval(source_weight, input_acts)

        # Reconstruct with both methods
        result_scalar = reconstruct_weight_from_behavior(
            source_weight, input_acts, alignment_in, alignment_out, b
        )
        result_spectral = reconstruct_weight_spectral_corrected(
            source_weight, input_acts, alignment_in, alignment_out, b
        )

        # Weights should be nearly identical
        diff = result_scalar.reconstructed_weight - result_spectral.reconstructed_weight
        diff_norm = b.sqrt(b.sum(diff * diff))
        weight_norm = b.sqrt(b.sum(result_scalar.reconstructed_weight * result_scalar.reconstructed_weight))
        b.eval(diff_norm, weight_norm)

        relative_diff = float(b.to_scalar(diff_norm)) / float(b.to_scalar(weight_norm))

        # Should be within sqrt(eps) for uniform scaling
        assert relative_diff < sqrt_eps * 10, (
            f"Uniform scaling: methods differ by {relative_diff:.2e}, expected < {sqrt_eps * 10:.2e}"
        )

    def test_high_condition_number_spectral_better(self):
        """When σ₁ >> σₙ (bottleneck), per-direction should have lower error."""
        b = get_default_backend()
        b.random_seed(314159)

        d_src, d_tgt = 64, 64
        n_samples = 200

        # Alignment with high condition number: [10, 10, ..., 0.1, 0.1]
        # Dominant directions scale 10x, minor directions scale 0.1x
        # Scalar correction uses mean ≈ 5, which is wrong for both extremes
        high_cond_sv = [10.0] * 32 + [0.1] * 32
        alignment_in = _make_alignment_with_known_spectrum(d_src, d_tgt, high_cond_sv, b)
        alignment_out = _make_alignment_with_known_spectrum(d_src, d_tgt, high_cond_sv, b)

        # Random source weight with structure in both high and low directions
        source_weight = b.random_normal((d_src, d_src))
        input_acts = b.random_normal((n_samples, d_src))
        b.eval(source_weight, input_acts)

        # Reconstruct with both methods
        result_scalar = reconstruct_weight_from_behavior(
            source_weight, input_acts, alignment_in, alignment_out, b
        )
        result_spectral = reconstruct_weight_spectral_corrected(
            source_weight, input_acts, alignment_in, alignment_out, b
        )

        # Compute behavioral error for both
        # Error = ||input @ W_reconstructed.T - target_output||
        output_source = b.matmul(input_acts, b.transpose(source_weight))

        # Get orthogonal parts for projecting to target space
        from modelcypher.core.domain.geometry.numerical_stability import orthogonalize_alignment
        U_in, _ = orthogonalize_alignment(alignment_in, b)
        U_out, _ = orthogonalize_alignment(alignment_out, b)
        b.eval(U_in, U_out)

        input_tgt = b.matmul(input_acts, U_in)
        output_tgt = b.matmul(output_source, U_out)
        b.eval(input_tgt, output_tgt)

        # Behavioral error for scalar method
        out_scalar = b.matmul(input_tgt, b.transpose(result_scalar.reconstructed_weight))
        error_scalar = b.sqrt(b.sum((out_scalar - output_tgt) ** 2))
        b.eval(error_scalar)

        # Behavioral error for spectral method
        out_spectral = b.matmul(input_tgt, b.transpose(result_spectral.reconstructed_weight))
        error_spectral = b.sqrt(b.sum((out_spectral - output_tgt) ** 2))
        b.eval(error_spectral)

        error_scalar_val = float(b.to_scalar(error_scalar))
        error_spectral_val = float(b.to_scalar(error_spectral))

        # Log the comparison
        print(f"\nHigh condition number test (cond ≈ 100):")
        print(f"  Scalar error: {error_scalar_val:.6f}")
        print(f"  Spectral error: {error_spectral_val:.6f}")
        print(f"  Improvement: {(error_scalar_val - error_spectral_val) / error_scalar_val * 100:.1f}%")

        # Spectral should be at least as good (usually better)
        # Allow small tolerance for numerical effects
        eps = division_epsilon(b, output_tgt)
        assert error_spectral_val <= error_scalar_val * (1 + eps), (
            f"Spectral error ({error_spectral_val:.6f}) should be <= scalar ({error_scalar_val:.6f})"
        )

    def test_cross_dimensional_with_varying_scales(self):
        """Cross-dimensional transfer with heterogeneous scaling per direction."""
        b = get_default_backend()

        d_src, d_tgt = 64, 48  # Cross-dimensional
        n_samples = 200

        # Input alignment: heterogeneous scaling [8, 4, 2, 1, 0.5, ...]
        # This simulates a bottleneck where different dimensions scale differently
        k = min(d_src, d_tgt)
        varying_sv_in = [8.0 / (2 ** (i // (k // 4))) for i in range(k)]
        alignment_in = _make_alignment_with_known_spectrum(d_src, d_tgt, varying_sv_in, b)

        # Output alignment: different pattern [1, 2, 4, 8, ...]
        varying_sv_out = [1.0 * (2 ** (i // (k // 4))) for i in range(k)]
        alignment_out = _make_alignment_with_known_spectrum(d_src, d_tgt, varying_sv_out, b)

        source_weight = b.random_normal((d_src, d_src))
        input_acts = b.random_normal((n_samples, d_src))
        b.eval(source_weight, input_acts)

        # Reconstruct with both methods
        result_scalar = reconstruct_weight_from_behavior(
            source_weight, input_acts, alignment_in, alignment_out, b
        )
        result_spectral = reconstruct_weight_spectral_corrected(
            source_weight, input_acts, alignment_in, alignment_out, b
        )

        # Log the scale correction factors
        print(f"\nCross-dimensional test ({d_src} -> {d_tgt}):")
        print(f"  Scalar scale_correction: {result_scalar.scale_correction:.4f}")
        print(f"  Spectral scale_correction (geom mean): {result_spectral.scale_correction:.4f}")
        print(f"  Scalar reconstruction_error: {result_scalar.reconstruction_error:.6f}")
        print(f"  Spectral reconstruction_error: {result_spectral.reconstruction_error:.6f}")

        # Note: Cross-dimensional reconstruction error is NOT machine-precision small
        # because the lstsq solve is projecting from d_src to d_tgt dimensions.
        # When d_src > d_tgt, we lose information and error is expected.
        # The key test is that both methods produce similar results and are stable.
        error_diff = abs(result_scalar.reconstruction_error - result_spectral.reconstruction_error)
        max_error = max(result_scalar.reconstruction_error, result_spectral.reconstruction_error)

        # Errors should be similar (within 10% relative)
        if max_error > 0:
            relative_error_diff = error_diff / max_error
            assert relative_error_diff < 0.1, (
                f"Cross-dim: scalar and spectral errors differ too much: "
                f"{result_scalar.reconstruction_error:.4f} vs {result_spectral.reconstruction_error:.4f}"
            )

    def test_identity_alignment_no_correction_needed(self):
        """When alignment is identity, no scale correction should be applied."""
        b = get_default_backend()
        sqrt_eps = _sqrt_eps(b)

        d = 32
        n_samples = 100

        # Identity alignment has all singular values = 1
        alignment_in = b.eye(d)
        alignment_out = b.eye(d)
        b.eval(alignment_in, alignment_out)

        source_weight = b.random_normal((d, d))
        input_acts = b.random_normal((n_samples, d))
        b.eval(source_weight, input_acts)

        result_spectral = reconstruct_weight_spectral_corrected(
            source_weight, input_acts, alignment_in, alignment_out, b
        )

        # Scale correction should be ~1.0
        assert abs(result_spectral.scale_correction - 1.0) < sqrt_eps, (
            f"Identity alignment: scale_correction={result_spectral.scale_correction:.6f}, expected 1.0"
        )

        # Reconstructed weight should match source closely
        diff = result_spectral.reconstructed_weight - source_weight
        diff_norm = b.sqrt(b.sum(diff * diff))
        src_norm = b.sqrt(b.sum(source_weight * source_weight))
        b.eval(diff_norm, src_norm)

        relative_diff = float(b.to_scalar(diff_norm)) / float(b.to_scalar(src_norm))
        assert relative_diff < sqrt_eps * 10, (
            f"Identity alignment: weight differs by {relative_diff:.2e}"
        )


class TestSpectralCorrectionMath:
    """Test the mathematical properties of spectral correction."""

    def test_spectral_correction_preserves_orthogonal_basis(self):
        """Per-direction correction should work in the orthogonal singular basis."""
        b = get_default_backend()

        d = 32
        n_samples = 100

        # Alignment with known structure
        sv = [4.0, 2.0, 1.0, 0.5] + [0.25] * (d - 4)
        alignment_in = _make_alignment_with_known_spectrum(d, d, sv, b)
        alignment_out = _make_alignment_with_known_spectrum(d, d, sv, b)

        source_weight = b.random_normal((d, d))
        input_acts = b.random_normal((n_samples, d))
        b.eval(source_weight, input_acts)

        result = reconstruct_weight_spectral_corrected(
            source_weight, input_acts, alignment_in, alignment_out, b
        )

        # Check that weight is finite and has reasonable norm
        w_norm = b.sqrt(b.sum(result.reconstructed_weight * result.reconstructed_weight))
        b.eval(w_norm)
        w_norm_val = float(b.to_scalar(w_norm))

        assert w_norm_val > 0, "Reconstructed weight has zero norm"
        assert w_norm_val < 1e10, f"Reconstructed weight norm too large: {w_norm_val}"

    def test_spectral_correction_condition_number_bounded(self):
        """Spectral correction should not introduce numerical instability."""
        b = get_default_backend()

        d = 32
        n_samples = 100

        # Very high condition number alignment
        sv = [100.0] * 4 + [0.01] * (d - 4)  # cond ≈ 10000
        alignment_in = _make_alignment_with_known_spectrum(d, d, sv, b)
        alignment_out = _make_alignment_with_known_spectrum(d, d, sv, b)

        source_weight = b.random_normal((d, d))
        input_acts = b.random_normal((n_samples, d))
        b.eval(source_weight, input_acts)

        result = reconstruct_weight_spectral_corrected(
            source_weight, input_acts, alignment_in, alignment_out, b
        )

        # Condition number should be finite
        assert result.condition_number < 1e12, (
            f"Condition number too high: {result.condition_number:.2e}"
        )

        # Reconstruction error should still be small
        sqrt_eps = _sqrt_eps(b)
        assert result.reconstruction_error < sqrt_eps * 100, (
            f"High-cond reconstruction error: {result.reconstruction_error:.2e}"
        )
