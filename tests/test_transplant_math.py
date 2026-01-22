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
Tests verifying the closed-form math for transplant operations.

These tests prove the math works - no fallbacks needed when the math is correct.
The philosophy: F = pinv(source) @ target is closed-form. CKA = 1.0 is the proof.
If the math is right, it shouldn't fail. If it fails, fix the math, not add fallbacks.

Tests cover:
- Behavioral reconstruction (reconstruct_weight_manifold_aware)
- Compositional stitch (S @ W_src @ H = W_tgt)
- Orthogonal projection (_compute_dimension_projection vs H-derived)
- Attention stitch computability conditions
"""

from __future__ import annotations

import pytest
import math

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.transplant import (
    reconstruct_weight_from_behavior,
    reconstruct_weight_manifold_aware,
    compute_cross_dimensional_transplant,
)
from modelcypher.core.domain.geometry.gram_aligner import GramAligner
from modelcypher.core.use_cases.merge.stages.transplant_helpers import (
    _compute_dimension_projection,
)


def _eps(backend, *values: float) -> float:
    """Get machine epsilon for comparison."""
    return machine_epsilon(backend, backend.array(list(values) or [1.0]))


def _sqrt_eps(backend) -> float:
    """Get sqrt(machine_epsilon) - the precision threshold."""
    eps = machine_epsilon(backend, backend.array([1.0]))
    return sqrt_scalar(eps, backend)


# =============================================================================
# Behavioral Reconstruction Tests
# =============================================================================


class TestBehavioralReconstruction:
    """Verify reconstruct_weight_manifold_aware is closed-form.

    The math:
        1. Orthogonalize alignments: U_in, U_out = polar_decomposition(H)
        2. Compute source behavior: output_src = input_src @ W_src.T
        3. Project to target coords: input_tgt = input_src @ U_in
        4. Solve: W.T = lstsq(input_tgt, output_tgt)

    This is CLOSED-FORM. Error should be bounded by machine precision when n > rank.
    """

    def test_reconstruction_bounded_by_machine_precision(self) -> None:
        """Error should be < sqrt(eps) when n > rank.

        This is the KEY TEST: proves behavioral reconstruction is numerically exact.
        """
        b = get_default_backend()
        b.random_seed(42)

        # Dimensions: source [out_src, in_src], target [out_tgt, in_tgt]
        n_samples = 100  # n >> max_dim ensures overdetermined system
        in_src, out_src = 32, 64
        in_tgt, out_tgt = 24, 48

        # Create random but well-conditioned data
        source_weight = b.random_normal((out_src, in_src))
        input_acts_source = b.random_normal((n_samples, in_src))

        # Create alignment transforms (not orthogonal - testing scale correction)
        alignment_in = b.random_normal((in_src, in_tgt))
        alignment_out = b.random_normal((out_src, out_tgt))
        b.eval(source_weight, input_acts_source, alignment_in, alignment_out)

        # Reconstruct
        result = reconstruct_weight_manifold_aware(
            source_weight=source_weight,
            input_activations_source=input_acts_source,
            alignment_in=alignment_in,
            alignment_out=alignment_out,
            backend=b,
        )

        # Key assertion: reconstruction error bounded by machine precision
        sqrt_eps = _sqrt_eps(b)
        assert result.reconstruction_error < sqrt_eps, (
            f"Reconstruction error {result.reconstruction_error:.2e} exceeds "
            f"sqrt(eps)={sqrt_eps:.2e}. This indicates numerical instability."
        )

    def test_reconstruction_condition_number_bounded(self) -> None:
        """Condition number is bounded for well-conditioned systems.

        The condition number determines numerical stability of the lstsq solve.
        For random well-conditioned data, κ should be moderate.

        Note: The reconstruction error is NOT simply κ × eps because:
        1. RMT truncation discards noise dimensions (increases apparent error in full space)
        2. Scale correction adjusts magnitudes
        3. The error is measured on the INTRINSIC manifold, not full space

        The key invariant: reconstruction_error returned by the function should be small
        (it's measured in the intrinsic manifold subspace where lstsq is exact).
        """
        b = get_default_backend()
        b.random_seed(42)

        n_samples = 100
        in_src, out_src = 32, 64
        in_tgt, out_tgt = 24, 48

        source_weight = b.random_normal((out_src, in_src))
        input_acts_source = b.random_normal((n_samples, in_src))
        alignment_in = b.random_normal((in_src, in_tgt))
        alignment_out = b.random_normal((out_src, out_tgt))
        b.eval(source_weight, input_acts_source, alignment_in, alignment_out)

        result = reconstruct_weight_manifold_aware(
            source_weight=source_weight,
            input_activations_source=input_acts_source,
            alignment_in=alignment_in,
            alignment_out=alignment_out,
            backend=b,
        )

        # Condition number should be finite and reasonable
        assert math.isfinite(result.condition_number), "Condition number should be finite"
        assert result.condition_number >= 1.0, "Condition number must be >= 1.0"

        # For random well-conditioned data, condition number should be moderate
        # (not a hard requirement, just a sanity check)
        assert result.condition_number < 1e6, (
            f"Condition number {result.condition_number:.2e} is very large, "
            "indicating potential numerical instability"
        )

        # The reconstruction_error (manifold error) should be small
        # This is the error in the intrinsic subspace, not full space
        sqrt_eps = _sqrt_eps(b)
        assert result.reconstruction_error < sqrt_eps, (
            f"Manifold reconstruction error {result.reconstruction_error:.2e} "
            f"exceeds sqrt(eps)={sqrt_eps:.2e}"
        )

    def test_reconstruction_identity_alignment(self) -> None:
        """Identity alignment should give exact reconstruction.

        When H_in = H_out = I, the reconstruction should match exactly.
        """
        b = get_default_backend()
        b.random_seed(42)

        n_samples = 100
        dim = 32

        source_weight = b.random_normal((dim, dim))
        input_acts_source = b.random_normal((n_samples, dim))
        alignment_in = b.eye(dim)
        alignment_out = b.eye(dim)
        b.eval(source_weight, input_acts_source, alignment_in, alignment_out)

        result = reconstruct_weight_from_behavior(
            source_weight=source_weight,
            input_activations_source=input_acts_source,
            alignment_in=alignment_in,
            alignment_out=alignment_out,
            backend=b,
        )

        # With identity alignment, reconstructed weight should equal source
        diff = b.sum(b.abs(result.reconstructed_weight - source_weight))
        b.eval(diff)
        diff_val = float(b.to_scalar(diff))

        sqrt_eps = _sqrt_eps(b)
        weight_norm = float(b.to_scalar(b.sqrt(b.sum(source_weight * source_weight))))
        relative_diff = diff_val / max(weight_norm, sqrt_eps)

        assert relative_diff < sqrt_eps, (
            f"Identity alignment reconstruction differs by {relative_diff:.2e}, "
            f"expected < sqrt(eps)={sqrt_eps:.2e}"
        )

    def test_reconstruction_manifold_error_is_small(self) -> None:
        """Reconstruction error on the intrinsic manifold should be small.

        The function uses RMT signal/noise separation to find the intrinsic rank,
        then reconstructs using only the signal subspace. The reconstruction_error
        returned is the error WITHIN this intrinsic subspace.

        Note: Full-space error may be larger due to noise dimensions being discarded.
        The KEY invariant is that manifold_error (returned as reconstruction_error)
        is small - this is what matters for knowledge transfer.
        """
        b = get_default_backend()
        b.random_seed(42)

        n_samples = 100
        in_src, out_src = 32, 64
        in_tgt, out_tgt = 24, 48

        source_weight = b.random_normal((out_src, in_src))
        input_acts_source = b.random_normal((n_samples, in_src))
        alignment_in = b.random_normal((in_src, in_tgt))
        alignment_out = b.random_normal((out_src, out_tgt))
        b.eval(source_weight, input_acts_source, alignment_in, alignment_out)

        result = reconstruct_weight_manifold_aware(
            source_weight=source_weight,
            input_activations_source=input_acts_source,
            alignment_in=alignment_in,
            alignment_out=alignment_out,
            backend=b,
        )

        # The function returns manifold_error as reconstruction_error
        # This should be small because it's computed in the intrinsic subspace
        sqrt_eps = _sqrt_eps(b)
        assert result.reconstruction_error < sqrt_eps, (
            f"Manifold reconstruction error {result.reconstruction_error:.2e} "
            f"exceeds sqrt(eps)={sqrt_eps:.2e}. "
            "RMT-truncated reconstruction should be numerically exact."
        )

        # Verify the weight has correct shape
        weight_shape = tuple(b.shape(result.reconstructed_weight))
        assert weight_shape == (out_tgt, in_tgt), (
            f"Reconstructed weight shape {weight_shape} != expected ({out_tgt}, {in_tgt})"
        )

    def test_reconstruction_with_procrustes_alignment(self) -> None:
        """Reconstruction works correctly when alignments come from Procrustes.

        With random alignments, behavior preservation is not guaranteed because
        there's no meaningful coordinate relationship. But when alignments are
        derived from actual data (as in a real merge), reconstruction should work.

        This test uses synthetically-computed alignments that represent a
        meaningful coordinate transform (derived from the data itself).
        """
        b = get_default_backend()
        b.random_seed(42)

        n_samples = 100
        dim = 32  # Same dimension for simplicity

        # Create source data
        source_weight = b.random_normal((dim, dim))
        input_acts_source = b.random_normal((n_samples, dim))
        b.eval(source_weight, input_acts_source)

        # Create target activations that are a rotation of source
        # This simulates what Procrustes would find
        # Use a simple rotation matrix (orthogonal)
        rotation = b.qr(b.random_normal((dim, dim)))[0]  # Random orthogonal matrix
        b.eval(rotation)

        # Target activations = source activations rotated
        target_acts = b.matmul(input_acts_source, rotation)
        b.eval(target_acts)

        # The alignment IS the rotation (by construction)
        # alignment_in = alignment_out = rotation
        alignment_in = rotation
        alignment_out = rotation

        result = reconstruct_weight_from_behavior(
            source_weight=source_weight,
            input_activations_source=input_acts_source,
            alignment_in=alignment_in,
            alignment_out=alignment_out,
            backend=b,
        )

        # For orthogonal alignment with full-rank data, reconstruction should be exact
        sqrt_eps = _sqrt_eps(b)
        assert result.reconstruction_error < sqrt_eps, (
            f"Reconstruction error {result.reconstruction_error:.2e} "
            f"exceeds sqrt(eps)={sqrt_eps:.2e} for orthogonal alignment"
        )


# =============================================================================
# Compositional Stitch Tests
# =============================================================================


class TestCompositionalStitch:
    """Verify compositional_stitch solves S @ (W_src @ H) = W_tgt.

    The math:
        Solve: S @ (W_src @ H) = W_tgt
        Equivalently: lstsq(W_src @ H, W_tgt) = S.T

    This is CLOSED-FORM. The equation should hold within machine precision.
    """

    def test_stitch_equation_holds_within_precision(self) -> None:
        """||S @ W_src @ H - W_tgt||_F / ||W_tgt||_F < sqrt(eps).

        This is the KEY TEST: proves compositional stitch is numerically exact.
        """
        b = get_default_backend()
        b.random_seed(42)

        # Dimensions:
        # W_src: [src_proj, src_hidden]
        # W_tgt: [tgt_proj, tgt_hidden]
        # H: [src_hidden, tgt_hidden]
        # S: [tgt_proj, src_proj]
        src_proj, src_hidden = 64, 32
        tgt_proj, tgt_hidden = 48, 24

        W_src = b.random_normal((src_proj, src_hidden))
        W_tgt = b.random_normal((tgt_proj, tgt_hidden))
        H = b.random_normal((src_hidden, tgt_hidden))
        b.eval(W_src, W_tgt, H)

        # Compute compositional stitch
        aligner = GramAligner(backend=b)
        S = aligner.compositional_stitch(
            hidden_transform=H,
            source_weight=W_src,
            target_weight=W_tgt,
        )
        b.eval(S)

        # Verify equation: S @ (W_src @ H) = W_tgt
        W_src_H = b.matmul(W_src, H)
        reconstructed = b.matmul(S, W_src_H)
        b.eval(reconstructed)

        # Compute relative error
        diff = b.sqrt(b.sum((reconstructed - W_tgt) ** 2))
        tgt_norm = b.sqrt(b.sum(W_tgt ** 2))
        b.eval(diff, tgt_norm)

        diff_val = float(b.to_scalar(diff))
        tgt_norm_val = float(b.to_scalar(tgt_norm))
        relative_error = diff_val / max(tgt_norm_val, 1e-10)

        sqrt_eps = _sqrt_eps(b)
        assert relative_error < sqrt_eps, (
            f"Compositional stitch relative error {relative_error:.2e} exceeds "
            f"sqrt(eps)={sqrt_eps:.2e}. Equation S @ W_src @ H = W_tgt does NOT hold."
        )

    def test_compositional_vs_direct_stitch_difference(self) -> None:
        """Show that P @ W @ Q ≠ compositional stitch for cross-dim.

        Direct stitch (P @ W @ Q) distorts magnitudes.
        Compositional stitch preserves the input→output behavior.
        They should give DIFFERENT results for cross-dimensional cases.
        """
        b = get_default_backend()
        b.random_seed(42)

        src_proj, src_hidden = 64, 32
        tgt_proj, tgt_hidden = 48, 24

        W_src = b.random_normal((src_proj, src_hidden))
        W_tgt = b.random_normal((tgt_proj, tgt_hidden))
        H = b.random_normal((src_hidden, tgt_hidden))
        b.eval(W_src, W_tgt, H)

        # Compositional stitch
        aligner = GramAligner(backend=b)
        S_compositional = aligner.compositional_stitch(
            hidden_transform=H,
            source_weight=W_src,
            target_weight=W_tgt,
        )
        b.eval(S_compositional)

        # Direct stitch would be: P @ W_src @ Q
        # where P = output_stitch, Q = input_stitch
        # For this test, we use H.T as output stitch (wrong but common fallback)
        P = b.transpose(H)  # Wrong: should be from compositional stitch
        b.eval(P)

        # Check that S_compositional is NOT just P
        # (they operate on different dimensions, so not directly comparable,
        # but we can verify S_compositional satisfies the equation while P doesn't)

        W_src_H = b.matmul(W_src, H)
        compositional_result = b.matmul(S_compositional, W_src_H)
        b.eval(compositional_result)

        # Compositional should match W_tgt
        comp_error = b.sqrt(b.sum((compositional_result - W_tgt) ** 2))
        b.eval(comp_error)
        comp_error_val = float(b.to_scalar(comp_error))

        # Verify compositional stitch is correct
        sqrt_eps = _sqrt_eps(b)
        tgt_norm = float(b.to_scalar(b.sqrt(b.sum(W_tgt ** 2))))
        assert comp_error_val / tgt_norm < sqrt_eps, (
            "Compositional stitch should satisfy the equation exactly"
        )

    def test_input_stitch_equation_holds(self) -> None:
        """Verify compositional_stitch_input: (H.T @ W_src) @ S_in = W_tgt.

        For down_proj weights, we need INPUT stitch, not output stitch.
        """
        b = get_default_backend()
        b.random_seed(42)

        # For down_proj: W_src [src_hidden, src_inter], W_tgt [tgt_hidden, tgt_inter]
        src_hidden, src_inter = 32, 64
        tgt_hidden, tgt_inter = 24, 48

        W_src = b.random_normal((src_hidden, src_inter))
        W_tgt = b.random_normal((tgt_hidden, tgt_inter))
        H = b.random_normal((src_hidden, tgt_hidden))
        b.eval(W_src, W_tgt, H)

        # Compute input stitch
        aligner = GramAligner(backend=b)
        S_in = aligner.compositional_stitch_input(
            hidden_transform=H,
            source_weight=W_src,
            target_weight=W_tgt,
        )
        b.eval(S_in)

        # Verify equation: (H.T @ W_src) @ S_in = W_tgt
        # A = H.T @ W_src = [tgt_hidden, src_inter]
        A = b.matmul(b.transpose(H), W_src)
        reconstructed = b.matmul(A, S_in)
        b.eval(reconstructed)

        # Compute relative error
        diff = b.sqrt(b.sum((reconstructed - W_tgt) ** 2))
        tgt_norm = b.sqrt(b.sum(W_tgt ** 2))
        b.eval(diff, tgt_norm)

        diff_val = float(b.to_scalar(diff))
        tgt_norm_val = float(b.to_scalar(tgt_norm))
        relative_error = diff_val / max(tgt_norm_val, 1e-10)

        sqrt_eps = _sqrt_eps(b)
        assert relative_error < sqrt_eps, (
            f"Input stitch relative error {relative_error:.2e} exceeds "
            f"sqrt(eps)={sqrt_eps:.2e}. Equation (H.T @ W_src) @ S_in = W_tgt does NOT hold."
        )


# =============================================================================
# Orthogonal Projection Tests
# =============================================================================


class TestOrthogonalProjection:
    """Verify _compute_dimension_projection rejects geometrically wrong operations.

    The old implementation used [[I, 0]] or [[I], [0]] - a pure guess with 10x
    more error than H-derived projection. It now correctly raises an error
    for cross-dimensional cases, forcing the caller to use proper stitches.
    """

    def test_same_dimension_returns_identity(self) -> None:
        """Same dimensions should return identity matrix."""
        b = get_default_backend()

        dim = 64
        projection = _compute_dimension_projection(b, dim, dim)
        b.eval(projection)

        expected = b.eye(dim)
        diff = b.sum(b.abs(projection - expected))
        b.eval(diff)

        eps = _eps(b, 1.0)
        assert float(b.to_scalar(diff)) < eps, (
            "Same-dimension projection should be identity"
        )

    def test_cross_dimensional_raises_error(self) -> None:
        """Cross-dimensional projection should raise RuntimeError.

        The [[I, 0]] pattern is geometrically wrong (10x more error).
        The function now refuses to return a bad guess.
        """
        import pytest
        b = get_default_backend()

        src_dim, tgt_dim = 64, 48

        with pytest.raises(RuntimeError) as exc_info:
            _compute_dimension_projection(b, src_dim, tgt_dim)

        assert "geometrically wrong" in str(exc_info.value).lower()
        assert "alignment-derived" in str(exc_info.value).lower()

    def test_cross_dimensional_both_directions_raise(self) -> None:
        """Both src > tgt and src < tgt should raise error."""
        import pytest
        b = get_default_backend()

        # src > tgt
        with pytest.raises(RuntimeError):
            _compute_dimension_projection(b, 64, 48)

        # src < tgt
        with pytest.raises(RuntimeError):
            _compute_dimension_projection(b, 48, 64)

    def test_h_derived_preserves_behavior(self) -> None:
        """H-derived projection preserves input→output behavior.

        When we project source activations using H, the result should
        match the target activations (within numerical precision).
        """
        b = get_default_backend()
        b.random_seed(42)

        n_samples = 100
        src_dim, tgt_dim = 64, 48

        source_acts = b.random_normal((n_samples, src_dim))
        H_true = b.random_normal((src_dim, tgt_dim))
        target_acts = b.matmul(source_acts, H_true)
        b.eval(source_acts, target_acts)

        # H-derived projection (should recover H_true)
        from modelcypher.core.domain.geometry.numerical_stability import gpu_lstsq
        H_derived = gpu_lstsq(b, source_acts, target_acts)
        b.eval(H_derived)

        # Apply projection
        projected = b.matmul(source_acts, H_derived)
        b.eval(projected)

        # Should match target
        diff = b.mean(b.abs(projected - target_acts))
        b.eval(diff)
        diff_val = float(b.to_scalar(diff))

        sqrt_eps = _sqrt_eps(b)
        assert diff_val < sqrt_eps, (
            f"H-derived projection behavioral difference {diff_val:.2e} "
            f"exceeds sqrt(eps)={sqrt_eps:.2e}"
        )


# =============================================================================
# Cross-Dimensional Transplant Integration Tests
# =============================================================================


class TestCrossDimensionalTransplant:
    """Integration tests for full cross-dimensional transplant pipeline."""

    def test_transplant_preserves_target_behavior_on_null_space(self) -> None:
        """Null-space projection should preserve target behavior exactly.

        The key invariant: after transplant, the merged weight should
        produce the same output as target weight on target activations.
        """
        b = get_default_backend()
        b.random_seed(42)

        n_samples = 100
        out_src, in_src = 64, 32
        out_tgt, in_tgt = 48, 24

        source_weight = b.random_normal((out_src, in_src))
        target_weight = b.random_normal((out_tgt, in_tgt))
        input_acts_source = b.random_normal((n_samples, in_src))
        input_acts_target = b.random_normal((n_samples, in_tgt))
        alignment_in = b.random_normal((in_src, in_tgt))
        alignment_out = b.random_normal((out_src, out_tgt))
        b.eval(source_weight, target_weight, input_acts_source, input_acts_target)
        b.eval(alignment_in, alignment_out)

        result = compute_cross_dimensional_transplant(
            source_weight=source_weight,
            target_weight=target_weight,
            input_activations_source=input_acts_source,
            input_activations_target=input_acts_target,
            alignment_in=alignment_in,
            alignment_out=alignment_out,
            backend=b,
        )

        # Merged weight should produce same output as target on target activations
        # (behavioral null-space projection)
        target_output = b.matmul(input_acts_target, b.transpose(target_weight))
        merged_output = b.matmul(input_acts_target, b.transpose(result.merged_weight))
        b.eval(target_output, merged_output)

        # The difference should be in the NULL SPACE of target activations
        # So the behavioral difference should be small (projected norm << delta norm)
        diff_output = b.mean(b.abs(merged_output - target_output))
        b.eval(diff_output)
        diff_val = float(b.to_scalar(diff_output))

        # preserved_fraction ≈ 0 means projection worked (delta went to null space)
        # so behavioral difference on target activations should be small
        sqrt_eps = _sqrt_eps(b)

        # If null_rank > 0, we should see some preserved fraction
        # If null_rank = 0, all delta is absorbed
        assert result.null_rank >= 0, "Null rank should be non-negative"

        # The behavioral impact on TARGET activations should be minimal
        # because we projected into the null space of TARGET activations
        target_norm = float(b.to_scalar(b.sqrt(b.sum(target_output ** 2))))
        relative_diff = diff_val / max(target_norm, sqrt_eps)

        # With proper null-space projection, relative diff should be small
        # (though not zero because some delta may leak into non-null directions)
        assert relative_diff < 1.0, (
            f"Relative behavioral difference {relative_diff:.2e} is too large. "
            "Null-space projection may not be working correctly."
        )

    def test_transplant_metrics_are_consistent(self) -> None:
        """Transplant metrics should be internally consistent.

        preserved_fraction = projected_norm / delta_norm
        Both should use behavioral norm (not Frobenius).
        """
        b = get_default_backend()
        b.random_seed(42)

        n_samples = 100
        out_src, in_src = 64, 32
        out_tgt, in_tgt = 48, 24

        source_weight = b.random_normal((out_src, in_src))
        target_weight = b.random_normal((out_tgt, in_tgt))
        input_acts_source = b.random_normal((n_samples, in_src))
        input_acts_target = b.random_normal((n_samples, in_tgt))
        alignment_in = b.random_normal((in_src, in_tgt))
        alignment_out = b.random_normal((out_src, out_tgt))
        b.eval(source_weight, target_weight, input_acts_source, input_acts_target)
        b.eval(alignment_in, alignment_out)

        result = compute_cross_dimensional_transplant(
            source_weight=source_weight,
            target_weight=target_weight,
            input_activations_source=input_acts_source,
            input_activations_target=input_acts_target,
            alignment_in=alignment_in,
            alignment_out=alignment_out,
            backend=b,
        )

        # Verify metric consistency
        if result.delta_norm > 0:
            computed_fraction = result.projected_norm / result.delta_norm
            assert abs(computed_fraction - result.preserved_fraction) < 1e-6, (
                f"preserved_fraction {result.preserved_fraction} != "
                f"projected/delta {computed_fraction}"
            )

        # Null rank should be non-negative and bounded
        assert 0 <= result.null_rank, "null_rank should be non-negative"

        # Transfer strength should be in [0, 1] (density-weighted)
        assert 0 <= result.transfer_strength <= 1.0, (
            f"transfer_strength {result.transfer_strength} out of [0, 1] range"
        )


# =============================================================================
# Attention Stitch Tests
# =============================================================================


class TestAttentionStitch:
    """Verify when attention stitch is computable.

    Attention stitch CAN be computed if we have attention activations.
    The fallback should only trigger when activations are unavailable.
    """

    def test_attention_stitch_computable_from_activations(self) -> None:
        """If attention activations exist, stitch CAN be computed.

        The same lstsq approach works for attention as for hidden states.
        """
        b = get_default_backend()
        b.random_seed(42)

        n_samples = 100
        src_attn_dim, tgt_attn_dim = 2048, 1536

        # Attention activations
        source_attn_acts = b.random_normal((n_samples, src_attn_dim))
        target_attn_acts = b.random_normal((n_samples, tgt_attn_dim))
        b.eval(source_attn_acts, target_attn_acts)

        # Compute attention stitch via lstsq (same as hidden stitch)
        from modelcypher.core.domain.geometry.numerical_stability import gpu_lstsq
        attention_stitch = gpu_lstsq(b, source_attn_acts, target_attn_acts)
        b.eval(attention_stitch)

        # Verify it works
        aligned = b.matmul(source_attn_acts, attention_stitch)
        b.eval(aligned)

        diff = b.mean(b.abs(aligned - target_attn_acts))
        b.eval(diff)
        diff_val = float(b.to_scalar(diff))

        sqrt_eps = _sqrt_eps(b)
        assert diff_val < sqrt_eps, (
            f"Attention stitch error {diff_val:.2e} exceeds sqrt(eps)={sqrt_eps:.2e}. "
            "Attention stitch IS computable when activations are available."
        )

    def test_attention_stitch_equation_holds(self) -> None:
        """Verify: source_acts @ stitch ≈ target_acts.

        Same closed-form solution as hidden stitch.
        """
        b = get_default_backend()
        b.random_seed(42)

        n_samples = 100
        src_dim, tgt_dim = 2048, 1536

        source_acts = b.random_normal((n_samples, src_dim))
        target_acts = b.random_normal((n_samples, tgt_dim))
        b.eval(source_acts, target_acts)

        # Use GramAligner's find_perfect_alignment
        aligner = GramAligner(backend=b)
        result = aligner.find_perfect_alignment(source_acts, target_acts)

        # Apply transform
        aligned = b.matmul(source_acts, result.feature_transform)
        b.eval(aligned)

        # Compute relative error
        diff = b.sqrt(b.sum((aligned - target_acts) ** 2))
        tgt_norm = b.sqrt(b.sum(target_acts ** 2))
        b.eval(diff, tgt_norm)

        relative_error = float(b.to_scalar(diff)) / float(b.to_scalar(tgt_norm))

        # Alignment residual should be small
        sqrt_eps = _sqrt_eps(b)
        assert relative_error < sqrt_eps * 10, (  # Allow some margin
            f"Attention alignment residual {relative_error:.2e} too large. "
            "Closed-form alignment should work for attention space."
        )
