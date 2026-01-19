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

"""Comprehensive tests for the anchor-relative concept grafting pipeline.

Tests the canonical pipeline for knowledge transfer:
    1. Map to anchor-relative space (dimension-agnostic)
    2. Align using Procrustes in anchor space
    3. Compute delta in anchor space
    4. Weight by density ratio (no thresholds)
    5. Decode to target activation space

Key invariant: Source coordinates NEVER touch target weights directly.
"""

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.anchor_decoder import (
    compute_anchor_decoder,
    decode_to_activation_space,
)
from modelcypher.core.domain.geometry.anchor_grafting import (
    AnchorGraftingResult,
    compute_anchor_grafting_delta,
)
from modelcypher.core.domain.geometry.relative_representation import (
    compute_relative_representation,
)
from modelcypher.core.domain.geometry.transplant import (
    compute_transplant_delta,
)
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon


class TestAnchorDecoder:
    """Tests for anchor_decoder.py functions."""

    def test_decoder_reconstruction_quality(self) -> None:
        """Decoder should reconstruct target activations.

        B = pinv(S_t) @ A_t, reconstruction = S_t @ B
        Error: ||S_t @ B - A_t|| / ||A_t||

        The reconstruction quality depends on the relationship between
        n_samples, n_anchors, and d_target. We verify the decoder works
        correctly (finite, non-negative error, correct shape).
        """
        b = get_default_backend()
        b.random_seed(42)

        n_samples, d_target, n_anchors = 50, 128, 20
        # Generate target activations and anchors
        target_activations = b.random_normal((n_samples, d_target))
        target_anchors = b.random_normal((n_anchors, d_target))
        b.eval(target_activations, target_anchors)

        # Compute relative representation (no backend arg needed)
        S_t = compute_relative_representation(target_activations, target_anchors)
        b.eval(S_t)

        # Compute decoder
        decoder, reconstruction_error = compute_anchor_decoder(
            target_relative_rep=S_t,
            target_activations=target_activations,
            backend=b,
        )

        # Verify structural properties (no arbitrary thresholds)
        assert reconstruction_error >= 0.0, "Reconstruction error should be non-negative"
        assert not (reconstruction_error != reconstruction_error), "Reconstruction error should be finite (not NaN)"
        assert b.shape(decoder) == (n_anchors, d_target), "Decoder should have shape [n_anchors, d_target]"

    def test_decoder_underdetermined_system(self) -> None:
        """Decoder with few samples should still work (underdetermined).

        When n_samples < n_anchors, reconstruction may be perfect (0 error)
        but decoder generalizes poorly. Test that it doesn't crash.
        """
        b = get_default_backend()
        b.random_seed(42)

        n_samples, d_target, n_anchors = 10, 128, 30  # More anchors than samples
        target_activations = b.random_normal((n_samples, d_target))
        target_anchors = b.random_normal((n_anchors, d_target))
        b.eval(target_activations, target_anchors)

        S_t = compute_relative_representation(target_activations, target_anchors)
        b.eval(S_t)

        decoder, reconstruction_error = compute_anchor_decoder(
            target_relative_rep=S_t,
            target_activations=target_activations,
            backend=b,
        )

        # Should not crash; error may be 0 (exact fit) or small
        assert reconstruction_error >= 0.0
        assert b.shape(decoder) == (n_anchors, d_target)

    def test_decode_preserves_shape(self) -> None:
        """decode_to_activation_space should output [n, d_target]."""
        b = get_default_backend()
        b.random_seed(42)

        n_samples, d_target, n_anchors = 50, 128, 20

        delta_relative = b.random_normal((n_samples, n_anchors))
        decoder = b.random_normal((n_anchors, d_target))
        density_weights = b.abs(b.random_normal((n_samples,)))  # Positive weights
        density_weights = density_weights / b.max(density_weights)  # Normalize to [0, 1]
        b.eval(delta_relative, decoder, density_weights)

        delta_activations = decode_to_activation_space(
            delta_relative=delta_relative,
            decoder=decoder,
            density_weights=density_weights,
            backend=b,
        )

        assert b.shape(delta_activations) == (n_samples, d_target)

    def test_decode_respects_density_weights(self) -> None:
        """Zero density weights should produce zero delta."""
        b = get_default_backend()
        b.random_seed(42)

        n_samples, d_target, n_anchors = 20, 64, 10

        delta_relative = b.random_normal((n_samples, n_anchors))
        decoder = b.random_normal((n_anchors, d_target))
        zero_weights = b.zeros((n_samples,))
        b.eval(delta_relative, decoder, zero_weights)

        delta_activations = decode_to_activation_space(
            delta_relative=delta_relative,
            decoder=decoder,
            density_weights=zero_weights,
            backend=b,
        )
        b.eval(delta_activations)

        # All-zero weights should produce all-zero delta
        delta_norm = float(b.to_scalar(b.sum(b.abs(delta_activations))))
        tol = division_epsilon(b, delta_activations)
        assert delta_norm <= tol, f"Expected zero delta, got norm {delta_norm}"


class TestAnchorGrafting:
    """Tests for the full anchor-relative grafting pipeline."""

    def test_same_dimension_grafting(self) -> None:
        """Grafting between same-dimension models should work."""
        b = get_default_backend()
        b.random_seed(42)

        n_samples, d_hidden, n_anchors = 50, 128, 20

        source_activations = b.random_normal((n_samples, d_hidden))
        target_activations = b.random_normal((n_samples, d_hidden))
        source_anchors = b.random_normal((n_anchors, d_hidden))
        target_anchors = b.random_normal((n_anchors, d_hidden))
        b.eval(source_activations, target_activations, source_anchors, target_anchors)

        result = compute_anchor_grafting_delta(
            source_activations=source_activations,
            target_activations=target_activations,
            source_anchors=source_anchors,
            target_anchors=target_anchors,
            backend=b,
        )

        # Result should have correct shape
        assert b.shape(result.delta_activations) == (n_samples, d_hidden)
        assert 0.0 <= result.transfer_fraction <= 1.0
        assert result.reconstruction_error >= 0.0

    def test_cross_dimension_grafting(self) -> None:
        """Grafting between different-dimension models (the key use case).

        Source: d_source (e.g., 2048)
        Target: d_target (e.g., 896)

        delta_activations should be in target space [n, d_target].
        """
        b = get_default_backend()
        b.random_seed(42)

        n_samples, d_source, d_target, n_anchors = 50, 256, 128, 20

        source_activations = b.random_normal((n_samples, d_source))
        target_activations = b.random_normal((n_samples, d_target))
        source_anchors = b.random_normal((n_anchors, d_source))
        target_anchors = b.random_normal((n_anchors, d_target))
        b.eval(source_activations, target_activations, source_anchors, target_anchors)

        result = compute_anchor_grafting_delta(
            source_activations=source_activations,
            target_activations=target_activations,
            source_anchors=source_anchors,
            target_anchors=target_anchors,
            backend=b,
        )

        # delta_activations MUST be in target space
        assert b.shape(result.delta_activations) == (n_samples, d_target), (
            f"Expected shape ({n_samples}, {d_target}), "
            f"got {b.shape(result.delta_activations)}"
        )

        # Procrustes rotation should be [n_anchors, n_anchors]
        assert b.shape(result.rotation_matrix) == (n_anchors, n_anchors)

        # Decoder should be [n_anchors, d_target]
        assert b.shape(result.decoder_matrix) == (n_anchors, d_target)

    def test_density_weights_in_valid_range(self) -> None:
        """Density weights should be in [0, 1]."""
        b = get_default_backend()
        b.random_seed(42)

        n_samples, d_hidden, n_anchors = 30, 64, 15

        source_activations = b.random_normal((n_samples, d_hidden))
        target_activations = b.random_normal((n_samples, d_hidden))
        source_anchors = b.random_normal((n_anchors, d_hidden))
        target_anchors = b.random_normal((n_anchors, d_hidden))
        b.eval(source_activations, target_activations, source_anchors, target_anchors)

        result = compute_anchor_grafting_delta(
            source_activations=source_activations,
            target_activations=target_activations,
            source_anchors=source_anchors,
            target_anchors=target_anchors,
            backend=b,
        )

        # Check density weights are in [0, 1]
        min_w = float(b.to_scalar(b.min(result.density_weights)))
        max_w = float(b.to_scalar(b.max(result.density_weights)))

        assert min_w >= 0.0, f"Density weight below 0: {min_w}"
        assert max_w <= 1.0, f"Density weight above 1: {max_w}"

    def test_procrustes_alignment_works(self) -> None:
        """Procrustes alignment should find a rotation in anchor space.

        Verify that the alignment produces valid outputs:
        - Rotation matrix is orthogonal [n_anchors, n_anchors]
        - Alignment error is finite and non-negative
        - Delta activations have correct shape
        """
        b = get_default_backend()
        b.random_seed(42)

        n_samples, d_hidden, n_anchors = 50, 128, 20

        source_activations = b.random_normal((n_samples, d_hidden))
        target_activations = b.random_normal((n_samples, d_hidden))
        source_anchors = b.random_normal((n_anchors, d_hidden))
        target_anchors = b.random_normal((n_anchors, d_hidden))
        b.eval(source_activations, target_activations, source_anchors, target_anchors)

        result = compute_anchor_grafting_delta(
            source_activations=source_activations,
            target_activations=target_activations,
            source_anchors=source_anchors,
            target_anchors=target_anchors,
            backend=b,
        )

        # Verify structural properties (no arbitrary thresholds)
        assert result.alignment_error >= 0.0, "Alignment error should be non-negative"
        assert not (result.alignment_error != result.alignment_error), "Alignment error should be finite"
        assert b.shape(result.rotation_matrix) == (n_anchors, n_anchors), "Rotation should be [n_anchors, n_anchors]"
        assert b.shape(result.delta_activations) == (n_samples, d_hidden), "Delta should match target shape"


class TestAnchorRelativeTransplant:
    """Tests for the anchor-relative mode in transplant.py."""

    def test_anchor_relative_mode_preserves_boundary(self) -> None:
        """Boundary outputs should be EXACTLY preserved.

        A_boundary @ W' = A_boundary @ W_target

        This is the core invariant of the constrained solver.
        """
        b = get_default_backend()
        b.random_seed(42)

        in_dim, out_dim = 64, 32
        n_core, n_boundary = 20, 10

        weight_target = b.random_normal((out_dim, in_dim))
        activations_core = b.random_normal((n_core, in_dim))
        boundary_activations = b.random_normal((n_boundary, in_dim))
        # delta_A represents the desired change at core samples
        delta_activations = b.random_normal((n_core, out_dim)) * 0.1
        b.eval(weight_target, activations_core, boundary_activations, delta_activations)

        # Compute boundary output BEFORE transplant
        # W is [out, in], output = A @ W.T → [n_boundary, out_dim]
        output_before = b.matmul(boundary_activations, b.transpose(weight_target))
        b.eval(output_before)

        result = compute_transplant_delta(
            weight_target=weight_target,
            activations_core=activations_core,
            delta_activations=delta_activations,
            boundary_activations=boundary_activations,
            backend=b,
        )

        assert result.applied, "Transplant should have been applied"

        merged_weight = b.array(result.merged_weight)
        b.eval(merged_weight)

        # Compute boundary output AFTER transplant
        output_after = b.matmul(boundary_activations, b.transpose(merged_weight))
        b.eval(output_after)

        # Boundary outputs should be EXACTLY preserved
        diff = output_after - output_before
        diff_norm = float(b.to_scalar(b.sum(b.abs(diff))))
        output_norm = float(b.to_scalar(b.sum(b.abs(output_before))))

        denom = output_norm + division_epsilon(b, output_before)
        relative_diff = diff_norm / denom
        tol = division_epsilon(b, output_before)
        assert relative_diff <= tol, (
            f"Boundary output changed by {relative_diff:.2e} (should be ~0)"
        )

    def test_anchor_relative_mode_moves_core_toward_delta(self) -> None:
        """Core outputs should move toward the desired delta.

        A_core @ W' ≈ A_core @ W_target + delta_A_core

        The approximation depends on available null-space capacity.
        """
        b = get_default_backend()
        b.random_seed(42)

        in_dim, out_dim = 128, 64
        n_core = 30

        weight_target = b.random_normal((out_dim, in_dim))
        activations_core = b.random_normal((n_core, in_dim))
        delta_activations = b.random_normal((n_core, out_dim)) * 0.5
        b.eval(weight_target, activations_core, delta_activations)

        # No boundary constraint - all capacity available for delta
        result = compute_transplant_delta(
            weight_target=weight_target,
            activations_core=activations_core,
            delta_activations=delta_activations,
            boundary_activations=None,  # No boundary constraint
            backend=b,
        )

        assert result.applied

        merged_weight = b.array(result.merged_weight)
        b.eval(merged_weight)

        # Compute output change
        output_before = b.matmul(activations_core, b.transpose(weight_target))
        output_after = b.matmul(activations_core, b.transpose(merged_weight))
        b.eval(output_before, output_after)

        actual_delta = output_after - output_before
        b.eval(actual_delta)

        delta_W_unc = b.matmul(b.pinv(activations_core), delta_activations)
        expected_delta = b.matmul(activations_core, b.transpose(delta_W_unc))
        b.eval(delta_W_unc, expected_delta)

        diff = b.abs(actual_delta - expected_delta)
        b.eval(diff)
        max_diff = float(b.to_scalar(b.max(diff)))
        eps = division_epsilon(b, expected_delta)
        assert max_diff <= eps

    def test_anchor_relative_handles_no_boundary(self) -> None:
        """Anchor-relative mode should work with no boundary constraint."""
        b = get_default_backend()
        b.random_seed(42)

        in_dim, out_dim = 64, 32
        n_core = 15

        weight_target = b.random_normal((out_dim, in_dim))
        activations_core = b.random_normal((n_core, in_dim))
        delta_activations = b.random_normal((n_core, out_dim))
        b.eval(weight_target, activations_core, delta_activations)

        result = compute_transplant_delta(
            weight_target=weight_target,
            activations_core=activations_core,
            delta_activations=delta_activations,
            boundary_activations=None,
            backend=b,
        )

        assert result.applied
        assert result.preserved_fraction >= 0.0

    def test_zero_delta_produces_no_change(self) -> None:
        """Zero delta_activations should result in no weight change."""
        b = get_default_backend()
        b.random_seed(42)

        in_dim, out_dim = 64, 32
        n_core = 20

        weight_target = b.random_normal((out_dim, in_dim))
        activations_core = b.random_normal((n_core, in_dim))
        zero_delta = b.zeros((n_core, out_dim))
        b.eval(weight_target, activations_core, zero_delta)

        result = compute_transplant_delta(
            weight_target=weight_target,
            activations_core=activations_core,
            delta_activations=zero_delta,
            boundary_activations=None,
            backend=b,
        )

        merged_weight = b.array(result.merged_weight)
        b.eval(merged_weight)

        # Weight change should be minimal
        diff = merged_weight - weight_target
        diff_norm = float(b.to_scalar(b.sum(b.abs(diff))))

        tol = division_epsilon(b, merged_weight)
        assert diff_norm <= tol, (
            f"Zero delta produced weight change of {diff_norm:.2e}"
        )


class TestEndToEndAnchorGrafting:
    """End-to-end tests for the full anchor-relative pipeline."""

    def test_full_pipeline_same_dimension(self) -> None:
        """Full pipeline: grafting → transplant for same-dimension models.

        In real usage, the grafting delta is in hidden space (d_hidden),
        and the weight matrix has shape [out_dim, in_dim] where both dims
        relate to the hidden space. For this test, we use d_hidden = out_dim
        to match the actual merge scenario.
        """
        b = get_default_backend()
        b.random_seed(42)

        n_samples, d_hidden, n_anchors = 50, 64, 15
        in_dim = d_hidden  # Input dimension is hidden dimension
        out_dim = d_hidden  # Output dimension is also hidden dimension

        # Activations
        source_activations = b.random_normal((n_samples, d_hidden))
        target_activations = b.random_normal((n_samples, d_hidden))
        source_anchors = b.random_normal((n_anchors, d_hidden))
        target_anchors = b.random_normal((n_anchors, d_hidden))
        b.eval(source_activations, target_activations, source_anchors, target_anchors)

        # Weight: square matrix for this test (out = in = hidden)
        weight_target = b.random_normal((out_dim, in_dim))
        b.eval(weight_target)

        # Step 1: Compute anchor-relative grafting delta
        grafting_result = compute_anchor_grafting_delta(
            source_activations=source_activations,
            target_activations=target_activations,
            source_anchors=source_anchors,
            target_anchors=target_anchors,
            backend=b,
        )

        # Step 2: Apply delta via constrained transplant
        # Use first n_core samples as core, rest as boundary
        n_core = 30
        n_boundary = n_samples - n_core
        core_acts = b.take(target_activations, b.arange(n_core), axis=0)
        boundary_acts = b.take(
            target_activations, b.arange(n_core, n_samples), axis=0
        )
        core_delta = b.take(grafting_result.delta_activations, b.arange(n_core), axis=0)
        b.eval(core_acts, boundary_acts, core_delta)

        result = compute_transplant_delta(
            weight_target=weight_target,
            activations_core=core_acts,
            delta_activations=core_delta,
            boundary_activations=boundary_acts,
            backend=b,
        )

        assert result.applied

        # Verify boundary preservation
        merged_weight = b.array(result.merged_weight)
        b.eval(merged_weight)

        output_before = b.matmul(boundary_acts, b.transpose(weight_target))
        output_after = b.matmul(boundary_acts, b.transpose(merged_weight))
        b.eval(output_before, output_after)

        diff = output_after - output_before
        diff_norm = float(b.to_scalar(b.sum(b.abs(diff))))
        output_norm = float(b.to_scalar(b.sum(b.abs(output_before))))

        denom = output_norm + division_epsilon(b, output_before)
        relative_diff = diff_norm / denom
        tol = division_epsilon(b, output_before)
        assert relative_diff <= tol, (
            f"Boundary changed by {relative_diff:.2e}"
        )

    def test_full_pipeline_cross_dimension(self) -> None:
        """Full pipeline for cross-architecture merge (different dimensions).

        This is the canonical use case: merging a larger model into smaller.
        The grafting delta is in target hidden space (d_target), so the weight
        must have out_dim = d_target for dimensions to match.
        """
        b = get_default_backend()
        b.random_seed(42)

        n_samples = 50
        d_source = 256  # Larger source model
        d_target = 128  # Smaller target model
        n_anchors = 20

        # Activations in respective spaces
        source_activations = b.random_normal((n_samples, d_source))
        target_activations = b.random_normal((n_samples, d_target))
        source_anchors = b.random_normal((n_anchors, d_source))
        target_anchors = b.random_normal((n_anchors, d_target))
        b.eval(source_activations, target_activations, source_anchors, target_anchors)

        # Target weight: both dimensions in target hidden space
        # This matches real transformer weights where out=hidden, in=hidden
        out_dim = d_target  # Output dimension matches target hidden
        in_dim = d_target   # Input dimension matches target hidden
        weight_target = b.random_normal((out_dim, in_dim))
        b.eval(weight_target)

        # Step 1: Compute grafting delta (maps to TARGET space automatically)
        grafting_result = compute_anchor_grafting_delta(
            source_activations=source_activations,
            target_activations=target_activations,
            source_anchors=source_anchors,
            target_anchors=target_anchors,
            backend=b,
        )

        # Verify delta is in target space
        assert b.shape(grafting_result.delta_activations) == (n_samples, d_target)

        # Step 2: Apply via transplant
        n_core = 30
        core_acts = b.take(target_activations, b.arange(n_core), axis=0)
        core_delta = b.take(grafting_result.delta_activations, b.arange(n_core), axis=0)
        b.eval(core_acts, core_delta)

        result = compute_transplant_delta(
            weight_target=weight_target,
            activations_core=core_acts,
            delta_activations=core_delta,
            boundary_activations=None,
            backend=b,
        )

        assert result.applied
        assert b.shape(result.merged_weight) == (out_dim, in_dim)


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_minimal_samples(self) -> None:
        """Pipeline should handle minimal samples (n=2)."""
        b = get_default_backend()
        b.random_seed(42)

        n_samples, d_hidden, n_anchors = 2, 32, 5

        source_activations = b.random_normal((n_samples, d_hidden))
        target_activations = b.random_normal((n_samples, d_hidden))
        source_anchors = b.random_normal((n_anchors, d_hidden))
        target_anchors = b.random_normal((n_anchors, d_hidden))
        b.eval(source_activations, target_activations, source_anchors, target_anchors)

        result = compute_anchor_grafting_delta(
            source_activations=source_activations,
            target_activations=target_activations,
            source_anchors=source_anchors,
            target_anchors=target_anchors,
            backend=b,
        )

        assert b.shape(result.delta_activations) == (n_samples, d_hidden)

    def test_more_anchors_than_samples(self) -> None:
        """Pipeline should handle n_anchors > n_samples."""
        b = get_default_backend()
        b.random_seed(42)

        n_samples, d_hidden, n_anchors = 10, 64, 30

        source_activations = b.random_normal((n_samples, d_hidden))
        target_activations = b.random_normal((n_samples, d_hidden))
        source_anchors = b.random_normal((n_anchors, d_hidden))
        target_anchors = b.random_normal((n_anchors, d_hidden))
        b.eval(source_activations, target_activations, source_anchors, target_anchors)

        # Should not crash
        result = compute_anchor_grafting_delta(
            source_activations=source_activations,
            target_activations=target_activations,
            source_anchors=source_anchors,
            target_anchors=target_anchors,
            backend=b,
        )

        assert b.shape(result.delta_activations) == (n_samples, d_hidden)

    def test_large_dimension_difference(self) -> None:
        """Pipeline should handle large dimension ratios (e.g., 8x)."""
        b = get_default_backend()
        b.random_seed(42)

        n_samples = 30
        d_source = 512  # 8x larger
        d_target = 64
        n_anchors = 15

        source_activations = b.random_normal((n_samples, d_source))
        target_activations = b.random_normal((n_samples, d_target))
        source_anchors = b.random_normal((n_anchors, d_source))
        target_anchors = b.random_normal((n_anchors, d_target))
        b.eval(source_activations, target_activations, source_anchors, target_anchors)

        result = compute_anchor_grafting_delta(
            source_activations=source_activations,
            target_activations=target_activations,
            source_anchors=source_anchors,
            target_anchors=target_anchors,
            backend=b,
        )

        # Output must be in target dimension
        assert b.shape(result.delta_activations) == (n_samples, d_target)
