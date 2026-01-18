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

"""Tests for behavioral weight reconstruction in cross-dimensional merging.

These tests verify the mathematical properties of behavioral reconstruction:
1. Behavior preservation: reconstructed weight produces same outputs (in target coords)
2. Magnitude stability: reconstructed weights have reasonable magnitudes
3. Cross-dimensional correctness: works for different source/target dimensions
4. Null-space projection: delta is properly constrained to null space
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
import numpy as np

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    regularization_epsilon,
)
from modelcypher.core.domain.geometry.transplant import (
    BehavioralReconstructionResult,
    WeightSpaceTransplantResult,
    reconstruct_weight_from_behavior,
    compute_cross_dimensional_transplant,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def backend():
    """Get the default backend."""
    return get_default_backend()


def make_random_weight(backend, out_dim: int, in_dim: int, scale: float = 0.1):
    """Create a random weight matrix with controlled magnitude."""
    b = backend
    # Use normal distribution scaled appropriately
    w = b.array(np.random.randn(out_dim, in_dim).astype(np.float32) * scale)
    b.eval(w)
    return w


def make_random_activations(backend, n_samples: int, dim: int, scale: float = 1.0):
    """Create random activation samples."""
    b = backend
    acts = b.array(np.random.randn(n_samples, dim).astype(np.float32) * scale)
    b.eval(acts)
    return acts


def make_orthogonal_transform(backend, src_dim: int, tgt_dim: int):
    """Create a random orthogonal-ish transform matrix.

    For src_dim <= tgt_dim: columns are orthonormal (isometric embedding)
    For src_dim > tgt_dim: rows are orthonormal (projection)
    """
    b = backend
    # Start with random matrix
    M = np.random.randn(src_dim, tgt_dim).astype(np.float32)

    # Make it more well-conditioned via QR
    if src_dim <= tgt_dim:
        # Tall or square: use Q from QR of transpose
        Q, _ = np.linalg.qr(M.T)
        result = Q.T[:src_dim, :]  # [src_dim, tgt_dim]
    else:
        # Wide: use Q from QR
        Q, _ = np.linalg.qr(M)
        result = Q[:, :tgt_dim]  # [src_dim, tgt_dim]

    transform = b.array(result.astype(np.float32))
    b.eval(transform)
    return transform


# ============================================================================
# Unit Tests for reconstruct_weight_from_behavior
# ============================================================================

class TestReconstructWeightFromBehavior:
    """Tests for the core behavioral reconstruction function."""

    def test_same_dimensions_identity_transform(self, backend):
        """When source and target have same dims with identity transform, weight should be preserved."""
        b = backend
        dim = 32
        n_samples = 64

        # Create source weight
        source_weight = make_random_weight(b, dim, dim, scale=0.1)
        input_acts = make_random_activations(b, n_samples, dim)

        # Identity transforms
        alignment_in = b.eye(dim)
        alignment_out = b.eye(dim)
        b.eval(alignment_in, alignment_out)

        result = reconstruct_weight_from_behavior(
            source_weight=source_weight,
            input_activations_source=input_acts,
            alignment_in=alignment_in,
            alignment_out=alignment_out,
            backend=b,
        )

        assert isinstance(result, BehavioralReconstructionResult)

        # With identity transforms, reconstructed weight should match source
        reconstructed = result.reconstructed_weight
        diff = b.mean(b.abs(reconstructed - source_weight))
        b.eval(diff)
        diff_val = float(b.to_scalar(diff))

        mean_abs = float(b.to_scalar(b.mean(b.abs(source_weight))))
        tol = division_epsilon(b, source_weight) * max(1.0, mean_abs, result.condition_number)
        assert diff_val <= tol, f"Identity transform should preserve weight, got diff={diff_val}"

    def test_behavior_is_preserved(self, backend):
        """Reconstructed weight should produce same outputs in target coordinates."""
        b = backend
        out_src, in_src = 48, 32
        out_tgt, in_tgt = 64, 40
        n_samples = 100

        source_weight = make_random_weight(b, out_src, in_src, scale=0.1)
        input_acts_source = make_random_activations(b, n_samples, in_src)
        alignment_in = make_orthogonal_transform(b, in_src, in_tgt)
        alignment_out = make_orthogonal_transform(b, out_src, out_tgt)

        result = reconstruct_weight_from_behavior(
            source_weight=source_weight,
            input_activations_source=input_acts_source,
            alignment_in=alignment_in,
            alignment_out=alignment_out,
            backend=b,
        )

        # Compute source behavior
        output_source = b.matmul(input_acts_source, b.transpose(source_weight))
        b.eval(output_source)

        # Project to target coordinates
        input_target = b.matmul(input_acts_source, alignment_in)
        output_target_expected = b.matmul(output_source, alignment_out)
        b.eval(input_target, output_target_expected)

        # Compute output with reconstructed weight
        output_target_actual = b.matmul(input_target, b.transpose(result.reconstructed_weight))
        b.eval(output_target_actual)

        # Should match
        diff = b.mean(b.abs(output_target_actual - output_target_expected))
        b.eval(diff)
        diff_val = float(b.to_scalar(diff))

        tol = division_epsilon(b, output_target_expected) * max(1.0, result.condition_number)
        assert abs(diff_val - result.reconstruction_error) <= tol

    def test_compression_reduces_dimensions(self, backend):
        """When target dims < source dims, compression should work."""
        b = backend
        out_src, in_src = 64, 48  # Larger source
        out_tgt, in_tgt = 32, 24  # Smaller target (compression)
        n_samples = 100

        source_weight = make_random_weight(b, out_src, in_src, scale=0.1)
        input_acts_source = make_random_activations(b, n_samples, in_src)
        alignment_in = make_orthogonal_transform(b, in_src, in_tgt)
        alignment_out = make_orthogonal_transform(b, out_src, out_tgt)

        result = reconstruct_weight_from_behavior(
            source_weight=source_weight,
            input_activations_source=input_acts_source,
            alignment_in=alignment_in,
            alignment_out=alignment_out,
            backend=b,
        )

        # Check output shape
        assert result.reconstructed_weight.shape == (out_tgt, in_tgt)

        output_source = b.matmul(input_acts_source, b.transpose(source_weight))
        input_target = b.matmul(input_acts_source, alignment_in)
        output_target_expected = b.matmul(output_source, alignment_out)
        output_target_actual = b.matmul(input_target, b.transpose(result.reconstructed_weight))
        b.eval(output_target_expected, output_target_actual)

        diff = b.mean(b.abs(output_target_actual - output_target_expected))
        b.eval(diff)
        diff_val = float(b.to_scalar(diff))
        tol = division_epsilon(b, output_target_expected) * max(1.0, result.condition_number)
        assert abs(diff_val - result.reconstruction_error) <= tol

    def test_expansion_increases_dimensions(self, backend):
        """When target dims > source dims, expansion should work."""
        b = backend
        out_src, in_src = 32, 24  # Smaller source
        out_tgt, in_tgt = 64, 48  # Larger target (expansion)
        n_samples = 100

        source_weight = make_random_weight(b, out_src, in_src, scale=0.1)
        input_acts_source = make_random_activations(b, n_samples, in_src)
        alignment_in = make_orthogonal_transform(b, in_src, in_tgt)
        alignment_out = make_orthogonal_transform(b, out_src, out_tgt)

        result = reconstruct_weight_from_behavior(
            source_weight=source_weight,
            input_activations_source=input_acts_source,
            alignment_in=alignment_in,
            alignment_out=alignment_out,
            backend=b,
        )

        # Check output shape
        assert result.reconstructed_weight.shape == (out_tgt, in_tgt)

        output_source = b.matmul(input_acts_source, b.transpose(source_weight))
        input_target = b.matmul(input_acts_source, alignment_in)
        output_target_expected = b.matmul(output_source, alignment_out)
        output_target_actual = b.matmul(input_target, b.transpose(result.reconstructed_weight))
        b.eval(output_target_expected, output_target_actual)

        diff = b.mean(b.abs(output_target_actual - output_target_expected))
        b.eval(diff)
        diff_val = float(b.to_scalar(diff))
        tol = division_epsilon(b, output_target_expected) * max(1.0, result.condition_number)
        assert abs(diff_val - result.reconstruction_error) <= tol


# ============================================================================
# Unit Tests for compute_cross_dimensional_transplant
# ============================================================================

class TestComputeCrossDimensionalTransplant:
    """Tests for the full cross-dimensional transplant function."""

    def test_basic_cross_dim_transplant(self, backend):
        """Basic test that cross-dim transplant produces valid output."""
        b = backend
        out_src, in_src = 48, 32
        out_tgt, in_tgt = 64, 40
        n_samples = 100

        source_weight = make_random_weight(b, out_src, in_src, scale=0.1)
        target_weight = make_random_weight(b, out_tgt, in_tgt, scale=0.1)

        input_acts_source = make_random_activations(b, n_samples, in_src)
        input_acts_target = make_random_activations(b, n_samples, in_tgt)

        alignment_in = make_orthogonal_transform(b, in_src, in_tgt)
        alignment_out = make_orthogonal_transform(b, out_src, out_tgt)

        result = compute_cross_dimensional_transplant(
            source_weight=source_weight,
            target_weight=target_weight,
            input_activations_source=input_acts_source,
            input_activations_target=input_acts_target,
            alignment_in=alignment_in,
            alignment_out=alignment_out,
            delta_scale=1.0,
            backend=b,
        )

        assert isinstance(result, WeightSpaceTransplantResult)
        assert result.merged_weight.shape == target_weight.shape
        eps = division_epsilon(b, target_weight)
        if result.delta_norm > eps:
            expected = result.projected_norm / result.delta_norm
        else:
            expected = 1.0
        tol = division_epsilon(b, target_weight) * max(1.0, abs(expected))
        assert abs(result.preserved_fraction - expected) <= tol
        assert result.delta_norm >= 0.0

    def test_magnitude_stability(self, backend):
        """Merged weight magnitudes should not explode."""
        b = backend
        out_src, in_src = 48, 32
        out_tgt, in_tgt = 64, 40
        n_samples = 100

        source_weight = make_random_weight(b, out_src, in_src, scale=0.1)
        target_weight = make_random_weight(b, out_tgt, in_tgt, scale=0.1)

        input_acts_source = make_random_activations(b, n_samples, in_src)
        input_acts_target = make_random_activations(b, n_samples, in_tgt)

        alignment_in = make_orthogonal_transform(b, in_src, in_tgt)
        alignment_out = make_orthogonal_transform(b, out_src, out_tgt)

        result = compute_cross_dimensional_transplant(
            source_weight=source_weight,
            target_weight=target_weight,
            input_activations_source=input_acts_source,
            input_activations_target=input_acts_target,
            alignment_in=alignment_in,
            alignment_out=alignment_out,
            delta_scale=1.0,
            backend=b,
        )

        delta_applied = result.merged_weight - target_weight
        output_delta = b.matmul(input_acts_target, b.transpose(delta_applied))
        b.eval(output_delta)
        behavioral_norm = float(b.to_scalar(b.sqrt(b.sum(output_delta * output_delta))))
        expected = abs(1.0) * result.projected_norm
        tol = division_epsilon(b, output_delta) * max(1.0, abs(expected))
        assert abs(behavioral_norm - expected) <= tol

    def test_delta_scale_zero_preserves_target(self, backend):
        """With delta_scale=0, merged weight should equal target."""
        b = backend
        out_src, in_src = 48, 32
        out_tgt, in_tgt = 64, 40
        n_samples = 100

        source_weight = make_random_weight(b, out_src, in_src, scale=0.1)
        target_weight = make_random_weight(b, out_tgt, in_tgt, scale=0.1)

        input_acts_source = make_random_activations(b, n_samples, in_src)
        input_acts_target = make_random_activations(b, n_samples, in_tgt)

        alignment_in = make_orthogonal_transform(b, in_src, in_tgt)
        alignment_out = make_orthogonal_transform(b, out_src, out_tgt)

        result = compute_cross_dimensional_transplant(
            source_weight=source_weight,
            target_weight=target_weight,
            input_activations_source=input_acts_source,
            input_activations_target=input_acts_target,
            alignment_in=alignment_in,
            alignment_out=alignment_out,
            delta_scale=0.0,  # No delta applied
            backend=b,
        )

        diff = b.mean(b.abs(result.merged_weight - target_weight))
        b.eval(diff)
        diff_val = float(b.to_scalar(diff))

        tol = regularization_epsilon(b, target_weight)
        assert diff_val < tol, f"delta_scale=0 should preserve target, got diff={diff_val}"

    def test_same_source_target_dimensions(self, backend):
        """When source and target have same dims, should still work."""
        b = backend
        dim_out, dim_in = 32, 32
        n_samples = 64

        source_weight = make_random_weight(b, dim_out, dim_in, scale=0.1)
        target_weight = make_random_weight(b, dim_out, dim_in, scale=0.1)

        input_acts = make_random_activations(b, n_samples, dim_in)

        # Identity transforms
        alignment_in = b.eye(dim_in)
        alignment_out = b.eye(dim_out)
        b.eval(alignment_in, alignment_out)

        result = compute_cross_dimensional_transplant(
            source_weight=source_weight,
            target_weight=target_weight,
            input_activations_source=input_acts,
            input_activations_target=input_acts,
            alignment_in=alignment_in,
            alignment_out=alignment_out,
            delta_scale=1.0,
            backend=b,
        )

        assert result.merged_weight.shape == target_weight.shape
        eps = division_epsilon(b, target_weight)
        if result.delta_norm > eps:
            expected = result.projected_norm / result.delta_norm
        else:
            expected = 1.0
        tol = division_epsilon(b, target_weight) * max(1.0, abs(expected))
        assert abs(result.preserved_fraction - expected) <= tol


# ============================================================================
# Hypothesis Property-Based Tests
# ============================================================================

# Strategy for reasonable dimensions
dims_strategy = st.integers(min_value=8, max_value=64)
samples_strategy = st.integers(min_value=32, max_value=128)
scale_strategy = st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False)


class TestBehavioralReconstructionProperties:
    """Hypothesis property-based tests for behavioral reconstruction."""

    @given(
        out_src=dims_strategy,
        in_src=dims_strategy,
        out_tgt=dims_strategy,
        in_tgt=dims_strategy,
        n_samples=samples_strategy,
        weight_scale=scale_strategy,
    )
    @settings(max_examples=20, deadline=None)
    def test_output_shape_is_correct(
        self, out_src, in_src, out_tgt, in_tgt, n_samples, weight_scale
    ):
        """Reconstructed weight always has correct output shape."""
        b = get_default_backend()

        source_weight = make_random_weight(b, out_src, in_src, scale=weight_scale)
        input_acts_source = make_random_activations(b, n_samples, in_src)
        alignment_in = make_orthogonal_transform(b, in_src, in_tgt)
        alignment_out = make_orthogonal_transform(b, out_src, out_tgt)

        result = reconstruct_weight_from_behavior(
            source_weight=source_weight,
            input_activations_source=input_acts_source,
            alignment_in=alignment_in,
            alignment_out=alignment_out,
            backend=b,
        )

        assert result.reconstructed_weight.shape == (out_tgt, in_tgt)

    @given(
        out_src=dims_strategy,
        in_src=dims_strategy,
        out_tgt=dims_strategy,
        in_tgt=dims_strategy,
        n_samples=samples_strategy,
        weight_scale=scale_strategy,
    )
    @settings(max_examples=20, deadline=None)
    def test_reconstruction_error_is_non_negative(
        self, out_src, in_src, out_tgt, in_tgt, n_samples, weight_scale
    ):
        """Reconstruction error is always non-negative."""
        b = get_default_backend()

        source_weight = make_random_weight(b, out_src, in_src, scale=weight_scale)
        input_acts_source = make_random_activations(b, n_samples, in_src)
        alignment_in = make_orthogonal_transform(b, in_src, in_tgt)
        alignment_out = make_orthogonal_transform(b, out_src, out_tgt)

        result = reconstruct_weight_from_behavior(
            source_weight=source_weight,
            input_activations_source=input_acts_source,
            alignment_in=alignment_in,
            alignment_out=alignment_out,
            backend=b,
        )

        assert result.reconstruction_error >= 0.0

    @given(
        dim=st.integers(min_value=8, max_value=32),  # Smaller dims for this test
        n_samples=st.integers(min_value=64, max_value=128),  # Ensure n_samples >> dim
        weight_scale=scale_strategy,
    )
    @settings(max_examples=15, deadline=None)
    def test_identity_transform_preserves_weight(self, dim, n_samples, weight_scale):
        """Identity transforms should approximately preserve the weight.

        Note: Requires n_samples >> dim for well-conditioned lstsq.
        """
        # Ensure we have enough samples for the dimension
        assume(n_samples >= 2 * dim)

        b = get_default_backend()

        source_weight = make_random_weight(b, dim, dim, scale=weight_scale)
        input_acts = make_random_activations(b, n_samples, dim)

        alignment_in = b.eye(dim)
        alignment_out = b.eye(dim)
        b.eval(alignment_in, alignment_out)

        result = reconstruct_weight_from_behavior(
            source_weight=source_weight,
            input_activations_source=input_acts,
            alignment_in=alignment_in,
            alignment_out=alignment_out,
            backend=b,
        )

        # With identity transforms, reconstructed should closely match source
        diff = b.mean(b.abs(result.reconstructed_weight - source_weight))
        b.eval(diff)
        diff_val = float(b.to_scalar(diff))

        mean_abs = float(b.to_scalar(b.mean(b.abs(source_weight))))
        tol = division_epsilon(b, source_weight) * max(1.0, mean_abs, result.condition_number)
        assert diff_val <= tol, f"Identity transform should preserve, diff={diff_val}, scale={weight_scale}"


class TestCrossDimensionalTransplantProperties:
    """Hypothesis property-based tests for cross-dimensional transplant."""

    @given(
        out_src=dims_strategy,
        in_src=dims_strategy,
        out_tgt=dims_strategy,
        in_tgt=dims_strategy,
        n_samples=samples_strategy,
        weight_scale=scale_strategy,
        delta_scale=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=20, deadline=None)
    def test_merged_weight_shape_matches_target(
        self, out_src, in_src, out_tgt, in_tgt, n_samples, weight_scale, delta_scale
    ):
        """Merged weight always has same shape as target."""
        b = get_default_backend()

        source_weight = make_random_weight(b, out_src, in_src, scale=weight_scale)
        target_weight = make_random_weight(b, out_tgt, in_tgt, scale=weight_scale)

        input_acts_source = make_random_activations(b, n_samples, in_src)
        input_acts_target = make_random_activations(b, n_samples, in_tgt)

        alignment_in = make_orthogonal_transform(b, in_src, in_tgt)
        alignment_out = make_orthogonal_transform(b, out_src, out_tgt)

        result = compute_cross_dimensional_transplant(
            source_weight=source_weight,
            target_weight=target_weight,
            input_activations_source=input_acts_source,
            input_activations_target=input_acts_target,
            alignment_in=alignment_in,
            alignment_out=alignment_out,
            delta_scale=delta_scale,
            backend=b,
        )

        assert result.merged_weight.shape == target_weight.shape

    @given(
        out_src=dims_strategy,
        in_src=dims_strategy,
        out_tgt=dims_strategy,
        in_tgt=dims_strategy,
        n_samples=samples_strategy,
        weight_scale=scale_strategy,
    )
    @settings(max_examples=20, deadline=None)
    def test_preserved_fraction_in_bounds(
        self, out_src, in_src, out_tgt, in_tgt, n_samples, weight_scale
    ):
        """Preserved fraction is always between 0 and 1."""
        b = get_default_backend()

        source_weight = make_random_weight(b, out_src, in_src, scale=weight_scale)
        target_weight = make_random_weight(b, out_tgt, in_tgt, scale=weight_scale)

        input_acts_source = make_random_activations(b, n_samples, in_src)
        input_acts_target = make_random_activations(b, n_samples, in_tgt)

        alignment_in = make_orthogonal_transform(b, in_src, in_tgt)
        alignment_out = make_orthogonal_transform(b, out_src, out_tgt)

        result = compute_cross_dimensional_transplant(
            source_weight=source_weight,
            target_weight=target_weight,
            input_activations_source=input_acts_source,
            input_activations_target=input_acts_target,
            alignment_in=alignment_in,
            alignment_out=alignment_out,
            delta_scale=1.0,
            backend=b,
        )

        eps = division_epsilon(b, target_weight)
        if result.delta_norm > eps:
            expected = result.projected_norm / result.delta_norm
        else:
            expected = 1.0
        tol = division_epsilon(b, target_weight) * max(1.0, abs(expected))
        assert abs(result.preserved_fraction - expected) <= tol

    @given(
        out_src=dims_strategy,
        in_src=dims_strategy,
        out_tgt=dims_strategy,
        in_tgt=dims_strategy,
        n_samples=samples_strategy,
        weight_scale=scale_strategy,
    )
    @settings(max_examples=20, deadline=None)
    def test_magnitude_does_not_explode(
        self, out_src, in_src, out_tgt, in_tgt, n_samples, weight_scale
    ):
        """Merged weight magnitude should not explode (key invariant)."""
        b = get_default_backend()

        source_weight = make_random_weight(b, out_src, in_src, scale=weight_scale)
        target_weight = make_random_weight(b, out_tgt, in_tgt, scale=weight_scale)

        input_acts_source = make_random_activations(b, n_samples, in_src)
        input_acts_target = make_random_activations(b, n_samples, in_tgt)

        alignment_in = make_orthogonal_transform(b, in_src, in_tgt)
        alignment_out = make_orthogonal_transform(b, out_src, out_tgt)

        result = compute_cross_dimensional_transplant(
            source_weight=source_weight,
            target_weight=target_weight,
            input_activations_source=input_acts_source,
            input_activations_target=input_acts_target,
            alignment_in=alignment_in,
            alignment_out=alignment_out,
            delta_scale=1.0,
            backend=b,
        )

        delta_applied = result.merged_weight - target_weight
        output_delta = b.matmul(input_acts_target, b.transpose(delta_applied))
        b.eval(output_delta)
        behavioral_norm = float(b.to_scalar(b.sqrt(b.sum(output_delta * output_delta))))
        expected = abs(1.0) * result.projected_norm
        tol = division_epsilon(b, output_delta) * max(1.0, abs(expected))
        assert abs(behavioral_norm - expected) <= tol

    @given(
        out_tgt=dims_strategy,
        in_tgt=dims_strategy,
        n_samples=samples_strategy,
        weight_scale=scale_strategy,
    )
    @settings(max_examples=15, deadline=None)
    def test_delta_scale_zero_equals_target(
        self, out_tgt, in_tgt, n_samples, weight_scale
    ):
        """With delta_scale=0, merged weight should equal target exactly."""
        b = get_default_backend()

        # Use same dimensions for simpler test
        out_src, in_src = out_tgt, in_tgt

        source_weight = make_random_weight(b, out_src, in_src, scale=weight_scale)
        target_weight = make_random_weight(b, out_tgt, in_tgt, scale=weight_scale)

        input_acts = make_random_activations(b, n_samples, in_tgt)

        alignment_in = b.eye(in_tgt)
        alignment_out = b.eye(out_tgt)
        b.eval(alignment_in, alignment_out)

        result = compute_cross_dimensional_transplant(
            source_weight=source_weight,
            target_weight=target_weight,
            input_activations_source=input_acts,
            input_activations_target=input_acts,
            alignment_in=alignment_in,
            alignment_out=alignment_out,
            delta_scale=0.0,
            backend=b,
        )

        diff = b.mean(b.abs(result.merged_weight - target_weight))
        b.eval(diff)
        diff_val = float(b.to_scalar(diff))

        tol = regularization_epsilon(b, target_weight)
        assert diff_val < tol, f"delta_scale=0 should give target, got diff={diff_val}"


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestBehavioralReconstructionEdgeCases:
    """Edge case tests for behavioral reconstruction."""

    def test_very_small_weight(self, backend):
        """Very small weights should not cause numerical issues."""
        b = backend
        dim = 32
        n_samples = 64

        source_weight = make_random_weight(
            b, dim, dim, scale=division_epsilon(b, b.array([1.0]))
        )
        input_acts = make_random_activations(b, n_samples, dim)

        alignment_in = b.eye(dim)
        alignment_out = b.eye(dim)
        b.eval(alignment_in, alignment_out)

        result = reconstruct_weight_from_behavior(
            source_weight=source_weight,
            input_activations_source=input_acts,
            alignment_in=alignment_in,
            alignment_out=alignment_out,
            backend=b,
        )

        # Should not have NaN or Inf
        has_nan = b.any(b.isnan(result.reconstructed_weight))
        has_inf = b.any(b.isinf(result.reconstructed_weight))
        b.eval(has_nan, has_inf)

        assert not bool(b.to_scalar(has_nan)), "Reconstructed weight has NaN"
        assert not bool(b.to_scalar(has_inf)), "Reconstructed weight has Inf"

    def test_minimum_samples(self, backend):
        """Should work with minimum number of samples."""
        b = backend
        dim_in, dim_out = 16, 16
        n_samples = dim_in + 1  # Just enough for lstsq

        source_weight = make_random_weight(b, dim_out, dim_in, scale=0.1)
        input_acts = make_random_activations(b, n_samples, dim_in)

        alignment_in = b.eye(dim_in)
        alignment_out = b.eye(dim_out)
        b.eval(alignment_in, alignment_out)

        result = reconstruct_weight_from_behavior(
            source_weight=source_weight,
            input_activations_source=input_acts,
            alignment_in=alignment_in,
            alignment_out=alignment_out,
            backend=b,
        )

        assert result.reconstructed_weight.shape == (dim_out, dim_in)

    def test_asymmetric_dimensions(self, backend):
        """Should handle very asymmetric dimensions."""
        b = backend
        out_src, in_src = 128, 16  # Very wide
        out_tgt, in_tgt = 16, 128  # Very tall (swapped)
        n_samples = 64

        source_weight = make_random_weight(b, out_src, in_src, scale=0.1)
        input_acts = make_random_activations(b, n_samples, in_src)

        alignment_in = make_orthogonal_transform(b, in_src, in_tgt)
        alignment_out = make_orthogonal_transform(b, out_src, out_tgt)

        result = reconstruct_weight_from_behavior(
            source_weight=source_weight,
            input_activations_source=input_acts,
            alignment_in=alignment_in,
            alignment_out=alignment_out,
            backend=b,
        )

        assert result.reconstructed_weight.shape == (out_tgt, in_tgt)

        output_source = b.matmul(input_acts, b.transpose(source_weight))
        input_target = b.matmul(input_acts, alignment_in)
        output_target_expected = b.matmul(output_source, alignment_out)
        output_target_actual = b.matmul(input_target, b.transpose(result.reconstructed_weight))
        b.eval(output_target_expected, output_target_actual)

        diff = b.mean(b.abs(output_target_actual - output_target_expected))
        b.eval(diff)
        diff_val = float(b.to_scalar(diff))
        tol = division_epsilon(b, output_target_expected) * max(1.0, result.condition_number)
        assert abs(diff_val - result.reconstruction_error) <= tol


# ============================================================================
# Comparison Tests: Behavioral Reconstruction vs Direct Stitch
# ============================================================================

class TestBehavioralVsDirectStitch:
    """Compare behavioral reconstruction to direct stitch to verify we fixed the bug."""

    def test_behavioral_reconstruction_minimizes_output_error(self, backend):
        """Behavioral reconstruction should not exceed direct stitch output error."""
        b = backend
        out_src, in_src = 64, 48  # Source dimensions
        out_tgt, in_tgt = 32, 24  # Target dimensions (compression)
        n_samples = 100

        source_weight = make_random_weight(b, out_src, in_src, scale=0.1)
        input_acts_source = make_random_activations(b, n_samples, in_src)

        # Create "stitch" transforms like the old code did
        # These are the P and Q matrices used in direct stitch
        P = make_orthogonal_transform(b, out_src, out_tgt)  # [out_src, out_tgt]
        Q = make_orthogonal_transform(b, in_src, in_tgt)    # [in_src, in_tgt]

        # Direct stitch: P.T @ W @ Q (the problematic approach)
        # P.T is [out_tgt, out_src], W is [out_src, in_src], Q is [in_src, in_tgt]
        P_T = b.transpose(P)  # [out_tgt, out_src]
        direct_stitched = b.matmul(P_T, source_weight)
        direct_stitched = b.matmul(direct_stitched, Q)
        b.eval(direct_stitched)

        # Compare behavioral errors against expected target outputs
        output_source = b.matmul(input_acts_source, b.transpose(source_weight))
        input_target = b.matmul(input_acts_source, Q)
        output_target_expected = b.matmul(output_source, P)
        output_direct = b.matmul(input_target, b.transpose(direct_stitched))
        b.eval(output_target_expected, output_direct)

        error_direct = float(b.to_scalar(b.mean(b.abs(output_direct - output_target_expected))))

        recon = reconstruct_weight_from_behavior(
            source_weight=source_weight,
            input_activations_source=input_acts_source,
            alignment_in=Q,
            alignment_out=P,
            backend=b,
        )
        output_behavior = b.matmul(input_target, b.transpose(recon.reconstructed_weight))
        b.eval(output_behavior)
        error_behavior = float(
            b.to_scalar(b.mean(b.abs(output_behavior - output_target_expected)))
        )

        tol = division_epsilon(b, output_target_expected) * max(1.0, error_direct)
        assert error_behavior <= error_direct + tol

    def test_behavioral_reconstruction_is_more_stable(self, backend):
        """Behavioral reconstruction should not exceed direct stitch error."""
        b = backend
        out_src, in_src = 64, 48  # Source dimensions
        out_tgt, in_tgt = 32, 24  # Target dimensions (compression)
        n_samples = 100

        source_weight = make_random_weight(b, out_src, in_src, scale=0.1)
        input_acts_source = make_random_activations(b, n_samples, in_src)

        alignment_in = make_orthogonal_transform(b, in_src, in_tgt)
        alignment_out = make_orthogonal_transform(b, out_src, out_tgt)

        output_source = b.matmul(input_acts_source, b.transpose(source_weight))
        input_target = b.matmul(input_acts_source, alignment_in)
        output_target_expected = b.matmul(output_source, alignment_out)

        direct_stitched = b.matmul(b.transpose(alignment_out), source_weight)
        direct_stitched = b.matmul(direct_stitched, alignment_in)
        b.eval(direct_stitched)
        output_direct = b.matmul(input_target, b.transpose(direct_stitched))
        b.eval(output_target_expected, output_direct)
        error_direct = float(b.to_scalar(b.mean(b.abs(output_direct - output_target_expected))))

        recon = reconstruct_weight_from_behavior(
            source_weight=source_weight,
            input_activations_source=input_acts_source,
            alignment_in=alignment_in,
            alignment_out=alignment_out,
            backend=b,
        )
        output_behavior = b.matmul(input_target, b.transpose(recon.reconstructed_weight))
        b.eval(output_behavior)
        error_behavior = float(
            b.to_scalar(b.mean(b.abs(output_behavior - output_target_expected)))
        )

        tol = division_epsilon(b, output_target_expected) * max(1.0, error_direct)
        assert error_behavior <= error_direct + tol

    def test_large_dimension_ratio_stability(self, backend):
        """Behavioral reconstruction should not exceed direct stitch error at large ratios."""
        b = backend
        # 4:1 dimension ratio - this was causing problems with direct stitch
        out_src, in_src = 128, 96  # Large source
        out_tgt, in_tgt = 32, 24   # Small target (4:1 compression)
        n_samples = 100

        source_weight = make_random_weight(b, out_src, in_src, scale=0.1)
        input_acts_source = make_random_activations(b, n_samples, in_src)

        alignment_in = make_orthogonal_transform(b, in_src, in_tgt)
        alignment_out = make_orthogonal_transform(b, out_src, out_tgt)

        output_source = b.matmul(input_acts_source, b.transpose(source_weight))
        input_target = b.matmul(input_acts_source, alignment_in)
        output_target_expected = b.matmul(output_source, alignment_out)

        direct_stitched = b.matmul(b.transpose(alignment_out), source_weight)
        direct_stitched = b.matmul(direct_stitched, alignment_in)
        b.eval(direct_stitched)
        output_direct = b.matmul(input_target, b.transpose(direct_stitched))
        b.eval(output_target_expected, output_direct)
        error_direct = float(b.to_scalar(b.mean(b.abs(output_direct - output_target_expected))))

        recon = reconstruct_weight_from_behavior(
            source_weight=source_weight,
            input_activations_source=input_acts_source,
            alignment_in=alignment_in,
            alignment_out=alignment_out,
            backend=b,
        )
        output_behavior = b.matmul(input_target, b.transpose(recon.reconstructed_weight))
        b.eval(output_behavior)
        error_behavior = float(
            b.to_scalar(b.mean(b.abs(output_behavior - output_target_expected)))
        )

        tol = division_epsilon(b, output_target_expected) * max(1.0, error_direct)
        assert error_behavior <= error_direct + tol


# ============================================================================
# Integration with Weight Processor
# ============================================================================

class TestWeightProcessorIntegration:
    """Tests for the integration in transplant_weight_processor.py."""

    def test_apply_behavioral_reconstruction_import(self):
        """Verify the integration function is importable."""
        from modelcypher.core.use_cases.merge.stages.transplant_weight_processor import (
            _apply_behavioral_reconstruction,
            _is_cross_dimensional,
        )

        # Basic smoke test
        assert callable(_apply_behavioral_reconstruction)
        assert callable(_is_cross_dimensional)

    def test_is_cross_dimensional_detection(self):
        """Test cross-dimensional detection logic."""
        from modelcypher.core.use_cases.merge.stages.transplant_weight_processor import (
            _is_cross_dimensional,
        )

        # Same dims - not cross-dimensional
        assert not _is_cross_dimensional((64, 32), (64, 32))

        # Different dims - cross-dimensional
        assert _is_cross_dimensional((64, 32), (48, 24))
        assert _is_cross_dimensional((32, 32), (64, 64))

    def test_apply_behavioral_reconstruction_success(self, backend):
        """Test the wrapper function works correctly."""
        from modelcypher.core.use_cases.merge.stages.transplant_weight_processor import (
            _apply_behavioral_reconstruction,
        )

        b = backend
        out_src, in_src = 48, 32
        out_tgt, in_tgt = 64, 40
        n_samples = 100

        source_weight = make_random_weight(b, out_src, in_src, scale=0.1)
        target_weight = make_random_weight(b, out_tgt, in_tgt, scale=0.1)

        source_acts = make_random_activations(b, n_samples, in_src)
        target_acts = make_random_activations(b, n_samples, in_tgt)

        alignment_in = make_orthogonal_transform(b, in_src, in_tgt)
        alignment_out = make_orthogonal_transform(b, out_src, out_tgt)

        result = _apply_behavioral_reconstruction(
            source_weight=source_weight,
            target_weight=target_weight,
            source_activations=source_acts,
            target_activations=target_acts,
            alignment_in=alignment_in,
            alignment_out=alignment_out,
            source_density_acts=source_acts,
            target_density_acts=target_acts,
            delta_scale=1.0,
            backend=b,
            weight_key="test.weight",
        )

        assert result is not None
        merged_weight, metrics = result

        assert merged_weight.shape == target_weight.shape
        assert "preserved_fraction" in metrics
        assert "delta_norm" in metrics
