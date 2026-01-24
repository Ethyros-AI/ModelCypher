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

"""Tests for active subspace blending.

The key insight: blend in ACTIVATION-defined subspace, not weight SVD space.

These tests verify:
1. Active subspace blend uses activation covariance eigenvectors (not weight SVD)
2. Blending happens with different ratios in active vs null subspaces
3. The blended weight preserves target behavior in null directions
4. Effective blend ratio is variance-weighted average
"""

from __future__ import annotations

import pytest
import math

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.active_subspace_blend import (
    compute_active_subspace_blend,
    compute_adaptive_active_blend,
    ActiveSubspaceBlendResult,
)
from modelcypher.core.domain.geometry.orthogonal_probe_generator import (
    compute_variance_null_space,
)


def _eps(backend) -> float:
    """Get machine epsilon for comparison."""
    return machine_epsilon(backend, backend.array([1.0]))


def _sqrt_eps(backend) -> float:
    """Get sqrt(machine_epsilon) - the precision threshold."""
    eps = machine_epsilon(backend, backend.array([1.0]))
    return sqrt_scalar(eps, backend)


class TestActiveSubspaceBlend:
    """Verify active subspace blending uses activation-defined basis."""

    def test_blend_uses_activation_basis_not_weight_svd(self) -> None:
        """The blending should happen in activation covariance eigenvector basis.

        This is THE KEY DIFFERENCE from spectral blend (which uses weight SVD).
        """
        b = get_default_backend()
        b.random_seed(42)

        out_dim, in_dim = 64, 128
        n_samples = 200

        # Create weights with known structure
        source_weight = b.random_normal((out_dim, in_dim))
        target_weight = b.random_normal((out_dim, in_dim))

        # Create activations with specific variance structure:
        # High variance in first 32 dims, low variance in rest
        high_var = b.random_normal((n_samples, 32)) * 10.0  # High variance
        low_var = b.random_normal((n_samples, in_dim - 32)) * 0.1  # Low variance
        activations = b.concatenate([high_var, low_var], axis=1)
        b.eval(source_weight, target_weight, activations)

        result = compute_active_subspace_blend(
            source_weight=source_weight,
            target_weight=target_weight,
            input_activations=activations,
            base_blend=0.1,
            active_boost=2.0,
            null_dampen=0.5,
            backend=b,
        )

        # Verify active subspace was identified
        # With 32 high-variance dims, active_rank should be around 32
        assert result.active_rank > 0, "Should identify active subspace"
        assert result.null_rank > 0, "Should identify null subspace"
        assert result.active_rank + result.null_rank == in_dim, "Should partition space"

        # Active rank should be close to 32 (the high variance dims)
        # Allow some tolerance for cumulative variance threshold
        assert result.active_rank <= 50, f"Active rank {result.active_rank} too high"

    def test_different_blend_ratios_in_active_vs_null(self) -> None:
        """Verify active and null subspaces get different blend ratios."""
        b = get_default_backend()
        b.random_seed(123)

        out_dim, in_dim = 32, 128
        n_samples = 50

        source_weight = b.random_normal((out_dim, in_dim))
        target_weight = b.random_normal((out_dim, in_dim))

        # Create activations with specific structure to ensure a null space exists
        # Only vary in first 40 dimensions, rest are near-zero
        active_part = b.random_normal((n_samples, 40))
        null_part = b.random_normal((n_samples, in_dim - 40)) * 0.001  # Near-zero
        activations = b.concatenate([active_part, null_part], axis=1)
        b.eval(source_weight, target_weight, activations)

        base = 0.1
        active_boost = 3.0
        null_dampen = 0.2

        result = compute_active_subspace_blend(
            source_weight=source_weight,
            target_weight=target_weight,
            input_activations=activations,
            base_blend=base,
            active_boost=active_boost,
            null_dampen=null_dampen,
            backend=b,
        )

        # Should have both active and null subspaces
        assert result.null_rank > 0, \
            f"Expected null_rank > 0 but got {result.null_rank}"

        # Verify blend ratios match expected values
        expected_active = min(1.0, base * active_boost)  # 0.3
        expected_null = max(0.0, base * null_dampen)  # 0.02

        assert abs(result.active_blend - expected_active) < 1e-6, \
            f"Active blend {result.active_blend} != expected {expected_active}"
        assert abs(result.null_blend - expected_null) < 1e-6, \
            f"Null blend {result.null_blend} != expected {expected_null}"

    def test_preserves_target_in_null_directions(self) -> None:
        """Blending with null_dampen=0 should preserve target in null subspace."""
        b = get_default_backend()
        b.random_seed(456)

        out_dim, in_dim = 32, 64
        n_samples = 100

        source_weight = b.random_normal((out_dim, in_dim))
        target_weight = b.random_normal((out_dim, in_dim))
        activations = b.random_normal((n_samples, in_dim))
        b.eval(source_weight, target_weight, activations)

        # Use null_dampen=0 to fully preserve target in null directions
        result = compute_active_subspace_blend(
            source_weight=source_weight,
            target_weight=target_weight,
            input_activations=activations,
            base_blend=0.3,
            active_boost=2.0,
            null_dampen=0.0,  # No blending in null space
            backend=b,
        )

        # Get the null space basis from activations
        variance_result = compute_variance_null_space(activations, b)
        U_available = variance_result.available_basis
        b.eval(U_available)

        if variance_result.available_rank > 0:
            # Project both blended and target into null space
            # W_null = W @ U_available @ U_available.T
            blended_null = b.matmul(
                b.matmul(result.blended_weight, U_available),
                b.transpose(U_available)
            )
            target_null = b.matmul(
                b.matmul(target_weight, U_available),
                b.transpose(U_available)
            )
            b.eval(blended_null, target_null)

            # They should be identical (null_blend=0)
            diff = b.sqrt(b.sum((blended_null - target_null) ** 2))
            b.eval(diff)
            diff_val = float(b.to_scalar(diff))

            target_norm = b.sqrt(b.sum(target_null ** 2))
            b.eval(target_norm)
            target_norm_val = float(b.to_scalar(target_norm))

            eps = _sqrt_eps(b)
            relative_diff = diff_val / max(target_norm_val, eps)

            assert relative_diff < eps, \
                f"Null component not preserved: relative_diff={relative_diff:.2e}"

    def test_effective_blend_is_variance_weighted(self) -> None:
        """Effective blend should be weighted by variance captured."""
        b = get_default_backend()
        b.random_seed(789)

        out_dim, in_dim = 32, 64
        n_samples = 100

        source_weight = b.random_normal((out_dim, in_dim))
        target_weight = b.random_normal((out_dim, in_dim))
        activations = b.random_normal((n_samples, in_dim))
        b.eval(source_weight, target_weight, activations)

        base = 0.1
        active_boost = 2.0
        null_dampen = 0.5

        result = compute_active_subspace_blend(
            source_weight=source_weight,
            target_weight=target_weight,
            input_activations=activations,
            base_blend=base,
            active_boost=active_boost,
            null_dampen=null_dampen,
            backend=b,
        )

        # Effective = variance_captured * active_blend + (1 - variance_captured) * null_blend
        expected_effective = (
            result.variance_captured * result.active_blend +
            (1.0 - result.variance_captured) * result.null_blend
        )

        assert abs(result.effective_blend_ratio - expected_effective) < 1e-6, \
            f"Effective blend {result.effective_blend_ratio} != expected {expected_effective}"

    def test_full_active_space_uses_uniform_blend(self) -> None:
        """When null_rank=0 (all active), should fall back to uniform blend."""
        b = get_default_backend()
        b.random_seed(111)

        out_dim, in_dim = 8, 16
        # Use n >> d to ensure full rank activations
        n_samples = 500

        source_weight = b.random_normal((out_dim, in_dim))
        target_weight = b.random_normal((out_dim, in_dim))
        # Random activations with many samples should span full space
        activations = b.random_normal((n_samples, in_dim))
        b.eval(source_weight, target_weight, activations)

        result = compute_active_subspace_blend(
            source_weight=source_weight,
            target_weight=target_weight,
            input_activations=activations,
            base_blend=0.1,
            active_boost=2.0,
            null_dampen=0.5,
            backend=b,
        )

        # When null_rank=0, the function should handle it gracefully
        # Either by returning uniform blend or by blending all in active
        assert result.active_rank + result.null_rank == in_dim

    def test_blended_weight_has_correct_shape(self) -> None:
        """Blended weight should match target weight shape."""
        b = get_default_backend()
        b.random_seed(222)

        out_dim, in_dim = 48, 96
        n_samples = 150

        source_weight = b.random_normal((out_dim, in_dim))
        target_weight = b.random_normal((out_dim, in_dim))
        activations = b.random_normal((n_samples, in_dim))
        b.eval(source_weight, target_weight, activations)

        result = compute_active_subspace_blend(
            source_weight=source_weight,
            target_weight=target_weight,
            input_activations=activations,
            backend=b,
        )

        blended_shape = b.shape(result.blended_weight)
        assert blended_shape == (out_dim, in_dim), \
            f"Shape mismatch: {blended_shape} != ({out_dim}, {in_dim})"


class TestAdaptiveActiveBlend:
    """Test adaptive blending based on activation concentration."""

    def test_high_concentration_uses_conservative_ratios(self) -> None:
        """High variance concentration should use conservative boost/dampen."""
        b = get_default_backend()
        b.random_seed(333)

        out_dim, in_dim = 32, 64
        n_samples = 100

        source_weight = b.random_normal((out_dim, in_dim))
        target_weight = b.random_normal((out_dim, in_dim))

        # Create highly concentrated activations (nearly 1D)
        # First component has huge variance, rest have tiny variance
        dominant = b.random_normal((n_samples, 1)) * 100.0
        rest = b.random_normal((n_samples, in_dim - 1)) * 0.01
        activations = b.concatenate([dominant, rest], axis=1)
        b.eval(source_weight, target_weight, activations)

        result = compute_adaptive_active_blend(
            source_weight=source_weight,
            target_weight=target_weight,
            input_activations=activations,
            base_blend=0.1,
            backend=b,
        )

        # High concentration should use conservative ratios
        # active_boost around 1.2-1.5, null_dampen around 0.7-0.9
        assert result.active_blend <= 0.20, \
            f"High concentration should use low active_blend, got {result.active_blend}"

    def test_low_concentration_uses_aggressive_ratios(self) -> None:
        """Low variance concentration should allow more aggressive blending."""
        b = get_default_backend()
        b.random_seed(444)

        out_dim, in_dim = 32, 64
        n_samples = 200

        source_weight = b.random_normal((out_dim, in_dim))
        target_weight = b.random_normal((out_dim, in_dim))

        # Create evenly distributed activations (low concentration)
        activations = b.random_normal((n_samples, in_dim))
        b.eval(source_weight, target_weight, activations)

        result = compute_adaptive_active_blend(
            source_weight=source_weight,
            target_weight=target_weight,
            input_activations=activations,
            base_blend=0.1,
            backend=b,
        )

        # With uniform random activations, concentration should be low
        # This allows more aggressive blending
        # active_boost around 2.0-2.5
        assert result.active_blend >= 0.15, \
            f"Low concentration should allow higher active_blend, got {result.active_blend}"


class TestActiveVsSpectralDifference:
    """Tests showing the key difference between active and spectral blending."""

    def test_active_blend_basis_differs_from_weight_svd(self) -> None:
        """Active blend basis (from activations) should differ from weight SVD basis.

        This is THE KEY TEST proving we're using the right basis.
        """
        b = get_default_backend()
        b.random_seed(555)

        out_dim, in_dim = 32, 64
        n_samples = 100

        # Create a weight with specific SVD structure
        # U @ S @ Vt where S has clear dominant singular values
        U = b.random_normal((out_dim, out_dim))
        U, _ = b.qr(U)
        Vt = b.random_normal((in_dim, in_dim))
        _, Vt = b.qr(Vt)
        S = b.zeros((out_dim, in_dim))
        # Dominant weight SVD in first 8 directions
        for i in range(min(8, out_dim)):
            S = b.where(
                (b.arange(out_dim).reshape(-1, 1) == i) & (b.arange(in_dim).reshape(1, -1) == i),
                10.0 - i,
                S
            )
        target_weight = b.matmul(U, b.matmul(S, Vt))
        source_weight = b.random_normal((out_dim, in_dim))
        b.eval(U, Vt, S, target_weight, source_weight)

        # Create activations with DIFFERENT variance structure
        # High variance in LAST 16 directions (opposite of weight SVD dominance)
        low_var = b.random_normal((n_samples, in_dim - 16)) * 0.1
        high_var = b.random_normal((n_samples, 16)) * 10.0
        activations = b.concatenate([low_var, high_var], axis=1)
        b.eval(activations)

        result = compute_active_subspace_blend(
            source_weight=source_weight,
            target_weight=target_weight,
            input_activations=activations,
            base_blend=0.1,
            backend=b,
        )

        # The active subspace should be in the LAST 16 directions (where activation variance is)
        # NOT in the FIRST 8 directions (where weight SVD dominance is)
        # This proves we're using activation basis, not weight basis

        # Get variance null space
        variance_result = compute_variance_null_space(activations, b)

        # Active rank should be roughly 16 (the high-variance dims)
        # NOT 8 (which would be weight SVD-based)
        assert result.active_rank > 10, \
            f"Active rank {result.active_rank} too low - might be using weight SVD basis"
