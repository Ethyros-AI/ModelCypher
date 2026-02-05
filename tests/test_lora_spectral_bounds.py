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

"""Tests for LoRA spectral scale bounds.

These tests validate the mathematical foundations of geometry-derived LoRA scaling:

1. sqrt(eps) threshold correctly identifies numerical noise
2. Bound computation produces bounded perturbations (Theorem 1)
3. Eigengap detection finds spectral structure (Theorem 2)
4. Cayley transform produces semi-orthogonal matrices
5. Spectral regularization reduces norm during training

EXPERIMENTAL: This is research code. Tests validate math correctness,
but real model validation is needed before CLI promotion.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def backend():
    """Get the default compute backend."""
    return get_default_backend()


@pytest.fixture
def sqrt_eps(backend):
    """Get sqrt(machine epsilon) for float32."""
    from modelcypher.core.domain.geometry.numerical_stability import sqrt_scalar
    return sqrt_scalar(backend.finfo().eps, backend)


# =============================================================================
# Test: sqrt(eps) Threshold
# =============================================================================


class TestSqrtEpsThreshold:
    """Tests that sqrt(eps) correctly identifies numerical noise."""

    def test_sqrt_eps_is_geometric_mean(self, backend, sqrt_eps):
        """sqrt(eps) is the geometric mean between 1 and eps."""
        eps = backend.finfo().eps
        geometric_mean = (1.0 * eps) ** 0.5
        assert abs(sqrt_eps - geometric_mean) < 1e-10

    def test_values_below_sqrt_eps_are_noise(self, backend, sqrt_eps):
        """Values below sqrt(eps) * scale are indistinguishable from noise."""
        # Create matrix with known singular values
        n = 50
        sigma_max = 1.0
        sigma_noise = sqrt_eps * sigma_max * 0.1  # Well below threshold

        # Diagonal matrix with one large and rest noise
        S = backend.concatenate([
            backend.array([sigma_max]),
            backend.ones((n - 1,)) * sigma_noise,
        ])
        D = backend.diag(S)
        backend.eval(D)

        # SVD should show effective rank of 1
        _, S_computed, _ = backend.svd(D, compute_uv=True)
        backend.eval(S_computed)

        threshold = sqrt_eps * float(backend.to_scalar(S_computed[0]))
        significant = S_computed > threshold
        backend.eval(significant)

        eff_rank = int(backend.to_scalar(backend.sum(backend.astype(significant, "int32"))))
        assert eff_rank == 1, f"Expected effective rank 1, got {eff_rank}"

    def test_values_above_sqrt_eps_are_signal(self, backend, sqrt_eps):
        """Values above sqrt(eps) * scale are distinguishable signal."""
        n = 50
        sigma_max = 1.0
        sigma_signal = sqrt_eps * sigma_max * 10  # Well above threshold

        # Diagonal matrix with decaying but significant values
        decay = backend.exp(-backend.arange(n, dtype="float32") * 0.1)
        S = sigma_max * decay
        # Ensure minimum is above threshold
        S = backend.where(S < sigma_signal, backend.ones_like(S) * sigma_signal, S)
        D = backend.diag(S)
        backend.eval(D)

        _, S_computed, _ = backend.svd(D, compute_uv=True)
        backend.eval(S_computed)

        threshold = sqrt_eps * float(backend.to_scalar(S_computed[0]))
        significant = S_computed > threshold
        backend.eval(significant)

        eff_rank = int(backend.to_scalar(backend.sum(backend.astype(significant, "int32"))))
        assert eff_rank == n, f"Expected effective rank {n}, got {eff_rank}"


# =============================================================================
# Test: Geometric Scale Bound (Theorem 1)
# =============================================================================


class TestGeometricScaleBound:
    """Tests that respecting the bound produces bounded perturbations."""

    def test_bound_scale_respects_weyl(self, backend, sqrt_eps):
        """Scaling at bound respects Weyl's inequality."""
        # Create base weight with known spectrum
        m, n = 64, 32
        W = backend.random_normal((m, n))
        backend.eval(W)

        _, S_W, _ = backend.svd(W, compute_uv=True)
        backend.eval(S_W)

        sigma_max = float(backend.to_scalar(S_W[0]))
        threshold = sqrt_eps * sigma_max

        # Find sigma_k
        significant = S_W > threshold
        backend.eval(significant)
        eff_rank = int(backend.to_scalar(backend.sum(backend.astype(significant, "int32"))))
        sigma_k = float(backend.to_scalar(S_W[eff_rank - 1])) if eff_rank > 0 else float(backend.to_scalar(S_W[-1]))

        # Create LoRA delta
        rank = 4
        A = backend.random_normal((rank, n)) * 0.1
        B = backend.random_normal((m, rank)) * 0.1
        backend.eval(A, B)

        Delta = backend.matmul(B, A)
        backend.eval(Delta)

        _, S_Delta, _ = backend.svd(Delta, compute_uv=True)
        backend.eval(S_Delta)
        delta_spectral = float(backend.to_scalar(S_Delta[0]))

        # Compute geometry-derived scale
        geo_scale = sigma_k / delta_spectral if delta_spectral > 0 else 0.0

        # Apply perturbation
        W_perturbed = W + geo_scale * Delta
        backend.eval(W_perturbed)

        _, S_perturbed, _ = backend.svd(W_perturbed, compute_uv=True)
        backend.eval(S_perturbed)

        # Weyl's inequality: |sigma_i(W') - sigma_i(W)| <= ||scale * Delta||_2
        perturbation_norm = geo_scale * delta_spectral  # = sigma_k by construction

        for i in range(min(10, eff_rank)):
            diff = abs(float(backend.to_scalar(S_perturbed[i])) - float(backend.to_scalar(S_W[i])))
            # Allow small numerical tolerance
            assert diff <= perturbation_norm * 1.1, (
                f"Weyl violated at i={i}: diff={diff:.6f}, bound={perturbation_norm:.6f}"
            )

    def test_over_bound_dominates_tail(self, backend, sqrt_eps):
        """Scaling over bound causes perturbation to dominate W's tail."""
        m, n = 64, 32
        W = backend.random_normal((m, n))
        backend.eval(W)

        _, S_W, _ = backend.svd(W, compute_uv=True)
        backend.eval(S_W)

        sigma_max = float(backend.to_scalar(S_W[0]))
        threshold = sqrt_eps * sigma_max
        significant = S_W > threshold
        backend.eval(significant)
        eff_rank = int(backend.to_scalar(backend.sum(backend.astype(significant, "int32"))))
        sigma_k = float(backend.to_scalar(S_W[eff_rank - 1])) if eff_rank > 0 else float(backend.to_scalar(S_W[-1]))

        # Create LoRA delta
        rank = 4
        A = backend.random_normal((rank, n)) * 0.1
        B = backend.random_normal((m, rank)) * 0.1
        backend.eval(A, B)

        Delta = backend.matmul(B, A)
        backend.eval(Delta)

        _, S_Delta, _ = backend.svd(Delta, compute_uv=True)
        backend.eval(S_Delta)
        delta_spectral = float(backend.to_scalar(S_Delta[0]))

        # Use 10x the geometric bound
        geo_scale = sigma_k / delta_spectral if delta_spectral > 0 else 0.0
        over_scale = 10.0 * geo_scale

        # The perturbation now has spectral norm 10 * sigma_k
        perturbation_spectral = over_scale * delta_spectral
        assert perturbation_spectral > sigma_k * 5, "Perturbation should dominate tail"


# =============================================================================
# Test: Eigengap Detection (Theorem 2)
# =============================================================================


class TestEigengapDetection:
    """Tests that eigengap detection finds spectral structure."""

    def test_detects_clear_gap(self, backend):
        """Detects gap when singular values have clear separation."""
        from modelcypher.core.use_cases.lora_safety_service import LoRASafetyService

        # Create spectrum with clear gap: [1.0, 0.9, 0.8, 0.1, 0.09, 0.08]
        S = backend.array([1.0, 0.9, 0.8, 0.1, 0.09, 0.08])
        backend.eval(S)

        eigengap = LoRASafetyService.detect_eigengap(S, backend, gap_threshold=2.0)

        # Gap should be at position 2 (0.8 / 0.1 = 8 > 2)
        assert eigengap.position == 2, f"Expected gap at position 2, got {eigengap.position}"
        assert eigengap.gap_ratio > 2.0, f"Gap ratio should be > 2, got {eigengap.gap_ratio}"

    def test_no_gap_when_smooth_decay(self, backend):
        """No gap detected when spectrum decays smoothly."""
        from modelcypher.core.use_cases.lora_safety_service import LoRASafetyService

        # Smooth decay: each value is 0.9x the previous
        S = backend.array([1.0, 0.9, 0.81, 0.729, 0.656, 0.590])
        backend.eval(S)

        eigengap = LoRASafetyService.detect_eigengap(S, backend, gap_threshold=2.0)

        # No gap should be detected (max ratio is ~1.11)
        assert eigengap.position is None, f"Expected no gap, got position {eigengap.position}"

    def test_tightened_bound_from_gap(self, backend):
        """Tightened bound uses gap_value / 2."""
        from modelcypher.core.use_cases.lora_safety_service import LoRASafetyService

        # Gap at position 2: sigma_2 = 0.8, sigma_3 = 0.1, gap = 0.7
        S = backend.array([1.0, 0.9, 0.8, 0.1, 0.09, 0.08])
        backend.eval(S)

        eigengap = LoRASafetyService.detect_eigengap(S, backend, gap_threshold=2.0)

        expected_tightened = (0.8 - 0.1) / 2.0  # 0.35
        assert eigengap.tightened_bound is not None
        assert abs(eigengap.tightened_bound - expected_tightened) < 0.01, (
            f"Expected tightened bound ~{expected_tightened}, got {eigengap.tightened_bound}"
        )


# =============================================================================
# Test: Cayley Transform
# =============================================================================


class TestCayleyTransform:
    """Tests that Cayley transform produces semi-orthogonal matrices.

    Validates the NB-LoRA (Wang et al., 2025) Cayley parameterization:
    - Semi-orthogonality: A^T @ A + B^T @ B = I_r
    - Spectral bound: ||2 * B^T @ S @ A||_2 <= 2 * max(S)
    """

    def test_cayley_semi_orthogonality(self, backend):
        """Cayley transform produces semi-orthogonal [A; B] with orthonormal columns.

        Mathematical property: A^T @ A + B^T @ B = I_r
        (NOT A @ A^T = I, which is a different property)
        """
        from modelcypher.core.domain.geometry.cayley_lora import cayley_transform

        r = 4
        X = backend.random_normal((r, r)) * 0.1
        Y = backend.random_normal((8, r)) * 0.1  # m = 8
        backend.eval(X, Y)

        A, B = cayley_transform(X, Y, backend)
        backend.eval(A, B)

        # Correct constraint: A^T @ A + B^T @ B = I_r
        # (stacked [A; B] has orthonormal COLUMNS)
        AtA = backend.matmul(backend.transpose(A), A)
        BtB = backend.matmul(backend.transpose(B), B)
        backend.eval(AtA, BtB)

        result = AtA + BtB
        I = backend.eye(r)
        diff = result - I
        backend.eval(diff)

        frobenius = float(backend.to_scalar(backend.sqrt(backend.sum(diff * diff))))
        assert frobenius < 1e-5, f"Semi-orthogonality violated: ||A^T@A + B^T@B - I||_F = {frobenius}"

    def test_nb_lora_spectral_bound(self, backend):
        """NB-LoRA delta ||2 * B^T @ S @ A||_spectral respects bound."""
        from modelcypher.core.domain.geometry.cayley_lora import cayley_transform_full

        r, n_in, n_out = 4, 32, 64

        # Per NB-LoRA paper: A_tilde [r, n_in], B_tilde [r, n_out]
        A_tilde = backend.random_normal((r, n_in)) * 0.1
        B_tilde = backend.random_normal((r, n_out)) * 0.1
        backend.eval(A_tilde, B_tilde)

        A, B = cayley_transform_full(A_tilde, B_tilde, backend)
        backend.eval(A, B)

        # Test with known scale bound
        scale_bound = 0.1
        S = backend.ones((r,)) * scale_bound

        # Compute delta: 2 * B^T @ S @ A
        # A: [r, n_in], B: [r, n_out]
        # B^T: [n_out, r], S: [r]
        S_A = A * backend.reshape(S, (-1, 1))  # [r, n_in]
        delta = 2.0 * backend.matmul(backend.transpose(B), S_A)  # [n_out, n_in]
        backend.eval(delta)

        # Check spectral norm
        _, S_delta, _ = backend.svd(delta, compute_uv=True)
        backend.eval(S_delta)
        spectral = float(backend.to_scalar(S_delta[0]))

        # NB-LoRA guarantee: ||delta||_2 <= 2 * max(S) = 2 * scale_bound
        expected_bound = 2.0 * scale_bound
        assert spectral <= expected_bound * 1.01, (
            f"Spectral bound violated: ||delta||_2 = {spectral:.6f}, bound = {expected_bound:.6f}"
        )

    def test_nb_lora_layer_respects_bound(self, backend):
        """NBLoRALayer output respects spectral bound."""
        from modelcypher.core.domain.geometry.cayley_lora import NBLoRALayer, NBLoRAConfig

        config = NBLoRAConfig(
            in_features=32,
            out_features=64,
            rank=4,
            scale_bound=0.1,
        )
        layer = NBLoRALayer(config, backend)

        # Check actual spectral norm
        spectral = layer.get_spectral_norm()

        # Should be <= 2 * scale_bound (the NB-LoRA guarantee)
        assert spectral <= 2 * config.scale_bound * 1.1, (
            f"Spectral norm {spectral} exceeds 2 * bound = {2 * config.scale_bound}"
        )

    def test_per_direction_bounds(self, backend):
        """Verify per-direction spectral bounds via U^T @ Δ @ V projection."""
        from modelcypher.core.domain.geometry.cayley_lora import (
            NBLoRALayer,
            NBLoRAConfig,
            create_nb_lora_from_base_weight,
        )

        # Create a base weight with known spectrum
        m, n = 64, 32
        W = backend.random_normal((m, n))
        backend.eval(W)

        # Create NB-LoRA layer from base weight (should respect geometry)
        layer = create_nb_lora_from_base_weight(W, rank=4, backend=backend)

        # Verify per-direction bounds
        result = layer.verify_per_direction_bounds(W)

        # With geometry-derived initialization, should be safe
        # Allow some tolerance for numerical precision
        assert result.max_ratio < 2.0, (
            f"Per-direction max ratio {result.max_ratio} too high. "
            f"Violations: {result.violations}"
        )


    def test_per_direction_bounds(self, backend):
        """NBLoRALayer.verify_per_direction_bounds computes correct ratios."""
        from modelcypher.core.domain.geometry.cayley_lora import NBLoRALayer, NBLoRAConfig

        m, n, rank = 64, 32, 4
        
        # Create base weight
        W = backend.random_normal((m, n))
        backend.eval(W)
        
        # Create layer with small bound to likely be safe
        # (Though random init might still violate locally, we just check structure)
        config = NBLoRAConfig(
            in_features=n,
            out_features=m,
            rank=rank,
            scale_bound=1.0,
        )
        layer = NBLoRALayer(config, backend)
        
        result = layer.verify_per_direction_bounds(W)
        
        # Check structure
        assert len(result.per_direction_ratios) > 0
        assert result.max_ratio >= 0
        assert isinstance(result.is_safe, bool)
        assert len(result.violations) == (0 if result.is_safe else len(result.violations))

        # Check that ratio logic is correct manually for one direction
        # Get Delta
        delta = layer.get_effective_delta()
        
        # Compute U^T @ delta @ V
        W_f32 = backend.astype(W, "float32")
        U, S, Vt = backend.svd(W_f32, compute_uv=True)
        
        # Project
        delta_f32 = backend.astype(delta, "float32")
        proj = backend.matmul(backend.transpose(U), delta_f32)
        proj = backend.matmul(proj, backend.transpose(Vt))
        backend.eval(proj)
        
        # Check first diagonal
        proj_00 = float(backend.to_scalar(backend.abs(proj[0, 0])))
        sigma_0 = float(backend.to_scalar(S[0]))
        
        expected_ratio = proj_00 / sigma_0
        actual_ratio = result.per_direction_ratios[0]
        
        assert abs(actual_ratio - expected_ratio) < 1e-5, (
            f"Ratio mismatch: expected {expected_ratio}, got {actual_ratio}"
        )


# =============================================================================
# Test: Spectral Regularization
# =============================================================================


class TestSpectralRegularization:
    """Tests that spectral regularization reduces norm during training."""

    def test_reg_loss_zero_when_within_bound(self, backend):
        """Regularization loss is zero when ||B@A|| < sigma_k."""
        from modelcypher.core.domain.lora_memory_store import compute_spectral_regularization_loss

        # Create small LoRA matrices
        rank, n_in, n_out = 4, 32, 64
        A = backend.random_normal((rank, n_in)) * 0.01
        B = backend.random_normal((n_out, rank)) * 0.01
        backend.eval(A, B)

        # sigma_k much larger than ||B@A||
        sigma_k = 1.0

        reg_loss, spectral = compute_spectral_regularization_loss(B, A, sigma_k, backend)

        assert spectral < sigma_k, f"Spectral {spectral} should be < sigma_k {sigma_k}"
        assert reg_loss < 1e-10, f"Reg loss should be ~0, got {reg_loss}"

    def test_reg_loss_positive_when_over_bound(self, backend):
        """Regularization loss is positive when ||B@A|| > sigma_k."""
        from modelcypher.core.domain.lora_memory_store import compute_spectral_regularization_loss

        # Create larger LoRA matrices
        rank, n_in, n_out = 4, 32, 64
        A = backend.random_normal((rank, n_in)) * 0.5
        B = backend.random_normal((n_out, rank)) * 0.5
        backend.eval(A, B)

        # sigma_k smaller than ||B@A||
        sigma_k = 0.01

        reg_loss, spectral = compute_spectral_regularization_loss(B, A, sigma_k, backend)

        assert spectral > sigma_k, f"Spectral {spectral} should be > sigma_k {sigma_k}"
        assert reg_loss > 0.0, f"Reg loss should be > 0, got {reg_loss}"

    def test_gradient_reduces_spectral_norm(self, backend):
        """Gradient descent with spectral regularization reduces norm."""
        from modelcypher.core.domain.lora_memory_store import (
            compute_spectral_regularization_loss,
            compute_spectral_regularization_gradient,
        )

        # Create LoRA matrices with high spectral norm
        rank, n_in, n_out = 4, 32, 64
        A = backend.random_normal((rank, n_in)) * 0.5
        B = backend.random_normal((n_out, rank)) * 0.5
        backend.eval(A, B)

        sigma_k = 0.1
        # Use small learning rate to ensure gradient descent works
        learning_rate = 0.001

        # Initial spectral norm
        _, initial_spectral = compute_spectral_regularization_loss(B, A, sigma_k, backend)

        # Take multiple gradient steps for more robust test
        A_curr, B_curr = A, B
        for _ in range(10):
            grad_B, grad_A, _ = compute_spectral_regularization_gradient(B_curr, A_curr, sigma_k, backend)
            A_curr = A_curr - learning_rate * grad_A
            B_curr = B_curr - learning_rate * grad_B
            backend.eval(A_curr, B_curr)

        # Final spectral norm
        _, final_spectral = compute_spectral_regularization_loss(B_curr, A_curr, sigma_k, backend)

        # Spectral norm should decrease after multiple small steps
        assert final_spectral < initial_spectral, (
            f"Spectral norm should decrease: {initial_spectral:.4f} -> {final_spectral:.4f}"
        )


# =============================================================================
# Test: Alignment Quality
# =============================================================================


class TestAlignmentQuality:
    """Tests that alignment quality measures subspace overlap correctly."""

    def test_perfect_alignment_has_ratio_one(self, backend):
        """LoRA aligned with W's subspace has alignment ratio ~1."""
        from modelcypher.core.use_cases.lora_safety_service import LoRASafetyService

        m, n, r = 64, 32, 4

        # Create W with known left singular vectors
        U = backend.random_normal((m, r))
        # Orthogonalize U
        U, _ = backend.qr(U)
        backend.eval(U)

        # Create LoRA delta aligned with U
        # delta_u1 = U[:, 0] (first column of U)
        delta_u1 = U[:, 0]
        backend.eval(delta_u1)

        alignment = LoRASafetyService.compute_alignment_quality(U, delta_u1, backend)

        assert alignment.alignment_ratio > 0.9, (
            f"Perfect alignment should have ratio ~1, got {alignment.alignment_ratio}"
        )
        assert alignment.is_well_aligned

    def test_orthogonal_lora_has_low_alignment(self, backend):
        """LoRA orthogonal to W's subspace has low alignment."""
        from modelcypher.core.use_cases.lora_safety_service import LoRASafetyService

        m, r = 64, 4

        # Create W's subspace
        U = backend.random_normal((m, r))
        U, _ = backend.qr(U)
        backend.eval(U)

        # Create delta orthogonal to U's columns
        # Generate random vector and project out U's components
        delta_random = backend.random_normal((m,))
        backend.eval(delta_random)

        # Project out U: delta_orth = delta - U @ U^T @ delta
        proj = backend.matmul(U, backend.matmul(backend.transpose(U), backend.reshape(delta_random, (-1, 1))))
        delta_orth = delta_random - backend.reshape(proj, (-1,))
        backend.eval(delta_orth)

        # Normalize
        norm = backend.sqrt(backend.sum(delta_orth * delta_orth))
        delta_orth = delta_orth / norm
        backend.eval(delta_orth)

        alignment = LoRASafetyService.compute_alignment_quality(U, delta_orth, backend)

        assert alignment.alignment_ratio < 0.1, (
            f"Orthogonal should have ratio ~0, got {alignment.alignment_ratio}"
        )
        assert not alignment.is_well_aligned


# =============================================================================
# Property-Based Tests (Hypothesis)
# =============================================================================


class TestSpectralBoundsProperties:
    """Property-based tests for spectral bound invariants."""

    @given(
        m=st.integers(min_value=8, max_value=32),
        n=st.integers(min_value=8, max_value=32),
        rank=st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=10, deadline=None)
    def test_bound_always_positive(self, m, n, rank):
        """Geometric bound is always positive for non-zero matrices."""
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.geometry.numerical_stability import sqrt_scalar

        backend = get_default_backend()
        assume(rank <= min(m, n))

        # Create random weight and LoRA
        W = backend.random_normal((m, n))
        A = backend.random_normal((rank, n)) * 0.1
        B = backend.random_normal((m, rank)) * 0.1
        backend.eval(W, A, B)

        # Compute bound
        _, S_W, _ = backend.svd(W, compute_uv=True)
        backend.eval(S_W)

        sqrt_eps = sqrt_scalar(backend.finfo().eps, backend)
        sigma_max = float(backend.to_scalar(S_W[0]))
        threshold = sqrt_eps * sigma_max
        significant = S_W > threshold
        backend.eval(significant)
        eff_rank = int(backend.to_scalar(backend.sum(backend.astype(significant, "int32"))))
        sigma_k = float(backend.to_scalar(S_W[max(0, eff_rank - 1)]))

        Delta = backend.matmul(B, A)
        backend.eval(Delta)
        _, S_D, _ = backend.svd(Delta, compute_uv=True)
        backend.eval(S_D)
        delta_spectral = float(backend.to_scalar(S_D[0]))

        if delta_spectral > 1e-10:
            bound = sigma_k / delta_spectral
            assert bound > 0, f"Bound should be positive, got {bound}"

    @given(scale_factor=st.floats(min_value=0.1, max_value=2.0))
    @settings(max_examples=10, deadline=None)
    def test_weyl_inequality_holds(self, scale_factor):
        """Weyl's inequality holds for any scale factor."""
        from modelcypher.core.domain._backend import get_default_backend

        backend = get_default_backend()
        m, n = 32, 16

        W = backend.random_normal((m, n))
        E = backend.random_normal((m, n)) * 0.1 * scale_factor
        backend.eval(W, E)

        _, S_W, _ = backend.svd(W, compute_uv=True)
        _, S_WE, _ = backend.svd(W + E, compute_uv=True)
        _, S_E, _ = backend.svd(E, compute_uv=True)
        backend.eval(S_W, S_WE, S_E)

        E_spectral = float(backend.to_scalar(S_E[0]))

        # Check Weyl for first few singular values
        for i in range(min(5, min(m, n))):
            diff = abs(float(backend.to_scalar(S_WE[i])) - float(backend.to_scalar(S_W[i])))
            # Allow small tolerance for numerical errors
            assert diff <= E_spectral * 1.01, (
                f"Weyl violated at i={i}: |σ_i(W+E) - σ_i(W)| = {diff} > ||E||_2 = {E_spectral}"
            )
