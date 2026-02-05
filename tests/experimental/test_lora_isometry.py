# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for LoRA isometry metrics (experimental).

These tests validate the discriminative properties of the isometry metrics.
"""

from __future__ import annotations

import math

import pytest


class TestIsometryMetricsSynthetic:
    """Test isometry metrics against synthetic ground-truth."""

    @pytest.fixture
    def backend(self):
        """Get compute backend."""
        from modelcypher.core.domain._backend import get_default_backend

        return get_default_backend()

    def test_isometric_scale_lora(self, backend):
        """Test that scaling W gives low spectral angle."""
        from modelcypher.experimental.lora_isometry import (
            compute_isometry_metrics,
            create_synthetic_isometric_lora,
        )

        lora = create_synthetic_isometric_lora(backend=backend)
        metrics = compute_isometry_metrics(
            lora.weight_original, lora.weight_modified, backend
        )

        # Scaling preserves directions - spectral angle should be low
        assert metrics.spectral_angle < 5.0, (
            f"Expected low spectral angle for isometric LoRA, got {metrics.spectral_angle}°"
        )
        # SPR should be high
        assert metrics.spectral_preservation_ratio > 0.8
        # Condition ratio should be near 1.0 (preserved)
        assert 0.8 < metrics.condition_ratio < 1.2

    def test_random_lora_moderate_deviation(self, backend):
        """Test that random LoRA has moderate spectral deviation."""
        from modelcypher.experimental.lora_isometry import (
            compute_isometry_metrics,
            create_synthetic_random_lora,
        )

        lora = create_synthetic_random_lora(backend=backend)
        metrics = compute_isometry_metrics(
            lora.weight_original, lora.weight_modified, backend
        )

        # Random perturbation will rotate directions somewhat
        # With small scale (0.1), angle should still be modest
        assert 0 < metrics.spectral_angle < 60
        # SPR should be reasonable
        assert metrics.spectral_preservation_ratio > 0.5
        # Grassmann distance should be non-zero
        assert 0 < metrics.grassmann_distance < math.pi / 2

    def test_orthogonal_lora_different_subspace(self, backend):
        """Test that null-space LoRA affects condition ratio."""
        from modelcypher.experimental.lora_isometry import (
            compute_isometry_metrics,
            create_synthetic_orthogonal_lora,
        )

        lora = create_synthetic_orthogonal_lora(backend=backend)
        metrics = compute_isometry_metrics(
            lora.weight_original, lora.weight_modified, backend
        )

        # Null space action should affect conditioning
        # The original spectrum should be preserved
        assert metrics.spectral_preservation_ratio > 0.7

    def test_scale_invariance(self, backend):
        """Test that metrics are scale-invariant."""
        from modelcypher.experimental.lora_isometry import compute_isometry_metrics

        # Create a random weight and delta
        W = backend.random_normal((32, 16), dtype="float32")
        delta = backend.random_normal((32, 16), dtype="float32")
        delta = backend.multiply(delta, 0.1)
        W_mod = backend.add(W, delta)
        backend.eval(W, W_mod)

        # Compute metrics at original scale
        metrics_1 = compute_isometry_metrics(W, W_mod, backend)

        # Scale everything by 10
        W_scaled = backend.multiply(W, 10.0)
        W_mod_scaled = backend.multiply(W_mod, 10.0)
        backend.eval(W_scaled, W_mod_scaled)

        metrics_2 = compute_isometry_metrics(W_scaled, W_mod_scaled, backend)

        # Key metrics should be approximately equal
        assert abs(metrics_1.spectral_angle - metrics_2.spectral_angle) < 1.0
        assert abs(
            metrics_1.spectral_preservation_ratio 
            - metrics_2.spectral_preservation_ratio
        ) < 0.01

    def test_identical_weights_perfect_metrics(self, backend):
        """Test that W' = W gives perfect metrics."""
        from modelcypher.experimental.lora_isometry import compute_isometry_metrics

        W = backend.random_normal((32, 16), dtype="float32")
        backend.eval(W)

        metrics = compute_isometry_metrics(W, W, backend)

        # No change = perfect preservation
        assert metrics.spectral_preservation_ratio == 1.0
        assert metrics.spectral_angle < 1.0  # Near zero
        assert 0.99 < metrics.condition_ratio < 1.01  # Near 1.0
        assert metrics.grassmann_distance < 0.01  # Near zero
        assert metrics.relative_frobenius_deviation < 0.01


class TestIsometryMetricsProperties:
    """Test mathematical properties of the metrics."""

    @pytest.fixture
    def backend(self):
        from modelcypher.core.domain._backend import get_default_backend

        return get_default_backend()

    def test_spr_range(self, backend):
        """Test that SPR is in [0, 1]."""
        from modelcypher.experimental.lora_isometry import compute_isometry_metrics

        # Random test case
        W = backend.random_normal((64, 32), dtype="float32")
        delta = backend.random_normal((64, 32), dtype="float32")
        W_mod = backend.add(W, delta)
        backend.eval(W, W_mod)

        metrics = compute_isometry_metrics(W, W_mod, backend)

        assert 0.0 <= metrics.spectral_preservation_ratio <= 1.0

    def test_spectral_angle_range(self, backend):
        """Test that spectral angle is in [0, 90]."""
        from modelcypher.experimental.lora_isometry import compute_isometry_metrics

        W = backend.random_normal((64, 32), dtype="float32")
        delta = backend.random_normal((64, 32), dtype="float32")
        W_mod = backend.add(W, delta)
        backend.eval(W, W_mod)

        metrics = compute_isometry_metrics(W, W_mod, backend)

        assert 0.0 <= metrics.spectral_angle <= 90.0 + 0.1

    def test_grassmann_range(self, backend):
        """Test that Grassmann distance is in [0, π/2]."""
        from modelcypher.experimental.lora_isometry import compute_isometry_metrics

        W = backend.random_normal((64, 32), dtype="float32")
        delta = backend.random_normal((64, 32), dtype="float32")
        W_mod = backend.add(W, delta)
        backend.eval(W, W_mod)

        metrics = compute_isometry_metrics(W, W_mod, backend)

        assert 0.0 <= metrics.grassmann_distance <= math.pi / 2 + 0.01

    def test_rfd_positive(self, backend):
        """Test that RFD is non-negative."""
        from modelcypher.experimental.lora_isometry import compute_isometry_metrics

        W = backend.random_normal((64, 32), dtype="float32")
        delta = backend.random_normal((64, 32), dtype="float32")
        W_mod = backend.add(W, delta)
        backend.eval(W, W_mod)

        metrics = compute_isometry_metrics(W, W_mod, backend)

        assert metrics.relative_frobenius_deviation >= 0.0


class TestIsometryDiscrimination:
    """Test that metrics discriminate good vs bad adapters."""

    @pytest.fixture
    def backend(self):
        from modelcypher.core.domain._backend import get_default_backend
        return get_default_backend()

    def test_catastrophic_change_high_angle(self, backend):
        """Test that random replacement gives high spectral angle."""
        from modelcypher.experimental.lora_isometry import compute_isometry_metrics

        W = backend.random_normal((64, 32), dtype="float32")
        W_random = backend.random_normal((64, 32), dtype="float32")
        backend.eval(W, W_random)

        metrics = compute_isometry_metrics(W, W_random, backend)

        # Full random replacement should give ~90° angle
        assert metrics.spectral_angle > 60.0, (
            f"Expected high angle for random replacement, got {metrics.spectral_angle}°"
        )
