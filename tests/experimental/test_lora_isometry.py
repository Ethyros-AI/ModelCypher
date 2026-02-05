# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for LoRA isometry metrics (experimental).

These tests validate the mathematical properties of the isometry metrics
using synthetic ground-truth data.
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
        """Test that scaling W gives perfect overlap and high SPR."""
        from modelcypher.experimental.lora_isometry import (
            compute_isometry_metrics,
            create_synthetic_isometric_lora,
        )

        lora = create_synthetic_isometric_lora(backend=backend)
        metrics = compute_isometry_metrics(
            lora.weight_original, lora.weight_modified, backend
        )

        # Scaling preserves subspace perfectly
        assert metrics.subspace_overlap > 0.95, (
            f"Expected high overlap for isometric LoRA, got {metrics.subspace_overlap}"
        )
        # SPR should be high (spectrum preserved, just scaled)
        # Note: SPR uses min(), so scaling UP reduces SPR
        assert metrics.spectral_preservation_ratio > 0.8, (
            f"Expected high SPR, got {metrics.spectral_preservation_ratio}"
        )
        # Combined IR should be good
        assert metrics.isometry_ratio > 0.8

    def test_random_lora_moderate_metrics(self, backend):
        """Test that random LoRA has moderate overlap."""
        from modelcypher.experimental.lora_isometry import (
            compute_isometry_metrics,
            create_synthetic_random_lora,
        )

        lora = create_synthetic_random_lora(backend=backend)
        metrics = compute_isometry_metrics(
            lora.weight_original, lora.weight_modified, backend
        )

        # Random direction will have some overlap by chance
        assert 0.0 < metrics.subspace_overlap < 1.0
        # SPR should be reasonable for small perturbation
        assert metrics.spectral_preservation_ratio > 0.5
        # Grassmann distance should be non-zero
        assert 0 < metrics.grassmann_distance < math.pi / 2

    def test_orthogonal_lora_low_overlap(self, backend):
        """Test that null-space LoRA has low overlap."""
        from modelcypher.experimental.lora_isometry import (
            compute_isometry_metrics,
            create_synthetic_orthogonal_lora,
        )

        lora = create_synthetic_orthogonal_lora(backend=backend)
        metrics = compute_isometry_metrics(
            lora.weight_original, lora.weight_modified, backend
        )

        # Null space action should have low overlap
        # (Not exactly 0 due to numerical precision and random components)
        assert metrics.subspace_overlap < 0.5, (
            f"Expected low overlap for orthogonal LoRA, got {metrics.subspace_overlap}"
        )
        # Original spectrum should be mostly preserved
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

        # Metrics should be approximately equal
        assert abs(metrics_1.subspace_overlap - metrics_2.subspace_overlap) < 0.01
        assert (
            abs(
                metrics_1.spectral_preservation_ratio
                - metrics_2.spectral_preservation_ratio
            )
            < 0.01
        )
        assert abs(metrics_1.isometry_ratio - metrics_2.isometry_ratio) < 0.01

    def test_identical_weights_perfect_metrics(self, backend):
        """Test that W' = W gives perfect metrics."""
        from modelcypher.experimental.lora_isometry import compute_isometry_metrics

        W = backend.random_normal((32, 16), dtype="float32")
        backend.eval(W)

        metrics = compute_isometry_metrics(W, W, backend)

        # No change = perfect preservation
        assert metrics.spectral_preservation_ratio == 1.0
        assert metrics.subspace_overlap == 1.0  # ΔW = 0, defined as perfect
        assert metrics.grassmann_distance < 0.01  # Identical subspaces
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

    def test_overlap_range(self, backend):
        """Test that subspace overlap is in [0, 1]."""
        from modelcypher.experimental.lora_isometry import compute_isometry_metrics

        W = backend.random_normal((64, 32), dtype="float32")
        delta = backend.random_normal((64, 32), dtype="float32")
        W_mod = backend.add(W, delta)
        backend.eval(W, W_mod)

        metrics = compute_isometry_metrics(W, W_mod, backend)

        assert 0.0 <= metrics.subspace_overlap <= 1.0

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
