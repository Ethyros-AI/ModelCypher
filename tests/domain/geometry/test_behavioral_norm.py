# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for behavioral norm: ||A @ ΔW.T||_F.

Behavioral norm measures the actual output change induced by a weight delta.
All tests verify strict mathematical properties of the Frobenius norm.
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain.geometry.behavioral_norm import behavioral_norm


class TestBehavioralNorm:
    """Mathematical properties of behavioral_norm."""

    def test_zero_delta_gives_zero(self, any_backend):
        """ΔW = 0 → norm = 0 (no perturbation, no output change)."""
        b = any_backend
        delta_W = b.zeros((4, 8))
        activations = b.random_normal((10, 8))
        b.eval(activations)

        result = behavioral_norm(delta_W, activations, b)

        assert result == 0.0

    def test_zero_activations_gives_zero(self, any_backend):
        """A = 0 → norm = 0 (no input, no output)."""
        b = any_backend
        delta_W = b.random_normal((4, 8))
        activations = b.zeros((10, 8))
        b.eval(delta_W)

        result = behavioral_norm(delta_W, activations, b)

        assert result == 0.0

    def test_always_nonnegative(self, any_backend):
        """Frobenius norm ≥ 0 for any input."""
        b = any_backend
        b.random_seed(42)
        delta_W = b.random_normal((4, 8))
        activations = b.random_normal((10, 8))
        b.eval(delta_W, activations)

        result = behavioral_norm(delta_W, activations, b)

        assert result >= 0.0

    def test_homogeneity_delta(self, any_backend):
        """norm(k*ΔW, A) = |k| * norm(ΔW, A) — absolute homogeneity in delta."""
        b = any_backend
        b.random_seed(42)
        delta_W = b.random_normal((4, 8))
        activations = b.random_normal((10, 8))
        b.eval(delta_W, activations)

        k = 3.0
        scaled_delta = delta_W * k
        b.eval(scaled_delta)

        norm_base = behavioral_norm(delta_W, activations, b)
        norm_scaled = behavioral_norm(scaled_delta, activations, b)

        assert abs(norm_scaled - abs(k) * norm_base) < 1e-4 * max(norm_scaled, 1e-8)

    def test_homogeneity_activations(self, any_backend):
        """norm(ΔW, k*A) = |k| * norm(ΔW, A) — absolute homogeneity in activations."""
        b = any_backend
        b.random_seed(42)
        delta_W = b.random_normal((4, 8))
        activations = b.random_normal((10, 8))
        b.eval(delta_W, activations)

        k = 2.5
        scaled_acts = activations * k
        b.eval(scaled_acts)

        norm_base = behavioral_norm(delta_W, activations, b)
        norm_scaled = behavioral_norm(delta_W, scaled_acts, b)

        assert abs(norm_scaled - abs(k) * norm_base) < 1e-4 * max(norm_scaled, 1e-8)

    def test_1d_delta_auto_reshape(self, any_backend):
        """1D delta [d] is treated as [1, d] — single output dimension."""
        b = any_backend
        b.random_seed(42)
        delta_1d = b.random_normal((8,))
        delta_2d = b.reshape(delta_1d, (1, 8))
        activations = b.random_normal((10, 8))
        b.eval(delta_1d, delta_2d, activations)

        norm_1d = behavioral_norm(delta_1d, activations, b)
        norm_2d = behavioral_norm(delta_2d, activations, b)

        assert abs(norm_1d - norm_2d) < 1e-6

    def test_known_value_identity(self, any_backend):
        """ΔW = I₂, A = I₂ → norm = sqrt(2).

        A @ ΔW.T = I @ I = I, and ||I||_F = sqrt(trace(I^T I)) = sqrt(2).
        """
        b = any_backend
        delta_W = b.eye(2)
        activations = b.eye(2)

        result = behavioral_norm(delta_W, activations, b)

        assert abs(result - 2.0**0.5) < 1e-5, f"Expected sqrt(2) ≈ 1.4142, got {result}"
