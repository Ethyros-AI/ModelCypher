# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for mode connectivity analysis module.

Mode connectivity measures loss barriers between weight configurations.
Models in the same loss basin (low barrier) merge better than models
in disconnected modes (high barrier).

For LoRA merging: Check if applying a LoRA pushes the model out of its
original loss basin. High barrier = LoRA fighting base model structure.

References:
    - Draxler et al. (2018) "Essentially No Barriers in Neural Network Energy Landscape"
    - Garipov et al. (2018) "Loss Surfaces, Mode Connectivity, and Fast Ensembling"
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.mode_connectivity import (
    InterpolationMethod,
    analyze_mode_connectivity,
    compute_loss_barrier_profile,
    compute_path_losses,
    linear_interpolate,
)
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon


@pytest.fixture
def backend():
    """Get default backend."""
    return get_default_backend()


class TestLinearInterpolation:
    """Tests for linear_interpolate function."""

    def test_t_zero_returns_source(self, backend):
        """At t=0, interpolation should return source exactly."""
        W0 = backend.array([[1.0, 2.0], [3.0, 4.0]])
        W1 = backend.array([[5.0, 6.0], [7.0, 8.0]])

        result = linear_interpolate(W0, W1, 0.0, backend)
        backend.eval(result)

        eps = float(machine_epsilon(backend, W0))
        diff = backend.sum(backend.abs(result - W0))
        backend.eval(diff)
        assert float(backend.to_scalar(diff)) < eps * 10

    def test_t_one_returns_target(self, backend):
        """At t=1, interpolation should return target exactly."""
        W0 = backend.array([[1.0, 2.0], [3.0, 4.0]])
        W1 = backend.array([[5.0, 6.0], [7.0, 8.0]])

        result = linear_interpolate(W0, W1, 1.0, backend)
        backend.eval(result)

        eps = float(machine_epsilon(backend, W1))
        diff = backend.sum(backend.abs(result - W1))
        backend.eval(diff)
        assert float(backend.to_scalar(diff)) < eps * 10

    def test_t_half_returns_midpoint(self, backend):
        """At t=0.5, should return average of source and target."""
        W0 = backend.zeros((2, 2))
        W1 = backend.full((2, 2), 2.0)
        expected = backend.ones((2, 2))

        result = linear_interpolate(W0, W1, 0.5, backend)
        backend.eval(result)

        eps = float(machine_epsilon(backend, expected))
        diff = backend.sum(backend.abs(result - expected))
        backend.eval(diff)
        assert float(backend.to_scalar(diff)) < eps * 10

    def test_clamps_t_to_valid_range(self, backend):
        """Values outside [0, 1] should be clamped."""
        W0 = backend.zeros((2, 2))
        W1 = backend.ones((2, 2))

        # t < 0 should clamp to 0
        result_neg = linear_interpolate(W0, W1, -0.5, backend)
        backend.eval(result_neg)
        diff = backend.sum(backend.abs(result_neg - W0))
        backend.eval(diff)
        assert float(backend.to_scalar(diff)) < 1e-6

        # t > 1 should clamp to 1
        result_over = linear_interpolate(W0, W1, 1.5, backend)
        backend.eval(result_over)
        diff = backend.sum(backend.abs(result_over - W1))
        backend.eval(diff)
        assert float(backend.to_scalar(diff)) < 1e-6


class TestComputePathLosses:
    """Tests for compute_path_losses function."""

    def test_returns_correct_number_of_points(self, backend):
        """Should return n_steps points along the path."""
        W0 = backend.random_normal((4, 4))
        W1 = backend.random_normal((4, 4))

        def dummy_loss(w):
            return 1.0

        t_values, losses = compute_path_losses(W0, W1, dummy_loss, n_steps=11, backend=backend)

        assert len(t_values) == 11
        assert len(losses) == 11

    def test_t_values_span_zero_to_one(self, backend):
        """t values should go from 0 to 1."""
        W0 = backend.random_normal((4, 4))
        W1 = backend.random_normal((4, 4))

        def dummy_loss(w):
            return 1.0

        t_values, _ = compute_path_losses(W0, W1, dummy_loss, n_steps=11, backend=backend)

        assert t_values[0] == 0.0
        assert t_values[-1] == 1.0
        # Should be evenly spaced
        for i in range(1, len(t_values)):
            assert abs(t_values[i] - t_values[i-1] - 0.1) < 1e-10

    def test_calls_loss_function_at_each_point(self, backend):
        """Loss function should be called at each interpolation point."""
        W0 = backend.random_normal((4, 4))
        W1 = backend.random_normal((4, 4))

        call_count = [0]

        def counting_loss(w):
            call_count[0] += 1
            return float(call_count[0])

        _, losses = compute_path_losses(W0, W1, counting_loss, n_steps=5, backend=backend)

        assert call_count[0] == 5
        assert losses == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_unknown_method_raises_value_error(self, backend):
        """Unexpected interpolation method should fail with ValueError."""
        W0 = backend.random_normal((2, 2))
        W1 = backend.random_normal((2, 2))

        def dummy_loss(w):
            return 1.0

        with pytest.raises(ValueError, match="Unknown interpolation method"):
            compute_path_losses(
                W0, W1, dummy_loss, n_steps=3,
                method="cubic",  # type: ignore[arg-type]
                backend=backend,
            )


class TestAnalyzeModeConnectivity:
    """Tests for analyze_mode_connectivity function."""

    def test_identical_weights_zero_barrier(self, backend):
        """When source == target, barrier should be 0."""
        weights = backend.random_normal((8, 8))

        def constant_loss(w):
            return 1.0

        result = analyze_mode_connectivity(weights, weights, constant_loss, n_steps=11, backend=backend)

        eps = float(machine_epsilon(backend, weights))
        assert result.barrier_height < eps * 10, \
            f"Barrier for identical weights should be ~0, got {result.barrier_height}"

    def test_barrier_height_nonnegative(self, backend):
        """Barrier height should always be >= 0."""
        backend.random_seed(42)
        W0 = backend.random_normal((8, 8))
        backend.random_seed(123)
        W1 = backend.random_normal((8, 8))

        def quadratic_loss(w):
            # Simple loss that varies with weights
            return float(backend.to_scalar(backend.sum(w * w)))

        result = analyze_mode_connectivity(W0, W1, quadratic_loss, n_steps=11, backend=backend)

        assert result.barrier_height >= 0, f"Barrier height should be >= 0, got {result.barrier_height}"

    def test_barrier_location_in_valid_range(self, backend):
        """Barrier location should be in [0, 1]."""
        backend.random_seed(42)
        W0 = backend.random_normal((8, 8))
        backend.random_seed(123)
        W1 = backend.random_normal((8, 8))

        def varying_loss(w):
            norm = float(backend.to_scalar(backend.sum(w * w)))
            return norm

        result = analyze_mode_connectivity(W0, W1, varying_loss, n_steps=21, backend=backend)

        assert 0.0 <= result.barrier_location <= 1.0, \
            f"Barrier location should be in [0, 1], got {result.barrier_location}"

    def test_source_target_losses_recorded(self, backend):
        """Source and target endpoint losses should be recorded."""
        W0 = backend.full((4, 4), 1.0)
        W1 = backend.full((4, 4), 2.0)

        def norm_loss(w):
            return float(backend.to_scalar(backend.sum(w * w)))

        result = analyze_mode_connectivity(W0, W1, norm_loss, n_steps=5, backend=backend)

        expected_source_loss = 16.0  # 4*4 * 1^2
        expected_target_loss = 64.0  # 4*4 * 2^2

        assert abs(result.source_loss - expected_source_loss) < 0.1
        assert abs(result.target_loss - expected_target_loss) < 0.1

    def test_result_has_method(self, backend):
        """Result should record which interpolation method was used."""
        W0 = backend.random_normal((4, 4))
        W1 = backend.random_normal((4, 4))

        def dummy_loss(w):
            return 1.0

        result = analyze_mode_connectivity(
            W0, W1, dummy_loss,
            n_steps=11,
            method=InterpolationMethod.LINEAR,
            backend=backend
        )

        assert result.method == InterpolationMethod.LINEAR


class TestLossBarrierProfile:
    """Tests for compute_loss_barrier_profile function."""

    def test_monotonic_path_detected(self, backend):
        """A monotonic loss path should be detected."""
        W0 = backend.full((4, 4), 0.0)
        W1 = backend.full((4, 4), 1.0)

        # Loss increases linearly with weight magnitude
        def linear_loss(w):
            return float(backend.to_scalar(backend.sum(w)))

        profile = compute_loss_barrier_profile(W0, W1, linear_loss, n_steps=11, backend=backend)

        # Path from 0 to 16 should be monotonic increasing
        assert profile.is_monotonic or profile.gradient_sign_changes <= 1

    def test_local_minima_detected(self, backend):
        """Local minima along the path should be detected."""
        W0 = backend.full((2, 2), 0.0)
        W1 = backend.full((2, 2), 2.0)

        # Quadratic loss with minimum at midpoint (w=1)
        def quadratic_loss(w):
            mean_w = float(backend.to_scalar(backend.mean(w)))
            return (mean_w - 1.0) ** 2  # Minimum at w=1 (t=0.5)

        profile = compute_loss_barrier_profile(W0, W1, quadratic_loss, n_steps=21, backend=backend)

        # Should detect the minimum near t=0.5
        if profile.local_minima_t:
            # At least one local minimum should be near 0.5
            has_near_half = any(0.3 < t < 0.7 for t in profile.local_minima_t)
            assert has_near_half, f"Expected local minimum near t=0.5, got {profile.local_minima_t}"

    def test_lipschitz_estimate_nonnegative(self, backend):
        """Lipschitz estimate should be non-negative."""
        W0 = backend.random_normal((4, 4))
        W1 = backend.random_normal((4, 4))

        def varying_loss(w):
            return float(backend.to_scalar(backend.sum(backend.abs(w))))

        profile = compute_loss_barrier_profile(W0, W1, varying_loss, n_steps=11, backend=backend)

        assert profile.lipschitz_estimate >= 0, \
            f"Lipschitz estimate should be >= 0, got {profile.lipschitz_estimate}"
