# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for CKA-based loss proxy for mode connectivity analysis."""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cka_loss_proxy import (
    make_cka_loss_proxy,
    make_simple_cka_loss_proxy,
)
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon


@pytest.fixture
def backend():
    """Get default backend."""
    return get_default_backend()


class TestSimpleCKALossProxy:
    """Tests for make_simple_cka_loss_proxy (activation interpolation)."""

    def test_source_point_zero_loss(self, backend):
        """At source (t=0), CKA=1.0 so loss=0."""
        backend.random_seed(42)
        source = backend.random_normal((50, 32))
        target = backend.random_normal((50, 32))

        loss_fn = make_simple_cka_loss_proxy(source, target, backend)
        loss_at_source = loss_fn(0.0)

        eps = float(machine_epsilon(backend, source))
        assert loss_at_source < eps * 10, f"Loss at source should be ~0, got {loss_at_source}"

    def test_target_point_positive_loss(self, backend):
        """At target (t=1), loss > 0 for different activations."""
        backend.random_seed(42)
        source = backend.random_normal((50, 32))
        backend.random_seed(123)  # Different seed for truly different activations
        target = backend.random_normal((50, 32))

        loss_fn = make_simple_cka_loss_proxy(source, target, backend)
        loss_at_target = loss_fn(1.0)

        assert loss_at_target > 0.0, f"Loss at target should be > 0, got {loss_at_target}"

    def test_identical_activations_always_zero_loss(self, backend):
        """When source == target, loss should be ~0 at all points."""
        backend.random_seed(42)
        activations = backend.random_normal((50, 32))

        loss_fn = make_simple_cka_loss_proxy(activations, activations, backend)

        eps = float(machine_epsilon(backend, activations))
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            loss = loss_fn(t)
            assert loss < eps * 10, f"Loss at t={t} should be ~0 for identical activations, got {loss}"

    def test_midpoint_has_intermediate_loss(self, backend):
        """Loss at t=0.5 should be between loss at t=0 and t=1."""
        backend.random_seed(42)
        source = backend.random_normal((50, 32))
        backend.random_seed(123)
        target = backend.random_normal((50, 32))

        loss_fn = make_simple_cka_loss_proxy(source, target, backend)

        loss_0 = loss_fn(0.0)
        loss_mid = loss_fn(0.5)
        loss_1 = loss_fn(1.0)

        # At midpoint, we should have positive loss (diverged from source)
        # but typically less than at full target
        assert loss_mid > loss_0, "Midpoint loss should exceed source loss"
        assert loss_mid <= loss_1 + 0.01, "Midpoint loss shouldn't exceed target loss much"

    def test_loss_values_in_valid_range(self, backend):
        """Loss values should be in [0, 2] (since CKA in [-1, 1])."""
        backend.random_seed(42)
        source = backend.random_normal((50, 32))
        backend.random_seed(999)
        target = backend.random_normal((50, 32))

        loss_fn = make_simple_cka_loss_proxy(source, target, backend)

        for t in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            loss = loss_fn(t)
            assert 0.0 <= loss <= 2.0, f"Loss at t={t} out of range: {loss}"


class TestCKALossProxyWithForward:
    """Tests for make_cka_loss_proxy (with forward function)."""

    def test_source_weights_zero_loss(self, backend):
        """Loss at source weights should be ~0."""
        backend.random_seed(42)

        # Simple linear layer weights
        weights = backend.random_normal((32, 64))
        probe_inputs = backend.random_normal((20, 32))

        # Simple forward: matmul
        def forward_fn(w, x):
            return backend.matmul(x, w)

        # Source activations from source weights
        source_activations = forward_fn(weights, probe_inputs)
        backend.eval(source_activations)

        loss_fn = make_cka_loss_proxy(source_activations, forward_fn, probe_inputs, backend)

        # Evaluate at source weights
        loss = loss_fn(weights)

        eps = float(machine_epsilon(backend, weights))
        assert loss < eps * 10, f"Loss at source weights should be ~0, got {loss}"

    def test_different_weights_positive_loss(self, backend):
        """Loss at different weights should be > 0."""
        backend.random_seed(42)

        source_weights = backend.random_normal((32, 64))
        probe_inputs = backend.random_normal((20, 32))

        def forward_fn(w, x):
            return backend.matmul(x, w)

        source_activations = forward_fn(source_weights, probe_inputs)
        backend.eval(source_activations)

        loss_fn = make_cka_loss_proxy(source_activations, forward_fn, probe_inputs, backend)

        # Evaluate at different weights
        backend.random_seed(999)
        other_weights = backend.random_normal((32, 64))
        loss = loss_fn(other_weights)

        assert loss > 0.0, f"Loss at different weights should be > 0, got {loss}"


class TestIntegrationWithModeConnectivity:
    """Integration tests with mode_connectivity module."""

    def test_simple_proxy_works_with_analyze_mode_connectivity(self, backend):
        """The simple proxy integrates with mode connectivity analysis."""
        from modelcypher.core.domain.geometry.mode_connectivity import (
            analyze_mode_connectivity,
        )

        backend.random_seed(42)
        source_weights = backend.random_normal((16, 16))
        backend.random_seed(123)
        target_weights = backend.random_normal((16, 16))

        # Create activations from weights (simple matmul with fixed input)
        probe = backend.random_normal((10, 16))
        source_acts = backend.matmul(probe, source_weights)
        target_acts = backend.matmul(probe, target_weights)
        backend.eval(source_acts, target_acts)

        # Use simple proxy (directly on activations)
        # We need to adapt: mode_connectivity expects loss_fn(weights) -> float
        # For this test, use a wrapper that interpolates weights, then activations

        def activation_based_loss(interpolated_weights):
            # Compute activations from interpolated weights
            acts = backend.matmul(probe, interpolated_weights)
            backend.eval(acts)

            # Center for CKA
            source_centered = source_acts - backend.mean(source_acts, axis=0, keepdims=True)
            acts_centered = acts - backend.mean(acts, axis=0, keepdims=True)
            backend.eval(source_centered, acts_centered)

            from modelcypher.core.domain.geometry.cka import compute_linear_cka_from_activations
            cka = compute_linear_cka_from_activations(source_centered, acts_centered, backend)
            return 1.0 - cka

        result = analyze_mode_connectivity(
            source_weights,
            target_weights,
            activation_based_loss,
            n_steps=11,
            backend=backend,
        )

        # Verify result structure
        assert len(result.path_losses) == 11
        assert len(result.path_t_values) == 11
        assert result.barrier_height >= 0
        assert 0.0 <= result.barrier_location <= 1.0

        # Source endpoint should have near-zero loss
        eps = float(machine_epsilon(backend, source_weights))
        assert result.source_loss < eps * 100, f"Source loss should be ~0, got {result.source_loss}"

    def test_identical_weights_zero_barrier(self, backend):
        """Identical source and target should have zero barrier."""
        from modelcypher.core.domain.geometry.mode_connectivity import (
            analyze_mode_connectivity,
        )

        backend.random_seed(42)
        weights = backend.random_normal((16, 16))
        probe = backend.random_normal((10, 16))
        activations = backend.matmul(probe, weights)
        backend.eval(activations)

        def constant_zero_loss(w):
            # When weights are identical, all interpolations produce same activations
            # So CKA = 1.0 always, loss = 0 always
            acts = backend.matmul(probe, w)
            acts_centered = acts - backend.mean(acts, axis=0, keepdims=True)
            source_centered = activations - backend.mean(activations, axis=0, keepdims=True)
            backend.eval(acts_centered, source_centered)

            from modelcypher.core.domain.geometry.cka import compute_linear_cka_from_activations
            cka = compute_linear_cka_from_activations(source_centered, acts_centered, backend)
            return 1.0 - cka

        result = analyze_mode_connectivity(
            weights,
            weights,  # Same as source
            constant_zero_loss,
            n_steps=11,
            backend=backend,
        )

        eps = float(machine_epsilon(backend, weights))
        assert result.barrier_height < eps * 100, f"Barrier should be ~0 for identical weights, got {result.barrier_height}"
