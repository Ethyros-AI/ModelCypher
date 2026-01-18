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

"""Extended tests for anchor-relative concept grafting.

Tests critical APIs:
- compute_anchor_grafting_delta(): Basic anchor-relative grafting pipeline
- compute_anchor_grafting_with_ghost_anchors(): Extended pipeline with Ghost Anchor synthesis
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.anchor_grafting import (
    compute_anchor_grafting_delta,
    compute_anchor_grafting_with_ghost_anchors,
)
from modelcypher.core.domain.geometry.numerical_stability import regularization_epsilon


@pytest.fixture
def backend():
    return get_default_backend()


class TestComputeAnchorGraftingDelta:
    """Tests for compute_anchor_grafting_delta()."""

    def test_basic_grafting(self, backend):
        """Basic grafting pipeline should work."""
        n_samples = 16
        d_source = 32
        d_target = 24
        n_anchors = 10

        source_act = backend.random_normal((n_samples, d_source))
        target_act = backend.random_normal((n_samples, d_target))
        source_anchors = backend.random_normal((n_anchors, d_source))
        target_anchors = backend.random_normal((n_anchors, d_target))
        backend.eval(source_act, target_act, source_anchors, target_anchors)

        result = compute_anchor_grafting_delta(
            source_act, target_act, source_anchors, target_anchors, backend
        )

        # Check result structure
        assert result.delta_activations is not None
        assert backend.shape(result.delta_activations) == (n_samples, d_target)
        assert result.rotation_matrix is not None
        assert backend.shape(result.rotation_matrix) == (n_anchors, n_anchors)

    def test_same_dimensions(self, backend):
        """Same source/target dimensions should work."""
        n_samples = 16
        d = 32
        n_anchors = 10

        source_act = backend.random_normal((n_samples, d))
        target_act = backend.random_normal((n_samples, d))
        source_anchors = backend.random_normal((n_anchors, d))
        target_anchors = backend.random_normal((n_anchors, d))
        backend.eval(source_act, target_act, source_anchors, target_anchors)

        result = compute_anchor_grafting_delta(
            source_act, target_act, source_anchors, target_anchors, backend
        )

        assert backend.shape(result.delta_activations) == (n_samples, d)

    def test_identical_activations_small_delta(self, backend):
        """Identical activations should produce small delta."""
        n_samples = 16
        d = 32
        n_anchors = 10

        # Use same activations for source and target
        activations = backend.random_normal((n_samples, d))
        source_anchors = backend.random_normal((n_anchors, d))
        # Use same anchors normalized
        backend.eval(activations, source_anchors)

        result = compute_anchor_grafting_delta(
            activations, activations, source_anchors, source_anchors, backend
        )

        # Delta should be very small when source == target
        delta_mean = backend.mean(backend.abs(result.delta_activations))
        backend.eval(delta_mean)
        assert float(backend.to_scalar(delta_mean)) < 0.1

    def test_density_weights_bounded(self, backend):
        """Density weights should be in [0, 1]."""
        n_samples = 16
        d = 32
        n_anchors = 10

        source_act = backend.random_normal((n_samples, d))
        target_act = backend.random_normal((n_samples, d))
        source_anchors = backend.random_normal((n_anchors, d))
        target_anchors = backend.random_normal((n_anchors, d))
        backend.eval(source_act, target_act, source_anchors, target_anchors)

        result = compute_anchor_grafting_delta(
            source_act, target_act, source_anchors, target_anchors, backend
        )

        min_w = backend.min(result.density_weights)
        max_w = backend.max(result.density_weights)
        backend.eval(min_w, max_w)

        tol = regularization_epsilon(backend, result.density_weights)
        assert float(backend.to_scalar(min_w)) >= -tol
        assert float(backend.to_scalar(max_w)) <= 1.0 + tol

    def test_transfer_fraction_bounded(self, backend):
        """Transfer fraction should be in [0, 1]."""
        n_samples = 16
        d = 32
        n_anchors = 10

        source_act = backend.random_normal((n_samples, d))
        target_act = backend.random_normal((n_samples, d))
        source_anchors = backend.random_normal((n_anchors, d))
        target_anchors = backend.random_normal((n_anchors, d))
        backend.eval(source_act, target_act, source_anchors, target_anchors)

        result = compute_anchor_grafting_delta(
            source_act, target_act, source_anchors, target_anchors, backend
        )

        assert 0.0 <= result.transfer_fraction <= 1.0

    def test_few_anchors(self, backend):
        """Should work with minimal anchors."""
        n_samples = 16
        d = 32
        n_anchors = 3  # Minimal

        source_act = backend.random_normal((n_samples, d))
        target_act = backend.random_normal((n_samples, d))
        source_anchors = backend.random_normal((n_anchors, d))
        target_anchors = backend.random_normal((n_anchors, d))
        backend.eval(source_act, target_act, source_anchors, target_anchors)

        result = compute_anchor_grafting_delta(
            source_act, target_act, source_anchors, target_anchors, backend
        )

        assert result.delta_activations is not None


class TestComputeAnchorGraftingWithGhostAnchors:
    """Tests for compute_anchor_grafting_with_ghost_anchors()."""

    def test_basic_ghost_anchor_grafting(self, backend):
        """Ghost anchor grafting should work."""
        n_samples = 16
        d_source = 32
        d_target = 24
        n_anchors = 10

        source_act = backend.random_normal((n_samples, d_source))
        target_act = backend.random_normal((n_samples, d_target))
        source_anchors = backend.random_normal((n_anchors, d_source))
        target_anchors = backend.random_normal((n_anchors, d_target))
        backend.eval(source_act, target_act, source_anchors, target_anchors)

        result = compute_anchor_grafting_with_ghost_anchors(
            source_act, target_act, source_anchors, target_anchors, backend=backend
        )

        assert result.delta_activations is not None
        assert backend.shape(result.delta_activations) == (n_samples, d_target)

    def test_with_anchor_names(self, backend):
        """Should accept custom anchor names."""
        n_samples = 16
        d = 32
        n_anchors = 5

        source_act = backend.random_normal((n_samples, d))
        target_act = backend.random_normal((n_samples, d))
        source_anchors = backend.random_normal((n_anchors, d))
        target_anchors = backend.random_normal((n_anchors, d))
        backend.eval(source_act, target_act, source_anchors, target_anchors)

        anchor_names = [f"custom_anchor_{i}" for i in range(n_anchors)]

        result = compute_anchor_grafting_with_ghost_anchors(
            source_act, target_act, source_anchors, target_anchors,
            anchor_names=anchor_names, backend=backend
        )

        assert result.delta_activations is not None

    def test_same_dimensions(self, backend):
        """Same dimensions should work with ghost anchors."""
        n_samples = 16
        d = 32
        n_anchors = 10

        source_act = backend.random_normal((n_samples, d))
        target_act = backend.random_normal((n_samples, d))
        source_anchors = backend.random_normal((n_anchors, d))
        target_anchors = backend.random_normal((n_anchors, d))
        backend.eval(source_act, target_act, source_anchors, target_anchors)

        result = compute_anchor_grafting_with_ghost_anchors(
            source_act, target_act, source_anchors, target_anchors, backend=backend
        )

        assert backend.shape(result.delta_activations) == (n_samples, d)

    def test_density_weights_bounded_ghost(self, backend):
        """Density weights should be in [0, 1] with ghost anchors."""
        n_samples = 16
        d = 32
        n_anchors = 10

        source_act = backend.random_normal((n_samples, d))
        target_act = backend.random_normal((n_samples, d))
        source_anchors = backend.random_normal((n_anchors, d))
        target_anchors = backend.random_normal((n_anchors, d))
        backend.eval(source_act, target_act, source_anchors, target_anchors)

        result = compute_anchor_grafting_with_ghost_anchors(
            source_act, target_act, source_anchors, target_anchors, backend=backend
        )

        min_w = backend.min(result.density_weights)
        max_w = backend.max(result.density_weights)
        backend.eval(min_w, max_w)

        tol = regularization_epsilon(backend, result.density_weights)
        assert float(backend.to_scalar(min_w)) >= -tol
        assert float(backend.to_scalar(max_w)) <= 1.0 + tol

    def test_identical_produces_small_delta(self, backend):
        """Identical activations should produce small delta even with ghost anchors."""
        n_samples = 16
        d = 32
        n_anchors = 10

        activations = backend.random_normal((n_samples, d))
        anchors = backend.random_normal((n_anchors, d))
        backend.eval(activations, anchors)

        result = compute_anchor_grafting_with_ghost_anchors(
            activations, activations, anchors, anchors, backend=backend
        )

        # Delta should be very small when source == target
        delta_mean = backend.mean(backend.abs(result.delta_activations))
        backend.eval(delta_mean)
        assert float(backend.to_scalar(delta_mean)) < 0.1


class TestAnchorGraftingMathematicalProperties:
    """Hypothesis-based tests for mathematical invariants."""

    @given(
        n_samples=st.integers(min_value=8, max_value=24),
        d=st.integers(min_value=16, max_value=48),
        n_anchors=st.integers(min_value=4, max_value=12),
    )
    @settings(max_examples=5, deadline=None)
    def test_delta_shape_correct(self, n_samples, d, n_anchors):
        """Delta activations should have correct shape [n, d_target]."""
        backend = get_default_backend()
        source_act = backend.random_normal((n_samples, d))
        target_act = backend.random_normal((n_samples, d))
        source_anchors = backend.random_normal((n_anchors, d))
        target_anchors = backend.random_normal((n_anchors, d))
        backend.eval(source_act, target_act, source_anchors, target_anchors)

        result = compute_anchor_grafting_delta(
            source_act, target_act, source_anchors, target_anchors, backend
        )

        assert backend.shape(result.delta_activations) == (n_samples, d)

    @given(
        n_samples=st.integers(min_value=8, max_value=24),
        d_source=st.integers(min_value=16, max_value=48),
        d_target=st.integers(min_value=16, max_value=48),
        n_anchors=st.integers(min_value=4, max_value=12),
    )
    @settings(max_examples=5, deadline=None)
    def test_cross_dim_delta_shape(self, n_samples, d_source, d_target, n_anchors):
        """Cross-dimension grafting should produce target-dim delta."""
        backend = get_default_backend()
        source_act = backend.random_normal((n_samples, d_source))
        target_act = backend.random_normal((n_samples, d_target))
        source_anchors = backend.random_normal((n_anchors, d_source))
        target_anchors = backend.random_normal((n_anchors, d_target))
        backend.eval(source_act, target_act, source_anchors, target_anchors)

        result = compute_anchor_grafting_delta(
            source_act, target_act, source_anchors, target_anchors, backend
        )

        assert backend.shape(result.delta_activations) == (n_samples, d_target)

    @given(
        n_samples=st.integers(min_value=8, max_value=24),
        d=st.integers(min_value=16, max_value=48),
        n_anchors=st.integers(min_value=4, max_value=12),
    )
    @settings(max_examples=5, deadline=None)
    def test_density_weights_valid(self, n_samples, d, n_anchors):
        """Density weights should all be in valid range."""
        backend = get_default_backend()
        source_act = backend.random_normal((n_samples, d))
        target_act = backend.random_normal((n_samples, d))
        source_anchors = backend.random_normal((n_anchors, d))
        target_anchors = backend.random_normal((n_anchors, d))
        backend.eval(source_act, target_act, source_anchors, target_anchors)

        result = compute_anchor_grafting_delta(
            source_act, target_act, source_anchors, target_anchors, backend
        )

        # Check bounds
        min_w = backend.min(result.density_weights)
        max_w = backend.max(result.density_weights)
        backend.eval(min_w, max_w)

        tol = regularization_epsilon(backend, result.density_weights)
        assert float(backend.to_scalar(min_w)) >= -tol
        assert float(backend.to_scalar(max_w)) <= 1.0 + tol

    @given(
        n_samples=st.integers(min_value=8, max_value=24),
        d=st.integers(min_value=16, max_value=48),
        n_anchors=st.integers(min_value=4, max_value=12),
    )
    @settings(max_examples=5, deadline=None)
    def test_rotation_matrix_square(self, n_samples, d, n_anchors):
        """Rotation matrix should be square [n_anchors, n_anchors]."""
        backend = get_default_backend()
        source_act = backend.random_normal((n_samples, d))
        target_act = backend.random_normal((n_samples, d))
        source_anchors = backend.random_normal((n_anchors, d))
        target_anchors = backend.random_normal((n_anchors, d))
        backend.eval(source_act, target_act, source_anchors, target_anchors)

        result = compute_anchor_grafting_delta(
            source_act, target_act, source_anchors, target_anchors, backend
        )

        assert backend.shape(result.rotation_matrix) == (n_anchors, n_anchors)
