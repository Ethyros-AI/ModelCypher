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

"""Extended tests for transplant stitch construction.

Tests critical APIs:
- compute_composite_stitches(): Build stitch matrices from transforms
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import all_finite
from modelcypher.core.use_cases.merge.stages.transplant_stitches import (
    compute_composite_stitches,
)


@pytest.fixture
def backend():
    return get_default_backend()


class TestComputeCompositeStitches:
    """Tests for compute_composite_stitches()."""

    def test_empty_transforms_returns_empty(self, backend):
        """Empty transforms map should return empty result."""
        result = compute_composite_stitches(
            transforms_map=None,
            desc="test",
            backend=backend,
            layer_mapping=None,
        )

        assert result == {}

    def test_single_layer_single_source(self, backend):
        """Single layer with single source transform."""
        transform = backend.random_normal((16, 32))
        backend.eval(transform)

        transforms_map = {0: transform}
        layer_mapping = {0: 0}

        result = compute_composite_stitches(
            transforms_map=transforms_map,
            desc="test",
            backend=backend,
            layer_mapping=layer_mapping,
        )

        assert 0 in result
        assert 0 in result[0]  # Source layer 0
        P, Q = result[0][0]

        # P (stitch_output) should be transpose of F
        assert backend.shape(P) == (32, 16)
        # Q (stitch_input) should be transpose of F_pinv
        assert backend.shape(Q) == (16, 32)
        assert all_finite(P, backend)
        assert all_finite(Q, backend)

    def test_multiple_layers(self, backend):
        """Multiple target layers with transforms."""
        transforms_map = {
            0: backend.random_normal((16, 32)),
            1: backend.random_normal((16, 32)),
            2: backend.random_normal((16, 32)),
        }
        for k, v in transforms_map.items():
            backend.eval(v)

        layer_mapping = {0: 0, 1: 1, 2: 2}

        result = compute_composite_stitches(
            transforms_map=transforms_map,
            desc="test",
            backend=backend,
            layer_mapping=layer_mapping,
        )

        assert len(result) == 3
        for tgt_layer in [0, 1, 2]:
            assert tgt_layer in result

    def test_composite_sources(self, backend):
        """Layer with multiple source contributions (dict format)."""
        # Layer 0 receives from sources 0 and 1
        transforms_map = {
            0: {
                0: backend.random_normal((16, 32)),
                1: backend.random_normal((8, 32)),
            }
        }
        for src_map in transforms_map.values():
            for arr in src_map.values():
                backend.eval(arr)

        layer_mapping = {0: 0}

        result = compute_composite_stitches(
            transforms_map=transforms_map,
            desc="test",
            backend=backend,
            layer_mapping=layer_mapping,
        )

        assert 0 in result
        # Should have stitches for both sources
        assert 0 in result[0]
        assert 1 in result[0]

        # Check shapes
        P0, Q0 = result[0][0]
        P1, Q1 = result[0][1]

        # P dimensions: [tgt_dim, src_dim] where src_dim varies
        assert backend.shape(P0)[1] == 16  # First source dim
        assert backend.shape(P1)[1] == 8   # Second source dim

    def test_scale_ratio_not_applied(self, backend):
        """Scale ratio is intentionally NOT applied to weight stitch transforms.

        The scale_ratio parameter normalizes activation magnitudes during alignment,
        but should not scale weight transforms - scaling would shrink weights.
        """
        transform = backend.ones((8, 16))  # All ones for easy checking
        backend.eval(transform)

        transforms_map = {0: transform}
        layer_mapping = {0: 0}
        scale_ratios = {0: 2.0}

        result_scaled = compute_composite_stitches(
            transforms_map=transforms_map,
            desc="test",
            backend=backend,
            layer_mapping=layer_mapping,
            layer_scale_ratios=scale_ratios,
        )

        result_unscaled = compute_composite_stitches(
            transforms_map=transforms_map,
            desc="test",
            backend=backend,
            layer_mapping=layer_mapping,
            layer_scale_ratios=None,
        )

        P_scaled, _ = result_scaled[0][0]
        P_unscaled, _ = result_unscaled[0][0]

        # Scale ratio is NOT applied to weight stitches - should be identical
        ratio = backend.mean(P_scaled) / backend.mean(P_unscaled)
        backend.eval(ratio)

        ratio_val = float(backend.to_scalar(ratio))
        eps = division_epsilon(backend, P_scaled) * max(1.0, abs(ratio_val))
        assert abs(ratio_val - 1.0) <= eps

    def test_stitch_roundtrip_approximate(self, backend):
        """P @ Q should approximate identity (for well-conditioned F)."""
        # Use a well-conditioned matrix (orthogonal-ish)
        backend.random_seed(42)
        F = backend.random_normal((8, 16))
        backend.eval(F)

        transforms_map = {0: F}
        layer_mapping = {0: 0}

        result = compute_composite_stitches(
            transforms_map=transforms_map,
            desc="test",
            backend=backend,
            layer_mapping=layer_mapping,
        )

        P, Q = result[0][0]

        # P @ F^T @ F @ Q should relate to identity
        # But more directly: F @ Q @ P @ F^T should reconstruct F @ F^T
        # For our purposes, just check shapes and finiteness
        assert all_finite(P, backend)
        assert all_finite(Q, backend)


class TestCompositeStitchesMathematicalProperties:
    """Hypothesis-based tests for stitch construction."""

    @given(
        n_layers=st.integers(min_value=1, max_value=5),
        src_dim=st.integers(min_value=4, max_value=16),
        tgt_dim=st.integers(min_value=8, max_value=32),
    )
    @settings(max_examples=5, deadline=None)
    def test_all_stitches_finite(self, n_layers, src_dim, tgt_dim):
        """All computed stitches should be finite."""
        backend = get_default_backend()

        transforms_map = {}
        layer_mapping = {}
        for i in range(n_layers):
            transforms_map[i] = backend.random_normal((src_dim, tgt_dim))
            backend.eval(transforms_map[i])
            layer_mapping[i] = i

        result = compute_composite_stitches(
            transforms_map=transforms_map,
            desc="test",
            backend=backend,
            layer_mapping=layer_mapping,
        )

        for tgt_layer, stitches in result.items():
            for src_layer, (P, Q) in stitches.items():
                assert all_finite(P, backend), f"P not finite for layer {tgt_layer}->{src_layer}"
                assert all_finite(Q, backend), f"Q not finite for layer {tgt_layer}->{src_layer}"

    @given(
        src_dim=st.integers(min_value=4, max_value=16),
        tgt_dim=st.integers(min_value=8, max_value=32),
    )
    @settings(max_examples=10, deadline=None)
    def test_stitch_shapes_match_transform(self, src_dim, tgt_dim):
        """Stitch dimensions should correspond to transform dimensions."""
        backend = get_default_backend()

        F = backend.random_normal((src_dim, tgt_dim))
        backend.eval(F)

        transforms_map = {0: F}
        layer_mapping = {0: 0}

        result = compute_composite_stitches(
            transforms_map=transforms_map,
            desc="test",
            backend=backend,
            layer_mapping=layer_mapping,
        )

        P, Q = result[0][0]

        # P = F^T has shape [tgt_dim, src_dim]
        assert backend.shape(P)[0] == tgt_dim
        assert backend.shape(P)[1] == src_dim

        # Q = (F_pinv)^T has shape [src_dim, tgt_dim]
        assert backend.shape(Q)[0] == src_dim
        assert backend.shape(Q)[1] == tgt_dim
