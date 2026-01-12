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

"""Extended tests for anchor-relative decoder.

Tests critical APIs:
- compute_anchor_decoder(): Compute decoder from anchor space to target space
- decode_to_activation_space(): Decode weighted delta to activation space
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.anchor_decoder import (
    compute_anchor_decoder,
    decode_to_activation_space,
)
from modelcypher.core.domain.geometry.numerical_stability import all_finite


@pytest.fixture
def backend():
    return get_default_backend()


class TestComputeAnchorDecoder:
    """Tests for compute_anchor_decoder()."""

    def test_basic_decoder_computation(self, backend):
        """Basic decoder computation should work."""
        n_samples = 16
        n_anchors = 8
        d_target = 32

        target_relative_rep = backend.random_normal((n_samples, n_anchors))
        target_activations = backend.random_normal((n_samples, d_target))
        backend.eval(target_relative_rep, target_activations)

        decoder, error = compute_anchor_decoder(
            target_relative_rep=target_relative_rep,
            target_activations=target_activations,
            backend=backend,
        )

        assert decoder is not None
        assert backend.shape(decoder) == (n_anchors, d_target)
        assert all_finite(decoder, backend)

    def test_decoder_reconstruction_error_bounded(self, backend):
        """Reconstruction error should be finite and non-negative."""
        n_samples = 16
        n_anchors = 8
        d_target = 32

        target_relative_rep = backend.random_normal((n_samples, n_anchors))
        target_activations = backend.random_normal((n_samples, d_target))
        backend.eval(target_relative_rep, target_activations)

        _, error = compute_anchor_decoder(
            target_relative_rep=target_relative_rep,
            target_activations=target_activations,
            backend=backend,
        )

        assert error >= 0.0
        assert not (error != error)  # Not NaN

    def test_decoder_square_system(self, backend):
        """Square system (n_anchors == n_samples) should work."""
        n = 16
        d_target = 32

        target_relative_rep = backend.random_normal((n, n))
        target_activations = backend.random_normal((n, d_target))
        backend.eval(target_relative_rep, target_activations)

        decoder, error = compute_anchor_decoder(
            target_relative_rep=target_relative_rep,
            target_activations=target_activations,
            backend=backend,
        )

        assert decoder is not None
        assert backend.shape(decoder) == (n, d_target)

    def test_decoder_overdetermined_system(self, backend):
        """Overdetermined system (n_samples > n_anchors) should work."""
        n_samples = 32
        n_anchors = 8
        d_target = 16

        target_relative_rep = backend.random_normal((n_samples, n_anchors))
        target_activations = backend.random_normal((n_samples, d_target))
        backend.eval(target_relative_rep, target_activations)

        decoder, _ = compute_anchor_decoder(
            target_relative_rep=target_relative_rep,
            target_activations=target_activations,
            backend=backend,
        )

        assert backend.shape(decoder) == (n_anchors, d_target)

    def test_decoder_underdetermined_system(self, backend):
        """Underdetermined system (n_samples < n_anchors) should work."""
        n_samples = 8
        n_anchors = 16
        d_target = 32

        target_relative_rep = backend.random_normal((n_samples, n_anchors))
        target_activations = backend.random_normal((n_samples, d_target))
        backend.eval(target_relative_rep, target_activations)

        decoder, _ = compute_anchor_decoder(
            target_relative_rep=target_relative_rep,
            target_activations=target_activations,
            backend=backend,
        )

        assert backend.shape(decoder) == (n_anchors, d_target)


class TestDecodeToActivationSpace:
    """Tests for decode_to_activation_space()."""

    def test_basic_decoding(self, backend):
        """Basic decoding should work."""
        n_samples = 16
        n_anchors = 8
        d_target = 32

        delta_relative = backend.random_normal((n_samples, n_anchors))
        decoder = backend.random_normal((n_anchors, d_target))
        density_weights = backend.abs(backend.random_normal((n_samples,)))
        density_weights = density_weights / backend.max(density_weights)
        backend.eval(delta_relative, decoder, density_weights)

        delta_activations = decode_to_activation_space(
            delta_relative=delta_relative,
            decoder=decoder,
            density_weights=density_weights,
            backend=backend,
        )

        assert delta_activations is not None
        assert backend.shape(delta_activations) == (n_samples, d_target)
        assert all_finite(delta_activations, backend)

    def test_decode_zero_weights_zero_output(self, backend):
        """Zero weights should produce zero delta."""
        n_samples = 16
        n_anchors = 8
        d_target = 32

        delta_relative = backend.random_normal((n_samples, n_anchors))
        decoder = backend.random_normal((n_anchors, d_target))
        density_weights = backend.zeros((n_samples,))
        backend.eval(delta_relative, decoder, density_weights)

        delta_activations = decode_to_activation_space(
            delta_relative=delta_relative,
            decoder=decoder,
            density_weights=density_weights,
            backend=backend,
        )

        # All zeros weights should produce zero output
        mean_abs = backend.mean(backend.abs(delta_activations))
        backend.eval(mean_abs)
        assert float(backend.to_scalar(mean_abs)) < 1e-6

    def test_decode_full_weights(self, backend):
        """Unit weights should pass through the full delta."""
        n_samples = 16
        n_anchors = 8
        d_target = 32

        delta_relative = backend.random_normal((n_samples, n_anchors))
        decoder = backend.random_normal((n_anchors, d_target))
        density_weights = backend.ones((n_samples,))
        backend.eval(delta_relative, decoder, density_weights)

        delta_activations = decode_to_activation_space(
            delta_relative=delta_relative,
            decoder=decoder,
            density_weights=density_weights,
            backend=backend,
        )

        # With unit weights, should equal delta_relative @ decoder
        expected = backend.matmul(delta_relative, decoder)
        backend.eval(expected)

        diff = backend.mean(backend.abs(delta_activations - expected))
        backend.eval(diff)
        assert float(backend.to_scalar(diff)) < 1e-5

    def test_decode_half_weights(self, backend):
        """Half weights should scale output by half."""
        n_samples = 16
        n_anchors = 8
        d_target = 32

        delta_relative = backend.random_normal((n_samples, n_anchors))
        decoder = backend.random_normal((n_anchors, d_target))
        backend.eval(delta_relative, decoder)

        full_weights = backend.ones((n_samples,))
        half_weights = backend.full((n_samples,), 0.5)
        backend.eval(full_weights, half_weights)

        delta_full = decode_to_activation_space(
            delta_relative=delta_relative,
            decoder=decoder,
            density_weights=full_weights,
            backend=backend,
        )

        delta_half = decode_to_activation_space(
            delta_relative=delta_relative,
            decoder=decoder,
            density_weights=half_weights,
            backend=backend,
        )

        # Half weights should produce half the magnitude
        ratio = backend.mean(backend.abs(delta_half)) / backend.mean(backend.abs(delta_full))
        backend.eval(ratio)
        assert abs(float(backend.to_scalar(ratio)) - 0.5) < 0.01


class TestAnchorDecoderRoundtrip:
    """Tests for decoder roundtrip properties."""

    def test_decoder_then_decode(self, backend):
        """Compute decoder then decode should produce valid output."""
        n_samples = 16
        n_anchors = 8
        d_target = 32

        target_relative_rep = backend.random_normal((n_samples, n_anchors))
        target_activations = backend.random_normal((n_samples, d_target))
        backend.eval(target_relative_rep, target_activations)

        # Compute decoder
        decoder, _ = compute_anchor_decoder(
            target_relative_rep=target_relative_rep,
            target_activations=target_activations,
            backend=backend,
        )

        # Decode with unit weights
        delta_relative = backend.random_normal((n_samples, n_anchors))
        density_weights = backend.ones((n_samples,))
        backend.eval(delta_relative, density_weights)

        delta_activations = decode_to_activation_space(
            delta_relative=delta_relative,
            decoder=decoder,
            density_weights=density_weights,
            backend=backend,
        )

        assert all_finite(delta_activations, backend)
        assert backend.shape(delta_activations) == (n_samples, d_target)


class TestAnchorDecoderMathematicalProperties:
    """Hypothesis-based tests for mathematical invariants."""

    @given(
        n_samples=st.integers(min_value=8, max_value=32),
        n_anchors=st.integers(min_value=4, max_value=16),
        d_target=st.integers(min_value=8, max_value=32),
    )
    @settings(max_examples=5, deadline=None)
    def test_decoder_shape_correct(self, n_samples, n_anchors, d_target):
        """Decoder shape should be [n_anchors, d_target]."""
        backend = get_default_backend()

        target_relative_rep = backend.random_normal((n_samples, n_anchors))
        target_activations = backend.random_normal((n_samples, d_target))
        backend.eval(target_relative_rep, target_activations)

        decoder, _ = compute_anchor_decoder(
            target_relative_rep=target_relative_rep,
            target_activations=target_activations,
            backend=backend,
        )

        assert backend.shape(decoder) == (n_anchors, d_target)

    @given(
        n_samples=st.integers(min_value=8, max_value=32),
        n_anchors=st.integers(min_value=4, max_value=16),
        d_target=st.integers(min_value=8, max_value=32),
    )
    @settings(max_examples=5, deadline=None)
    def test_decode_output_shape_correct(self, n_samples, n_anchors, d_target):
        """Decoded output shape should be [n_samples, d_target]."""
        backend = get_default_backend()

        delta_relative = backend.random_normal((n_samples, n_anchors))
        decoder = backend.random_normal((n_anchors, d_target))
        density_weights = backend.ones((n_samples,))
        backend.eval(delta_relative, decoder, density_weights)

        delta_activations = decode_to_activation_space(
            delta_relative=delta_relative,
            decoder=decoder,
            density_weights=density_weights,
            backend=backend,
        )

        assert backend.shape(delta_activations) == (n_samples, d_target)

    @given(
        n_samples=st.integers(min_value=8, max_value=32),
        n_anchors=st.integers(min_value=4, max_value=16),
        d_target=st.integers(min_value=8, max_value=32),
    )
    @settings(max_examples=5, deadline=None)
    def test_decoder_finite(self, n_samples, n_anchors, d_target):
        """Decoder should always be finite."""
        backend = get_default_backend()

        target_relative_rep = backend.random_normal((n_samples, n_anchors))
        target_activations = backend.random_normal((n_samples, d_target))
        backend.eval(target_relative_rep, target_activations)

        decoder, error = compute_anchor_decoder(
            target_relative_rep=target_relative_rep,
            target_activations=target_activations,
            backend=backend,
        )

        assert all_finite(decoder, backend)
        assert error >= 0.0
