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

"""Property-based tests for entropy calculations (requires MLX)."""

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Attempt MLX import - skip module entirely if unavailable
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore

# Skip all tests in this module if MLX unavailable
pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available (requires Apple Silicon)")

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.entropy.logit_entropy_calculator import (
    LogitEntropyCalculator,
)
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon


def _eps(*values: float) -> float:
    backend = get_default_backend()
    return division_epsilon(backend, backend.array(list(values) or [1.0]))


# Strategy for generating valid logit arrays
@st.composite
def logits_array(draw, size=st.integers(2, 1000)):
    """Generate a logits array with random floats."""
    n = draw(size)
    values = [
        draw(st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False))
        for _ in range(n)
    ]
    return mx.array(values)


@st.composite
def uniform_logits(draw, size=st.integers(2, 100)):
    """Generate uniform logits (all same value)."""
    n = draw(size)
    value = draw(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False))
    return mx.full((n,), value)


@st.composite
def peaked_logits(draw, size=st.integers(2, 100)):
    """Generate peaked logits (one high, rest low)."""
    n = draw(size)
    base_value = draw(st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False))
    peak_value = base_value + draw(st.floats(min_value=10, max_value=50))
    peak_idx = draw(st.integers(0, n - 1))

    values = [base_value] * n
    values[peak_idx] = peak_value
    return mx.array(values)


class TestEntropyProperties:
    """Property-based tests for entropy calculation."""

    @given(logits_array())
    @settings(max_examples=50, deadline=None)
    def test_entropy_is_non_negative(self, logits):
        """Entropy should always be non-negative."""
        calc = LogitEntropyCalculator(top_k=None)

        entropy, _ = calc.compute(logits)

        eps = _eps(entropy, 0.0)
        assert entropy >= -eps

    @given(uniform_logits())
    @settings(max_examples=30, deadline=None)
    def test_uniform_distribution_maximum_entropy(self, logits):
        """Uniform distribution should have maximum entropy."""
        calc = LogitEntropyCalculator(top_k=None)

        entropy, _ = calc.compute(logits)

        # Maximum entropy for n outcomes is ln(n)
        n = logits.shape[0]
        max_entropy = math.log(n)

        # Should be close to maximum
        eps = _eps(entropy, max_entropy)
        assert abs(entropy - max_entropy) <= eps

    @given(peaked_logits())
    @settings(max_examples=30, deadline=None)
    def test_peaked_distribution_low_entropy(self, logits):
        """Highly peaked distribution should have low entropy."""
        calc = LogitEntropyCalculator(top_k=None)

        entropy, _ = calc.compute(logits)

        n = logits.shape[0]
        uniform = mx.zeros((n,))
        entropy_uniform, _ = calc.compute(uniform)
        eps = _eps(entropy, entropy_uniform)
        assert entropy <= entropy_uniform + eps

    @given(logits_array())
    @settings(max_examples=50, deadline=None)
    def test_variance_is_non_negative(self, logits):
        """Variance should always be non-negative."""
        calc = LogitEntropyCalculator(top_k=None)

        _, variance = calc.compute(logits)

        eps = _eps(variance, 0.0)
        assert variance >= -eps

    @given(logits_array(), logits_array())
    @settings(max_examples=30, deadline=None)
    def test_batch_compute_length_matches(self, logits_a, logits_b):
        """Batch compute should return correct number of results."""
        calc = LogitEntropyCalculator(top_k=None)

        batch = [logits_a, logits_b]
        results = calc.compute_batch(batch)

        assert len(results) == 2

    @given(st.lists(logits_array(size=st.just(100)), min_size=0, max_size=5))
    @settings(max_examples=30, deadline=None)
    def test_batch_compute_empty_batch(self, batch):
        """Batch compute should handle any size batch."""
        calc = LogitEntropyCalculator(top_k=None)

        results = calc.compute_batch(batch)

        assert len(results) == len(batch)

    @given(logits_array())
    @settings(max_examples=50, deadline=None)
    def test_skip_variance_returns_zero(self, logits):
        """When skipping variance, should return 0."""
        calc = LogitEntropyCalculator(top_k=None)

        _, variance = calc.compute(logits, skip_variance=True)

        eps = _eps(variance, 0.0)
        assert abs(variance - 0.0) <= eps
