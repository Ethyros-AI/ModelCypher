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

from modelcypher.core.domain.entropy.logit_entropy_calculator import (
    LogitEntropyCalculator,
)


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
        calc = LogitEntropyCalculator()

        entropy, _ = calc.compute(logits)

        assert entropy >= 0.0

    @given(uniform_logits())
    @settings(max_examples=30, deadline=None)
    def test_uniform_distribution_maximum_entropy(self, logits):
        """Uniform distribution should have maximum entropy."""
        calc = LogitEntropyCalculator()

        entropy, _ = calc.compute(logits)

        # Maximum entropy for n outcomes is ln(n)
        n = logits.shape[0]
        max_entropy = math.log(n)

        # Should be close to maximum
        assert entropy >= max_entropy * 0.9

    @given(peaked_logits())
    @settings(max_examples=30, deadline=None)
    def test_peaked_distribution_low_entropy(self, logits):
        """Highly peaked distribution should have low entropy."""
        calc = LogitEntropyCalculator()

        entropy, _ = calc.compute(logits)

        # Should be much lower than maximum
        n = logits.shape[0]
        max_entropy = math.log(n)

        assert entropy < max_entropy * 0.5

    @given(logits_array())
    @settings(max_examples=50, deadline=None)
    def test_variance_is_non_negative(self, logits):
        """Variance should always be non-negative."""
        calc = LogitEntropyCalculator()

        _, variance = calc.compute(logits)

        assert variance >= 0.0

    @given(st.floats(min_value=0, max_value=10, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50, deadline=None)
    def test_thresholds_are_monotonic(self, entropy_value):
        """Thresholds should follow low < high < circuit_breaker ordering.

        Tests the mathematical property that entropy thresholds are properly ordered.
        Classification is now done by comparing raw entropy values against these thresholds.
        """
        calc = LogitEntropyCalculator()
        thresholds = calc.thresholds

        # Verify threshold ordering - this is a structural invariant
        assert thresholds.low < thresholds.high < thresholds.circuit_breaker

        # Verify circuit breaker correctly identifies high entropy
        if entropy_value >= thresholds.circuit_breaker:
            assert calc.should_trip_circuit_breaker(entropy_value)
        else:
            assert not calc.should_trip_circuit_breaker(entropy_value)

    @given(logits_array(), logits_array())
    @settings(max_examples=30, deadline=None)
    def test_batch_compute_length_matches(self, logits_a, logits_b):
        """Batch compute should return correct number of results."""
        calc = LogitEntropyCalculator()

        batch = [logits_a, logits_b]
        results = calc.compute_batch(batch)

        assert len(results) == 2

    @given(st.lists(logits_array(size=st.just(100)), min_size=0, max_size=5))
    @settings(max_examples=30, deadline=None)
    def test_batch_compute_empty_batch(self, batch):
        """Batch compute should handle any size batch."""
        calc = LogitEntropyCalculator()

        results = calc.compute_batch(batch)

        assert len(results) == len(batch)

    @given(logits_array())
    @settings(max_examples=50, deadline=None)
    def test_skip_variance_returns_zero(self, logits):
        """When skipping variance, should return 0."""
        calc = LogitEntropyCalculator()

        _, variance = calc.compute(logits, skip_variance=True)

        assert variance == 0.0

    @given(st.floats(min_value=0, max_value=20, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50, deadline=None)
    def test_circuit_breaker_respects_threshold(self, entropy_value):
        """Circuit breaker should trip if and only if entropy >= threshold.

        Tests the precise boundary condition for circuit breaker activation.
        """
        calc = LogitEntropyCalculator()
        threshold = calc.thresholds.circuit_breaker

        should_trip = calc.should_trip_circuit_breaker(entropy_value)

        if entropy_value >= threshold:
            assert should_trip, f"Should trip at {entropy_value} >= {threshold}"
        else:
            assert not should_trip, f"Should not trip at {entropy_value} < {threshold}"
