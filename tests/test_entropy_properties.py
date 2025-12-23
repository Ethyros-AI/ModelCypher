"""Property-based tests for entropy calculations using Hypothesis."""

import math
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
import mlx.core as mx

from modelcypher.core.domain.entropy.logit_entropy_calculator import (
    LogitEntropyCalculator,
    EntropyLevel,
    EntropyThresholds,
)


# Strategy for generating valid logit arrays
@st.composite
def logits_array(draw, size=st.integers(2, 1000)):
    """Generate a logits array with random floats."""
    n = draw(size)
    values = [draw(st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False))
              for _ in range(n)]
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
    def test_classification_is_deterministic(self, entropy_value):
        """Classification should be deterministic for same input."""
        calc = LogitEntropyCalculator()

        level1 = calc.classify(entropy_value)
        level2 = calc.classify(entropy_value)

        assert level1 == level2

    @given(st.floats(min_value=0, max_value=1.49, allow_nan=False, allow_infinity=False))
    @settings(max_examples=30, deadline=None)
    def test_low_entropy_classification(self, entropy_value):
        """Values below low threshold should classify as LOW."""
        calc = LogitEntropyCalculator()

        level = calc.classify(entropy_value)

        assert level == EntropyLevel.LOW

    @given(st.floats(min_value=3.01, max_value=100, allow_nan=False, allow_infinity=False))
    @settings(max_examples=30, deadline=None)
    def test_high_entropy_classification(self, entropy_value):
        """Values above high threshold should classify as HIGH."""
        calc = LogitEntropyCalculator()

        level = calc.classify(entropy_value)

        assert level == EntropyLevel.HIGH

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

    @given(st.floats(min_value=4.01, max_value=100, allow_nan=False, allow_infinity=False))
    @settings(max_examples=30, deadline=None)
    def test_circuit_breaker_trips_above_threshold(self, entropy_value):
        """Circuit breaker should trip above threshold."""
        calc = LogitEntropyCalculator()

        assert calc.should_trip_circuit_breaker(entropy_value)

    @given(st.floats(min_value=0, max_value=3.99, allow_nan=False, allow_infinity=False))
    @settings(max_examples=30, deadline=None)
    def test_circuit_breaker_does_not_trip_below_threshold(self, entropy_value):
        """Circuit breaker should not trip below threshold."""
        calc = LogitEntropyCalculator()

        assert not calc.should_trip_circuit_breaker(entropy_value)


class TestEntropyThresholdsProperties:
    """Property-based tests for EntropyThresholds."""

    @given(
        st.floats(min_value=0.1, max_value=5, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.1, max_value=5, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.1, max_value=5, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30, deadline=None)
    def test_thresholds_preserve_values(self, low, high, circuit_breaker):
        """Threshold values should be preserved."""
        thresholds = EntropyThresholds(low=low, high=high, circuit_breaker=circuit_breaker)

        assert thresholds.low == low
        assert thresholds.high == high
        assert thresholds.circuit_breaker == circuit_breaker
