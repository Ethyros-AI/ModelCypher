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

"""Property-based tests for dynamics module (requires MLX or JAX backend)."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

# Get default backend - skip tests if no backend available
try:
    backend = get_default_backend()
    HAS_BACKEND = True
except Exception:
    HAS_BACKEND = False
    backend = None  # type: ignore

pytestmark = pytest.mark.skipif(not HAS_BACKEND, reason="No backend available")

if HAS_BACKEND:
    from modelcypher.core.domain.dynamics.regime_state_detector import RegimeStateDetector


def _eps(*values: float) -> float:
    if backend is None:
        return 0.0
    return division_epsilon(backend, backend.array(list(values) or [1.0]))


def _log_scalar(value: float) -> float:
    if backend is None:
        return 0.0
    arr = backend.array([value])
    log_val = backend.log(arr)
    backend.eval(log_val)
    return float(backend.to_scalar(log_val))

# Strategy for generating valid logit arrays
@st.composite
def logits_array(draw, min_size=2, max_size=1000):
    """Generate a logits array with random floats."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    values = [
        draw(st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False))
        for _ in range(n)
    ]
    return backend.array(values)


@st.composite
def temperature(draw):
    """Generate a valid temperature value."""
    min_temp = _eps(1.0)
    return draw(st.floats(min_value=min_temp, max_value=10.0, allow_nan=False, allow_infinity=False))


@pytest.fixture
def detector():
    """Create a RegimeStateDetector instance."""
    return RegimeStateDetector(backend=backend)


class TestRegimeStateDetectorProperties:
    """Property-based tests for RegimeStateDetector computations."""

    @given(logits_array(), temperature())
    @settings(max_examples=50, deadline=None)
    def test_compute_entropy_non_negative(self, logits, temp):
        """Entropy should always be non-negative."""
        detector = RegimeStateDetector(backend=backend)
        entropy = detector.compute_entropy(logits, temp)
        eps = _eps(entropy, 0.0)
        assert entropy >= -eps, f"Entropy should be >= 0, got {entropy}"

    @given(logits_array(), temperature())
    @settings(max_examples=50, deadline=None)
    def test_compute_entropy_bounded_by_log_vocab(self, logits, temp):
        """Entropy should be bounded by log(vocab_size)."""
        detector = RegimeStateDetector(backend=backend)
        entropy = detector.compute_entropy(logits, temp)
        vocab_size = logits.shape[-1]
        max_entropy = _log_scalar(float(vocab_size))
        # Allow small numerical tolerance
        eps = _eps(entropy, max_entropy)
        assert entropy <= max_entropy + eps, f"Entropy {entropy} exceeds max {max_entropy}"

    @given(logits_array(), temperature())
    @settings(max_examples=50, deadline=None)
    def test_compute_logit_variance_non_negative(self, logits, temp):
        """Logit variance should always be non-negative."""
        detector = RegimeStateDetector(backend=backend)
        variance = detector.compute_logit_variance(logits, temp)
        eps = _eps(variance, 0.0)
        assert variance >= -eps, f"Variance should be >= 0, got {variance}"

    @given(logits_array(), temperature())
    @settings(max_examples=50, deadline=None)
    def test_effective_vocabulary_size_at_least_one(self, logits, temp):
        """Effective vocabulary size should be at least 1."""
        detector = RegimeStateDetector(backend=backend)
        eff_vocab = detector.effective_vocabulary_size(logits, temp)
        eps = _eps(eff_vocab, 1.0)
        assert eff_vocab >= 1.0 - eps, f"Effective vocab should be >= 1, got {eff_vocab}"

    @given(logits_array(), temperature())
    @settings(max_examples=50, deadline=None)
    def test_effective_vocabulary_size_bounded_by_vocab(self, logits, temp):
        """Effective vocabulary size should not exceed actual vocab size."""
        detector = RegimeStateDetector(backend=backend)
        eff_vocab = detector.effective_vocabulary_size(logits, temp)
        vocab_size = logits.shape[-1]
        eps = _eps(eff_vocab, float(vocab_size))
        assert eff_vocab <= float(vocab_size) + eps, (
            f"Effective vocab {eff_vocab} exceeds vocab {vocab_size}"
        )

    @given(logits_array())
    @settings(max_examples=50, deadline=None)
    def test_logit_statistics_variance_equals_std_squared(self, logits):
        """Variance should equal std^2 (within numerical tolerance)."""
        detector = RegimeStateDetector(backend=backend)
        mean, variance, std = detector.compute_logit_statistics(logits)
        expected_variance = std * std
        eps = _eps(variance, expected_variance)
        assert abs(variance - expected_variance) <= eps, (
            f"Variance {variance} != std^2 {expected_variance}"
        )

    @given(logits_array())
    @settings(max_examples=50, deadline=None)
    def test_logit_statistics_std_non_negative(self, logits):
        """Standard deviation should be non-negative."""
        detector = RegimeStateDetector(backend=backend)
        _, _, std = detector.compute_logit_statistics(logits)
        eps = _eps(std, 0.0)
        assert std >= -eps, f"Std should be >= 0, got {std}"

    def test_zero_temperature_returns_zero_entropy(self):
        """Zero temperature should return zero entropy."""
        detector = RegimeStateDetector(backend=backend)
        logits = backend.array([1.0, 2.0, 3.0])
        entropy = detector.compute_entropy(logits, temperature=0.0)
        eps = _eps(entropy, 0.0)
        assert abs(entropy - 0.0) <= eps

    def test_zero_temperature_returns_zero_variance(self):
        """Zero temperature should return zero variance."""
        detector = RegimeStateDetector(backend=backend)
        logits = backend.array([1.0, 2.0, 3.0])
        variance = detector.compute_logit_variance(logits, temperature=0.0)
        eps = _eps(variance, 0.0)
        assert abs(variance - 0.0) <= eps

    def test_uniform_logits_maximize_entropy(self):
        """Uniform logits should produce maximum entropy (log n)."""
        detector = RegimeStateDetector(backend=backend)
        n = 100
        logits = backend.array([1.0] * n)  # All equal
        entropy = detector.compute_entropy(logits, temperature=1.0)
        expected = _log_scalar(float(n))
        eps = _eps(entropy, expected)
        assert abs(entropy - expected) <= eps, f"Expected {expected}, got {entropy}"
