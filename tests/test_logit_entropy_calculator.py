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

"""Tests for LogitEntropyCalculator (requires backend)."""

import pytest

from modelcypher.core.domain.entropy.logit_entropy_calculator import (
    LogitEntropyCalculator,
    LogitEntropySample,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    log_scalar,
)


def _eps(backend, *values: float) -> float:
    return division_epsilon(backend, backend.array(list(values) or [1.0]))


class TestLogitEntropyCalculator:
    """Tests for LogitEntropyCalculator."""

    def test_initialization(self, any_backend):
        """Should initialize without configuration knobs."""
        calc = LogitEntropyCalculator(backend=any_backend)

        assert calc.epsilon is None

    def test_compute_uniform_distribution(self, any_backend):
        """Uniform logits should have high entropy."""
        calc = LogitEntropyCalculator(backend=any_backend)

        vocab_size = 100
        logits = any_backend.zeros((vocab_size,))

        entropy, variance = calc.compute(logits)

        expected_entropy = log_scalar(float(vocab_size), any_backend)
        assert abs(entropy - expected_entropy) < _eps(any_backend, entropy, expected_entropy)
        assert abs(variance - 0.0) < _eps(any_backend, variance, 0.0)

    def test_compute_peaked_distribution(self, any_backend):
        """Peaked logits should have low entropy."""
        calc = LogitEntropyCalculator(backend=any_backend)

        vocab_size = 100
        values = [0.0] * vocab_size
        values[0] = 100.0
        logits = any_backend.array(values)

        entropy, _ = calc.compute(logits)

        assert entropy <= _eps(any_backend, entropy, 0.0)

    def test_flatten_to_vocab_1d(self, any_backend):
        """1D input should pass through."""
        calc = LogitEntropyCalculator(backend=any_backend)
        logits = any_backend.array([1.0, 2.0, 3.0])

        result = calc._flatten_to_vocab(logits)

        assert result.shape == (3,)

    def test_flatten_to_vocab_2d(self, any_backend):
        """2D input [batch, vocab] should extract batch 0."""
        calc = LogitEntropyCalculator(backend=any_backend)
        logits = any_backend.zeros((2, 100))

        result = calc._flatten_to_vocab(logits)

        assert result.shape == (100,)

    def test_flatten_to_vocab_3d(self, any_backend):
        """3D input [batch, seq, vocab] should extract last token."""
        calc = LogitEntropyCalculator(backend=any_backend)
        logits = any_backend.zeros((2, 5, 100))

        result = calc._flatten_to_vocab(logits)

        assert result.shape == (100,)

    def test_compute_with_skip_variance(self, any_backend):
        """Should return 0 variance when skipped."""
        calc = LogitEntropyCalculator(backend=any_backend)
        logits = any_backend.zeros((100,))

        entropy, variance = calc.compute(logits, skip_variance=True)

        assert variance == 0.0
        assert entropy > 0

    def test_compute_batch(self, any_backend):
        """Should compute entropy for batch of logits."""
        calc = LogitEntropyCalculator(backend=any_backend)

        # Uniform distribution
        uniform = any_backend.zeros((100,))
        # Peaked distribution (one high value)
        peaked_vals = [0.0] * 100
        peaked_vals[0] = 100.0
        peaked = any_backend.array(peaked_vals)

        batch = [uniform, peaked]

        results = calc.compute_batch(batch)

        assert len(results) == 2
        assert results[0][0] > results[1][0]

    def test_compute_batch_empty(self, any_backend):
        """Should handle empty batch."""
        calc = LogitEntropyCalculator(backend=any_backend)

        results = calc.compute_batch([])

        assert results == []


class TestLogitEntropySample:
    """Tests for LogitEntropySample."""

    def test_from_computation(self, any_backend):
        """Should create sample from computed values."""
        sample = LogitEntropySample.from_computation(
            entropy=2.5,
            variance=0.5,
            token_start=0,
            token_end=10,
            latency_ms=5.0,
            source="test",
        )

        assert sample.logit_entropy == 2.5
        assert sample.logit_variance == 0.5
        assert sample.latency_ms == 5.0
        assert sample.source == "test"
        assert sample.window_id is not None


class TestEdgeCases:
    """Edge case tests for numerical stability."""

    def test_compute_with_inf_logits_does_not_crash(self, any_backend):
        """Compute should complete without raising on inf input."""
        calc = LogitEntropyCalculator(backend=any_backend)
        logits = any_backend.array([float("inf"), 0.0, -1.0])

        calc.compute(logits)
