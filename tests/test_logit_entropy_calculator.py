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

"""Tests for LogitEntropyCalculator (requires MLX)."""

import pytest

# Attempt MLX import - skip module entirely if unavailable
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available (requires Apple Silicon)")

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.entropy.logit_entropy_calculator import (
    LogitEntropyCalculator,
    LogitEntropySample,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    log_scalar,
)


def _eps(*values: float) -> float:
    backend = get_default_backend()
    return division_epsilon(backend, backend.array(list(values) or [1.0]))


class TestLogitEntropyCalculator:
    """Tests for LogitEntropyCalculator."""

    def test_initialization(self):
        """Should initialize without configuration knobs."""
        calc = LogitEntropyCalculator()

        assert calc.epsilon is None

    def test_compute_uniform_distribution(self):
        """Uniform logits should have high entropy."""
        calc = LogitEntropyCalculator()

        vocab_size = 100
        logits = mx.zeros((vocab_size,))

        entropy, variance = calc.compute(logits)

        expected_entropy = log_scalar(float(vocab_size), get_default_backend())
        assert abs(entropy - expected_entropy) < _eps(entropy, expected_entropy)
        assert abs(variance - 0.0) < _eps(variance, 0.0)

    def test_compute_peaked_distribution(self):
        """Peaked logits should have low entropy."""
        calc = LogitEntropyCalculator()

        vocab_size = 100
        logits = mx.zeros((vocab_size,))
        logits = logits.at[0].add(100.0)

        entropy, _ = calc.compute(logits)

        assert entropy <= _eps(entropy, 0.0)

    def test_flatten_to_vocab_1d(self):
        """1D input should pass through."""
        calc = LogitEntropyCalculator()
        logits = mx.array([1.0, 2.0, 3.0])

        result = calc._flatten_to_vocab(logits)

        assert result.shape == (3,)

    def test_flatten_to_vocab_2d(self):
        """2D input [batch, vocab] should extract batch 0."""
        calc = LogitEntropyCalculator()
        logits = mx.zeros((2, 100))

        result = calc._flatten_to_vocab(logits)

        assert result.shape == (100,)

    def test_flatten_to_vocab_3d(self):
        """3D input [batch, seq, vocab] should extract last token."""
        calc = LogitEntropyCalculator()
        logits = mx.zeros((2, 5, 100))

        result = calc._flatten_to_vocab(logits)

        assert result.shape == (100,)

    def test_compute_with_skip_variance(self):
        """Should return 0 variance when skipped."""
        calc = LogitEntropyCalculator()
        logits = mx.zeros((100,))

        entropy, variance = calc.compute(logits, skip_variance=True)

        assert variance == 0.0
        assert entropy > 0

    def test_compute_batch(self):
        """Should compute entropy for batch of logits."""
        calc = LogitEntropyCalculator()

        batch = [
            mx.zeros((100,)),
            mx.zeros((100,)).at[0].add(100.0),
        ]

        results = calc.compute_batch(batch)

        assert len(results) == 2
        assert results[0][0] > results[1][0]

    def test_compute_batch_empty(self):
        """Should handle empty batch."""
        calc = LogitEntropyCalculator()

        results = calc.compute_batch([])

        assert results == []


class TestLogitEntropySample:
    """Tests for LogitEntropySample."""

    def test_from_computation(self):
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
        assert sample.top_k_variance == 0.5
        assert sample.latency_ms == 5.0
        assert sample.source == "test"
        assert sample.window_id is not None


class TestEdgeCases:
    """Edge case tests for numerical stability."""

    def test_compute_with_inf_logits_does_not_crash(self):
        """Compute should complete without raising on inf input."""
        calc = LogitEntropyCalculator(backend=get_default_backend())
        logits = mx.array([float("inf"), 0.0, -1.0])

        calc.compute(logits)
