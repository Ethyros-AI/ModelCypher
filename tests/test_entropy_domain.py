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

"""Entropy domain tests requiring MLX (Apple Silicon)."""

import pytest

from tests.conftest import HAS_MLX

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available (requires Apple Silicon)")
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.entropy.conflict_score import ConflictScoreCalculator
from modelcypher.core.domain.entropy.entropy_tracker import (
    EntropyTracker,
)
from modelcypher.core.domain.entropy.model_state_classifier import (
    CalibratedBaseline,
)
from modelcypher.core.domain.entropy.logit_entropy_calculator import (
    LogitEntropyCalculator,
    LogitEntropySample,
)
from modelcypher.core.domain.entropy.metrics_ring_buffer import MetricsRingBuffer
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    log_scalar,
)


def _eps(*values: float) -> float:
    backend = get_default_backend()
    return division_epsilon(backend, backend.array(list(values) or [1.0]))


def _create_test_baseline() -> CalibratedBaseline:
    """Create a calibrated baseline for testing."""
    return CalibratedBaseline(
        mean=2.5,
        std_dev=1.0,
        percentile_25=1.8,
        percentile_75=3.2,
        percentile_95=4.5,
        vocab_size=32768,
        model_id="test-model",
        sample_count=100,
    )


# --- LogitEntropyCalculator Tests ---


def test_logit_entropy_calculator_uniform():
    """Uniform distribution should have maximum entropy."""
    backend = get_default_backend()
    vocab_size = 32768
    logits = backend.zeros((vocab_size,))
    calculator = LogitEntropyCalculator()

    entropy, variance = calculator.compute(logits)

    expected_entropy = log_scalar(float(vocab_size), get_default_backend())
    assert abs(entropy - expected_entropy) <= _eps(entropy, expected_entropy)
    assert abs(variance - 0.0) <= _eps(variance, 0.0)


def test_logit_entropy_calculator_delta():
    """One-hot distribution (delta) should have zero entropy."""
    backend = get_default_backend()
    vocab_size = 100
    values = [-1e9] * vocab_size
    values[0] = 1e9
    logits = backend.array(values)

    calculator = LogitEntropyCalculator()
    entropy, _ = calculator.compute(logits)

    assert abs(entropy - 0.0) <= _eps(entropy, 0.0)


def test_logit_entropy_batch():
    backend = get_default_backend()
    calculator = LogitEntropyCalculator()
    logits_batch = [backend.zeros((10,)), backend.ones((10,))]
    results = calculator.compute_batch(logits_batch)

    assert len(results) == 2
    expected_entropy = log_scalar(10.0, get_default_backend())
    assert abs(results[0][0] - expected_entropy) <= _eps(results[0][0], expected_entropy)


# --- ConflictScoreCalculator Tests ---


def test_conflict_score_calculation():
    """Test conflict score with disagreeing distributions."""
    backend = get_default_backend()
    base_logits = backend.array([10.0, 0.0, 0.0])
    adapted_logits = backend.array([0.0, 10.0, 0.0])

    calculator = ConflictScoreCalculator()
    result = calculator.compute(base_logits, adapted_logits, sampled_token=1)

    expected_kl = calculator._compute_kl_divergence(adapted_logits, base_logits)
    expected_frontier_rate = 0.0
    expected_conflict = expected_kl * (1.0 - expected_frontier_rate)

    eps = _eps(result.mean_kl, expected_kl, expected_conflict)
    assert abs(result.mean_kl - expected_kl) <= eps
    assert abs(result.base_frontier_rate - expected_frontier_rate) <= eps
    assert abs(result.conflict_score - expected_conflict) <= eps


def test_conflict_score_agreement():
    """Test conflict score with identical distributions."""
    backend = get_default_backend()
    logits = backend.array([10.0, 0.0, 0.0])
    calculator = ConflictScoreCalculator()
    result = calculator.compute(logits, logits, sampled_token=0)

    expected_kl = calculator._compute_kl_divergence(logits, logits)
    expected_frontier_rate = 1.0
    expected_conflict = expected_kl * (1.0 - expected_frontier_rate)

    eps = _eps(result.mean_kl, expected_kl, expected_conflict, expected_frontier_rate)
    assert abs(result.mean_kl - expected_kl) <= eps
    assert abs(result.base_frontier_rate - expected_frontier_rate) <= eps
    assert abs(result.conflict_score - expected_conflict) <= eps


# --- EntropyTracker Tests ---


def test_entropy_tracker_session():
    """Test EntropyTracker session management."""
    import asyncio

    baseline = _create_test_baseline()
    tracker = EntropyTracker(baseline=baseline, source="test")

    tracker.start_session()
    assert tracker.is_session_active

    async def record_values():
        for i in range(5):
            await tracker.record_entropy(entropy=2.0, variance=0.1, token_index=i)

    asyncio.run(record_values())

    sample = tracker.end_session()
    assert sample is not None
    assert not tracker.is_session_active


def test_entropy_tracker_state_measurements():
    """EntropyTracker tracks raw entropy/variance/z-score values."""
    import asyncio

    baseline = _create_test_baseline()
    tracker = EntropyTracker(baseline=baseline, source="test")
    tracker.start_session()

    async def record_high_entropy():
        for i in range(5):
            await tracker.record_entropy(entropy=4.2, variance=0.1, token_index=i)

    asyncio.run(record_high_entropy())

    assert abs(tracker.current_entropy - 4.2) <= _eps(tracker.current_entropy, 4.2)
    assert abs(tracker.current_variance - 0.1) <= _eps(tracker.current_variance, 0.1)
    assert abs(tracker.current_z_score - 1.7) <= _eps(tracker.current_z_score, 1.7)
    tracker.end_session()


# --- MetricsRingBuffer Tests ---


def test_metrics_ring_buffer_wraparound():
    """Test MetricsRingBuffer wraps around correctly."""
    buffer = MetricsRingBuffer(capacity=3)
    buffer.append_values(timestamp=1.0, loss=1.0)
    buffer.append_values(timestamp=2.0, loss=2.0)
    buffer.append_values(timestamp=3.0, loss=3.0)
    buffer.append_values(timestamp=4.0, loss=4.0)

    points = buffer.all_points()
    assert len(points) == 3
    losses = [p.loss for p in points]
    assert 4.0 in losses
    assert 1.0 not in losses


def test_metrics_ring_buffer_stats():
    """Test MetricsRingBuffer tracks stats correctly."""
    buffer = MetricsRingBuffer(capacity=10)
    for v in [10, 20, 30]:
        buffer.append_values(timestamp=float(v), loss=float(v))

    assert buffer.count == 3
    expected_max = max(10.0, 20.0, 30.0)
    eps = _eps(buffer.max_y, expected_max)
    assert abs(buffer.max_y - expected_max) <= eps


# --- EntropyWindow Tests ---


def test_entropy_window_sliding():
    """Test EntropyWindow sliding statistics."""
    from modelcypher.core.domain.entropy.entropy_window import EntropyWindow

    window = EntropyWindow(sample_count=25)

    for i, val in enumerate([1.0, 1.1, 1.2, 5.0, 1.1]):
        status = window.add(entropy=val, variance=0.1, token_index=i)

    assert status.sample_count == 5
    assert abs(status.max_entropy - 5.0) <= _eps(status.max_entropy, 5.0)


# --- LogitEntropySample Tests ---


def test_logit_entropy_sample_creation():
    """Test LogitEntropySample creation from raw values."""
    sample = LogitEntropySample.from_computation(
        entropy=2.2, variance=1.5, token_start=0, token_end=1
    )

    eps = _eps(sample.logit_entropy, sample.logit_variance)
    assert abs(sample.logit_entropy - 2.2) <= eps
    assert abs(sample.logit_variance - 1.5) <= eps
    assert sample.token_start == 0
    assert sample.token_end == 1
