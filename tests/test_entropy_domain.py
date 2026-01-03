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

# Attempt MLX import - skip module entirely if unavailable
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available (requires Apple Silicon)")
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.entropy.conflict_score import ConflictScoreCalculator
from modelcypher.core.domain.entropy.entropy_tracker import (
    EntropyTracker,
    EntropyTrackerConfig,
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


def _create_tracker_config(window_size: int) -> EntropyTrackerConfig:
    return EntropyTrackerConfig(
        top_k=10,
        window_size=window_size,
        emit_interval=1,
        source="EntropyTracker",
    )


# --- LogitEntropyCalculator Tests ---


def test_logit_entropy_calculator_uniform():
    """Uniform distribution should have maximum entropy."""
    vocab_size = 32768
    logits = mx.zeros((vocab_size,))
    calculator = LogitEntropyCalculator(top_k=None)

    entropy, variance = calculator.compute(logits)

    expected_entropy = log_scalar(float(vocab_size), get_default_backend())
    assert abs(entropy - expected_entropy) <= _eps(entropy, expected_entropy)
    assert abs(variance - 0.0) <= _eps(variance, 0.0)


def test_logit_entropy_calculator_delta():
    """One-hot distribution (delta) should have zero entropy."""
    vocab_size = 100
    logits = mx.array([-1e9] * vocab_size)
    logits[0] = 1e9

    calculator = LogitEntropyCalculator(top_k=None)
    entropy, _ = calculator.compute(logits)

    assert abs(entropy - 0.0) <= _eps(entropy, 0.0)


def test_logit_entropy_batch():
    calculator = LogitEntropyCalculator(top_k=None)
    logits_batch = [mx.zeros((10,)), mx.ones((10,))]
    results = calculator.compute_batch(logits_batch)

    assert len(results) == 2
    expected_entropy = log_scalar(10.0, get_default_backend())
    assert abs(results[0][0] - expected_entropy) <= _eps(results[0][0], expected_entropy)


# --- ConflictScoreCalculator Tests ---


def test_conflict_score_calculation():
    """Test conflict score with disagreeing distributions."""
    base_logits = mx.array([10.0, 0.0, 0.0])
    adapted_logits = mx.array([0.0, 10.0, 0.0])

    calculator = ConflictScoreCalculator()
    result = calculator.compute(base_logits, adapted_logits, sampled_token=1)

    eps = _eps(result.mean_kl, result.conflict_score, result.base_frontier_rate)
    assert result.mean_kl > eps
    assert abs(result.base_frontier_rate - 0.0) <= eps
    assert result.conflict_score > eps


def test_conflict_score_agreement():
    """Test conflict score with identical distributions."""
    logits = mx.array([10.0, 0.0, 0.0])
    calculator = ConflictScoreCalculator()
    result = calculator.compute(logits, logits, sampled_token=0)

    assert abs(result.mean_kl - 0.0) <= _eps(result.mean_kl, 0.0)
    eps = _eps(result.base_frontier_rate, 1.0)
    assert abs(result.base_frontier_rate - 1.0) <= eps
    assert abs(result.conflict_score - 0.0) <= _eps(result.conflict_score, 0.0)


# --- EntropyTracker Tests ---


def test_entropy_tracker_session():
    """Test EntropyTracker session management."""
    import asyncio

    config = _create_tracker_config(window_size=5)
    baseline = _create_test_baseline()
    tracker = EntropyTracker(baseline=baseline, config=config)

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

    config = _create_tracker_config(window_size=10)
    baseline = _create_test_baseline()
    tracker = EntropyTracker(baseline=baseline, config=config)
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
    eps = _eps(buffer.max_y, 30.0)
    assert buffer.max_y >= 30.0 - eps


# --- EntropyWindow Tests ---


def test_entropy_window_sliding():
    """Test EntropyWindow sliding statistics."""
    from modelcypher.core.domain.entropy.entropy_window import (
        EntropyWindow,
        EntropyWindowConfig,
    )

    config = EntropyWindowConfig(window_size=5)
    window = EntropyWindow(config=config)

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

    eps = _eps(sample.logit_entropy, sample.top_k_variance)
    assert abs(sample.logit_entropy - 2.2) <= eps
    assert abs(sample.top_k_variance - 1.5) <= eps
    assert sample.token_start == 0
    assert sample.token_end == 1
