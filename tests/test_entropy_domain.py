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

import math

import pytest

# Attempt MLX import - skip module entirely if unavailable
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore

# Skip all tests in this module if MLX unavailable
pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available (requires Apple Silicon)")
from modelcypher.core.domain.entropy.conflict_score import ConflictScoreCalculator
from modelcypher.core.domain.entropy.entropy_tracker import EntropyTracker
from modelcypher.core.domain.entropy.logit_entropy_calculator import (
    EntropyLevel,
    EntropyThresholds,
    LogitEntropyCalculator,
    LogitEntropySample,
)
from modelcypher.core.domain.entropy.metrics_ring_buffer import MetricsRingBuffer

# --- LogitEntropyCalculator Tests ---


def test_logit_entropy_calculator_uniform():
    """Uniform distribution should have maximum entropy."""
    # 32K vocab
    vocab_size = 32768
    logits = mx.zeros((vocab_size,))
    calculator = LogitEntropyCalculator()

    entropy, variance = calculator.compute(logits)

    # Entropy should be ln(vocab_size)
    assert entropy == pytest.approx(math.log(vocab_size), rel=1e-5)
    # Variance of zeros should be 0
    assert variance == pytest.approx(0.0)


def test_logit_entropy_calculator_delta():
    """One-hot distribution (delta) should have zero entropy."""
    vocab_size = 100
    logits = mx.array([-1e9] * vocab_size)
    logits[0] = 1e9  # Massive spike at index 0

    calculator = LogitEntropyCalculator()
    entropy, variance = calculator.compute(logits)

    assert entropy == pytest.approx(0.0, abs=1e-5)


def test_logit_entropy_classification():
    calculator = LogitEntropyCalculator()

    assert calculator.classify(0.5) == EntropyLevel.LOW
    assert calculator.classify(2.0) == EntropyLevel.MODERATE
    assert calculator.classify(3.5) == EntropyLevel.HIGH


def test_logit_entropy_circuit_breaker():
    calculator = LogitEntropyCalculator()
    assert calculator.should_trip_circuit_breaker(4.5) is True
    assert calculator.should_trip_circuit_breaker(2.0) is False


def test_logit_entropy_batch():
    calculator = LogitEntropyCalculator()
    logits_batch = [mx.zeros((10,)), mx.ones((10,))]
    results = calculator.compute_batch(logits_batch)

    assert len(results) == 2
    assert results[0][0] == pytest.approx(math.log(10))


# --- ConflictScoreCalculator Tests ---


def test_conflict_score_calculation():
    """Test conflict score with disagreeing distributions."""
    # Base model prefers token 0, adapted prefers token 1
    base_logits = mx.array([10.0, 0.0, 0.0])
    adapted_logits = mx.array([0.0, 10.0, 0.0])

    # Use top_k=1 so token 1 is NOT in base model's top-K (only token 0 is)
    calculator = ConflictScoreCalculator(top_k=1)
    # sampled_token=1 means we sampled a token NOT in base model's top-1
    result = calculator.compute(base_logits, adapted_logits, sampled_token=1)

    # High KL divergence expected due to disagreement
    assert result.mean_kl > 0.5
    # base_approval_rate should be 0 (token 1 is not in base's top-1)
    assert result.base_approval_rate == 0.0
    # Conflict score = mean_kl * (1 - approval_rate) = mean_kl * 1 = mean_kl > 0
    assert result.conflict_score > 0.0


def test_conflict_score_agreement():
    """Test conflict score with identical distributions."""
    logits = mx.array([10.0, 0.0, 0.0])
    calculator = ConflictScoreCalculator(top_k=3)
    # sampled_token=0 is in the top-K of both
    result = calculator.compute(logits, logits, sampled_token=0)

    # KL divergence should be ~0 for identical distributions
    assert result.mean_kl == pytest.approx(0.0, abs=1e-3)
    # Approval rate should be 1.0 (sampled token in base's top-K)
    assert result.base_approval_rate == 1.0
    # Conflict score = KL * (1 - 1) = 0
    assert result.conflict_score == pytest.approx(0.0, abs=1e-3)


# --- EntropyTracker Tests ---


def test_entropy_tracker_session():
    """Test EntropyTracker session management."""
    import asyncio

    from modelcypher.core.domain.entropy.entropy_tracker import EntropyTrackerConfig

    config = EntropyTrackerConfig(window_size=5)
    tracker = EntropyTracker(config=config)

    # Start a session
    tracker.start_session()
    assert tracker.is_session_active

    # Record some entropy values using async method
    async def record_values():
        for i in range(5):
            await tracker.record_entropy(entropy=2.0, variance=0.1, token_index=i)

    asyncio.run(record_values())

    # End session returns an EntropySample
    sample = tracker.end_session()
    assert sample is not None
    assert not tracker.is_session_active


def test_entropy_tracker_state_classification():
    """Test EntropyTracker tracks raw entropy/variance values correctly."""
    import asyncio

    from modelcypher.core.domain.entropy.entropy_tracker import EntropyTrackerConfig

    config = EntropyTrackerConfig(window_size=10)
    tracker = EntropyTracker(config=config)
    tracker.start_session()

    # Record high entropy values
    async def record_high_entropy():
        for i in range(5):
            await tracker.record_entropy(entropy=4.0, variance=0.1, token_index=i)

    asyncio.run(record_high_entropy())

    # Should have high entropy (raw value IS the state)
    assert tracker.current_entropy >= 3.5  # Distress threshold
    assert tracker.current_variance <= 0.2  # Low variance = distress signature
    # requires_caution should be True for high-entropy states
    assert tracker.requires_caution
    tracker.end_session()


# --- MetricsRingBuffer Tests ---


def test_metrics_ring_buffer_wraparound():
    """Test MetricsRingBuffer wraps around correctly."""
    buffer = MetricsRingBuffer(capacity=3)
    # Use append_values which creates MetricSample objects
    buffer.append_values(timestamp=1.0, loss=1.0)
    buffer.append_values(timestamp=2.0, loss=2.0)
    buffer.append_values(timestamp=3.0, loss=3.0)
    buffer.append_values(timestamp=4.0, loss=4.0)  # Should overwrite first

    points = buffer.all_points()
    assert len(points) == 3
    # Should contain the last 3 samples (loss 2.0, 3.0, 4.0)
    losses = [p.loss for p in points]
    assert 4.0 in losses
    assert 1.0 not in losses


def test_metrics_ring_buffer_stats():
    """Test MetricsRingBuffer tracks stats correctly."""
    buffer = MetricsRingBuffer(capacity=10)
    for v in [10, 20, 30]:
        buffer.append_values(timestamp=float(v), loss=float(v))

    # Check count
    assert buffer.count == 3
    # Check max_y (derived from max loss/entropy)
    assert buffer.max_y >= 30.0


# --- EntropyWindow Tests ---


def test_entropy_window_sliding():
    """Test EntropyWindow detects high entropy conditions."""
    from modelcypher.core.domain.entropy.entropy_window import (
        EntropyWindow,
        EntropyWindowConfig,
    )

    config = EntropyWindowConfig(
        window_size=5,
        high_entropy_threshold=3.0,
        sustained_high_count=1,  # Trip on any high sample
    )
    window = EntropyWindow(config=config)

    # Add samples with one high entropy spike
    for i, val in enumerate([1.0, 1.1, 1.2, 5.0, 1.1]):
        status = window.add(entropy=val, variance=0.1, token_index=i)

    # The circuit breaker should trip due to the 5.0 spike
    assert status.should_trip_circuit_breaker is True


# --- LogitEntropySample Tests ---


def test_logit_entropy_sample_creation():
    calculator = LogitEntropyCalculator()
    sample = LogitEntropySample.from_computation(
        entropy=2.2, variance=1.5, token_start=0, token_end=1, calculator=calculator
    )

    assert sample.level == EntropyLevel.MODERATE
    assert sample.logit_entropy == 2.2


def test_logit_entropy_threshold_customization():
    thresholds = EntropyThresholds(low=1.0, high=2.0, circuit_breaker=3.0)
    calculator = LogitEntropyCalculator()

    assert calculator.classify(1.5, thresholds=thresholds) == EntropyLevel.MODERATE
    assert calculator.classify(2.5, thresholds=thresholds) == EntropyLevel.HIGH
