import mlx.core as mx
import pytest
import math
from modelcypher.core.domain.entropy.logit_entropy_calculator import (
    LogitEntropyCalculator,
    EntropyLevel,
    EntropyThresholds,
    LogitEntropySample
)
from modelcypher.core.domain.entropy.conflict_score import ConflictScoreCalculator
from modelcypher.core.domain.entropy.entropy_tracker import EntropyTracker
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
    logits[0] = 1e9 # Massive spike at index 0
    
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
    # Mock logits
    source_logits = mx.array([10.0, 0.0, 0.0])
    target_logits = mx.array([0.0, 10.0, 0.0]) # Complete disagreement
    
    # We need to see how ConflictScoreCalculator is implemented
    # Assuming it computes KL divergence or similar
    from modelcypher.core.domain.entropy.conflict_score import ConflictScoreCalculator
    calculator = ConflictScoreCalculator()
    score = calculator.compute(source_logits, target_logits)
    
    assert score > 5.0 # High conflict


def test_conflict_score_agreement():
    logits = mx.array([10.0, 0.0, 0.0])
    calculator = ConflictScoreCalculator()
    score = calculator.compute(logits, logits)
    assert score == pytest.approx(0.0, abs=1e-3)


# --- EntropyTracker Tests ---

def test_entropy_tracker_moving_average():
    tracker = EntropyTracker(window_size=5)
    for _ in range(5):
        tracker.add_sample(2.0)
    
    assert tracker.moving_average == pytest.approx(2.0)
    
    tracker.add_sample(4.0)
    # Average of [2, 2, 2, 2, 4] = 12/5 = 2.4
    assert tracker.moving_average == pytest.approx(2.4)


def test_entropy_tracker_trend():
    tracker = EntropyTracker(window_size=10)
    # Increasing entropy trend
    for i in range(10):
        tracker.add_sample(float(i))
    
    assert tracker.trend > 0 # Upward


# --- MetricsRingBuffer Tests ---

def test_metrics_ring_buffer_wraparound():
    buffer = MetricsRingBuffer(capacity=3)
    buffer.append(1.0)
    buffer.append(2.0)
    buffer.append(3.0)
    buffer.append(4.0) # Should overwrite 1.0
    
    values = buffer.get_all()
    assert values == [2.0, 3.0, 4.0]


def test_metrics_ring_buffer_stats():
    buffer = MetricsRingBuffer(capacity=10)
    for v in [10, 20, 30]:
        buffer.append(float(v))
    
    assert buffer.mean == 20.0
    assert buffer.min == 10.0
    assert buffer.max == 30.0


# --- EntropyWindow Tests ---

def test_entropy_window_sliding():
    from modelcypher.core.domain.entropy.entropy_window import EntropyWindow
    window = EntropyWindow(size=5)
    
    # Check if a window of entropy samples reports triggers correctly
    for val in [1.0, 1.1, 1.2, 5.0, 1.1]:
        window.add(val)
        
    assert window.has_anomaly is True # Due to 5.0 spike


# --- LogitEntropySample Tests ---

def test_logit_entropy_sample_creation():
    calculator = LogitEntropyCalculator()
    sample = LogitEntropySample.from_computation(
        entropy=2.2,
        variance=1.5,
        token_start=0,
        token_end=1,
        calculator=calculator
    )
    
    assert sample.level == EntropyLevel.MODERATE
    assert sample.logit_entropy == 2.2


def test_logit_entropy_threshold_customization():
    thresholds = EntropyThresholds(low=1.0, high=2.0)
    calculator = LogitEntropyCalculator()
    
    assert calculator.classify(1.5, thresholds=thresholds) == EntropyLevel.MODERATE
    assert calculator.classify(2.5, thresholds=thresholds) == EntropyLevel.HIGH
