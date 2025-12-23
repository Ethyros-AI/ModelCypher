"""
Tests for LogitEntropyCalculator.
"""
import math

import pytest
import mlx.core as mx

from modelcypher.core.domain.entropy.logit_entropy_calculator import (
    LogitEntropyCalculator,
    EntropyThresholds,
    EntropyLevel,
    LogitEntropySample,
)


class TestEntropyThresholds:
    """Tests for EntropyThresholds."""
    
    def test_default_values(self):
        """Should have calibrated full-vocab thresholds."""
        t = EntropyThresholds.default()
        
        assert t.low == 1.5
        assert t.high == 3.0
        assert t.circuit_breaker == 4.0
    
    def test_custom_values(self):
        """Should accept custom thresholds."""
        t = EntropyThresholds(low=1.0, high=2.0, circuit_breaker=3.0)
        
        assert t.low == 1.0
        assert t.high == 2.0
        assert t.circuit_breaker == 3.0


class TestLogitEntropyCalculator:
    """Tests for LogitEntropyCalculator."""
    
    def test_initialization(self):
        """Should initialize with default top_k."""
        calc = LogitEntropyCalculator()
        
        assert calc.top_k == 10
        assert calc.epsilon > 0
    
    def test_custom_top_k(self):
        """Should accept custom top_k."""
        calc = LogitEntropyCalculator(top_k=5)
        
        assert calc.top_k == 5
    
    def test_compute_uniform_distribution(self):
        """Uniform logits should have high entropy."""
        calc = LogitEntropyCalculator()
        
        # Uniform distribution (equal logits)
        vocab_size = 100
        logits = mx.zeros((vocab_size,))
        
        entropy, variance = calc.compute(logits)
        
        # Entropy should be ln(vocab_size) ≈ 4.6
        import math
        expected_entropy = math.log(vocab_size)
        assert abs(entropy - expected_entropy) < 0.1
    
    def test_compute_peaked_distribution(self):
        """Peaked logits should have low entropy."""
        calc = LogitEntropyCalculator()
        
        # One very high logit, rest low
        vocab_size = 100
        logits = mx.zeros((vocab_size,))
        logits = logits.at[0].add(100.0)  # One dominant
        
        entropy, variance = calc.compute(logits)
        
        # Entropy should be near 0
        assert entropy < 0.1
    
    def test_flatten_to_vocab_1d(self):
        """1D input should pass through."""
        calc = LogitEntropyCalculator()
        logits = mx.array([1.0, 2.0, 3.0])
        
        result = calc._flatten_to_vocab(logits)
        
        assert result.shape == (3,)
    
    def test_flatten_to_vocab_2d(self):
        """2D input [batch, vocab] should extract batch 0."""
        calc = LogitEntropyCalculator()
        logits = mx.zeros((2, 100))  # batch=2, vocab=100
        
        result = calc._flatten_to_vocab(logits)
        
        assert result.shape == (100,)
    
    def test_flatten_to_vocab_3d(self):
        """3D input [batch, seq, vocab] should extract last token."""
        calc = LogitEntropyCalculator()
        logits = mx.zeros((2, 5, 100))  # batch=2, seq=5, vocab=100
        
        result = calc._flatten_to_vocab(logits)
        
        assert result.shape == (100,)
    
    def test_compute_with_skip_variance(self):
        """Should return 0 variance when skipped."""
        calc = LogitEntropyCalculator()
        logits = mx.zeros((100,))
        
        entropy, variance = calc.compute(logits, skip_variance=True)
        
        assert variance == 0.0
        assert entropy > 0  # Entropy still computed
    
    def test_compute_batch(self):
        """Should compute entropy for batch of logits."""
        calc = LogitEntropyCalculator()
        
        batch = [
            mx.zeros((100,)),  # Uniform
            mx.zeros((100,)).at[0].add(100.0),  # Peaked
        ]
        
        results = calc.compute_batch(batch)
        
        assert len(results) == 2
        # First should have high entropy, second low
        assert results[0][0] > results[1][0]
    
    def test_compute_batch_empty(self):
        """Should handle empty batch."""
        calc = LogitEntropyCalculator()
        
        results = calc.compute_batch([])
        
        assert results == []


class TestEntropyLevel:
    """Tests for EntropyLevel enum."""
    
    def test_values(self):
        """Should have expected values."""
        assert EntropyLevel.LOW.value == "low"
        assert EntropyLevel.MODERATE.value == "moderate"
        assert EntropyLevel.HIGH.value == "high"


class TestClassification:
    """Tests for entropy level classification."""
    
    def test_classify_low(self):
        """Entropy below low threshold should be LOW."""
        calc = LogitEntropyCalculator()
        
        level = calc.classify(1.0)
        
        assert level == EntropyLevel.LOW
    
    def test_classify_moderate(self):
        """Entropy between thresholds should be MODERATE."""
        calc = LogitEntropyCalculator()
        
        level = calc.classify(2.0)
        
        assert level == EntropyLevel.MODERATE
    
    def test_classify_high(self):
        """Entropy above high threshold should be HIGH."""
        calc = LogitEntropyCalculator()
        
        level = calc.classify(5.0)
        
        assert level == EntropyLevel.HIGH
    
    def test_should_trip_circuit_breaker_false(self):
        """Entropy below threshold should not trip."""
        calc = LogitEntropyCalculator()
        
        assert not calc.should_trip_circuit_breaker(3.0)
    
    def test_should_trip_circuit_breaker_true(self):
        """Entropy above threshold should trip."""
        calc = LogitEntropyCalculator()
        
        assert calc.should_trip_circuit_breaker(5.0)


class TestLogitEntropySample:
    """Tests for LogitEntropySample."""
    
    def test_from_computation(self):
        """Should create sample from computed values."""
        calc = LogitEntropyCalculator()
        
        sample = LogitEntropySample.from_computation(
            entropy=2.5,
            variance=0.5,
            token_start=0,
            token_end=10,
            calculator=calc,
            latency_ms=5.0,
            source="test",
        )
        
        assert sample.logit_entropy == 2.5
        assert sample.top_k_variance == 0.5
        assert sample.level == EntropyLevel.MODERATE
        assert sample.latency_ms == 5.0
        assert sample.source == "test"
        assert sample.window_id is not None


class TestEdgeCases:
    """Edge case tests for numerical stability.

    These tests verify the calculator handles degenerate inputs without crashing.
    The goal is robustness, not correctness for invalid inputs.
    """

    def test_compute_with_inf_logits_does_not_crash(self):
        """Compute should complete without raising on inf input.

        When inf appears in logits, softmax produces 0/1 probabilities
        which may result in nan entropy. The key property tested is
        that the function doesn't raise an exception.
        """
        calc = LogitEntropyCalculator()

        logits = mx.array([float('inf'), 1.0, 2.0, 3.0])

        # Should complete without raising
        entropy, variance = calc.compute(logits)

        # Returns a float (may be nan/inf but should return)
        assert isinstance(entropy, float)
        assert isinstance(variance, float)

    def test_compute_with_neg_inf_logits_produces_finite_result(self):
        """Compute with -inf should produce valid entropy.

        -inf logits become 0 probability after softmax, which is well-defined.
        The remaining tokens should have valid entropy.
        """
        calc = LogitEntropyCalculator()

        # -inf token has 0 probability, others share the mass
        logits = mx.array([float('-inf'), 1.0, 1.0, 1.0])

        entropy, variance = calc.compute(logits)

        # With -inf token excluded, it's a 3-way uniform -> entropy = ln(3)
        import math
        assert mx.isfinite(mx.array(entropy))
        assert abs(entropy - math.log(3)) < 0.1

    def test_compute_with_nan_logits_propagates_nan(self):
        """Compute with nan input should propagate nan (IEEE 754 semantics)."""
        calc = LogitEntropyCalculator()

        logits = mx.array([float('nan'), 1.0, 2.0, 3.0])

        entropy, variance = calc.compute(logits)

        # NaN should propagate through the computation
        assert math.isnan(entropy)

    def test_log_zero_protection(self):
        """Should protect against log(0) with epsilon."""
        calc = LogitEntropyCalculator()

        # Very peaked distribution that could cause log(0) issues
        logits = mx.zeros((1000,))
        logits = logits.at[0].add(1000.0)  # Extremely peaked

        # Should not raise
        entropy, variance = calc.compute(logits)

        # Entropy should be near 0 but finite
        assert mx.isfinite(mx.array(entropy))
        assert entropy >= 0

    def test_softmax_numerical_stability_large_values(self):
        """Should handle very large logit values (numerical stability of softmax)."""
        calc = LogitEntropyCalculator()

        # Very large values that could cause overflow in naive softmax
        logits = mx.array([1000.0, 1001.0, 1002.0])

        # Should not overflow
        entropy, variance = calc.compute(logits)

        assert mx.isfinite(mx.array(entropy))

    def test_compute_with_all_identical_logits(self):
        """Should handle all identical logits (uniform distribution)."""
        calc = LogitEntropyCalculator()

        # All same value = uniform distribution
        logits = mx.full((100,), 5.0)

        entropy, variance = calc.compute(logits)

        import math
        expected = math.log(100)
        assert abs(entropy - expected) < 0.1

    def test_compute_with_single_element(self):
        """Should handle single element logit array."""
        calc = LogitEntropyCalculator()

        logits = mx.array([1.0])

        entropy, variance = calc.compute(logits)

        # Single element = zero entropy (no uncertainty)
        assert entropy == pytest.approx(0.0, abs=1e-6)

    def test_compute_with_two_elements(self):
        """Should handle two element logit array."""
        calc = LogitEntropyCalculator()

        # Equal logits = maximum entropy for 2 choices
        logits = mx.array([1.0, 1.0])

        entropy, variance = calc.compute(logits)

        import math
        expected = math.log(2)
        assert abs(entropy - expected) < 0.1
