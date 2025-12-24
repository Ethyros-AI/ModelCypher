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


# =============================================================================
# Mathematical Invariant Tests
# =============================================================================


class TestEntropyBoundsInvariants:
    """Tests for Shannon entropy bound invariants."""

    @pytest.mark.parametrize("seed", range(5))
    def test_entropy_non_negative(self, seed: int) -> None:
        """Entropy must be >= 0.

        Mathematical property: Shannon entropy H = -∑p*log(p) ≥ 0.
        """
        import numpy as np
        rng = np.random.default_rng(seed)
        calc = LogitEntropyCalculator()

        # Random logits
        logits = mx.array(rng.standard_normal(100).astype("float32"))
        entropy, _ = calc.compute(logits)

        assert entropy >= 0.0

    @pytest.mark.parametrize("vocab_size", [10, 100, 1000])
    def test_entropy_bounded_by_log_vocab(self, vocab_size: int) -> None:
        """Entropy must be <= ln(V).

        Mathematical property: Maximum entropy is ln(V) for uniform distribution.
        """
        calc = LogitEntropyCalculator()

        # Any logit distribution
        logits = mx.zeros((vocab_size,))
        entropy, _ = calc.compute(logits)

        max_entropy = math.log(vocab_size)
        assert entropy <= max_entropy + 1e-6

    @pytest.mark.parametrize("seed", range(5))
    def test_variance_non_negative(self, seed: int) -> None:
        """Variance must be >= 0.

        Mathematical property: Variance is a squared quantity.
        """
        import numpy as np
        rng = np.random.default_rng(seed)
        calc = LogitEntropyCalculator()

        logits = mx.array(rng.standard_normal(100).astype("float32"))
        _, variance = calc.compute(logits)

        assert variance >= 0.0

    @pytest.mark.parametrize("seed", range(5))
    def test_normalized_entropy_in_zero_one(self, seed: int) -> None:
        """Normalized entropy must be in [0, 1].

        Mathematical property: Normalization clamps to [0, 1].
        """
        import numpy as np
        rng = np.random.default_rng(seed)
        calc = LogitEntropyCalculator()

        logits = mx.array(rng.standard_normal(100).astype("float32"))
        _, _, normalized = calc.compute_with_normalization(logits)

        assert 0.0 <= normalized <= 1.0


class TestEntropyMonotonicity:
    """Tests for entropy monotonicity properties."""

    def test_more_uniform_higher_entropy(self) -> None:
        """More uniform distributions should have higher entropy.

        Mathematical property: Entropy is maximized at uniform distribution.
        """
        calc = LogitEntropyCalculator()

        # Peaked distribution
        peaked = mx.zeros((100,))
        peaked = peaked.at[0].add(50.0)

        # More uniform
        uniform = mx.zeros((100,))

        h_peaked, _ = calc.compute(peaked)
        h_uniform, _ = calc.compute(uniform)

        assert h_uniform > h_peaked

    def test_single_dominant_near_zero(self) -> None:
        """Single dominant token should have entropy near 0.

        Mathematical property: Certainty means zero entropy.
        """
        calc = LogitEntropyCalculator()

        logits = mx.zeros((1000,))
        logits = logits.at[0].add(1000.0)  # Overwhelmingly dominant

        entropy, _ = calc.compute(logits)

        assert entropy < 0.01


class TestEntropyNormalization:
    """Tests for entropy normalization to [0, 1] range."""

    def test_normalize_zero_entropy(self):
        """Zero entropy should normalize to 0."""
        normalized = LogitEntropyCalculator.normalize_entropy(0.0, vocab_size=32000)
        assert normalized == 0.0

    def test_normalize_max_entropy(self):
        """Maximum entropy (ln(vocab_size)) should normalize to 1."""
        vocab_size = 32000
        max_entropy = math.log(vocab_size)
        normalized = LogitEntropyCalculator.normalize_entropy(max_entropy, vocab_size)
        assert normalized == pytest.approx(1.0, abs=1e-6)

    def test_normalize_typical_values(self):
        """Typical entropy values should normalize correctly."""
        vocab_size = 32000
        max_entropy = math.log(vocab_size)  # ~10.37

        # Low entropy (confident) -> low normalized
        low_raw = 1.5
        low_norm = LogitEntropyCalculator.normalize_entropy(low_raw, vocab_size)
        assert 0.1 < low_norm < 0.2  # ~0.14

        # High entropy (uncertain) -> higher normalized
        high_raw = 5.0
        high_norm = LogitEntropyCalculator.normalize_entropy(high_raw, vocab_size)
        assert 0.4 < high_norm < 0.6  # ~0.48

    def test_normalize_clamps_negative(self):
        """Negative entropy should clamp to 0."""
        normalized = LogitEntropyCalculator.normalize_entropy(-1.0, vocab_size=32000)
        assert normalized == 0.0

    def test_normalize_clamps_above_max(self):
        """Entropy above max should clamp to 1."""
        vocab_size = 1000
        max_entropy = math.log(vocab_size)  # ~6.9
        normalized = LogitEntropyCalculator.normalize_entropy(max_entropy + 1.0, vocab_size)
        assert normalized == 1.0

    def test_normalize_small_vocab(self):
        """Should handle small vocabulary correctly."""
        vocab_size = 10
        max_entropy = math.log(vocab_size)  # ~2.3

        # Half of max entropy
        normalized = LogitEntropyCalculator.normalize_entropy(max_entropy / 2, vocab_size)
        assert normalized == pytest.approx(0.5, abs=0.01)

    def test_normalize_vocab_size_1(self):
        """Vocab size 1 should return 0 (no uncertainty possible)."""
        normalized = LogitEntropyCalculator.normalize_entropy(1.0, vocab_size=1)
        assert normalized == 0.0

    def test_normalize_vocab_size_0(self):
        """Vocab size 0 should return 0 (edge case)."""
        normalized = LogitEntropyCalculator.normalize_entropy(1.0, vocab_size=0)
        assert normalized == 0.0

    def test_compute_with_normalization(self):
        """compute_with_normalization should return raw, variance, and normalized."""
        calc = LogitEntropyCalculator()

        # Uniform distribution over 100 tokens
        logits = mx.zeros((100,))
        raw, variance, normalized = calc.compute_with_normalization(logits)

        # Raw should be ln(100) ≈ 4.6
        assert raw == pytest.approx(math.log(100), abs=0.1)

        # Normalized should be raw/ln(vocab) = ln(100)/ln(100) = 1.0
        assert normalized == pytest.approx(1.0, abs=0.01)

    def test_compute_with_normalization_peaked(self):
        """Peaked distribution should have low normalized entropy."""
        calc = LogitEntropyCalculator()

        logits = mx.zeros((100,))
        logits = logits.at[0].add(100.0)  # Very peaked

        raw, variance, normalized = calc.compute_with_normalization(logits)

        # Raw entropy near 0
        assert raw < 0.1

        # Normalized also near 0
        assert normalized < 0.1

    def test_compute_with_normalization_explicit_vocab_size(self):
        """Should use explicit vocab_size if provided."""
        calc = LogitEntropyCalculator()

        # 100-token uniform distribution
        logits = mx.zeros((100,))
        raw, _, normalized = calc.compute_with_normalization(logits, vocab_size=32000)

        # Raw entropy is ln(100) ≈ 4.6
        assert raw == pytest.approx(math.log(100), abs=0.1)

        # But normalized against 32K vocab: ln(100)/ln(32000) ≈ 0.44
        assert 0.4 < normalized < 0.5
