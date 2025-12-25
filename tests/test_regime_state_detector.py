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

"""Tests for RegimeStateDetector (requires MLX).

Tests the thermodynamic regime classification that determines
ordered, critical, or disordered states based on logit statistics.
"""

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

from modelcypher.core.domain.dynamics.regime_state_detector import (
    BasinTopology,
    RegimeAnalysis,
    RegimeState,
    RegimeStateDetector,
)


class TestRegimeState:
    """Tests for RegimeState enum."""

    def test_display_names(self):
        """Each state should have a display name."""
        assert "Ordered" in RegimeState.ORDERED.display_name
        assert "Critical" in RegimeState.CRITICAL.display_name
        assert "Disordered" in RegimeState.DISORDERED.display_name

    def test_expected_modifier_effects(self):
        """Each state should have expected modifier effect description."""
        assert "cooling" in RegimeState.ORDERED.expected_modifier_effect.lower()
        assert "unpredictable" in RegimeState.CRITICAL.expected_modifier_effect.lower()
        assert "heating" in RegimeState.DISORDERED.expected_modifier_effect.lower()


class TestBasinTopology:
    """Tests for BasinTopology dataclass."""

    def test_from_logit_geometry(self):
        """Topology should be derived from entropy landscape."""
        # Low entropy regime: refusal basin should be deep
        topology_low = BasinTopology.from_logit_geometry(
            entropy=0.5,
            max_entropy=3.0,
            temperature=0.5,
            critical_temperature=1.0,
        )
        # Basin depths should be finite and reasonable
        assert 0.0 <= topology_low.refusal_depth <= 1.0
        assert 0.0 <= topology_low.solution_depth <= 1.0

        # High entropy regime: basins shallower
        topology_high = BasinTopology.from_logit_geometry(
            entropy=2.5,
            max_entropy=3.0,
            temperature=1.5,
            critical_temperature=1.0,
        )
        assert 0.0 <= topology_high.refusal_depth <= 1.0

    def test_basin_weights_low_temperature(self):
        """Low temperature should favor deepest basin."""
        # Create topology where refusal is deepest
        topology = BasinTopology(
            refusal_depth=0.0,  # Deepest
            caution_depth=0.3,
            solution_depth=0.5,
            transition_ridge=0.8,
        )
        refusal, caution, solution = topology.basin_weights(0.1)

        # At very low T, deepest basin should dominate
        assert refusal > caution
        assert refusal > solution

    def test_basin_weights_high_temperature(self):
        """High temperature should give more uniform weights."""
        topology = BasinTopology(
            refusal_depth=0.0,
            caution_depth=0.3,
            solution_depth=0.5,
            transition_ridge=0.8,
        )
        refusal, caution, solution = topology.basin_weights(10.0)

        # At high T, weights become more uniform
        assert abs(refusal - caution) < 0.3
        assert abs(caution - solution) < 0.3

    def test_basin_weights_zero_temperature(self):
        """Zero temperature should put all weight in deepest basin."""
        topology = BasinTopology(
            refusal_depth=0.0,  # Deepest
            caution_depth=0.3,
            solution_depth=0.5,
            transition_ridge=0.8,
        )
        refusal, caution, solution = topology.basin_weights(0.0)

        # All weight in deepest basin
        assert refusal == 1.0
        assert caution == 0.0
        assert solution == 0.0

    def test_basin_weights_sum_to_one(self):
        """Basin weights should sum to 1."""
        topology = BasinTopology(
            refusal_depth=0.1,
            caution_depth=0.3,
            solution_depth=0.5,
            transition_ridge=0.8,
        )

        for temp in [0.1, 0.5, 1.0, 2.0, 10.0]:
            refusal, caution, solution = topology.basin_weights(temp)
            total = refusal + caution + solution
            assert abs(total - 1.0) < 1e-6


class TestClassifyState:
    """Tests for RegimeStateDetector.classify_state().

    The tolerance for classifying CRITICAL is derived from logit_variance:
    tolerance = sqrt(variance) / T_c, clamped to [0.05, 0.5].
    """

    def test_ordered_below_tc(self):
        """Temperature well below T_c should be ordered."""
        # Low variance = tight tolerance (~0.1 for variance=0.01, tc=1.0)
        state = RegimeStateDetector.classify_state(
            temperature=0.5, critical_temperature=1.0, logit_variance=0.01
        )
        assert state == RegimeState.ORDERED

    def test_disordered_above_tc(self):
        """Temperature well above T_c should be disordered."""
        state = RegimeStateDetector.classify_state(
            temperature=2.0, critical_temperature=1.0, logit_variance=0.01
        )
        assert state == RegimeState.DISORDERED

    def test_critical_near_tc(self):
        """Temperature near T_c should be critical."""
        state = RegimeStateDetector.classify_state(
            temperature=1.0, critical_temperature=1.0, logit_variance=0.01
        )
        assert state == RegimeState.CRITICAL

    def test_critical_region_scales_with_variance(self):
        """Higher variance = wider critical region."""
        # With low variance (0.01), tolerance is ~0.1, so 1.15 is disordered
        state_low_var = RegimeStateDetector.classify_state(
            temperature=1.15, critical_temperature=1.0, logit_variance=0.01
        )
        # With high variance (0.25), tolerance is ~0.5, so 1.15 is still critical
        state_high_var = RegimeStateDetector.classify_state(
            temperature=1.15, critical_temperature=1.0, logit_variance=0.25
        )

        assert state_low_var == RegimeState.DISORDERED
        assert state_high_var == RegimeState.CRITICAL

    def test_zero_tc_returns_ordered(self):
        """Zero critical temperature should return ordered."""
        state = RegimeStateDetector.classify_state(
            temperature=1.0, critical_temperature=0.0, logit_variance=0.1
        )
        assert state == RegimeState.ORDERED


class TestLogitStatistics:
    """Tests for logit statistics computation."""

    def test_compute_logit_variance_uniform(self):
        """Uniform logits should have low variance."""
        logits = mx.array([1.0, 1.0, 1.0, 1.0])
        variance = RegimeStateDetector().compute_logit_variance(logits, temperature=1.0)

        assert variance >= 0.0
        assert variance < 0.1  # Near zero for uniform

    def test_compute_logit_variance_peaked(self):
        """Peaked logits should have higher variance."""
        logits = mx.array([10.0, 0.0, 0.0, 0.0])
        variance = RegimeStateDetector().compute_logit_variance(logits, temperature=1.0)

        assert variance > 0.0

    def test_compute_logit_variance_zero_temp(self):
        """Zero temperature should return zero variance."""
        logits = mx.array([1.0, 2.0, 3.0])
        variance = RegimeStateDetector().compute_logit_variance(logits, temperature=0.0)

        assert variance == 0.0

    def test_compute_logit_statistics(self):
        """Should return mean, variance, std_dev."""
        logits = mx.array([0.0, 2.0, 4.0])
        mean, variance, std_dev = RegimeStateDetector().compute_logit_statistics(logits)

        assert abs(mean - 2.0) < 0.01
        # Variance of [0, 2, 4] is (4+0+4)/3 = 2.67
        assert variance > 0
        assert abs(std_dev - math.sqrt(variance)) < 0.01


class TestCriticalTemperature:
    """Tests for critical temperature estimation."""

    def test_estimate_critical_temperature(self):
        """T_c should depend on std_dev and vocab size."""
        # T_c = σ / sqrt(2 * ln(V_eff))
        std_dev = 1.0
        vocab_size = 100  # ln(100) ≈ 4.6

        tc = RegimeStateDetector.estimate_critical_temperature(std_dev, vocab_size)

        expected = 1.0 / math.sqrt(2.0 * math.log(100))
        assert abs(tc - expected) < 0.01

    def test_estimate_critical_temperature_small_vocab(self):
        """Small vocab should give T_c = 1.0."""
        tc = RegimeStateDetector.estimate_critical_temperature(1.0, 1)
        assert tc == 1.0

    def test_estimate_critical_temperature_zero_vocab(self):
        """Zero vocab should give T_c = 1.0."""
        tc = RegimeStateDetector.estimate_critical_temperature(1.0, 0)
        assert tc == 1.0


class TestEffectiveVocabularySize:
    """Tests for effective vocabulary size computation."""

    def test_effective_vocab_uniform(self):
        """Uniform distribution should have high effective vocab."""
        logits = mx.zeros((100,))  # Uniform over 100 tokens
        v_eff = RegimeStateDetector().effective_vocabulary_size(logits, temperature=1.0)

        # Most tokens should be above threshold
        assert v_eff > 50

    def test_effective_vocab_peaked(self):
        """Peaked distribution should have low effective vocab."""
        logits = mx.zeros((100,))
        logits[0] = 100.0  # Very peaked
        v_eff = RegimeStateDetector().effective_vocabulary_size(logits, temperature=1.0)

        # Should be very concentrated
        assert v_eff < 10

    def test_effective_vocab_zero_temp(self):
        """Zero temperature should return 1."""
        logits = mx.array([1.0, 2.0, 3.0])
        v_eff = RegimeStateDetector().effective_vocabulary_size(logits, temperature=0.0)

        assert v_eff == 1


class TestEntropy:
    """Tests for entropy computation."""

    def test_compute_entropy_uniform(self):
        """Uniform distribution should have high entropy."""
        logits = mx.zeros((10,))  # Uniform
        entropy = RegimeStateDetector().compute_entropy(logits, temperature=1.0)

        # Max entropy for 10 tokens is ln(10) ≈ 2.3
        assert entropy > 2.0

    def test_compute_entropy_peaked(self):
        """Peaked distribution should have low entropy."""
        logits = mx.zeros((10,))
        logits[0] = 100.0  # Very peaked
        entropy = RegimeStateDetector().compute_entropy(logits, temperature=1.0)

        assert entropy < 0.5

    def test_compute_entropy_zero_temp(self):
        """Zero temperature should return zero entropy."""
        logits = mx.array([1.0, 2.0, 3.0])
        entropy = RegimeStateDetector().compute_entropy(logits, temperature=0.0)

        assert entropy == 0.0


class TestPredictModifierEffect:
    """Tests for modifier effect prediction.

    All effect magnitudes and confidences are derived from the geometry:
    - Distance from T_c (how stable the regime is)
    - Logit variance (susceptibility to perturbation)
    - Base entropy (room for change)
    """

    def test_ordered_state_predicts_cooling(self):
        """Ordered state should predict negative (cooling) effect."""
        delta_h, confidence = RegimeStateDetector.predict_modifier_effect(
            state=RegimeState.ORDERED,
            intensity_score=0.5,
            base_entropy=1.0,
            temperature=0.5,  # Well below T_c
            critical_temperature=1.0,
            logit_variance=0.1,  # Low variance = high confidence
        )

        assert delta_h < 0  # Cooling (entropy reduction)
        assert confidence > 0  # Positive confidence

    def test_disordered_state_predicts_heating(self):
        """Disordered state should predict positive (heating) effect."""
        delta_h, confidence = RegimeStateDetector.predict_modifier_effect(
            state=RegimeState.DISORDERED,
            intensity_score=0.5,
            base_entropy=1.0,
            temperature=2.0,  # Well above T_c
            critical_temperature=1.0,
            logit_variance=0.1,
        )

        assert delta_h > 0  # Heating (entropy increase)
        assert confidence > 0

    def test_critical_state_is_uncertain(self):
        """Critical state should have low confidence."""
        delta_h, confidence = RegimeStateDetector.predict_modifier_effect(
            state=RegimeState.CRITICAL,
            intensity_score=0.5,
            base_entropy=1.0,
            temperature=1.0,  # At T_c
            critical_temperature=1.0,
            logit_variance=0.1,
        )

        assert delta_h == 0.0  # Unpredictable
        # Confidence is low when near critical point
        assert confidence < 0.5

    def test_effect_scales_with_distance_from_tc(self):
        """Effect magnitude should be larger further from T_c."""
        # Far from T_c
        delta_far, _ = RegimeStateDetector.predict_modifier_effect(
            state=RegimeState.ORDERED,
            intensity_score=0.5,
            base_entropy=1.0,
            temperature=0.3,  # Far below T_c
            critical_temperature=1.0,
            logit_variance=0.1,
        )

        # Closer to T_c
        delta_near, _ = RegimeStateDetector.predict_modifier_effect(
            state=RegimeState.ORDERED,
            intensity_score=0.5,
            base_entropy=1.0,
            temperature=0.8,  # Closer to T_c
            critical_temperature=1.0,
            logit_variance=0.1,
        )

        # Larger distance = larger effect magnitude
        assert abs(delta_far) > abs(delta_near)


class TestAnalyze:
    """Tests for full analyze() method."""

    def test_analyze_uniform_logits(self):
        """Analyze should work with uniform logits."""
        logits = mx.zeros((100,))
        result = RegimeStateDetector().analyze(logits, temperature=1.0)

        assert isinstance(result, RegimeAnalysis)
        assert result.temperature == 1.0
        # Uniform logits have zero std_dev, so T_c = 0
        assert result.estimated_tc >= 0
        assert result.state in [RegimeState.ORDERED, RegimeState.CRITICAL, RegimeState.DISORDERED]
        assert result.effective_vocab_size > 0
        assert result.basin_weights is not None
        assert len(result.basin_weights) == 3

    def test_analyze_peaked_logits(self):
        """Analyze should handle peaked logits with concentrated probability."""
        logits = mx.zeros((100,))
        logits[0] = 10.0
        result = RegimeStateDetector().analyze(logits, temperature=1.0)

        # Peaked logits should have low effective vocab (probability concentrated)
        assert result.effective_vocab_size < 50  # Much less than 100 tokens
        assert result.effective_vocab_size >= 1  # At least 1 token
        # Peaked logits should have non-zero variance (higher than perfectly uniform)
        assert result.logit_variance > 0.0

    def test_analyze_derives_topology_from_geometry(self):
        """Analyze should derive topology from the logit geometry."""
        logits = mx.zeros((50,))
        result = RegimeStateDetector().analyze(logits, temperature=1.0)

        # Topology is derived, not passed - basin weights should still exist
        assert result.basin_weights is not None
        # Weights should sum to 1
        assert abs(sum(result.basin_weights) - 1.0) < 1e-6

    def test_analyze_with_intensity_score(self):
        """Intensity score should affect modifier prediction."""
        logits = mx.zeros((50,))

        result_low = RegimeStateDetector().analyze(logits, temperature=0.5, intensity_score=0.1)
        result_high = RegimeStateDetector().analyze(logits, temperature=0.5, intensity_score=0.9)

        # Higher intensity should have larger predicted effect
        assert abs(result_high.predicted_modifier_effect) >= abs(
            result_low.predicted_modifier_effect
        )


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_temperature(self):
        """Should handle very small temperatures."""
        logits = mx.array([1.0, 2.0, 3.0])
        result = RegimeStateDetector().analyze(logits, temperature=1e-10)

        assert result.state == RegimeState.ORDERED

    def test_very_large_temperature(self):
        """Should handle very large temperatures."""
        logits = mx.array([1.0, 2.0, 3.0])
        result = RegimeStateDetector().analyze(logits, temperature=1000.0)

        assert result.state == RegimeState.DISORDERED

    def test_single_token_logits(self):
        """Should handle single token logits."""
        logits = mx.array([5.0])
        result = RegimeStateDetector().analyze(logits, temperature=1.0)

        assert result.effective_vocab_size == 1

    def test_large_logit_values(self):
        """Should handle large logit values without overflow."""
        logits = mx.array([100.0, 0.0, -100.0])
        result = RegimeStateDetector().analyze(logits, temperature=1.0)

        assert math.isfinite(result.logit_variance)
        assert math.isfinite(result.estimated_tc)

    def test_2d_logits_batch(self):
        """Should handle batched 2D logits."""
        logits = mx.random.normal((4, 100))  # Batch of 4
        variance = RegimeStateDetector().compute_logit_variance(logits, temperature=1.0)

        assert variance >= 0
        assert math.isfinite(variance)
