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

"""Tests for PhaseTransitionTheory.

Tests the statistical mechanics framework for language model phase transitions.
Validates the Softmax-Boltzmann equivalence and critical temperature derivation.

Key invariants tested:
- T_c ≈ 1.0 for typical LLM parameters (σ_z ≈ 4.0, V_eff ≈ 2000)
- dH/dT = Var(z) / T³ (entropy derivative formula)
- Phase classification respects T < T_c (ordered) vs T > T_c (disordered)
- Entropy increases monotonically with temperature
"""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain.thermo.phase_transition_theory import (
    BasinTopology,
    Phase,
    PhaseTransitionTheory,
)


class TestCriticalTemperatureEstimation:
    """Tests for T_c estimation from first principles."""

    def test_theoretical_tc_approximately_one(self) -> None:
        """T_c should be ≈ 1.0 for typical LLM parameters.

        From the derivation:
        T_c = σ_z / √(2 × ln(V_eff))

        With σ_z = 4.0 and V_eff = 2000:
        T_c = 4.0 / √(2 × 7.6) = 4.0 / 3.9 ≈ 1.03
        """
        tc = PhaseTransitionTheory.theoretical_tc()
        assert 0.9 <= tc <= 1.2, f"Theoretical T_c={tc} should be ≈ 1.0"

    def test_tc_formula_with_typical_values(self) -> None:
        """T_c formula should give expected result for typical values."""
        # σ_z = 4.0 (typical logit std dev)
        # V_eff = 2000 (typical effective vocab)
        tc = PhaseTransitionTheory.estimate_critical_temperature(
            logit_std_dev=4.0,
            effective_vocab_size=2000,
        )

        # T_c = 4.0 / √(2 × ln(2000)) = 4.0 / √(2 × 7.6) ≈ 1.03
        expected = 4.0 / math.sqrt(2 * math.log(2000))
        assert abs(tc - expected) < 0.01, f"T_c={tc}, expected={expected}"

    def test_tc_increases_with_logit_std_dev(self) -> None:
        """Higher logit variance should increase T_c."""
        tc_low = PhaseTransitionTheory.estimate_critical_temperature(2.0, 2000)
        tc_high = PhaseTransitionTheory.estimate_critical_temperature(6.0, 2000)

        assert tc_high > tc_low

    def test_tc_decreases_with_vocab_size(self) -> None:
        """Larger vocabulary should decrease T_c."""
        tc_small = PhaseTransitionTheory.estimate_critical_temperature(4.0, 500)
        tc_large = PhaseTransitionTheory.estimate_critical_temperature(4.0, 10000)

        assert tc_small > tc_large

    def test_tc_handles_edge_cases(self) -> None:
        """T_c should handle degenerate cases gracefully."""
        # Single effective token
        tc = PhaseTransitionTheory.estimate_critical_temperature(4.0, 1)
        assert tc == 1.0  # Fallback value

        # Zero vocab (invalid)
        tc = PhaseTransitionTheory.estimate_critical_temperature(4.0, 0)
        assert tc == 1.0


class TestEntropyDerivative:
    """Tests for dH/dT = Var(z) / T³ formula."""

    def test_entropy_derivative_formula(self) -> None:
        """dH/dT should follow Var(z) / T³ relationship."""
        logits = [1.0, 2.0, 3.0, 4.0, 5.0]
        temp = 1.0

        deriv = PhaseTransitionTheory.entropy_derivative(logits, temp)
        variance = PhaseTransitionTheory.compute_logit_variance(logits, temp)

        # dH/dT = Var(z) / T³
        expected = variance / (temp**3)
        assert abs(deriv - expected) < 1e-6

    def test_derivative_positive_for_positive_variance(self) -> None:
        """Entropy derivative should be positive when variance > 0."""
        logits = [1.0, 2.0, 3.0]  # Non-uniform
        temp = 1.0

        deriv = PhaseTransitionTheory.entropy_derivative(logits, temp)
        assert deriv > 0

    def test_derivative_decreases_with_temperature(self) -> None:
        """Higher temperature should decrease the derivative (T³ in denominator)."""
        logits = [1.0, 2.0, 3.0, 4.0]

        deriv_low = PhaseTransitionTheory.entropy_derivative(logits, 0.5)
        deriv_high = PhaseTransitionTheory.entropy_derivative(logits, 2.0)

        assert deriv_low > deriv_high


class TestPhaseClassification:
    """Tests for phase classification based on temperature."""

    def test_ordered_phase_below_tc(self) -> None:
        """T < T_c should be classified as ORDERED."""
        phase = PhaseTransitionTheory.classify_phase(
            temperature=0.5,
            critical_temperature=1.0,
        )
        assert phase == Phase.ORDERED

    def test_disordered_phase_above_tc(self) -> None:
        """T > T_c should be classified as DISORDERED."""
        phase = PhaseTransitionTheory.classify_phase(
            temperature=1.5,
            critical_temperature=1.0,
        )
        assert phase == Phase.DISORDERED

    def test_critical_phase_near_tc(self) -> None:
        """T ≈ T_c should be classified as CRITICAL."""
        phase = PhaseTransitionTheory.classify_phase(
            temperature=1.0,
            critical_temperature=1.0,
            tolerance=0.15,
        )
        assert phase == Phase.CRITICAL

    def test_phase_display_names(self) -> None:
        """Phases should have descriptive display names."""
        assert "Ordered" in Phase.ORDERED.display_name
        assert "Critical" in Phase.CRITICAL.display_name
        assert "Disordered" in Phase.DISORDERED.display_name

    def test_phase_modifier_effects(self) -> None:
        """Phases should have expected modifier effect descriptions."""
        assert "reduction" in Phase.ORDERED.expected_modifier_effect.lower()
        assert "unpredictable" in Phase.CRITICAL.expected_modifier_effect.lower()
        assert "increase" in Phase.DISORDERED.expected_modifier_effect.lower()


class TestEntropyComputation:
    """Tests for Shannon entropy computation."""

    def test_entropy_non_negative(self) -> None:
        """Entropy should always be non-negative."""
        logits = [1.0, 2.0, 3.0, 4.0]
        for temp in [0.5, 1.0, 2.0]:
            entropy = PhaseTransitionTheory.compute_entropy(logits, temp)
            assert entropy >= 0

    def test_entropy_zero_for_deterministic(self) -> None:
        """Entropy should be ≈ 0 when one logit dominates (low T)."""
        logits = [10.0, 0.0, 0.0]  # One very high logit
        entropy = PhaseTransitionTheory.compute_entropy(logits, 0.1)
        # At very low T, probability concentrates on max logit
        assert entropy < 0.1

    def test_entropy_increases_with_temperature(self) -> None:
        """Higher temperature should generally increase entropy."""
        logits = [1.0, 2.0, 3.0, 4.0]

        entropy_low = PhaseTransitionTheory.compute_entropy(logits, 0.5)
        entropy_high = PhaseTransitionTheory.compute_entropy(logits, 2.0)

        assert entropy_high > entropy_low

    def test_entropy_max_for_uniform(self) -> None:
        """Uniform logits should give maximum entropy (log(n))."""
        n = 4
        logits = [1.0] * n  # Uniform

        entropy = PhaseTransitionTheory.compute_entropy(logits, 1.0)
        max_entropy = math.log(n)

        # Should be close to max entropy
        assert abs(entropy - max_entropy) < 0.01


class TestLogitStatistics:
    """Tests for logit statistics computation."""

    def test_compute_logit_statistics(self) -> None:
        """Should compute correct mean, variance, std_dev."""
        logits = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = PhaseTransitionTheory.compute_logit_statistics(logits)

        assert stats.mean == 3.0
        # Sample variance = Σ(x - mean)² / (n-1)
        expected_var = 10.0 / 4  # (4+1+0+1+4) / 4
        assert abs(stats.variance - expected_var) < 1e-6
        assert abs(stats.std_dev - math.sqrt(expected_var)) < 1e-6

    def test_empty_logits(self) -> None:
        """Empty logits should return zero statistics."""
        stats = PhaseTransitionTheory.compute_logit_statistics([])
        assert stats.mean == 0.0
        assert stats.variance == 0.0
        assert stats.std_dev == 0.0


class TestEffectiveVocabSize:
    """Tests for effective vocabulary size computation."""

    def test_effective_vocab_decreases_with_lower_temperature(self) -> None:
        """Lower temperature should concentrate probability on fewer tokens."""
        logits = [i * 0.1 for i in range(100)]

        v_eff_low = PhaseTransitionTheory.effective_vocabulary_size(logits, 0.1)
        v_eff_high = PhaseTransitionTheory.effective_vocabulary_size(logits, 2.0)

        assert v_eff_high > v_eff_low

    def test_effective_vocab_at_least_one(self) -> None:
        """Should always return at least 1."""
        v_eff = PhaseTransitionTheory.effective_vocabulary_size([10.0, 0.0, 0.0], 0.01)
        assert v_eff >= 1


class TestBasinTopology:
    """Tests for behavioral basin topology."""

    def test_escape_probability_increases_with_temperature(self) -> None:
        """Higher temperature should increase escape probability."""
        topology = BasinTopology.default()

        p_low = topology.escape_probability(0.5)
        p_high = topology.escape_probability(2.0)

        assert p_high > p_low

    def test_escape_probability_bounded(self) -> None:
        """Escape probability should be in [0, 1]."""
        topology = BasinTopology.default()

        for temp in [0.1, 0.5, 1.0, 2.0, 5.0]:
            p = topology.escape_probability(temp)
            assert 0 <= p <= 1

    def test_basin_weights_sum_to_one(self) -> None:
        """Basin weights should sum to 1 (probability distribution)."""
        topology = BasinTopology.default()

        for temp in [0.5, 1.0, 2.0]:
            weights = topology.basin_weights(temp)
            total = weights.refusal + weights.caution + weights.solution
            assert abs(total - 1.0) < 1e-6

    def test_refusal_dominates_at_zero_temperature(self) -> None:
        """At T=0, all probability should go to deepest basin (refusal)."""
        topology = BasinTopology.default()
        weights = topology.basin_weights(0.0)

        assert weights.refusal == 1.0
        assert weights.caution == 0.0
        assert weights.solution == 0.0


class TestModifierEffectPrediction:
    """Tests for intensity modifier effect prediction."""

    def test_ordered_phase_predicts_cooling(self) -> None:
        """Ordered phase should predict entropy reduction (cooling)."""
        prediction = PhaseTransitionTheory.predict_modifier_effect(
            phase=Phase.ORDERED,
            intensity_score=0.8,
            base_entropy=2.0,
        )

        assert prediction.predicted_delta_h < 0  # Cooling = negative delta
        assert prediction.confidence > 0.8

    def test_disordered_phase_predicts_heating(self) -> None:
        """Disordered phase should predict entropy increase (heating)."""
        prediction = PhaseTransitionTheory.predict_modifier_effect(
            phase=Phase.DISORDERED,
            intensity_score=0.8,
            base_entropy=2.0,
        )

        assert prediction.predicted_delta_h > 0  # Heating = positive delta
        assert prediction.confidence > 0.5

    def test_critical_phase_has_low_confidence(self) -> None:
        """Critical phase should have low prediction confidence."""
        prediction = PhaseTransitionTheory.predict_modifier_effect(
            phase=Phase.CRITICAL,
            intensity_score=0.8,
            base_entropy=2.0,
        )

        assert prediction.confidence < 0.5


class TestTemperatureSweep:
    """Tests for temperature sweep analysis."""

    def test_sweep_returns_expected_structure(self) -> None:
        """Temperature sweep should return all expected fields."""
        logits = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = PhaseTransitionTheory.temperature_sweep(logits)

        assert len(result.temperatures) > 0
        assert len(result.entropies) == len(result.temperatures)
        assert len(result.derivatives) == len(result.temperatures)
        assert result.estimated_tc > 0

    def test_entropy_monotonic_in_sweep(self) -> None:
        """Entropy should generally increase with temperature."""
        logits = [1.0, 2.0, 3.0, 4.0]
        result = PhaseTransitionTheory.temperature_sweep(logits)

        # Check most consecutive pairs are increasing
        increasing_count = sum(
            1
            for i in range(len(result.entropies) - 1)
            if result.entropies[i + 1] >= result.entropies[i]
        )
        assert increasing_count >= len(result.entropies) - 2


class TestPhaseAnalysis:
    """Tests for complete phase analysis."""

    def test_analyze_returns_complete_result(self) -> None:
        """Analyze should return all expected fields."""
        logits = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = PhaseTransitionTheory.analyze(
            logits=logits,
            temperature=1.0,
            intensity_score=0.5,
        )

        assert result.temperature == 1.0
        assert result.estimated_tc > 0
        assert result.phase in Phase
        assert result.logit_variance >= 0
        assert result.effective_vocab_size >= 1
        assert result.confidence >= 0
        assert result.basin_weights is not None

    def test_analyze_with_custom_topology(self) -> None:
        """Analyze should accept custom basin topology."""
        logits = [1.0, 2.0, 3.0]
        custom_topology = BasinTopology(
            refusal_depth=0.1,
            caution_depth=0.3,
            transition_ridge=0.9,
            solution_depth=0.5,
        )

        result = PhaseTransitionTheory.analyze(
            logits=logits,
            temperature=1.0,
            topology=custom_topology,
        )

        assert result.basin_weights is not None


class TestValidation:
    """Tests for T_c validation."""

    def test_valid_estimation(self) -> None:
        """Estimation within tolerance should be valid."""
        assert PhaseTransitionTheory.validate_tc_estimation(
            estimated_tc=1.0,
            observed_tc=1.1,
            tolerance=0.2,
        )

    def test_invalid_estimation(self) -> None:
        """Estimation outside tolerance should be invalid."""
        assert not PhaseTransitionTheory.validate_tc_estimation(
            estimated_tc=1.0,
            observed_tc=1.5,
            tolerance=0.2,
        )


class TestMathematicalInvariants:
    """Property-based tests for mathematical invariants."""

    @given(
        logits=st.lists(
            st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
            min_size=3,
            max_size=20,
        ),
        temperature=st.floats(min_value=0.1, max_value=5.0),
    )
    @settings(max_examples=30)
    def test_entropy_always_non_negative(self, logits: list[float], temperature: float) -> None:
        """Entropy should always be non-negative (within floating point tolerance)."""
        entropy = PhaseTransitionTheory.compute_entropy(logits, temperature)
        # Allow tiny negative values due to floating point precision
        assert entropy >= -1e-9

    @given(
        logit_std=st.floats(min_value=0.1, max_value=10.0),
        vocab_size=st.integers(min_value=2, max_value=100000),
    )
    @settings(max_examples=30)
    def test_tc_always_positive(self, logit_std: float, vocab_size: int) -> None:
        """Critical temperature should always be positive."""
        tc = PhaseTransitionTheory.estimate_critical_temperature(logit_std, vocab_size)
        assert tc > 0

    @given(
        temperature=st.floats(min_value=0.01, max_value=10.0),
    )
    @settings(max_examples=20)
    def test_basin_weights_are_valid_distribution(self, temperature: float) -> None:
        """Basin weights should form a valid probability distribution."""
        topology = BasinTopology.default()
        weights = topology.basin_weights(temperature)

        total = weights.refusal + weights.caution + weights.solution
        assert abs(total - 1.0) < 1e-6
        assert weights.refusal >= 0
        assert weights.caution >= 0
        assert weights.solution >= 0
