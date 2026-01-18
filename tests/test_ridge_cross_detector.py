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

"""Tests for RidgeCrossDetector module."""

from __future__ import annotations

from uuid import uuid4

import pytest

from modelcypher.core.domain.thermo.linguistic_thermodynamics import (
    AttractorBasin,
    BehavioralOutcome,
    LinguisticModifier,
    PerturbedPrompt,
    ThermoMeasurement,
)
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.thermo.ridge_cross_detector import (
    RidgeCrossDetector,
    RidgeCrossEvent,
    RidgeCrossRateStats,
    TransitionAnalysis,
)
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon


def make_prompt(modifier: LinguisticModifier = LinguisticModifier.BASELINE) -> PerturbedPrompt:
    """Create a test PerturbedPrompt."""
    return PerturbedPrompt(
        base_content="Test prompt",
        full_prompt="Test prompt",
        modifier=modifier,
    )


def make_measurement(
    modifier: LinguisticModifier = LinguisticModifier.BASELINE,
    mean_entropy: float = 2.5,
    behavioral_outcome: BehavioralOutcome = BehavioralOutcome.HEDGED,
    delta_h: float | None = None,
) -> ThermoMeasurement:
    """Create a test ThermoMeasurement."""
    return ThermoMeasurement(
        id=uuid4(),
        prompt=make_prompt(modifier),
        first_token_entropy=2.0,
        mean_entropy=mean_entropy,
        entropy_variance=0.1,
        entropy_trajectory=[2.0, 2.5, 2.3],
        top_k_concentration=0.8,
        behavioral_outcome=behavioral_outcome,
        delta_h=delta_h,
    )


class TestRidgeCrossEvent:
    """Tests for RidgeCrossEvent dataclass."""

    def test_is_solution_crossing_true(self):
        """Test is_solution_crossing for refusal -> solution."""
        event = RidgeCrossEvent(
            from_basin=AttractorBasin.REFUSAL,
            to_basin=AttractorBasin.SOLUTION,
            trigger_modifier=LinguisticModifier.DIRECT,
            delta_h=-0.3,
            from_outcome=BehavioralOutcome.REFUSED,
            to_outcome=BehavioralOutcome.SOLVED,
        )
        assert event.is_solution_crossing is True
        assert event.is_refusal_crossing is False

    def test_is_solution_crossing_from_caution(self):
        """Test is_solution_crossing from caution basin."""
        event = RidgeCrossEvent(
            from_basin=AttractorBasin.CAUTION,
            to_basin=AttractorBasin.TRANSITION,
            trigger_modifier=LinguisticModifier.URGENT,
            delta_h=-0.2,
            from_outcome=BehavioralOutcome.HEDGED,
            to_outcome=BehavioralOutcome.ATTEMPTED,
        )
        assert event.is_solution_crossing is True

    def test_is_refusal_crossing_true(self):
        """Test is_refusal_crossing for solution -> refusal."""
        event = RidgeCrossEvent(
            from_basin=AttractorBasin.SOLUTION,
            to_basin=AttractorBasin.REFUSAL,
            trigger_modifier=LinguisticModifier.NEGATION,
            delta_h=0.4,
            from_outcome=BehavioralOutcome.SOLVED,
            to_outcome=BehavioralOutcome.REFUSED,
        )
        assert event.is_refusal_crossing is True
        assert event.is_solution_crossing is False

    def test_neither_crossing(self):
        """Test when neither solution nor refusal crossing."""
        event = RidgeCrossEvent(
            from_basin=AttractorBasin.REFUSAL,
            to_basin=AttractorBasin.CAUTION,
            trigger_modifier=LinguisticModifier.POLITE,
            delta_h=-0.1,
            from_outcome=BehavioralOutcome.REFUSED,
            to_outcome=BehavioralOutcome.HEDGED,
        )
        assert event.is_solution_crossing is False
        assert event.is_refusal_crossing is False

    def test_description_solution_crossing(self):
        """Test description for solution crossing."""
        event = RidgeCrossEvent(
            from_basin=AttractorBasin.REFUSAL,
            to_basin=AttractorBasin.SOLUTION,
            trigger_modifier=LinguisticModifier.DIRECT,
            delta_h=-0.3,
            from_outcome=BehavioralOutcome.REFUSED,
            to_outcome=BehavioralOutcome.SOLVED,
        )
        desc = event.description
        assert "→ solution" in desc
        assert "Direct" in desc
        assert "-0.300" in desc

    def test_description_refusal_crossing(self):
        """Test description for refusal crossing."""
        event = RidgeCrossEvent(
            from_basin=AttractorBasin.SOLUTION,
            to_basin=AttractorBasin.REFUSAL,
            trigger_modifier=LinguisticModifier.NEGATION,
            delta_h=0.4,
            from_outcome=BehavioralOutcome.SOLVED,
            to_outcome=BehavioralOutcome.REFUSED,
        )
        desc = event.description
        assert "→ refusal" in desc


class TestRidgeCrossRateStats:
    """Tests for RidgeCrossRateStats dataclass."""

    def test_confidence_interval_property(self):
        """Test confidence_interval property."""
        stats = RidgeCrossRateStats(
            modifier=LinguisticModifier.DIRECT,
            rate=0.75,
            sample_count=100,
            crossed_count=75,
            confidence_interval_lower=0.65,
            confidence_interval_upper=0.83,
        )
        assert stats.confidence_interval == (0.65, 0.83)

    def test_display_string(self):
        """Test display_string property."""
        stats = RidgeCrossRateStats(
            modifier=LinguisticModifier.DIRECT,
            rate=0.75,
            sample_count=100,
            crossed_count=75,
            confidence_interval_lower=0.65,
            confidence_interval_upper=0.83,
        )
        display = stats.display_string
        assert "75.0%" in display
        assert "65.0%" in display
        assert "83.0%" in display
        assert "n=100" in display


class TestRidgeCrossDetectorDetectCrossings:
    """Tests for RidgeCrossDetector.detect_crossings method."""

    def test_no_crossings_when_no_change(self):
        """Test no crossings detected when no basin change."""
        detector = RidgeCrossDetector()
        baseline = make_measurement(
            modifier=LinguisticModifier.BASELINE,
            mean_entropy=2.5,
            behavioral_outcome=BehavioralOutcome.HEDGED,
        )
        variants = [
            make_measurement(
                modifier=LinguisticModifier.DIRECT,
                mean_entropy=2.4,  # Only 0.1 delta
                behavioral_outcome=BehavioralOutcome.HEDGED,  # Same outcome
            )
        ]

        events = detector.detect_crossings(baseline, variants)

        assert len(events) == 0

    def test_crossing_detected_with_basin_change(self):
        """Test crossing detected with basin change and significant delta."""
        detector = RidgeCrossDetector()
        baseline = make_measurement(
            modifier=LinguisticModifier.BASELINE,
            mean_entropy=2.5,
            behavioral_outcome=BehavioralOutcome.REFUSED,
        )
        variants = [
            make_measurement(
                modifier=LinguisticModifier.DIRECT,
                mean_entropy=2.2,  # 0.3 delta (significant)
                behavioral_outcome=BehavioralOutcome.SOLVED,  # Changed
            )
        ]

        events = detector.detect_crossings(baseline, variants)

        assert len(events) == 1
        assert events[0].trigger_modifier == LinguisticModifier.DIRECT
        assert events[0].is_solution_crossing is True

    def test_skips_baseline_variants(self):
        """Test baseline variants are skipped."""
        detector = RidgeCrossDetector()
        baseline = make_measurement(
            modifier=LinguisticModifier.BASELINE,
            mean_entropy=2.5,
            behavioral_outcome=BehavioralOutcome.REFUSED,
        )
        variants = [
            make_measurement(
                modifier=LinguisticModifier.BASELINE,  # Should be skipped
                mean_entropy=2.2,
                behavioral_outcome=BehavioralOutcome.SOLVED,
            )
        ]

        events = detector.detect_crossings(baseline, variants)

        assert len(events) == 0

    def test_uses_explicit_delta_h(self):
        """Test explicit delta_h is used when provided."""
        detector = RidgeCrossDetector()
        baseline = make_measurement(
            modifier=LinguisticModifier.BASELINE,
            mean_entropy=2.5,
            behavioral_outcome=BehavioralOutcome.REFUSED,
        )
        variants = [
            make_measurement(
                modifier=LinguisticModifier.DIRECT,
                mean_entropy=2.5,  # Same as baseline
                behavioral_outcome=BehavioralOutcome.SOLVED,
                delta_h=-0.5,  # Explicit large delta
            )
        ]

        events = detector.detect_crossings(baseline, variants)

        assert len(events) == 1
        assert events[0].delta_h == -0.5

class TestRidgeCrossDetectorRidgeCrossRate:
    """Tests for RidgeCrossDetector.ridge_cross_rate method."""

    def test_empty_measurements(self):
        """Test empty measurements returns 0.0."""
        detector = RidgeCrossDetector()
        rate = detector.ridge_cross_rate(LinguisticModifier.DIRECT, [])
        assert rate == 0.0

    def test_no_crossings(self):
        """Test rate when no measurements crossed."""
        detector = RidgeCrossDetector()
        measurements = [
            make_measurement(behavioral_outcome=BehavioralOutcome.REFUSED),
            make_measurement(behavioral_outcome=BehavioralOutcome.HEDGED),
        ]

        rate = detector.ridge_cross_rate(LinguisticModifier.DIRECT, measurements)

        assert rate == 0.0

    def test_all_crossed(self):
        """Test rate when all measurements crossed."""
        detector = RidgeCrossDetector()
        measurements = [
            make_measurement(behavioral_outcome=BehavioralOutcome.ATTEMPTED),
            make_measurement(behavioral_outcome=BehavioralOutcome.SOLVED),
        ]

        rate = detector.ridge_cross_rate(LinguisticModifier.DIRECT, measurements)

        assert rate == 1.0

    def test_partial_crossings(self):
        """Test rate with partial crossings."""
        detector = RidgeCrossDetector()
        measurements = [
            make_measurement(behavioral_outcome=BehavioralOutcome.REFUSED),
            make_measurement(behavioral_outcome=BehavioralOutcome.SOLVED),
            make_measurement(behavioral_outcome=BehavioralOutcome.HEDGED),
            make_measurement(behavioral_outcome=BehavioralOutcome.ATTEMPTED),
        ]

        rate = detector.ridge_cross_rate(LinguisticModifier.DIRECT, measurements)

        assert rate == 0.5  # 2 out of 4


class TestRidgeCrossDetectorRidgeCrossRates:
    """Tests for RidgeCrossDetector.ridge_cross_rates method."""

    def test_empty_input(self):
        """Test empty input returns empty dict."""
        detector = RidgeCrossDetector()
        stats = detector.ridge_cross_rates({})
        assert stats == {}

    def test_skips_empty_lists(self):
        """Test empty measurement lists are skipped."""
        detector = RidgeCrossDetector()
        stats = detector.ridge_cross_rates({LinguisticModifier.DIRECT: []})
        assert LinguisticModifier.DIRECT not in stats

    def test_computes_stats_for_each_modifier(self):
        """Test stats computed for each modifier."""
        detector = RidgeCrossDetector()
        measurements = {
            LinguisticModifier.DIRECT: [
                make_measurement(behavioral_outcome=BehavioralOutcome.SOLVED),
                make_measurement(behavioral_outcome=BehavioralOutcome.SOLVED),
            ],
            LinguisticModifier.URGENT: [
                make_measurement(behavioral_outcome=BehavioralOutcome.REFUSED),
                make_measurement(behavioral_outcome=BehavioralOutcome.HEDGED),
            ],
        }

        stats = detector.ridge_cross_rates(measurements)

        assert LinguisticModifier.DIRECT in stats
        assert LinguisticModifier.URGENT in stats
        assert stats[LinguisticModifier.DIRECT].rate == 1.0
        assert stats[LinguisticModifier.URGENT].rate == 0.0

    def test_confidence_interval_bounds(self):
        """Test confidence intervals are properly bounded."""
        detector = RidgeCrossDetector()
        measurements = {
            LinguisticModifier.DIRECT: [
                make_measurement(behavioral_outcome=BehavioralOutcome.SOLVED)
                for _ in range(10)
            ],
        }

        stats = detector.ridge_cross_rates(measurements)

        ci = stats[LinguisticModifier.DIRECT].confidence_interval
        assert 0.0 <= ci[0] <= ci[1] <= 1.0


class TestRidgeCrossDetectorEffectSize:
    """Tests for RidgeCrossDetector.compute_effect_size method."""

    def test_empty_baseline(self):
        """Test empty baseline returns None."""
        detector = RidgeCrossDetector()
        result = detector.compute_effect_size(
            [], [make_measurement(mean_entropy=2.0)]
        )
        assert result is None

    def test_empty_variant(self):
        """Test empty variant returns None."""
        detector = RidgeCrossDetector()
        result = detector.compute_effect_size(
            [make_measurement(mean_entropy=2.0)], []
        )
        assert result is None

    def test_identical_groups(self):
        """Test identical groups have zero effect size."""
        detector = RidgeCrossDetector()
        measurements = [make_measurement(mean_entropy=2.5) for _ in range(5)]

        result = detector.compute_effect_size(measurements, measurements)

        assert result is not None
        eps = division_epsilon(get_default_backend(), get_default_backend().array([1.0]))
        assert abs(result) <= eps

    def test_different_groups(self):
        """Test different groups have non-zero effect size."""
        detector = RidgeCrossDetector()
        baseline = [make_measurement(mean_entropy=2.5) for _ in range(5)]
        variant = [make_measurement(mean_entropy=1.5) for _ in range(5)]

        # compute_effect_size expects (baseline, variant) order
        result = detector.compute_effect_size(baseline, variant)

        assert result is not None
        assert result < 0  # Variant has lower entropy (negative effect)


class TestRidgeCrossDetectorCohensD:
    """Tests for RidgeCrossDetector._cohens_d method."""

    def test_empty_groups(self):
        """Test empty groups return 0."""
        detector = RidgeCrossDetector()
        assert detector._cohens_d([], [1, 2, 3]) == 0.0
        assert detector._cohens_d([1, 2, 3], []) == 0.0

    def test_same_values(self):
        """Test same values return 0."""
        detector = RidgeCrossDetector()
        result = detector._cohens_d([2.0, 2.0, 2.0], [2.0, 2.0, 2.0])
        eps = division_epsilon(get_default_backend(), get_default_backend().array([1.0]))
        assert abs(result) <= eps

    def test_large_effect(self):
        """Test large effect size."""
        detector = RidgeCrossDetector()
        group1 = [10.0, 11.0, 12.0]
        group2 = [1.0, 2.0, 3.0]

        result = detector._cohens_d(group1, group2)

        expected = 9.0
        eps = division_epsilon(get_default_backend(), get_default_backend().array([expected]))
        assert abs(result - expected) <= eps


class TestRidgeCrossDetectorAnalyzeTransitions:
    """Tests for RidgeCrossDetector.analyze_transitions method."""

    def test_no_transitions(self):
        """Test analysis with no transitions."""
        detector = RidgeCrossDetector()
        baseline = make_measurement(
            modifier=LinguisticModifier.BASELINE,
            mean_entropy=2.5,
            behavioral_outcome=BehavioralOutcome.HEDGED,
        )
        variants = []

        analysis = detector.analyze_transitions(baseline, variants)

        assert analysis.events == []
        assert analysis.solution_crossings == 0
        assert analysis.most_effective_modifier is None
        assert analysis.mean_successful_delta_h is None
        assert "Total crossings detected: 0" in analysis.summary

    def test_with_solution_crossings(self):
        """Test analysis with solution crossings."""
        detector = RidgeCrossDetector()
        baseline = make_measurement(
            modifier=LinguisticModifier.BASELINE,
            mean_entropy=2.5,
            behavioral_outcome=BehavioralOutcome.REFUSED,
        )
        variants = [
            make_measurement(
                modifier=LinguisticModifier.DIRECT,
                mean_entropy=2.0,
                behavioral_outcome=BehavioralOutcome.SOLVED,
            ),
            make_measurement(
                modifier=LinguisticModifier.URGENT,
                mean_entropy=2.1,
                behavioral_outcome=BehavioralOutcome.ATTEMPTED,
            ),
        ]

        analysis = detector.analyze_transitions(baseline, variants)

        assert analysis.solution_crossings == 2
        assert analysis.most_effective_modifier is not None

    def test_threshold_delta_h(self):
        """Test threshold_delta_h is minimum successful delta."""
        detector = RidgeCrossDetector()
        baseline = make_measurement(
            modifier=LinguisticModifier.BASELINE,
            mean_entropy=3.0,
            behavioral_outcome=BehavioralOutcome.REFUSED,
        )
        variants = [
            make_measurement(
                modifier=LinguisticModifier.DIRECT,
                mean_entropy=2.5,  # delta = -0.5
                behavioral_outcome=BehavioralOutcome.SOLVED,
            ),
            make_measurement(
                modifier=LinguisticModifier.URGENT,
                mean_entropy=2.7,  # delta = -0.3
                behavioral_outcome=BehavioralOutcome.ATTEMPTED,
            ),
        ]

        analysis = detector.analyze_transitions(baseline, variants)

        # Threshold should be minimum delta that achieved solution crossing
        assert analysis.threshold_delta_h is not None
        assert analysis.threshold_delta_h == min(e.delta_h for e in analysis.events if e.is_solution_crossing)

    def test_summary_includes_key_info(self):
        """Test summary includes key information."""
        detector = RidgeCrossDetector()
        baseline = make_measurement(
            modifier=LinguisticModifier.BASELINE,
            mean_entropy=2.5,
            behavioral_outcome=BehavioralOutcome.REFUSED,
        )
        variants = [
            make_measurement(
                modifier=LinguisticModifier.DIRECT,
                mean_entropy=2.0,
                behavioral_outcome=BehavioralOutcome.SOLVED,
            ),
        ]

        analysis = detector.analyze_transitions(baseline, variants)

        assert "Transition Analysis Summary" in analysis.summary
        assert "Solution crossings:" in analysis.summary


class TestTransitionAnalysisDataclass:
    """Tests for TransitionAnalysis dataclass."""

    def test_fields(self):
        """Test all fields are accessible."""
        analysis = TransitionAnalysis(
            events=[],
            solution_crossings=5,
            most_effective_modifier=LinguisticModifier.DIRECT,
            mean_successful_delta_h=0.3,
            threshold_delta_h=0.15,
            summary="Test summary",
        )

        assert analysis.events == []
        assert analysis.solution_crossings == 5
        assert analysis.most_effective_modifier == LinguisticModifier.DIRECT
        assert analysis.mean_successful_delta_h == 0.3
        assert analysis.threshold_delta_h == 0.15
        assert analysis.summary == "Test summary"
