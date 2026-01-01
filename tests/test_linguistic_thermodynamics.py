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

"""Tests for LinguisticThermodynamics.

Tests the linguistic thermodynamics types that model prompt engineering
as entropy reduction (cooling) rather than injection.
"""

from uuid import UUID

import pytest

from modelcypher.core.domain.thermo.linguistic_thermodynamics import (
    AttractorBasin,
    BehavioralOutcome,
    EntropyDirection,
    LanguageResourceLevel,
    LinguisticModifier,
    LocalizedModifiers,
    ModifierMechanism,
    MultilingualMeasurement,
    MultilingualPerturbedPrompt,
    PerturbedPrompt,
    PromptLanguage,
    ThermoMeasurement,
)


class TestLinguisticModifier:
    """Tests for LinguisticModifier enum."""

    def test_all_have_display_names(self):
        """All modifiers should have display names."""
        for modifier in LinguisticModifier:
            assert modifier.display_name is not None
            assert len(modifier.display_name) > 0

    def test_baseline_is_neutral_direction(self):
        """Baseline should have neutral expected direction."""
        assert LinguisticModifier.BASELINE.expected_direction == EntropyDirection.NEUTRAL

    def test_non_baseline_decreases_entropy(self):
        """Non-baseline modifiers should expect entropy decrease."""
        for modifier in LinguisticModifier:
            if modifier != LinguisticModifier.BASELINE:
                assert modifier.expected_direction == EntropyDirection.DECREASE

    def test_all_have_mechanisms(self):
        """All modifiers should have a mechanism classification."""
        for modifier in LinguisticModifier:
            assert modifier.mechanism is not None
            assert isinstance(modifier.mechanism, ModifierMechanism)


class TestBehavioralOutcome:
    """Tests for BehavioralOutcome enum."""

    def test_all_have_display_names(self):
        """All outcomes should have display names."""
        for outcome in BehavioralOutcome:
            assert outcome.display_name is not None
            assert len(outcome.display_name) > 0

    def test_ridge_crossing_classification(self):
        """Ridge crossing should be ATTEMPTED or SOLVED."""
        assert not BehavioralOutcome.REFUSED.is_ridge_crossed
        assert not BehavioralOutcome.HEDGED.is_ridge_crossed
        assert BehavioralOutcome.ATTEMPTED.is_ridge_crossed
        assert BehavioralOutcome.SOLVED.is_ridge_crossed

    def test_basin_mapping(self):
        """Each outcome should map to a valid basin."""
        expected_basins = {
            BehavioralOutcome.REFUSED: AttractorBasin.REFUSAL,
            BehavioralOutcome.HEDGED: AttractorBasin.CAUTION,
            BehavioralOutcome.ATTEMPTED: AttractorBasin.TRANSITION,
            BehavioralOutcome.SOLVED: AttractorBasin.SOLUTION,
        }
        for outcome, expected_basin in expected_basins.items():
            assert outcome.basin == expected_basin

    def test_all_have_display_colors(self):
        """All outcomes should have display colors."""
        for outcome in BehavioralOutcome:
            assert outcome.display_color is not None
            assert len(outcome.display_color) > 0


class TestPromptLanguage:
    """Tests for PromptLanguage enum."""

    def test_all_have_display_names(self):
        """All languages should have display names."""
        for lang in PromptLanguage:
            assert lang.display_name is not None
            assert len(lang.display_name) > 0

    def test_iso_codes_are_valid(self):
        """ISO codes should be 2-letter codes."""
        for lang in PromptLanguage:
            assert len(lang.iso_code) == 2
            assert lang.iso_code.islower()

    def test_resource_levels_assigned(self):
        """All languages should have resource level classification."""
        for lang in PromptLanguage:
            assert lang.resource_level in LanguageResourceLevel

    def test_english_is_high_resource(self):
        """English should be high resource."""
        assert PromptLanguage.ENGLISH.resource_level == LanguageResourceLevel.HIGH

    def test_swahili_is_low_resource(self):
        """Swahili should be low resource."""
        assert PromptLanguage.SWAHILI.resource_level == LanguageResourceLevel.LOW


class TestPerturbedPrompt:
    """Tests for PerturbedPrompt dataclass."""

    def test_create_baseline(self):
        """Baseline should not modify content."""
        content = "Tell me about cats"
        prompt = PerturbedPrompt.create(content, LinguisticModifier.BASELINE)
        assert prompt.full_prompt == content
        assert prompt.base_content == content
        assert prompt.modifier == LinguisticModifier.BASELINE

    def test_create_caps(self):
        """CAPS should uppercase content."""
        content = "Tell me about cats"
        prompt = PerturbedPrompt.create(content, LinguisticModifier.CAPS)
        assert prompt.full_prompt == "TELL ME ABOUT CATS"

    def test_create_polite(self):
        """Polite should add prefix."""
        content = "Tell me about cats"
        prompt = PerturbedPrompt.create(content, LinguisticModifier.POLITE)
        assert "Could you please" in prompt.full_prompt
        assert content in prompt.full_prompt

    def test_create_negation(self):
        """Negation should add suffix."""
        content = "Tell me about cats"
        prompt = PerturbedPrompt.create(content, LinguisticModifier.NEGATION)
        assert "Don't hedge" in prompt.full_prompt
        assert content in prompt.full_prompt

    def test_create_combined(self):
        """Combined should apply multiple transformations."""
        content = "Tell me about cats"
        prompt = PerturbedPrompt.create(content, LinguisticModifier.COMBINED)
        # Should be uppercase
        assert "TELL ME ABOUT CATS" in prompt.full_prompt
        # Should have urgency prefix
        assert "NOW" in prompt.full_prompt
        # Should have negation suffix
        assert "Don't hedge" in prompt.full_prompt


class TestThermoMeasurement:
    """Tests for ThermoMeasurement dataclass."""

    @pytest.fixture
    def sample_prompt(self):
        return PerturbedPrompt.create("Test content", LinguisticModifier.BASELINE)

    def test_create_generates_uuid(self, sample_prompt):
        """Create should generate a valid UUID."""
        measurement = ThermoMeasurement.create(
            prompt=sample_prompt,
            first_token_entropy=1.5,
            mean_entropy=1.2,
            entropy_variance=0.1,
            entropy_trajectory=[1.5, 1.2, 1.0],
            top_k_concentration=0.3,
            model_state="normal",
            behavioral_outcome=BehavioralOutcome.SOLVED,
            generated_text="Test output",
            token_count=10,
            stop_reason="max_tokens",
        )
        assert isinstance(measurement.id, UUID)

    def test_modifier_accessor(self, sample_prompt):
        """Modifier accessor should return prompt's modifier."""
        measurement = ThermoMeasurement.create(
            prompt=sample_prompt,
            first_token_entropy=1.5,
            mean_entropy=1.2,
            entropy_variance=0.1,
            entropy_trajectory=[1.5, 1.2, 1.0],
            top_k_concentration=0.3,
            model_state="normal",
            behavioral_outcome=BehavioralOutcome.SOLVED,
            generated_text="Test output",
            token_count=10,
            stop_reason="max_tokens",
        )
        assert measurement.modifier == LinguisticModifier.BASELINE

    def test_ridge_crossed(self, sample_prompt):
        """Ridge crossed should match behavioral outcome."""
        measurement = ThermoMeasurement.create(
            prompt=sample_prompt,
            first_token_entropy=1.5,
            mean_entropy=1.2,
            entropy_variance=0.1,
            entropy_trajectory=[1.5, 1.2, 1.0],
            top_k_concentration=0.3,
            model_state="normal",
            behavioral_outcome=BehavioralOutcome.SOLVED,
            generated_text="Test output",
            token_count=10,
            stop_reason="max_tokens",
        )
        assert measurement.ridge_crossed is True

    def test_entropy_trend_decreasing(self, sample_prompt):
        """Decreasing trajectory should report DECREASE."""
        measurement = ThermoMeasurement.create(
            prompt=sample_prompt,
            first_token_entropy=2.0,
            mean_entropy=1.5,
            entropy_variance=0.1,
            entropy_trajectory=[2.0, 1.8, 1.6, 1.4, 1.2, 1.0],  # Decreasing
            top_k_concentration=0.3,
            model_state="normal",
            behavioral_outcome=BehavioralOutcome.SOLVED,
            generated_text="Test output",
            token_count=10,
            stop_reason="max_tokens",
        )
        assert measurement.entropy_trend == EntropyDirection.DECREASE

    def test_entropy_trend_increasing(self, sample_prompt):
        """Increasing trajectory should report INCREASE."""
        measurement = ThermoMeasurement.create(
            prompt=sample_prompt,
            first_token_entropy=1.0,
            mean_entropy=1.5,
            entropy_variance=0.1,
            entropy_trajectory=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0],  # Increasing
            top_k_concentration=0.3,
            model_state="normal",
            behavioral_outcome=BehavioralOutcome.SOLVED,
            generated_text="Test output",
            token_count=10,
            stop_reason="max_tokens",
        )
        assert measurement.entropy_trend == EntropyDirection.INCREASE

    def test_entropy_trend_short_trajectory(self, sample_prompt):
        """Short trajectory should report NEUTRAL."""
        measurement = ThermoMeasurement.create(
            prompt=sample_prompt,
            first_token_entropy=1.5,
            mean_entropy=1.5,
            entropy_variance=0.0,
            entropy_trajectory=[1.5, 1.5],  # Too short
            top_k_concentration=0.3,
            model_state="normal",
            behavioral_outcome=BehavioralOutcome.SOLVED,
            generated_text="Test output",
            token_count=10,
            stop_reason="max_tokens",
        )
        assert measurement.entropy_trend == EntropyDirection.NEUTRAL

    def test_distress_signature_detection(self, sample_prompt):
        """High entropy + low variance should show distress."""
        measurement = ThermoMeasurement.create(
            prompt=sample_prompt,
            first_token_entropy=3.5,
            mean_entropy=3.5,
            entropy_variance=0.01,
            entropy_trajectory=[3.5, 3.5, 3.5],
            top_k_concentration=0.1,  # Low concentration
            model_state="distressed",
            behavioral_outcome=BehavioralOutcome.HEDGED,
            generated_text="I'm not sure...",
            token_count=10,
            stop_reason="max_tokens",
        )
        assert measurement.shows_distress_signature is True


class TestLocalizedModifiers:
    """Tests for LocalizedModifiers class."""

    def test_caps_works_universally(self):
        """CAPS should work the same in all languages."""
        content = "Hello World"
        for lang in PromptLanguage:
            result = LocalizedModifiers.apply(LinguisticModifier.CAPS, content, lang)
            assert result == "HELLO WORLD"

    def test_english_templates_exist(self):
        """All modifiers should have English templates."""
        for modifier in LinguisticModifier:
            prefix, suffix = LocalizedModifiers.template(modifier, PromptLanguage.ENGLISH)
            # At least one should exist or modifier is baseline/direct/caps
            if modifier in (
                LinguisticModifier.BASELINE,
                LinguisticModifier.DIRECT,
                LinguisticModifier.CAPS,
            ):
                assert prefix is None and suffix is None
            else:
                assert prefix is not None or suffix is not None

    def test_polite_prefix_in_different_languages(self):
        """Polite should have different prefixes per language."""
        content = "Help me"
        english = LocalizedModifiers.apply(
            LinguisticModifier.POLITE, content, PromptLanguage.ENGLISH
        )
        chinese = LocalizedModifiers.apply(
            LinguisticModifier.POLITE, content, PromptLanguage.CHINESE
        )

        assert "please" in english.lower()
        assert "请" in chinese  # "Please" in Chinese

    def test_combined_applies_caps_and_prefix_suffix(self):
        """Combined should uppercase and add prefix/suffix."""
        content = "Help me"
        result = LocalizedModifiers.apply(
            LinguisticModifier.COMBINED, content, PromptLanguage.ENGLISH
        )
        assert "HELP ME" in result
        assert "expert" in result.lower()


class TestMultilingualPerturbedPrompt:
    """Tests for MultilingualPerturbedPrompt dataclass."""

    def test_create_english(self):
        """Should create English perturbed prompt."""
        content = "Tell me about cats"
        prompt = MultilingualPerturbedPrompt.create(
            base_content=content,
            modifier=LinguisticModifier.POLITE,
            language=PromptLanguage.ENGLISH,
        )
        assert prompt.language == PromptLanguage.ENGLISH
        assert prompt.modifier == LinguisticModifier.POLITE
        assert "please" in prompt.full_prompt.lower()

    def test_create_chinese(self):
        """Should create Chinese perturbed prompt."""
        content = "告诉我关于猫的事"
        prompt = MultilingualPerturbedPrompt.create(
            base_content=content,
            modifier=LinguisticModifier.POLITE,
            language=PromptLanguage.CHINESE,
        )
        assert prompt.language == PromptLanguage.CHINESE
        assert "请" in prompt.full_prompt


class TestMultilingualMeasurement:
    """Tests for MultilingualMeasurement dataclass."""

    @pytest.fixture
    def sample_prompt(self):
        return MultilingualPerturbedPrompt.create(
            base_content="Test content",
            modifier=LinguisticModifier.CAPS,
            language=PromptLanguage.ENGLISH,
        )

    def test_create_generates_uuid(self, sample_prompt):
        """Create should generate a valid UUID."""
        measurement = MultilingualMeasurement.create(
            prompt=sample_prompt,
            baseline_entropy=2.0,
            modified_entropy=1.5,
            token_count=10,
        )
        assert isinstance(measurement.id, UUID)

    def test_delta_h_computation(self, sample_prompt):
        """Delta H should be modified - baseline."""
        measurement = MultilingualMeasurement.create(
            prompt=sample_prompt,
            baseline_entropy=2.0,
            modified_entropy=1.5,
            token_count=10,
        )
        assert measurement.delta_h == pytest.approx(-0.5)

    def test_shows_cooling_negative_delta(self, sample_prompt):
        """Negative delta should show cooling."""
        measurement = MultilingualMeasurement.create(
            prompt=sample_prompt,
            baseline_entropy=2.0,
            modified_entropy=1.5,  # Lower = cooling
            token_count=10,
        )
        assert measurement.shows_cooling is True

    def test_no_cooling_positive_delta(self, sample_prompt):
        """Positive delta should not show cooling."""
        measurement = MultilingualMeasurement.create(
            prompt=sample_prompt,
            baseline_entropy=1.5,
            modified_entropy=2.0,  # Higher = heating
            token_count=10,
        )
        assert measurement.shows_cooling is False

    def test_accessors(self, sample_prompt):
        """Accessors should return correct values."""
        measurement = MultilingualMeasurement.create(
            prompt=sample_prompt,
            baseline_entropy=2.0,
            modified_entropy=1.5,
            token_count=10,
        )
        assert measurement.language == PromptLanguage.ENGLISH
        assert measurement.modifier == LinguisticModifier.CAPS
