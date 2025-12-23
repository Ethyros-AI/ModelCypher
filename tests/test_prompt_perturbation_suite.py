"""
Tests for PromptPerturbationSuite.

This tests the linguistic modifier variant generation for entropy experiments.
"""
from __future__ import annotations

import pytest

from modelcypher.core.domain.dynamics.prompt_perturbation_suite import (
    LinguisticModifier,
    ModifierMechanism,
    ModifierTemplate,
    PerturbationConfig,
    PerturbedPrompt,
    PromptPerturbationSuite,
    TextTransform,
)


# =============================================================================
# TextTransform Tests
# =============================================================================


class TestTextTransform:
    """Tests for TextTransform."""

    def test_uppercase(self) -> None:
        """Test uppercase transform."""
        result = TextTransform.uppercase.apply("Hello World")
        assert result == "HELLO WORLD"

    def test_lowercase(self) -> None:
        """Test lowercase transform."""
        result = TextTransform.lowercase.apply("Hello World")
        assert result == "hello world"

    def test_title_case(self) -> None:
        """Test title case transform."""
        result = TextTransform.title_case.apply("hello world")
        assert result == "Hello World"


# =============================================================================
# ModifierTemplate Tests
# =============================================================================


class TestModifierTemplate:
    """Tests for ModifierTemplate."""

    def test_empty_template(self) -> None:
        """Test empty template leaves content unchanged."""
        template = ModifierTemplate()
        result = template.apply("Hello")
        assert result == "Hello"

    def test_prefix_only(self) -> None:
        """Test template with prefix only."""
        template = ModifierTemplate(prefix="URGENT: ")
        result = template.apply("Please help")
        assert result == "URGENT: Please help"

    def test_suffix_only(self) -> None:
        """Test template with suffix only."""
        template = ModifierTemplate(suffix=" NOW!")
        result = template.apply("Do this")
        assert result == "Do this NOW!"

    def test_prefix_and_suffix(self) -> None:
        """Test template with both prefix and suffix."""
        template = ModifierTemplate(prefix="[START] ", suffix=" [END]")
        result = template.apply("content")
        assert result == "[START] content [END]"

    def test_transform_only(self) -> None:
        """Test template with transform only."""
        template = ModifierTemplate(transform=TextTransform.uppercase)
        result = template.apply("hello")
        assert result == "HELLO"

    def test_full_template(self) -> None:
        """Test template with all options."""
        template = ModifierTemplate(
            prefix=">> ",
            suffix=" <<",
            transform=TextTransform.uppercase,
        )
        result = template.apply("hello")
        # Transform applied first, then prefix/suffix
        assert result == ">> HELLO <<"


# =============================================================================
# LinguisticModifier Tests
# =============================================================================


class TestLinguisticModifier:
    """Tests for LinguisticModifier."""

    def test_intensity_scores(self) -> None:
        """Test that intensity scores are ordered correctly."""
        scores = [m.intensity_score for m in LinguisticModifier]
        assert scores[0] == 0.0  # baseline
        assert LinguisticModifier.baseline.intensity_score < LinguisticModifier.caps.intensity_score
        assert LinguisticModifier.caps.intensity_score < LinguisticModifier.combined.intensity_score
        assert LinguisticModifier.combined.intensity_score == 1.0

    def test_mechanism_mapping(self) -> None:
        """Test mechanism mapping for modifiers."""
        assert LinguisticModifier.baseline.mechanism == ModifierMechanism.none
        assert LinguisticModifier.caps.mechanism == ModifierMechanism.typography
        assert LinguisticModifier.polite.mechanism == ModifierMechanism.framing
        assert LinguisticModifier.roleplay.mechanism == ModifierMechanism.persona
        assert LinguisticModifier.combined.mechanism == ModifierMechanism.compound


# =============================================================================
# PerturbedPrompt Tests
# =============================================================================


class TestPerturbedPrompt:
    """Tests for PerturbedPrompt."""

    def test_apply_modifier_baseline(self) -> None:
        """Test applying baseline modifier."""
        result = PerturbedPrompt.apply_modifier(
            LinguisticModifier.baseline,
            "Hello world"
        )
        assert result == "Hello world"

    def test_apply_modifier_caps(self) -> None:
        """Test applying caps modifier."""
        result = PerturbedPrompt.apply_modifier(
            LinguisticModifier.caps,
            "Hello world"
        )
        assert result == "HELLO WORLD"

    def test_apply_modifier_polite(self) -> None:
        """Test applying polite modifier."""
        result = PerturbedPrompt.apply_modifier(
            LinguisticModifier.polite,
            "Help me"
        )
        assert "Could you please" in result
        assert "Help me" in result


# =============================================================================
# PerturbationConfig Tests
# =============================================================================


class TestPerturbationConfig:
    """Tests for PerturbationConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = PerturbationConfig.default()
        assert config.always_include_baseline is True
        assert len(config.default_modifiers) == len(LinguisticModifier)

    def test_minimal_config(self) -> None:
        """Test minimal configuration."""
        config = PerturbationConfig.minimal()
        assert len(config.default_modifiers) == 4
        assert LinguisticModifier.baseline in config.default_modifiers
        assert LinguisticModifier.caps in config.default_modifiers


# =============================================================================
# PromptPerturbationSuite Tests
# =============================================================================


class TestPromptPerturbationSuite:
    """Tests for PromptPerturbationSuite."""

    def test_generate_variants_default(self) -> None:
        """Test generating variants with default config."""
        suite = PromptPerturbationSuite()
        variants = suite.generate_variants("Test prompt")

        assert len(variants) == len(LinguisticModifier)
        assert all(isinstance(v, PerturbedPrompt) for v in variants)
        assert all(v.base_content == "Test prompt" for v in variants)

    def test_generate_variants_specific_modifiers(self) -> None:
        """Test generating variants with specific modifiers."""
        suite = PromptPerturbationSuite()
        modifiers = [LinguisticModifier.caps, LinguisticModifier.polite]
        variants = suite.generate_variants("Test", modifiers=modifiers)

        # Should include baseline + specified
        assert len(variants) == 3  # baseline + 2 specified
        variant_modifiers = [v.modifier for v in variants]
        assert LinguisticModifier.baseline in variant_modifiers

    def test_generate_variants_no_baseline(self) -> None:
        """Test generating variants without always including baseline."""
        config = PerturbationConfig(
            default_modifiers=[LinguisticModifier.caps],
            always_include_baseline=False,
        )
        suite = PromptPerturbationSuite(config=config)
        variants = suite.generate_variants("Test")

        assert len(variants) == 1
        assert variants[0].modifier == LinguisticModifier.caps

    def test_generate_single_variant(self) -> None:
        """Test generating a single variant."""
        suite = PromptPerturbationSuite()
        variant = suite.generate_variant("Hello", LinguisticModifier.caps)

        assert variant.base_content == "Hello"
        assert variant.modifier == LinguisticModifier.caps
        assert variant.full_prompt == "HELLO"

    def test_generate_variants_by_mechanism(self) -> None:
        """Test grouping variants by mechanism."""
        suite = PromptPerturbationSuite()
        grouped = suite.generate_variants_by_mechanism("Test")

        assert ModifierMechanism.none in grouped
        assert ModifierMechanism.typography in grouped
        assert ModifierMechanism.framing in grouped

    def test_generate_intensity_gradient(self) -> None:
        """Test generating intensity gradient."""
        suite = PromptPerturbationSuite()
        gradient = suite.generate_intensity_gradient("Test")

        # Check that it's sorted by intensity
        intensities = [v.modifier.intensity_score for v in gradient]
        assert intensities == sorted(intensities)
        assert gradient[0].modifier == LinguisticModifier.baseline
        assert gradient[-1].modifier == LinguisticModifier.combined

    def test_research_suite(self) -> None:
        """Test research-grade suite creation."""
        suite = PromptPerturbationSuite.research()
        variants = suite.generate_variants("Test prompt")

        # Research templates should be different from defaults
        polite_variant = next(v for v in variants if v.modifier == LinguisticModifier.polite)
        assert "greatly appreciate" in polite_variant.full_prompt

    def test_custom_templates(self) -> None:
        """Test using custom templates."""
        custom = {
            LinguisticModifier.baseline: ModifierTemplate(prefix="[CUSTOM] "),
        }
        config = PerturbationConfig(
            default_modifiers=[LinguisticModifier.baseline],
            custom_templates=custom,
        )
        suite = PromptPerturbationSuite(config=config)

        variants = suite.generate_variants("Test")
        assert variants[0].full_prompt == "[CUSTOM] Test"


# =============================================================================
# Batch Generation Tests
# =============================================================================


class TestBatchGeneration:
    """Tests for batch generation methods."""

    def test_generate_batch_variants(self) -> None:
        """Test batch variant generation."""
        suite = PromptPerturbationSuite(
            config=PerturbationConfig.minimal()
        )
        prompts = ["Prompt A", "Prompt B", "Prompt C"]

        batch = suite.generate_batch_variants(prompts)

        assert len(batch) == 3
        assert "Prompt A" in batch
        assert "Prompt B" in batch
        assert len(batch["Prompt A"]) == 4  # minimal has 4 modifiers

    def test_generate_cross_product(self) -> None:
        """Test cross-product generation."""
        suite = PromptPerturbationSuite(
            config=PerturbationConfig(
                default_modifiers=[LinguisticModifier.baseline, LinguisticModifier.caps],
                always_include_baseline=False,
            )
        )
        prompts = ["A", "B"]

        cross = suite.generate_cross_product(prompts)

        # 2 prompts × 2 modifiers = 4 variants
        assert len(cross) == 4


# =============================================================================
# Analysis Helper Tests
# =============================================================================


class TestAnalysisHelpers:
    """Tests for analysis helper methods."""

    def test_estimate_token_overhead(self) -> None:
        """Test token overhead estimation."""
        suite = PromptPerturbationSuite(
            config=PerturbationConfig.minimal()
        )

        avg, max_overhead = suite.estimate_token_overhead(100)

        assert avg >= 0
        assert max_overhead >= avg
        # Combined template is longest, so max should be > 0
        assert max_overhead > 0

    def test_default_templates_exist(self) -> None:
        """Test that default templates exist for all modifiers."""
        templates = PromptPerturbationSuite.default_templates()

        for modifier in LinguisticModifier:
            assert modifier in templates

    def test_research_templates_exist(self) -> None:
        """Test that research templates exist for all modifiers."""
        templates = PromptPerturbationSuite.research_templates()

        for modifier in LinguisticModifier:
            assert modifier in templates


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the perturbation suite."""

    def test_full_workflow(self) -> None:
        """Test a complete workflow."""
        # Create suite
        suite = PromptPerturbationSuite()

        # Generate variants
        base_prompt = "How do I solve this math problem?"
        variants = suite.generate_variants(base_prompt)

        # Verify all modifiers are represented
        modifier_set = {v.modifier for v in variants}
        assert modifier_set == set(LinguisticModifier)

        # Verify base content is preserved
        assert all(v.base_content == base_prompt for v in variants)

        # Verify some expected transformations
        caps_variant = next(v for v in variants if v.modifier == LinguisticModifier.caps)
        assert caps_variant.full_prompt.isupper()

        baseline_variant = next(v for v in variants if v.modifier == LinguisticModifier.baseline)
        assert baseline_variant.full_prompt == base_prompt

    def test_gradient_entropy_experiment_setup(self) -> None:
        """Test setting up a gradient entropy experiment."""
        suite = PromptPerturbationSuite()

        # Setup for entropy comparison experiment
        prompts = [
            "Explain quantum physics",
            "How do I bake a cake?",
            "What is the meaning of life?",
        ]

        all_variants = suite.generate_cross_product(prompts)

        # Should have prompt_count × modifier_count variants
        expected_count = len(prompts) * len(LinguisticModifier)
        assert len(all_variants) == expected_count
