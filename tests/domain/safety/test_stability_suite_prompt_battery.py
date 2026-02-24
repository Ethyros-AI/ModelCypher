"""Tests for StabilitySuitePromptBattery prompt counts, categories, and tiers."""

import pytest

from modelcypher.core.domain.safety.stability_suite.stability_suite_models import (
    StabilitySuitePromptCategory,
    StabilitySuiteTier,
)
from modelcypher.core.domain.safety.stability_suite.stability_suite_prompt_battery import (
    StabilitySuitePromptBattery,
)

# ---------------------------------------------------------------------------
# Tier prompt counts
# ---------------------------------------------------------------------------

class TestTierPromptCounts:
    """Each tier returns the documented number of prompts."""

    def test_quick_returns_6(self):
        prompts = StabilitySuitePromptBattery.prompts(StabilitySuiteTier.QUICK)
        assert len(prompts) == 6

    def test_standard_returns_12(self):
        prompts = StabilitySuitePromptBattery.prompts(StabilitySuiteTier.STANDARD)
        assert len(prompts) == 12

    def test_full_returns_19(self):
        prompts = StabilitySuitePromptBattery.prompts(StabilitySuiteTier.FULL)
        assert len(prompts) == 19


# ---------------------------------------------------------------------------
# Prompt uniqueness
# ---------------------------------------------------------------------------

class TestPromptUniqueness:
    """All prompt IDs are unique within each tier."""

    @pytest.mark.parametrize("tier", list(StabilitySuiteTier))
    def test_unique_ids(self, tier):
        prompts = StabilitySuitePromptBattery.prompts(tier)
        ids = [p.id for p in prompts]
        assert len(ids) == len(set(ids)), f"Duplicate IDs in {tier.value}: {ids}"


# ---------------------------------------------------------------------------
# Category validity
# ---------------------------------------------------------------------------

class TestCategoryValidity:
    """All prompts belong to valid categories."""

    @pytest.mark.parametrize("tier", list(StabilitySuiteTier))
    def test_all_categories_valid(self, tier):
        prompts = StabilitySuitePromptBattery.prompts(tier)
        valid = set(StabilitySuitePromptCategory)
        for p in prompts:
            assert p.category in valid, f"{p.id} has invalid category {p.category}"


# ---------------------------------------------------------------------------
# Quick is subset of Standard
# ---------------------------------------------------------------------------

class TestTierSubset:
    """QUICK prompts are a prefix of STANDARD prompts."""

    def test_quick_is_subset_of_standard(self):
        quick = StabilitySuitePromptBattery.prompts(StabilitySuiteTier.QUICK)
        standard = StabilitySuitePromptBattery.prompts(StabilitySuiteTier.STANDARD)
        quick_ids = [p.id for p in quick]
        standard_ids = [p.id for p in standard]
        assert quick_ids == standard_ids[:6]

    def test_standard_is_subset_of_full(self):
        standard = StabilitySuitePromptBattery.prompts(StabilitySuiteTier.STANDARD)
        full = StabilitySuitePromptBattery.prompts(StabilitySuiteTier.FULL)
        standard_ids = set(p.id for p in standard)
        full_ids = set(p.id for p in full)
        assert standard_ids.issubset(full_ids)


# ---------------------------------------------------------------------------
# categories()
# ---------------------------------------------------------------------------

class TestCategories:
    """categories() returns all enum values."""

    def test_returns_all_categories(self):
        cats = StabilitySuitePromptBattery.categories()
        assert set(cats) == set(StabilitySuitePromptCategory)

    def test_returns_list(self):
        cats = StabilitySuitePromptBattery.categories()
        assert isinstance(cats, list)


# ---------------------------------------------------------------------------
# prompts_by_category
# ---------------------------------------------------------------------------

class TestPromptsByCategory:
    """Grouping by category."""

    def test_groups_correctly(self):
        grouped = StabilitySuitePromptBattery.prompts_by_category(StabilitySuiteTier.STANDARD)
        all_prompts = StabilitySuitePromptBattery.prompts(StabilitySuiteTier.STANDARD)
        total_in_groups = sum(len(v) for v in grouped.values())
        assert total_in_groups == len(all_prompts)

    def test_each_prompt_in_correct_category(self):
        grouped = StabilitySuitePromptBattery.prompts_by_category(StabilitySuiteTier.FULL)
        for cat, prompts in grouped.items():
            for p in prompts:
                assert p.category == cat, f"{p.id} in wrong group"

    def test_all_keys_are_valid_categories(self):
        grouped = StabilitySuitePromptBattery.prompts_by_category(StabilitySuiteTier.STANDARD)
        for key in grouped.keys():
            assert isinstance(key, StabilitySuitePromptCategory)

    @pytest.mark.parametrize("tier", list(StabilitySuiteTier))
    def test_no_empty_groups(self, tier):
        grouped = StabilitySuitePromptBattery.prompts_by_category(tier)
        for cat, prompts in grouped.items():
            assert len(prompts) > 0, f"Empty group for {cat.value}"


# ---------------------------------------------------------------------------
# Structured output prompts
# ---------------------------------------------------------------------------

class TestStructuredOutputPrompts:
    """Structured output prompts have expects_agent_action_json=True."""

    def test_structured_output_prompts_expect_json(self):
        prompts = StabilitySuitePromptBattery.prompts(StabilitySuiteTier.FULL)
        structured = [
            p for p in prompts
            if p.category == StabilitySuitePromptCategory.STRUCTURED_OUTPUT
        ]
        assert len(structured) > 0, "No structured output prompts found"
        for p in structured:
            assert p.expects_agent_action_json is True, (
                f"{p.id} should expect agent action JSON"
            )

    def test_non_structured_prompts_do_not_expect_json(self):
        prompts = StabilitySuitePromptBattery.prompts(StabilitySuiteTier.FULL)
        non_structured = [
            p for p in prompts
            if p.category != StabilitySuitePromptCategory.STRUCTURED_OUTPUT
        ]
        for p in non_structured:
            assert p.expects_agent_action_json is False, (
                f"{p.id} should not expect agent action JSON"
            )


# ---------------------------------------------------------------------------
# Prompt content
# ---------------------------------------------------------------------------

class TestPromptContent:
    """All prompts have non-empty title and user_prompt."""

    @pytest.mark.parametrize("tier", list(StabilitySuiteTier))
    def test_all_have_nonempty_title(self, tier):
        for p in StabilitySuitePromptBattery.prompts(tier):
            assert p.title.strip(), f"{p.id} has empty title"

    @pytest.mark.parametrize("tier", list(StabilitySuiteTier))
    def test_all_have_nonempty_user_prompt(self, tier):
        for p in StabilitySuitePromptBattery.prompts(tier):
            assert p.user_prompt.strip(), f"{p.id} has empty user_prompt"

    @pytest.mark.parametrize("tier", list(StabilitySuiteTier))
    def test_all_have_nonempty_id(self, tier):
        for p in StabilitySuitePromptBattery.prompts(tier):
            assert p.id.strip(), "prompt has empty id"
