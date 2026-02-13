# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import dataclasses

import pytest

import modelcypher.core.domain.geometry.sparse_region_domains as mod
from modelcypher.core.domain.geometry.sparse_region_domains import (
    DomainCategory,
    DomainDefinition,
    ProbeCorpus,
    SparseRegionDomains,
    create_probe_corpora,
)


# ---------------------------------------------------------------------------
# DomainCategory enum
# ---------------------------------------------------------------------------


class TestDomainCategory:
    def test_is_str_subclass(self) -> None:
        assert isinstance(DomainCategory.technical, str)

    def test_all_seven_values(self) -> None:
        expected = {"technical", "scientific", "creative", "reasoning", "knowledge", "safety", "custom"}
        actual = {m.name for m in DomainCategory}
        assert actual == expected

    def test_values(self) -> None:
        assert DomainCategory.technical.value == "Technical"
        assert DomainCategory.scientific.value == "Scientific"
        assert DomainCategory.creative.value == "Creative"
        assert DomainCategory.reasoning.value == "Reasoning"
        assert DomainCategory.knowledge.value == "Knowledge"
        assert DomainCategory.safety.value == "Safety"
        assert DomainCategory.custom.value == "Custom"

    def test_count(self) -> None:
        assert len(DomainCategory) == 7


# ---------------------------------------------------------------------------
# DomainDefinition
# ---------------------------------------------------------------------------


class TestDomainDefinition:
    def test_instantiation(self) -> None:
        dd = DomainDefinition(
            name="test",
            description="A test domain",
            category=DomainCategory.custom,
            probe_prompts=["prompt1", "prompt2"],
        )
        assert dd.name == "test"
        assert dd.description == "A test domain"
        assert dd.category == DomainCategory.custom
        assert dd.probe_prompts == ["prompt1", "prompt2"]

    def test_id_property_returns_name(self) -> None:
        dd = DomainDefinition(
            name="MyDomain",
            description="d",
            category=DomainCategory.technical,
            probe_prompts=[],
        )
        assert dd.id == "MyDomain"

    def test_frozen_enforcement(self) -> None:
        dd = DomainDefinition(
            name="frozen",
            description="d",
            category=DomainCategory.custom,
            probe_prompts=[],
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            dd.name = "changed"

    def test_post_init_initializes_keywords_from_none(self) -> None:
        """When keywords is None (default), __post_init__ sets it to []."""
        dd = DomainDefinition(
            name="test",
            description="d",
            category=DomainCategory.custom,
            probe_prompts=[],
        )
        assert dd.keywords == []

    def test_explicit_keywords_not_overwritten(self) -> None:
        """When keywords is explicitly provided, __post_init__ does not overwrite."""
        dd = DomainDefinition(
            name="test",
            description="d",
            category=DomainCategory.custom,
            probe_prompts=[],
            keywords=["alpha", "beta"],
        )
        assert dd.keywords == ["alpha", "beta"]

    def test_expected_active_layer_range_default_none(self) -> None:
        dd = DomainDefinition(
            name="test",
            description="d",
            category=DomainCategory.custom,
            probe_prompts=[],
        )
        assert dd.expected_active_layer_range is None

    def test_expected_active_layer_range_set(self) -> None:
        dd = DomainDefinition(
            name="test",
            description="d",
            category=DomainCategory.custom,
            probe_prompts=[],
            expected_active_layer_range=(0.2, 0.8),
        )
        assert dd.expected_active_layer_range == (0.2, 0.8)


# ---------------------------------------------------------------------------
# SparseRegionDomains class attributes
# ---------------------------------------------------------------------------


class TestSparseRegionDomainsAttributes:
    def test_code_is_domain_definition(self) -> None:
        assert isinstance(SparseRegionDomains.code, DomainDefinition)

    def test_code_category_is_technical(self) -> None:
        assert SparseRegionDomains.code.category == DomainCategory.technical

    def test_code_has_probe_prompts(self) -> None:
        assert len(SparseRegionDomains.code.probe_prompts) > 0

    def test_code_has_keywords(self) -> None:
        assert SparseRegionDomains.code.keywords is not None
        assert len(SparseRegionDomains.code.keywords) > 0

    def test_math_category_is_scientific(self) -> None:
        assert SparseRegionDomains.math.category == DomainCategory.scientific

    def test_safety_category_is_safety(self) -> None:
        assert SparseRegionDomains.safety.category == DomainCategory.safety

    def test_creative_category_is_creative(self) -> None:
        assert SparseRegionDomains.creative.category == DomainCategory.creative

    def test_reasoning_category_is_reasoning(self) -> None:
        assert SparseRegionDomains.reasoning.category == DomainCategory.reasoning


# ---------------------------------------------------------------------------
# SparseRegionDomains.all_built_in
# ---------------------------------------------------------------------------


class TestAllBuiltIn:
    def test_count(self) -> None:
        assert len(SparseRegionDomains.all_built_in) == 10

    def test_all_are_domain_definitions(self) -> None:
        for d in SparseRegionDomains.all_built_in:
            assert isinstance(d, DomainDefinition)

    def test_names_are_unique(self) -> None:
        names = [d.name for d in SparseRegionDomains.all_built_in]
        assert len(names) == len(set(names))

    def test_expected_names(self) -> None:
        names = {d.name for d in SparseRegionDomains.all_built_in}
        expected = {
            "code", "math", "medical", "legal", "creative",
            "reasoning", "physics", "history", "safety", "baseline",
        }
        assert names == expected


# ---------------------------------------------------------------------------
# SparseRegionDomains.domain_named
# ---------------------------------------------------------------------------


class TestDomainNamed:
    def test_existing_domain(self) -> None:
        result = SparseRegionDomains.domain_named("code")
        assert result is not None
        assert result.name == "code"

    def test_case_insensitive(self) -> None:
        result = SparseRegionDomains.domain_named("CODE")
        assert result is not None
        assert result.name == "code"

    def test_nonexistent_returns_none(self) -> None:
        assert SparseRegionDomains.domain_named("nonexistent") is None

    def test_all_built_in_findable(self) -> None:
        for domain in SparseRegionDomains.all_built_in:
            found = SparseRegionDomains.domain_named(domain.name)
            assert found is not None
            assert found.name == domain.name


# ---------------------------------------------------------------------------
# SparseRegionDomains.domains_in_category
# ---------------------------------------------------------------------------


class TestDomainsInCategory:
    def test_technical_includes_code(self) -> None:
        results = SparseRegionDomains.domains_in_category(DomainCategory.technical)
        names = [d.name for d in results]
        assert "code" in names

    def test_scientific_includes_math_medical_physics(self) -> None:
        results = SparseRegionDomains.domains_in_category(DomainCategory.scientific)
        names = {d.name for d in results}
        assert {"math", "medical", "physics"}.issubset(names)

    def test_custom_returns_empty_for_built_in(self) -> None:
        results = SparseRegionDomains.domains_in_category(DomainCategory.custom)
        assert results == []

    def test_all_results_have_matching_category(self) -> None:
        for cat in DomainCategory:
            results = SparseRegionDomains.domains_in_category(cat)
            for d in results:
                assert d.category == cat


# ---------------------------------------------------------------------------
# SparseRegionDomains.custom
# ---------------------------------------------------------------------------


class TestCustomDomain:
    def test_creates_domain_definition(self) -> None:
        dd = SparseRegionDomains.custom(
            name="my_custom",
            description="A custom domain",
            probe_prompts=["What is X?"],
        )
        assert isinstance(dd, DomainDefinition)
        assert dd.name == "my_custom"
        assert dd.category == DomainCategory.custom

    def test_custom_with_keywords(self) -> None:
        dd = SparseRegionDomains.custom(
            name="kw",
            description="d",
            probe_prompts=["p1"],
            keywords=["k1", "k2"],
        )
        assert dd.keywords == ["k1", "k2"]

    def test_custom_without_keywords_defaults_to_empty(self) -> None:
        dd = SparseRegionDomains.custom(
            name="nokw",
            description="d",
            probe_prompts=["p1"],
        )
        assert dd.keywords == []


# ---------------------------------------------------------------------------
# ProbeCorpus
# ---------------------------------------------------------------------------


class TestProbeCorpus:
    def test_instantiation_from_domain(self) -> None:
        domain = SparseRegionDomains.code
        corpus = ProbeCorpus(domain, shuffle=False)
        assert corpus.domain is domain
        assert corpus.count == len(domain.probe_prompts)
        assert set(corpus.prompts) == set(domain.probe_prompts)

    def test_max_prompts_limits_count(self) -> None:
        domain = SparseRegionDomains.code
        corpus = ProbeCorpus(domain, max_prompts=3, shuffle=False)
        assert corpus.count == 3
        assert len(corpus.prompts) == 3

    def test_max_prompts_none_uses_all(self) -> None:
        domain = SparseRegionDomains.math
        corpus = ProbeCorpus(domain, max_prompts=None, shuffle=False)
        assert corpus.count == len(domain.probe_prompts)

    def test_shuffle_preserves_count(self) -> None:
        domain = SparseRegionDomains.reasoning
        corpus = ProbeCorpus(domain, shuffle=True)
        assert corpus.count == len(domain.probe_prompts)
        assert set(corpus.prompts) == set(domain.probe_prompts)

    def test_count_matches_len_prompts(self) -> None:
        domain = SparseRegionDomains.safety
        corpus = ProbeCorpus(domain, max_prompts=5, shuffle=False)
        assert corpus.count == len(corpus.prompts)


# ---------------------------------------------------------------------------
# create_probe_corpora
# ---------------------------------------------------------------------------


class TestCreateProbeCorpora:
    def test_returns_tuple_of_two(self) -> None:
        target, baseline = create_probe_corpora(SparseRegionDomains.code)
        assert isinstance(target, ProbeCorpus)
        assert isinstance(baseline, ProbeCorpus)

    def test_target_domain_matches(self) -> None:
        target, _ = create_probe_corpora(SparseRegionDomains.math)
        assert target.domain is SparseRegionDomains.math

    def test_default_baseline_is_baseline_domain(self) -> None:
        _, baseline = create_probe_corpora(SparseRegionDomains.code)
        assert baseline.domain is SparseRegionDomains.baseline

    def test_custom_baseline(self) -> None:
        _, baseline = create_probe_corpora(
            SparseRegionDomains.code,
            baseline_domain=SparseRegionDomains.physics,
        )
        assert baseline.domain is SparseRegionDomains.physics

    def test_prompts_per_domain_limits_count(self) -> None:
        target, baseline = create_probe_corpora(
            SparseRegionDomains.code,
            prompts_per_domain=4,
        )
        assert target.count <= 4
        assert baseline.count <= 4
