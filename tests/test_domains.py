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

"""Tests for unified domain resolution.

These tests ensure that:
1. AtlasDomain is the single source of truth for all domain definitions
2. All domain aliases resolve correctly
3. The resolve_domain function handles user input properly
"""

import pytest

from modelcypher.core.domain.domains import (
    AtlasDomain,
    list_all_domains,
    list_domain_aliases,
    resolve_domain,
    resolve_domains,
)


class TestAtlasDomainEnum:
    """Tests for the AtlasDomain enum."""

    def test_all_domains_exist(self):
        """Verify all expected domains are defined."""
        expected = {
            "mathematical",
            "logical",
            "linguistic",
            "mental",
            "computational",
            "structural",
            "affective",
            "relational",
            "temporal",
            "spatial",
            "moral",
            "safety",
            "philosophical",
            "factual",
            "physical",
        }
        actual = {d.value for d in AtlasDomain}
        assert actual == expected

    def test_domain_values_are_lowercase(self):
        """All domain values should be lowercase strings."""
        for domain in AtlasDomain:
            assert domain.value == domain.value.lower()
            assert domain.value.isalpha()

    def test_domain_is_string_enum(self):
        """AtlasDomain should be a string enum."""
        assert AtlasDomain.MATHEMATICAL == "mathematical"
        assert str(AtlasDomain.MATHEMATICAL) == "AtlasDomain.MATHEMATICAL"


class TestResolveDomain:
    """Tests for the resolve_domain function."""

    def test_resolve_exact_name(self):
        """Resolving exact domain name should work."""
        assert resolve_domain("mathematical") == AtlasDomain.MATHEMATICAL
        assert resolve_domain("logical") == AtlasDomain.LOGICAL
        assert resolve_domain("spatial") == AtlasDomain.SPATIAL

    def test_resolve_case_insensitive(self):
        """Domain resolution should be case-insensitive."""
        assert resolve_domain("MATHEMATICAL") == AtlasDomain.MATHEMATICAL
        assert resolve_domain("Mathematical") == AtlasDomain.MATHEMATICAL
        assert resolve_domain("mAtHeMaTiCaL") == AtlasDomain.MATHEMATICAL

    def test_resolve_with_whitespace(self):
        """Domain resolution should strip whitespace."""
        assert resolve_domain("  mathematical  ") == AtlasDomain.MATHEMATICAL
        assert resolve_domain("\tlogical\n") == AtlasDomain.LOGICAL

    def test_resolve_common_aliases(self):
        """Common aliases should resolve correctly."""
        # Mathematical aliases
        assert resolve_domain("math") == AtlasDomain.MATHEMATICAL
        assert resolve_domain("numeric") == AtlasDomain.MATHEMATICAL
        assert resolve_domain("quantitative") == AtlasDomain.MATHEMATICAL

        # Logical aliases
        assert resolve_domain("logic") == AtlasDomain.LOGICAL
        assert resolve_domain("boolean") == AtlasDomain.LOGICAL
        assert resolve_domain("inference") == AtlasDomain.LOGICAL

        # Computational aliases
        assert resolve_domain("coding") == AtlasDomain.COMPUTATIONAL
        assert resolve_domain("code") == AtlasDomain.COMPUTATIONAL
        assert resolve_domain("programming") == AtlasDomain.COMPUTATIONAL

    def test_resolve_social_to_relational(self):
        """'social' should resolve to RELATIONAL for backward compatibility."""
        assert resolve_domain("social") == AtlasDomain.RELATIONAL
        assert resolve_domain("interpersonal") == AtlasDomain.RELATIONAL

    def test_resolve_unknown_returns_none(self):
        """Unknown domains should return None."""
        assert resolve_domain("unknown_domain") is None
        assert resolve_domain("foobar") is None
        assert resolve_domain("") is None

    def test_resolve_all_taxonomy_domains(self):
        """All domains from domain_taxonomy.yaml should resolve."""
        # Core domains from taxonomy
        taxonomy_domains = [
            "mathematical",
            "logical",
            "computational",
            "linguistic",
            "affective",
            "relational",
            "temporal",
            "spatial",
            "moral",
            "philosophical",
            "factual",
        ]
        for domain in taxonomy_domains:
            result = resolve_domain(domain)
            assert result is not None, f"Domain '{domain}' should resolve"


class TestResolveDomains:
    """Tests for the resolve_domains function."""

    def test_resolve_multiple_domains(self):
        """Should resolve multiple domains correctly."""
        result = resolve_domains(["mathematical", "logical", "spatial"])
        assert len(result) == 3
        assert AtlasDomain.MATHEMATICAL in result
        assert AtlasDomain.LOGICAL in result
        assert AtlasDomain.SPATIAL in result

    def test_resolve_with_aliases(self):
        """Should resolve aliases in a list."""
        result = resolve_domains(["math", "coding", "social"])
        assert len(result) == 3
        assert AtlasDomain.MATHEMATICAL in result
        assert AtlasDomain.COMPUTATIONAL in result
        assert AtlasDomain.RELATIONAL in result

    def test_skip_unknown_domains(self):
        """Should skip unknown domains and log warning."""
        result = resolve_domains(["mathematical", "unknown_domain", "logical"])
        assert len(result) == 2
        assert AtlasDomain.MATHEMATICAL in result
        assert AtlasDomain.LOGICAL in result

    def test_no_duplicates(self):
        """Should not include duplicate domains."""
        result = resolve_domains(["math", "mathematical", "numeric"])
        assert len(result) == 1
        assert result[0] == AtlasDomain.MATHEMATICAL

    def test_empty_input(self):
        """Empty input should return empty list."""
        assert resolve_domains([]) == []

    def test_all_unknown_returns_empty(self):
        """All unknown domains should return empty list."""
        result = resolve_domains(["foo", "bar", "baz"])
        assert result == []


class TestUnifiedAtlasIntegration:
    """Tests for integration with unified_atlas module."""

    def test_import_atlas_domain_from_unified_atlas(self):
        """Should be able to import AtlasDomain from unified_atlas."""
        from modelcypher.core.domain.agents.unified_atlas import AtlasDomain as UnifiedDomain

        assert UnifiedDomain is AtlasDomain

    def test_unified_atlas_domain_mappings(self):
        """Unified atlas domain mappings should use canonical AtlasDomain."""
        from modelcypher.core.domain.agents.unified_atlas import (
            _SEQUENCE_DOMAIN_MAP,
            AtlasDomain as UnifiedDomain,
        )

        # All values in the mapping should be AtlasDomain instances
        for value in _SEQUENCE_DOMAIN_MAP.values():
            assert isinstance(value, UnifiedDomain)
            assert isinstance(value, AtlasDomain)


class TestListFunctions:
    """Tests for list_all_domains and list_domain_aliases."""

    def test_list_all_domains(self):
        """list_all_domains should return all AtlasDomain values."""
        domains = list_all_domains()
        assert len(domains) == len(AtlasDomain)
        for domain in AtlasDomain:
            assert domain in domains

    def test_list_domain_aliases(self):
        """list_domain_aliases should return alias mappings."""
        aliases = list_domain_aliases()
        assert isinstance(aliases, dict)
        assert len(aliases) > 0

        # Check some expected aliases
        assert aliases.get("math") == AtlasDomain.MATHEMATICAL
        assert aliases.get("social") == AtlasDomain.RELATIONAL


class TestMergePipelineServiceIntegration:
    """Tests ensuring merge service uses unified domains correctly."""

    def test_merge_pipeline_imports_from_domains(self):
        """merge service should import from domains module."""
        # This test verifies the import works without errors
        from modelcypher.core.use_cases.merge.service import MergePipelineService

        assert MergePipelineService is not None

    def test_resolve_transplant_domains(self):
        """Common transplant domain inputs should resolve."""
        # These are the domains users typically specify for transplant
        common_inputs = [
            ["mathematical", "logical"],
            ["math", "logic"],
            ["coding", "reasoning"],
            ["spatial", "temporal", "moral"],
            ["social", "emotional"],  # social -> RELATIONAL, emotional -> AFFECTIVE
        ]

        for inputs in common_inputs:
            result = resolve_domains(inputs)
            assert len(result) > 0, f"Inputs {inputs} should resolve to at least one domain"
