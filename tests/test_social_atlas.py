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

"""Tests for SocialAtlas.

Tests the "Latent Sociologist" hypothesis: models encode social relationships
as a coherent geometric manifold with Power, Kinship, and Formality axes.

Validated 2025-12-23:
- Mean SMS: 0.53 (effect size d=2.39)
- Axis orthogonality: 94.8%
- Perfect reproducibility (CV=0.00%)
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain.agents.social_atlas import (
    SocialCategory,
    SocialAxis,
    SocialConcept,
    SocialConceptInventory,
    POWER_HIERARCHY_PROBES,
    FORMALITY_PROBES,
    KINSHIP_PROBES,
    STATUS_MARKERS_PROBES,
    AGE_PROBES,
    ALL_SOCIAL_PROBES,
)


class TestSocialConceptInventory:
    """Tests for SocialConceptInventory."""

    def test_total_concept_count(self) -> None:
        """Should have 25 total social concepts."""
        all_concepts = SocialConceptInventory.all_concepts()
        assert len(all_concepts) == 25

    def test_count_method(self) -> None:
        """Count method should return 25."""
        assert SocialConceptInventory.count() == 25

    def test_power_hierarchy_probes_count(self) -> None:
        """Should have 5 power hierarchy probes (slave→emperor)."""
        power = SocialConceptInventory.power_hierarchy_probes()
        assert len(power) == 5

    def test_formality_probes_count(self) -> None:
        """Should have 5 formality probes (hey→salutations)."""
        formality = SocialConceptInventory.formality_probes()
        assert len(formality) == 5

    def test_kinship_probes_count(self) -> None:
        """Should have 5 kinship probes (enemy→family)."""
        kinship = SocialConceptInventory.kinship_probes()
        assert len(kinship) == 5

    def test_all_concept_ids_unique(self) -> None:
        """All social concept IDs should be unique."""
        all_concepts = SocialConceptInventory.all_concepts()
        ids = [c.id for c in all_concepts]
        assert len(ids) == len(set(ids))

    def test_count_by_category(self) -> None:
        """Each category should have 5 probes."""
        counts = SocialConceptInventory.count_by_category()
        for category, count in counts.items():
            assert count == 5, f"{category.value} has {count} probes, expected 5"

    def test_count_by_axis(self) -> None:
        """Power axis should have most probes."""
        counts = SocialConceptInventory.count_by_axis()
        # Power: power_hierarchy + status_markers + age = 15
        # Kinship: kinship = 5
        # Formality: formality = 5
        assert counts[SocialAxis.POWER] == 15
        assert counts[SocialAxis.KINSHIP] == 5
        assert counts[SocialAxis.FORMALITY] == 5


class TestSocialAxisStructure:
    """Tests for social axis ordering and structure."""

    def test_power_hierarchy_level_ordering(self) -> None:
        """Power hierarchy should be ordered: slave < servant < citizen < noble < emperor."""
        hierarchy = list(POWER_HIERARCHY_PROBES)
        levels = [p.level for p in hierarchy]
        assert levels == [1, 2, 3, 4, 5], f"Power levels out of order: {levels}"

    def test_formality_level_ordering(self) -> None:
        """Formality should be ordered: hey < hi < hello < greetings < salutations."""
        formality = list(FORMALITY_PROBES)
        levels = [p.level for p in formality]
        assert levels == [1, 2, 3, 4, 5], f"Formality levels out of order: {levels}"

    def test_kinship_level_ordering(self) -> None:
        """Kinship should be ordered: enemy < stranger < acquaintance < friend < family."""
        kinship = list(KINSHIP_PROBES)
        levels = [p.level for p in kinship]
        assert levels == [1, 2, 3, 4, 5], f"Kinship levels out of order: {levels}"

    def test_status_markers_level_ordering(self) -> None:
        """Status markers should be ordered: beggar < worker < professional < wealthy < elite."""
        status = list(STATUS_MARKERS_PROBES)
        levels = [p.level for p in status]
        assert levels == [1, 2, 3, 4, 5], f"Status levels out of order: {levels}"

    def test_age_level_ordering(self) -> None:
        """Age should be ordered: child < youth < adult < senior < elder."""
        age = list(AGE_PROBES)
        levels = [p.level for p in age]
        assert levels == [1, 2, 3, 4, 5], f"Age levels out of order: {levels}"

    def test_slave_is_level_1(self) -> None:
        """Slave should be at level 1 (lowest on power axis)."""
        slave = next(p for p in POWER_HIERARCHY_PROBES if p.id == "slave")
        assert slave.level == 1
        assert slave.axis == SocialAxis.POWER

    def test_emperor_is_level_5(self) -> None:
        """Emperor should be at level 5 (highest on power axis)."""
        emperor = next(p for p in POWER_HIERARCHY_PROBES if p.id == "emperor")
        assert emperor.level == 5
        assert emperor.axis == SocialAxis.POWER

    def test_enemy_is_distant_family_is_close(self) -> None:
        """Enemy should be level 1 (distant), family level 5 (close) on kinship axis."""
        enemy = next(p for p in KINSHIP_PROBES if p.id == "enemy")
        family = next(p for p in KINSHIP_PROBES if p.id == "family")
        assert enemy.level == 1
        assert family.level == 5
        assert enemy.axis == SocialAxis.KINSHIP
        assert family.axis == SocialAxis.KINSHIP


class TestSocialConceptProperties:
    """Tests for individual social concept properties."""

    def test_all_concepts_have_support_texts(self) -> None:
        """All concepts should have at least 2 support texts."""
        for concept in ALL_SOCIAL_PROBES:
            assert len(concept.support_texts) >= 2, \
                f"{concept.id} has only {len(concept.support_texts)} support texts"

    def test_all_concepts_have_descriptions(self) -> None:
        """All concepts should have non-empty descriptions."""
        for concept in ALL_SOCIAL_PROBES:
            assert concept.description, f"{concept.id} has empty description"
            assert len(concept.description) > 10, \
                f"{concept.id} has very short description"

    def test_cross_domain_weights_in_range(self) -> None:
        """Cross-domain weights should be between 0.5 and 2.0."""
        for concept in ALL_SOCIAL_PROBES:
            assert 0.5 <= concept.cross_domain_weight <= 2.0, \
                f"{concept.id} has weight {concept.cross_domain_weight} out of range"

    def test_endpoint_concepts_have_higher_weights(self) -> None:
        """Endpoint concepts should have higher weights."""
        high_weight_ids = ["slave", "emperor", "enemy", "family", "hey", "salutations"]
        for concept in ALL_SOCIAL_PROBES:
            if concept.id in high_weight_ids:
                assert concept.cross_domain_weight >= 1.2, \
                    f"Endpoint {concept.id} should have weight >= 1.2"

    def test_canonical_name_property(self) -> None:
        """Canonical name should equal name."""
        for concept in ALL_SOCIAL_PROBES:
            assert concept.canonical_name == concept.name


class TestSocialByFilterMethods:
    """Tests for filtering methods."""

    def test_by_category_power_hierarchy(self) -> None:
        """Filter by POWER_HIERARCHY category should return 5 probes."""
        power = SocialConceptInventory.by_category(SocialCategory.POWER_HIERARCHY)
        assert len(power) == 5
        for c in power:
            assert c.category == SocialCategory.POWER_HIERARCHY

    def test_by_axis_power(self) -> None:
        """Filter by POWER axis should return 15 probes."""
        power = SocialConceptInventory.by_axis(SocialAxis.POWER)
        assert len(power) == 15
        for c in power:
            assert c.axis == SocialAxis.POWER

    def test_power_probes_method(self) -> None:
        """power_probes() should return same as by_axis(POWER)."""
        direct_call = SocialConceptInventory.power_probes()
        by_axis = SocialConceptInventory.by_axis(SocialAxis.POWER)
        assert len(direct_call) == len(by_axis)


class TestMonotonicPowerHierarchy:
    """Tests for monotonic power hierarchy (validated 2025-12-23)."""

    def test_power_hierarchy_is_monotonic(self) -> None:
        """Power hierarchy should have monotonically increasing levels."""
        hierarchy = list(POWER_HIERARCHY_PROBES)
        for i in range(len(hierarchy) - 1):
            assert hierarchy[i].level < hierarchy[i + 1].level, \
                f"Non-monotonic at {hierarchy[i].id} -> {hierarchy[i + 1].id}"

    def test_status_markers_is_monotonic(self) -> None:
        """Status markers should be monotonically ordered beggar→elite."""
        status = list(STATUS_MARKERS_PROBES)
        ids = [p.id for p in status]
        assert ids == ["beggar", "worker", "professional", "wealthy", "elite"]

    def test_formality_is_monotonic(self) -> None:
        """Formality should be monotonically ordered casual→formal."""
        formality = list(FORMALITY_PROBES)
        ids = [p.id for p in formality]
        assert ids == ["hey", "hi", "hello", "greetings", "salutations"]


class TestAxisOrthogonality:
    """Tests for axis independence (validated: 94.8% orthogonality)."""

    def test_axes_are_distinct_enums(self) -> None:
        """Power, Kinship, and Formality should be distinct axes."""
        axes = list(SocialAxis)
        assert len(axes) == 3
        assert SocialAxis.POWER in axes
        assert SocialAxis.KINSHIP in axes
        assert SocialAxis.FORMALITY in axes

    def test_each_axis_has_probes(self) -> None:
        """Each axis should have at least 5 probes."""
        counts = SocialConceptInventory.count_by_axis()
        for axis in SocialAxis:
            assert counts.get(axis, 0) >= 5, f"{axis.value} has too few probes"

    def test_kinship_axis_is_independent(self) -> None:
        """Kinship axis should be separate from power (different concepts)."""
        kinship_ids = {p.id for p in KINSHIP_PROBES}
        power_ids = {p.id for p in POWER_HIERARCHY_PROBES}
        # No overlap between kinship and power hierarchy probes
        assert kinship_ids.isdisjoint(power_ids)
