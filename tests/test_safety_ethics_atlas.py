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

"""Tests for safety ethics atlas (ethical concept probes)."""

import pytest

from modelcypher.core.domain.agents.safety_ethics_atlas import (
    ALL_SAFETY_PROBES,
    AUTONOMY_PROBES,
    BOUNDARIES_PROBES,
    ECONOMIC_COERCION_PROBES,
    INFORMED_CONSENT_PROBES,
    PHYSICAL_COERCION_PROBES,
    PSYCHOLOGICAL_COERCION_PROBES,
    SOCIAL_COERCION_PROBES,
    VOLUNTARY_CONSENT_PROBES,
    VULNERABILITY_PROBES,
    CoercionType,
    ConsentType,
    SafetyCategory,
    SafetyConcept,
    SafetyEthicsInventory,
)


class TestSafetyCategory:
    """Tests for SafetyCategory enum."""

    def test_consent_value(self):
        assert SafetyCategory.CONSENT.value == "consent"

    def test_autonomy_value(self):
        assert SafetyCategory.AUTONOMY.value == "autonomy"

    def test_coercion_value(self):
        assert SafetyCategory.COERCION.value == "coercion"

    def test_boundaries_value(self):
        assert SafetyCategory.BOUNDARIES.value == "boundaries"

    def test_vulnerability_value(self):
        assert SafetyCategory.VULNERABILITY.value == "vulnerability"

    def test_all_categories_count(self):
        assert len(SafetyCategory) == 5


class TestConsentType:
    """Tests for ConsentType enum."""

    def test_informed_value(self):
        assert ConsentType.INFORMED.value == "informed"

    def test_voluntary_value(self):
        assert ConsentType.VOLUNTARY.value == "voluntary"

    def test_revocable_value(self):
        assert ConsentType.REVOCABLE.value == "revocable"

    def test_capacity_value(self):
        assert ConsentType.CAPACITY.value == "capacity"

    def test_all_types_count(self):
        assert len(ConsentType) == 4


class TestCoercionType:
    """Tests for CoercionType enum."""

    def test_physical_value(self):
        assert CoercionType.PHYSICAL.value == "physical"

    def test_psychological_value(self):
        assert CoercionType.PSYCHOLOGICAL.value == "psychological"

    def test_economic_value(self):
        assert CoercionType.ECONOMIC.value == "economic"

    def test_social_value(self):
        assert CoercionType.SOCIAL.value == "social"

    def test_all_types_count(self):
        assert len(CoercionType) == 4


class TestSafetyConcept:
    """Tests for SafetyConcept dataclass."""

    def test_required_fields(self):
        concept = SafetyConcept(
            id="test_id",
            category=SafetyCategory.CONSENT,
            subcategory="informed",
            level=3,
            name="Test Concept",
            description="A test concept description.",
            support_texts=("Text 1", "Text 2"),
        )
        assert concept.id == "test_id"
        assert concept.category == SafetyCategory.CONSENT
        assert concept.subcategory == "informed"
        assert concept.level == 3
        assert concept.name == "Test Concept"
        assert len(concept.support_texts) == 2

    def test_default_cross_domain_weight(self):
        concept = SafetyConcept(
            id="test",
            category=SafetyCategory.CONSENT,
            subcategory="informed",
            level=1,
            name="Test",
            description="Test",
            support_texts=(),
        )
        assert concept.cross_domain_weight == 1.0

    def test_custom_cross_domain_weight(self):
        concept = SafetyConcept(
            id="test",
            category=SafetyCategory.CONSENT,
            subcategory="informed",
            level=1,
            name="Test",
            description="Test",
            support_texts=(),
            cross_domain_weight=1.5,
        )
        assert concept.cross_domain_weight == 1.5

    def test_canonical_name_property(self):
        concept = SafetyConcept(
            id="test",
            category=SafetyCategory.CONSENT,
            subcategory="informed",
            level=1,
            name="My Test Name",
            description="Test",
            support_texts=(),
        )
        assert concept.canonical_name == "My Test Name"

    def test_prompt_property(self):
        concept = SafetyConcept(
            id="test",
            category=SafetyCategory.CONSENT,
            subcategory="informed",
            level=1,
            name="Informed Choice",
            description="Test",
            support_texts=(),
        )
        assert concept.prompt == "The concept of informed choice represents"

    def test_frozen_dataclass(self):
        concept = SafetyConcept(
            id="test",
            category=SafetyCategory.CONSENT,
            subcategory="informed",
            level=1,
            name="Test",
            description="Test",
            support_texts=(),
        )
        with pytest.raises(AttributeError):
            concept.name = "New Name"


class TestProbeTuples:
    """Tests for probe tuple definitions."""

    def test_informed_consent_probes_count(self):
        assert len(INFORMED_CONSENT_PROBES) == 3

    def test_voluntary_consent_probes_count(self):
        assert len(VOLUNTARY_CONSENT_PROBES) == 4

    def test_autonomy_probes_count(self):
        assert len(AUTONOMY_PROBES) == 6

    def test_physical_coercion_probes_count(self):
        assert len(PHYSICAL_COERCION_PROBES) == 3

    def test_psychological_coercion_probes_count(self):
        assert len(PSYCHOLOGICAL_COERCION_PROBES) == 4

    def test_economic_coercion_probes_count(self):
        assert len(ECONOMIC_COERCION_PROBES) == 2

    def test_social_coercion_probes_count(self):
        assert len(SOCIAL_COERCION_PROBES) == 2

    def test_boundaries_probes_count(self):
        assert len(BOUNDARIES_PROBES) == 4

    def test_vulnerability_probes_count(self):
        assert len(VULNERABILITY_PROBES) == 6

    def test_all_safety_probes_count(self):
        expected = (
            len(INFORMED_CONSENT_PROBES)
            + len(VOLUNTARY_CONSENT_PROBES)
            + len(AUTONOMY_PROBES)
            + len(PHYSICAL_COERCION_PROBES)
            + len(PSYCHOLOGICAL_COERCION_PROBES)
            + len(ECONOMIC_COERCION_PROBES)
            + len(SOCIAL_COERCION_PROBES)
            + len(BOUNDARIES_PROBES)
            + len(VULNERABILITY_PROBES)
        )
        assert len(ALL_SAFETY_PROBES) == expected
        assert len(ALL_SAFETY_PROBES) == 34  # As documented


class TestProbeContent:
    """Tests for probe content validity."""

    def test_all_probes_have_unique_ids(self):
        ids = [probe.id for probe in ALL_SAFETY_PROBES]
        assert len(ids) == len(set(ids))

    def test_all_probes_have_support_texts(self):
        for probe in ALL_SAFETY_PROBES:
            assert len(probe.support_texts) >= 1, f"{probe.id} has no support texts"

    def test_all_probes_have_valid_level(self):
        for probe in ALL_SAFETY_PROBES:
            assert 1 <= probe.level <= 5, f"{probe.id} has invalid level {probe.level}"

    def test_all_probes_have_valid_category(self):
        for probe in ALL_SAFETY_PROBES:
            assert probe.category in SafetyCategory

    def test_consent_probes_have_consent_category(self):
        all_consent = INFORMED_CONSENT_PROBES + VOLUNTARY_CONSENT_PROBES
        for probe in all_consent:
            assert probe.category == SafetyCategory.CONSENT

    def test_coercion_probes_have_coercion_category(self):
        all_coercion = (
            PHYSICAL_COERCION_PROBES
            + PSYCHOLOGICAL_COERCION_PROBES
            + ECONOMIC_COERCION_PROBES
            + SOCIAL_COERCION_PROBES
        )
        for probe in all_coercion:
            assert probe.category == SafetyCategory.COERCION


class TestSafetyEthicsInventory:
    """Tests for SafetyEthicsInventory class."""

    def test_all_concepts_returns_list(self):
        concepts = SafetyEthicsInventory.all_concepts()
        assert isinstance(concepts, list)
        assert len(concepts) == 34

    def test_by_category_consent(self):
        consent = SafetyEthicsInventory.by_category(SafetyCategory.CONSENT)
        assert len(consent) == 7
        for probe in consent:
            assert probe.category == SafetyCategory.CONSENT

    def test_by_category_autonomy(self):
        autonomy = SafetyEthicsInventory.by_category(SafetyCategory.AUTONOMY)
        assert len(autonomy) == 6
        for probe in autonomy:
            assert probe.category == SafetyCategory.AUTONOMY

    def test_by_category_coercion(self):
        coercion = SafetyEthicsInventory.by_category(SafetyCategory.COERCION)
        assert len(coercion) == 11
        for probe in coercion:
            assert probe.category == SafetyCategory.COERCION

    def test_by_category_boundaries(self):
        boundaries = SafetyEthicsInventory.by_category(SafetyCategory.BOUNDARIES)
        assert len(boundaries) == 4
        for probe in boundaries:
            assert probe.category == SafetyCategory.BOUNDARIES

    def test_by_category_vulnerability(self):
        vulnerability = SafetyEthicsInventory.by_category(SafetyCategory.VULNERABILITY)
        assert len(vulnerability) == 6
        for probe in vulnerability:
            assert probe.category == SafetyCategory.VULNERABILITY

    def test_consent_probes_method(self):
        consent = SafetyEthicsInventory.consent_probes()
        assert len(consent) == 7

    def test_autonomy_probes_method(self):
        autonomy = SafetyEthicsInventory.autonomy_probes()
        assert len(autonomy) == 6

    def test_coercion_probes_method(self):
        coercion = SafetyEthicsInventory.coercion_probes()
        assert len(coercion) == 11

    def test_boundaries_probes_method(self):
        boundaries = SafetyEthicsInventory.boundaries_probes()
        assert len(boundaries) == 4

    def test_vulnerability_probes_method(self):
        vulnerability = SafetyEthicsInventory.vulnerability_probes()
        assert len(vulnerability) == 6

    def test_high_severity_probes(self):
        high_severity = SafetyEthicsInventory.high_severity_probes()
        assert len(high_severity) > 0
        for probe in high_severity:
            assert probe.level == 5

    def test_count(self):
        assert SafetyEthicsInventory.count() == 34

    def test_count_by_category(self):
        counts = SafetyEthicsInventory.count_by_category()
        assert counts[SafetyCategory.CONSENT] == 7
        assert counts[SafetyCategory.AUTONOMY] == 6
        assert counts[SafetyCategory.COERCION] == 11
        assert counts[SafetyCategory.BOUNDARIES] == 4
        assert counts[SafetyCategory.VULNERABILITY] == 6
        assert sum(counts.values()) == 34


class TestSpecificProbes:
    """Tests for specific probe definitions."""

    def test_informed_choice_probe(self):
        probe = next(p for p in INFORMED_CONSENT_PROBES if p.id == "informed_choice")
        assert probe.name == "Informed Choice"
        assert probe.level == 5
        assert probe.cross_domain_weight == 1.5

    def test_bodily_autonomy_probe(self):
        probe = next(p for p in AUTONOMY_PROBES if p.id == "bodily_autonomy")
        assert probe.name == "Bodily Autonomy"
        assert probe.subcategory == "bodily"
        assert probe.level == 5

    def test_manipulation_probe(self):
        probe = next(p for p in PSYCHOLOGICAL_COERCION_PROBES if p.id == "manipulation")
        assert probe.name == "Manipulation"
        assert probe.subcategory == CoercionType.PSYCHOLOGICAL.value
        assert probe.level == 5

    def test_power_imbalance_probe(self):
        probe = next(p for p in VULNERABILITY_PROBES if p.id == "power_imbalance")
        assert probe.name == "Power Imbalance"
        assert probe.subcategory == "power"
        assert probe.level == 5
