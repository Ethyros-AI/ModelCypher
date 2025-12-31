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

"""Tests for philosophical atlas (philosophical concept probes)."""

import pytest

from modelcypher.core.domain.agents.philosophical_atlas import (
    ALL_PHILOSOPHICAL_PROBES,
    EPISTEMOLOGICAL_PROBES,
    LOGICAL_PROBES,
    MEREOLOGICAL_PROBES,
    MODAL_PROBES,
    ONTOLOGICAL_PROBES,
    PhilosophicalAxis,
    PhilosophicalCategory,
    PhilosophicalConcept,
    PhilosophicalConceptInventory,
)


class TestPhilosophicalCategory:
    """Tests for PhilosophicalCategory enum."""

    def test_ontological_value(self):
        assert PhilosophicalCategory.ONTOLOGICAL.value == "ontological"

    def test_epistemological_value(self):
        assert PhilosophicalCategory.EPISTEMOLOGICAL.value == "epistemological"

    def test_logical_value(self):
        assert PhilosophicalCategory.LOGICAL.value == "logical"

    def test_modal_value(self):
        assert PhilosophicalCategory.MODAL.value == "modal"

    def test_mereological_value(self):
        assert PhilosophicalCategory.MEREOLOGICAL.value == "mereological"

    def test_all_categories_count(self):
        assert len(PhilosophicalCategory) == 5

    def test_string_enum(self):
        assert PhilosophicalCategory.ONTOLOGICAL == "ontological"


class TestPhilosophicalAxis:
    """Tests for PhilosophicalAxis enum."""

    def test_being_value(self):
        assert PhilosophicalAxis.BEING.value == "being"

    def test_truth_value(self):
        assert PhilosophicalAxis.TRUTH.value == "truth"

    def test_unity_value(self):
        assert PhilosophicalAxis.UNITY.value == "unity"

    def test_all_axes_count(self):
        assert len(PhilosophicalAxis) == 3

    def test_string_enum(self):
        assert PhilosophicalAxis.BEING == "being"


class TestPhilosophicalConcept:
    """Tests for PhilosophicalConcept dataclass."""

    @pytest.fixture
    def sample_concept(self):
        return PhilosophicalConcept(
            id="test_concept",
            category=PhilosophicalCategory.ONTOLOGICAL,
            axis=PhilosophicalAxis.BEING,
            level=1,
            name="Test Concept",
            description="A test concept for testing.",
            support_texts=("First support text.", "Second support text."),
        )

    def test_required_fields(self, sample_concept):
        assert sample_concept.id == "test_concept"
        assert sample_concept.category == PhilosophicalCategory.ONTOLOGICAL
        assert sample_concept.axis == PhilosophicalAxis.BEING
        assert sample_concept.level == 1
        assert sample_concept.name == "Test Concept"
        assert sample_concept.description == "A test concept for testing."

    def test_support_texts_is_tuple(self, sample_concept):
        assert isinstance(sample_concept.support_texts, tuple)
        assert len(sample_concept.support_texts) == 2

    def test_default_cross_domain_weight(self, sample_concept):
        assert sample_concept.cross_domain_weight == 1.0

    def test_custom_cross_domain_weight(self):
        concept = PhilosophicalConcept(
            id="weighted",
            category=PhilosophicalCategory.LOGICAL,
            axis=PhilosophicalAxis.TRUTH,
            level=3,
            name="Weighted",
            description="Custom weight.",
            support_texts=("Text.",),
            cross_domain_weight=0.8,
        )
        assert concept.cross_domain_weight == 0.8

    def test_canonical_name_property(self, sample_concept):
        assert sample_concept.canonical_name == "Test Concept"
        assert sample_concept.canonical_name == sample_concept.name

    def test_frozen_dataclass(self, sample_concept):
        with pytest.raises(AttributeError):
            sample_concept.id = "new_id"  # type: ignore

    def test_hashable(self, sample_concept):
        # Frozen dataclasses are hashable
        concept_set = {sample_concept}
        assert sample_concept in concept_set


class TestOntologicalProbes:
    """Tests for ONTOLOGICAL_PROBES tuple."""

    def test_probe_count(self):
        assert len(ONTOLOGICAL_PROBES) == 6

    def test_all_ontological_category(self):
        for probe in ONTOLOGICAL_PROBES:
            assert probe.category == PhilosophicalCategory.ONTOLOGICAL

    def test_all_being_axis(self):
        for probe in ONTOLOGICAL_PROBES:
            assert probe.axis == PhilosophicalAxis.BEING

    def test_levels_one_to_six(self):
        levels = [p.level for p in ONTOLOGICAL_PROBES]
        assert sorted(levels) == [1, 2, 3, 4, 5, 6]

    def test_non_being_is_first(self):
        assert ONTOLOGICAL_PROBES[0].id == "non_being"
        assert ONTOLOGICAL_PROBES[0].level == 1

    def test_necessary_being_is_last(self):
        assert ONTOLOGICAL_PROBES[-1].id == "necessary_being"
        assert ONTOLOGICAL_PROBES[-1].level == 6

    def test_all_have_support_texts(self):
        for probe in ONTOLOGICAL_PROBES:
            assert len(probe.support_texts) >= 1

    def test_contains_key_concepts(self):
        ids = [p.id for p in ONTOLOGICAL_PROBES]
        assert "non_being" in ids
        assert "potential" in ids
        assert "becoming" in ids
        assert "actual" in ids
        assert "substance" in ids
        assert "necessary_being" in ids


class TestEpistemologicalProbes:
    """Tests for EPISTEMOLOGICAL_PROBES tuple."""

    def test_probe_count(self):
        assert len(EPISTEMOLOGICAL_PROBES) == 6

    def test_all_epistemological_category(self):
        for probe in EPISTEMOLOGICAL_PROBES:
            assert probe.category == PhilosophicalCategory.EPISTEMOLOGICAL

    def test_all_truth_axis(self):
        for probe in EPISTEMOLOGICAL_PROBES:
            assert probe.axis == PhilosophicalAxis.TRUTH

    def test_levels_ordered(self):
        levels = [p.level for p in EPISTEMOLOGICAL_PROBES]
        assert sorted(levels) == [1, 2, 3, 4, 5, 6]

    def test_ignorance_to_wisdom_gradient(self):
        assert EPISTEMOLOGICAL_PROBES[0].id == "ignorance"
        assert EPISTEMOLOGICAL_PROBES[-1].id == "wisdom"

    def test_contains_key_concepts(self):
        ids = [p.id for p in EPISTEMOLOGICAL_PROBES]
        assert "ignorance" in ids
        assert "opinion" in ids
        assert "belief" in ids
        assert "understanding" in ids
        assert "knowledge" in ids
        assert "wisdom" in ids


class TestLogicalProbes:
    """Tests for LOGICAL_PROBES tuple."""

    def test_probe_count(self):
        assert len(LOGICAL_PROBES) == 6

    def test_all_logical_category(self):
        for probe in LOGICAL_PROBES:
            assert probe.category == PhilosophicalCategory.LOGICAL

    def test_all_truth_axis(self):
        for probe in LOGICAL_PROBES:
            assert probe.axis == PhilosophicalAxis.TRUTH

    def test_levels_ordered(self):
        levels = [p.level for p in LOGICAL_PROBES]
        assert sorted(levels) == [1, 2, 3, 4, 5, 6]

    def test_contradiction_to_necessity_gradient(self):
        assert LOGICAL_PROBES[0].id == "contradiction"
        assert LOGICAL_PROBES[-1].id == "necessity"

    def test_contains_key_concepts(self):
        ids = [p.id for p in LOGICAL_PROBES]
        assert "contradiction" in ids
        assert "negation" in ids
        assert "contingency" in ids
        assert "implication" in ids
        assert "identity" in ids
        assert "necessity" in ids


class TestModalProbes:
    """Tests for MODAL_PROBES tuple."""

    def test_probe_count(self):
        assert len(MODAL_PROBES) == 6

    def test_all_modal_category(self):
        for probe in MODAL_PROBES:
            assert probe.category == PhilosophicalCategory.MODAL

    def test_all_being_axis(self):
        for probe in MODAL_PROBES:
            assert probe.axis == PhilosophicalAxis.BEING

    def test_levels_ordered(self):
        levels = [p.level for p in MODAL_PROBES]
        assert sorted(levels) == [1, 2, 3, 4, 5, 6]

    def test_impossibility_to_absolute_gradient(self):
        assert MODAL_PROBES[0].id == "impossibility"
        assert MODAL_PROBES[-1].id == "absolute"

    def test_contains_key_concepts(self):
        ids = [p.id for p in MODAL_PROBES]
        assert "impossibility" in ids
        assert "possibility" in ids
        assert "contingent_being" in ids
        assert "actuality_modal" in ids
        assert "necessity_modal" in ids
        assert "absolute" in ids


class TestMereologicalProbes:
    """Tests for MEREOLOGICAL_PROBES tuple."""

    def test_probe_count(self):
        assert len(MEREOLOGICAL_PROBES) == 6

    def test_all_mereological_category(self):
        for probe in MEREOLOGICAL_PROBES:
            assert probe.category == PhilosophicalCategory.MEREOLOGICAL

    def test_all_unity_axis(self):
        for probe in MEREOLOGICAL_PROBES:
            assert probe.axis == PhilosophicalAxis.UNITY

    def test_levels_ordered(self):
        levels = [p.level for p in MEREOLOGICAL_PROBES]
        assert sorted(levels) == [1, 2, 3, 4, 5, 6]

    def test_plurality_to_unity_gradient(self):
        assert MEREOLOGICAL_PROBES[0].id == "plurality"
        assert MEREOLOGICAL_PROBES[-1].id == "unity"

    def test_contains_key_concepts(self):
        ids = [p.id for p in MEREOLOGICAL_PROBES]
        assert "plurality" in ids
        assert "part" in ids
        assert "aggregate" in ids
        assert "whole" in ids
        assert "composition" in ids
        assert "unity" in ids


class TestAllPhilosophicalProbes:
    """Tests for ALL_PHILOSOPHICAL_PROBES tuple."""

    def test_total_count(self):
        assert len(ALL_PHILOSOPHICAL_PROBES) == 30

    def test_all_unique_ids(self):
        ids = [p.id for p in ALL_PHILOSOPHICAL_PROBES]
        assert len(ids) == len(set(ids))

    def test_contains_all_categories(self):
        categories = {p.category for p in ALL_PHILOSOPHICAL_PROBES}
        assert len(categories) == 5

    def test_contains_all_axes(self):
        axes = {p.axis for p in ALL_PHILOSOPHICAL_PROBES}
        assert len(axes) == 3

    def test_all_have_descriptions(self):
        for probe in ALL_PHILOSOPHICAL_PROBES:
            assert len(probe.description) > 0

    def test_all_have_support_texts(self):
        for probe in ALL_PHILOSOPHICAL_PROBES:
            assert len(probe.support_texts) >= 1

    def test_all_weights_positive(self):
        for probe in ALL_PHILOSOPHICAL_PROBES:
            assert probe.cross_domain_weight > 0


class TestPhilosophicalConceptInventory:
    """Tests for PhilosophicalConceptInventory class."""

    def test_all_concepts_returns_list(self):
        concepts = PhilosophicalConceptInventory.all_concepts()
        assert isinstance(concepts, list)
        assert len(concepts) == 30

    def test_count(self):
        assert PhilosophicalConceptInventory.count() == 30


class TestPhilosophicalConceptInventoryByCategory:
    """Tests for by_category method."""

    def test_ontological_count(self):
        probes = PhilosophicalConceptInventory.by_category(
            PhilosophicalCategory.ONTOLOGICAL
        )
        assert len(probes) == 6

    def test_epistemological_count(self):
        probes = PhilosophicalConceptInventory.by_category(
            PhilosophicalCategory.EPISTEMOLOGICAL
        )
        assert len(probes) == 6

    def test_logical_count(self):
        probes = PhilosophicalConceptInventory.by_category(
            PhilosophicalCategory.LOGICAL
        )
        assert len(probes) == 6

    def test_modal_count(self):
        probes = PhilosophicalConceptInventory.by_category(PhilosophicalCategory.MODAL)
        assert len(probes) == 6

    def test_mereological_count(self):
        probes = PhilosophicalConceptInventory.by_category(
            PhilosophicalCategory.MEREOLOGICAL
        )
        assert len(probes) == 6

    def test_all_categories_match(self):
        for category in PhilosophicalCategory:
            probes = PhilosophicalConceptInventory.by_category(category)
            for probe in probes:
                assert probe.category == category


class TestPhilosophicalConceptInventoryByAxis:
    """Tests for by_axis method."""

    def test_being_axis_count(self):
        # Ontological (6) + Modal (6) = 12
        probes = PhilosophicalConceptInventory.by_axis(PhilosophicalAxis.BEING)
        assert len(probes) == 12

    def test_truth_axis_count(self):
        # Epistemological (6) + Logical (6) = 12
        probes = PhilosophicalConceptInventory.by_axis(PhilosophicalAxis.TRUTH)
        assert len(probes) == 12

    def test_unity_axis_count(self):
        # Mereological (6) = 6
        probes = PhilosophicalConceptInventory.by_axis(PhilosophicalAxis.UNITY)
        assert len(probes) == 6

    def test_all_axes_match(self):
        for axis in PhilosophicalAxis:
            probes = PhilosophicalConceptInventory.by_axis(axis)
            for probe in probes:
                assert probe.axis == axis


class TestPhilosophicalConceptInventoryCategoryMethods:
    """Tests for category-specific methods."""

    def test_ontological_probes(self):
        probes = PhilosophicalConceptInventory.ontological_probes()
        assert len(probes) == 6
        assert all(p.category == PhilosophicalCategory.ONTOLOGICAL for p in probes)

    def test_epistemological_probes(self):
        probes = PhilosophicalConceptInventory.epistemological_probes()
        assert len(probes) == 6
        assert all(p.category == PhilosophicalCategory.EPISTEMOLOGICAL for p in probes)

    def test_logical_probes(self):
        probes = PhilosophicalConceptInventory.logical_probes()
        assert len(probes) == 6
        assert all(p.category == PhilosophicalCategory.LOGICAL for p in probes)

    def test_modal_probes(self):
        probes = PhilosophicalConceptInventory.modal_probes()
        assert len(probes) == 6
        assert all(p.category == PhilosophicalCategory.MODAL for p in probes)

    def test_mereological_probes(self):
        probes = PhilosophicalConceptInventory.mereological_probes()
        assert len(probes) == 6
        assert all(p.category == PhilosophicalCategory.MEREOLOGICAL for p in probes)


class TestPhilosophicalConceptInventoryAxisMethods:
    """Tests for axis-specific methods."""

    def test_being_axis_probes(self):
        probes = PhilosophicalConceptInventory.being_axis_probes()
        assert len(probes) == 12
        assert all(p.axis == PhilosophicalAxis.BEING for p in probes)

    def test_truth_axis_probes(self):
        probes = PhilosophicalConceptInventory.truth_axis_probes()
        assert len(probes) == 12
        assert all(p.axis == PhilosophicalAxis.TRUTH for p in probes)

    def test_unity_axis_probes(self):
        probes = PhilosophicalConceptInventory.unity_axis_probes()
        assert len(probes) == 6
        assert all(p.axis == PhilosophicalAxis.UNITY for p in probes)


class TestPhilosophicalConceptInventoryCountMethods:
    """Tests for count methods."""

    def test_count_by_category(self):
        counts = PhilosophicalConceptInventory.count_by_category()
        assert isinstance(counts, dict)
        assert len(counts) == 5
        assert all(v == 6 for v in counts.values())

    def test_count_by_category_keys(self):
        counts = PhilosophicalConceptInventory.count_by_category()
        for category in PhilosophicalCategory:
            assert category in counts

    def test_count_by_axis(self):
        counts = PhilosophicalConceptInventory.count_by_axis()
        assert isinstance(counts, dict)
        assert len(counts) == 3

    def test_count_by_axis_values(self):
        counts = PhilosophicalConceptInventory.count_by_axis()
        assert counts[PhilosophicalAxis.BEING] == 12
        assert counts[PhilosophicalAxis.TRUTH] == 12
        assert counts[PhilosophicalAxis.UNITY] == 6

    def test_count_by_axis_sums_to_total(self):
        counts = PhilosophicalConceptInventory.count_by_axis()
        assert sum(counts.values()) == 30


class TestProbeStructure:
    """Tests for structural properties of probes."""

    def test_all_probes_have_at_least_4_support_texts(self):
        # Per module docstring, each probe has multiple support texts
        for probe in ALL_PHILOSOPHICAL_PROBES:
            assert len(probe.support_texts) >= 4

    def test_all_probes_have_unique_names(self):
        names = [p.name for p in ALL_PHILOSOPHICAL_PROBES]
        # Names should be mostly unique (some may overlap like "Necessity")
        unique_names = set(names)
        # At least 25 unique names out of 30
        assert len(unique_names) >= 25

    def test_level_range(self):
        for probe in ALL_PHILOSOPHICAL_PROBES:
            assert 1 <= probe.level <= 6

    def test_all_ids_are_lowercase_underscore(self):
        for probe in ALL_PHILOSOPHICAL_PROBES:
            assert probe.id == probe.id.lower()
            assert " " not in probe.id
