# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import modelcypher.core.domain.atlas.moral_concepts as moral_concepts


# ---------------------------------------------------------------------------
# MoralAxis enum
# ---------------------------------------------------------------------------


class TestMoralAxis:
    """Tests for the MoralAxis enum."""

    def test_valence_value(self) -> None:
        assert moral_concepts.MoralAxis.VALENCE == "valence"
        assert moral_concepts.MoralAxis.VALENCE.value == "valence"

    def test_agency_value(self) -> None:
        assert moral_concepts.MoralAxis.AGENCY == "agency"
        assert moral_concepts.MoralAxis.AGENCY.value == "agency"

    def test_scope_value(self) -> None:
        assert moral_concepts.MoralAxis.SCOPE == "scope"
        assert moral_concepts.MoralAxis.SCOPE.value == "scope"

    def test_str_subclass(self) -> None:
        assert isinstance(moral_concepts.MoralAxis.VALENCE, str)
        assert isinstance(moral_concepts.MoralAxis.AGENCY, str)
        assert isinstance(moral_concepts.MoralAxis.SCOPE, str)

    def test_all_members_count(self) -> None:
        assert len(moral_concepts.MoralAxis) == 3

    def test_from_value(self) -> None:
        assert moral_concepts.MoralAxis("valence") is moral_concepts.MoralAxis.VALENCE
        assert moral_concepts.MoralAxis("agency") is moral_concepts.MoralAxis.AGENCY
        assert moral_concepts.MoralAxis("scope") is moral_concepts.MoralAxis.SCOPE


# ---------------------------------------------------------------------------
# MoralFoundation enum
# ---------------------------------------------------------------------------


class TestMoralFoundation:
    """Tests for the MoralFoundation enum."""

    def test_care_harm(self) -> None:
        assert moral_concepts.MoralFoundation.CARE_HARM == "care_harm"
        assert moral_concepts.MoralFoundation.CARE_HARM.value == "care_harm"

    def test_fairness_cheating(self) -> None:
        assert moral_concepts.MoralFoundation.FAIRNESS_CHEATING == "fairness_cheating"

    def test_sanctity_degradation(self) -> None:
        assert moral_concepts.MoralFoundation.SANCTITY_DEGRADATION == "sanctity_degradation"

    def test_liberty_oppression(self) -> None:
        assert moral_concepts.MoralFoundation.LIBERTY_OPPRESSION == "liberty_oppression"

    def test_all_4_values(self) -> None:
        assert len(moral_concepts.MoralFoundation) == 4

    def test_str_subclass(self) -> None:
        for f in moral_concepts.MoralFoundation:
            assert isinstance(f, str)


# ---------------------------------------------------------------------------
# MoralConcept frozen dataclass
# ---------------------------------------------------------------------------


class TestMoralConcept:
    """Tests for the MoralConcept frozen dataclass."""

    def test_instantiation(self) -> None:
        concept = moral_concepts.MoralConcept(
            id="test_concept",
            name="Test",
            axis=moral_concepts.MoralAxis.VALENCE,
            foundation=moral_concepts.MoralFoundation.CARE_HARM,
            level=0,
            description="A test concept.",
            support_texts=("text one", "text two", "text three"),
        )
        assert concept.id == "test_concept"
        assert concept.name == "Test"
        assert concept.axis is moral_concepts.MoralAxis.VALENCE
        assert concept.foundation is moral_concepts.MoralFoundation.CARE_HARM
        assert concept.level == 0
        assert concept.description == "A test concept."
        assert len(concept.support_texts) == 3

    def test_field_access(self) -> None:
        concept = moral_concepts.MoralConcept(
            id="cruelty",
            name="Cruelty",
            axis=moral_concepts.MoralAxis.VALENCE,
            foundation=moral_concepts.MoralFoundation.CARE_HARM,
            level=-2,
            description="Deliberate infliction of suffering.",
            support_texts=(
                "Cruelty is the willful causing of pain.",
                "To be cruel is to delight in suffering.",
                "Cruelty shows disregard for others' wellbeing.",
            ),
        )
        assert concept.axis == "valence"
        assert concept.foundation == "care_harm"
        assert concept.level == -2

    def test_frozen_enforcement(self) -> None:
        concept = moral_concepts.MoralConcept(
            id="test",
            name="Test",
            axis=moral_concepts.MoralAxis.VALENCE,
            foundation=moral_concepts.MoralFoundation.CARE_HARM,
            level=0,
            description="Test",
            support_texts=("a", "b", "c"),
        )
        with pytest.raises(FrozenInstanceError):
            concept.id = "changed"  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            concept.name = "Changed"  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            concept.level = 5  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            concept.axis = moral_concepts.MoralAxis.AGENCY  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            concept.support_texts = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MoralConceptInventory
# ---------------------------------------------------------------------------


class TestMoralConceptInventory:
    """Tests for the MoralConceptInventory class methods."""

    def setup_method(self) -> None:
        moral_concepts.MoralConceptInventory.clear_cache()

    def teardown_method(self) -> None:
        moral_concepts.MoralConceptInventory.clear_cache()

    def test_all_concepts_returns_20(self) -> None:
        concepts = moral_concepts.MoralConceptInventory.all_concepts()
        assert len(concepts) == 20

    def test_all_concepts_returns_moral_concept_instances(self) -> None:
        concepts = moral_concepts.MoralConceptInventory.all_concepts()
        for c in concepts:
            assert isinstance(c, moral_concepts.MoralConcept)

    def test_by_foundation_care_harm(self) -> None:
        care_harm = moral_concepts.MoralConceptInventory.by_foundation(
            moral_concepts.MoralFoundation.CARE_HARM
        )
        assert len(care_harm) == 5
        expected_ids = {"cruelty", "neglect", "indifference", "kindness", "compassion"}
        actual_ids = {c.id for c in care_harm}
        assert actual_ids == expected_ids

    def test_by_foundation_fairness_cheating(self) -> None:
        result = moral_concepts.MoralConceptInventory.by_foundation(
            moral_concepts.MoralFoundation.FAIRNESS_CHEATING
        )
        assert len(result) == 5

    def test_by_foundation_sanctity_degradation(self) -> None:
        result = moral_concepts.MoralConceptInventory.by_foundation(
            moral_concepts.MoralFoundation.SANCTITY_DEGRADATION
        )
        assert len(result) == 5

    def test_by_foundation_liberty_oppression(self) -> None:
        result = moral_concepts.MoralConceptInventory.by_foundation(
            moral_concepts.MoralFoundation.LIBERTY_OPPRESSION
        )
        assert len(result) == 5

    def test_by_axis_valence(self) -> None:
        valence = moral_concepts.MoralConceptInventory.by_axis(
            moral_concepts.MoralAxis.VALENCE
        )
        # CARE_HARM (5) + FAIRNESS_CHEATING (5) + SANCTITY_DEGRADATION (5) = 15
        assert len(valence) == 15
        for c in valence:
            assert c.axis is moral_concepts.MoralAxis.VALENCE

    def test_by_axis_agency(self) -> None:
        agency = moral_concepts.MoralConceptInventory.by_axis(
            moral_concepts.MoralAxis.AGENCY
        )
        # LIBERTY_OPPRESSION (5)
        assert len(agency) == 5
        for c in agency:
            assert c.axis is moral_concepts.MoralAxis.AGENCY
            assert c.foundation is moral_concepts.MoralFoundation.LIBERTY_OPPRESSION

    def test_by_axis_scope(self) -> None:
        scope = moral_concepts.MoralConceptInventory.by_axis(
            moral_concepts.MoralAxis.SCOPE
        )
        # No concepts are on the SCOPE axis in the current inventory
        assert len(scope) == 0

    def test_clear_cache(self) -> None:
        concepts_first = moral_concepts.MoralConceptInventory.all_concepts()
        moral_concepts.MoralConceptInventory.clear_cache()
        concepts_second = moral_concepts.MoralConceptInventory.all_concepts()
        assert len(concepts_first) == len(concepts_second)
        for c1, c2 in zip(concepts_first, concepts_second):
            assert c1.id == c2.id

    def test_clear_cache_resets_internal_state(self) -> None:
        moral_concepts.MoralConceptInventory.all_concepts()
        assert moral_concepts.MoralConceptInventory._cached is not None
        moral_concepts.MoralConceptInventory.clear_cache()
        assert moral_concepts.MoralConceptInventory._cached is None

    def test_all_concepts_returns_fresh_list(self) -> None:
        """Verify all_concepts returns a copy, not the cached reference."""
        a = moral_concepts.MoralConceptInventory.all_concepts()
        b = moral_concepts.MoralConceptInventory.all_concepts()
        assert a is not b
        assert a == b


# ---------------------------------------------------------------------------
# Concept field validation
# ---------------------------------------------------------------------------


class TestMoralConceptFieldValidation:
    """Validate all loaded concepts have proper field values."""

    def setup_method(self) -> None:
        moral_concepts.MoralConceptInventory.clear_cache()

    def teardown_method(self) -> None:
        moral_concepts.MoralConceptInventory.clear_cache()

    def test_all_ids_non_empty(self) -> None:
        for c in moral_concepts.MoralConceptInventory.all_concepts():
            assert c.id, f"Concept has empty id: {c}"
            assert isinstance(c.id, str)

    def test_all_names_non_empty(self) -> None:
        for c in moral_concepts.MoralConceptInventory.all_concepts():
            assert c.name, f"Concept has empty name: {c}"
            assert isinstance(c.name, str)

    def test_all_descriptions_non_empty(self) -> None:
        for c in moral_concepts.MoralConceptInventory.all_concepts():
            assert c.description, f"Concept has empty description: {c}"
            assert isinstance(c.description, str)

    def test_all_support_texts_have_3_strings(self) -> None:
        for c in moral_concepts.MoralConceptInventory.all_concepts():
            assert isinstance(c.support_texts, tuple), (
                f"support_texts not tuple for {c.id}"
            )
            assert len(c.support_texts) == 3, (
                f"Expected 3 support_texts for {c.id}, got {len(c.support_texts)}"
            )
            for t in c.support_texts:
                assert isinstance(t, str)
                assert t, f"Empty support_text in {c.id}"

    def test_level_values_span_minus2_to_plus2_per_foundation(self) -> None:
        """Each foundation should have levels spanning -2 to +2."""
        for foundation in moral_concepts.MoralFoundation:
            concepts = moral_concepts.MoralConceptInventory.by_foundation(foundation)
            levels = sorted(c.level for c in concepts)
            assert levels == [-2, -1, 0, 1, 2], (
                f"Foundation {foundation.value} levels: {levels}, "
                f"expected [-2, -1, 0, 1, 2]"
            )

    def test_all_ids_unique(self) -> None:
        concepts = moral_concepts.MoralConceptInventory.all_concepts()
        ids = [c.id for c in concepts]
        assert len(ids) == len(set(ids)), "Duplicate concept ids found"
