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

"""Tests for MoralAtlas.

Tests the "Latent Ethicist" hypothesis: models encode moral reasoning as a
coherent geometric manifold with Valence, Agency, and Scope axes.

Based on Moral Foundations Theory (Haidt, 2012):
- Care/Harm, Fairness/Cheating, Loyalty/Betrayal
- Authority/Subversion, Sanctity/Degradation, Liberty/Oppression
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain.agents.moral_atlas import (
    MoralFoundation,
    MoralAxis,
    MoralConcept,
    MoralConceptInventory,
    CARE_HARM_PROBES,
    FAIRNESS_CHEATING_PROBES,
    LOYALTY_BETRAYAL_PROBES,
    AUTHORITY_SUBVERSION_PROBES,
    SANCTITY_DEGRADATION_PROBES,
    LIBERTY_OPPRESSION_PROBES,
    ALL_MORAL_PROBES,
)


class TestMoralConceptInventory:
    """Tests for MoralConceptInventory."""

    def test_total_concept_count(self) -> None:
        """Should have 30 total moral concepts (6 foundations x 5 probes)."""
        all_concepts = MoralConceptInventory.all_concepts()
        assert len(all_concepts) == 30

    def test_count_method(self) -> None:
        """Count method should return 30."""
        assert MoralConceptInventory.count() == 30

    def test_care_harm_probes_count(self) -> None:
        """Should have 5 care/harm probes (cruelty→compassion)."""
        care = MoralConceptInventory.care_harm_probes()
        assert len(care) == 5

    def test_fairness_probes_count(self) -> None:
        """Should have 5 fairness probes (exploitation→justice)."""
        fairness = MoralConceptInventory.fairness_probes()
        assert len(fairness) == 5

    def test_all_concept_ids_unique(self) -> None:
        """All moral concept IDs should be unique."""
        all_concepts = MoralConceptInventory.all_concepts()
        ids = [c.id for c in all_concepts]
        assert len(ids) == len(set(ids))

    def test_count_by_foundation(self) -> None:
        """Each foundation should have 5 probes."""
        counts = MoralConceptInventory.count_by_foundation()
        for foundation, count in counts.items():
            assert count == 5, f"{foundation.value} has {count} probes, expected 5"

    def test_count_by_axis(self) -> None:
        """Probes should be distributed across 3 axes."""
        counts = MoralConceptInventory.count_by_axis()
        # Valence: care_harm + fairness_cheating = 10
        # Agency: loyalty_betrayal + authority_subversion = 10
        # Scope: sanctity_degradation + liberty_oppression = 10
        assert counts[MoralAxis.VALENCE] == 10
        assert counts[MoralAxis.AGENCY] == 10
        assert counts[MoralAxis.SCOPE] == 10


class TestMoralFoundations:
    """Tests for Moral Foundations Theory structure."""

    def test_six_moral_foundations(self) -> None:
        """Should have exactly 6 moral foundations."""
        foundations = list(MoralFoundation)
        assert len(foundations) == 6

    def test_foundation_names(self) -> None:
        """Foundation names should match Haidt's theory."""
        expected = {
            "care_harm", "fairness_cheating", "loyalty_betrayal",
            "authority_subversion", "sanctity_degradation", "liberty_oppression"
        }
        actual = {f.value for f in MoralFoundation}
        assert actual == expected

    def test_care_harm_ordering(self) -> None:
        """Care/harm should be ordered: cruelty < neglect < indifference < kindness < compassion."""
        probes = list(CARE_HARM_PROBES)
        levels = [p.level for p in probes]
        assert levels == [1, 2, 3, 4, 5]
        ids = [p.id for p in probes]
        assert ids == ["cruelty", "neglect", "indifference", "kindness", "compassion"]

    def test_fairness_cheating_ordering(self) -> None:
        """Fairness/cheating should be ordered: exploitation → justice."""
        probes = list(FAIRNESS_CHEATING_PROBES)
        levels = [p.level for p in probes]
        assert levels == [1, 2, 3, 4, 5]
        ids = [p.id for p in probes]
        assert ids == ["exploitation", "cheating", "impartiality", "fairness", "justice"]

    def test_loyalty_betrayal_ordering(self) -> None:
        """Loyalty/betrayal should be ordered: betrayal → devotion."""
        probes = list(LOYALTY_BETRAYAL_PROBES)
        levels = [p.level for p in probes]
        assert levels == [1, 2, 3, 4, 5]
        ids = [p.id for p in probes]
        assert ids == ["betrayal", "treachery", "neutrality", "loyalty", "devotion"]

    def test_authority_subversion_ordering(self) -> None:
        """Authority/subversion should be ordered: rebellion → obedience."""
        probes = list(AUTHORITY_SUBVERSION_PROBES)
        levels = [p.level for p in probes]
        assert levels == [1, 2, 3, 4, 5]
        ids = [p.id for p in probes]
        assert ids == ["rebellion", "disobedience", "autonomy", "respect", "obedience"]

    def test_sanctity_degradation_ordering(self) -> None:
        """Sanctity/degradation should be ordered: defilement → sanctity."""
        probes = list(SANCTITY_DEGRADATION_PROBES)
        levels = [p.level for p in probes]
        assert levels == [1, 2, 3, 4, 5]
        ids = [p.id for p in probes]
        assert ids == ["defilement", "degradation", "mundane", "purity", "sanctity"]

    def test_liberty_oppression_ordering(self) -> None:
        """Liberty/oppression should be ordered: tyranny → liberation."""
        probes = list(LIBERTY_OPPRESSION_PROBES)
        levels = [p.level for p in probes]
        assert levels == [1, 2, 3, 4, 5]
        ids = [p.id for p in probes]
        assert ids == ["tyranny", "oppression", "constraint", "freedom", "liberation"]


class TestMoralAxisStructure:
    """Tests for moral axis structure."""

    def test_three_moral_axes(self) -> None:
        """Should have exactly 3 moral axes."""
        axes = list(MoralAxis)
        assert len(axes) == 3

    def test_axis_names(self) -> None:
        """Axis names should be valence, agency, and scope."""
        expected = {"valence", "agency", "scope"}
        actual = {a.value for a in MoralAxis}
        assert actual == expected

    def test_valence_axis_probes(self) -> None:
        """Valence axis should have care/harm and fairness/cheating."""
        valence = MoralConceptInventory.valence_probes()
        assert len(valence) == 10
        foundations = {p.foundation for p in valence}
        assert MoralFoundation.CARE_HARM in foundations
        assert MoralFoundation.FAIRNESS_CHEATING in foundations

    def test_agency_axis_probes(self) -> None:
        """Agency axis should have loyalty/betrayal and authority/subversion."""
        agency = MoralConceptInventory.agency_probes()
        assert len(agency) == 10
        foundations = {p.foundation for p in agency}
        assert MoralFoundation.LOYALTY_BETRAYAL in foundations
        assert MoralFoundation.AUTHORITY_SUBVERSION in foundations

    def test_scope_axis_probes(self) -> None:
        """Scope axis should have sanctity/degradation and liberty/oppression."""
        scope = MoralConceptInventory.scope_probes()
        assert len(scope) == 10
        foundations = {p.foundation for p in scope}
        assert MoralFoundation.SANCTITY_DEGRADATION in foundations
        assert MoralFoundation.LIBERTY_OPPRESSION in foundations


class TestMoralConceptProperties:
    """Tests for individual moral concept properties."""

    def test_all_concepts_have_support_texts(self) -> None:
        """All concepts should have at least 2 support texts."""
        for concept in ALL_MORAL_PROBES:
            assert len(concept.support_texts) >= 2, \
                f"{concept.id} has only {len(concept.support_texts)} support texts"

    def test_all_concepts_have_descriptions(self) -> None:
        """All concepts should have non-empty descriptions."""
        for concept in ALL_MORAL_PROBES:
            assert concept.description, f"{concept.id} has empty description"
            assert len(concept.description) > 10, \
                f"{concept.id} has very short description"

    def test_cross_domain_weights_in_range(self) -> None:
        """Cross-domain weights should be between 0.5 and 2.0."""
        for concept in ALL_MORAL_PROBES:
            assert 0.5 <= concept.cross_domain_weight <= 2.0, \
                f"{concept.id} has weight {concept.cross_domain_weight} out of range"

    def test_virtue_vice_extremes_have_higher_weights(self) -> None:
        """Virtue (level 5) and vice (level 1) should have higher weights."""
        for concept in ALL_MORAL_PROBES:
            if concept.level in (1, 5):  # Extremes (virtue or vice)
                assert concept.cross_domain_weight >= 1.2, \
                    f"Extreme {concept.id} should have weight >= 1.2"

    def test_canonical_name_property(self) -> None:
        """Canonical name should equal name."""
        for concept in ALL_MORAL_PROBES:
            assert concept.canonical_name == concept.name


class TestVirtueViceGradient:
    """Tests for virtue-vice gradient (monotonic valence ordering)."""

    def test_care_harm_is_monotonic(self) -> None:
        """Care/harm should be monotonically ordered cruelty→compassion."""
        probes = list(CARE_HARM_PROBES)
        for i in range(len(probes) - 1):
            assert probes[i].level < probes[i + 1].level, \
                f"Non-monotonic at {probes[i].id} -> {probes[i + 1].id}"

    def test_cruelty_is_vice_compassion_is_virtue(self) -> None:
        """Cruelty should be level 1 (vice), compassion level 5 (virtue)."""
        cruelty = next(p for p in CARE_HARM_PROBES if p.id == "cruelty")
        compassion = next(p for p in CARE_HARM_PROBES if p.id == "compassion")
        assert cruelty.level == 1
        assert compassion.level == 5

    def test_exploitation_is_vice_justice_is_virtue(self) -> None:
        """Exploitation should be level 1 (vice), justice level 5 (virtue)."""
        exploitation = next(p for p in FAIRNESS_CHEATING_PROBES if p.id == "exploitation")
        justice = next(p for p in FAIRNESS_CHEATING_PROBES if p.id == "justice")
        assert exploitation.level == 1
        assert justice.level == 5

    def test_betrayal_is_vice_devotion_is_virtue(self) -> None:
        """Betrayal should be level 1 (vice), devotion level 5 (virtue)."""
        betrayal = next(p for p in LOYALTY_BETRAYAL_PROBES if p.id == "betrayal")
        devotion = next(p for p in LOYALTY_BETRAYAL_PROBES if p.id == "devotion")
        assert betrayal.level == 1
        assert devotion.level == 5


class TestMoralByFilterMethods:
    """Tests for filtering methods."""

    def test_by_foundation_care_harm(self) -> None:
        """Filter by CARE_HARM foundation should return 5 probes."""
        care = MoralConceptInventory.by_foundation(MoralFoundation.CARE_HARM)
        assert len(care) == 5
        for c in care:
            assert c.foundation == MoralFoundation.CARE_HARM

    def test_by_axis_valence(self) -> None:
        """Filter by VALENCE axis should return 10 probes."""
        valence = MoralConceptInventory.by_axis(MoralAxis.VALENCE)
        assert len(valence) == 10
        for c in valence:
            assert c.axis == MoralAxis.VALENCE

    def test_valence_probes_method(self) -> None:
        """valence_probes() should return same as by_axis(VALENCE)."""
        direct_call = MoralConceptInventory.valence_probes()
        by_axis = MoralConceptInventory.by_axis(MoralAxis.VALENCE)
        assert len(direct_call) == len(by_axis)


class TestMoralHypotheses:
    """Tests aligned with stated moral atlas hypotheses."""

    def test_h1_baseline_probe_count(self) -> None:
        """H1: Models should have 30+ probes for statistical power."""
        assert MoralConceptInventory.count() >= 30

    def test_h2_axis_independence(self) -> None:
        """H2: Three axes should be distinct (for orthogonality testing)."""
        axes = list(MoralAxis)
        assert len(axes) == 3
        assert len(set(axes)) == 3  # All unique

    def test_h3_valence_gradient_structure(self) -> None:
        """H3: Valence probes should have 5 levels for monotonicity testing."""
        for probe in ALL_MORAL_PROBES:
            assert 1 <= probe.level <= 5, \
                f"{probe.id} has invalid level {probe.level}"

    def test_h4_foundation_clusters(self) -> None:
        """H4: Each foundation should be a distinct cluster."""
        # All foundations should have exactly 5 probes
        counts = MoralConceptInventory.count_by_foundation()
        for foundation in MoralFoundation:
            assert counts[foundation] == 5

    def test_h5_reproducibility_structure(self) -> None:
        """H5: Probes should be deterministic (frozen dataclasses)."""
        # Test that probes are hashable (frozen)
        probe_set = set(ALL_MORAL_PROBES)
        assert len(probe_set) == 30
