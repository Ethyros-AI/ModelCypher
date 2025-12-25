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

"""Tests for TemporalAtlas.

Tests the "Latent Chronologist" hypothesis: models encode time as a coherent
geometric manifold with Direction, Duration, and Causality axes.
"""

from __future__ import annotations

from modelcypher.core.domain.agents.temporal_atlas import (
    ALL_TEMPORAL_PROBES,
    CAUSALITY_PROBES,
    DURATION_PROBES,
    LIFECYCLE_PROBES,
    SEQUENCE_PROBES,
    TENSE_PROBES,
    TemporalAxis,
    TemporalCategory,
    TemporalConceptInventory,
)


class TestTemporalConceptInventory:
    """Tests for TemporalConceptInventory."""

    def test_total_concept_count(self) -> None:
        """Should have 25 total temporal concepts."""
        all_concepts = TemporalConceptInventory.all_concepts()
        assert len(all_concepts) == 25

    def test_count_method(self) -> None:
        """Count method should return 25."""
        assert TemporalConceptInventory.count() == 25

    def test_tense_probes_count(self) -> None:
        """Should have 5 tense probes."""
        tense = TemporalConceptInventory.tense_probes()
        assert len(tense) == 5

    def test_duration_probes_count(self) -> None:
        """Should have 5 duration probes (moment→century)."""
        duration = TemporalConceptInventory.duration_probes()
        assert len(duration) == 5

    def test_causality_probes_count(self) -> None:
        """Should have 5 causality probes (because→therefore)."""
        causality = TemporalConceptInventory.causality_probes()
        assert len(causality) == 5

    def test_lifecycle_probes_count(self) -> None:
        """Should have 5 lifecycle probes (birth→death)."""
        lifecycle = TemporalConceptInventory.lifecycle_probes()
        assert len(lifecycle) == 5

    def test_sequence_probes_count(self) -> None:
        """Should have 5 sequence probes (beginning→ending)."""
        sequence = TemporalConceptInventory.sequence_probes()
        assert len(sequence) == 5

    def test_all_concept_ids_unique(self) -> None:
        """All temporal concept IDs should be unique."""
        all_concepts = TemporalConceptInventory.all_concepts()
        ids = [c.id for c in all_concepts]
        assert len(ids) == len(set(ids))

    def test_count_by_category(self) -> None:
        """Each category should have 5 probes."""
        counts = TemporalConceptInventory.count_by_category()
        for category, count in counts.items():
            assert count == 5, f"{category.value} has {count} probes, expected 5"

    def test_count_by_axis(self) -> None:
        """Direction axis should have most probes (tense + lifecycle + sequence)."""
        counts = TemporalConceptInventory.count_by_axis()
        assert counts[TemporalAxis.DIRECTION] == 15  # tense + lifecycle + sequence
        assert counts[TemporalAxis.DURATION] == 5
        assert counts[TemporalAxis.CAUSALITY] == 5


class TestTemporalAxisStructure:
    """Tests for temporal axis ordering and structure."""

    def test_tense_level_ordering(self) -> None:
        """Tense probes should be ordered: past < yesterday < today < tomorrow < future."""
        tense = list(TENSE_PROBES)
        levels = [p.level for p in tense]
        assert levels == [1, 2, 3, 4, 5], f"Tense levels out of order: {levels}"

    def test_duration_level_ordering(self) -> None:
        """Duration probes should be ordered: moment < hour < day < year < century."""
        duration = list(DURATION_PROBES)
        levels = [p.level for p in duration]
        assert levels == [1, 2, 3, 4, 5], f"Duration levels out of order: {levels}"

    def test_causality_level_ordering(self) -> None:
        """Causality probes should be ordered: because < causes < leads_to < therefore < results_in."""
        causality = list(CAUSALITY_PROBES)
        levels = [p.level for p in causality]
        assert levels == [1, 2, 3, 4, 5], f"Causality levels out of order: {levels}"

    def test_lifecycle_level_ordering(self) -> None:
        """Lifecycle probes should be ordered: birth < childhood < adulthood < elderly < death."""
        lifecycle = list(LIFECYCLE_PROBES)
        levels = [p.level for p in lifecycle]
        assert levels == [1, 2, 3, 4, 5], f"Lifecycle levels out of order: {levels}"

    def test_sequence_level_ordering(self) -> None:
        """Sequence probes should be ordered: beginning < first < middle < last < ending."""
        sequence = list(SEQUENCE_PROBES)
        levels = [p.level for p in sequence]
        assert levels == [1, 2, 3, 4, 5], f"Sequence levels out of order: {levels}"

    def test_past_is_level_1(self) -> None:
        """Past should be at level 1 (lowest on direction axis)."""
        past = next(p for p in TENSE_PROBES if p.id == "past")
        assert past.level == 1
        assert past.axis == TemporalAxis.DIRECTION

    def test_future_is_level_5(self) -> None:
        """Future should be at level 5 (highest on direction axis)."""
        future = next(p for p in TENSE_PROBES if p.id == "future")
        assert future.level == 5
        assert future.axis == TemporalAxis.DIRECTION

    def test_because_precedes_therefore(self) -> None:
        """Because (cause) should precede therefore (effect) on causality axis."""
        because = next(p for p in CAUSALITY_PROBES if p.id == "because")
        therefore = next(p for p in CAUSALITY_PROBES if p.id == "therefore")
        assert because.level < therefore.level, "Cause should precede effect"


class TestTemporalConceptProperties:
    """Tests for individual temporal concept properties."""

    def test_all_concepts_have_support_texts(self) -> None:
        """All concepts should have at least 2 support texts."""
        for concept in ALL_TEMPORAL_PROBES:
            assert len(concept.support_texts) >= 2, (
                f"{concept.id} has only {len(concept.support_texts)} support texts"
            )

    def test_all_concepts_have_descriptions(self) -> None:
        """All concepts should have non-empty descriptions."""
        for concept in ALL_TEMPORAL_PROBES:
            assert concept.description, f"{concept.id} has empty description"
            assert len(concept.description) > 10, f"{concept.id} has very short description"

    def test_cross_domain_weights_in_range(self) -> None:
        """Cross-domain weights should be between 0.5 and 2.0."""
        for concept in ALL_TEMPORAL_PROBES:
            assert 0.5 <= concept.cross_domain_weight <= 2.0, (
                f"{concept.id} has weight {concept.cross_domain_weight} out of range"
            )

    def test_endpoint_concepts_have_higher_weights(self) -> None:
        """Endpoint concepts (past, future, birth, death) should have higher weights."""
        endpoints = ["past", "future", "birth", "death", "beginning", "ending"]
        for concept in ALL_TEMPORAL_PROBES:
            if concept.id in endpoints:
                assert concept.cross_domain_weight >= 1.2, (
                    f"Endpoint {concept.id} should have weight >= 1.2"
                )

    def test_canonical_name_property(self) -> None:
        """Canonical name should equal name."""
        for concept in ALL_TEMPORAL_PROBES:
            assert concept.canonical_name == concept.name


class TestTemporalByFilterMethods:
    """Tests for filtering methods."""

    def test_by_category_tense(self) -> None:
        """Filter by TENSE category should return 5 tense probes."""
        tense = TemporalConceptInventory.by_category(TemporalCategory.TENSE)
        assert len(tense) == 5
        for c in tense:
            assert c.category == TemporalCategory.TENSE

    def test_by_axis_direction(self) -> None:
        """Filter by DIRECTION axis should return 15 probes."""
        direction = TemporalConceptInventory.by_axis(TemporalAxis.DIRECTION)
        assert len(direction) == 15
        for c in direction:
            assert c.axis == TemporalAxis.DIRECTION

    def test_direction_probes_method(self) -> None:
        """direction_probes() should return same as by_axis(DIRECTION)."""
        direct_call = TemporalConceptInventory.direction_probes()
        by_axis = TemporalConceptInventory.by_axis(TemporalAxis.DIRECTION)
        assert len(direct_call) == len(by_axis)


class TestArrowOfTime:
    """Tests for Arrow of Time hypothesis (monotonic gradient past→future)."""

    def test_tense_is_monotonic(self) -> None:
        """Tense probes should have monotonically increasing levels."""
        tense = list(TENSE_PROBES)
        for i in range(len(tense) - 1):
            assert tense[i].level < tense[i + 1].level, (
                f"Non-monotonic at {tense[i].id} -> {tense[i + 1].id}"
            )

    def test_lifecycle_is_monotonic(self) -> None:
        """Lifecycle probes should be monotonically ordered birth→death."""
        lifecycle = list(LIFECYCLE_PROBES)
        ids = [p.id for p in lifecycle]
        assert ids == ["birth", "childhood", "adulthood", "elderly", "death"]

    def test_causality_preserves_direction(self) -> None:
        """Causality axis should preserve temporal direction (cause→effect)."""
        causality = list(CAUSALITY_PROBES)
        # because (1) should precede results_in (5)
        because_level = next(p.level for p in causality if p.id == "because")
        results_level = next(p.level for p in causality if p.id == "results_in")
        assert because_level < results_level
