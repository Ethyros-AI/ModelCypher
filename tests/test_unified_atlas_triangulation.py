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

"""Tests for UnifiedAtlas triangulation scoring.

Tests the cross-atlas triangulation math that boosts confidence when
concepts are detected across multiple atlas sources and domains.

Mathematical invariants tested:
- Source multiplier: 1.0 + (source_count - 1) * 0.1
- Domain multiplier: 1.0 + (domain_count - 1) * 0.15
- Combined multiplier: geometric mean = sqrt(source * domain)
- Detection across more sources/domains increases score
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings, strategies as st

from modelcypher.core.domain.agents.unified_atlas import (
    AtlasSource,
    AtlasDomain,
    AtlasProbe,
    UnifiedAtlasInventory,
    MultiAtlasTriangulationScore,
    MultiAtlasTriangulationScorer,
    get_probe_ids,
    DEFAULT_ATLAS_SOURCES,
    ALL_ATLAS_SOURCES,
    MATHEMATICAL_DOMAINS,
    LINGUISTIC_DOMAINS,
)


def make_probe(
    source: AtlasSource,
    domain: AtlasDomain,
    id: str = "test",
) -> AtlasProbe:
    """Create a test probe with specified source and domain."""
    return AtlasProbe(
        id=id,
        source=source,
        domain=domain,
        name=f"Test {id}",
        description="Test probe",
        cross_domain_weight=1.0,
        category_name="test",
    )


class TestUnifiedAtlasInventory:
    """Tests for UnifiedAtlasInventory probe loading."""

    def test_total_probe_count_343(self) -> None:
        """Should have exactly 343 probes across all sources."""
        probes = UnifiedAtlasInventory.all_probes()
        assert len(probes) == 343

    def test_probes_by_source_returns_correct_counts(self) -> None:
        """Each source should have expected probe count."""
        expected_counts = {
            AtlasSource.SEQUENCE_INVARIANT: 68,
            AtlasSource.SEMANTIC_PRIME: 65,
            AtlasSource.COMPUTATIONAL_GATE: 76,
            AtlasSource.EMOTION_CONCEPT: 32,
            AtlasSource.TEMPORAL_CONCEPT: 25,
            AtlasSource.SOCIAL_CONCEPT: 25,
            AtlasSource.MORAL_CONCEPT: 30,
        }

        for source, expected in expected_counts.items():
            probes = UnifiedAtlasInventory.probes_by_source({source})
            assert len(probes) == expected, f"{source.value} has {len(probes)}, expected {expected}"

    def test_all_probes_have_valid_source(self) -> None:
        """All probes should have a valid AtlasSource."""
        for probe in UnifiedAtlasInventory.all_probes():
            assert probe.source in AtlasSource

    def test_all_probes_have_valid_domain(self) -> None:
        """All probes should have a valid AtlasDomain."""
        for probe in UnifiedAtlasInventory.all_probes():
            assert probe.domain in AtlasDomain

    def test_probe_ids_are_unique(self) -> None:
        """All probe_ids should be unique."""
        probes = UnifiedAtlasInventory.all_probes()
        ids = [p.probe_id for p in probes]
        assert len(ids) == len(set(ids))


class TestSourceMultiplier:
    """Tests for source multiplier calculation."""

    def test_single_source_multiplier_is_one(self) -> None:
        """1 source detected should give multiplier of 1.0."""
        probe = make_probe(AtlasSource.SEQUENCE_INVARIANT, AtlasDomain.MATHEMATICAL)
        activations = {probe: 0.5}

        score = MultiAtlasTriangulationScorer.compute_score(activations)

        assert score.source_multiplier == 1.0
        assert len(score.sources_detected) == 1

    def test_two_sources_multiplier_is_1_1(self) -> None:
        """2 sources detected should give multiplier of 1.1."""
        probes = [
            make_probe(AtlasSource.SEQUENCE_INVARIANT, AtlasDomain.MATHEMATICAL, "a"),
            make_probe(AtlasSource.SEMANTIC_PRIME, AtlasDomain.LINGUISTIC, "b"),
        ]
        activations = {p: 0.5 for p in probes}

        score = MultiAtlasTriangulationScorer.compute_score(activations)

        assert score.source_multiplier == pytest.approx(1.1)
        assert len(score.sources_detected) == 2

    def test_source_multiplier_formula(self) -> None:
        """Source multiplier should follow: 1.0 + (count - 1) * 0.1"""
        # Test with 3 sources
        probes = [
            make_probe(AtlasSource.SEQUENCE_INVARIANT, AtlasDomain.MATHEMATICAL, "a"),
            make_probe(AtlasSource.SEMANTIC_PRIME, AtlasDomain.LINGUISTIC, "b"),
            make_probe(AtlasSource.COMPUTATIONAL_GATE, AtlasDomain.COMPUTATIONAL, "c"),
        ]
        activations = {p: 0.5 for p in probes}

        score = MultiAtlasTriangulationScorer.compute_score(activations)

        # 1.0 + (3 - 1) * 0.1 = 1.2
        expected = 1.0 + (3 - 1) * 0.1
        assert score.source_multiplier == pytest.approx(expected)

    def test_all_seven_sources_multiplier(self) -> None:
        """All 7 sources should give multiplier of 1.6."""
        probes = [
            make_probe(AtlasSource.SEQUENCE_INVARIANT, AtlasDomain.MATHEMATICAL, "a"),
            make_probe(AtlasSource.SEMANTIC_PRIME, AtlasDomain.LINGUISTIC, "b"),
            make_probe(AtlasSource.COMPUTATIONAL_GATE, AtlasDomain.COMPUTATIONAL, "c"),
            make_probe(AtlasSource.EMOTION_CONCEPT, AtlasDomain.AFFECTIVE, "d"),
            make_probe(AtlasSource.TEMPORAL_CONCEPT, AtlasDomain.TEMPORAL, "e"),
            make_probe(AtlasSource.SOCIAL_CONCEPT, AtlasDomain.RELATIONAL, "f"),
            make_probe(AtlasSource.MORAL_CONCEPT, AtlasDomain.MORAL, "g"),
        ]
        activations = {p: 0.5 for p in probes}

        score = MultiAtlasTriangulationScorer.compute_score(activations)

        # 1.0 + (7 - 1) * 0.1 = 1.6
        expected = 1.0 + (7 - 1) * 0.1
        assert score.source_multiplier == pytest.approx(expected)


class TestDomainMultiplier:
    """Tests for domain multiplier calculation."""

    def test_single_domain_multiplier_is_one(self) -> None:
        """1 domain detected should give multiplier of 1.0."""
        probe = make_probe(AtlasSource.SEQUENCE_INVARIANT, AtlasDomain.MATHEMATICAL)
        activations = {probe: 0.5}

        score = MultiAtlasTriangulationScorer.compute_score(activations)

        assert score.domain_multiplier == 1.0
        assert len(score.domains_detected) == 1

    def test_two_domains_multiplier_is_1_15(self) -> None:
        """2 domains detected should give multiplier of 1.15."""
        probes = [
            make_probe(AtlasSource.SEQUENCE_INVARIANT, AtlasDomain.MATHEMATICAL, "a"),
            make_probe(AtlasSource.SEMANTIC_PRIME, AtlasDomain.LINGUISTIC, "b"),
        ]
        activations = {p: 0.5 for p in probes}

        score = MultiAtlasTriangulationScorer.compute_score(activations)

        assert score.domain_multiplier == pytest.approx(1.15)
        assert len(score.domains_detected) == 2

    def test_domain_multiplier_formula(self) -> None:
        """Domain multiplier should follow: 1.0 + (count - 1) * 0.15"""
        # Test with 4 domains
        probes = [
            make_probe(AtlasSource.SEQUENCE_INVARIANT, AtlasDomain.MATHEMATICAL, "a"),
            make_probe(AtlasSource.SEMANTIC_PRIME, AtlasDomain.LINGUISTIC, "b"),
            make_probe(AtlasSource.COMPUTATIONAL_GATE, AtlasDomain.COMPUTATIONAL, "c"),
            make_probe(AtlasSource.EMOTION_CONCEPT, AtlasDomain.AFFECTIVE, "d"),
        ]
        activations = {p: 0.5 for p in probes}

        score = MultiAtlasTriangulationScorer.compute_score(activations)

        # 1.0 + (4 - 1) * 0.15 = 1.45
        expected = 1.0 + (4 - 1) * 0.15
        assert score.domain_multiplier == pytest.approx(expected)


class TestCombinedMultiplier:
    """Tests for combined geometric mean multiplier."""

    def test_combined_is_geometric_mean(self) -> None:
        """Combined multiplier should be sqrt(source * domain)."""
        probes = [
            make_probe(AtlasSource.SEQUENCE_INVARIANT, AtlasDomain.MATHEMATICAL, "a"),
            make_probe(AtlasSource.SEMANTIC_PRIME, AtlasDomain.LINGUISTIC, "b"),
        ]
        activations = {p: 0.5 for p in probes}

        score = MultiAtlasTriangulationScorer.compute_score(activations)

        expected = math.sqrt(score.source_multiplier * score.domain_multiplier)
        assert score.combined_multiplier == pytest.approx(expected)

    def test_combined_always_at_least_one(self) -> None:
        """Combined multiplier should be >= 1.0 when any probe detected."""
        probe = make_probe(AtlasSource.SEQUENCE_INVARIANT, AtlasDomain.MATHEMATICAL)
        activations = {probe: 0.5}

        score = MultiAtlasTriangulationScorer.compute_score(activations)

        assert score.combined_multiplier >= 1.0

    def test_combined_increases_with_more_sources(self) -> None:
        """More sources should increase combined multiplier."""
        # 1 source
        probes_1 = [make_probe(AtlasSource.SEQUENCE_INVARIANT, AtlasDomain.MATHEMATICAL)]
        score_1 = MultiAtlasTriangulationScorer.compute_score({p: 0.5 for p in probes_1})

        # 3 sources
        probes_3 = [
            make_probe(AtlasSource.SEQUENCE_INVARIANT, AtlasDomain.MATHEMATICAL, "a"),
            make_probe(AtlasSource.SEMANTIC_PRIME, AtlasDomain.LINGUISTIC, "b"),
            make_probe(AtlasSource.COMPUTATIONAL_GATE, AtlasDomain.COMPUTATIONAL, "c"),
        ]
        score_3 = MultiAtlasTriangulationScorer.compute_score({p: 0.5 for p in probes_3})

        assert score_3.combined_multiplier > score_1.combined_multiplier


class TestActivationThreshold:
    """Tests for activation threshold filtering."""

    def test_below_threshold_not_counted(self) -> None:
        """Activations below threshold should not be counted."""
        probes = [
            make_probe(AtlasSource.SEQUENCE_INVARIANT, AtlasDomain.MATHEMATICAL, "a"),
            make_probe(AtlasSource.SEMANTIC_PRIME, AtlasDomain.LINGUISTIC, "b"),
        ]
        # One above threshold, one below
        activations = {probes[0]: 0.5, probes[1]: 0.1}

        score = MultiAtlasTriangulationScorer.compute_score(activations, threshold=0.3)

        assert len(score.sources_detected) == 1
        assert AtlasSource.SEQUENCE_INVARIANT in score.sources_detected
        assert AtlasSource.SEMANTIC_PRIME not in score.sources_detected

    def test_exactly_at_threshold_not_counted(self) -> None:
        """Activation exactly at threshold should not be counted (> not >=)."""
        probe = make_probe(AtlasSource.SEQUENCE_INVARIANT, AtlasDomain.MATHEMATICAL)
        activations = {probe: 0.3}

        score = MultiAtlasTriangulationScorer.compute_score(activations, threshold=0.3)

        assert len(score.sources_detected) == 0

    def test_empty_activations(self) -> None:
        """Empty activations should return multiplier of 1.0."""
        score = MultiAtlasTriangulationScorer.compute_score({})

        assert score.source_multiplier == 1.0
        assert score.domain_multiplier == 1.0
        assert score.combined_multiplier == 1.0


class TestGetProbeIds:
    """Tests for get_probe_ids utility function."""

    def test_get_all_probe_ids(self) -> None:
        """Should return all 343 probe IDs."""
        ids = get_probe_ids(None)
        assert len(ids) == 343

    def test_get_probe_ids_for_single_source(self) -> None:
        """Should filter probe IDs by source."""
        ids = get_probe_ids({AtlasSource.SEQUENCE_INVARIANT})
        assert len(ids) == 68
        for id in ids:
            assert id.startswith("sequence_invariant:")

    def test_get_probe_ids_for_multiple_sources(self) -> None:
        """Should combine probes from multiple sources."""
        ids = get_probe_ids({
            AtlasSource.SEQUENCE_INVARIANT,
            AtlasSource.SEMANTIC_PRIME,
        })
        assert len(ids) == 68 + 65  # 133


class TestDomainSets:
    """Tests for domain convenience sets."""

    def test_mathematical_domains(self) -> None:
        """MATHEMATICAL_DOMAINS should contain expected domains."""
        assert AtlasDomain.MATHEMATICAL in MATHEMATICAL_DOMAINS
        assert AtlasDomain.LOGICAL in MATHEMATICAL_DOMAINS
        assert len(MATHEMATICAL_DOMAINS) == 2

    def test_linguistic_domains(self) -> None:
        """LINGUISTIC_DOMAINS should contain expected domains."""
        assert AtlasDomain.LINGUISTIC in LINGUISTIC_DOMAINS
        assert AtlasDomain.MENTAL in LINGUISTIC_DOMAINS
        assert len(LINGUISTIC_DOMAINS) == 2


class TestScoreDataclass:
    """Tests for MultiAtlasTriangulationScore dataclass."""

    def test_score_is_frozen(self) -> None:
        """Score should be immutable."""
        score = MultiAtlasTriangulationScore(
            layer_index=0,
            sources_detected={AtlasSource.SEQUENCE_INVARIANT},
            domains_detected={AtlasDomain.MATHEMATICAL},
            source_multiplier=1.0,
            domain_multiplier=1.0,
            combined_multiplier=1.0,
        )

        with pytest.raises(AttributeError):
            score.layer_index = 1  # type: ignore


class TestMathematicalInvariants:
    """Property-based tests for triangulation math invariants."""

    @given(
        n_sources=st.integers(min_value=1, max_value=7),
    )
    @settings(max_examples=20)
    def test_source_multiplier_monotonic(self, n_sources: int) -> None:
        """More sources should always give higher or equal multiplier."""
        # Create n_sources unique probes
        sources = list(AtlasSource)[:n_sources]
        probes = [
            make_probe(s, AtlasDomain.MATHEMATICAL, f"probe_{i}")
            for i, s in enumerate(sources)
        ]
        activations = {p: 0.5 for p in probes}

        score = MultiAtlasTriangulationScorer.compute_score(activations)

        expected = 1.0 + (n_sources - 1) * 0.1
        assert score.source_multiplier == pytest.approx(expected)

    @given(
        n_domains=st.integers(min_value=1, max_value=len(AtlasDomain)),
    )
    @settings(max_examples=20)
    def test_domain_multiplier_monotonic(self, n_domains: int) -> None:
        """More domains should always give higher or equal multiplier."""
        domains = list(AtlasDomain)[:n_domains]
        probes = [
            make_probe(AtlasSource.SEQUENCE_INVARIANT, d, f"probe_{i}")
            for i, d in enumerate(domains)
        ]
        activations = {p: 0.5 for p in probes}

        score = MultiAtlasTriangulationScorer.compute_score(activations)

        expected = 1.0 + (n_domains - 1) * 0.15
        assert score.domain_multiplier == pytest.approx(expected)

    def test_combined_multiplier_is_geometric_mean(self) -> None:
        """Combined should always equal sqrt(source * domain)."""
        probes = [
            make_probe(AtlasSource.SEQUENCE_INVARIANT, AtlasDomain.MATHEMATICAL, "a"),
            make_probe(AtlasSource.SEMANTIC_PRIME, AtlasDomain.LINGUISTIC, "b"),
            make_probe(AtlasSource.COMPUTATIONAL_GATE, AtlasDomain.STRUCTURAL, "c"),
            make_probe(AtlasSource.EMOTION_CONCEPT, AtlasDomain.AFFECTIVE, "d"),
        ]
        activations = {p: 0.5 for p in probes}

        score = MultiAtlasTriangulationScorer.compute_score(activations)

        expected = math.sqrt(score.source_multiplier * score.domain_multiplier)
        assert score.combined_multiplier == pytest.approx(expected, rel=1e-6)
