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

import modelcypher.core.domain.atlas.unified_atlas as unified_atlas
from modelcypher.core.domain.domains import AtlasDomain

# ---------------------------------------------------------------------------
# AtlasSource enum
# ---------------------------------------------------------------------------


class TestAtlasSource:
    """Tests for the AtlasSource enum."""

    _EXPECTED_MEMBERS = [
        "SEQUENCE_INVARIANT",
        "SEMANTIC_PRIME",
        "COMPUTATIONAL_GATE",
        "EMOTION_CONCEPT",
        "TEMPORAL_CONCEPT",
        "SPATIAL_CONCEPT",
        "SOCIAL_CONCEPT",
        "MORAL_CONCEPT",
        "COMPOSITIONAL",
        "PHILOSOPHICAL_CONCEPT",
        "CONCEPTUAL_GENEALOGY",
        "METAPHOR_INVARIANT",
        "CONCEPTUAL_METAPHOR",
        "SYNTAX_CONCEPT",
        "SAFETY_ETHICS",
        "PHYSICAL_EXISTENCE",
        "PERCEPTUAL",
        "NUMERIC",
        "COMMON_OBJECT",
        "ACTION_VERB",
        "DOMAIN_SPECIFIC",
        "ABSTRACT_RELATION",
        "PRONOUN_PERSPECTIVE",
        "PRIME_NUMBER",
    ]

    def test_has_24_members(self) -> None:
        assert len(unified_atlas.AtlasSource) == 24

    def test_all_values_accessible(self) -> None:
        for name in self._EXPECTED_MEMBERS:
            member = unified_atlas.AtlasSource[name]
            assert member is not None

    def test_str_subclass(self) -> None:
        for source in unified_atlas.AtlasSource:
            assert isinstance(source, str)

    def test_semantic_prime_value(self) -> None:
        assert unified_atlas.AtlasSource.SEMANTIC_PRIME == "semantic_prime"
        assert unified_atlas.AtlasSource.SEMANTIC_PRIME.value == "semantic_prime"

    def test_computational_gate_value(self) -> None:
        assert unified_atlas.AtlasSource.COMPUTATIONAL_GATE == "computational_gate"

    def test_from_value(self) -> None:
        src = unified_atlas.AtlasSource("semantic_prime")
        assert src is unified_atlas.AtlasSource.SEMANTIC_PRIME


# ---------------------------------------------------------------------------
# AtlasProbe frozen dataclass
# ---------------------------------------------------------------------------


class TestAtlasProbe:
    """Tests for the AtlasProbe frozen dataclass."""

    def _make_probe(self, **overrides) -> unified_atlas.AtlasProbe:
        defaults = dict(
            id="test_probe_1",
            source=unified_atlas.AtlasSource.SEMANTIC_PRIME,
            domain=AtlasDomain.LINGUISTIC,
            name="Test Probe",
            description="A test probe",
            cross_domain_weight=1.0,
            category_name="substantives",
        )
        defaults.update(overrides)
        return unified_atlas.AtlasProbe(**defaults)

    def test_instantiation(self) -> None:
        probe = self._make_probe()
        assert probe.id == "test_probe_1"
        assert probe.source is unified_atlas.AtlasSource.SEMANTIC_PRIME
        assert probe.domain is AtlasDomain.LINGUISTIC
        assert probe.name == "Test Probe"
        assert probe.description == "A test probe"
        assert probe.cross_domain_weight == 1.0
        assert probe.category_name == "substantives"
        assert probe.support_texts == ()
        assert probe.verification_depth is None

    def test_probe_id_property(self) -> None:
        probe = self._make_probe(
            id="my_probe",
            source=unified_atlas.AtlasSource.COMPUTATIONAL_GATE,
        )
        assert probe.probe_id == "computational_gate:my_probe"

    def test_probe_id_format(self) -> None:
        probe = self._make_probe(
            id="kindness",
            source=unified_atlas.AtlasSource.MORAL_CONCEPT,
        )
        assert probe.probe_id == "moral_concept:kindness"
        assert ":" in probe.probe_id

    def test_frozen_enforcement(self) -> None:
        probe = self._make_probe()
        with pytest.raises(FrozenInstanceError):
            probe.id = "changed"  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            probe.name = "Changed"  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            probe.source = unified_atlas.AtlasSource.EMOTION_CONCEPT  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            probe.cross_domain_weight = 0.5  # type: ignore[misc]

    def test_optional_support_texts(self) -> None:
        probe = self._make_probe(
            support_texts=("text1", "text2"),
        )
        assert probe.support_texts == ("text1", "text2")

    def test_optional_verification_depth(self) -> None:
        probe = self._make_probe(verification_depth=3)
        assert probe.verification_depth == 3


# ---------------------------------------------------------------------------
# UnifiedAtlasInventory
# ---------------------------------------------------------------------------


class TestUnifiedAtlasInventory:
    """Tests for the UnifiedAtlasInventory class methods."""

    def setup_method(self) -> None:
        unified_atlas.UnifiedAtlasInventory.clear_cache()

    def teardown_method(self) -> None:
        unified_atlas.UnifiedAtlasInventory.clear_cache()

    def test_all_probes_returns_non_empty_list(self) -> None:
        probes = unified_atlas.UnifiedAtlasInventory.all_probes()
        assert isinstance(probes, list)
        assert len(probes) > 0, "Expected at least one probe"

    def test_all_probes_returns_atlas_probe_instances(self) -> None:
        probes = unified_atlas.UnifiedAtlasInventory.all_probes()
        for p in probes:
            assert isinstance(p, unified_atlas.AtlasProbe)

    def test_all_probes_returns_fresh_list(self) -> None:
        a = unified_atlas.UnifiedAtlasInventory.all_probes()
        b = unified_atlas.UnifiedAtlasInventory.all_probes()
        assert a is not b
        assert len(a) == len(b)

    def test_probes_by_source(self) -> None:
        sources = {unified_atlas.AtlasSource.SEMANTIC_PRIME}
        result = unified_atlas.UnifiedAtlasInventory.probes_by_source(sources)
        assert isinstance(result, list)
        assert len(result) > 0, "Expected at least one SEMANTIC_PRIME probe"
        for p in result:
            assert p.source is unified_atlas.AtlasSource.SEMANTIC_PRIME

    def test_probes_by_source_subset(self) -> None:
        """Probes by source should be a subset of all probes."""
        all_probes = unified_atlas.UnifiedAtlasInventory.all_probes()
        subset = unified_atlas.UnifiedAtlasInventory.probes_by_source(
            {unified_atlas.AtlasSource.SEMANTIC_PRIME}
        )
        assert len(subset) <= len(all_probes)

    def test_probes_by_domain(self) -> None:
        domains = {AtlasDomain.LINGUISTIC}
        result = unified_atlas.UnifiedAtlasInventory.probes_by_domain(domains)
        assert isinstance(result, list)
        assert len(result) > 0, "Expected at least one LINGUISTIC domain probe"
        for p in result:
            assert p.domain is AtlasDomain.LINGUISTIC

    def test_probes_by_domain_subset(self) -> None:
        all_probes = unified_atlas.UnifiedAtlasInventory.all_probes()
        subset = unified_atlas.UnifiedAtlasInventory.probes_by_domain(
            {AtlasDomain.MATHEMATICAL}
        )
        assert len(subset) <= len(all_probes)

    def test_probe_count(self) -> None:
        counts = unified_atlas.UnifiedAtlasInventory.probe_count()
        assert isinstance(counts, dict)
        assert len(counts) > 0
        for key, value in counts.items():
            assert isinstance(key, unified_atlas.AtlasSource)
            assert isinstance(value, int)
            assert value > 0

    def test_probe_count_sums_to_total(self) -> None:
        counts = unified_atlas.UnifiedAtlasInventory.probe_count()
        total_from_counts = sum(counts.values())
        total = unified_atlas.UnifiedAtlasInventory.total_probe_count()
        assert total_from_counts == total

    def test_total_probe_count(self) -> None:
        total = unified_atlas.UnifiedAtlasInventory.total_probe_count()
        assert isinstance(total, int)
        assert total > 0

    def test_total_probe_count_matches_all_probes(self) -> None:
        probes = unified_atlas.UnifiedAtlasInventory.all_probes()
        total = unified_atlas.UnifiedAtlasInventory.total_probe_count()
        assert total == len(probes)

    def test_clear_cache(self) -> None:
        probes_first = unified_atlas.UnifiedAtlasInventory.all_probes()
        unified_atlas.UnifiedAtlasInventory.clear_cache()
        probes_second = unified_atlas.UnifiedAtlasInventory.all_probes()
        assert len(probes_first) == len(probes_second)

    def test_clear_cache_resets_internal_state(self) -> None:
        unified_atlas.UnifiedAtlasInventory.all_probes()
        assert unified_atlas.UnifiedAtlasInventory._cached_probes is not None
        unified_atlas.UnifiedAtlasInventory.clear_cache()
        assert unified_atlas.UnifiedAtlasInventory._cached_probes is None


# ---------------------------------------------------------------------------
# MultiAtlasTriangulationScore frozen dataclass
# ---------------------------------------------------------------------------


class TestMultiAtlasTriangulationScore:
    """Tests for the MultiAtlasTriangulationScore frozen dataclass."""

    def test_instantiation(self) -> None:
        score = unified_atlas.MultiAtlasTriangulationScore(
            layer_index=5,
            sources_detected={unified_atlas.AtlasSource.SEMANTIC_PRIME},
            domains_detected={AtlasDomain.LINGUISTIC},
            source_multiplier=1.0,
            domain_multiplier=1.0,
            combined_multiplier=1.0,
        )
        assert score.layer_index == 5
        assert unified_atlas.AtlasSource.SEMANTIC_PRIME in score.sources_detected
        assert AtlasDomain.LINGUISTIC in score.domains_detected
        assert score.source_multiplier == 1.0
        assert score.domain_multiplier == 1.0
        assert score.combined_multiplier == 1.0

    def test_frozen_enforcement(self) -> None:
        score = unified_atlas.MultiAtlasTriangulationScore(
            layer_index=0,
            sources_detected=set(),
            domains_detected=set(),
            source_multiplier=1.0,
            domain_multiplier=1.0,
            combined_multiplier=1.0,
        )
        with pytest.raises(FrozenInstanceError):
            score.layer_index = 10  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            score.source_multiplier = 2.0  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            score.combined_multiplier = 5.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MultiAtlasTriangulationScorer
# ---------------------------------------------------------------------------


class TestMultiAtlasTriangulationScorer:
    """Tests for the MultiAtlasTriangulationScorer."""

    def _make_probe(self, source, domain, probe_id="p1") -> unified_atlas.AtlasProbe:
        return unified_atlas.AtlasProbe(
            id=probe_id,
            source=source,
            domain=domain,
            name="test",
            description="test probe",
            cross_domain_weight=1.0,
            category_name="test",
        )

    def test_compute_score_empty_activations(self, any_backend, monkeypatch) -> None:
        monkeypatch.setattr(
            unified_atlas, "get_default_backend", lambda: any_backend
        )
        score = unified_atlas.MultiAtlasTriangulationScorer.compute_score({})
        assert isinstance(score, unified_atlas.MultiAtlasTriangulationScore)
        assert score.source_multiplier == 1.0
        assert score.domain_multiplier == 1.0
        assert score.combined_multiplier == 1.0
        assert len(score.sources_detected) == 0
        assert len(score.domains_detected) == 0

    def test_compute_score_single_probe(self, any_backend, monkeypatch) -> None:
        monkeypatch.setattr(
            unified_atlas, "get_default_backend", lambda: any_backend
        )
        probe = self._make_probe(
            unified_atlas.AtlasSource.SEMANTIC_PRIME,
            AtlasDomain.LINGUISTIC,
        )
        activations = {probe: 5.0}
        score = unified_atlas.MultiAtlasTriangulationScorer.compute_score(activations)
        assert isinstance(score, unified_atlas.MultiAtlasTriangulationScore)
        assert unified_atlas.AtlasSource.SEMANTIC_PRIME in score.sources_detected
        assert AtlasDomain.LINGUISTIC in score.domains_detected
        assert score.source_multiplier >= 1.0
        assert score.domain_multiplier >= 1.0
        assert score.combined_multiplier >= 1.0

    def test_compute_score_multiple_sources(self, any_backend, monkeypatch) -> None:
        monkeypatch.setattr(
            unified_atlas, "get_default_backend", lambda: any_backend
        )
        probe_a = self._make_probe(
            unified_atlas.AtlasSource.SEMANTIC_PRIME,
            AtlasDomain.LINGUISTIC,
            probe_id="p1",
        )
        probe_b = self._make_probe(
            unified_atlas.AtlasSource.EMOTION_CONCEPT,
            AtlasDomain.AFFECTIVE,
            probe_id="p2",
        )
        activations = {probe_a: 10.0, probe_b: 10.0}
        score = unified_atlas.MultiAtlasTriangulationScorer.compute_score(activations)
        assert len(score.sources_detected) >= 1
        assert score.combined_multiplier >= 1.0

    def test_compute_score_layer_index_is_minus1(self, any_backend, monkeypatch) -> None:
        """The scorer sets layer_index to -1 (to be set by caller)."""
        monkeypatch.setattr(
            unified_atlas, "get_default_backend", lambda: any_backend
        )
        score = unified_atlas.MultiAtlasTriangulationScorer.compute_score({})
        assert score.layer_index == -1


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module-level constants."""

    def test_all_atlas_sources_is_frozenset(self) -> None:
        assert isinstance(unified_atlas.ALL_ATLAS_SOURCES, frozenset)

    def test_all_atlas_sources_contains_all_members(self) -> None:
        for source in unified_atlas.AtlasSource:
            assert source in unified_atlas.ALL_ATLAS_SOURCES

    def test_all_atlas_sources_count(self) -> None:
        assert len(unified_atlas.ALL_ATLAS_SOURCES) == len(unified_atlas.AtlasSource)

    def test_mathematical_domains_is_frozenset(self) -> None:
        assert isinstance(unified_atlas.MATHEMATICAL_DOMAINS, frozenset)
        assert AtlasDomain.MATHEMATICAL in unified_atlas.MATHEMATICAL_DOMAINS
        assert AtlasDomain.LOGICAL in unified_atlas.MATHEMATICAL_DOMAINS

    def test_linguistic_domains_is_frozenset(self) -> None:
        assert isinstance(unified_atlas.LINGUISTIC_DOMAINS, frozenset)
        assert AtlasDomain.LINGUISTIC in unified_atlas.LINGUISTIC_DOMAINS
        assert AtlasDomain.MENTAL in unified_atlas.LINGUISTIC_DOMAINS

    def test_computational_domains_is_frozenset(self) -> None:
        assert isinstance(unified_atlas.COMPUTATIONAL_DOMAINS, frozenset)
        assert AtlasDomain.COMPUTATIONAL in unified_atlas.COMPUTATIONAL_DOMAINS
        assert AtlasDomain.STRUCTURAL in unified_atlas.COMPUTATIONAL_DOMAINS

    def test_affective_domains_is_frozenset(self) -> None:
        assert isinstance(unified_atlas.AFFECTIVE_DOMAINS, frozenset)
        assert AtlasDomain.AFFECTIVE in unified_atlas.AFFECTIVE_DOMAINS
        assert AtlasDomain.RELATIONAL in unified_atlas.AFFECTIVE_DOMAINS

    def test_spatiotemporal_domains_is_frozenset(self) -> None:
        assert isinstance(unified_atlas.SPATIOTEMPORAL_DOMAINS, frozenset)
        assert AtlasDomain.TEMPORAL in unified_atlas.SPATIOTEMPORAL_DOMAINS
        assert AtlasDomain.SPATIAL in unified_atlas.SPATIOTEMPORAL_DOMAINS

    def test_moral_domains_is_frozenset(self) -> None:
        assert isinstance(unified_atlas.MORAL_DOMAINS, frozenset)
        assert AtlasDomain.MORAL in unified_atlas.MORAL_DOMAINS

    def test_philosophical_domains_is_frozenset(self) -> None:
        assert isinstance(unified_atlas.PHILOSOPHICAL_DOMAINS, frozenset)
        assert AtlasDomain.PHILOSOPHICAL in unified_atlas.PHILOSOPHICAL_DOMAINS
        assert AtlasDomain.LOGICAL in unified_atlas.PHILOSOPHICAL_DOMAINS

    def test_safety_domains_is_frozenset(self) -> None:
        assert isinstance(unified_atlas.SAFETY_DOMAINS, frozenset)
        assert AtlasDomain.SAFETY in unified_atlas.SAFETY_DOMAINS

    def test_default_atlas_sources_equals_all(self) -> None:
        assert unified_atlas.DEFAULT_ATLAS_SOURCES == unified_atlas.ALL_ATLAS_SOURCES


# ---------------------------------------------------------------------------
# _DEFAULT_WEIGHTS
# ---------------------------------------------------------------------------


class TestDefaultWeights:
    """Tests for the _DEFAULT_WEIGHTS dict."""

    def test_all_sources_have_weight(self) -> None:
        for source in unified_atlas.AtlasSource:
            assert source in unified_atlas._DEFAULT_WEIGHTS

    def test_all_weights_are_1_0(self) -> None:
        for source, weight in unified_atlas._DEFAULT_WEIGHTS.items():
            assert weight == 1.0, f"Source {source.value} has weight {weight}, expected 1.0"

    def test_weight_count_matches_source_count(self) -> None:
        assert len(unified_atlas._DEFAULT_WEIGHTS) == len(unified_atlas.AtlasSource)


# ---------------------------------------------------------------------------
# get_probe_ids
# ---------------------------------------------------------------------------


class TestGetProbeIds:
    """Tests for the get_probe_ids function."""

    def setup_method(self) -> None:
        unified_atlas.UnifiedAtlasInventory.clear_cache()

    def teardown_method(self) -> None:
        unified_atlas.UnifiedAtlasInventory.clear_cache()

    def test_returns_list_of_strings(self) -> None:
        ids = unified_atlas.get_probe_ids()
        assert isinstance(ids, list)
        for pid in ids:
            assert isinstance(pid, str)

    def test_all_ids_contain_colon(self) -> None:
        ids = unified_atlas.get_probe_ids()
        for pid in ids:
            assert ":" in pid, f"Probe id missing colon separator: {pid}"

    def test_source_colon_id_format(self) -> None:
        ids = unified_atlas.get_probe_ids()
        source_values = {s.value for s in unified_atlas.AtlasSource}
        for pid in ids:
            parts = pid.split(":", 1)
            assert len(parts) == 2, f"Probe id has wrong format: {pid}"
            source_part = parts[0]
            assert source_part in source_values, (
                f"Probe id has unknown source prefix: {source_part}"
            )

    def test_with_specific_sources(self) -> None:
        sources = {unified_atlas.AtlasSource.SEMANTIC_PRIME}
        ids = unified_atlas.get_probe_ids(sources)
        assert len(ids) > 0
        for pid in ids:
            assert pid.startswith("semantic_prime:")

    def test_none_sources_returns_all(self) -> None:
        all_ids = unified_atlas.get_probe_ids(None)
        total = unified_atlas.UnifiedAtlasInventory.total_probe_count()
        assert len(all_ids) == total

    def test_no_args_returns_all(self) -> None:
        all_ids = unified_atlas.get_probe_ids()
        total = unified_atlas.UnifiedAtlasInventory.total_probe_count()
        assert len(all_ids) == total
