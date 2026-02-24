# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

import modelcypher.core.domain.geometry.atlas_registry as mod

# ---------------------------------------------------------------------------
# Fake protocol implementations
# ---------------------------------------------------------------------------


@dataclass
class _FakeAtlasProbe:
    probe_id: str = "p1"
    name: str = "probe"
    description: str = "desc"
    support_texts: list[str] = field(default_factory=lambda: ["a"])
    source: str = "src"
    domain: str = "dom"
    category_name: str = "cat"
    cross_domain_weight: float = 1.0
    verification_depth: int | None = None


@dataclass
class _FakeSequenceInvariant:
    id: str = "si1"
    family: str = "fam"
    domain: str = "dom"
    support_texts: list[str] = field(default_factory=lambda: ["x"])
    cross_domain_weight: float = 0.5


@dataclass
class _FakeTriangulatedScore:
    base: float = 1.0
    cross_domain_multiplier: float = 1.5
    relationship_bonus: float = 0.2
    coherence_bonus: float = 0.1


@dataclass
class _FakeGate:
    id: str = "g1"
    name: str = "gate"
    description: str = "desc"
    examples: list[str] = field(default_factory=lambda: ["e"])
    polyglot_examples: list[str] = field(default_factory=lambda: ["pe"])


@dataclass
class _FakeSpatialConcept:
    id: str = "sp1"
    name: str = "near"
    prompt: str = "?"
    expected_x: float = 0.0
    expected_y: float = 1.0
    expected_z: float = 0.0
    category: str = "spatial"
    axis: str = "x"


@dataclass
class _FakeSocialConcept:
    id: str = "so1"
    name: str = "trust"
    axis: str = "social"
    category: str = "social"
    level: int = 1
    description: str = "desc"
    support_texts: list[str] = field(default_factory=lambda: ["t"])


@dataclass
class _FakeTemporalConcept:
    id: str = "tc1"
    name: str = "now"
    axis: str = "temporal"
    category: str = "temporal"
    level: int = 2
    description: str = "desc"
    support_texts: list[str] = field(default_factory=lambda: ["u"])


@dataclass
class _FakeMoralConcept:
    id: str = "mc1"
    name: str = "care"
    axis: str = "moral"
    foundation: str = "care_harm"
    level: int = 3
    description: str = "desc"
    support_texts: list[str] = field(default_factory=lambda: ["v"])


@dataclass
class _FakeMetaphorInvariant:
    id: str = "mi1"
    family: str = "fam"


# ---------------------------------------------------------------------------
# Fixture to reset all module-level globals between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_registry(monkeypatch):
    """Reset all registry globals to None before each test."""
    monkeypatch.setattr(mod, "_ATLAS_PROBES", None)
    monkeypatch.setattr(mod, "_SEQUENCE_INVARIANTS", None)
    monkeypatch.setattr(mod, "_SEQUENCE_TRIANGULATION_SCORER", None)
    monkeypatch.setattr(mod, "_GATE_INVENTORY", None)
    monkeypatch.setattr(mod, "_SPATIAL_CONCEPTS", None)
    monkeypatch.setattr(mod, "_SOCIAL_CONCEPTS", None)
    monkeypatch.setattr(mod, "_TEMPORAL_CONCEPTS", None)
    monkeypatch.setattr(mod, "_MORAL_CONCEPTS", None)
    monkeypatch.setattr(mod, "_METAPHOR_INVARIANTS", None)


# ---------------------------------------------------------------------------
# Atlas probes
# ---------------------------------------------------------------------------


class TestAtlasProbes:
    def test_get_before_register_returns_empty(self) -> None:
        assert mod.get_atlas_probes() == ()

    def test_register_and_get(self) -> None:
        probes = [_FakeAtlasProbe(probe_id="a"), _FakeAtlasProbe(probe_id="b")]
        mod.register_atlas_probes(probes)
        result = mod.get_atlas_probes()
        assert len(result) == 2
        assert result[0].probe_id == "a"
        assert result[1].probe_id == "b"

    def test_register_converts_to_tuple(self) -> None:
        mod.register_atlas_probes([_FakeAtlasProbe()])
        assert isinstance(mod.get_atlas_probes(), tuple)


# ---------------------------------------------------------------------------
# Sequence invariants
# ---------------------------------------------------------------------------


class TestSequenceInvariants:
    def test_get_before_register_returns_empty(self) -> None:
        assert mod.get_sequence_invariants() == ()

    def test_register_and_get(self) -> None:
        invariants = [_FakeSequenceInvariant(id="x")]
        mod.register_sequence_invariants(invariants)
        result = mod.get_sequence_invariants()
        assert len(result) == 1
        assert result[0].id == "x"

    def test_register_converts_to_tuple(self) -> None:
        mod.register_sequence_invariants([_FakeSequenceInvariant()])
        assert isinstance(mod.get_sequence_invariants(), tuple)


# ---------------------------------------------------------------------------
# Sequence triangulation scorer
# ---------------------------------------------------------------------------


class TestSequenceTriangulationScorer:
    def test_get_before_register_returns_none(self) -> None:
        assert mod.get_sequence_triangulation_scorer() is None

    def test_register_and_get(self) -> None:
        def _scorer(scores, invariant, ctx=None):
            return _FakeTriangulatedScore()

        mod.register_sequence_triangulation_scorer(_scorer)
        scorer = mod.get_sequence_triangulation_scorer()
        assert scorer is not None
        assert callable(scorer)

    def test_registered_scorer_is_callable(self) -> None:
        def _scorer(scores, invariant, ctx=None):
            return _FakeTriangulatedScore(base=42.0)

        mod.register_sequence_triangulation_scorer(_scorer)
        scorer = mod.get_sequence_triangulation_scorer()
        result = scorer({}, None, None)
        assert result.base == 42.0


# ---------------------------------------------------------------------------
# Gate inventory
# ---------------------------------------------------------------------------


class TestGateInventory:
    def test_get_before_register_returns_empty(self) -> None:
        assert mod.get_gate_inventory() == ()

    def test_register_and_get(self) -> None:
        gates = [_FakeGate(id="g1"), _FakeGate(id="g2")]
        mod.register_gate_inventory(gates)
        result = mod.get_gate_inventory()
        assert len(result) == 2
        assert result[0].id == "g1"

    def test_register_converts_to_tuple(self) -> None:
        mod.register_gate_inventory([_FakeGate()])
        assert isinstance(mod.get_gate_inventory(), tuple)


# ---------------------------------------------------------------------------
# Spatial concepts
# ---------------------------------------------------------------------------


class TestSpatialConcepts:
    def test_get_before_register_returns_empty(self) -> None:
        assert mod.get_spatial_concepts() == ()

    def test_register_and_get(self) -> None:
        concepts = [_FakeSpatialConcept(id="sp1")]
        mod.register_spatial_concepts(concepts)
        result = mod.get_spatial_concepts()
        assert len(result) == 1
        assert result[0].id == "sp1"

    def test_register_converts_to_tuple(self) -> None:
        mod.register_spatial_concepts([_FakeSpatialConcept()])
        assert isinstance(mod.get_spatial_concepts(), tuple)


# ---------------------------------------------------------------------------
# Social concepts
# ---------------------------------------------------------------------------


class TestSocialConcepts:
    def test_get_before_register_returns_empty(self) -> None:
        assert mod.get_social_concepts() == ()

    def test_register_and_get(self) -> None:
        concepts = [_FakeSocialConcept(id="so1")]
        mod.register_social_concepts(concepts)
        result = mod.get_social_concepts()
        assert len(result) == 1
        assert result[0].id == "so1"

    def test_register_converts_to_tuple(self) -> None:
        mod.register_social_concepts([_FakeSocialConcept()])
        assert isinstance(mod.get_social_concepts(), tuple)


# ---------------------------------------------------------------------------
# Temporal concepts
# ---------------------------------------------------------------------------


class TestTemporalConcepts:
    def test_get_before_register_returns_empty(self) -> None:
        assert mod.get_temporal_concepts() == ()

    def test_register_and_get(self) -> None:
        concepts = [_FakeTemporalConcept(id="tc1")]
        mod.register_temporal_concepts(concepts)
        result = mod.get_temporal_concepts()
        assert len(result) == 1
        assert result[0].id == "tc1"

    def test_register_converts_to_tuple(self) -> None:
        mod.register_temporal_concepts([_FakeTemporalConcept()])
        assert isinstance(mod.get_temporal_concepts(), tuple)


# ---------------------------------------------------------------------------
# Moral concepts
# ---------------------------------------------------------------------------


class TestMoralConcepts:
    def test_get_before_register_returns_empty(self) -> None:
        assert mod.get_moral_concepts() == ()

    def test_register_and_get(self) -> None:
        concepts = [_FakeMoralConcept(id="mc1")]
        mod.register_moral_concepts(concepts)
        result = mod.get_moral_concepts()
        assert len(result) == 1
        assert result[0].id == "mc1"

    def test_register_converts_to_tuple(self) -> None:
        mod.register_moral_concepts([_FakeMoralConcept()])
        assert isinstance(mod.get_moral_concepts(), tuple)


# ---------------------------------------------------------------------------
# Metaphor invariants
# ---------------------------------------------------------------------------


class TestMetaphorInvariants:
    def test_get_before_register_returns_empty(self) -> None:
        assert mod.get_metaphor_invariants() == ()

    def test_register_and_get(self) -> None:
        invariants = [_FakeMetaphorInvariant(id="mi1")]
        mod.register_metaphor_invariants(invariants)
        result = mod.get_metaphor_invariants()
        assert len(result) == 1
        assert result[0].id == "mi1"

    def test_register_converts_to_tuple(self) -> None:
        mod.register_metaphor_invariants([_FakeMetaphorInvariant()])
        assert isinstance(mod.get_metaphor_invariants(), tuple)


# ---------------------------------------------------------------------------
# Cross-cutting: overwrite behavior
# ---------------------------------------------------------------------------


class TestOverwrite:
    def test_second_register_replaces_first(self) -> None:
        mod.register_atlas_probes([_FakeAtlasProbe(probe_id="first")])
        mod.register_atlas_probes([_FakeAtlasProbe(probe_id="second")])
        result = mod.get_atlas_probes()
        assert len(result) == 1
        assert result[0].probe_id == "second"

    def test_register_empty_clears(self) -> None:
        mod.register_gate_inventory([_FakeGate()])
        assert len(mod.get_gate_inventory()) == 1
        mod.register_gate_inventory([])
        assert mod.get_gate_inventory() == ()
