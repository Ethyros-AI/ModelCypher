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
from enum import Enum

import modelcypher.core.domain.geometry.atlas_protocols as mod


# ---------------------------------------------------------------------------
# Helper enums and dataclasses for protocol conformance testing
# ---------------------------------------------------------------------------


class _Color(Enum):
    RED = "red"
    GREEN = "green"


@dataclass
class _FakeAtlasProbe:
    probe_id: str = "p1"
    name: str = "probe"
    description: str = "desc"
    support_texts: list[str] = field(default_factory=lambda: ["a", "b"])
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
class _FakeSpatialConcept:
    id: str = "sp1"
    name: str = "near"
    prompt: str = "Is it near?"
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
class _FakeComputationalGate:
    id: str = "cg1"
    name: str = "gate"
    description: str = "desc"
    examples: list[str] = field(default_factory=lambda: ["e1"])
    polyglot_examples: list[str] = field(default_factory=lambda: ["pe1"])


@dataclass
class _FakeMetaphorInvariant:
    id: str = "mi1"
    family: str = "fam"


# ---------------------------------------------------------------------------
# enum_key
# ---------------------------------------------------------------------------


class TestEnumKey:
    def test_enum_key_with_enum(self) -> None:
        assert mod.enum_key(_Color.RED) == "red"

    def test_enum_key_with_string(self) -> None:
        assert mod.enum_key("hello") == "hello"

    def test_enum_key_with_int_enum(self) -> None:
        """An object with a .value attribute is treated like an enum."""

        class _Level(Enum):
            HIGH = 3

        assert mod.enum_key(_Level.HIGH) == "3"


# ---------------------------------------------------------------------------
# enum_key_set
# ---------------------------------------------------------------------------


class TestEnumKeySet:
    def test_enum_key_set_with_enums(self) -> None:
        result = mod.enum_key_set([_Color.RED, _Color.GREEN])
        assert result == {"red", "green"}

    def test_enum_key_set_with_strings(self) -> None:
        result = mod.enum_key_set(["alpha", "beta"])
        assert result == {"alpha", "beta"}

    def test_enum_key_set_empty(self) -> None:
        assert mod.enum_key_set([]) == set()

    def test_enum_key_set_mixed(self) -> None:
        result = mod.enum_key_set([_Color.RED, "plain"])
        assert result == {"red", "plain"}


# ---------------------------------------------------------------------------
# axis_key
# ---------------------------------------------------------------------------


class TestAxisKey:
    def test_axis_key_lowercases_enum(self) -> None:
        class _Axis(Enum):
            X = "X_AXIS"

        assert mod.axis_key(_Axis.X) == "x_axis"

    def test_axis_key_lowercases_string(self) -> None:
        assert mod.axis_key("HELLO") == "hello"

    def test_axis_key_already_lower(self) -> None:
        assert mod.axis_key("already") == "already"


# ---------------------------------------------------------------------------
# Protocol isinstance checks (runtime_checkable)
# ---------------------------------------------------------------------------


class TestAtlasProbeProtocol:
    def test_isinstance(self) -> None:
        probe = _FakeAtlasProbe()
        assert isinstance(probe, mod.AtlasProbeProtocol)

    def test_field_access(self) -> None:
        probe = _FakeAtlasProbe(probe_id="abc", name="test")
        assert probe.probe_id == "abc"
        assert probe.name == "test"


class TestSequenceInvariantProtocol:
    def test_isinstance(self) -> None:
        inv = _FakeSequenceInvariant()
        assert isinstance(inv, mod.SequenceInvariantProtocol)

    def test_field_access(self) -> None:
        inv = _FakeSequenceInvariant(id="xyz", cross_domain_weight=0.9)
        assert inv.id == "xyz"
        assert inv.cross_domain_weight == 0.9


class TestTriangulatedScoreProtocol:
    def test_isinstance(self) -> None:
        score = _FakeTriangulatedScore()
        assert isinstance(score, mod.TriangulatedScoreProtocol)

    def test_field_access(self) -> None:
        score = _FakeTriangulatedScore(base=2.0, coherence_bonus=0.5)
        assert score.base == 2.0
        assert score.coherence_bonus == 0.5


class TestSpatialConceptProtocol:
    def test_isinstance(self) -> None:
        concept = _FakeSpatialConcept()
        assert isinstance(concept, mod.SpatialConceptProtocol)

    def test_field_access(self) -> None:
        concept = _FakeSpatialConcept(expected_x=1.0, expected_y=2.0, expected_z=3.0)
        assert concept.expected_x == 1.0
        assert concept.expected_y == 2.0
        assert concept.expected_z == 3.0


class TestSocialConceptProtocol:
    def test_isinstance(self) -> None:
        concept = _FakeSocialConcept()
        assert isinstance(concept, mod.SocialConceptProtocol)

    def test_field_access(self) -> None:
        concept = _FakeSocialConcept(level=5)
        assert concept.level == 5


class TestTemporalConceptProtocol:
    def test_isinstance(self) -> None:
        concept = _FakeTemporalConcept()
        assert isinstance(concept, mod.TemporalConceptProtocol)

    def test_field_access(self) -> None:
        concept = _FakeTemporalConcept(name="past")
        assert concept.name == "past"


class TestMoralConceptProtocol:
    def test_isinstance(self) -> None:
        concept = _FakeMoralConcept()
        assert isinstance(concept, mod.MoralConceptProtocol)

    def test_field_access(self) -> None:
        concept = _FakeMoralConcept(foundation="loyalty")
        assert concept.foundation == "loyalty"


class TestComputationalGateProtocol:
    def test_isinstance(self) -> None:
        gate = _FakeComputationalGate()
        assert isinstance(gate, mod.ComputationalGateProtocol)

    def test_field_access(self) -> None:
        gate = _FakeComputationalGate(name="relu_gate")
        assert gate.name == "relu_gate"


class TestMetaphorInvariantProtocol:
    def test_isinstance(self) -> None:
        inv = _FakeMetaphorInvariant()
        assert isinstance(inv, mod.MetaphorInvariantProtocol)

    def test_field_access(self) -> None:
        inv = _FakeMetaphorInvariant(id="m99", family="container")
        assert inv.id == "m99"
        assert inv.family == "container"


# ---------------------------------------------------------------------------
# Non-conforming objects should NOT match protocols
# ---------------------------------------------------------------------------


class TestProtocolNonConformance:
    def test_empty_object_not_atlas_probe(self) -> None:
        """An object missing required attributes is not a match."""

        class _Empty:
            pass

        assert not isinstance(_Empty(), mod.AtlasProbeProtocol)

    def test_partial_object_not_sequence_invariant(self) -> None:
        """An object with only some attributes does not satisfy the protocol."""

        @dataclass
        class _Partial:
            id: str = "x"

        assert not isinstance(_Partial(), mod.SequenceInvariantProtocol)
