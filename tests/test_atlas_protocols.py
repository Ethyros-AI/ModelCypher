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

"""Tests for atlas_protocols.py utility functions and protocol checks.

Tests cover:
- enum_key: Normalize Enum-like values to stable string keys
- enum_key_set: Normalize a collection of values to string key sets
- axis_key: Convert axis enum/value to lowercase string key
- Protocol runtime_checkable behavior
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pytest

from modelcypher.core.domain.geometry.atlas_protocols import (
    AtlasProbeProtocol,
    ComputationalGateProtocol,
    MoralConceptProtocol,
    SequenceInvariantProtocol,
    SocialConceptProtocol,
    SpatialConceptProtocol,
    TemporalConceptProtocol,
    TriangulatedScoreProtocol,
    axis_key,
    enum_key,
    enum_key_set,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class MockEnum(Enum):
    """Mock enum for testing."""

    VALUE_A = "value_a"
    VALUE_B = "value_b"
    UPPERCASE = "UPPERCASE"


class MockEnumInt(Enum):
    """Mock enum with int values."""

    ONE = 1
    TWO = 2


@dataclass
class MockAtlasProbe:
    """Mock implementation of AtlasProbeProtocol."""

    probe_id: str
    name: str
    description: str
    support_texts: list[str]
    source: str
    domain: str
    category_name: str
    cross_domain_weight: float


@dataclass
class MockSpatialConcept:
    """Mock implementation of SpatialConceptProtocol."""

    id: str
    name: str
    prompt: str
    expected_x: float
    expected_y: float
    expected_z: float
    category: str
    axis: str


# =============================================================================
# enum_key Tests
# =============================================================================


class TestEnumKey:
    """Tests for enum_key function."""

    def test_enum_key_with_string_enum(self):
        """enum_key extracts value from string enum."""
        result = enum_key(MockEnum.VALUE_A)
        assert result == "value_a"

    def test_enum_key_with_int_enum(self):
        """enum_key converts int enum value to string."""
        result = enum_key(MockEnumInt.ONE)
        assert result == "1"

    def test_enum_key_with_plain_string(self):
        """enum_key returns string unchanged."""
        result = enum_key("plain_string")
        assert result == "plain_string"

    def test_enum_key_with_plain_int(self):
        """enum_key converts plain int to string."""
        result = enum_key(42)
        assert result == "42"

    def test_enum_key_with_float(self):
        """enum_key converts float to string."""
        result = enum_key(3.14)
        assert result == "3.14"

    def test_enum_key_preserves_case(self):
        """enum_key preserves original case."""
        result = enum_key(MockEnum.UPPERCASE)
        assert result == "UPPERCASE"

    def test_enum_key_with_none_attribute(self):
        """enum_key handles objects without .value attribute."""

        class NoValue:
            """Object without value attribute."""

        obj = NoValue()
        result = enum_key(obj)
        assert "NoValue" in result  # str(obj) contains class name


# =============================================================================
# enum_key_set Tests
# =============================================================================


class TestEnumKeySet:
    """Tests for enum_key_set function."""

    def test_enum_key_set_with_enum_list(self):
        """enum_key_set normalizes list of enums."""
        result = enum_key_set([MockEnum.VALUE_A, MockEnum.VALUE_B])
        assert result == {"value_a", "value_b"}

    def test_enum_key_set_with_mixed_types(self):
        """enum_key_set handles mixed types."""
        result = enum_key_set([MockEnum.VALUE_A, "plain", 42])
        assert result == {"value_a", "plain", "42"}

    def test_enum_key_set_with_empty_iterable(self):
        """enum_key_set returns empty set for empty input."""
        result = enum_key_set([])
        assert result == set()

    def test_enum_key_set_with_duplicates(self):
        """enum_key_set deduplicates values."""
        result = enum_key_set([MockEnum.VALUE_A, MockEnum.VALUE_A])
        assert result == {"value_a"}

    def test_enum_key_set_with_generator(self):
        """enum_key_set works with generators."""
        gen = (x for x in [MockEnum.VALUE_A, "test"])
        result = enum_key_set(gen)
        assert result == {"value_a", "test"}


# =============================================================================
# axis_key Tests
# =============================================================================


class TestAxisKey:
    """Tests for axis_key function."""

    def test_axis_key_lowercases_enum(self):
        """axis_key lowercases enum value."""
        result = axis_key(MockEnum.UPPERCASE)
        assert result == "uppercase"

    def test_axis_key_lowercases_string(self):
        """axis_key lowercases plain string."""
        result = axis_key("TEMPORAL")
        assert result == "temporal"

    def test_axis_key_preserves_already_lowercase(self):
        """axis_key preserves already lowercase strings."""
        result = axis_key("spatial")
        assert result == "spatial"

    def test_axis_key_with_mixed_case(self):
        """axis_key handles mixed case."""
        result = axis_key("MoralFoundation")
        assert result == "moralfoundation"


# =============================================================================
# Protocol Runtime Checkable Tests
# =============================================================================


class TestAtlasProbeProtocol:
    """Tests for AtlasProbeProtocol runtime checkable."""

    def test_isinstance_with_valid_implementation(self):
        """isinstance returns True for valid implementation."""
        probe = MockAtlasProbe(
            probe_id="p1",
            name="Test",
            description="Desc",
            support_texts=["text"],
            source="src",
            domain="dom",
            category_name="cat",
            cross_domain_weight=1.0,
        )
        assert isinstance(probe, AtlasProbeProtocol)

    def test_isinstance_with_invalid_implementation(self):
        """isinstance returns False for incomplete implementation."""

        @dataclass
        class Incomplete:
            probe_id: str  # Missing other required fields

        obj = Incomplete(probe_id="p1")
        assert not isinstance(obj, AtlasProbeProtocol)


class TestSpatialConceptProtocol:
    """Tests for SpatialConceptProtocol runtime checkable."""

    def test_isinstance_with_valid_implementation(self):
        """isinstance returns True for valid implementation."""
        concept = MockSpatialConcept(
            id="s1",
            name="Above",
            prompt="above prompt",
            expected_x=0.0,
            expected_y=1.0,
            expected_z=0.0,
            category="vertical",
            axis="y",
        )
        assert isinstance(concept, SpatialConceptProtocol)


class TestSequenceInvariantProtocol:
    """Tests for SequenceInvariantProtocol runtime checkable."""

    def test_isinstance_with_valid_implementation(self):
        """isinstance returns True for valid implementation."""

        @dataclass
        class MockSequenceInvariant:
            id: str
            family: str
            domain: str
            support_texts: list[str]
            cross_domain_weight: float

        inv = MockSequenceInvariant(
            id="seq1", family="fam", domain="dom", support_texts=["t"], cross_domain_weight=1.0
        )
        assert isinstance(inv, SequenceInvariantProtocol)


class TestTriangulatedScoreProtocol:
    """Tests for TriangulatedScoreProtocol runtime checkable."""

    def test_isinstance_with_valid_implementation(self):
        """isinstance returns True for valid implementation."""

        @dataclass
        class MockScore:
            base: float
            cross_domain_multiplier: float
            relationship_bonus: float
            coherence_bonus: float

        score = MockScore(base=0.8, cross_domain_multiplier=1.2, relationship_bonus=0.1, coherence_bonus=0.05)
        assert isinstance(score, TriangulatedScoreProtocol)


class TestSocialConceptProtocol:
    """Tests for SocialConceptProtocol runtime checkable."""

    def test_isinstance_with_valid_implementation(self):
        """isinstance returns True for valid implementation."""

        @dataclass
        class MockSocialConcept:
            id: str
            name: str
            axis: str
            category: str
            level: int
            description: str
            support_texts: list[str]

        concept = MockSocialConcept(
            id="soc1", name="Trust", axis="x", category="cat", level=1, description="desc", support_texts=["t"]
        )
        assert isinstance(concept, SocialConceptProtocol)


class TestTemporalConceptProtocol:
    """Tests for TemporalConceptProtocol runtime checkable."""

    def test_isinstance_with_valid_implementation(self):
        """isinstance returns True for valid implementation."""

        @dataclass
        class MockTemporalConcept:
            id: str
            name: str
            axis: str
            category: str
            level: int
            description: str
            support_texts: list[str]

        concept = MockTemporalConcept(
            id="t1", name="Past", axis="time", category="cat", level=1, description="desc", support_texts=["t"]
        )
        assert isinstance(concept, TemporalConceptProtocol)


class TestMoralConceptProtocol:
    """Tests for MoralConceptProtocol runtime checkable."""

    def test_isinstance_with_valid_implementation(self):
        """isinstance returns True for valid implementation."""

        @dataclass
        class MockMoralConcept:
            id: str
            name: str
            axis: str
            foundation: str
            level: int
            description: str
            support_texts: list[str]

        concept = MockMoralConcept(
            id="m1", name="Care", axis="x", foundation="harm", level=1, description="desc", support_texts=["t"]
        )
        assert isinstance(concept, MoralConceptProtocol)


class TestComputationalGateProtocol:
    """Tests for ComputationalGateProtocol runtime checkable."""

    def test_isinstance_with_valid_implementation(self):
        """isinstance returns True for valid implementation."""

        @dataclass
        class MockComputationalGate:
            id: str
            name: str
            description: str
            examples: list[str]
            polyglot_examples: list[str]

        gate = MockComputationalGate(
            id="g1", name="Addition", description="Add numbers", examples=["1+1"], polyglot_examples=["add"]
        )
        assert isinstance(gate, ComputationalGateProtocol)
