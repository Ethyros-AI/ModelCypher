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

"""Tests for atlas_registry.py - global registry for probe inventories.

Tests cover all 18 register/get function pairs:
- atlas_probes
- sequence_invariants
- sequence_triangulation_scorer
- gate_inventory
- spatial_concepts
- social_concepts
- temporal_concepts
- moral_concepts
- metaphor_invariants
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

# Import the module to access the private globals for cleanup
from modelcypher.core.domain.geometry import atlas_registry
from modelcypher.core.domain.geometry.atlas_registry import (
    get_atlas_probes,
    get_gate_inventory,
    get_metaphor_invariants,
    get_moral_concepts,
    get_sequence_invariants,
    get_sequence_triangulation_scorer,
    get_social_concepts,
    get_spatial_concepts,
    get_temporal_concepts,
    register_atlas_probes,
    register_gate_inventory,
    register_metaphor_invariants,
    register_moral_concepts,
    register_sequence_invariants,
    register_sequence_triangulation_scorer,
    register_social_concepts,
    register_spatial_concepts,
    register_temporal_concepts,
)


# =============================================================================
# Mock Implementations
# =============================================================================


@dataclass
class MockAtlasProbe:
    probe_id: str
    name: str
    description: str
    support_texts: list[str]
    source: str
    domain: str
    category_name: str
    cross_domain_weight: float


@dataclass
class MockSequenceInvariant:
    id: str
    family: str
    domain: str
    support_texts: list[str]
    cross_domain_weight: float


@dataclass
class MockTriangulatedScore:
    base: float
    cross_domain_multiplier: float
    relationship_bonus: float
    coherence_bonus: float


@dataclass
class MockComputationalGate:
    id: str
    name: str
    description: str
    examples: list[str]
    polyglot_examples: list[str]


@dataclass
class MockSpatialConcept:
    id: str
    name: str
    prompt: str
    expected_x: float
    expected_y: float
    expected_z: float
    category: str
    axis: str


@dataclass
class MockSocialConcept:
    id: str
    name: str
    axis: str
    category: str
    level: int
    description: str
    support_texts: list[str]


@dataclass
class MockTemporalConcept:
    id: str
    name: str
    axis: str
    category: str
    level: int
    description: str
    support_texts: list[str]


@dataclass
class MockMoralConcept:
    id: str
    name: str
    axis: str
    foundation: str
    level: int
    description: str
    support_texts: list[str]


@dataclass
class MockMetaphorInvariant:
    id: str
    family: str


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset all registry globals before and after each test."""
    # Clear before test
    atlas_registry._ATLAS_PROBES = None
    atlas_registry._SEQUENCE_INVARIANTS = None
    atlas_registry._SEQUENCE_TRIANGULATION_SCORER = None
    atlas_registry._GATE_INVENTORY = None
    atlas_registry._SPATIAL_CONCEPTS = None
    atlas_registry._SOCIAL_CONCEPTS = None
    atlas_registry._TEMPORAL_CONCEPTS = None
    atlas_registry._MORAL_CONCEPTS = None
    atlas_registry._METAPHOR_INVARIANTS = None
    
    yield
    
    # Clear after test
    atlas_registry._ATLAS_PROBES = None
    atlas_registry._SEQUENCE_INVARIANTS = None
    atlas_registry._SEQUENCE_TRIANGULATION_SCORER = None
    atlas_registry._GATE_INVENTORY = None
    atlas_registry._SPATIAL_CONCEPTS = None
    atlas_registry._SOCIAL_CONCEPTS = None
    atlas_registry._TEMPORAL_CONCEPTS = None
    atlas_registry._MORAL_CONCEPTS = None
    atlas_registry._METAPHOR_INVARIANTS = None


# =============================================================================
# Atlas Probes Tests
# =============================================================================


class TestAtlasProbes:
    """Tests for atlas probe registry."""

    def test_get_returns_empty_when_not_registered(self):
        """get_atlas_probes returns empty sequence when nothing registered."""
        result = get_atlas_probes()
        assert result == ()

    def test_register_and_get(self):
        """register_atlas_probes stores probes that get_atlas_probes retrieves."""
        probes = [
            MockAtlasProbe("p1", "Probe1", "Desc", ["text"], "src", "dom", "cat", 1.0),
            MockAtlasProbe("p2", "Probe2", "Desc2", ["text2"], "src", "dom", "cat", 0.5),
        ]
        register_atlas_probes(probes)
        
        result = get_atlas_probes()
        
        assert len(result) == 2
        assert result[0].probe_id == "p1"
        assert result[1].probe_id == "p2"

    def test_register_converts_to_tuple(self):
        """register_atlas_probes converts input to tuple (immutable)."""
        probes = [MockAtlasProbe("p1", "Probe1", "Desc", [], "src", "dom", "cat", 1.0)]
        register_atlas_probes(probes)
        
        result = get_atlas_probes()
        
        assert isinstance(result, tuple)

    def test_register_overwrites_previous(self):
        """Subsequent register calls overwrite previous registration."""
        probes1 = [MockAtlasProbe("p1", "First", "Desc", [], "src", "dom", "cat", 1.0)]
        probes2 = [MockAtlasProbe("p2", "Second", "Desc", [], "src", "dom", "cat", 1.0)]
        
        register_atlas_probes(probes1)
        register_atlas_probes(probes2)
        
        result = get_atlas_probes()
        assert len(result) == 1
        assert result[0].probe_id == "p2"


# =============================================================================
# Sequence Invariants Tests
# =============================================================================


class TestSequenceInvariants:
    """Tests for sequence invariant registry."""

    def test_get_returns_empty_when_not_registered(self):
        """get_sequence_invariants returns empty sequence when nothing registered."""
        result = get_sequence_invariants()
        assert result == ()

    def test_register_and_get(self):
        """register_sequence_invariants stores invariants."""
        invariants = [
            MockSequenceInvariant("i1", "fam1", "dom1", ["text"], 1.0),
        ]
        register_sequence_invariants(invariants)
        
        result = get_sequence_invariants()
        
        assert len(result) == 1
        assert result[0].id == "i1"

    def test_register_converts_to_tuple(self):
        """register_sequence_invariants converts to tuple."""
        invariants = [MockSequenceInvariant("i1", "fam1", "dom1", [], 1.0)]
        register_sequence_invariants(invariants)
        
        assert isinstance(get_sequence_invariants(), tuple)


# =============================================================================
# Sequence Triangulation Scorer Tests
# =============================================================================


class TestSequenceTriangulationScorer:
    """Tests for sequence triangulation scorer registry."""

    def test_get_returns_none_when_not_registered(self):
        """get_sequence_triangulation_scorer returns None when not registered."""
        result = get_sequence_triangulation_scorer()
        assert result is None

    def test_register_and_get(self):
        """register_sequence_triangulation_scorer stores callable."""

        def scorer(activations, probe, relationships=None):
            return MockTriangulatedScore(0.8, 1.0, 0.1, 0.05)

        register_sequence_triangulation_scorer(scorer)
        
        result = get_sequence_triangulation_scorer()
        
        assert result is not None
        assert callable(result)
        
        # Invoke the scorer to verify it works
        score = result({}, None, None)
        assert score.base == 0.8


# =============================================================================
# Gate Inventory Tests
# =============================================================================


class TestGateInventory:
    """Tests for computational gate inventory registry."""

    def test_get_returns_empty_when_not_registered(self):
        """get_gate_inventory returns empty sequence when not registered."""
        result = get_gate_inventory()
        assert result == ()

    def test_register_and_get(self):
        """register_gate_inventory stores gates."""
        gates = [
            MockComputationalGate("g1", "Add", "Addition", ["1+1"], ["plus"]),
        ]
        register_gate_inventory(gates)
        
        result = get_gate_inventory()
        
        assert len(result) == 1
        assert result[0].id == "g1"

    def test_register_converts_to_tuple(self):
        """register_gate_inventory converts to tuple."""
        gates = [MockComputationalGate("g1", "Add", "Addition", [], [])]
        register_gate_inventory(gates)
        
        assert isinstance(get_gate_inventory(), tuple)


# =============================================================================
# Spatial Concepts Tests
# =============================================================================


class TestSpatialConcepts:
    """Tests for spatial concepts registry."""

    def test_get_returns_empty_when_not_registered(self):
        """get_spatial_concepts returns empty sequence when not registered."""
        result = get_spatial_concepts()
        assert result == ()

    def test_register_and_get(self):
        """register_spatial_concepts stores concepts."""
        concepts = [
            MockSpatialConcept("s1", "Above", "prompt", 0.0, 1.0, 0.0, "vertical", "y"),
        ]
        register_spatial_concepts(concepts)
        
        result = get_spatial_concepts()
        
        assert len(result) == 1
        assert result[0].id == "s1"


# =============================================================================
# Social Concepts Tests
# =============================================================================


class TestSocialConcepts:
    """Tests for social concepts registry."""

    def test_get_returns_empty_when_not_registered(self):
        """get_social_concepts returns empty sequence when not registered."""
        result = get_social_concepts()
        assert result == ()

    def test_register_and_get(self):
        """register_social_concepts stores concepts."""
        concepts = [
            MockSocialConcept("soc1", "Trust", "x", "cat", 1, "desc", ["text"]),
        ]
        register_social_concepts(concepts)
        
        result = get_social_concepts()
        
        assert len(result) == 1
        assert result[0].id == "soc1"


# =============================================================================
# Temporal Concepts Tests
# =============================================================================


class TestTemporalConcepts:
    """Tests for temporal concepts registry."""

    def test_get_returns_empty_when_not_registered(self):
        """get_temporal_concepts returns empty sequence when not registered."""
        result = get_temporal_concepts()
        assert result == ()

    def test_register_and_get(self):
        """register_temporal_concepts stores concepts."""
        concepts = [
            MockTemporalConcept("t1", "Past", "time", "cat", 1, "desc", ["text"]),
        ]
        register_temporal_concepts(concepts)
        
        result = get_temporal_concepts()
        
        assert len(result) == 1
        assert result[0].id == "t1"


# =============================================================================
# Moral Concepts Tests
# =============================================================================


class TestMoralConcepts:
    """Tests for moral concepts registry."""

    def test_get_returns_empty_when_not_registered(self):
        """get_moral_concepts returns empty sequence when not registered."""
        result = get_moral_concepts()
        assert result == ()

    def test_register_and_get(self):
        """register_moral_concepts stores concepts."""
        concepts = [
            MockMoralConcept("m1", "Care", "x", "harm", 1, "desc", ["text"]),
        ]
        register_moral_concepts(concepts)
        
        result = get_moral_concepts()
        
        assert len(result) == 1
        assert result[0].id == "m1"


# =============================================================================
# Metaphor Invariants Tests
# =============================================================================


class TestMetaphorInvariants:
    """Tests for metaphor invariants registry."""

    def test_get_returns_empty_when_not_registered(self):
        """get_metaphor_invariants returns empty sequence when not registered."""
        result = get_metaphor_invariants()
        assert result == ()

    def test_register_and_get(self):
        """register_metaphor_invariants stores invariants."""
        invariants = [
            MockMetaphorInvariant("inv1", "family1"),
        ]
        register_metaphor_invariants(invariants)
        
        result = get_metaphor_invariants()
        
        assert len(result) == 1
        assert result[0].id == "inv1"

    def test_register_converts_to_tuple(self):
        """register_metaphor_invariants converts to tuple."""
        invariants = [MockMetaphorInvariant("inv1", "fam1")]
        register_metaphor_invariants(invariants)
        
        assert isinstance(get_metaphor_invariants(), tuple)
