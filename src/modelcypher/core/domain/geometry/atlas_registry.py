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

"""Registry for optional probe inventories used by geometry modules.

Outer layers (CLI/MCP/use_cases/tests) register inventories from agents here.
Geometry consumes the registry without importing agents, preserving the hexagon.
"""

from __future__ import annotations

from typing import Callable, Sequence

from modelcypher.core.domain.geometry.atlas_protocols import (
    AtlasProbeProtocol,
    ComputationalGateProtocol,
    MetaphorInvariantProtocol,
    MoralConceptProtocol,
    SequenceInvariantProtocol,
    SocialConceptProtocol,
    SpatialConceptProtocol,
    TemporalConceptProtocol,
    TriangulatedScoreProtocol,
)

_ATLAS_PROBES: Sequence[AtlasProbeProtocol] | None = None
_SEQUENCE_INVARIANTS: Sequence[SequenceInvariantProtocol] | None = None
_SEQUENCE_TRIANGULATION_SCORER: Callable[
    [dict, object, dict | None], TriangulatedScoreProtocol
] | None = None
_GATE_INVENTORY: Sequence[ComputationalGateProtocol] | None = None
_SPATIAL_CONCEPTS: Sequence[SpatialConceptProtocol] | None = None
_SOCIAL_CONCEPTS: Sequence[SocialConceptProtocol] | None = None
_TEMPORAL_CONCEPTS: Sequence[TemporalConceptProtocol] | None = None
_MORAL_CONCEPTS: Sequence[MoralConceptProtocol] | None = None
_METAPHOR_INVARIANTS: Sequence[MetaphorInvariantProtocol] | None = None


def register_atlas_probes(probes: Sequence[AtlasProbeProtocol]) -> None:
    global _ATLAS_PROBES
    _ATLAS_PROBES = tuple(probes)


def get_atlas_probes() -> Sequence[AtlasProbeProtocol]:
    return _ATLAS_PROBES or ()


def register_sequence_invariants(invariants: Sequence[SequenceInvariantProtocol]) -> None:
    global _SEQUENCE_INVARIANTS
    _SEQUENCE_INVARIANTS = tuple(invariants)


def get_sequence_invariants() -> Sequence[SequenceInvariantProtocol]:
    return _SEQUENCE_INVARIANTS or ()


def register_sequence_triangulation_scorer(
    scorer: Callable[[dict, object, dict | None], TriangulatedScoreProtocol],
) -> None:
    global _SEQUENCE_TRIANGULATION_SCORER
    _SEQUENCE_TRIANGULATION_SCORER = scorer


def get_sequence_triangulation_scorer() -> (
    Callable[[dict, object, dict | None], TriangulatedScoreProtocol] | None
):
    return _SEQUENCE_TRIANGULATION_SCORER


def register_gate_inventory(gates: Sequence[ComputationalGateProtocol]) -> None:
    global _GATE_INVENTORY
    _GATE_INVENTORY = tuple(gates)


def get_gate_inventory() -> Sequence[ComputationalGateProtocol]:
    return _GATE_INVENTORY or ()


def register_spatial_concepts(concepts: Sequence[SpatialConceptProtocol]) -> None:
    global _SPATIAL_CONCEPTS
    _SPATIAL_CONCEPTS = tuple(concepts)


def get_spatial_concepts() -> Sequence[SpatialConceptProtocol]:
    return _SPATIAL_CONCEPTS or ()


def register_social_concepts(concepts: Sequence[SocialConceptProtocol]) -> None:
    global _SOCIAL_CONCEPTS
    _SOCIAL_CONCEPTS = tuple(concepts)


def get_social_concepts() -> Sequence[SocialConceptProtocol]:
    return _SOCIAL_CONCEPTS or ()


def register_temporal_concepts(concepts: Sequence[TemporalConceptProtocol]) -> None:
    global _TEMPORAL_CONCEPTS
    _TEMPORAL_CONCEPTS = tuple(concepts)


def get_temporal_concepts() -> Sequence[TemporalConceptProtocol]:
    return _TEMPORAL_CONCEPTS or ()


def register_moral_concepts(concepts: Sequence[MoralConceptProtocol]) -> None:
    global _MORAL_CONCEPTS
    _MORAL_CONCEPTS = tuple(concepts)


def get_moral_concepts() -> Sequence[MoralConceptProtocol]:
    return _MORAL_CONCEPTS or ()


def register_metaphor_invariants(
    invariants: Sequence[MetaphorInvariantProtocol],
) -> None:
    global _METAPHOR_INVARIANTS
    _METAPHOR_INVARIANTS = tuple(invariants)


def get_metaphor_invariants() -> Sequence[MetaphorInvariantProtocol]:
    return _METAPHOR_INVARIANTS or ()
