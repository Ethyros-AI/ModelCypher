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

"""Protocol helpers for geometry modules that operate on probe inventories.

Geometry must not import the agents domain directly. These lightweight
protocols let geometry stay pure while outer layers inject inventories
from agents (or other sources) without coupling.
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol, Sequence, runtime_checkable


def enum_key(value: Any) -> str:
    """Normalize Enum-like values to stable string keys."""
    return str(getattr(value, "value", value))


def enum_key_set(values: Iterable[Any]) -> set[str]:
    """Normalize a collection of Enum-like values to string keys."""
    return {enum_key(v) for v in values}


@runtime_checkable
class AtlasProbeProtocol(Protocol):
    probe_id: str
    name: str
    description: str
    support_texts: Sequence[str]
    source: Any
    domain: Any
    category_name: str
    cross_domain_weight: float


@runtime_checkable
class SequenceInvariantProtocol(Protocol):
    id: str
    family: Any
    domain: Any
    support_texts: Sequence[str]
    cross_domain_weight: float


@runtime_checkable
class TriangulatedScoreProtocol(Protocol):
    base: float
    cross_domain_multiplier: float
    relationship_bonus: float
    coherence_bonus: float


@runtime_checkable
class SpatialConceptProtocol(Protocol):
    id: str
    name: str
    prompt: str
    expected_x: float
    expected_y: float
    expected_z: float
    category: Any
    axis: Any


@runtime_checkable
class SocialConceptProtocol(Protocol):
    id: str
    name: str
    axis: Any
    category: Any
    level: int
    description: str
    support_texts: Sequence[str]


@runtime_checkable
class TemporalConceptProtocol(Protocol):
    id: str
    name: str
    axis: Any
    category: Any
    level: int
    description: str
    support_texts: Sequence[str]


@runtime_checkable
class MoralConceptProtocol(Protocol):
    id: str
    name: str
    axis: Any
    foundation: Any
    level: int
    description: str
    support_texts: Sequence[str]


@runtime_checkable
class ComputationalGateProtocol(Protocol):
    id: str
    name: str
    description: str
    examples: Sequence[str]
    polyglot_examples: Sequence[str]


@runtime_checkable
class MetaphorInvariantProtocol(Protocol):
    id: str
    family: Any
