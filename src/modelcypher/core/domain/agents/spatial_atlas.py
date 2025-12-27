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

"""
Spatial Atlas.

Spatial probes for triangulating 3D structure in LLM representations.
Encodes 3D spatial anchors across vertical, lateral, depth, mass, and
furniture categories with expected coordinates for geometric validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SpatialAxis(str, Enum):
    """The three primitive axes of 3D space."""

    X_LATERAL = "x_lateral"  # Left <-> Right
    Y_VERTICAL = "y_vertical"  # Up <-> Down (Gravity)
    Z_DEPTH = "z_depth"  # Forward <-> Backward (Perspective)


class SpatialCategory(str, Enum):
    """Categories of spatial probes."""

    VERTICAL = "vertical"
    LATERAL = "lateral"
    DEPTH = "depth"
    MASS = "mass"
    FURNITURE = "furniture"


_CATEGORY_AXIS_MAP: dict[SpatialCategory, SpatialAxis] = {
    SpatialCategory.VERTICAL: SpatialAxis.Y_VERTICAL,
    SpatialCategory.MASS: SpatialAxis.Y_VERTICAL,
    SpatialCategory.FURNITURE: SpatialAxis.Y_VERTICAL,
    SpatialCategory.LATERAL: SpatialAxis.X_LATERAL,
    SpatialCategory.DEPTH: SpatialAxis.Z_DEPTH,
}


@dataclass(frozen=True)
class SpatialConcept:
    """A spatial probe with expected 3D coordinates."""

    id: str
    name: str
    prompt: str
    expected_x: float  # -1 (left) to +1 (right)
    expected_y: float  # -1 (down) to +1 (up)
    expected_z: float  # -1 (far) to +1 (near)
    category: SpatialCategory

    @property
    def axis(self) -> SpatialAxis:
        return _CATEGORY_AXIS_MAP[self.category]

    @property
    def canonical_name(self) -> str:
        return self.name


ALL_SPATIAL_PROBES: tuple[SpatialConcept, ...] = (
    # Vertical axis (Y) - Gravity gradient
    SpatialConcept("ceiling", "ceiling", "The ceiling is above.", 0.0, 1.0, 0.0, SpatialCategory.VERTICAL),
    SpatialConcept("floor", "floor", "The floor is below.", 0.0, -1.0, 0.0, SpatialCategory.VERTICAL),
    SpatialConcept("sky", "sky", "The sky stretches overhead.", 0.0, 1.0, 0.5, SpatialCategory.VERTICAL),
    SpatialConcept("ground", "ground", "The ground beneath our feet.", 0.0, -1.0, 0.0, SpatialCategory.VERTICAL),
    SpatialConcept("cloud", "cloud", "A cloud floats high above.", 0.0, 0.8, 0.3, SpatialCategory.VERTICAL),
    SpatialConcept("basement", "basement", "The basement is underground.", 0.0, -0.9, 0.0, SpatialCategory.VERTICAL),
    # Lateral axis (X) - Sidedness
    SpatialConcept("left_hand", "left_hand", "My left hand is on my left side.", -1.0, 0.0, 0.5, SpatialCategory.LATERAL),
    SpatialConcept("right_hand", "right_hand", "My right hand is on my right side.", 1.0, 0.0, 0.5, SpatialCategory.LATERAL),
    SpatialConcept("west", "west", "The sun sets in the west.", -0.8, 0.0, 0.0, SpatialCategory.LATERAL),
    SpatialConcept("east", "east", "The sun rises in the east.", 0.8, 0.0, 0.0, SpatialCategory.LATERAL),
    # Depth axis (Z) - Perspective
    SpatialConcept("foreground", "foreground", "The object in the foreground is close.", 0.0, 0.0, 1.0, SpatialCategory.DEPTH),
    SpatialConcept("background", "background", "The mountains in the background are distant.", 0.0, 0.0, -1.0, SpatialCategory.DEPTH),
    SpatialConcept("horizon", "horizon", "The horizon line marks the far distance.", 0.0, 0.0, -0.9, SpatialCategory.DEPTH),
    SpatialConcept("here", "here", "I am standing right here.", 0.0, 0.0, 1.0, SpatialCategory.DEPTH),
    SpatialConcept("there", "there", "The building is over there.", 0.0, 0.0, -0.5, SpatialCategory.DEPTH),
    # Physical objects with mass (Gravity test)
    SpatialConcept("balloon", "balloon", "A helium balloon floats upward.", 0.0, 0.7, 0.5, SpatialCategory.MASS),
    SpatialConcept("stone", "stone", "A heavy stone falls downward.", 0.0, -0.7, 0.5, SpatialCategory.MASS),
    SpatialConcept("feather", "feather", "A light feather drifts slowly.", 0.0, 0.3, 0.5, SpatialCategory.MASS),
    SpatialConcept("anvil", "anvil", "The anvil sinks like a rock.", 0.0, -0.9, 0.5, SpatialCategory.MASS),
    # Furniture (Virtual room test)
    SpatialConcept("chair", "chair", "A chair sits on the floor.", 0.0, -0.5, 0.5, SpatialCategory.FURNITURE),
    SpatialConcept("table", "table", "A table stands in the room.", 0.0, -0.3, 0.5, SpatialCategory.FURNITURE),
    SpatialConcept("lamp", "lamp", "A lamp hangs from the ceiling.", 0.0, 0.7, 0.5, SpatialCategory.FURNITURE),
    SpatialConcept("rug", "rug", "A rug lies flat on the floor.", 0.0, -0.9, 0.5, SpatialCategory.FURNITURE),
)


class SpatialConceptInventory:
    """Complete inventory of spatial concepts for manifold analysis."""

    @staticmethod
    def all_concepts() -> list[SpatialConcept]:
        return list(ALL_SPATIAL_PROBES)

    @staticmethod
    def by_category(category: SpatialCategory) -> list[SpatialConcept]:
        return [c for c in ALL_SPATIAL_PROBES if c.category == category]

    @staticmethod
    def by_axis(axis: SpatialAxis) -> list[SpatialConcept]:
        return [c for c in ALL_SPATIAL_PROBES if c.axis == axis]

    @staticmethod
    def count() -> int:
        return len(ALL_SPATIAL_PROBES)
