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
Physical Existence Atlas.

3D probes for embodied grounding - testing if a model understands
PHYSICAL EXISTENCE, not just conceptual reasoning.

Without physical grounding, models build on quicksand.

Categories:
- DYNAMICS: Physical cause → effect relationships
- CONSTRAINTS: Physical impossibilities
- PERMANENCE: Object persistence when unobserved
- EMBODIMENT: Proprioception, bodily sensation
- CONTINUITY: Identity through physical change
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PhysicalCategory(str, Enum):
    """Categories of physical existence probes.
    
    NOTE: These test MACROSCOPIC/CLASSICAL physics intuitions.
    At quantum level (0D), many "constraints" break - particles CAN be
    in superposition, CAN tunnel through barriers, etc. We're testing
    whether a model has grounded 3D physical intuition, not absolute physics.
    """

    DYNAMICS = "dynamics"  # Cause → effect
    CLASSICAL_CONSTRAINTS = "classical_constraints"  # Macroscopic physics (NOT quantum)
    PERMANENCE = "permanence"  # Object persistence
    EMBODIMENT = "embodiment"  # Bodily experience
    CONTINUITY = "continuity"  # Identity through change


@dataclass(frozen=True)
class PhysicalConcept:
    """A physical existence probe."""

    id: str
    name: str
    prompt: str
    category: PhysicalCategory
    expected_understanding: str  # What a physically-grounded model should understand

    @property
    def canonical_name(self) -> str:
        return self.name


# =============================================================================
# PHYSICAL DYNAMICS: Cause → Effect
# =============================================================================

DYNAMICS_PROBES: tuple[PhysicalConcept, ...] = (
    PhysicalConcept(
        "ball_window", "ball_window",
        "When a ball hits a glass window, the window...",
        PhysicalCategory.DYNAMICS,
        "breaks/shatters",
    ),
    PhysicalConcept(
        "water_heat", "water_heat",
        "When water is heated to 100°C, it...",
        PhysicalCategory.DYNAMICS,
        "boils/evaporates",
    ),
    PhysicalConcept(
        "egg_drop", "egg_drop",
        "When you drop an egg on a hard floor, the egg...",
        PhysicalCategory.DYNAMICS,
        "breaks/cracks",
    ),
    PhysicalConcept(
        "ice_sun", "ice_sun",
        "When ice is left in the sun, it...",
        PhysicalCategory.DYNAMICS,
        "melts",
    ),
    PhysicalConcept(
        "fire_paper", "fire_paper",
        "When paper touches fire, the paper...",
        PhysicalCategory.DYNAMICS,
        "burns",
    ),
    PhysicalConcept(
        "push_ball", "push_ball",
        "When you push a ball, the ball...",
        PhysicalCategory.DYNAMICS,
        "rolls/moves",
    ),
    PhysicalConcept(
        "release_balloon", "release_balloon",
        "When you release a helium balloon, it...",
        PhysicalCategory.DYNAMICS,
        "rises/floats up",
    ),
    PhysicalConcept(
        "drop_stone", "drop_stone",
        "When you drop a stone in water, it...",
        PhysicalCategory.DYNAMICS,
        "sinks",
    ),
)


# =============================================================================
# CLASSICAL CONSTRAINTS: Macroscopic physics (NOT quantum)
# NOTE: At quantum level, particles CAN be in superposition, CAN tunnel, etc.
# These test whether a model has grounded CLASSICAL 3D intuition.
# =============================================================================

CLASSICAL_CONSTRAINTS_PROBES: tuple[PhysicalConcept, ...] = (
    PhysicalConcept(
        "ball_wall", "ball_wall",
        "Can a solid ball pass through a brick wall without breaking it?",
        PhysicalCategory.CLASSICAL_CONSTRAINTS,
        "No (classical physics)",
    ),
    PhysicalConcept(
        "two_places", "two_places",
        "Can a person be in New York and Tokyo at the exact same moment?",
        PhysicalCategory.CLASSICAL_CONSTRAINTS,
        "No (classical physics)",
    ),
    PhysicalConcept(
        "reverse_time", "reverse_time",
        "Can a broken egg unbreak itself?",
        PhysicalCategory.CLASSICAL_CONSTRAINTS,
        "No (entropy)",
    ),
    PhysicalConcept(
        "object_vanish", "object_vanish",
        "If I close my eyes, does the table in front of me disappear?",
        PhysicalCategory.CLASSICAL_CONSTRAINTS,
        "No",
    ),
    PhysicalConcept(
        "table_cup", "table_cup",
        "If I remove the table, what happens to the cup that was on it?",
        PhysicalCategory.CLASSICAL_CONSTRAINTS,
        "Falls (gravity)",
    ),
    PhysicalConcept(
        "walk_water", "walk_water",
        "Can a person walk on liquid water without sinking?",
        PhysicalCategory.CLASSICAL_CONSTRAINTS,
        "No (density)",
    ),
    PhysicalConcept(
        "fly_unaided", "fly_unaided",
        "Can a person fly by flapping their arms?",
        PhysicalCategory.CLASSICAL_CONSTRAINTS,
        "No (biomechanics)",
    ),
)


# =============================================================================
# OBJECT PERMANENCE: Things exist when unobserved
# =============================================================================

PERMANENCE_PROBES: tuple[PhysicalConcept, ...] = (
    PhysicalConcept(
        "book_drawer", "book_drawer",
        "I put a book in a drawer and close it. Is the book still in the drawer?",
        PhysicalCategory.PERMANENCE,
        "Yes",
    ),
    PhysicalConcept(
        "moon_cloud", "moon_cloud",
        "The moon goes behind a cloud. Does the moon still exist?",
        PhysicalCategory.PERMANENCE,
        "Yes",
    ),
    PhysicalConcept(
        "car_garage", "car_garage",
        "I park my car in the garage and walk away. Is my car still there?",
        PhysicalCategory.PERMANENCE,
        "Yes",
    ),
    PhysicalConcept(
        "ball_blanket", "ball_blanket",
        "A ball rolls under a blanket. Can I find the ball if I lift the blanket?",
        PhysicalCategory.PERMANENCE,
        "Yes",
    ),
    PhysicalConcept(
        "friend_room", "friend_room",
        "My friend walks into another room. Does my friend still exist?",
        PhysicalCategory.PERMANENCE,
        "Yes",
    ),
)


# =============================================================================
# EMBODIMENT: Proprioception and bodily sensation
# =============================================================================

EMBODIMENT_PROBES: tuple[PhysicalConcept, ...] = (
    PhysicalConcept(
        "stub_toe", "stub_toe",
        "When I stub my toe on a rock, I feel...",
        PhysicalCategory.EMBODIMENT,
        "pain",
    ),
    PhysicalConcept(
        "hungry_stomach", "hungry_stomach",
        "When I haven't eaten for hours, my stomach...",
        PhysicalCategory.EMBODIMENT,
        "growls/aches/feels empty",
    ),
    PhysicalConcept(
        "cold_shiver", "cold_shiver",
        "When I'm very cold, I tend to...",
        PhysicalCategory.EMBODIMENT,
        "shiver",
    ),
    PhysicalConcept(
        "hot_sweat", "hot_sweat",
        "When I'm very hot, I tend to...",
        PhysicalCategory.EMBODIMENT,
        "sweat",
    ),
    PhysicalConcept(
        "tired_yawn", "tired_yawn",
        "When I'm tired, I often...",
        PhysicalCategory.EMBODIMENT,
        "yawn",
    ),
    PhysicalConcept(
        "touch_ice", "touch_ice",
        "When I touch ice, my fingers feel...",
        PhysicalCategory.EMBODIMENT,
        "cold",
    ),
    PhysicalConcept(
        "touch_fire", "touch_fire",
        "When I get too close to fire, I feel...",
        PhysicalCategory.EMBODIMENT,
        "heat/pain",
    ),
    PhysicalConcept(
        "eyes_closed", "eyes_closed",
        "When I close my eyes, I can still feel...",
        PhysicalCategory.EMBODIMENT,
        "my body/myself existing",
    ),
)


# =============================================================================
# PHYSICAL CONTINUITY: Identity through change
# =============================================================================

CONTINUITY_PROBES: tuple[PhysicalConcept, ...] = (
    PhysicalConcept(
        "chair_paint", "chair_paint",
        "If I paint a red chair blue, is it still the same chair?",
        PhysicalCategory.CONTINUITY,
        "Yes",
    ),
    PhysicalConcept(
        "cup_break", "cup_break",
        "If I break a cup into many pieces, is it still a cup?",
        PhysicalCategory.CONTINUITY,
        "No",
    ),
    PhysicalConcept(
        "ship_planks", "ship_planks",
        "If I replace every plank of a wooden ship over time, is it the same ship?",
        PhysicalCategory.CONTINUITY,
        "Debatable/Ship of Theseus",
    ),
    PhysicalConcept(
        "caterpillar_butterfly", "caterpillar_butterfly",
        "A caterpillar becomes a butterfly. Is it the same creature?",
        PhysicalCategory.CONTINUITY,
        "Yes",
    ),
    PhysicalConcept(
        "ice_water", "ice_water",
        "Ice melts into water. Is it the same substance?",
        PhysicalCategory.CONTINUITY,
        "Yes (H2O)",
    ),
    PhysicalConcept(
        "person_age", "person_age",
        "A person at age 5 and age 50 - are they the same person?",
        PhysicalCategory.CONTINUITY,
        "Yes",
    ),
)


# =============================================================================
# INVENTORY
# =============================================================================

ALL_PHYSICAL_PROBES: tuple[PhysicalConcept, ...] = (
    DYNAMICS_PROBES +
    CLASSICAL_CONSTRAINTS_PROBES +
    PERMANENCE_PROBES +
    EMBODIMENT_PROBES +
    CONTINUITY_PROBES
)


class PhysicalExistenceInventory:
    """Complete inventory of physical existence probes."""

    @staticmethod
    def all_concepts() -> list[PhysicalConcept]:
        return list(ALL_PHYSICAL_PROBES)

    @staticmethod
    def by_category(category: PhysicalCategory) -> list[PhysicalConcept]:
        return [c for c in ALL_PHYSICAL_PROBES if c.category == category]

    @staticmethod
    def count() -> int:
        return len(ALL_PHYSICAL_PROBES)

    @staticmethod
    def count_by_category() -> dict[str, int]:
        return {
            "dynamics": len(DYNAMICS_PROBES),
            "constraints": len(CONSTRAINTS_PROBES),
            "permanence": len(PERMANENCE_PROBES),
            "embodiment": len(EMBODIMENT_PROBES),
            "continuity": len(CONTINUITY_PROBES),
        }


__all__ = [
    "PhysicalCategory",
    "PhysicalConcept",
    "PhysicalExistenceInventory",
    "ALL_PHYSICAL_PROBES",
]
