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

"""Conceptual Metaphor Theory (CMT) atlas for geometric metaphor analysis.

Provides structured source→target domain mappings based on Lakoff & Johnson's
foundational work "Metaphors We Live By" (1980). Each CMT mapping includes:

- Source domain: The concrete domain providing structure (e.g., MONEY, WAR)
- Target domain: The abstract domain being understood (e.g., TIME, ARGUMENT)
- Exemplars: Words/phrases from each domain for activation collection
- Bridging expressions: Metaphorical phrases that cross domains

Research Foundation:
    - Lakoff & Johnson (1980) "Metaphors We Live By"
    - arXiv 2502.01901 "Conceptual Metaphor Theory as Prompting Paradigm"
    - arXiv 2505.22563 "Do LLMs Think Like the Brain?" (layer-wise analysis)
    - arXiv 2405.07987 "The Platonic Representation Hypothesis"

Usage:
    The CMT mappings are used for layer-wise trajectory analysis:
    1. Collect activations for source_exemplars at each layer
    2. Collect activations for target_exemplars at each layer
    3. Measure CKA between source/target activations per layer
    4. Find convergence_layer where CKA peaks (metaphor mapping occurs)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CMTFamily(str, Enum):
    """Conceptual Metaphor Theory family categories."""

    TIME_AS_RESOURCE = "time_as_resource"
    """TIME IS MONEY, TIME IS LIMITED - treating time as spendable/wasteable."""

    ARGUMENT_AS_CONFLICT = "argument_as_conflict"
    """ARGUMENT IS WAR - debates as battles with attacks and defenses."""

    LIFE_AS_JOURNEY = "life_as_journey"
    """LIFE IS A JOURNEY - existence as travel with paths and destinations."""

    IDEAS_AS_OBJECTS = "ideas_as_objects"
    """IDEAS ARE FOOD/TOOLS - concepts as consumable or usable things."""

    EMOTIONS_AS_SUBSTANCES = "emotions_as_substances"
    """EMOTIONS ARE FLUIDS - feelings as liquids that fill, overflow, drain."""

    MIND_AS_SPACE = "mind_as_space"
    """MIND IS A CONTAINER - consciousness as bounded space."""

    UNDERSTANDING_AS_PERCEPTION = "understanding_as_perception"
    """UNDERSTANDING IS SEEING - comprehension as visual perception."""

    RELATIONSHIPS_AS_JOURNEYS = "relationships_as_journeys"
    """LOVE IS A JOURNEY - relationships as shared travel."""


@dataclass(frozen=True)
class CMTMapping:
    """Conceptual Metaphor Theory mapping: SOURCE IS TARGET.

    Attributes:
        id: Unique identifier for the mapping.
        name: Human-readable name (e.g., "TIME IS MONEY").
        family: CMT family category.
        source_domain: The concrete domain providing structure.
        target_domain: The abstract domain being understood.
        source_exemplars: Words/phrases from source domain for activation.
        target_exemplars: Words/phrases from target domain for activation.
        bridging_expressions: Metaphorical expressions that cross domains.
    """

    id: str
    name: str
    family: CMTFamily
    source_domain: str
    target_domain: str
    source_exemplars: tuple[str, ...] = field(default_factory=tuple)
    target_exemplars: tuple[str, ...] = field(default_factory=tuple)
    bridging_expressions: tuple[str, ...] = field(default_factory=tuple)


class ConceptualMetaphorInventory:
    """Inventory of Conceptual Metaphor Theory mappings.

    Based on Lakoff & Johnson (1980) "Metaphors We Live By" with extensions
    from subsequent cognitive linguistics research.

    Total: 8 CMT mappings across 8 families.
    """

    # MARK: - Time as Resource

    TIME_IS_MONEY = CMTMapping(
        id="cmt_time_is_money",
        name="TIME IS MONEY",
        family=CMTFamily.TIME_AS_RESOURCE,
        source_domain="MONEY",
        target_domain="TIME",
        source_exemplars=(
            "dollar",
            "cent",
            "budget",
            "spend",
            "save",
            "invest",
            "waste",
            "cost",
            "profit",
            "value",
            "rich",
            "poor",
            "bankrupt",
            "currency",
            "transaction",
        ),
        target_exemplars=(
            "hour",
            "minute",
            "day",
            "week",
            "month",
            "year",
            "moment",
            "duration",
            "period",
            "deadline",
            "schedule",
            "calendar",
            "clock",
            "time",
            "instant",
        ),
        bridging_expressions=(
            "spend time",
            "waste time",
            "save time",
            "invest time",
            "time is money",
            "budget your time",
            "running out of time",
            "time well spent",
            "precious time",
            "time is valuable",
        ),
    )

    # MARK: - Argument as Conflict

    ARGUMENT_IS_WAR = CMTMapping(
        id="cmt_argument_is_war",
        name="ARGUMENT IS WAR",
        family=CMTFamily.ARGUMENT_AS_CONFLICT,
        source_domain="WAR",
        target_domain="ARGUMENT",
        source_exemplars=(
            "attack",
            "defend",
            "battle",
            "fight",
            "enemy",
            "ally",
            "strategy",
            "weapon",
            "victory",
            "defeat",
            "surrender",
            "retreat",
            "target",
            "ammunition",
            "fortress",
        ),
        target_exemplars=(
            "debate",
            "discuss",
            "claim",
            "argument",
            "position",
            "conclusion",
            "premise",
            "logic",
            "reason",
            "evidence",
            "opinion",
            "viewpoint",
            "thesis",
            "rebuttal",
            "counterpoint",
        ),
        bridging_expressions=(
            "defend your position",
            "attack the argument",
            "win the debate",
            "shoot down ideas",
            "target weak points",
            "strategic argument",
            "demolish the claim",
            "indefensible position",
            "ammunition for debate",
            "war of words",
        ),
    )

    # MARK: - Life as Journey

    LIFE_IS_A_JOURNEY = CMTMapping(
        id="cmt_life_is_journey",
        name="LIFE IS A JOURNEY",
        family=CMTFamily.LIFE_AS_JOURNEY,
        source_domain="JOURNEY",
        target_domain="LIFE",
        source_exemplars=(
            "path",
            "road",
            "destination",
            "crossroads",
            "journey",
            "travel",
            "map",
            "compass",
            "direction",
            "milestone",
            "obstacle",
            "vehicle",
            "passenger",
            "guide",
            "horizon",
        ),
        target_exemplars=(
            "life",
            "existence",
            "career",
            "purpose",
            "goal",
            "future",
            "past",
            "choice",
            "decision",
            "growth",
            "progress",
            "setback",
            "success",
            "failure",
            "meaning",
        ),
        bridging_expressions=(
            "crossroads in life",
            "life path",
            "dead end",
            "new direction",
            "reach a milestone",
            "hit a roadblock",
            "journey of life",
            "long road ahead",
            "on the right track",
            "lost in life",
        ),
    )

    # MARK: - Ideas as Objects

    IDEAS_ARE_FOOD = CMTMapping(
        id="cmt_ideas_are_food",
        name="IDEAS ARE FOOD",
        family=CMTFamily.IDEAS_AS_OBJECTS,
        source_domain="FOOD",
        target_domain="IDEAS",
        source_exemplars=(
            "food",
            "meal",
            "eat",
            "digest",
            "swallow",
            "taste",
            "chew",
            "cook",
            "raw",
            "baked",
            "fresh",
            "stale",
            "appetite",
            "hunger",
            "nourish",
        ),
        target_exemplars=(
            "idea",
            "concept",
            "thought",
            "theory",
            "notion",
            "understanding",
            "knowledge",
            "information",
            "insight",
            "wisdom",
            "learning",
            "comprehension",
            "thesis",
            "hypothesis",
            "proposal",
        ),
        bridging_expressions=(
            "digest the information",
            "half-baked idea",
            "food for thought",
            "chew on that",
            "swallow the truth",
            "raw data",
            "meaty argument",
            "spoon-fed",
            "intellectual appetite",
            "hungry for knowledge",
        ),
    )

    # MARK: - Relationships as Journeys

    LOVE_IS_A_JOURNEY = CMTMapping(
        id="cmt_love_is_journey",
        name="LOVE IS A JOURNEY",
        family=CMTFamily.RELATIONSHIPS_AS_JOURNEYS,
        source_domain="JOURNEY",
        target_domain="LOVE",
        source_exemplars=(
            "path",
            "road",
            "destination",
            "crossroads",
            "journey",
            "travel",
            "direction",
            "together",
            "apart",
            "baggage",
            "companion",
            "fork",
            "detour",
            "distance",
            "arrival",
        ),
        target_exemplars=(
            "love",
            "relationship",
            "marriage",
            "partner",
            "romance",
            "couple",
            "commitment",
            "bond",
            "connection",
            "intimacy",
            "devotion",
            "affection",
            "passion",
            "attachment",
            "union",
        ),
        bridging_expressions=(
            "at a crossroads",
            "going nowhere",
            "long road together",
            "bumpy relationship",
            "parting ways",
            "stuck in a rut",
            "moving forward together",
            "emotional baggage",
            "on the same path",
            "relationship journey",
        ),
    )

    # MARK: - Understanding as Perception

    UNDERSTANDING_IS_SEEING = CMTMapping(
        id="cmt_understanding_is_seeing",
        name="UNDERSTANDING IS SEEING",
        family=CMTFamily.UNDERSTANDING_AS_PERCEPTION,
        source_domain="SEEING",
        target_domain="UNDERSTANDING",
        source_exemplars=(
            "see",
            "look",
            "view",
            "vision",
            "sight",
            "light",
            "dark",
            "blind",
            "eye",
            "focus",
            "clarity",
            "illuminate",
            "reveal",
            "hidden",
            "visible",
        ),
        target_exemplars=(
            "understand",
            "know",
            "comprehend",
            "realize",
            "grasp",
            "learn",
            "recognize",
            "perceive",
            "cognize",
            "discern",
            "apprehend",
            "fathom",
            "intuit",
            "conceive",
            "interpret",
        ),
        bridging_expressions=(
            "I see what you mean",
            "shed light on",
            "blind to the truth",
            "clear understanding",
            "point of view",
            "in the dark",
            "enlighten me",
            "see the light",
            "vision of the future",
            "illuminate the issue",
        ),
    )

    # MARK: - Emotions as Substances

    EMOTIONS_ARE_FLUIDS = CMTMapping(
        id="cmt_emotions_are_fluids",
        name="EMOTIONS ARE FLUIDS",
        family=CMTFamily.EMOTIONS_AS_SUBSTANCES,
        source_domain="FLUIDS",
        target_domain="EMOTIONS",
        source_exemplars=(
            "water",
            "liquid",
            "flow",
            "pour",
            "overflow",
            "drain",
            "flood",
            "wave",
            "tide",
            "boil",
            "simmer",
            "steam",
            "pressure",
            "container",
            "vessel",
        ),
        target_exemplars=(
            "anger",
            "joy",
            "sadness",
            "fear",
            "love",
            "hate",
            "happiness",
            "grief",
            "anxiety",
            "excitement",
            "frustration",
            "contentment",
            "despair",
            "elation",
            "emotion",
        ),
        bridging_expressions=(
            "boiling with rage",
            "overflow with joy",
            "drained of emotion",
            "flood of feelings",
            "wave of sadness",
            "bottled up emotions",
            "let off steam",
            "bubbling with excitement",
            "emotional outpouring",
            "tide of anger",
        ),
    )

    # MARK: - Mind as Space

    MIND_IS_A_CONTAINER = CMTMapping(
        id="cmt_mind_is_container",
        name="MIND IS A CONTAINER",
        family=CMTFamily.MIND_AS_SPACE,
        source_domain="CONTAINER",
        target_domain="MIND",
        source_exemplars=(
            "container",
            "box",
            "space",
            "room",
            "full",
            "empty",
            "open",
            "closed",
            "enter",
            "exit",
            "inside",
            "outside",
            "boundary",
            "walls",
            "capacity",
        ),
        target_exemplars=(
            "mind",
            "brain",
            "consciousness",
            "thought",
            "memory",
            "imagination",
            "intellect",
            "cognition",
            "awareness",
            "psyche",
            "mental",
            "thinking",
            "reasoning",
            "perception",
            "attention",
        ),
        bridging_expressions=(
            "open mind",
            "closed mind",
            "empty-headed",
            "full of ideas",
            "in the back of my mind",
            "keep in mind",
            "out of mind",
            "mental space",
            "room for thought",
            "bounded thinking",
        ),
    )

    # All CMT mappings
    ALL_MAPPINGS: list[CMTMapping] = [
        TIME_IS_MONEY,
        ARGUMENT_IS_WAR,
        LIFE_IS_A_JOURNEY,
        IDEAS_ARE_FOOD,
        LOVE_IS_A_JOURNEY,
        UNDERSTANDING_IS_SEEING,
        EMOTIONS_ARE_FLUIDS,
        MIND_IS_A_CONTAINER,
    ]

    @classmethod
    def mappings_by_family(cls, family: CMTFamily) -> list[CMTMapping]:
        """Get all mappings for a given family."""
        return [m for m in cls.ALL_MAPPINGS if m.family == family]

    @classmethod
    def get_by_id(cls, mapping_id: str) -> CMTMapping | None:
        """Get a mapping by its ID."""
        for mapping in cls.ALL_MAPPINGS:
            if mapping.id == mapping_id:
                return mapping
        return None

    @classmethod
    def get_by_name(cls, name: str) -> CMTMapping | None:
        """Get a mapping by its name (case-insensitive)."""
        name_upper = name.upper().replace("_", " ")
        for mapping in cls.ALL_MAPPINGS:
            if mapping.name.upper() == name_upper:
                return mapping
        return None
