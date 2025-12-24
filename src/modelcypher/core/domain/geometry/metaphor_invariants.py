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
Metaphor Invariants for Extended Probing.

Provides probe texts based on cross-cultural conceptual metaphor mappings.
Tests whether models preserve conceptual structure when the surface metaphor varies.

Reference: Lakoff & Johnson (1980) "Metaphors We Live By"
Reference: Kövecses (2005) "Metaphor in Culture: Universality and Variation"
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MetaphorDomain(str, Enum):
    """Source domains for conceptual metaphors."""

    TIME = "time"
    EMOTION = "emotion"
    ARGUMENT = "argument"
    LIFE = "life"
    KNOWLEDGE = "knowledge"
    MORALITY = "morality"
    ECONOMICS = "economics"
    POWER = "power"


class CulturalContext(str, Enum):
    """Cultural contexts for metaphor variation."""

    WESTERN = "western"  # Primarily English/European
    EASTERN = "eastern"  # Primarily Chinese/Japanese
    UNIVERSAL = "universal"  # Cross-cultural
    ARABIC = "arabic"  # Middle Eastern
    AFRICAN = "african"  # Sub-Saharan African


@dataclass(frozen=True)
class MetaphorMapping:
    """A conceptual metaphor mapping between source and target domains."""

    target_domain: MetaphorDomain
    source_concept: str  # e.g., "MONEY", "JOURNEY", "WAR"
    cultural_context: CulturalContext
    description: str


@dataclass(frozen=True)
class MetaphorProbe:
    """A probe text derived from a conceptual metaphor."""

    domain: MetaphorDomain
    source_concept: str
    cultural_context: CulturalContext
    probe_text: str
    probe_id: str
    underlying_concept: str  # The abstract concept being tested

    @property
    def category(self) -> str:
        return f"metaphor_{self.domain.value}"


# =============================================================================
# Core Metaphor Mappings
# =============================================================================

# TIME metaphors
TIME_METAPHORS = [
    # Western: TIME IS MONEY
    MetaphorMapping(
        target_domain=MetaphorDomain.TIME,
        source_concept="MONEY",
        cultural_context=CulturalContext.WESTERN,
        description="Time as a valuable commodity that can be spent, saved, wasted",
    ),
    # Eastern: TIME IS A RIVER
    MetaphorMapping(
        target_domain=MetaphorDomain.TIME,
        source_concept="RIVER",
        cultural_context=CulturalContext.EASTERN,
        description="Time as a flowing river carrying us along",
    ),
    # Universal: TIME IS MOTION
    MetaphorMapping(
        target_domain=MetaphorDomain.TIME,
        source_concept="MOTION",
        cultural_context=CulturalContext.UNIVERSAL,
        description="Time as movement through space",
    ),
]

# EMOTION metaphors
EMOTION_METAPHORS = [
    # Universal: ANGER IS HEAT
    MetaphorMapping(
        target_domain=MetaphorDomain.EMOTION,
        source_concept="HEAT",
        cultural_context=CulturalContext.UNIVERSAL,
        description="Anger as rising temperature, boiling, explosion",
    ),
    # Western: HAPPINESS IS UP
    MetaphorMapping(
        target_domain=MetaphorDomain.EMOTION,
        source_concept="VERTICALITY",
        cultural_context=CulturalContext.WESTERN,
        description="Positive emotions as upward movement",
    ),
    # Eastern: HAPPINESS IS BALANCE
    MetaphorMapping(
        target_domain=MetaphorDomain.EMOTION,
        source_concept="BALANCE",
        cultural_context=CulturalContext.EASTERN,
        description="Positive states as harmony and equilibrium",
    ),
]

# ARGUMENT metaphors
ARGUMENT_METAPHORS = [
    # Western: ARGUMENT IS WAR
    MetaphorMapping(
        target_domain=MetaphorDomain.ARGUMENT,
        source_concept="WAR",
        cultural_context=CulturalContext.WESTERN,
        description="Arguments as battles with winners and losers",
    ),
    # Eastern: ARGUMENT IS DANCE
    MetaphorMapping(
        target_domain=MetaphorDomain.ARGUMENT,
        source_concept="DANCE",
        cultural_context=CulturalContext.EASTERN,
        description="Arguments as collaborative movement toward truth",
    ),
]

# LIFE metaphors
LIFE_METAPHORS = [
    # Western: LIFE IS A JOURNEY
    MetaphorMapping(
        target_domain=MetaphorDomain.LIFE,
        source_concept="JOURNEY",
        cultural_context=CulturalContext.WESTERN,
        description="Life as travel with destinations, paths, obstacles",
    ),
    # Universal: LIFE IS A STORY
    MetaphorMapping(
        target_domain=MetaphorDomain.LIFE,
        source_concept="STORY",
        cultural_context=CulturalContext.UNIVERSAL,
        description="Life as narrative with chapters, plot, meaning",
    ),
    # Eastern: LIFE IS A DREAM
    MetaphorMapping(
        target_domain=MetaphorDomain.LIFE,
        source_concept="DREAM",
        cultural_context=CulturalContext.EASTERN,
        description="Life as illusion or transient experience",
    ),
]

# KNOWLEDGE metaphors
KNOWLEDGE_METAPHORS = [
    # Universal: UNDERSTANDING IS SEEING
    MetaphorMapping(
        target_domain=MetaphorDomain.KNOWLEDGE,
        source_concept="SEEING",
        cultural_context=CulturalContext.UNIVERSAL,
        description="Knowledge acquisition as visual perception",
    ),
    # Western: IDEAS ARE OBJECTS
    MetaphorMapping(
        target_domain=MetaphorDomain.KNOWLEDGE,
        source_concept="OBJECTS",
        cultural_context=CulturalContext.WESTERN,
        description="Ideas as things that can be grasped, held, given",
    ),
    # Eastern: KNOWLEDGE IS LIGHT
    MetaphorMapping(
        target_domain=MetaphorDomain.KNOWLEDGE,
        source_concept="LIGHT",
        cultural_context=CulturalContext.EASTERN,
        description="Knowledge as illumination dispelling darkness",
    ),
]


# =============================================================================
# Probe Text Generation
# =============================================================================


def generate_time_probes() -> list[MetaphorProbe]:
    """Generate probes for TIME metaphors."""
    probes = []

    # TIME IS MONEY (Western)
    money_texts = [
        ("You're wasting my time.", "TIME_VALUE"),
        ("This will save you hours.", "TIME_VALUE"),
        ("I invested a week in this project.", "TIME_VALUE"),
        ("Don't spend all your time on that.", "TIME_VALUE"),
        ("Time is running out.", "TIME_SCARCITY"),
    ]
    for i, (text, concept) in enumerate(money_texts):
        probes.append(
            MetaphorProbe(
                domain=MetaphorDomain.TIME,
                source_concept="MONEY",
                cultural_context=CulturalContext.WESTERN,
                probe_text=text,
                probe_id=f"time_money_{i}",
                underlying_concept=concept,
            )
        )

    # TIME IS A RIVER (Eastern)
    river_texts = [
        ("Time flows on, carrying all things.", "TIME_FLOW"),
        ("We drift along the stream of time.", "TIME_FLOW"),
        ("The river of time cannot be reversed.", "TIME_IRREVERSIBILITY"),
        ("Let time carry you forward.", "TIME_FLOW"),
    ]
    for i, (text, concept) in enumerate(river_texts):
        probes.append(
            MetaphorProbe(
                domain=MetaphorDomain.TIME,
                source_concept="RIVER",
                cultural_context=CulturalContext.EASTERN,
                probe_text=text,
                probe_id=f"time_river_{i}",
                underlying_concept=concept,
            )
        )

    return probes


def generate_emotion_probes() -> list[MetaphorProbe]:
    """Generate probes for EMOTION metaphors."""
    probes = []

    # ANGER IS HEAT (Universal)
    heat_texts = [
        ("She was boiling with rage.", "ANGER_INTENSITY"),
        ("His anger reached the boiling point.", "ANGER_INTENSITY"),
        ("Cool down before you speak.", "ANGER_CONTROL"),
        ("He's hot-headed.", "ANGER_DISPOSITION"),
        ("The heated argument continued.", "ANGER_INTERACTION"),
    ]
    for i, (text, concept) in enumerate(heat_texts):
        probes.append(
            MetaphorProbe(
                domain=MetaphorDomain.EMOTION,
                source_concept="HEAT",
                cultural_context=CulturalContext.UNIVERSAL,
                probe_text=text,
                probe_id=f"emotion_heat_{i}",
                underlying_concept=concept,
            )
        )

    # HAPPINESS IS UP (Western)
    up_texts = [
        ("I'm feeling up today.", "MOOD_POSITIVE"),
        ("That really lifted my spirits.", "MOOD_CHANGE"),
        ("She's on top of the world.", "HAPPINESS_PEAK"),
        ("My heart soared with joy.", "HAPPINESS_INTENSITY"),
    ]
    for i, (text, concept) in enumerate(up_texts):
        probes.append(
            MetaphorProbe(
                domain=MetaphorDomain.EMOTION,
                source_concept="VERTICALITY",
                cultural_context=CulturalContext.WESTERN,
                probe_text=text,
                probe_id=f"emotion_up_{i}",
                underlying_concept=concept,
            )
        )

    return probes


def generate_argument_probes() -> list[MetaphorProbe]:
    """Generate probes for ARGUMENT metaphors."""
    probes = []

    # ARGUMENT IS WAR (Western)
    war_texts = [
        ("He attacked every point I made.", "ARGUMENT_AGGRESSION"),
        ("Her arguments were indefensible.", "ARGUMENT_WEAKNESS"),
        ("I demolished his position.", "ARGUMENT_VICTORY"),
        ("She shot down all my proposals.", "ARGUMENT_REJECTION"),
        ("He won the argument.", "ARGUMENT_OUTCOME"),
    ]
    for i, (text, concept) in enumerate(war_texts):
        probes.append(
            MetaphorProbe(
                domain=MetaphorDomain.ARGUMENT,
                source_concept="WAR",
                cultural_context=CulturalContext.WESTERN,
                probe_text=text,
                probe_id=f"argument_war_{i}",
                underlying_concept=concept,
            )
        )

    # ARGUMENT IS DANCE (Eastern/Collaborative)
    dance_texts = [
        ("We moved toward understanding together.", "ARGUMENT_COOPERATION"),
        ("Our ideas flowed in harmony.", "ARGUMENT_HARMONY"),
        ("We found a rhythm in our discussion.", "ARGUMENT_FLOW"),
        ("The conversation was a graceful exchange.", "ARGUMENT_ELEGANCE"),
    ]
    for i, (text, concept) in enumerate(dance_texts):
        probes.append(
            MetaphorProbe(
                domain=MetaphorDomain.ARGUMENT,
                source_concept="DANCE",
                cultural_context=CulturalContext.EASTERN,
                probe_text=text,
                probe_id=f"argument_dance_{i}",
                underlying_concept=concept,
            )
        )

    return probes


def generate_life_probes() -> list[MetaphorProbe]:
    """Generate probes for LIFE metaphors."""
    probes = []

    # LIFE IS A JOURNEY (Western)
    journey_texts = [
        ("He's at a crossroads in his life.", "LIFE_DECISION"),
        ("She's come a long way.", "LIFE_PROGRESS"),
        ("I took the road less traveled.", "LIFE_CHOICE"),
        ("We've hit a dead end.", "LIFE_OBSTACLE"),
        ("She's lost her way.", "LIFE_CONFUSION"),
    ]
    for i, (text, concept) in enumerate(journey_texts):
        probes.append(
            MetaphorProbe(
                domain=MetaphorDomain.LIFE,
                source_concept="JOURNEY",
                cultural_context=CulturalContext.WESTERN,
                probe_text=text,
                probe_id=f"life_journey_{i}",
                underlying_concept=concept,
            )
        )

    # LIFE IS A DREAM (Eastern)
    dream_texts = [
        ("Life is but a passing dream.", "LIFE_TRANSIENCE"),
        ("We awaken from the dream of self.", "LIFE_AWAKENING"),
        ("Reality dissolves like a dream.", "LIFE_ILLUSION"),
        ("This world is maya, an illusion.", "LIFE_ILLUSION"),
    ]
    for i, (text, concept) in enumerate(dream_texts):
        probes.append(
            MetaphorProbe(
                domain=MetaphorDomain.LIFE,
                source_concept="DREAM",
                cultural_context=CulturalContext.EASTERN,
                probe_text=text,
                probe_id=f"life_dream_{i}",
                underlying_concept=concept,
            )
        )

    return probes


def generate_knowledge_probes() -> list[MetaphorProbe]:
    """Generate probes for KNOWLEDGE metaphors."""
    probes = []

    # UNDERSTANDING IS SEEING (Universal)
    seeing_texts = [
        ("I see what you mean.", "UNDERSTANDING"),
        ("Can you shed light on this?", "CLARIFICATION"),
        ("That's a clear explanation.", "CLARITY"),
        ("I'm in the dark about this.", "IGNORANCE"),
        ("The truth came to light.", "REVELATION"),
    ]
    for i, (text, concept) in enumerate(seeing_texts):
        probes.append(
            MetaphorProbe(
                domain=MetaphorDomain.KNOWLEDGE,
                source_concept="SEEING",
                cultural_context=CulturalContext.UNIVERSAL,
                probe_text=text,
                probe_id=f"knowledge_seeing_{i}",
                underlying_concept=concept,
            )
        )

    # IDEAS ARE OBJECTS (Western)
    object_texts = [
        ("Let me give you an idea.", "IDEA_TRANSFER"),
        ("I can't grasp this concept.", "COMPREHENSION"),
        ("Hold that thought.", "IDEA_RETENTION"),
        ("She's full of ideas.", "CREATIVITY"),
    ]
    for i, (text, concept) in enumerate(object_texts):
        probes.append(
            MetaphorProbe(
                domain=MetaphorDomain.KNOWLEDGE,
                source_concept="OBJECTS",
                cultural_context=CulturalContext.WESTERN,
                probe_text=text,
                probe_id=f"knowledge_object_{i}",
                underlying_concept=concept,
            )
        )

    return probes


# =============================================================================
# Main Generation Functions
# =============================================================================


def generate_all_metaphor_probes(
    domains: set[MetaphorDomain] | None = None,
) -> list[MetaphorProbe]:
    """
    Generate all metaphor probes for specified domains.

    Args:
        domains: Set of domains to include (None = all)

    Returns:
        List of all metaphor probes
    """
    if domains is None:
        domains = set(MetaphorDomain)

    all_probes = []

    if MetaphorDomain.TIME in domains:
        all_probes.extend(generate_time_probes())
    if MetaphorDomain.EMOTION in domains:
        all_probes.extend(generate_emotion_probes())
    if MetaphorDomain.ARGUMENT in domains:
        all_probes.extend(generate_argument_probes())
    if MetaphorDomain.LIFE in domains:
        all_probes.extend(generate_life_probes())
    if MetaphorDomain.KNOWLEDGE in domains:
        all_probes.extend(generate_knowledge_probes())

    return all_probes


@dataclass(frozen=True)
class MetaphorInvariantPair:
    """
    A pair of metaphor probes testing the same underlying concept
    through different cultural metaphors.

    Used to test whether models preserve conceptual structure
    across different surface realizations.
    """

    concept: str
    probe_a: MetaphorProbe
    probe_b: MetaphorProbe

    @property
    def is_cross_cultural(self) -> bool:
        return self.probe_a.cultural_context != self.probe_b.cultural_context


def generate_cross_cultural_pairs() -> list[MetaphorInvariantPair]:
    """
    Generate pairs of probes that express the same concept
    through different cultural metaphors.
    """
    pairs = []

    # TIME VALUE: Money vs River
    pairs.append(
        MetaphorInvariantPair(
            concept="TIME_VALUE",
            probe_a=MetaphorProbe(
                domain=MetaphorDomain.TIME,
                source_concept="MONEY",
                cultural_context=CulturalContext.WESTERN,
                probe_text="Don't waste your precious time.",
                probe_id="time_pair_west",
                underlying_concept="TIME_VALUE",
            ),
            probe_b=MetaphorProbe(
                domain=MetaphorDomain.TIME,
                source_concept="RIVER",
                cultural_context=CulturalContext.EASTERN,
                probe_text="Let each moment flow with purpose.",
                probe_id="time_pair_east",
                underlying_concept="TIME_VALUE",
            ),
        )
    )

    # ARGUMENT RESOLUTION: War vs Dance
    pairs.append(
        MetaphorInvariantPair(
            concept="ARGUMENT_RESOLUTION",
            probe_a=MetaphorProbe(
                domain=MetaphorDomain.ARGUMENT,
                source_concept="WAR",
                cultural_context=CulturalContext.WESTERN,
                probe_text="We reached a ceasefire in our debate.",
                probe_id="argument_pair_west",
                underlying_concept="ARGUMENT_RESOLUTION",
            ),
            probe_b=MetaphorProbe(
                domain=MetaphorDomain.ARGUMENT,
                source_concept="DANCE",
                cultural_context=CulturalContext.EASTERN,
                probe_text="Our discussion came to a graceful close.",
                probe_id="argument_pair_east",
                underlying_concept="ARGUMENT_RESOLUTION",
            ),
        )
    )

    return pairs


@dataclass
class MetaphorInvariantConfig:
    """Configuration for metaphor invariant probing."""

    domains: set[MetaphorDomain] | None = None  # None = all
    include_cross_cultural_pairs: bool = True
    cultural_contexts: set[CulturalContext] | None = None  # None = all
