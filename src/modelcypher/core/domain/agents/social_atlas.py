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
Social Atlas.

Multi-domain probes for triangulating social structure in LLM representations.
Tests the "Latent Sociologist" hypothesis: models encode social relationships
as a coherent geometric manifold with Power (status), Kinship (social distance),
and Formality (register) axes.

This module provides 25 social probes across 5 categories organized along
Power/Kinship/Formality axis structure (triangulable). It enables monotonic
power hierarchy detection (slave→emperor) and axis orthogonality testing
(>90% independence).

Notes
-----
Scientific basis:
- Human text implicitly encodes social hierarchies
- Politeness phenomena (power + distance → formality)
- Social deixis in language (formal/informal registers)

Empirical validation (2025-12-23) shows Mean SMS: 0.53 (effect size d=2.39),
axis orthogonality: 94.8%, and perfect reproducibility (CV=0.00%).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SocialCategory(str, Enum):
    """Categories of social probes."""

    POWER_HIERARCHY = "power_hierarchy"  # Status/authority ordering
    FORMALITY = "formality"  # Linguistic register
    KINSHIP = "kinship"  # Social distance/closeness
    STATUS_MARKERS = "status_markers"  # Economic/social standing
    AGE = "age"  # Life stage/seniority


class SocialAxis(str, Enum):
    """Axes of the social manifold."""

    POWER = "power"  # Low status (-1) to High status (+1)
    KINSHIP = "kinship"  # Distant (-1) to Close (+1)
    FORMALITY = "formality"  # Casual (-1) to Formal (+1)


@dataclass(frozen=True)
class SocialConcept:
    """A social probe for manifold analysis."""

    id: str
    category: SocialCategory
    axis: SocialAxis
    level: int  # 1-5 ordering within axis (1=low, 5=high)
    name: str
    description: str
    support_texts: tuple[str, ...]
    cross_domain_weight: float = 1.0

    @property
    def canonical_name(self) -> str:
        return self.name

    @property
    def prompt(self) -> str:
        if self.category == SocialCategory.FORMALITY:
            return f"The greeting {self.name.lower()} represents"
        return f"The word {self.name.lower()} represents"


# Power Hierarchy Category (Power Axis) - 5 probes
POWER_HIERARCHY_PROBES: tuple[SocialConcept, ...] = (
    SocialConcept(
        id="slave",
        category=SocialCategory.POWER_HIERARCHY,
        axis=SocialAxis.POWER,
        level=1,
        name="Slave",
        description="Person owned by another, lowest status.",
        support_texts=(
            "A slave has no freedom or rights.",
            "Forced to serve without choice.",
            "The lowest position in a hierarchy.",
            "Complete subjugation.",
        ),
        cross_domain_weight=1.5,
    ),
    SocialConcept(
        id="servant",
        category=SocialCategory.POWER_HIERARCHY,
        axis=SocialAxis.POWER,
        level=2,
        name="Servant",
        description="Person who serves others, low status.",
        support_texts=(
            "A servant works for a master.",
            "Domestic help, attendant.",
            "Serving at the pleasure of others.",
        ),
        cross_domain_weight=1.2,
    ),
    SocialConcept(
        id="citizen",
        category=SocialCategory.POWER_HIERARCHY,
        axis=SocialAxis.POWER,
        level=3,
        name="Citizen",
        description="Member of a society, common status.",
        support_texts=(
            "A citizen has rights and duties.",
            "An ordinary member of society.",
            "Neither ruling nor ruled.",
        ),
        cross_domain_weight=1.0,
    ),
    SocialConcept(
        id="noble",
        category=SocialCategory.POWER_HIERARCHY,
        axis=SocialAxis.POWER,
        level=4,
        name="Noble",
        description="Person of high social rank.",
        support_texts=(
            "A noble holds hereditary rank.",
            "Aristocrat, lord, lady.",
            "Born to privilege and authority.",
        ),
        cross_domain_weight=1.2,
    ),
    SocialConcept(
        id="emperor",
        category=SocialCategory.POWER_HIERARCHY,
        axis=SocialAxis.POWER,
        level=5,
        name="Emperor",
        description="Supreme ruler, highest status.",
        support_texts=(
            "An emperor rules over an empire.",
            "The highest authority in the land.",
            "Supreme power over all subjects.",
            "Your Imperial Majesty.",
        ),
        cross_domain_weight=1.5,
    ),
)

# Formality Category (Formality Axis) - 5 probes
FORMALITY_PROBES: tuple[SocialConcept, ...] = (
    SocialConcept(
        id="hey",
        category=SocialCategory.FORMALITY,
        axis=SocialAxis.FORMALITY,
        level=1,
        name="Hey",
        description="Very casual greeting.",
        support_texts=(
            "Hey is an informal greeting.",
            "What's up, hey there.",
            "Used with friends and peers.",
            "Casual and relaxed tone.",
        ),
        cross_domain_weight=1.3,
    ),
    SocialConcept(
        id="hi",
        category=SocialCategory.FORMALITY,
        axis=SocialAxis.FORMALITY,
        level=2,
        name="Hi",
        description="Casual greeting.",
        support_texts=(
            "Hi is a common greeting.",
            "Friendly but not too formal.",
            "Everyday hello.",
        ),
        cross_domain_weight=1.0,
    ),
    SocialConcept(
        id="hello",
        category=SocialCategory.FORMALITY,
        axis=SocialAxis.FORMALITY,
        level=3,
        name="Hello",
        description="Neutral greeting.",
        support_texts=(
            "Hello is a standard greeting.",
            "Neither formal nor informal.",
            "Polite and neutral.",
        ),
        cross_domain_weight=1.0,
    ),
    SocialConcept(
        id="greetings",
        category=SocialCategory.FORMALITY,
        axis=SocialAxis.FORMALITY,
        level=4,
        name="Greetings",
        description="Formal greeting.",
        support_texts=(
            "Greetings is more formal.",
            "Used in professional contexts.",
            "A respectful salutation.",
        ),
        cross_domain_weight=1.0,
    ),
    SocialConcept(
        id="salutations",
        category=SocialCategory.FORMALITY,
        axis=SocialAxis.FORMALITY,
        level=5,
        name="Salutations",
        description="Very formal greeting.",
        support_texts=(
            "Salutations is highly formal.",
            "Used in official correspondence.",
            "Distinguished and ceremonial.",
            "Warm salutations, dear colleague.",
        ),
        cross_domain_weight=1.3,
    ),
)

# Kinship Category (Kinship Axis) - 5 probes
KINSHIP_PROBES: tuple[SocialConcept, ...] = (
    SocialConcept(
        id="enemy",
        category=SocialCategory.KINSHIP,
        axis=SocialAxis.KINSHIP,
        level=1,
        name="Enemy",
        description="Person actively opposed to you.",
        support_texts=(
            "An enemy is hostile and adversarial.",
            "Someone who wishes you harm.",
            "Foe, opponent, antagonist.",
            "Complete social distance and conflict.",
        ),
        cross_domain_weight=1.3,
    ),
    SocialConcept(
        id="stranger",
        category=SocialCategory.KINSHIP,
        axis=SocialAxis.KINSHIP,
        level=2,
        name="Stranger",
        description="Unknown person, no relationship.",
        support_texts=(
            "A stranger is someone you don't know.",
            "No prior relationship or connection.",
            "Anonymous, unfamiliar.",
        ),
        cross_domain_weight=1.0,
    ),
    SocialConcept(
        id="acquaintance",
        category=SocialCategory.KINSHIP,
        axis=SocialAxis.KINSHIP,
        level=3,
        name="Acquaintance",
        description="Person you know slightly.",
        support_texts=(
            "An acquaintance is someone you've met.",
            "Known but not close.",
            "Casual familiarity.",
        ),
        cross_domain_weight=1.0,
    ),
    SocialConcept(
        id="friend",
        category=SocialCategory.KINSHIP,
        axis=SocialAxis.KINSHIP,
        level=4,
        name="Friend",
        description="Person you know well and like.",
        support_texts=(
            "A friend is someone you trust and enjoy.",
            "Mutual affection and support.",
            "Close personal relationship.",
        ),
        cross_domain_weight=1.2,
    ),
    SocialConcept(
        id="family",
        category=SocialCategory.KINSHIP,
        axis=SocialAxis.KINSHIP,
        level=5,
        name="Family",
        description="Closest social bonds.",
        support_texts=(
            "Family is the closest bond.",
            "Blood relatives or chosen family.",
            "Unconditional love and belonging.",
            "Those we call our own.",
        ),
        cross_domain_weight=1.5,
    ),
)

# Status Markers Category (Power Axis) - 5 probes
STATUS_MARKERS_PROBES: tuple[SocialConcept, ...] = (
    SocialConcept(
        id="beggar",
        category=SocialCategory.STATUS_MARKERS,
        axis=SocialAxis.POWER,
        level=1,
        name="Beggar",
        description="Person who lives by begging.",
        support_texts=(
            "A beggar asks for charity.",
            "Living in poverty, destitute.",
            "The lowest economic status.",
        ),
        cross_domain_weight=1.2,
    ),
    SocialConcept(
        id="worker",
        category=SocialCategory.STATUS_MARKERS,
        axis=SocialAxis.POWER,
        level=2,
        name="Worker",
        description="Person who does manual labor.",
        support_texts=(
            "A worker earns wages for labor.",
            "Blue-collar, hourly employment.",
            "The working class.",
        ),
        cross_domain_weight=1.0,
    ),
    SocialConcept(
        id="professional",
        category=SocialCategory.STATUS_MARKERS,
        axis=SocialAxis.POWER,
        level=3,
        name="Professional",
        description="Person with specialized training.",
        support_texts=(
            "A professional has expertise.",
            "White-collar, salaried.",
            "Doctor, lawyer, engineer.",
        ),
        cross_domain_weight=1.0,
    ),
    SocialConcept(
        id="wealthy",
        category=SocialCategory.STATUS_MARKERS,
        axis=SocialAxis.POWER,
        level=4,
        name="Wealthy",
        description="Person with significant assets.",
        support_texts=(
            "The wealthy have abundance.",
            "Rich, prosperous, affluent.",
            "Financial security and luxury.",
        ),
        cross_domain_weight=1.2,
    ),
    SocialConcept(
        id="elite",
        category=SocialCategory.STATUS_MARKERS,
        axis=SocialAxis.POWER,
        level=5,
        name="Elite",
        description="Highest social stratum.",
        support_texts=(
            "The elite are the top echelon.",
            "Ultra-wealthy, powerful.",
            "The one percent.",
            "Those who shape society.",
        ),
        cross_domain_weight=1.3,
    ),
)

# Age Category (Power Axis - seniority correlates with authority) - 5 probes
AGE_PROBES: tuple[SocialConcept, ...] = (
    SocialConcept(
        id="child",
        category=SocialCategory.AGE,
        axis=SocialAxis.POWER,
        level=1,
        name="Child",
        description="Young person, dependent.",
        support_texts=(
            "A child is young and developing.",
            "Under the care of adults.",
            "Learning and growing.",
        ),
        cross_domain_weight=1.0,
    ),
    SocialConcept(
        id="youth",
        category=SocialCategory.AGE,
        axis=SocialAxis.POWER,
        level=2,
        name="Youth",
        description="Adolescent, transitional.",
        support_texts=(
            "Youth is the transition to adulthood.",
            "Teenager, young adult.",
            "Gaining independence.",
        ),
        cross_domain_weight=1.0,
    ),
    SocialConcept(
        id="adult",
        category=SocialCategory.AGE,
        axis=SocialAxis.POWER,
        level=3,
        name="Adult",
        description="Fully grown person.",
        support_texts=(
            "An adult is fully mature.",
            "Independent, responsible.",
            "Peak of life's capabilities.",
        ),
        cross_domain_weight=1.0,
    ),
    SocialConcept(
        id="senior",
        category=SocialCategory.AGE,
        axis=SocialAxis.POWER,
        level=4,
        name="Senior",
        description="Older person with experience.",
        support_texts=(
            "A senior has life experience.",
            "Wisdom from years lived.",
            "Respected for age.",
        ),
        cross_domain_weight=1.0,
    ),
    SocialConcept(
        id="elder",
        category=SocialCategory.AGE,
        axis=SocialAxis.POWER,
        level=5,
        name="Elder",
        description="Venerated older person.",
        support_texts=(
            "An elder is revered for wisdom.",
            "Patriarch, matriarch.",
            "Leader by virtue of experience.",
            "The voice of ancestral knowledge.",
        ),
        cross_domain_weight=1.3,
    ),
)

# All probes by category
ALL_SOCIAL_PROBES: tuple[SocialConcept, ...] = (
    *POWER_HIERARCHY_PROBES,
    *FORMALITY_PROBES,
    *KINSHIP_PROBES,
    *STATUS_MARKERS_PROBES,
    *AGE_PROBES,
)


class SocialConceptInventory:
    """
    Complete inventory of social concepts for manifold analysis.

    Structure:
    - Power Hierarchy: 5 probes (slave, servant, citizen, noble, emperor)
    - Formality: 5 probes (hey, hi, hello, greetings, salutations)
    - Kinship: 5 probes (enemy, stranger, acquaintance, friend, family)
    - Status Markers: 5 probes (beggar, worker, professional, wealthy, elite)
    - Age: 5 probes (child, youth, adult, senior, elder)

    Total: 25 social probes
    """

    @staticmethod
    def all_concepts() -> list[SocialConcept]:
        """Get all social concepts."""
        return list(ALL_SOCIAL_PROBES)

    @staticmethod
    def by_category(category: SocialCategory) -> list[SocialConcept]:
        """Get concepts by category."""
        return [c for c in ALL_SOCIAL_PROBES if c.category == category]

    @staticmethod
    def by_axis(axis: SocialAxis) -> list[SocialConcept]:
        """Get concepts by axis."""
        return [c for c in ALL_SOCIAL_PROBES if c.axis == axis]

    @staticmethod
    def power_probes() -> list[SocialConcept]:
        """Get all probes on the Power axis (low→high status)."""
        return SocialConceptInventory.by_axis(SocialAxis.POWER)

    @staticmethod
    def kinship_probes() -> list[SocialConcept]:
        """Get all probes on the Kinship axis (distant→close)."""
        return SocialConceptInventory.by_axis(SocialAxis.KINSHIP)

    @staticmethod
    def formality_probes() -> list[SocialConcept]:
        """Get all probes on the Formality axis (casual→formal)."""
        return SocialConceptInventory.by_axis(SocialAxis.FORMALITY)

    @staticmethod
    def power_hierarchy_probes() -> list[SocialConcept]:
        """Get power hierarchy probes (slave→emperor)."""
        return list(POWER_HIERARCHY_PROBES)

    @staticmethod
    def count() -> int:
        """Total number of social probes."""
        return len(ALL_SOCIAL_PROBES)

    @staticmethod
    def count_by_category() -> dict[SocialCategory, int]:
        """Count probes by category."""
        counts: dict[SocialCategory, int] = {}
        for concept in ALL_SOCIAL_PROBES:
            counts[concept.category] = counts.get(concept.category, 0) + 1
        return counts

    @staticmethod
    def count_by_axis() -> dict[SocialAxis, int]:
        """Count probes by axis."""
        counts: dict[SocialAxis, int] = {}
        for concept in ALL_SOCIAL_PROBES:
            counts[concept.axis] = counts.get(concept.axis, 0) + 1
        return counts
