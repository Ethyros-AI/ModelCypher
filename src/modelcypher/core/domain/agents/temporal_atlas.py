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
Temporal Atlas.

Multi-domain probes for triangulating temporal structure in LLM representations.
Tests the "Latent Chronologist" hypothesis: models encode time as a coherent
geometric manifold with Direction (past→future), Duration (moment→eternity),
and Causality (cause→effect) axes.

This module provides 25 temporal probes across 5 categories organized along
Direction/Duration/Causality axis structure (triangulable). It enables Arrow of
Time detection via monotonic gradient past→future, and tests causal asymmetry
(because ≠ therefore).

Notes
-----
Scientific basis:
- Narrative structure (stories have beginning/middle/end)
- Causal reasoning in text (because/therefore asymmetry)
- Temporal deixis (yesterday/today/tomorrow ordering)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TemporalCategory(str, Enum):
    """Categories of temporal probes."""

    TENSE = "tense"  # Past/present/future deixis
    DURATION = "duration"  # Temporal extent (moment→eternity)
    CAUSALITY = "causality"  # Cause→effect relationships
    LIFECYCLE = "lifecycle"  # Birth→death trajectory
    SEQUENCE = "sequence"  # Ordering (first, then, finally)


class TemporalAxis(str, Enum):
    """Axes of the temporal manifold."""

    DIRECTION = "direction"  # Past (-1) to Future (+1)
    DURATION = "duration"  # Short (-1) to Long (+1)
    CAUSALITY = "causality"  # Cause (-1) to Effect (+1)


@dataclass(frozen=True)
class TemporalConcept:
    """A temporal probe for manifold analysis."""

    id: str
    category: TemporalCategory
    axis: TemporalAxis
    level: int  # 1-5 ordering within axis (1=low/past, 5=high/future)
    name: str
    description: str
    support_texts: tuple[str, ...]
    cross_domain_weight: float = 1.0

    @property
    def canonical_name(self) -> str:
        return self.name

    @property
    def prompt(self) -> str:
        return f"The word {self.name.lower()} represents"


# Tense Category (Direction Axis) - 5 probes
TENSE_PROBES: tuple[TemporalConcept, ...] = (
    TemporalConcept(
        id="past",
        category=TemporalCategory.TENSE,
        axis=TemporalAxis.DIRECTION,
        level=1,
        name="Past",
        description="Time before the present moment.",
        support_texts=(
            "The past is what has already happened.",
            "Looking back at what was.",
            "Events that occurred before now.",
            "History and memory.",
        ),
        cross_domain_weight=1.5,
    ),
    TemporalConcept(
        id="yesterday",
        category=TemporalCategory.TENSE,
        axis=TemporalAxis.DIRECTION,
        level=2,
        name="Yesterday",
        description="The day before today.",
        support_texts=(
            "Yesterday was the day before today.",
            "What happened 24 hours ago.",
            "The recent past, just behind us.",
        ),
        cross_domain_weight=1.2,
    ),
    TemporalConcept(
        id="today",
        category=TemporalCategory.TENSE,
        axis=TemporalAxis.DIRECTION,
        level=3,
        name="Today",
        description="The current day, the present.",
        support_texts=(
            "Today is the present moment.",
            "Right now, this very day.",
            "The current time period.",
        ),
        cross_domain_weight=1.0,
    ),
    TemporalConcept(
        id="tomorrow",
        category=TemporalCategory.TENSE,
        axis=TemporalAxis.DIRECTION,
        level=4,
        name="Tomorrow",
        description="The day after today.",
        support_texts=(
            "Tomorrow is the day after today.",
            "What will happen in 24 hours.",
            "The near future, just ahead.",
        ),
        cross_domain_weight=1.2,
    ),
    TemporalConcept(
        id="future",
        category=TemporalCategory.TENSE,
        axis=TemporalAxis.DIRECTION,
        level=5,
        name="Future",
        description="Time after the present moment.",
        support_texts=(
            "The future is what will happen.",
            "Looking forward to what will be.",
            "Events that have not yet occurred.",
            "Anticipation and prediction.",
        ),
        cross_domain_weight=1.5,
    ),
)

# Duration Category (Duration Axis) - 5 probes
DURATION_PROBES: tuple[TemporalConcept, ...] = (
    TemporalConcept(
        id="moment",
        category=TemporalCategory.DURATION,
        axis=TemporalAxis.DURATION,
        level=1,
        name="Moment",
        description="An extremely brief period of time.",
        support_texts=(
            "A moment is a brief instant.",
            "Just a split second.",
            "The blink of an eye.",
            "A fleeting instant.",
        ),
        cross_domain_weight=1.2,
    ),
    TemporalConcept(
        id="hour",
        category=TemporalCategory.DURATION,
        axis=TemporalAxis.DURATION,
        level=2,
        name="Hour",
        description="A period of 60 minutes.",
        support_texts=(
            "An hour is 60 minutes.",
            "Enough time for a meeting.",
            "A short but measurable duration.",
        ),
        cross_domain_weight=1.0,
    ),
    TemporalConcept(
        id="day",
        category=TemporalCategory.DURATION,
        axis=TemporalAxis.DURATION,
        level=3,
        name="Day",
        description="A period of 24 hours.",
        support_texts=(
            "A day is 24 hours.",
            "From sunrise to sunrise.",
            "One full rotation of Earth.",
        ),
        cross_domain_weight=1.0,
    ),
    TemporalConcept(
        id="year",
        category=TemporalCategory.DURATION,
        axis=TemporalAxis.DURATION,
        level=4,
        name="Year",
        description="A period of 365 days.",
        support_texts=(
            "A year is about 365 days.",
            "One full orbit around the sun.",
            "Four seasons passing.",
            "Twelve months.",
        ),
        cross_domain_weight=1.0,
    ),
    TemporalConcept(
        id="century",
        category=TemporalCategory.DURATION,
        axis=TemporalAxis.DURATION,
        level=5,
        name="Century",
        description="A period of 100 years.",
        support_texts=(
            "A century is 100 years.",
            "Several generations of humans.",
            "A long span of history.",
            "Major civilizational changes.",
        ),
        cross_domain_weight=1.2,
    ),
)

# Causality Category (Causality Axis) - 5 probes
CAUSALITY_PROBES: tuple[TemporalConcept, ...] = (
    TemporalConcept(
        id="because",
        category=TemporalCategory.CAUSALITY,
        axis=TemporalAxis.CAUSALITY,
        level=1,
        name="Because",
        description="Indicating the cause or reason.",
        support_texts=(
            "Because explains why something happened.",
            "The reason behind an event.",
            "Due to, owing to.",
            "The causal antecedent.",
        ),
        cross_domain_weight=1.5,
    ),
    TemporalConcept(
        id="causes",
        category=TemporalCategory.CAUSALITY,
        axis=TemporalAxis.CAUSALITY,
        level=2,
        name="Causes",
        description="The act of bringing about an effect.",
        support_texts=(
            "X causes Y means X brings about Y.",
            "The mechanism of causation.",
            "Making something happen.",
        ),
        cross_domain_weight=1.2,
    ),
    TemporalConcept(
        id="leads_to",
        category=TemporalCategory.CAUSALITY,
        axis=TemporalAxis.CAUSALITY,
        level=3,
        name="Leads to",
        description="Resulting in or causing eventually.",
        support_texts=(
            "One thing leads to another.",
            "A chain of events unfolds.",
            "The path from cause to effect.",
        ),
        cross_domain_weight=1.0,
    ),
    TemporalConcept(
        id="therefore",
        category=TemporalCategory.CAUSALITY,
        axis=TemporalAxis.CAUSALITY,
        level=4,
        name="Therefore",
        description="For that reason, consequently.",
        support_texts=(
            "Therefore indicates a conclusion.",
            "As a result, consequently.",
            "The logical consequence.",
            "It follows that.",
        ),
        cross_domain_weight=1.5,
    ),
    TemporalConcept(
        id="results_in",
        category=TemporalCategory.CAUSALITY,
        axis=TemporalAxis.CAUSALITY,
        level=5,
        name="Results in",
        description="The outcome or effect produced.",
        support_texts=(
            "The action results in a consequence.",
            "The end state after causation.",
            "The effect that follows.",
        ),
        cross_domain_weight=1.2,
    ),
)

# Lifecycle Category (Direction Axis) - 5 probes
LIFECYCLE_PROBES: tuple[TemporalConcept, ...] = (
    TemporalConcept(
        id="birth",
        category=TemporalCategory.LIFECYCLE,
        axis=TemporalAxis.DIRECTION,
        level=1,
        name="Birth",
        description="The beginning of life.",
        support_texts=(
            "Birth is the start of existence.",
            "Coming into the world.",
            "The origin of a being.",
            "A new life begins.",
        ),
        cross_domain_weight=1.3,
    ),
    TemporalConcept(
        id="childhood",
        category=TemporalCategory.LIFECYCLE,
        axis=TemporalAxis.DIRECTION,
        level=2,
        name="Childhood",
        description="The early period of life.",
        support_texts=(
            "Childhood is the early years.",
            "Growing up, learning, playing.",
            "Before adulthood.",
        ),
        cross_domain_weight=1.0,
    ),
    TemporalConcept(
        id="adulthood",
        category=TemporalCategory.LIFECYCLE,
        axis=TemporalAxis.DIRECTION,
        level=3,
        name="Adulthood",
        description="The mature period of life.",
        support_texts=(
            "Adulthood is full maturity.",
            "Responsibility and independence.",
            "The prime of life.",
        ),
        cross_domain_weight=1.0,
    ),
    TemporalConcept(
        id="elderly",
        category=TemporalCategory.LIFECYCLE,
        axis=TemporalAxis.DIRECTION,
        level=4,
        name="Elderly",
        description="The late period of life.",
        support_texts=(
            "The elderly are in late life.",
            "Old age, wisdom, experience.",
            "The twilight years.",
        ),
        cross_domain_weight=1.0,
    ),
    TemporalConcept(
        id="death",
        category=TemporalCategory.LIFECYCLE,
        axis=TemporalAxis.DIRECTION,
        level=5,
        name="Death",
        description="The end of life.",
        support_texts=(
            "Death is the end of existence.",
            "Leaving the world.",
            "The final moment.",
            "Life's conclusion.",
        ),
        cross_domain_weight=1.3,
    ),
)

# Sequence Category (Direction Axis) - 5 probes
SEQUENCE_PROBES: tuple[TemporalConcept, ...] = (
    TemporalConcept(
        id="beginning",
        category=TemporalCategory.SEQUENCE,
        axis=TemporalAxis.DIRECTION,
        level=1,
        name="Beginning",
        description="The first part of something.",
        support_texts=(
            "The beginning is where it starts.",
            "The opening, the first chapter.",
            "Once upon a time.",
            "In the beginning.",
        ),
        cross_domain_weight=1.3,
    ),
    TemporalConcept(
        id="first",
        category=TemporalCategory.SEQUENCE,
        axis=TemporalAxis.DIRECTION,
        level=2,
        name="First",
        description="Coming before all others.",
        support_texts=(
            "First means before everything else.",
            "The initial item or event.",
            "Number one in order.",
        ),
        cross_domain_weight=1.0,
    ),
    TemporalConcept(
        id="middle",
        category=TemporalCategory.SEQUENCE,
        axis=TemporalAxis.DIRECTION,
        level=3,
        name="Middle",
        description="The central part of something.",
        support_texts=(
            "The middle is between beginning and end.",
            "The central portion.",
            "Halfway through.",
        ),
        cross_domain_weight=1.0,
    ),
    TemporalConcept(
        id="last",
        category=TemporalCategory.SEQUENCE,
        axis=TemporalAxis.DIRECTION,
        level=4,
        name="Last",
        description="Coming after all others.",
        support_texts=(
            "Last means after everything else.",
            "The final item or event.",
            "At the end of the sequence.",
        ),
        cross_domain_weight=1.0,
    ),
    TemporalConcept(
        id="ending",
        category=TemporalCategory.SEQUENCE,
        axis=TemporalAxis.DIRECTION,
        level=5,
        name="Ending",
        description="The final part of something.",
        support_texts=(
            "The ending is where it concludes.",
            "The closing, the final chapter.",
            "And they lived happily ever after.",
            "The end.",
        ),
        cross_domain_weight=1.3,
    ),
)

# All probes by category
ALL_TEMPORAL_PROBES: tuple[TemporalConcept, ...] = (
    *TENSE_PROBES,
    *DURATION_PROBES,
    *CAUSALITY_PROBES,
    *LIFECYCLE_PROBES,
    *SEQUENCE_PROBES,
)


class TemporalConceptInventory:
    """
    Complete inventory of temporal concepts for manifold analysis.

    Structure:
    - Tense: 5 probes (past, yesterday, today, tomorrow, future)
    - Duration: 5 probes (moment, hour, day, year, century)
    - Causality: 5 probes (because, causes, leads_to, therefore, results_in)
    - Lifecycle: 5 probes (birth, childhood, adulthood, elderly, death)
    - Sequence: 5 probes (beginning, first, middle, last, ending)

    Total: 25 temporal probes
    """

    @staticmethod
    def all_concepts() -> list[TemporalConcept]:
        """Get all temporal concepts."""
        return list(ALL_TEMPORAL_PROBES)

    @staticmethod
    def by_category(category: TemporalCategory) -> list[TemporalConcept]:
        """Get concepts by category."""
        return [c for c in ALL_TEMPORAL_PROBES if c.category == category]

    @staticmethod
    def by_axis(axis: TemporalAxis) -> list[TemporalConcept]:
        """Get concepts by axis."""
        return [c for c in ALL_TEMPORAL_PROBES if c.axis == axis]

    @staticmethod
    def direction_probes() -> list[TemporalConcept]:
        """Get all probes on the Direction axis (past→future)."""
        return TemporalConceptInventory.by_axis(TemporalAxis.DIRECTION)

    @staticmethod
    def duration_probes() -> list[TemporalConcept]:
        """Get all probes on the Duration axis (moment→century)."""
        return TemporalConceptInventory.by_axis(TemporalAxis.DURATION)

    @staticmethod
    def causality_probes() -> list[TemporalConcept]:
        """Get all probes on the Causality axis (because→therefore)."""
        return TemporalConceptInventory.by_axis(TemporalAxis.CAUSALITY)

    @staticmethod
    def tense_probes() -> list[TemporalConcept]:
        """Get tense probes (deixis)."""
        return list(TENSE_PROBES)

    @staticmethod
    def lifecycle_probes() -> list[TemporalConcept]:
        """Get lifecycle probes (birth→death)."""
        return list(LIFECYCLE_PROBES)

    @staticmethod
    def sequence_probes() -> list[TemporalConcept]:
        """Get sequence probes (beginning→ending)."""
        return list(SEQUENCE_PROBES)

    @staticmethod
    def count() -> int:
        """Total number of temporal probes."""
        return len(ALL_TEMPORAL_PROBES)

    @staticmethod
    def count_by_category() -> dict[TemporalCategory, int]:
        """Count probes by category."""
        counts: dict[TemporalCategory, int] = {}
        for concept in ALL_TEMPORAL_PROBES:
            counts[concept.category] = counts.get(concept.category, 0) + 1
        return counts

    @staticmethod
    def count_by_axis() -> dict[TemporalAxis, int]:
        """Count probes by axis."""
        counts: dict[TemporalAxis, int] = {}
        for concept in ALL_TEMPORAL_PROBES:
            counts[concept.axis] = counts.get(concept.axis, 0) + 1
        return counts
