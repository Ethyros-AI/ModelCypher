# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Moral Concepts - Domain model for moral geometry analysis.

Provides MoralConcept dataclass with axis, foundation, and level
metadata required for moral geometry analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MoralAxis(str, Enum):
    """Moral reasoning axis (orthogonal dimensions of moral space)."""

    VALENCE = "valence"  # evil → good
    AGENCY = "agency"  # victim → perpetrator
    SCOPE = "scope"  # self → universal


class MoralFoundation(str, Enum):
    """Moral Foundations Theory categories (Haidt, 2012)."""

    CARE_HARM = "care_harm"
    FAIRNESS_CHEATING = "fairness_cheating"
    SANCTITY_DEGRADATION = "sanctity_degradation"
    LIBERTY_OPPRESSION = "liberty_oppression"


@dataclass(frozen=True)
class MoralConcept:
    """
    Moral concept for geometry analysis.

    Satisfies MoralConceptProtocol with:
    - id: Unique identifier
    - name: Human-readable name
    - axis: MoralAxis (valence, agency, scope)
    - foundation: MoralFoundation
    - level: Position on gradient (-2 to +2 typically)
    - description: Brief description
    - support_texts: Example sentences for embedding
    """

    id: str
    name: str
    axis: MoralAxis
    foundation: MoralFoundation
    level: int
    description: str
    support_texts: tuple[str, ...]


# Define the moral concept inventory
# Based on Moral Foundations Theory with gradient levels
_MORAL_CONCEPTS = [
    # CARE/HARM foundation - VALENCE axis
    MoralConcept(
        id="cruelty",
        name="Cruelty",
        axis=MoralAxis.VALENCE,
        foundation=MoralFoundation.CARE_HARM,
        level=-2,
        description="Deliberate infliction of suffering.",
        support_texts=(
            "Cruelty is the willful causing of pain.",
            "To be cruel is to delight in suffering.",
            "Cruelty shows disregard for others' wellbeing.",
        ),
    ),
    MoralConcept(
        id="neglect",
        name="Neglect",
        axis=MoralAxis.VALENCE,
        foundation=MoralFoundation.CARE_HARM,
        level=-1,
        description="Failure to provide care.",
        support_texts=(
            "Neglect is the absence of care.",
            "To neglect is to ignore needs.",
            "Passive harm through inaction.",
        ),
    ),
    MoralConcept(
        id="indifference",
        name="Indifference",
        axis=MoralAxis.VALENCE,
        foundation=MoralFoundation.CARE_HARM,
        level=0,
        description="Lack of concern for others.",
        support_texts=(
            "Indifference is neither caring nor harming.",
            "Neutral stance toward suffering.",
            "The absence of moral engagement.",
        ),
    ),
    MoralConcept(
        id="kindness",
        name="Kindness",
        axis=MoralAxis.VALENCE,
        foundation=MoralFoundation.CARE_HARM,
        level=1,
        description="Gentle concern for others.",
        support_texts=(
            "Kindness is gentle care for others.",
            "To be kind is to show compassion.",
            "Small acts of consideration.",
        ),
    ),
    MoralConcept(
        id="compassion",
        name="Compassion",
        axis=MoralAxis.VALENCE,
        foundation=MoralFoundation.CARE_HARM,
        level=2,
        description="Deep empathy and care for suffering.",
        support_texts=(
            "Compassion is suffering with others.",
            "Deep empathy that motivates action.",
            "The highest expression of care.",
        ),
    ),
    # FAIRNESS/CHEATING foundation - VALENCE axis
    MoralConcept(
        id="exploitation",
        name="Exploitation",
        axis=MoralAxis.VALENCE,
        foundation=MoralFoundation.FAIRNESS_CHEATING,
        level=-2,
        description="Taking unfair advantage of others.",
        support_texts=(
            "Exploitation is using others unfairly.",
            "To exploit is to take advantage.",
            "Treating people as means only.",
        ),
    ),
    MoralConcept(
        id="cheating",
        name="Cheating",
        axis=MoralAxis.VALENCE,
        foundation=MoralFoundation.FAIRNESS_CHEATING,
        level=-1,
        description="Breaking rules for personal gain.",
        support_texts=(
            "Cheating is breaking agreed rules.",
            "Gaining advantage through deception.",
            "Violation of fair play.",
        ),
    ),
    MoralConcept(
        id="impartiality",
        name="Impartiality",
        axis=MoralAxis.VALENCE,
        foundation=MoralFoundation.FAIRNESS_CHEATING,
        level=0,
        description="Treating all equally without bias.",
        support_texts=(
            "Impartiality is equal treatment.",
            "Neither favoring nor disfavoring.",
            "The neutral stance of fairness.",
        ),
    ),
    MoralConcept(
        id="fairness",
        name="Fairness",
        axis=MoralAxis.VALENCE,
        foundation=MoralFoundation.FAIRNESS_CHEATING,
        level=1,
        description="Just and equitable treatment.",
        support_texts=(
            "Fairness is giving each their due.",
            "Just treatment according to merit.",
            "Equitable distribution of goods.",
        ),
    ),
    MoralConcept(
        id="justice",
        name="Justice",
        axis=MoralAxis.VALENCE,
        foundation=MoralFoundation.FAIRNESS_CHEATING,
        level=2,
        description="Righteous judgment and equity.",
        support_texts=(
            "Justice is giving each their due.",
            "The highest principle of fairness.",
            "Righting wrongs and restoring balance.",
        ),
    ),
    # SANCTITY/DEGRADATION foundation - VALENCE axis
    MoralConcept(
        id="defilement",
        name="Defilement",
        axis=MoralAxis.VALENCE,
        foundation=MoralFoundation.SANCTITY_DEGRADATION,
        level=-2,
        description="Corruption of what is sacred.",
        support_texts=(
            "Defilement is making impure.",
            "Violation of the sacred.",
            "Spiritual contamination.",
        ),
    ),
    MoralConcept(
        id="degradation",
        name="Degradation",
        axis=MoralAxis.VALENCE,
        foundation=MoralFoundation.SANCTITY_DEGRADATION,
        level=-1,
        description="Lowering of dignity or worth.",
        support_texts=(
            "Degradation is reducing worth.",
            "Treating the noble as base.",
            "Loss of dignity.",
        ),
    ),
    MoralConcept(
        id="mundane",
        name="Mundane",
        axis=MoralAxis.VALENCE,
        foundation=MoralFoundation.SANCTITY_DEGRADATION,
        level=0,
        description="Neither sacred nor profane.",
        support_texts=(
            "The mundane is the ordinary.",
            "Neither elevated nor degraded.",
            "The everyday world.",
        ),
    ),
    MoralConcept(
        id="purity",
        name="Purity",
        axis=MoralAxis.VALENCE,
        foundation=MoralFoundation.SANCTITY_DEGRADATION,
        level=1,
        description="Cleanness and wholesomeness.",
        support_texts=(
            "Purity is freedom from contamination.",
            "Clean body, clean mind.",
            "Wholesome and untainted.",
        ),
    ),
    MoralConcept(
        id="sanctity",
        name="Sanctity",
        axis=MoralAxis.VALENCE,
        foundation=MoralFoundation.SANCTITY_DEGRADATION,
        level=2,
        description="Sacred and inviolable holiness.",
        support_texts=(
            "Sanctity is the sacred.",
            "That which must not be violated.",
            "The holy and inviolable.",
        ),
    ),
    # LIBERTY/OPPRESSION foundation - AGENCY axis
    MoralConcept(
        id="tyranny",
        name="Tyranny",
        axis=MoralAxis.AGENCY,
        foundation=MoralFoundation.LIBERTY_OPPRESSION,
        level=-2,
        description="Cruel and oppressive rule.",
        support_texts=(
            "Tyranny is rule by fear.",
            "Oppressive domination.",
            "The crushing of freedom.",
        ),
    ),
    MoralConcept(
        id="oppression",
        name="Oppression",
        axis=MoralAxis.AGENCY,
        foundation=MoralFoundation.LIBERTY_OPPRESSION,
        level=-1,
        description="Unjust exercise of power.",
        support_texts=(
            "Oppression is unjust constraint.",
            "Keeping others down.",
            "Systematic denial of rights.",
        ),
    ),
    MoralConcept(
        id="constraint",
        name="Constraint",
        axis=MoralAxis.AGENCY,
        foundation=MoralFoundation.LIBERTY_OPPRESSION,
        level=0,
        description="Necessary limitation of freedom.",
        support_texts=(
            "Constraint is bounded freedom.",
            "Neither free nor enslaved.",
            "Reasonable limits.",
        ),
    ),
    MoralConcept(
        id="freedom",
        name="Freedom",
        axis=MoralAxis.AGENCY,
        foundation=MoralFoundation.LIBERTY_OPPRESSION,
        level=1,
        description="Absence of unjust constraint.",
        support_texts=(
            "Freedom is self-determination.",
            "Liberty to act as one chooses.",
            "Absence of oppression.",
        ),
    ),
    MoralConcept(
        id="liberation",
        name="Liberation",
        axis=MoralAxis.AGENCY,
        foundation=MoralFoundation.LIBERTY_OPPRESSION,
        level=2,
        description="Complete freedom from oppression.",
        support_texts=(
            "Liberation is breaking chains.",
            "Complete emancipation.",
            "Freedom for all.",
        ),
    ),
]


class MoralConceptInventory:
    """Inventory of moral concepts for geometry analysis."""

    _cached: list[MoralConcept] | None = None

    @classmethod
    def all_concepts(cls) -> list[MoralConcept]:
        """Get all moral concepts."""
        if cls._cached is None:
            cls._cached = list(_MORAL_CONCEPTS)
        return list(cls._cached)

    @classmethod
    def by_foundation(cls, foundation: MoralFoundation) -> list[MoralConcept]:
        """Get concepts for a specific foundation."""
        return [c for c in cls.all_concepts() if c.foundation == foundation]

    @classmethod
    def by_axis(cls, axis: MoralAxis) -> list[MoralConcept]:
        """Get concepts for a specific axis."""
        return [c for c in cls.all_concepts() if c.axis == axis]

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the concept cache."""
        cls._cached = None
