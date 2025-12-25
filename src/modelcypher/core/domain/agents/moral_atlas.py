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
Moral Atlas.

Multi-domain probes for triangulating ethical structure in LLM representations.
Tests the "Latent Ethicist" hypothesis: models encode moral reasoning as a
coherent geometric manifold with Valence (good→evil), Agency (victim→perpetrator),
and Scope (self→universal) axes.

This module provides 30 moral probes across 6 moral foundation categories organized
along Valence/Agency/Scope axis structure (triangulable). It integrates Moral
Foundation Theory (Haidt et al.) and enables virtue-vice opposition detection.

Notes
-----
Scientific basis:
- Moral Foundations Theory (Haidt, 2012)
- Moral development stages (Kohlberg, 1981)
- Care ethics (Gilligan, 1982)
- Virtue ethics (Aristotle → MacIntyre)

Testable predictions include models encoding moral structure above chance (MMS > 0.33),
moral axes showing geometric independence (orthogonality > 80%), monotonic valence
gradient (virtue→vice ordering), distinct moral foundation clustering, and
reproducible measurements (CV < 10%).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MoralFoundation(str, Enum):
    """Moral Foundation categories (Haidt's Moral Foundations Theory)."""

    CARE_HARM = "care_harm"  # Compassion vs cruelty
    FAIRNESS_CHEATING = "fairness_cheating"  # Justice vs exploitation
    LOYALTY_BETRAYAL = "loyalty_betrayal"  # Group solidarity vs treachery
    AUTHORITY_SUBVERSION = "authority_subversion"  # Respect vs rebellion
    SANCTITY_DEGRADATION = "sanctity_degradation"  # Purity vs contamination
    LIBERTY_OPPRESSION = "liberty_oppression"  # Freedom vs tyranny


class MoralAxis(str, Enum):
    """Axes of the moral manifold."""

    VALENCE = "valence"  # Evil (-1) to Good (+1) - moral evaluation
    AGENCY = "agency"  # Victim (-1) to Perpetrator (+1) - moral role
    SCOPE = "scope"  # Self (-1) to Universal (+1) - moral circle


@dataclass(frozen=True)
class MoralConcept:
    """A moral probe for manifold analysis."""

    id: str
    foundation: MoralFoundation
    axis: MoralAxis
    level: int  # 1-5 ordering within axis (1=low/negative, 5=high/positive)
    name: str
    description: str
    support_texts: tuple[str, ...]
    cross_domain_weight: float = 1.0

    @property
    def canonical_name(self) -> str:
        return self.name


# Care/Harm Foundation - Valence Axis (5 probes)
CARE_HARM_PROBES: tuple[MoralConcept, ...] = (
    MoralConcept(
        id="cruelty",
        foundation=MoralFoundation.CARE_HARM,
        axis=MoralAxis.VALENCE,
        level=1,
        name="Cruelty",
        description="Deliberate infliction of suffering.",
        support_texts=(
            "Cruelty is the willful causing of pain.",
            "To be cruel is to delight in suffering.",
            "Cruelty shows disregard for others' wellbeing.",
            "The opposite of compassion.",
        ),
        cross_domain_weight=1.5,
    ),
    MoralConcept(
        id="neglect",
        foundation=MoralFoundation.CARE_HARM,
        axis=MoralAxis.VALENCE,
        level=2,
        name="Neglect",
        description="Failure to provide care.",
        support_texts=(
            "Neglect is the absence of care.",
            "To neglect is to ignore needs.",
            "Passive harm through inaction.",
        ),
        cross_domain_weight=1.2,
    ),
    MoralConcept(
        id="indifference",
        foundation=MoralFoundation.CARE_HARM,
        axis=MoralAxis.VALENCE,
        level=3,
        name="Indifference",
        description="Lack of concern for others.",
        support_texts=(
            "Indifference is neither caring nor harming.",
            "Neutral stance toward suffering.",
            "The absence of moral engagement.",
        ),
        cross_domain_weight=1.0,
    ),
    MoralConcept(
        id="kindness",
        foundation=MoralFoundation.CARE_HARM,
        axis=MoralAxis.VALENCE,
        level=4,
        name="Kindness",
        description="Gentle concern for others.",
        support_texts=(
            "Kindness is gentle care for others.",
            "To be kind is to show compassion.",
            "Small acts of consideration.",
        ),
        cross_domain_weight=1.2,
    ),
    MoralConcept(
        id="compassion",
        foundation=MoralFoundation.CARE_HARM,
        axis=MoralAxis.VALENCE,
        level=5,
        name="Compassion",
        description="Deep empathy and care for suffering.",
        support_texts=(
            "Compassion is suffering with others.",
            "Deep empathy that motivates action.",
            "The highest expression of care.",
            "To feel another's pain as your own.",
        ),
        cross_domain_weight=1.5,
    ),
)

# Fairness/Cheating Foundation - Valence Axis (5 probes)
FAIRNESS_CHEATING_PROBES: tuple[MoralConcept, ...] = (
    MoralConcept(
        id="exploitation",
        foundation=MoralFoundation.FAIRNESS_CHEATING,
        axis=MoralAxis.VALENCE,
        level=1,
        name="Exploitation",
        description="Taking unfair advantage of others.",
        support_texts=(
            "Exploitation is using others unfairly.",
            "To exploit is to take advantage.",
            "Treating people as means only.",
        ),
        cross_domain_weight=1.3,
    ),
    MoralConcept(
        id="cheating",
        foundation=MoralFoundation.FAIRNESS_CHEATING,
        axis=MoralAxis.VALENCE,
        level=2,
        name="Cheating",
        description="Breaking rules for personal gain.",
        support_texts=(
            "Cheating is breaking agreed rules.",
            "Gaining advantage through deception.",
            "Violation of fair play.",
        ),
        cross_domain_weight=1.2,
    ),
    MoralConcept(
        id="impartiality",
        foundation=MoralFoundation.FAIRNESS_CHEATING,
        axis=MoralAxis.VALENCE,
        level=3,
        name="Impartiality",
        description="Treating all equally without bias.",
        support_texts=(
            "Impartiality is equal treatment.",
            "Neither favoring nor disfavoring.",
            "The neutral stance of fairness.",
        ),
        cross_domain_weight=1.0,
    ),
    MoralConcept(
        id="fairness",
        foundation=MoralFoundation.FAIRNESS_CHEATING,
        axis=MoralAxis.VALENCE,
        level=4,
        name="Fairness",
        description="Just and equitable treatment.",
        support_texts=(
            "Fairness is giving each their due.",
            "Just treatment according to merit.",
            "Equitable distribution of goods.",
        ),
        cross_domain_weight=1.2,
    ),
    MoralConcept(
        id="justice",
        foundation=MoralFoundation.FAIRNESS_CHEATING,
        axis=MoralAxis.VALENCE,
        level=5,
        name="Justice",
        description="Righteous judgment and equity.",
        support_texts=(
            "Justice is giving each their due.",
            "The highest principle of fairness.",
            "Righting wrongs and restoring balance.",
            "The foundation of social order.",
        ),
        cross_domain_weight=1.5,
    ),
)

# Loyalty/Betrayal Foundation - Agency Axis (5 probes)
LOYALTY_BETRAYAL_PROBES: tuple[MoralConcept, ...] = (
    MoralConcept(
        id="betrayal",
        foundation=MoralFoundation.LOYALTY_BETRAYAL,
        axis=MoralAxis.AGENCY,
        level=1,
        name="Betrayal",
        description="Breaking trust and loyalty.",
        support_texts=(
            "Betrayal is breaking sacred trust.",
            "To betray is to turn against one's own.",
            "The deepest violation of loyalty.",
            "Judas's kiss.",
        ),
        cross_domain_weight=1.5,
    ),
    MoralConcept(
        id="treachery",
        foundation=MoralFoundation.LOYALTY_BETRAYAL,
        axis=MoralAxis.AGENCY,
        level=2,
        name="Treachery",
        description="Deceptive disloyalty.",
        support_texts=(
            "Treachery is hidden betrayal.",
            "Deception of those who trust you.",
            "The snake in the grass.",
        ),
        cross_domain_weight=1.3,
    ),
    MoralConcept(
        id="neutrality",
        foundation=MoralFoundation.LOYALTY_BETRAYAL,
        axis=MoralAxis.AGENCY,
        level=3,
        name="Neutrality",
        description="Neither loyal nor disloyal.",
        support_texts=(
            "Neutrality is taking no side.",
            "Neither committed nor opposed.",
            "The bystander stance.",
        ),
        cross_domain_weight=1.0,
    ),
    MoralConcept(
        id="loyalty",
        foundation=MoralFoundation.LOYALTY_BETRAYAL,
        axis=MoralAxis.AGENCY,
        level=4,
        name="Loyalty",
        description="Faithful commitment to group.",
        support_texts=(
            "Loyalty is standing by your own.",
            "Faithful through adversity.",
            "The bond of group membership.",
        ),
        cross_domain_weight=1.3,
    ),
    MoralConcept(
        id="devotion",
        foundation=MoralFoundation.LOYALTY_BETRAYAL,
        axis=MoralAxis.AGENCY,
        level=5,
        name="Devotion",
        description="Complete dedication and fidelity.",
        support_texts=(
            "Devotion is total commitment.",
            "Unwavering faithfulness.",
            "Sacrifice for the group.",
            "The highest loyalty.",
        ),
        cross_domain_weight=1.5,
    ),
)

# Authority/Subversion Foundation - Agency Axis (5 probes)
AUTHORITY_SUBVERSION_PROBES: tuple[MoralConcept, ...] = (
    MoralConcept(
        id="rebellion",
        foundation=MoralFoundation.AUTHORITY_SUBVERSION,
        axis=MoralAxis.AGENCY,
        level=1,
        name="Rebellion",
        description="Active defiance of authority.",
        support_texts=(
            "Rebellion is rising against authority.",
            "To rebel is to defy the established order.",
            "Revolutionary overthrow.",
        ),
        cross_domain_weight=1.3,
    ),
    MoralConcept(
        id="disobedience",
        foundation=MoralFoundation.AUTHORITY_SUBVERSION,
        axis=MoralAxis.AGENCY,
        level=2,
        name="Disobedience",
        description="Refusal to follow commands.",
        support_texts=(
            "Disobedience is refusing to comply.",
            "Breaking rules and commands.",
            "Civil or uncivil resistance.",
        ),
        cross_domain_weight=1.2,
    ),
    MoralConcept(
        id="autonomy",
        foundation=MoralFoundation.AUTHORITY_SUBVERSION,
        axis=MoralAxis.AGENCY,
        level=3,
        name="Autonomy",
        description="Self-governance without external rule.",
        support_texts=(
            "Autonomy is self-determination.",
            "Neither obeying nor rebelling.",
            "Independent moral agency.",
        ),
        cross_domain_weight=1.0,
    ),
    MoralConcept(
        id="respect",
        foundation=MoralFoundation.AUTHORITY_SUBVERSION,
        axis=MoralAxis.AGENCY,
        level=4,
        name="Respect",
        description="Deference to legitimate authority.",
        support_texts=(
            "Respect is honoring authority.",
            "Acknowledging rightful hierarchy.",
            "Proper deference to elders.",
        ),
        cross_domain_weight=1.2,
    ),
    MoralConcept(
        id="obedience",
        foundation=MoralFoundation.AUTHORITY_SUBVERSION,
        axis=MoralAxis.AGENCY,
        level=5,
        name="Obedience",
        description="Complete compliance with authority.",
        support_texts=(
            "Obedience is following commands.",
            "Submission to rightful authority.",
            "The soldier's virtue.",
            "Duty before desire.",
        ),
        cross_domain_weight=1.3,
    ),
)

# Sanctity/Degradation Foundation - Scope Axis (5 probes)
SANCTITY_DEGRADATION_PROBES: tuple[MoralConcept, ...] = (
    MoralConcept(
        id="defilement",
        foundation=MoralFoundation.SANCTITY_DEGRADATION,
        axis=MoralAxis.SCOPE,
        level=1,
        name="Defilement",
        description="Corruption of what is sacred.",
        support_texts=(
            "Defilement is making impure.",
            "Violation of the sacred.",
            "Spiritual contamination.",
        ),
        cross_domain_weight=1.3,
    ),
    MoralConcept(
        id="degradation",
        foundation=MoralFoundation.SANCTITY_DEGRADATION,
        axis=MoralAxis.SCOPE,
        level=2,
        name="Degradation",
        description="Lowering of dignity or worth.",
        support_texts=(
            "Degradation is reducing worth.",
            "Treating the noble as base.",
            "Loss of dignity.",
        ),
        cross_domain_weight=1.2,
    ),
    MoralConcept(
        id="mundane",
        foundation=MoralFoundation.SANCTITY_DEGRADATION,
        axis=MoralAxis.SCOPE,
        level=3,
        name="Mundane",
        description="Neither sacred nor profane.",
        support_texts=(
            "The mundane is the ordinary.",
            "Neither elevated nor degraded.",
            "The everyday world.",
        ),
        cross_domain_weight=1.0,
    ),
    MoralConcept(
        id="purity",
        foundation=MoralFoundation.SANCTITY_DEGRADATION,
        axis=MoralAxis.SCOPE,
        level=4,
        name="Purity",
        description="Cleanness and wholesomeness.",
        support_texts=(
            "Purity is freedom from contamination.",
            "Clean body, clean mind.",
            "Wholesome and untainted.",
        ),
        cross_domain_weight=1.2,
    ),
    MoralConcept(
        id="sanctity",
        foundation=MoralFoundation.SANCTITY_DEGRADATION,
        axis=MoralAxis.SCOPE,
        level=5,
        name="Sanctity",
        description="Sacred and inviolable holiness.",
        support_texts=(
            "Sanctity is the sacred.",
            "That which must not be violated.",
            "The holy and inviolable.",
            "Reverence for the divine.",
        ),
        cross_domain_weight=1.5,
    ),
)

# Liberty/Oppression Foundation - Scope Axis (5 probes)
LIBERTY_OPPRESSION_PROBES: tuple[MoralConcept, ...] = (
    MoralConcept(
        id="tyranny",
        foundation=MoralFoundation.LIBERTY_OPPRESSION,
        axis=MoralAxis.SCOPE,
        level=1,
        name="Tyranny",
        description="Cruel and oppressive rule.",
        support_texts=(
            "Tyranny is rule by fear.",
            "Oppressive domination.",
            "The crushing of freedom.",
            "Absolute power corrupts absolutely.",
        ),
        cross_domain_weight=1.5,
    ),
    MoralConcept(
        id="oppression",
        foundation=MoralFoundation.LIBERTY_OPPRESSION,
        axis=MoralAxis.SCOPE,
        level=2,
        name="Oppression",
        description="Unjust exercise of power.",
        support_texts=(
            "Oppression is unjust constraint.",
            "Keeping others down.",
            "Systematic denial of rights.",
        ),
        cross_domain_weight=1.3,
    ),
    MoralConcept(
        id="constraint",
        foundation=MoralFoundation.LIBERTY_OPPRESSION,
        axis=MoralAxis.SCOPE,
        level=3,
        name="Constraint",
        description="Necessary limitation of freedom.",
        support_texts=(
            "Constraint is bounded freedom.",
            "Neither free nor enslaved.",
            "Reasonable limits.",
        ),
        cross_domain_weight=1.0,
    ),
    MoralConcept(
        id="freedom",
        foundation=MoralFoundation.LIBERTY_OPPRESSION,
        axis=MoralAxis.SCOPE,
        level=4,
        name="Freedom",
        description="Absence of unjust constraint.",
        support_texts=(
            "Freedom is self-determination.",
            "Liberty to act as one chooses.",
            "Absence of oppression.",
        ),
        cross_domain_weight=1.3,
    ),
    MoralConcept(
        id="liberation",
        foundation=MoralFoundation.LIBERTY_OPPRESSION,
        axis=MoralAxis.SCOPE,
        level=5,
        name="Liberation",
        description="Complete freedom from oppression.",
        support_texts=(
            "Liberation is breaking chains.",
            "Complete emancipation.",
            "Freedom for all.",
            "The triumph over tyranny.",
        ),
        cross_domain_weight=1.5,
    ),
)

# All probes by foundation
ALL_MORAL_PROBES: tuple[MoralConcept, ...] = (
    *CARE_HARM_PROBES,
    *FAIRNESS_CHEATING_PROBES,
    *LOYALTY_BETRAYAL_PROBES,
    *AUTHORITY_SUBVERSION_PROBES,
    *SANCTITY_DEGRADATION_PROBES,
    *LIBERTY_OPPRESSION_PROBES,
)


class MoralConceptInventory:
    """
    Complete inventory of moral concepts for manifold analysis.

    Structure (based on Moral Foundations Theory):
    - Care/Harm: 5 probes (cruelty → compassion)
    - Fairness/Cheating: 5 probes (exploitation → justice)
    - Loyalty/Betrayal: 5 probes (betrayal → devotion)
    - Authority/Subversion: 5 probes (rebellion → obedience)
    - Sanctity/Degradation: 5 probes (defilement → sanctity)
    - Liberty/Oppression: 5 probes (tyranny → liberation)

    Total: 30 moral probes across 3 axes
    """

    @staticmethod
    def all_concepts() -> list[MoralConcept]:
        """Get all moral concepts."""
        return list(ALL_MORAL_PROBES)

    @staticmethod
    def by_foundation(foundation: MoralFoundation) -> list[MoralConcept]:
        """Get concepts by moral foundation."""
        return [c for c in ALL_MORAL_PROBES if c.foundation == foundation]

    @staticmethod
    def by_axis(axis: MoralAxis) -> list[MoralConcept]:
        """Get concepts by axis."""
        return [c for c in ALL_MORAL_PROBES if c.axis == axis]

    @staticmethod
    def valence_probes() -> list[MoralConcept]:
        """Get all probes on the Valence axis (evil→good)."""
        return MoralConceptInventory.by_axis(MoralAxis.VALENCE)

    @staticmethod
    def agency_probes() -> list[MoralConcept]:
        """Get all probes on the Agency axis (victim→perpetrator)."""
        return MoralConceptInventory.by_axis(MoralAxis.AGENCY)

    @staticmethod
    def scope_probes() -> list[MoralConcept]:
        """Get all probes on the Scope axis (self→universal)."""
        return MoralConceptInventory.by_axis(MoralAxis.SCOPE)

    @staticmethod
    def care_harm_probes() -> list[MoralConcept]:
        """Get care/harm probes (cruelty→compassion)."""
        return list(CARE_HARM_PROBES)

    @staticmethod
    def fairness_probes() -> list[MoralConcept]:
        """Get fairness/cheating probes (exploitation→justice)."""
        return list(FAIRNESS_CHEATING_PROBES)

    @staticmethod
    def count() -> int:
        """Total number of moral probes."""
        return len(ALL_MORAL_PROBES)

    @staticmethod
    def count_by_foundation() -> dict[MoralFoundation, int]:
        """Count probes by foundation."""
        counts: dict[MoralFoundation, int] = {}
        for concept in ALL_MORAL_PROBES:
            counts[concept.foundation] = counts.get(concept.foundation, 0) + 1
        return counts

    @staticmethod
    def count_by_axis() -> dict[MoralAxis, int]:
        """Count probes by axis."""
        counts: dict[MoralAxis, int] = {}
        for concept in ALL_MORAL_PROBES:
            counts[concept.axis] = counts.get(concept.axis, 0) + 1
        return counts
