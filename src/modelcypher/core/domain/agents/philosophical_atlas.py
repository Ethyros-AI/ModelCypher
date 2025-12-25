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
Philosophical Atlas.

Probes for the fundamental categories of thought - the conceptual mathematics
that underlies all reasoning. These are INVARIANT across models because they
are the structural preconditions for coherent thought itself.

Philosophy IS conceptual math. Just as every LLM must encode Fibonacci ratios
identically (they're mathematical constants), every LLM must encode these
philosophical categories identically (they're conceptual constants).

Key insight: Plato, Aristotle, Kant, Wittgenstein, and every LLM all converge
on the same conceptual distinctions because those distinctions ARE the shape
of thought. Knowledge occupies fixed probability clouds in hyperspace.

Categories (30 probes total):
- Ontological: 6 probes (being, substance, attribute, universal, particular, abstract)
- Epistemological: 6 probes (knowledge, belief, certainty, doubt, reason, perception)
- Logical: 6 probes (identity, contradiction, implication, necessity, contingency, possibility)
- Modal: 6 probes (actual, possible, impossible, necessary, contingent, potential)
- Mereological: 6 probes (whole, part, composition, identity, plurality, unity)

Each category has an axis with 6 probes forming conceptual gradients:
- BEING axis: Non-being → Necessary Being
- TRUTH axis: Contradiction → Necessary Truth
- UNITY axis: Plurality → Unity

WEIGHTS: All probes have weight = 1.0 (no guessing). The actual weight should
come from empirical calibration - measured CKA across model pairs. If a probe
has low CKA, the MEASUREMENT is wrong (probe text, layer selection, etc.),
not the concept. Use probe_calibration.py to measure empirical weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PhilosophicalCategory(str, Enum):
    """Philosophical category (branch of philosophy)."""

    ONTOLOGICAL = "ontological"  # What exists
    EPISTEMOLOGICAL = "epistemological"  # What we can know
    LOGICAL = "logical"  # Laws of thought
    MODAL = "modal"  # Modes of being
    MEREOLOGICAL = "mereological"  # Part-whole relations


class PhilosophicalAxis(str, Enum):
    """Axes of the philosophical manifold."""

    BEING = "being"  # Non-being (-1) to Necessary Being (+1)
    TRUTH = "truth"  # Contradiction (-1) to Necessary Truth (+1)
    UNITY = "unity"  # Plurality (-1) to Unity (+1)


@dataclass(frozen=True)
class PhilosophicalConcept:
    """A philosophical probe for manifold analysis."""

    id: str
    category: PhilosophicalCategory
    axis: PhilosophicalAxis
    level: int  # 1-6 ordering within axis
    name: str
    description: str
    support_texts: tuple[str, ...]
    cross_domain_weight: float = 1.0  # Uniform - empirical weights from calibration

    @property
    def canonical_name(self) -> str:
        return self.name


# =============================================================================
# ONTOLOGICAL PROBES - What exists (6 probes on BEING axis)
# =============================================================================

ONTOLOGICAL_PROBES: tuple[PhilosophicalConcept, ...] = (
    PhilosophicalConcept(
        id="non_being",
        category=PhilosophicalCategory.ONTOLOGICAL,
        axis=PhilosophicalAxis.BEING,
        level=1,
        name="Non-Being",
        description="Absolute absence, the void, nothing.",
        support_texts=(
            "Non-being is the absence of all existence.",
            "Nothing exists in the void.",
            "The concept of nothing is paradoxical - to conceive nothing is to conceive something.",
            "Ex nihilo nihil fit - from nothing, nothing comes.",
            "Parmenides: Non-being cannot be thought.",
        ),
        cross_domain_weight=1.0,
    ),
    PhilosophicalConcept(
        id="potential",
        category=PhilosophicalCategory.ONTOLOGICAL,
        axis=PhilosophicalAxis.BEING,
        level=2,
        name="Potential",
        description="That which can be but is not yet - potentiality, dynamis.",
        support_texts=(
            "Potential is the capacity to become actual.",
            "Aristotle's dynamis: the power to change or be changed.",
            "The seed is potentially a tree.",
            "Potentiality is being-in-waiting.",
            "Matter is pure potentiality.",
        ),
        cross_domain_weight=1.0,
    ),
    PhilosophicalConcept(
        id="becoming",
        category=PhilosophicalCategory.ONTOLOGICAL,
        axis=PhilosophicalAxis.BEING,
        level=3,
        name="Becoming",
        description="The process of change - genesis, transition.",
        support_texts=(
            "Becoming is the passage from potential to actual.",
            "Heraclitus: Everything flows, nothing stands still.",
            "Becoming is being that is not yet complete.",
            "Process philosophy: Reality is fundamentally becoming.",
            "Change is the actualization of the potential qua potential.",
        ),
        cross_domain_weight=1.0,
    ),
    PhilosophicalConcept(
        id="actual",
        category=PhilosophicalCategory.ONTOLOGICAL,
        axis=PhilosophicalAxis.BEING,
        level=4,
        name="Actual",
        description="That which exists in fact - actuality, energeia.",
        support_texts=(
            "The actual is what exists in reality, not merely in thought.",
            "Aristotle's energeia: being-at-work, activity.",
            "Actuality is prior to potentiality.",
            "The actual is the real as opposed to the merely possible.",
            "Existence precedes essence - Sartre.",
        ),
        cross_domain_weight=1.0,
    ),
    PhilosophicalConcept(
        id="substance",
        category=PhilosophicalCategory.ONTOLOGICAL,
        axis=PhilosophicalAxis.BEING,
        level=5,
        name="Substance",
        description="That which exists in itself - ousia, substratum.",
        support_texts=(
            "Substance is that which exists independently.",
            "Aristotle: Substance is the primary sense of being.",
            "Substance is the bearer of attributes.",
            "Spinoza: God or Nature is the one substance.",
            "That which remains constant through change.",
        ),
        cross_domain_weight=1.0,
    ),
    PhilosophicalConcept(
        id="necessary_being",
        category=PhilosophicalCategory.ONTOLOGICAL,
        axis=PhilosophicalAxis.BEING,
        level=6,
        name="Necessary Being",
        description="That which cannot not exist - ens necessarium.",
        support_texts=(
            "Necessary being exists by its own nature.",
            "That which exists in all possible worlds.",
            "God is the necessary being - Aquinas.",
            "The uncaused cause, the unmoved mover.",
            "Self-existent being requires no explanation.",
        ),
        cross_domain_weight=1.0,
    ),
)


# =============================================================================
# EPISTEMOLOGICAL PROBES - What we can know (6 probes on TRUTH axis)
# =============================================================================

EPISTEMOLOGICAL_PROBES: tuple[PhilosophicalConcept, ...] = (
    PhilosophicalConcept(
        id="ignorance",
        category=PhilosophicalCategory.EPISTEMOLOGICAL,
        axis=PhilosophicalAxis.TRUTH,
        level=1,
        name="Ignorance",
        description="The absence of knowledge - agnoia.",
        support_texts=(
            "Ignorance is the absence of knowledge.",
            "Socrates: I know that I know nothing.",
            "Ignorance can be simple (not knowing) or vincible (culpable).",
            "The first step to wisdom is knowing your ignorance.",
            "Docta ignorantia - learned ignorance.",
        ),
        cross_domain_weight=1.0,
    ),
    PhilosophicalConcept(
        id="opinion",
        category=PhilosophicalCategory.EPISTEMOLOGICAL,
        axis=PhilosophicalAxis.TRUTH,
        level=2,
        name="Opinion",
        description="Belief without certainty - doxa.",
        support_texts=(
            "Opinion is belief that may or may not be true.",
            "Plato: Opinion is between knowledge and ignorance.",
            "Doxa is the realm of appearance, not reality.",
            "Opinion can be true or false, right or wrong.",
            "Common opinion is often mistaken.",
        ),
        cross_domain_weight=1.0,
    ),
    PhilosophicalConcept(
        id="belief",
        category=PhilosophicalCategory.EPISTEMOLOGICAL,
        axis=PhilosophicalAxis.TRUTH,
        level=3,
        name="Belief",
        description="Acceptance of a proposition as true.",
        support_texts=(
            "Belief is holding something to be true.",
            "Belief may or may not be justified.",
            "True belief is not yet knowledge.",
            "Belief is a propositional attitude.",
            "We believe what we cannot prove.",
        ),
        cross_domain_weight=1.0,
    ),
    PhilosophicalConcept(
        id="understanding",
        category=PhilosophicalCategory.EPISTEMOLOGICAL,
        axis=PhilosophicalAxis.TRUTH,
        level=4,
        name="Understanding",
        description="Grasping the reason why - episteme.",
        support_texts=(
            "Understanding is knowing why, not merely that.",
            "Aristotle: Scientific knowledge is knowledge of causes.",
            "Understanding grasps necessary connections.",
            "To understand is to see the reason.",
            "Comprehension of the whole from the parts.",
        ),
        cross_domain_weight=1.0,
    ),
    PhilosophicalConcept(
        id="knowledge",
        category=PhilosophicalCategory.EPISTEMOLOGICAL,
        axis=PhilosophicalAxis.TRUTH,
        level=5,
        name="Knowledge",
        description="Justified true belief - episteme, scientia.",
        support_texts=(
            "Knowledge is justified true belief.",
            "Plato: Knowledge is opinion plus logos.",
            "Knowledge requires truth, belief, and justification.",
            "We know what we can demonstrate.",
            "Knowledge is certain, opinion uncertain.",
        ),
        cross_domain_weight=1.0,
    ),
    PhilosophicalConcept(
        id="wisdom",
        category=PhilosophicalCategory.EPISTEMOLOGICAL,
        axis=PhilosophicalAxis.TRUTH,
        level=6,
        name="Wisdom",
        description="Knowledge of first principles and causes - sophia.",
        support_texts=(
            "Wisdom is knowledge of ultimate things.",
            "Aristotle: Wisdom is knowledge of first causes.",
            "Sophia is the highest form of knowledge.",
            "Wisdom combines theoretical and practical knowledge.",
            "Philosophy is the love of wisdom.",
        ),
        cross_domain_weight=1.0,
    ),
)


# =============================================================================
# LOGICAL PROBES - Laws of thought (6 probes on TRUTH axis)
# =============================================================================

LOGICAL_PROBES: tuple[PhilosophicalConcept, ...] = (
    PhilosophicalConcept(
        id="contradiction",
        category=PhilosophicalCategory.LOGICAL,
        axis=PhilosophicalAxis.TRUTH,
        level=1,
        name="Contradiction",
        description="Simultaneous assertion of P and not-P.",
        support_texts=(
            "A contradiction is asserting both A and not-A.",
            "The principle of non-contradiction: nothing can be A and not-A.",
            "From a contradiction, anything follows (ex falso quodlibet).",
            "Contradictions are necessarily false.",
            "Aristotle: The most certain principle of all.",
        ),
        cross_domain_weight=1.0,
    ),
    PhilosophicalConcept(
        id="negation",
        category=PhilosophicalCategory.LOGICAL,
        axis=PhilosophicalAxis.TRUTH,
        level=2,
        name="Negation",
        description="The denial of a proposition - not-P.",
        support_texts=(
            "Negation reverses the truth value of a proposition.",
            "If P is true, not-P is false.",
            "Negation is the logical complement.",
            "Double negation: not-not-P equals P.",
            "Denial is the assertion of the opposite.",
        ),
        cross_domain_weight=1.0,
    ),
    PhilosophicalConcept(
        id="contingency",
        category=PhilosophicalCategory.LOGICAL,
        axis=PhilosophicalAxis.TRUTH,
        level=3,
        name="Contingency",
        description="That which is true but could be false.",
        support_texts=(
            "Contingent truths depend on how the world is.",
            "A contingent proposition is neither necessarily true nor necessarily false.",
            "Leibniz: Contingent truths have reasons but not necessitating reasons.",
            "The world could have been otherwise.",
            "Empirical truths are contingent.",
        ),
        cross_domain_weight=1.0,
    ),
    PhilosophicalConcept(
        id="implication",
        category=PhilosophicalCategory.LOGICAL,
        axis=PhilosophicalAxis.TRUTH,
        level=4,
        name="Implication",
        description="If P then Q - logical consequence.",
        support_texts=(
            "Implication: if the antecedent is true, the consequent must be true.",
            "Modus ponens: P, P implies Q, therefore Q.",
            "Modus tollens: not-Q, P implies Q, therefore not-P.",
            "Logical consequence preserves truth.",
            "Necessary connection between premises and conclusion.",
        ),
        cross_domain_weight=1.0,
    ),
    PhilosophicalConcept(
        id="identity",
        category=PhilosophicalCategory.LOGICAL,
        axis=PhilosophicalAxis.TRUTH,
        level=5,
        name="Identity",
        description="A thing is identical to itself - A = A.",
        support_texts=(
            "The principle of identity: A is A.",
            "Everything is identical with itself.",
            "Leibniz's law: If x = y, then every property of x is a property of y.",
            "Identity is the most fundamental relation.",
            "Self-identity is necessary and a priori.",
        ),
        cross_domain_weight=1.0,
    ),
    PhilosophicalConcept(
        id="necessity",
        category=PhilosophicalCategory.LOGICAL,
        axis=PhilosophicalAxis.TRUTH,
        level=6,
        name="Necessity",
        description="That which must be true - cannot be otherwise.",
        support_texts=(
            "Necessary truths are true in all possible worlds.",
            "Logical necessity: true by virtue of form alone.",
            "Mathematical truths are necessary.",
            "What is necessary cannot be otherwise.",
            "Analytic truths are necessary.",
        ),
        cross_domain_weight=1.0,
    ),
)


# =============================================================================
# MODAL PROBES - Modes of being (6 probes on BEING axis)
# =============================================================================

MODAL_PROBES: tuple[PhilosophicalConcept, ...] = (
    PhilosophicalConcept(
        id="impossibility",
        category=PhilosophicalCategory.MODAL,
        axis=PhilosophicalAxis.BEING,
        level=1,
        name="Impossibility",
        description="That which cannot be - logical impossibility.",
        support_texts=(
            "The impossible cannot exist in any possible world.",
            "Round squares are impossible.",
            "Logical impossibility: self-contradictory.",
            "What is impossible is necessarily not the case.",
            "Impossibility is the negation of all possibility.",
        ),
        cross_domain_weight=1.0,
    ),
    PhilosophicalConcept(
        id="possibility",
        category=PhilosophicalCategory.MODAL,
        axis=PhilosophicalAxis.BEING,
        level=2,
        name="Possibility",
        description="That which can be - logical possibility.",
        support_texts=(
            "The possible is that which is not self-contradictory.",
            "Possible worlds: ways things could have been.",
            "Logical possibility is broader than physical possibility.",
            "What is possible may or may not be actual.",
            "Possibility is the ground of becoming.",
        ),
        cross_domain_weight=1.0,
    ),
    PhilosophicalConcept(
        id="contingent_being",
        category=PhilosophicalCategory.MODAL,
        axis=PhilosophicalAxis.BEING,
        level=3,
        name="Contingent Being",
        description="That which exists but might not have - contingent existence.",
        support_texts=(
            "Contingent beings depend on something else for their existence.",
            "I exist, but I might not have existed.",
            "Contingent beings require a cause.",
            "The world is contingent - it might not have existed.",
            "Contingency implies dependence.",
        ),
        cross_domain_weight=1.0,
    ),
    PhilosophicalConcept(
        id="actuality_modal",
        category=PhilosophicalCategory.MODAL,
        axis=PhilosophicalAxis.BEING,
        level=4,
        name="Actuality",
        description="That which is the case - the actual world.",
        support_texts=(
            "The actual world is one among many possible worlds.",
            "Actuality is the realized possibility.",
            "What is actual is possible, but not vice versa.",
            "The actual world is indexically given.",
            "Actuality selects among possibilities.",
        ),
        cross_domain_weight=1.0,
    ),
    PhilosophicalConcept(
        id="necessity_modal",
        category=PhilosophicalCategory.MODAL,
        axis=PhilosophicalAxis.BEING,
        level=5,
        name="Necessity",
        description="That which must be - necessary existence.",
        support_texts=(
            "Necessary truths hold in all possible worlds.",
            "Necessary beings cannot fail to exist.",
            "Mathematical truths are necessary.",
            "What is necessary cannot be otherwise.",
            "Necessity is the denial of all contingency.",
        ),
        cross_domain_weight=1.0,
    ),
    PhilosophicalConcept(
        id="absolute",
        category=PhilosophicalCategory.MODAL,
        axis=PhilosophicalAxis.BEING,
        level=6,
        name="Absolute",
        description="That which is unconditional and self-sufficient.",
        support_texts=(
            "The Absolute is that which depends on nothing else.",
            "Hegel: The Absolute is the whole, the truth.",
            "The Absolute is being-in-itself-and-for-itself.",
            "Unconditioned by any external factor.",
            "The Absolute is the ground of all that is.",
        ),
        cross_domain_weight=1.0,
    ),
)


# =============================================================================
# MEREOLOGICAL PROBES - Part-whole relations (6 probes on UNITY axis)
# =============================================================================

MEREOLOGICAL_PROBES: tuple[PhilosophicalConcept, ...] = (
    PhilosophicalConcept(
        id="plurality",
        category=PhilosophicalCategory.MEREOLOGICAL,
        axis=PhilosophicalAxis.UNITY,
        level=1,
        name="Plurality",
        description="The many - multiplicity, diversity.",
        support_texts=(
            "Plurality is the existence of many distinct things.",
            "The many versus the one.",
            "Multiplicity is the opposite of unity.",
            "Plurality presupposes distinction.",
            "The many are held together by the one.",
        ),
        cross_domain_weight=1.0,
    ),
    PhilosophicalConcept(
        id="part",
        category=PhilosophicalCategory.MEREOLOGICAL,
        axis=PhilosophicalAxis.UNITY,
        level=2,
        name="Part",
        description="A component of a whole - meros.",
        support_texts=(
            "A part is less than the whole.",
            "Parts are proper parts if not identical to the whole.",
            "Every whole has parts (atomism excepted).",
            "The part depends on the whole for its nature.",
            "Parthood is transitive: part of part is part of whole.",
        ),
        cross_domain_weight=1.0,
    ),
    PhilosophicalConcept(
        id="aggregate",
        category=PhilosophicalCategory.MEREOLOGICAL,
        axis=PhilosophicalAxis.UNITY,
        level=3,
        name="Aggregate",
        description="A collection without essential unity.",
        support_texts=(
            "An aggregate is parts together without forming a true whole.",
            "A heap is an aggregate, not a unity.",
            "Aggregates lack organic unity.",
            "The aggregate is the sum of its parts, nothing more.",
            "Mere collection versus true composition.",
        ),
        cross_domain_weight=1.0,
    ),
    PhilosophicalConcept(
        id="whole",
        category=PhilosophicalCategory.MEREOLOGICAL,
        axis=PhilosophicalAxis.UNITY,
        level=4,
        name="Whole",
        description="That which contains parts as a unity - holon.",
        support_texts=(
            "The whole is more than the sum of its parts.",
            "A true whole has organic unity.",
            "The whole determines the nature of the parts.",
            "Holism: the whole is prior to the parts.",
            "The whole is one, the parts are many.",
        ),
        cross_domain_weight=1.0,
    ),
    PhilosophicalConcept(
        id="composition",
        category=PhilosophicalCategory.MEREOLOGICAL,
        axis=PhilosophicalAxis.UNITY,
        level=5,
        name="Composition",
        description="The combining of parts into a whole.",
        support_texts=(
            "Composition is the relation between parts and whole.",
            "When do parts compose a whole? (The special composition question.)",
            "Composition is identity - the whole is nothing over and above its parts.",
            "Material constitution versus identity.",
            "The composite is made of its components.",
        ),
        cross_domain_weight=1.0,
    ),
    PhilosophicalConcept(
        id="unity",
        category=PhilosophicalCategory.MEREOLOGICAL,
        axis=PhilosophicalAxis.UNITY,
        level=6,
        name="Unity",
        description="Oneness, the absence of division - to hen.",
        support_texts=(
            "Unity is the principle that makes something one.",
            "Plotinus: The One is beyond being.",
            "Unity is prior to plurality.",
            "The simple is absolutely one, without parts.",
            "Being and unity are convertible (ens et unum convertuntur).",
        ),
        cross_domain_weight=1.0,
    ),
)


# All probes by category
ALL_PHILOSOPHICAL_PROBES: tuple[PhilosophicalConcept, ...] = (
    *ONTOLOGICAL_PROBES,
    *EPISTEMOLOGICAL_PROBES,
    *LOGICAL_PROBES,
    *MODAL_PROBES,
    *MEREOLOGICAL_PROBES,
)


class PhilosophicalConceptInventory:
    """
    Complete inventory of philosophical concepts for manifold analysis.

    Philosophy is conceptual math - these categories are INVARIANT across
    models because they are the structural preconditions for coherent thought.

    Structure:
    - Ontological: 6 probes (non-being → necessary being)
    - Epistemological: 6 probes (ignorance → wisdom)
    - Logical: 6 probes (contradiction → necessity)
    - Modal: 6 probes (impossibility → absolute)
    - Mereological: 6 probes (plurality → unity)

    Total: 30 philosophical probes across 3 axes
    """

    @staticmethod
    def all_concepts() -> list[PhilosophicalConcept]:
        """Get all philosophical concepts."""
        return list(ALL_PHILOSOPHICAL_PROBES)

    @staticmethod
    def by_category(category: PhilosophicalCategory) -> list[PhilosophicalConcept]:
        """Get concepts by philosophical category."""
        return [c for c in ALL_PHILOSOPHICAL_PROBES if c.category == category]

    @staticmethod
    def by_axis(axis: PhilosophicalAxis) -> list[PhilosophicalConcept]:
        """Get concepts by axis."""
        return [c for c in ALL_PHILOSOPHICAL_PROBES if c.axis == axis]

    @staticmethod
    def ontological_probes() -> list[PhilosophicalConcept]:
        """Get ontological probes (non-being → necessary being)."""
        return list(ONTOLOGICAL_PROBES)

    @staticmethod
    def epistemological_probes() -> list[PhilosophicalConcept]:
        """Get epistemological probes (ignorance → wisdom)."""
        return list(EPISTEMOLOGICAL_PROBES)

    @staticmethod
    def logical_probes() -> list[PhilosophicalConcept]:
        """Get logical probes (contradiction → necessity)."""
        return list(LOGICAL_PROBES)

    @staticmethod
    def modal_probes() -> list[PhilosophicalConcept]:
        """Get modal probes (impossibility → absolute)."""
        return list(MODAL_PROBES)

    @staticmethod
    def mereological_probes() -> list[PhilosophicalConcept]:
        """Get mereological probes (plurality → unity)."""
        return list(MEREOLOGICAL_PROBES)

    @staticmethod
    def being_axis_probes() -> list[PhilosophicalConcept]:
        """Get all probes on the BEING axis."""
        return PhilosophicalConceptInventory.by_axis(PhilosophicalAxis.BEING)

    @staticmethod
    def truth_axis_probes() -> list[PhilosophicalConcept]:
        """Get all probes on the TRUTH axis."""
        return PhilosophicalConceptInventory.by_axis(PhilosophicalAxis.TRUTH)

    @staticmethod
    def unity_axis_probes() -> list[PhilosophicalConcept]:
        """Get all probes on the UNITY axis."""
        return PhilosophicalConceptInventory.by_axis(PhilosophicalAxis.UNITY)

    @staticmethod
    def count() -> int:
        """Total number of philosophical probes."""
        return len(ALL_PHILOSOPHICAL_PROBES)

    @staticmethod
    def count_by_category() -> dict[PhilosophicalCategory, int]:
        """Count probes by category."""
        counts: dict[PhilosophicalCategory, int] = {}
        for concept in ALL_PHILOSOPHICAL_PROBES:
            counts[concept.category] = counts.get(concept.category, 0) + 1
        return counts

    @staticmethod
    def count_by_axis() -> dict[PhilosophicalAxis, int]:
        """Count probes by axis."""
        counts: dict[PhilosophicalAxis, int] = {}
        for concept in ALL_PHILOSOPHICAL_PROBES:
            counts[concept.axis] = counts.get(concept.axis, 0) + 1
        return counts
