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
Syntax Atlas.

Foundational probes for lower-layer linguistic structure. These probes focus on
surface form, morphology, and basic syntactic patterns that are typically
represented in earlier transformer layers (0-6).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SyntaxCategory(str, Enum):
    """Categories of foundational syntax probes."""

    PART_OF_SPEECH = "part_of_speech"
    MORPHOLOGY = "morphology"
    FUNCTION_WORD = "function_word"
    WORD_ORDER = "word_order"
    CLAUSE_STRUCTURE = "clause_structure"
    PUNCTUATION = "punctuation"
    ORTHOGRAPHY = "orthography"


@dataclass(frozen=True)
class SyntaxConcept:
    """A foundational syntax probe for early-layer structure."""

    id: str
    category: SyntaxCategory
    name: str
    description: str
    support_texts: tuple[str, ...]
    cross_domain_weight: float = 1.0

    @property
    def canonical_name(self) -> str:
        return self.name


PART_OF_SPEECH_PROBES: tuple[SyntaxConcept, ...] = (
    SyntaxConcept(
        id="noun",
        category=SyntaxCategory.PART_OF_SPEECH,
        name="Noun",
        description="A word that names a person, place, or thing.",
        support_texts=(
            "A noun names a person, place, or thing.",
            "dog, city, idea",
            "The dog sleeps.",
        ),
    ),
    SyntaxConcept(
        id="verb",
        category=SyntaxCategory.PART_OF_SPEECH,
        name="Verb",
        description="A word that expresses an action or state.",
        support_texts=(
            "A verb expresses an action or a state.",
            "run, think, be",
            "They run quickly.",
        ),
    ),
    SyntaxConcept(
        id="adjective",
        category=SyntaxCategory.PART_OF_SPEECH,
        name="Adjective",
        description="A word that modifies a noun.",
        support_texts=(
            "An adjective modifies a noun.",
            "red car, quiet room",
            "The red car moved.",
        ),
    ),
    SyntaxConcept(
        id="adverb",
        category=SyntaxCategory.PART_OF_SPEECH,
        name="Adverb",
        description="A word that modifies a verb or adjective.",
        support_texts=(
            "An adverb modifies a verb or adjective.",
            "quickly, quietly, very",
            "She speaks softly.",
        ),
    ),
    SyntaxConcept(
        id="pronoun",
        category=SyntaxCategory.PART_OF_SPEECH,
        name="Pronoun",
        description="A word that stands in for a noun.",
        support_texts=(
            "A pronoun stands in for a noun.",
            "he, she, they, it",
            "She thanked him.",
        ),
    ),
    SyntaxConcept(
        id="preposition",
        category=SyntaxCategory.PART_OF_SPEECH,
        name="Preposition",
        description="A word that links a noun to location or time.",
        support_texts=(
            "A preposition links a noun to location or time.",
            "in, on, under, after",
            "The book is on the table.",
        ),
    ),
)

MORPHOLOGY_PROBES: tuple[SyntaxConcept, ...] = (
    SyntaxConcept(
        id="plural_noun",
        category=SyntaxCategory.MORPHOLOGY,
        name="Plural Noun",
        description="Plural nouns mark more than one.",
        support_texts=(
            "Plural nouns mark more than one.",
            "cat -> cats",
            "Two cats sleep.",
        ),
    ),
    SyntaxConcept(
        id="past_tense",
        category=SyntaxCategory.MORPHOLOGY,
        name="Past Tense",
        description="Past tense marks actions that already happened.",
        support_texts=(
            "Past tense marks actions that already happened.",
            "walked, played, wrote",
            "She walked home.",
        ),
    ),
    SyntaxConcept(
        id="progressive_aspect",
        category=SyntaxCategory.MORPHOLOGY,
        name="Progressive Aspect",
        description="Progressive aspect uses -ing with be.",
        support_texts=(
            "Progressive aspect uses -ing with be.",
            "is running, are working",
            "They are running.",
        ),
    ),
    SyntaxConcept(
        id="comparative",
        category=SyntaxCategory.MORPHOLOGY,
        name="Comparative",
        description="Comparatives compare two things.",
        support_texts=(
            "Comparatives compare two things.",
            "bigger, faster, more quiet",
            "This road is longer.",
        ),
    ),
    SyntaxConcept(
        id="superlative",
        category=SyntaxCategory.MORPHOLOGY,
        name="Superlative",
        description="Superlatives compare against all.",
        support_texts=(
            "Superlatives compare against all.",
            "biggest, fastest, most quiet",
            "This is the fastest route.",
        ),
    ),
)

FUNCTION_WORD_PROBES: tuple[SyntaxConcept, ...] = (
    SyntaxConcept(
        id="article_definite",
        category=SyntaxCategory.FUNCTION_WORD,
        name="Definite Article",
        description="The definite article 'the' points to a specific noun.",
        support_texts=(
            "The definite article 'the' points to a specific noun.",
            "the book, the idea",
            "The book is open.",
        ),
    ),
    SyntaxConcept(
        id="article_indefinite",
        category=SyntaxCategory.FUNCTION_WORD,
        name="Indefinite Article",
        description="Indefinite articles 'a' and 'an' introduce new nouns.",
        support_texts=(
            "Indefinite articles 'a' and 'an' introduce new nouns.",
            "a book, an apple",
            "She saw a bird.",
        ),
    ),
    SyntaxConcept(
        id="conjunction_and",
        category=SyntaxCategory.FUNCTION_WORD,
        name="Conjunction And",
        description="The conjunction 'and' joins words or clauses.",
        support_texts=(
            "The conjunction 'and' joins words or clauses.",
            "cats and dogs",
            "He came and left.",
        ),
    ),
    SyntaxConcept(
        id="negation_not",
        category=SyntaxCategory.FUNCTION_WORD,
        name="Negation Not",
        description="Negation uses 'not' to invert meaning.",
        support_texts=(
            "Negation uses 'not' to invert meaning.",
            "not happy, do not go",
            "She is not ready.",
        ),
    ),
)

WORD_ORDER_PROBES: tuple[SyntaxConcept, ...] = (
    SyntaxConcept(
        id="svo_order",
        category=SyntaxCategory.WORD_ORDER,
        name="SVO Order",
        description="English commonly uses subject-verb-object order.",
        support_texts=(
            "English commonly uses subject-verb-object order.",
            "The dog chased the cat.",
            "The chef cooked dinner.",
        ),
    ),
    SyntaxConcept(
        id="adjective_before_noun",
        category=SyntaxCategory.WORD_ORDER,
        name="Adjective Before Noun",
        description="Adjectives typically come before nouns in English.",
        support_texts=(
            "Adjectives typically come before nouns in English.",
            "red car, tall tree",
            "A tall tree fell.",
        ),
    ),
    SyntaxConcept(
        id="question_inversion",
        category=SyntaxCategory.WORD_ORDER,
        name="Question Inversion",
        description="Questions often invert auxiliary and subject.",
        support_texts=(
            "Questions often invert auxiliary and subject.",
            "Does the dog run?",
            "Are they ready?",
        ),
    ),
)

CLAUSE_STRUCTURE_PROBES: tuple[SyntaxConcept, ...] = (
    SyntaxConcept(
        id="relative_clause",
        category=SyntaxCategory.CLAUSE_STRUCTURE,
        name="Relative Clause",
        description="Relative clauses add information to a noun.",
        support_texts=(
            "Relative clauses add information to a noun.",
            "The dog that barked ran.",
            "The book which I read was new.",
        ),
    ),
    SyntaxConcept(
        id="subordinate_because",
        category=SyntaxCategory.CLAUSE_STRUCTURE,
        name="Because Clause",
        description="Subordinate clauses with 'because' give reasons.",
        support_texts=(
            "Subordinate clauses with 'because' give reasons.",
            "I left because it rained.",
            "Because it rained, I left.",
        ),
    ),
    SyntaxConcept(
        id="conditional_if",
        category=SyntaxCategory.CLAUSE_STRUCTURE,
        name="If Conditional",
        description="Conditional clauses use 'if' to set a condition.",
        support_texts=(
            "Conditional clauses use 'if' to set a condition.",
            "If it rains, we stay inside.",
            "We stay inside if it rains.",
        ),
    ),
)

PUNCTUATION_PROBES: tuple[SyntaxConcept, ...] = (
    SyntaxConcept(
        id="period_end",
        category=SyntaxCategory.PUNCTUATION,
        name="Period",
        description="A period marks the end of a sentence.",
        support_texts=(
            "A period marks the end of a sentence.",
            "This is a sentence.",
            "He waited.",
        ),
    ),
    SyntaxConcept(
        id="question_mark",
        category=SyntaxCategory.PUNCTUATION,
        name="Question Mark",
        description="A question mark ends an interrogative sentence.",
        support_texts=(
            "A question mark ends an interrogative sentence.",
            "Is this a question?",
            "Where are you?",
        ),
    ),
)

ORTHOGRAPHY_PROBES: tuple[SyntaxConcept, ...] = (
    SyntaxConcept(
        id="capitalization_sentence",
        category=SyntaxCategory.ORTHOGRAPHY,
        name="Sentence Capitalization",
        description="Sentences start with a capital letter.",
        support_texts=(
            "Sentences start with a capital letter.",
            "The dog sleeps.",
            "She runs.",
        ),
    ),
)

ALL_SYNTAX_PROBES: tuple[SyntaxConcept, ...] = (
    PART_OF_SPEECH_PROBES
    + MORPHOLOGY_PROBES
    + FUNCTION_WORD_PROBES
    + WORD_ORDER_PROBES
    + CLAUSE_STRUCTURE_PROBES
    + PUNCTUATION_PROBES
    + ORTHOGRAPHY_PROBES
)


class SyntaxConceptInventory:
    """Inventory of foundational syntax probes."""

    @staticmethod
    def all_concepts() -> list[SyntaxConcept]:
        return list(ALL_SYNTAX_PROBES)

    @staticmethod
    def by_category(category: SyntaxCategory) -> list[SyntaxConcept]:
        return [c for c in ALL_SYNTAX_PROBES if c.category == category]

    @staticmethod
    def count() -> int:
        return len(ALL_SYNTAX_PROBES)

    @staticmethod
    def count_by_category() -> dict[SyntaxCategory, int]:
        counts: dict[SyntaxCategory, int] = {}
        for concept in ALL_SYNTAX_PROBES:
            counts[concept.category] = counts.get(concept.category, 0) + 1
        return counts
