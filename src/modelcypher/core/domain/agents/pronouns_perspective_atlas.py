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
Pronouns and Perspective Atlas - Reference terms for cross-dimensional alignment.

Covers pronominal and deictic concepts:
- Personal pronouns (I, you, he, she)
- Possessives (my, your, his, her)
- Reflexives (myself, yourself)
- Demonstratives (this, that, here, there)
- Interrogatives (who, what, where)
- Indefinites (someone, anyone, everyone)
- Relatives (who, which, that)

Total: ~50 probes for perspective grounding.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PronounCategory(str, Enum):
    """Category of pronoun/perspective concept."""
    
    PERSONAL = "personal"
    POSSESSIVE = "possessive"
    REFLEXIVE = "reflexive"
    DEMONSTRATIVE = "demonstrative"
    INTERROGATIVE = "interrogative"
    INDEFINITE = "indefinite"
    RELATIVE = "relative"


@dataclass(frozen=True)
class PronounConcept:
    """A pronoun/perspective concept."""
    
    id: str
    name: str
    category: PronounCategory
    description: str
    support_texts: tuple[str, ...]
    cross_domain_weight: float = 1.0


class PronounPerspectiveInventory:
    """Inventory of pronoun and perspective concepts."""
    
    # Personal pronouns
    PERSONAL = (
        PronounConcept("i_pron", "I", PronounCategory.PERSONAL, "First person singular", ("I", "me", "myself")),
        PronounConcept("you_pron", "you", PronounCategory.PERSONAL, "Second person", ("you", "yourself",)),
        PronounConcept("he", "he", PronounCategory.PERSONAL, "Third person masculine", ("he", "him", "himself")),
        PronounConcept("she", "she", PronounCategory.PERSONAL, "Third person feminine", ("she", "her", "herself")),
        PronounConcept("it", "it", PronounCategory.PERSONAL, "Third person neuter", ("it", "itself",)),
        PronounConcept("we", "we", PronounCategory.PERSONAL, "First person plural", ("we", "us", "ourselves")),
        PronounConcept("they", "they", PronounCategory.PERSONAL, "Third person plural", ("they", "them", "themselves")),
    )
    
    # Possessive pronouns
    POSSESSIVE = (
        PronounConcept("my", "my", PronounCategory.POSSESSIVE, "First person possession", ("my", "mine",)),
        PronounConcept("your", "your", PronounCategory.POSSESSIVE, "Second person possession", ("your", "yours",)),
        PronounConcept("his", "his", PronounCategory.POSSESSIVE, "Third person masculine possession", ("his",)),
        PronounConcept("her_poss", "her", PronounCategory.POSSESSIVE, "Third person feminine possession", ("her", "hers",)),
        PronounConcept("its", "its", PronounCategory.POSSESSIVE, "Third person neuter possession", ("its",)),
        PronounConcept("our", "our", PronounCategory.POSSESSIVE, "First person plural possession", ("our", "ours",)),
        PronounConcept("their", "their", PronounCategory.POSSESSIVE, "Third person plural possession", ("their", "theirs",)),
    )
    
    # Reflexive pronouns
    REFLEXIVE = (
        PronounConcept("myself", "myself", PronounCategory.REFLEXIVE, "First person reflexive", ("myself",)),
        PronounConcept("yourself", "yourself", PronounCategory.REFLEXIVE, "Second person reflexive", ("yourself",)),
        PronounConcept("himself", "himself", PronounCategory.REFLEXIVE, "Third person masculine reflexive", ("himself",)),
        PronounConcept("herself", "herself", PronounCategory.REFLEXIVE, "Third person feminine reflexive", ("herself",)),
        PronounConcept("itself", "itself", PronounCategory.REFLEXIVE, "Third person neuter reflexive", ("itself",)),
        PronounConcept("ourselves", "ourselves", PronounCategory.REFLEXIVE, "First person plural reflexive", ("ourselves",)),
        PronounConcept("themselves", "themselves", PronounCategory.REFLEXIVE, "Third person plural reflexive", ("themselves",)),
    )
    
    # Demonstrative pronouns
    DEMONSTRATIVE = (
        PronounConcept("this_dem", "this", PronounCategory.DEMONSTRATIVE, "Near singular", ("this",)),
        PronounConcept("that_dem", "that", PronounCategory.DEMONSTRATIVE, "Far singular", ("that",)),
        PronounConcept("these", "these", PronounCategory.DEMONSTRATIVE, "Near plural", ("these",)),
        PronounConcept("those", "those", PronounCategory.DEMONSTRATIVE, "Far plural", ("those",)),
        PronounConcept("here_dem", "here", PronounCategory.DEMONSTRATIVE, "Near location", ("here",)),
        PronounConcept("there_dem", "there", PronounCategory.DEMONSTRATIVE, "Far location", ("there",)),
    )
    
    # Interrogative pronouns
    INTERROGATIVE = (
        PronounConcept("who_int", "who", PronounCategory.INTERROGATIVE, "Person question", ("who",)),
        PronounConcept("what", "what", PronounCategory.INTERROGATIVE, "Thing question", ("what",)),
        PronounConcept("where", "where", PronounCategory.INTERROGATIVE, "Place question", ("where",)),
        PronounConcept("when", "when", PronounCategory.INTERROGATIVE, "Time question", ("when",)),
        PronounConcept("why", "why", PronounCategory.INTERROGATIVE, "Reason question", ("why",)),
        PronounConcept("how", "how", PronounCategory.INTERROGATIVE, "Manner question", ("how",)),
        PronounConcept("which", "which", PronounCategory.INTERROGATIVE, "Selection question", ("which",)),
    )
    
    # Indefinite pronouns
    INDEFINITE = (
        PronounConcept("someone", "someone", PronounCategory.INDEFINITE, "Unknown person", ("someone", "somebody",)),
        PronounConcept("anyone", "anyone", PronounCategory.INDEFINITE, "Any person", ("anyone", "anybody",)),
        PronounConcept("everyone", "everyone", PronounCategory.INDEFINITE, "All persons", ("everyone", "everybody",)),
        PronounConcept("nobody_indef", "nobody", PronounCategory.INDEFINITE, "No person", ("nobody", "no one",)),
        PronounConcept("something", "something", PronounCategory.INDEFINITE, "Unknown thing", ("something",)),
        PronounConcept("anything", "anything", PronounCategory.INDEFINITE, "Any thing", ("anything",)),
        PronounConcept("everything", "everything", PronounCategory.INDEFINITE, "All things", ("everything",)),
        PronounConcept("nothing", "nothing", PronounCategory.INDEFINITE, "No thing", ("nothing",)),
    )
    
    # Relative pronouns
    RELATIVE = (
        PronounConcept("who_rel", "who", PronounCategory.RELATIVE, "Person relative", ("who",)),
        PronounConcept("whom", "whom", PronounCategory.RELATIVE, "Object person relative", ("whom",)),
        PronounConcept("whose", "whose", PronounCategory.RELATIVE, "Possession relative", ("whose",)),
        PronounConcept("which_rel", "which", PronounCategory.RELATIVE, "Thing relative", ("which",)),
        PronounConcept("that_rel", "that", PronounCategory.RELATIVE, "General relative", ("that",)),
        PronounConcept("where_rel", "where", PronounCategory.RELATIVE, "Place relative", ("where",)),
    )
    
    @classmethod
    def all_concepts(cls) -> list[PronounConcept]:
        """Get all pronoun concepts."""
        concepts: list[PronounConcept] = []
        concepts.extend(cls.PERSONAL)
        concepts.extend(cls.POSSESSIVE)
        concepts.extend(cls.REFLEXIVE)
        concepts.extend(cls.DEMONSTRATIVE)
        concepts.extend(cls.INTERROGATIVE)
        concepts.extend(cls.INDEFINITE)
        concepts.extend(cls.RELATIVE)
        return concepts
    
    @classmethod
    def concepts_by_category(cls, category: PronounCategory) -> list[PronounConcept]:
        """Get concepts for a specific category."""
        return [c for c in cls.all_concepts() if c.category == category]
