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
Numeric Atlas - Quantitative concepts for cross-dimensional alignment.

Covers numerical and mathematical concepts:
- Cardinal numbers (one through twenty, hundred, thousand, million)
- Ordinal numbers (first, second, last)
- Comparatives (more, less, equal)
- Mathematical operations (add, subtract, multiply)

Total: ~50 probes for numeric grounding.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NumericCategory(str, Enum):
    """Category of numeric concept."""
    
    CARDINAL = "cardinal"
    ORDINAL = "ordinal"
    COMPARATIVE = "comparative"
    OPERATION = "operation"


@dataclass(frozen=True)
class NumericConcept:
    """A numeric/quantitative concept."""
    
    id: str
    name: str
    category: NumericCategory
    description: str
    support_texts: tuple[str, ...]
    cross_domain_weight: float = 1.0


class NumericConceptInventory:
    """Inventory of numeric concepts."""
    
    # Cardinal numbers
    CARDINALS = (
        NumericConcept("one", "one", NumericCategory.CARDINAL, "The number 1", ("one", "1", "single")),
        NumericConcept("two", "two", NumericCategory.CARDINAL, "The number 2", ("two", "2", "pair")),
        NumericConcept("three", "three", NumericCategory.CARDINAL, "The number 3", ("three", "3", "triple")),
        NumericConcept("four", "four", NumericCategory.CARDINAL, "The number 4", ("four", "4", "quartet")),
        NumericConcept("five", "five", NumericCategory.CARDINAL, "The number 5", ("five", "5", "quintet")),
        NumericConcept("six", "six", NumericCategory.CARDINAL, "The number 6", ("six", "6", "half dozen")),
        NumericConcept("seven", "seven", NumericCategory.CARDINAL, "The number 7", ("seven", "7",)),
        NumericConcept("eight", "eight", NumericCategory.CARDINAL, "The number 8", ("eight", "8",)),
        NumericConcept("nine", "nine", NumericCategory.CARDINAL, "The number 9", ("nine", "9",)),
        NumericConcept("ten", "ten", NumericCategory.CARDINAL, "The number 10", ("ten", "10", "decade")),
        NumericConcept("eleven", "eleven", NumericCategory.CARDINAL, "The number 11", ("eleven", "11",)),
        NumericConcept("twelve", "twelve", NumericCategory.CARDINAL, "The number 12", ("twelve", "12", "dozen")),
        NumericConcept("thirteen", "thirteen", NumericCategory.CARDINAL, "The number 13", ("thirteen", "13",)),
        NumericConcept("fourteen", "fourteen", NumericCategory.CARDINAL, "The number 14", ("fourteen", "14",)),
        NumericConcept("fifteen", "fifteen", NumericCategory.CARDINAL, "The number 15", ("fifteen", "15",)),
        NumericConcept("sixteen", "sixteen", NumericCategory.CARDINAL, "The number 16", ("sixteen", "16",)),
        NumericConcept("seventeen", "seventeen", NumericCategory.CARDINAL, "The number 17", ("seventeen", "17",)),
        NumericConcept("eighteen", "eighteen", NumericCategory.CARDINAL, "The number 18", ("eighteen", "18",)),
        NumericConcept("nineteen", "nineteen", NumericCategory.CARDINAL, "The number 19", ("nineteen", "19",)),
        NumericConcept("twenty", "twenty", NumericCategory.CARDINAL, "The number 20", ("twenty", "20", "score")),
        NumericConcept("hundred", "hundred", NumericCategory.CARDINAL, "The number 100", ("hundred", "100", "century")),
        NumericConcept("thousand", "thousand", NumericCategory.CARDINAL, "The number 1000", ("thousand", "1000", "grand")),
        NumericConcept("million", "million", NumericCategory.CARDINAL, "The number 1000000", ("million", "1000000",)),
        NumericConcept("zero", "zero", NumericCategory.CARDINAL, "The number 0", ("zero", "0", "none")),
        NumericConcept("half", "half", NumericCategory.CARDINAL, "One divided by two", ("half", "1/2", "fifty percent")),
    )
    
    # Ordinal numbers
    ORDINALS = (
        NumericConcept("first", "first", NumericCategory.ORDINAL, "Position 1", ("first", "1st", "initial")),
        NumericConcept("second", "second", NumericCategory.ORDINAL, "Position 2", ("second", "2nd",)),
        NumericConcept("third", "third", NumericCategory.ORDINAL, "Position 3", ("third", "3rd",)),
        NumericConcept("fourth", "fourth", NumericCategory.ORDINAL, "Position 4", ("fourth", "4th",)),
        NumericConcept("fifth", "fifth", NumericCategory.ORDINAL, "Position 5", ("fifth", "5th",)),
        NumericConcept("last", "last", NumericCategory.ORDINAL, "Final position", ("last", "final", "ultimate")),
        NumericConcept("next", "next", NumericCategory.ORDINAL, "Following position", ("next", "following", "subsequent")),
        NumericConcept("previous", "previous", NumericCategory.ORDINAL, "Preceding position", ("previous", "prior", "preceding")),
        NumericConcept("middle", "middle", NumericCategory.ORDINAL, "Center position", ("middle", "center", "median")),
        NumericConcept("beginning", "beginning", NumericCategory.ORDINAL, "Start position", ("beginning", "start", "origin")),
    )
    
    # Comparatives
    COMPARATIVES = (
        NumericConcept("more", "more", NumericCategory.COMPARATIVE, "Greater quantity", ("more", "additional", "extra")),
        NumericConcept("less", "less", NumericCategory.COMPARATIVE, "Smaller quantity", ("less", "fewer", "reduced")),
        NumericConcept("equal", "equal", NumericCategory.COMPARATIVE, "Same quantity", ("equal", "same", "equivalent")),
        NumericConcept("greater", "greater", NumericCategory.COMPARATIVE, "Larger than", ("greater", "larger", "bigger")),
        NumericConcept("smaller", "smaller", NumericCategory.COMPARATIVE, "Less than", ("smaller", "lesser", "tinier")),
        NumericConcept("most", "most", NumericCategory.COMPARATIVE, "Maximum quantity", ("most", "maximum", "greatest")),
        NumericConcept("least", "least", NumericCategory.COMPARATIVE, "Minimum quantity", ("least", "minimum", "smallest")),
        NumericConcept("enough", "enough", NumericCategory.COMPARATIVE, "Sufficient quantity", ("enough", "sufficient", "adequate")),
    )
    
    # Mathematical operations
    OPERATIONS = (
        NumericConcept("add", "add", NumericCategory.OPERATION, "Combine quantities", ("add", "plus", "sum")),
        NumericConcept("subtract", "subtract", NumericCategory.OPERATION, "Remove quantity", ("subtract", "minus", "difference")),
        NumericConcept("multiply", "multiply", NumericCategory.OPERATION, "Repeated addition", ("multiply", "times", "product")),
        NumericConcept("divide", "divide", NumericCategory.OPERATION, "Split quantity", ("divide", "split", "quotient")),
        NumericConcept("calculate", "calculate", NumericCategory.OPERATION, "Compute result", ("calculate", "compute", "figure")),
        NumericConcept("count", "count", NumericCategory.OPERATION, "Enumerate items", ("count", "tally", "enumerate")),
        NumericConcept("measure", "measure", NumericCategory.OPERATION, "Determine size", ("measure", "gauge", "assess")),
    )
    
    @classmethod
    def all_concepts(cls) -> list[NumericConcept]:
        """Get all numeric concepts."""
        concepts: list[NumericConcept] = []
        concepts.extend(cls.CARDINALS)
        concepts.extend(cls.ORDINALS)
        concepts.extend(cls.COMPARATIVES)
        concepts.extend(cls.OPERATIONS)
        return concepts
    
    @classmethod
    def concepts_by_category(cls, category: NumericCategory) -> list[NumericConcept]:
        """Get concepts for a specific category."""
        return [c for c in cls.all_concepts() if c.category == category]
