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
Prime Number Atlas - Comprehensive prime number knowledge space for geometric analysis.

Maps the full "prime" conceptual space including:
- Prime numerals (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, ...)
- Prime words (two, three, five, seven, eleven, ...)
- Composite numerals (4, 6, 8, 9, 10, 12, 14, 15, ...)
- Composite words (four, six, eight, nine, ...)
- Primality concepts (prime, composite, indivisible, factorizable)
- Number theory concepts (divisor, factor, multiple, coprime)
- Mathematical operations (factorize, divide, modulo)
- Famous primes (Mersenne, twin primes, Sophie Germain)

The atlas enables geometric analysis of how LLMs encode prime structure
in their invariant representation space. CKA = 1.0 guarantees the relational
structure is identical across models - the question is whether primes form
a distinct geometric pattern within that invariant structure.

Total: ~120 probes covering the prime knowledge space.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PrimeCategory(str, Enum):
    """Category of prime-related concept."""

    PRIME_NUMERAL = "prime_numeral"  # Prime numbers as digits: 2, 3, 5, 7, ...
    PRIME_WORD = "prime_word"  # Prime numbers as words: two, three, five, ...
    COMPOSITE_NUMERAL = "composite_numeral"  # Composite numbers: 4, 6, 8, 9, ...
    COMPOSITE_WORD = "composite_word"  # Composite as words: four, six, ...
    PRIMALITY = "primality"  # Concepts: prime, composite, indivisible
    NUMBER_THEORY = "number_theory"  # Divisor, factor, multiple, coprime
    OPERATIONS = "operations"  # Factorize, divide, modulo
    FAMOUS_PRIMES = "famous_primes"  # Mersenne, twin, Sophie Germain
    SPECIAL_NUMBERS = "special_numbers"  # 0, 1, infinity


@dataclass(frozen=True)
class PrimeConcept:
    """A prime-related concept for probing."""

    id: str
    name: str
    category: PrimeCategory
    description: str
    support_texts: tuple[str, ...]
    is_prime: bool | None = None  # True for primes, False for composites, None for concepts
    cross_domain_weight: float = 1.0


class PrimeConceptInventory:
    """Inventory of prime-related concepts."""

    # First 25 prime numbers as numerals
    PRIME_NUMERALS = (
        PrimeConcept("p2", "2", PrimeCategory.PRIME_NUMERAL, "First prime", ("2", "the number 2"), is_prime=True),
        PrimeConcept("p3", "3", PrimeCategory.PRIME_NUMERAL, "Second prime", ("3", "the number 3"), is_prime=True),
        PrimeConcept("p5", "5", PrimeCategory.PRIME_NUMERAL, "Third prime", ("5", "the number 5"), is_prime=True),
        PrimeConcept("p7", "7", PrimeCategory.PRIME_NUMERAL, "Fourth prime", ("7", "the number 7"), is_prime=True),
        PrimeConcept("p11", "11", PrimeCategory.PRIME_NUMERAL, "Fifth prime", ("11", "the number 11"), is_prime=True),
        PrimeConcept("p13", "13", PrimeCategory.PRIME_NUMERAL, "Sixth prime", ("13", "the number 13"), is_prime=True),
        PrimeConcept("p17", "17", PrimeCategory.PRIME_NUMERAL, "Seventh prime", ("17", "the number 17"), is_prime=True),
        PrimeConcept("p19", "19", PrimeCategory.PRIME_NUMERAL, "Eighth prime", ("19", "the number 19"), is_prime=True),
        PrimeConcept("p23", "23", PrimeCategory.PRIME_NUMERAL, "Ninth prime", ("23", "the number 23"), is_prime=True),
        PrimeConcept("p29", "29", PrimeCategory.PRIME_NUMERAL, "Tenth prime", ("29", "the number 29"), is_prime=True),
        PrimeConcept("p31", "31", PrimeCategory.PRIME_NUMERAL, "Eleventh prime", ("31", "the number 31"), is_prime=True),
        PrimeConcept("p37", "37", PrimeCategory.PRIME_NUMERAL, "Twelfth prime", ("37", "the number 37"), is_prime=True),
        PrimeConcept("p41", "41", PrimeCategory.PRIME_NUMERAL, "Thirteenth prime", ("41", "the number 41"), is_prime=True),
        PrimeConcept("p43", "43", PrimeCategory.PRIME_NUMERAL, "Fourteenth prime", ("43", "the number 43"), is_prime=True),
        PrimeConcept("p47", "47", PrimeCategory.PRIME_NUMERAL, "Fifteenth prime", ("47", "the number 47"), is_prime=True),
        PrimeConcept("p53", "53", PrimeCategory.PRIME_NUMERAL, "Sixteenth prime", ("53", "the number 53"), is_prime=True),
        PrimeConcept("p59", "59", PrimeCategory.PRIME_NUMERAL, "Seventeenth prime", ("59", "the number 59"), is_prime=True),
        PrimeConcept("p61", "61", PrimeCategory.PRIME_NUMERAL, "Eighteenth prime", ("61", "the number 61"), is_prime=True),
        PrimeConcept("p67", "67", PrimeCategory.PRIME_NUMERAL, "Nineteenth prime", ("67", "the number 67"), is_prime=True),
        PrimeConcept("p71", "71", PrimeCategory.PRIME_NUMERAL, "Twentieth prime", ("71", "the number 71"), is_prime=True),
        PrimeConcept("p73", "73", PrimeCategory.PRIME_NUMERAL, "Twenty-first prime", ("73", "the number 73"), is_prime=True),
        PrimeConcept("p79", "79", PrimeCategory.PRIME_NUMERAL, "Twenty-second prime", ("79", "the number 79"), is_prime=True),
        PrimeConcept("p83", "83", PrimeCategory.PRIME_NUMERAL, "Twenty-third prime", ("83", "the number 83"), is_prime=True),
        PrimeConcept("p89", "89", PrimeCategory.PRIME_NUMERAL, "Twenty-fourth prime", ("89", "the number 89"), is_prime=True),
        PrimeConcept("p97", "97", PrimeCategory.PRIME_NUMERAL, "Twenty-fifth prime", ("97", "the number 97"), is_prime=True),
    )

    # Prime numbers as words
    PRIME_WORDS = (
        PrimeConcept("pw_two", "two", PrimeCategory.PRIME_WORD, "First prime as word", ("two", "the number two"), is_prime=True),
        PrimeConcept("pw_three", "three", PrimeCategory.PRIME_WORD, "Second prime as word", ("three", "the number three"), is_prime=True),
        PrimeConcept("pw_five", "five", PrimeCategory.PRIME_WORD, "Third prime as word", ("five", "the number five"), is_prime=True),
        PrimeConcept("pw_seven", "seven", PrimeCategory.PRIME_WORD, "Fourth prime as word", ("seven", "the number seven"), is_prime=True),
        PrimeConcept("pw_eleven", "eleven", PrimeCategory.PRIME_WORD, "Fifth prime as word", ("eleven", "the number eleven"), is_prime=True),
        PrimeConcept("pw_thirteen", "thirteen", PrimeCategory.PRIME_WORD, "Sixth prime as word", ("thirteen", "the number thirteen"), is_prime=True),
        PrimeConcept("pw_seventeen", "seventeen", PrimeCategory.PRIME_WORD, "Seventh prime as word", ("seventeen", "the number seventeen"), is_prime=True),
        PrimeConcept("pw_nineteen", "nineteen", PrimeCategory.PRIME_WORD, "Eighth prime as word", ("nineteen", "the number nineteen"), is_prime=True),
        PrimeConcept("pw_twenty_three", "twenty-three", PrimeCategory.PRIME_WORD, "Ninth prime as word", ("twenty-three", "the number twenty-three"), is_prime=True),
        PrimeConcept("pw_twenty_nine", "twenty-nine", PrimeCategory.PRIME_WORD, "Tenth prime as word", ("twenty-nine", "the number twenty-nine"), is_prime=True),
    )

    # First 25 composite numbers as numerals (matched to prime range)
    COMPOSITE_NUMERALS = (
        PrimeConcept("c4", "4", PrimeCategory.COMPOSITE_NUMERAL, "First composite", ("4", "the number 4"), is_prime=False),
        PrimeConcept("c6", "6", PrimeCategory.COMPOSITE_NUMERAL, "Second composite", ("6", "the number 6"), is_prime=False),
        PrimeConcept("c8", "8", PrimeCategory.COMPOSITE_NUMERAL, "Third composite", ("8", "the number 8"), is_prime=False),
        PrimeConcept("c9", "9", PrimeCategory.COMPOSITE_NUMERAL, "Fourth composite", ("9", "the number 9"), is_prime=False),
        PrimeConcept("c10", "10", PrimeCategory.COMPOSITE_NUMERAL, "Fifth composite", ("10", "the number 10"), is_prime=False),
        PrimeConcept("c12", "12", PrimeCategory.COMPOSITE_NUMERAL, "Sixth composite", ("12", "the number 12"), is_prime=False),
        PrimeConcept("c14", "14", PrimeCategory.COMPOSITE_NUMERAL, "Seventh composite", ("14", "the number 14"), is_prime=False),
        PrimeConcept("c15", "15", PrimeCategory.COMPOSITE_NUMERAL, "Eighth composite", ("15", "the number 15"), is_prime=False),
        PrimeConcept("c16", "16", PrimeCategory.COMPOSITE_NUMERAL, "Ninth composite", ("16", "the number 16"), is_prime=False),
        PrimeConcept("c18", "18", PrimeCategory.COMPOSITE_NUMERAL, "Tenth composite", ("18", "the number 18"), is_prime=False),
        PrimeConcept("c20", "20", PrimeCategory.COMPOSITE_NUMERAL, "Eleventh composite", ("20", "the number 20"), is_prime=False),
        PrimeConcept("c21", "21", PrimeCategory.COMPOSITE_NUMERAL, "Twelfth composite", ("21", "the number 21"), is_prime=False),
        PrimeConcept("c22", "22", PrimeCategory.COMPOSITE_NUMERAL, "Thirteenth composite", ("22", "the number 22"), is_prime=False),
        PrimeConcept("c24", "24", PrimeCategory.COMPOSITE_NUMERAL, "Fourteenth composite", ("24", "the number 24"), is_prime=False),
        PrimeConcept("c25", "25", PrimeCategory.COMPOSITE_NUMERAL, "Fifteenth composite", ("25", "the number 25"), is_prime=False),
        PrimeConcept("c26", "26", PrimeCategory.COMPOSITE_NUMERAL, "Sixteenth composite", ("26", "the number 26"), is_prime=False),
        PrimeConcept("c27", "27", PrimeCategory.COMPOSITE_NUMERAL, "Seventeenth composite", ("27", "the number 27"), is_prime=False),
        PrimeConcept("c28", "28", PrimeCategory.COMPOSITE_NUMERAL, "Eighteenth composite", ("28", "the number 28"), is_prime=False),
        PrimeConcept("c30", "30", PrimeCategory.COMPOSITE_NUMERAL, "Nineteenth composite", ("30", "the number 30"), is_prime=False),
        PrimeConcept("c32", "32", PrimeCategory.COMPOSITE_NUMERAL, "Twentieth composite", ("32", "the number 32"), is_prime=False),
        PrimeConcept("c33", "33", PrimeCategory.COMPOSITE_NUMERAL, "Twenty-first composite", ("33", "the number 33"), is_prime=False),
        PrimeConcept("c34", "34", PrimeCategory.COMPOSITE_NUMERAL, "Twenty-second composite", ("34", "the number 34"), is_prime=False),
        PrimeConcept("c35", "35", PrimeCategory.COMPOSITE_NUMERAL, "Twenty-third composite", ("35", "the number 35"), is_prime=False),
        PrimeConcept("c36", "36", PrimeCategory.COMPOSITE_NUMERAL, "Twenty-fourth composite", ("36", "the number 36"), is_prime=False),
        PrimeConcept("c38", "38", PrimeCategory.COMPOSITE_NUMERAL, "Twenty-fifth composite", ("38", "the number 38"), is_prime=False),
    )

    # Composite numbers as words
    COMPOSITE_WORDS = (
        PrimeConcept("cw_four", "four", PrimeCategory.COMPOSITE_WORD, "First composite as word", ("four", "the number four"), is_prime=False),
        PrimeConcept("cw_six", "six", PrimeCategory.COMPOSITE_WORD, "Second composite as word", ("six", "the number six"), is_prime=False),
        PrimeConcept("cw_eight", "eight", PrimeCategory.COMPOSITE_WORD, "Third composite as word", ("eight", "the number eight"), is_prime=False),
        PrimeConcept("cw_nine", "nine", PrimeCategory.COMPOSITE_WORD, "Fourth composite as word", ("nine", "the number nine"), is_prime=False),
        PrimeConcept("cw_ten", "ten", PrimeCategory.COMPOSITE_WORD, "Fifth composite as word", ("ten", "the number ten"), is_prime=False),
        PrimeConcept("cw_twelve", "twelve", PrimeCategory.COMPOSITE_WORD, "Sixth composite as word", ("twelve", "the number twelve"), is_prime=False),
        PrimeConcept("cw_fourteen", "fourteen", PrimeCategory.COMPOSITE_WORD, "Seventh composite as word", ("fourteen", "the number fourteen"), is_prime=False),
        PrimeConcept("cw_fifteen", "fifteen", PrimeCategory.COMPOSITE_WORD, "Eighth composite as word", ("fifteen", "the number fifteen"), is_prime=False),
        PrimeConcept("cw_sixteen", "sixteen", PrimeCategory.COMPOSITE_WORD, "Ninth composite as word", ("sixteen", "the number sixteen"), is_prime=False),
        PrimeConcept("cw_eighteen", "eighteen", PrimeCategory.COMPOSITE_WORD, "Tenth composite as word", ("eighteen", "the number eighteen"), is_prime=False),
    )

    # Primality concepts
    PRIMALITY = (
        PrimeConcept("prime", "prime", PrimeCategory.PRIMALITY, "A prime number", ("prime", "prime number", "indivisible")),
        PrimeConcept("composite", "composite", PrimeCategory.PRIMALITY, "A composite number", ("composite", "composite number", "non-prime")),
        PrimeConcept("indivisible", "indivisible", PrimeCategory.PRIMALITY, "Cannot be divided evenly", ("indivisible", "not divisible", "atomic")),
        PrimeConcept("divisible", "divisible", PrimeCategory.PRIMALITY, "Can be divided evenly", ("divisible", "evenly divided", "factorable")),
        PrimeConcept("factorizable", "factorizable", PrimeCategory.PRIMALITY, "Can be expressed as product", ("factorizable", "has factors", "can be factored")),
        PrimeConcept("irreducible", "irreducible", PrimeCategory.PRIMALITY, "Cannot be reduced", ("irreducible", "cannot reduce", "minimal")),
        PrimeConcept("primality", "primality", PrimeCategory.PRIMALITY, "The property of being prime", ("primality", "prime property", "is prime")),
    )

    # Number theory concepts
    NUMBER_THEORY = (
        PrimeConcept("divisor", "divisor", PrimeCategory.NUMBER_THEORY, "A number that divides evenly", ("divisor", "divides", "factor of")),
        PrimeConcept("factor", "factor", PrimeCategory.NUMBER_THEORY, "A divisor of a number", ("factor", "prime factor", "factorization")),
        PrimeConcept("multiple", "multiple", PrimeCategory.NUMBER_THEORY, "Product with an integer", ("multiple", "times", "product")),
        PrimeConcept("coprime", "coprime", PrimeCategory.NUMBER_THEORY, "Share no common factors", ("coprime", "relatively prime", "no common divisor")),
        PrimeConcept("gcd", "greatest common divisor", PrimeCategory.NUMBER_THEORY, "Largest shared factor", ("gcd", "greatest common divisor", "highest common factor")),
        PrimeConcept("lcm", "least common multiple", PrimeCategory.NUMBER_THEORY, "Smallest shared multiple", ("lcm", "least common multiple", "lowest common multiple")),
        PrimeConcept("modulo", "modulo", PrimeCategory.NUMBER_THEORY, "Remainder after division", ("modulo", "mod", "remainder")),
        PrimeConcept("congruent", "congruent", PrimeCategory.NUMBER_THEORY, "Same remainder mod n", ("congruent", "congruence", "equivalent mod")),
    )

    # Mathematical operations
    OPERATIONS = (
        PrimeConcept("factorize", "factorize", PrimeCategory.OPERATIONS, "Find prime factors", ("factorize", "factor", "decompose")),
        PrimeConcept("divide", "divide", PrimeCategory.OPERATIONS, "Split into equal parts", ("divide", "division", "quotient")),
        PrimeConcept("remainder", "remainder", PrimeCategory.OPERATIONS, "What's left after division", ("remainder", "leftover", "residue")),
        PrimeConcept("sieve", "sieve", PrimeCategory.OPERATIONS, "Filter primes from composites", ("sieve", "filter", "Eratosthenes")),
        PrimeConcept("test_primality", "primality test", PrimeCategory.OPERATIONS, "Check if prime", ("primality test", "is prime", "check prime")),
    )

    # Famous primes and prime types
    FAMOUS_PRIMES = (
        PrimeConcept("mersenne", "Mersenne prime", PrimeCategory.FAMOUS_PRIMES, "Prime of form 2^n - 1", ("Mersenne", "Mersenne prime", "2^n minus 1")),
        PrimeConcept("twin_prime", "twin prime", PrimeCategory.FAMOUS_PRIMES, "Primes differing by 2", ("twin prime", "twin primes", "prime pair")),
        PrimeConcept("sophie_germain", "Sophie Germain prime", PrimeCategory.FAMOUS_PRIMES, "p where 2p+1 is also prime", ("Sophie Germain", "safe prime", "Germain")),
        PrimeConcept("fermat", "Fermat prime", PrimeCategory.FAMOUS_PRIMES, "Prime of form 2^(2^n) + 1", ("Fermat prime", "Fermat number", "Fermat")),
        PrimeConcept("perfect_number", "perfect number", PrimeCategory.FAMOUS_PRIMES, "Sum of divisors equals itself", ("perfect number", "perfect", "6 28 496")),
        PrimeConcept("prime_gap", "prime gap", PrimeCategory.FAMOUS_PRIMES, "Distance between consecutive primes", ("prime gap", "gap", "distance between primes")),
    )

    # Special numbers (edge cases)
    SPECIAL_NUMBERS = (
        PrimeConcept("zero", "0", PrimeCategory.SPECIAL_NUMBERS, "Zero - neither prime nor composite", ("0", "zero", "nothing"), is_prime=None),
        PrimeConcept("one", "1", PrimeCategory.SPECIAL_NUMBERS, "One - the unit, not prime", ("1", "one", "unity"), is_prime=None),
        PrimeConcept("infinity", "infinity", PrimeCategory.SPECIAL_NUMBERS, "Unbounded - infinitely many primes", ("infinity", "infinite", "unbounded")),
    )

    @classmethod
    def all_concepts(cls) -> list[PrimeConcept]:
        """Get all prime-related concepts."""
        concepts: list[PrimeConcept] = []
        concepts.extend(cls.PRIME_NUMERALS)
        concepts.extend(cls.PRIME_WORDS)
        concepts.extend(cls.COMPOSITE_NUMERALS)
        concepts.extend(cls.COMPOSITE_WORDS)
        concepts.extend(cls.PRIMALITY)
        concepts.extend(cls.NUMBER_THEORY)
        concepts.extend(cls.OPERATIONS)
        concepts.extend(cls.FAMOUS_PRIMES)
        concepts.extend(cls.SPECIAL_NUMBERS)
        return concepts

    @classmethod
    def concepts_by_category(cls, category: PrimeCategory) -> list[PrimeConcept]:
        """Get concepts for a specific category."""
        return [c for c in cls.all_concepts() if c.category == category]

    @classmethod
    def primes_only(cls) -> list[PrimeConcept]:
        """Get only prime number concepts (numerals and words)."""
        return [c for c in cls.all_concepts() if c.is_prime is True]

    @classmethod
    def composites_only(cls) -> list[PrimeConcept]:
        """Get only composite number concepts (numerals and words)."""
        return [c for c in cls.all_concepts() if c.is_prime is False]

    @classmethod
    def numerals_only(cls) -> list[PrimeConcept]:
        """Get only numeral concepts (both prime and composite)."""
        return [
            c for c in cls.all_concepts()
            if c.category in (PrimeCategory.PRIME_NUMERAL, PrimeCategory.COMPOSITE_NUMERAL)
        ]

    @classmethod
    def words_only(cls) -> list[PrimeConcept]:
        """Get only word concepts (both prime and composite)."""
        return [
            c for c in cls.all_concepts()
            if c.category in (PrimeCategory.PRIME_WORD, PrimeCategory.COMPOSITE_WORD)
        ]

    @classmethod
    def conceptual_only(cls) -> list[PrimeConcept]:
        """Get only abstract concepts (primality, number theory, etc.)."""
        return [
            c for c in cls.all_concepts()
            if c.category in (
                PrimeCategory.PRIMALITY,
                PrimeCategory.NUMBER_THEORY,
                PrimeCategory.OPERATIONS,
                PrimeCategory.FAMOUS_PRIMES,
                PrimeCategory.SPECIAL_NUMBERS,
            )
        ]

    @classmethod
    def count(cls) -> int:
        """Total number of concepts."""
        return len(cls.all_concepts())
