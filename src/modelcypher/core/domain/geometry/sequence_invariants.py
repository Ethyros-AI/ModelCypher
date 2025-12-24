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
Sequence Invariants for Extended Probing.

Provides probe texts based on mathematical sequences (Fibonacci, Lucas, Primes, Catalan)
for testing model understanding of numerical patterns and relationships.

These invariants complement semantic primes by testing computational/mathematical
dimensions of model representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SequenceFamily(str, Enum):
    """Families of mathematical sequences for probing."""

    FIBONACCI = "fibonacci"
    LUCAS = "lucas"
    PRIMES = "primes"
    CATALAN = "catalan"


@dataclass(frozen=True)
class SequenceProbe:
    """A probe text derived from a mathematical sequence."""

    sequence_family: SequenceFamily
    sequence_index: int
    sequence_value: int
    probe_text: str
    probe_id: str

    @property
    def category(self) -> str:
        return f"sequence_{self.sequence_family.value}"


# =============================================================================
# Sequence Generators
# =============================================================================


def fibonacci(n: int) -> int:
    """Compute the nth Fibonacci number (0-indexed)."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def lucas(n: int) -> int:
    """Compute the nth Lucas number (0-indexed)."""
    if n == 0:
        return 2
    if n == 1:
        return 1
    a, b = 2, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def is_prime(n: int) -> bool:
    """Check if n is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def nth_prime(n: int) -> int:
    """Compute the nth prime number (1-indexed: nth_prime(1) = 2)."""
    if n <= 0:
        return 2
    count = 0
    candidate = 1
    while count < n:
        candidate += 1
        if is_prime(candidate):
            count += 1
    return candidate


def catalan(n: int) -> int:
    """Compute the nth Catalan number (0-indexed)."""
    if n <= 1:
        return 1
    # C(n) = C(2n, n) / (n + 1)
    # Using iterative computation to avoid large factorials
    result = 1
    for i in range(n):
        result = result * 2 * (2 * i + 1) // (i + 2)
    return result


# =============================================================================
# Probe Text Generation
# =============================================================================


def fibonacci_probe_text(n: int) -> str:
    """Generate probe text for Fibonacci sequence."""
    value = fibonacci(n)
    templates = [
        f"The {n}th Fibonacci number is {value}.",
        f"In the Fibonacci sequence, position {n} contains {value}.",
        f"Fibonacci({n}) equals {value}.",
        f"The sequence 1, 1, 2, 3, 5, 8... at position {n} is {value}.",
    ]
    return templates[n % len(templates)]


def lucas_probe_text(n: int) -> str:
    """Generate probe text for Lucas sequence."""
    value = lucas(n)
    templates = [
        f"The {n}th Lucas number is {value}.",
        f"In the Lucas sequence, position {n} contains {value}.",
        f"Lucas({n}) equals {value}.",
        f"The sequence 2, 1, 3, 4, 7, 11... at position {n} is {value}.",
    ]
    return templates[n % len(templates)]


def primes_probe_text(n: int) -> str:
    """Generate probe text for prime numbers."""
    value = nth_prime(n)
    templates = [
        f"The {n}th prime number is {value}.",
        f"Prime number {n} in sequence is {value}.",
        f"The prime at position {n} is {value}.",
        f"{value} is the {n}th prime number.",
    ]
    return templates[n % len(templates)]


def catalan_probe_text(n: int) -> str:
    """Generate probe text for Catalan numbers."""
    value = catalan(n)
    templates = [
        f"The {n}th Catalan number is {value}.",
        f"Catalan({n}) equals {value}.",
        f"The number of ways to triangulate a {n+2}-gon is {value}.",
        f"The {n}th Catalan number, counting valid parentheses, is {value}.",
    ]
    return templates[n % len(templates)]


# =============================================================================
# Probe Generation
# =============================================================================


def generate_sequence_probes(
    family: SequenceFamily,
    start: int = 1,
    count: int = 10,
) -> list[SequenceProbe]:
    """
    Generate a list of sequence probes for a given family.

    Args:
        family: Which sequence family to use
        start: Starting index (1-indexed for primes, 0-indexed for others)
        count: Number of probes to generate

    Returns:
        List of SequenceProbe objects
    """
    probes = []

    for i in range(count):
        n = start + i

        if family == SequenceFamily.FIBONACCI:
            value = fibonacci(n)
            text = fibonacci_probe_text(n)
        elif family == SequenceFamily.LUCAS:
            value = lucas(n)
            text = lucas_probe_text(n)
        elif family == SequenceFamily.PRIMES:
            value = nth_prime(n)
            text = primes_probe_text(n)
        elif family == SequenceFamily.CATALAN:
            value = catalan(n)
            text = catalan_probe_text(n)
        else:
            continue

        probe_id = f"{family.value}_{n}"

        probes.append(
            SequenceProbe(
                sequence_family=family,
                sequence_index=n,
                sequence_value=value,
                probe_text=text,
                probe_id=probe_id,
            )
        )

    return probes


def generate_all_sequence_probes(
    families: set[SequenceFamily] | None = None,
    count_per_family: int = 10,
) -> list[SequenceProbe]:
    """
    Generate probes for all (or specified) sequence families.

    Args:
        families: Set of families to include (None = all)
        count_per_family: Number of probes per family

    Returns:
        List of all sequence probes
    """
    if families is None:
        families = set(SequenceFamily)

    all_probes = []

    for family in families:
        probes = generate_sequence_probes(
            family=family,
            start=1,
            count=count_per_family,
        )
        all_probes.extend(probes)

    return all_probes


@dataclass
class SequenceInvariantConfig:
    """Configuration for sequence invariant probing."""

    families: set[SequenceFamily] = None  # None = all families
    count_per_family: int = 10
    start_index: int = 1

    def __post_init__(self):
        if self.families is None:
            self.families = set(SequenceFamily)


# =============================================================================
# Invariant Anchors
# =============================================================================


@dataclass(frozen=True)
class SequenceInvariantAnchor:
    """
    An invariant anchor based on sequence relationships.

    These anchors test whether the model understands the relationship
    between sequence values, not just individual values.
    """

    family: SequenceFamily
    relationship: str  # e.g., "successor", "sum", "ratio"
    probe_text: str
    anchor_id: str


def generate_fibonacci_relationship_anchors() -> list[SequenceInvariantAnchor]:
    """Generate anchors testing Fibonacci relationships."""
    anchors = []

    # Successor relationship: F(n) + F(n-1) = F(n+1)
    for n in range(2, 10):
        f_n = fibonacci(n)
        f_n1 = fibonacci(n - 1)
        f_n2 = fibonacci(n + 1)
        text = f"{f_n1} plus {f_n} equals {f_n2} in the Fibonacci sequence."
        anchors.append(
            SequenceInvariantAnchor(
                family=SequenceFamily.FIBONACCI,
                relationship="successor",
                probe_text=text,
                anchor_id=f"fib_successor_{n}",
            )
        )

    # Golden ratio approximation
    for n in range(5, 12):
        f_n = fibonacci(n)
        f_n1 = fibonacci(n - 1)
        ratio = f_n / f_n1 if f_n1 > 0 else 0
        text = f"The ratio {f_n}/{f_n1} is approximately {ratio:.4f}, approaching phi."
        anchors.append(
            SequenceInvariantAnchor(
                family=SequenceFamily.FIBONACCI,
                relationship="golden_ratio",
                probe_text=text,
                anchor_id=f"fib_ratio_{n}",
            )
        )

    return anchors


def generate_prime_relationship_anchors() -> list[SequenceInvariantAnchor]:
    """Generate anchors testing prime number relationships."""
    anchors = []

    # Twin primes
    twin_primes = [(3, 5), (5, 7), (11, 13), (17, 19), (29, 31), (41, 43)]
    for i, (p1, p2) in enumerate(twin_primes):
        text = f"{p1} and {p2} are twin primes, differing by 2."
        anchors.append(
            SequenceInvariantAnchor(
                family=SequenceFamily.PRIMES,
                relationship="twin_primes",
                probe_text=text,
                anchor_id=f"prime_twin_{i}",
            )
        )

    # Prime gaps
    for n in range(1, 8):
        p1 = nth_prime(n)
        p2 = nth_prime(n + 1)
        gap = p2 - p1
        text = f"The gap between prime {p1} and {p2} is {gap}."
        anchors.append(
            SequenceInvariantAnchor(
                family=SequenceFamily.PRIMES,
                relationship="prime_gap",
                probe_text=text,
                anchor_id=f"prime_gap_{n}",
            )
        )

    return anchors


def generate_all_sequence_anchors() -> list[SequenceInvariantAnchor]:
    """Generate all sequence invariant anchors."""
    anchors = []
    anchors.extend(generate_fibonacci_relationship_anchors())
    anchors.extend(generate_prime_relationship_anchors())
    return anchors
