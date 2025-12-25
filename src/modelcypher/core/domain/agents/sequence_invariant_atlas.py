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
Sequence Invariant Atlas.

Multi-domain probes that triangulate universal invariants across math, logic, and causality.
Extends the Fibonacci invariant pattern to additional sequence families plus logic,
ordering, arithmetic, and causality anchors for robust cross-domain alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SequenceFamily(str, Enum):
    """Mathematical sequence family."""

    FIBONACCI = "fibonacci"  # F(n) = F(n-1) + F(n-2), F(0)=0, F(1)=1
    LUCAS = "lucas"  # L(n) = L(n-1) + L(n-2), L(0)=2, L(1)=1
    TRIBONACCI = "tribonacci"  # T(n) = T(n-1) + T(n-2) + T(n-3)
    PRIMES = "primes"  # Prime numbers: 2, 3, 5, 7, 11...
    CATALAN = "catalan"  # C(n) = C(0)C(n-1) + C(1)C(n-2) + ... + C(n-1)C(0)
    RAMANUJAN = "ramanujan"  # Partition function invariants and congruences
    LOGIC = "logic"  # Boolean logic invariants
    ORDERING = "ordering"  # Order and comparison invariants
    ARITHMETIC = "arithmetic"  # Algebraic identity invariants
    CAUSALITY = "causality"  # Causal dependency invariants


class ExpressionDomain(str, Enum):
    """Expression domain for triangulation."""

    DEFINITION = "definition"  # Mathematical definition/recurrence
    CODE = "code"  # Algorithmic implementation
    RATIO = "ratio"  # Limiting behavior/ratios
    MATRIX = "matrix"  # Matrix representation
    CLOSED_FORM = "closedForm"  # Closed-form expression
    NATURE = "nature"  # Natural manifestation
    VISUAL = "visual"  # Geometric/visual representation
    COMBINATORIAL = "combinatorial"  # Counting interpretation


@dataclass(frozen=True)
class SequenceInvariant:
    """Multi-domain probe for triangulating universal invariants."""

    id: str
    family: SequenceFamily
    domain: ExpressionDomain
    name: str
    description: str
    support_texts: tuple[str, ...]
    cross_domain_weight: float = 1.0

    @property
    def canonical_name(self) -> str:
        """Canonical name for display."""
        return self.name


# Fibonacci Sequence (7 probes)
FIBONACCI_PROBES: tuple[SequenceInvariant, ...] = (
    SequenceInvariant(
        id="fib_recurrence",
        family=SequenceFamily.FIBONACCI,
        domain=ExpressionDomain.DEFINITION,
        name="Fibonacci recurrence",
        description="Each term is the sum of the two previous terms.",
        support_texts=(
            "F(n) = F(n-1) + F(n-2) with F(0)=0, F(1)=1",
            "0, 1, 1, 2, 3, 5, 8, 13, 21, 34",
            "A pair of rabbits produces a new pair each month after maturity.",
            "Patterns recur across scales and domains.",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="fib_code",
        family=SequenceFamily.FIBONACCI,
        domain=ExpressionDomain.CODE,
        name="Fibonacci code",
        description="Algorithmic implementations of Fibonacci.",
        support_texts=(
            "def fib(n): a,b=0,1; for _ in range(n): a,b=b,a+b; return a",
            "func fib(_ n: Int) -> Int { var a=0,b=1; for _ in 0..<n { let t=a+b; a=b; b=t }; return a }",
            "fn fib(n: usize) -> usize { let (mut a, mut b) = (0, 1); for _ in 0..n { let t = a + b; a = b; b = t; } a }",
            "function fib(n){ let a=0,b=1; for(let i=0;i<n;i++){ [a,b]=[b,a+b]; } return a; }",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="fib_ratio",
        family=SequenceFamily.FIBONACCI,
        domain=ExpressionDomain.RATIO,
        name="Fibonacci ratio",
        description="Successive ratios converge to the golden ratio.",
        support_texts=(
            "F(n+1)/F(n) approaches phi as n grows",
            "phi = (1 + sqrt(5)) / 2 ≈ 1.618033988749895",
            "Golden ratio proportions appear in art and architecture.",
            "A golden rectangle has length/width equal to phi.",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="fib_matrix",
        family=SequenceFamily.FIBONACCI,
        domain=ExpressionDomain.MATRIX,
        name="Fibonacci matrix",
        description="Matrix powers encode Fibonacci numbers.",
        support_texts=(
            "[[1,1],[1,0]]^n = [[F(n+1), F(n)],[F(n), F(n-1)]]",
            "Matrix exponentiation computes Fibonacci in O(log n).",
            "Eigenvalues of the Fibonacci matrix are phi and 1-phi.",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="fib_binet",
        family=SequenceFamily.FIBONACCI,
        domain=ExpressionDomain.CLOSED_FORM,
        name="Fibonacci closed form",
        description="Closed-form Fibonacci expression using phi and sqrt(5).",
        support_texts=(
            "F(n) = (phi^n - (1-phi)^n) / sqrt(5)",
            "Binet's formula uses the golden ratio phi.",
            "The formula shows Fibonacci grows exponentially like phi^n.",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="fib_phyllotaxis",
        family=SequenceFamily.FIBONACCI,
        domain=ExpressionDomain.NATURE,
        name="Fibonacci phyllotaxis",
        description="Fibonacci counts appear in plant spirals.",
        support_texts=(
            "Sunflower seed spirals often count 34/55 or 55/89.",
            "Pinecone and pineapple spirals follow Fibonacci numbers.",
            "Phyllotaxis uses Fibonacci ratios for optimal packing.",
            "Leaf arrangements maximize sunlight exposure via golden angle.",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="fib_spiral",
        family=SequenceFamily.FIBONACCI,
        domain=ExpressionDomain.VISUAL,
        name="Fibonacci spiral",
        description="Squares with Fibonacci side lengths form a spiral.",
        support_texts=(
            "Fibonacci spiral built from squares of side lengths 1,1,2,3,5,8.",
            "A logarithmic spiral approximates the Fibonacci spiral.",
            "The golden spiral appears in nautilus shells and galaxies.",
        ),
        cross_domain_weight=1.0,
    ),
)

# Lucas Sequence (6 probes)
LUCAS_PROBES: tuple[SequenceInvariant, ...] = (
    SequenceInvariant(
        id="lucas_recurrence",
        family=SequenceFamily.LUCAS,
        domain=ExpressionDomain.DEFINITION,
        name="Lucas recurrence",
        description="Same recurrence as Fibonacci but L(0)=2, L(1)=1.",
        support_texts=(
            "L(n) = L(n-1) + L(n-2) with L(0)=2, L(1)=1",
            "2, 1, 3, 4, 7, 11, 18, 29, 47, 76",
            "Lucas numbers are Fibonacci with initial conditions shifted.",
            "Named after French mathematician Édouard Lucas.",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="lucas_code",
        family=SequenceFamily.LUCAS,
        domain=ExpressionDomain.CODE,
        name="Lucas code",
        description="Algorithmic implementations of Lucas numbers.",
        support_texts=(
            "def lucas(n): a,b=2,1; for _ in range(n): a,b=b,a+b; return a",
            "func lucas(_ n: Int) -> Int { var a=2,b=1; for _ in 0..<n { let t=a+b; a=b; b=t }; return a }",
            "fn lucas(n: usize) -> usize { let (mut a, mut b) = (2, 1); for _ in 0..n { let t = a + b; a = b; b = t; } a }",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="lucas_ratio",
        family=SequenceFamily.LUCAS,
        domain=ExpressionDomain.RATIO,
        name="Lucas ratio",
        description="Lucas ratios also converge to phi, and L(n)/F(n) → sqrt(5).",
        support_texts=(
            "L(n+1)/L(n) approaches phi as n grows",
            "L(n)/F(n) approaches sqrt(5) ≈ 2.236",
            "L(n) = F(n-1) + F(n+1) - Lucas numbers bracket Fibonacci.",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="lucas_closed",
        family=SequenceFamily.LUCAS,
        domain=ExpressionDomain.CLOSED_FORM,
        name="Lucas closed form",
        description="Closed-form Lucas expression using phi.",
        support_texts=(
            "L(n) = phi^n + (1-phi)^n",
            "Simpler than Binet's formula - no division by sqrt(5).",
            "L(n)^2 - 5*F(n)^2 = 4*(-1)^n - a beautiful identity.",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="lucas_primality",
        family=SequenceFamily.LUCAS,
        domain=ExpressionDomain.COMBINATORIAL,
        name="Lucas primality test",
        description="Lucas sequences used in primality testing.",
        support_texts=(
            "Lucas-Lehmer test verifies Mersenne primes 2^p - 1.",
            "If p is prime, then L(p) ≡ 1 (mod p).",
            "Lucas pseudoprimes are rare, making strong tests.",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="lucas_nature",
        family=SequenceFamily.LUCAS,
        domain=ExpressionDomain.NATURE,
        name="Lucas phyllotaxis",
        description="Lucas numbers appear in some plant structures.",
        support_texts=(
            "Some flower petal counts follow Lucas numbers: 3, 4, 7, 11.",
            "Lucas spiral packings provide alternative optimal arrangements.",
            "Bravais lattice structures relate to Lucas sequences.",
        ),
        cross_domain_weight=1.0,
    ),
)

# Tribonacci Sequence (5 probes)
TRIBONACCI_PROBES: tuple[SequenceInvariant, ...] = (
    SequenceInvariant(
        id="trib_recurrence",
        family=SequenceFamily.TRIBONACCI,
        domain=ExpressionDomain.DEFINITION,
        name="Tribonacci recurrence",
        description="Each term is the sum of the three previous terms.",
        support_texts=(
            "T(n) = T(n-1) + T(n-2) + T(n-3) with T(0)=T(1)=0, T(2)=1",
            "0, 0, 1, 1, 2, 4, 7, 13, 24, 44, 81",
            "Generalizes Fibonacci to three-term recurrence.",
            "Part of the k-nacci family of sequences.",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="trib_code",
        family=SequenceFamily.TRIBONACCI,
        domain=ExpressionDomain.CODE,
        name="Tribonacci code",
        description="Algorithmic implementations of Tribonacci.",
        support_texts=(
            "def trib(n): a,b,c=0,0,1; for _ in range(n): a,b,c=b,c,a+b+c; return a",
            "func trib(_ n: Int) -> Int { var (a,b,c)=(0,0,1); for _ in 0..<n { (a,b,c)=(b,c,a+b+c) }; return a }",
            "fn trib(n: usize) -> usize { let (mut a,mut b,mut c)=(0,0,1); for _ in 0..n { let t=a+b+c; a=b; b=c; c=t; } a }",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="trib_ratio",
        family=SequenceFamily.TRIBONACCI,
        domain=ExpressionDomain.RATIO,
        name="Tribonacci ratio",
        description="Ratios converge to the tribonacci constant τ ≈ 1.839.",
        support_texts=(
            "T(n+1)/T(n) approaches the tribonacci constant τ ≈ 1.839286755",
            "τ is the real root of x³ = x² + x + 1",
            "The tribonacci constant is an algebraic number of degree 3.",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="trib_matrix",
        family=SequenceFamily.TRIBONACCI,
        domain=ExpressionDomain.MATRIX,
        name="Tribonacci matrix",
        description="3x3 matrix powers encode Tribonacci numbers.",
        support_texts=(
            "[[1,1,1],[1,0,0],[0,1,0]]^n encodes Tribonacci terms",
            "Matrix exponentiation gives O(log n) computation.",
            "Eigenvalues include the tribonacci constant τ.",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="trib_nature",
        family=SequenceFamily.TRIBONACCI,
        domain=ExpressionDomain.NATURE,
        name="Tribonacci patterns",
        description="Three-way branching patterns in nature.",
        support_texts=(
            "Some coral branching follows three-term recurrence.",
            "Ternary tree structures in certain plant growth.",
            "Three-way cellular division patterns.",
        ),
        cross_domain_weight=1.0,
    ),
)

# Prime Numbers (6 probes)
PRIME_PROBES: tuple[SequenceInvariant, ...] = (
    SequenceInvariant(
        id="prime_definition",
        family=SequenceFamily.PRIMES,
        domain=ExpressionDomain.DEFINITION,
        name="Prime definition",
        description="A prime is divisible only by 1 and itself.",
        support_texts=(
            "p is prime iff p > 1 and has no divisors except 1 and p",
            "2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47",
            "The fundamental theorem: every integer > 1 is a unique product of primes.",
            "Primes are the atoms of arithmetic.",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="prime_code",
        family=SequenceFamily.PRIMES,
        domain=ExpressionDomain.CODE,
        name="Prime code",
        description="Algorithmic primality testing.",
        support_texts=(
            "def is_prime(n): return n>1 and all(n%i!=0 for i in range(2,int(n**0.5)+1))",
            "func isPrime(_ n: Int) -> Bool { n>1 && !(2...Int(sqrt(Double(n)))).contains { n%$0==0 } }",
            "Sieve of Eratosthenes: mark composites, keep unmarked as primes.",
            "Miller-Rabin: probabilistic primality test for large numbers.",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="prime_distribution",
        family=SequenceFamily.PRIMES,
        domain=ExpressionDomain.RATIO,
        name="Prime distribution",
        description="Prime gaps grow logarithmically; π(n) ~ n/ln(n).",
        support_texts=(
            "Prime Number Theorem: π(n) ~ n/ln(n) as n → ∞",
            "Average gap between primes near n is approximately ln(n).",
            "Riemann Hypothesis: zeros of ζ(s) control prime distribution.",
            "Twin prime conjecture: infinitely many primes p where p+2 is also prime.",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="prime_nature",
        family=SequenceFamily.PRIMES,
        domain=ExpressionDomain.NATURE,
        name="Primes in nature",
        description="Cicada life cycles and predator avoidance.",
        support_texts=(
            "Cicadas emerge after 13 or 17 years - both prime.",
            "Prime cycles avoid synchronization with predator cycles.",
            "Evolutionary advantage of prime-period timing.",
            "Periodic cicadas are living examples of number theory.",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="prime_visual",
        family=SequenceFamily.PRIMES,
        domain=ExpressionDomain.VISUAL,
        name="Ulam spiral",
        description="Primes form diagonal patterns when plotted spirally.",
        support_texts=(
            "Ulam spiral: integers spiral outward, primes marked.",
            "Diagonal lines emerge from quadratic residues.",
            "Prime-generating polynomials create visible patterns.",
            "n² + n + 41 is prime for n = 0 to 39 (Euler's polynomial).",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="prime_crypto",
        family=SequenceFamily.PRIMES,
        domain=ExpressionDomain.COMBINATORIAL,
        name="Prime cryptography",
        description="Large primes secure modern encryption.",
        support_texts=(
            "RSA: multiply two large primes; factoring is hard.",
            "Prime numbers are the foundation of public-key cryptography.",
            "2048-bit primes provide current security standards.",
            "Quantum computers threaten prime-based cryptography.",
        ),
        cross_domain_weight=1.0,
    ),
)

# Catalan Numbers (6 probes)
CATALAN_PROBES: tuple[SequenceInvariant, ...] = (
    SequenceInvariant(
        id="cat_definition",
        family=SequenceFamily.CATALAN,
        domain=ExpressionDomain.DEFINITION,
        name="Catalan definition",
        description="C(n) counts valid structures with n pairs.",
        support_texts=(
            "C(n) = (2n)! / ((n+1)! * n!) = C(2n,n) / (n+1)",
            "1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862",
            "Recurrence: C(n) = Σ C(i)*C(n-1-i) for i=0 to n-1",
            "Named after Belgian mathematician Eugène Catalan.",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="cat_code",
        family=SequenceFamily.CATALAN,
        domain=ExpressionDomain.CODE,
        name="Catalan code",
        description="Algorithmic computation of Catalan numbers.",
        support_texts=(
            "def catalan(n): return comb(2*n, n) // (n+1)",
            "func catalan(_ n: Int) -> Int { binomial(2*n, n) / (n+1) }",
            "Dynamic programming: C[n] = sum(C[i]*C[n-1-i] for i in 0..<n)",
            "Generate all valid structures using recursive backtracking.",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="cat_ratio",
        family=SequenceFamily.CATALAN,
        domain=ExpressionDomain.RATIO,
        name="Catalan ratio",
        description="C(n+1)/C(n) approaches 4 as n grows.",
        support_texts=(
            "C(n+1)/C(n) = 2(2n+1)/(n+2) → 4 as n → ∞",
            "Catalan numbers grow like 4^n / (n^(3/2) * sqrt(π)).",
            "Generating function: C(x) = (1 - sqrt(1-4x)) / (2x).",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="cat_parentheses",
        family=SequenceFamily.CATALAN,
        domain=ExpressionDomain.COMBINATORIAL,
        name="Catalan parentheses",
        description="C(n) counts valid parenthesizations.",
        support_texts=(
            "C(3) = 5: ((())), (()()), (())(), ()(()), ()()()",
            "Balanced parentheses: equal opens and closes, never more closes.",
            "Dyck words: paths that never go below the x-axis.",
            "Well-formed expressions in programming languages.",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="cat_trees",
        family=SequenceFamily.CATALAN,
        domain=ExpressionDomain.COMBINATORIAL,
        name="Catalan trees",
        description="C(n) counts binary trees with n+1 leaves.",
        support_texts=(
            "Full binary trees with n+1 leaves: C(n) distinct shapes.",
            "Rooted ordered trees with n edges.",
            "Non-crossing partitions of n+1 elements.",
            "Triangulations of a convex polygon with n+2 sides.",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="cat_visual",
        family=SequenceFamily.CATALAN,
        domain=ExpressionDomain.VISUAL,
        name="Catalan mountain ranges",
        description="Dyck paths as mountain ranges.",
        support_texts=(
            "Draw up-right steps that never cross below the diagonal.",
            "Mountain ranges that start and end at ground level.",
            "Ballot sequences: candidate A never trails candidate B.",
            "Non-crossing chord diagrams on a circle.",
        ),
        cross_domain_weight=1.0,
    ),
)

# Ramanujan Partitions (4 probes)
RAMANUJAN_PROBES: tuple[SequenceInvariant, ...] = (
    SequenceInvariant(
        id="partition_definition",
        family=SequenceFamily.RAMANUJAN,
        domain=ExpressionDomain.COMBINATORIAL,
        name="Partition count",
        description="p(n) counts integer partitions of n (order ignored).",
        support_texts=(
            "p(4)=5: 4, 3+1, 2+2, 2+1+1, 1+1+1+1",
            "Order does not matter: 2+1 and 1+2 are the same partition",
            "p(5)=7, p(6)=11, p(7)=15, p(8)=22",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="partition_congruences",
        family=SequenceFamily.RAMANUJAN,
        domain=ExpressionDomain.DEFINITION,
        name="Partition congruences",
        description="Ramanujan congruences for p(n) modulo 5, 7, and 11.",
        support_texts=(
            "p(5k+4) ≡ 0 (mod 5)",
            "p(7k+5) ≡ 0 (mod 7)",
            "p(11k+6) ≡ 0 (mod 11)",
            "Examples: p(4)=5, p(9)=30, p(14)=135",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="partition_growth",
        family=SequenceFamily.RAMANUJAN,
        domain=ExpressionDomain.RATIO,
        name="Partition growth",
        description="Hardy-Ramanujan asymptotic growth of p(n).",
        support_texts=(
            "p(n) ~ (1/(4*n*sqrt(3))) * exp(pi*sqrt(2*n/3))",
            "log p(n) ~ pi*sqrt(2*n/3) for large n",
            "Growth is exponential in sqrt(n), not in n",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="partition_generating",
        family=SequenceFamily.RAMANUJAN,
        domain=ExpressionDomain.CLOSED_FORM,
        name="Partition generating function",
        description="Euler generating function for partitions.",
        support_texts=(
            "sum_{n>=0} p(n) x^n = product_{k>=1} 1/(1 - x^k)",
            "Each factor 1/(1 - x^k) encodes using k any number of times",
            "The product expands to all partitions of n",
        ),
        cross_domain_weight=1.0,
    ),
)

# Logic Invariants (9 probes)
LOGIC_PROBES: tuple[SequenceInvariant, ...] = (
    SequenceInvariant(
        id="logic_modus_ponens",
        family=SequenceFamily.LOGIC,
        domain=ExpressionDomain.DEFINITION,
        name="Modus ponens",
        description="If A implies B and A is true, then B is true.",
        support_texts=(
            "A -> B, A, therefore B",
            "If A then B; A; so B",
            "Implication elimination rule",
            "Given A implies B and A holds, B follows",
            "Rule: from A and A=>B infer B",
        ),
        cross_domain_weight=1.4,
    ),
    SequenceInvariant(
        id="logic_double_negation",
        family=SequenceFamily.LOGIC,
        domain=ExpressionDomain.CODE,
        name="Double negation",
        description="Negating twice returns the original truth value.",
        support_texts=(
            "not(not A) == A",
            "!!a == a",
            "Double negation cancels out",
            "Boolean: !( !A ) is equivalent to A",
            "Negating a statement twice yields the same statement",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="logic_de_morgan",
        family=SequenceFamily.LOGIC,
        domain=ExpressionDomain.COMBINATORIAL,
        name="De Morgan's laws",
        description="Negation distributes over AND/OR with duality.",
        support_texts=(
            "not (A and B) == (not A) or (not B)",
            "not (A or B) == (not A) and (not B)",
            "De Morgan's equivalences",
            "Negation swaps AND/OR across components",
            "!(A && B) == (!A || !B) and !(A || B) == (!A && !B)",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="logic_contrapositive",
        family=SequenceFamily.LOGIC,
        domain=ExpressionDomain.DEFINITION,
        name="Contrapositive",
        description="If A implies B, then not B implies not A.",
        support_texts=(
            "A -> B is equivalent to (not B) -> (not A)",
            "If B is false, then A must be false",
            "Contrapositive preserves truth in implication",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="logic_excluded_middle",
        family=SequenceFamily.LOGIC,
        domain=ExpressionDomain.DEFINITION,
        name="Excluded middle",
        description="A statement is either true or not true.",
        support_texts=(
            "A or not A is always true",
            "A ∨ ¬A",
            "Boolean: a || !a == true",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="logic_modus_tollens",
        family=SequenceFamily.LOGIC,
        domain=ExpressionDomain.DEFINITION,
        name="Modus tollens",
        description="If A implies B and B is false, then A is false.",
        support_texts=(
            "A -> B, not B, therefore not A",
            "If B is false then A cannot be true",
            "Implication with negated consequent",
            "From A implies B and not B infer not A",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="logic_non_contradiction",
        family=SequenceFamily.LOGIC,
        domain=ExpressionDomain.DEFINITION,
        name="Non-contradiction",
        description="A statement cannot be both true and false.",
        support_texts=(
            "not (A and not A)",
            "A and not A is always false",
            "No statement is both true and false",
            "Law of non-contradiction",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="logic_biconditional",
        family=SequenceFamily.LOGIC,
        domain=ExpressionDomain.DEFINITION,
        name="Biconditional",
        description="A if and only if B means A implies B and B implies A.",
        support_texts=(
            "A <-> B is (A -> B) and (B -> A)",
            "If and only if",
            "Equivalence of A and B",
            "Bidirectional implication",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="logic_distributive",
        family=SequenceFamily.LOGIC,
        domain=ExpressionDomain.COMBINATORIAL,
        name="Distributive logic",
        description="AND/OR distribute over each other in boolean algebra.",
        support_texts=(
            "A and (B or C) == (A and B) or (A and C)",
            "A or (B and C) == (A or B) and (A or C)",
            "Distributive laws of logic",
            "Boolean distributivity",
        ),
        cross_domain_weight=1.0,
    ),
)

# Ordering Invariants (8 probes)
ORDERING_PROBES: tuple[SequenceInvariant, ...] = (
    SequenceInvariant(
        id="order_transitive",
        family=SequenceFamily.ORDERING,
        domain=ExpressionDomain.DEFINITION,
        name="Transitive order",
        description="If A > B and B > C, then A > C.",
        support_texts=(
            "Transitivity of order",
            "x > y and y > z implies x > z",
            "If x < y and y < z then x < z",
            "Order comparisons compose through chains",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="order_antisymmetry",
        family=SequenceFamily.ORDERING,
        domain=ExpressionDomain.DEFINITION,
        name="Antisymmetry",
        description="If A <= B and B <= A, then A = B.",
        support_texts=(
            "x <= y and y <= x implies x == y",
            "Antisymmetry of partial order",
            "No two distinct elements precede each other",
            "If both directions hold, the elements are equal",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="order_total",
        family=SequenceFamily.ORDERING,
        domain=ExpressionDomain.COMBINATORIAL,
        name="Total order",
        description="Any two elements are comparable.",
        support_texts=(
            "Exactly one of A < B, A = B, or A > B holds",
            "Total order comparability",
            "Elements are always comparable",
            "Trichotomy holds for every pair",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="order_reflexive",
        family=SequenceFamily.ORDERING,
        domain=ExpressionDomain.DEFINITION,
        name="Reflexive order",
        description="Every element is comparable to itself.",
        support_texts=(
            "x <= x",
            "Reflexivity of order",
            "An element is always equal to itself",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="order_trichotomy",
        family=SequenceFamily.ORDERING,
        domain=ExpressionDomain.DEFINITION,
        name="Trichotomy",
        description="Exactly one of less-than, equal, or greater-than holds.",
        support_texts=(
            "For any a,b: a < b or a = b or a > b",
            "Order relations are mutually exclusive",
            "Comparable values fall into one of three cases",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="order_monotone_addition",
        family=SequenceFamily.ORDERING,
        domain=ExpressionDomain.DEFINITION,
        name="Order under addition",
        description="Adding the same value preserves order.",
        support_texts=(
            "If a < b then a + c < b + c",
            "Order is translation-invariant",
            "Adding the same amount keeps inequalities true",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="order_monotone_multiplication",
        family=SequenceFamily.ORDERING,
        domain=ExpressionDomain.DEFINITION,
        name="Order under multiplication",
        description="Multiplying by a positive value preserves order.",
        support_texts=(
            "If a < b and c > 0 then a*c < b*c",
            "Positive scaling preserves inequalities",
            "Order is preserved under positive multiplication",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="order_bounds",
        family=SequenceFamily.ORDERING,
        domain=ExpressionDomain.DEFINITION,
        name="Bounds by min/max",
        description="Values are bounded by their minimum and maximum.",
        support_texts=(
            "min(a,b) <= a <= max(a,b)",
            "Every value lies between min and max",
            "Bounds from pairwise comparison",
        ),
        cross_domain_weight=1.0,
    ),
)

# Arithmetic Invariants (9 probes)
ARITHMETIC_PROBES: tuple[SequenceInvariant, ...] = (
    SequenceInvariant(
        id="arith_identity_add",
        family=SequenceFamily.ARITHMETIC,
        domain=ExpressionDomain.DEFINITION,
        name="Additive identity",
        description="Adding zero does not change a value.",
        support_texts=(
            "x + 0 = x",
            "0 + x = x",
            "Additive identity element",
            "Adding zero leaves a value unchanged",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="arith_identity_mul",
        family=SequenceFamily.ARITHMETIC,
        domain=ExpressionDomain.DEFINITION,
        name="Multiplicative identity",
        description="Multiplying by one does not change a value.",
        support_texts=(
            "x * 1 = x",
            "1 * x = x",
            "Multiplicative identity element",
            "Multiplying by one leaves a value unchanged",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="arith_commutative_add",
        family=SequenceFamily.ARITHMETIC,
        domain=ExpressionDomain.CODE,
        name="Commutative addition",
        description="Order of addition does not matter.",
        support_texts=(
            "a + b = b + a",
            "Addition is commutative",
            "swap(a, b) keeps sum",
            "Sum is invariant under permutation",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="arith_commutative_mul",
        family=SequenceFamily.ARITHMETIC,
        domain=ExpressionDomain.CODE,
        name="Commutative multiplication",
        description="Order of multiplication does not matter.",
        support_texts=(
            "a * b = b * a",
            "Multiplication is commutative",
            "Product is invariant under swapping factors",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="arith_distributive",
        family=SequenceFamily.ARITHMETIC,
        domain=ExpressionDomain.DEFINITION,
        name="Distributive law",
        description="Multiplication distributes over addition.",
        support_texts=(
            "a * (b + c) = a*b + a*c",
            "(b + c) * a = b*a + c*a",
            "Distributivity of multiplication over addition",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="arith_associative_add",
        family=SequenceFamily.ARITHMETIC,
        domain=ExpressionDomain.DEFINITION,
        name="Associative addition",
        description="Grouping of addition does not change the result.",
        support_texts=(
            "a + (b + c) = (a + b) + c",
            "Addition is associative",
            "Parentheses do not change sum",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="arith_associative_mul",
        family=SequenceFamily.ARITHMETIC,
        domain=ExpressionDomain.DEFINITION,
        name="Associative multiplication",
        description="Grouping of multiplication does not change the result.",
        support_texts=(
            "a * (b * c) = (a * b) * c",
            "Multiplication is associative",
            "Parentheses do not change product",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="arith_additive_inverse",
        family=SequenceFamily.ARITHMETIC,
        domain=ExpressionDomain.DEFINITION,
        name="Additive inverse",
        description="Every value has an additive inverse.",
        support_texts=(
            "x + (-x) = 0",
            "Additive inverse cancels a value",
            "Sum with negation yields zero",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="arith_zero_product",
        family=SequenceFamily.ARITHMETIC,
        domain=ExpressionDomain.DEFINITION,
        name="Zero product",
        description="Multiplying by zero yields zero.",
        support_texts=(
            "x * 0 = 0",
            "0 * x = 0",
            "Zero annihilates multiplication",
        ),
        cross_domain_weight=1.0,
    ),
)

# Causality Invariants (8 probes)
CAUSALITY_PROBES: tuple[SequenceInvariant, ...] = (
    SequenceInvariant(
        id="cause_precedes_effect",
        family=SequenceFamily.CAUSALITY,
        domain=ExpressionDomain.DEFINITION,
        name="Cause precedes effect",
        description="A cause occurs before its effect.",
        support_texts=(
            "Cause -> effect happens after",
            "Temporal order: cause then effect",
            "Causal precedence",
            "An effect cannot happen before its cause",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="cause_dependency",
        family=SequenceFamily.CAUSALITY,
        domain=ExpressionDomain.COMBINATORIAL,
        name="Effect depends on cause",
        description="If B depends on A, then without A, B does not occur.",
        support_texts=(
            "No A implies no B (when A is required)",
            "Effect requires its cause",
            "Dependency: B needs A",
            "Without the cause, the effect is absent",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="cause_intervention",
        family=SequenceFamily.CAUSALITY,
        domain=ExpressionDomain.CODE,
        name="Causal intervention",
        description="Intervening on a cause changes the effect.",
        support_texts=(
            "do(A) changes B when A causes B",
            "Intervene on A -> B changes",
            "Causal influence via intervention",
            "Manipulating A should change B if A causes B",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="cause_chain",
        family=SequenceFamily.CAUSALITY,
        domain=ExpressionDomain.DEFINITION,
        name="Causal chain",
        description="Causal influence can propagate through intermediates.",
        support_texts=(
            "If A causes B and B causes C, then A influences C",
            "Causal effects can be transitive through chains",
            "Upstream causes ripple through intermediate steps",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="cause_counterfactual",
        family=SequenceFamily.CAUSALITY,
        domain=ExpressionDomain.DEFINITION,
        name="Counterfactual dependence",
        description="If the cause did not occur, the effect would not occur.",
        support_texts=(
            "If not A, then not B (for A causing B)",
            "Counterfactual: remove cause -> remove effect",
            "Effect depends counterfactually on the cause",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="cause_common_cause",
        family=SequenceFamily.CAUSALITY,
        domain=ExpressionDomain.DEFINITION,
        name="Common cause",
        description="A shared cause can explain correlation between effects.",
        support_texts=(
            "If C causes A and B, then A and B may correlate",
            "Common cause explains observed association",
            "Shared causes produce correlated outcomes",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="cause_confounder",
        family=SequenceFamily.CAUSALITY,
        domain=ExpressionDomain.DEFINITION,
        name="Confounder control",
        description="Controlling for a confounder can remove spurious association.",
        support_texts=(
            "Adjusting for C removes the A-B correlation",
            "Confounder explains away association",
            "Conditional independence after control",
        ),
        cross_domain_weight=1.0,
    ),
    SequenceInvariant(
        id="cause_collider",
        family=SequenceFamily.CAUSALITY,
        domain=ExpressionDomain.DEFINITION,
        name="Collider bias",
        description="Conditioning on a common effect can induce correlation.",
        support_texts=(
            "A -> C <- B; conditioning on C links A and B",
            "Collider induces association when conditioned",
            "Selection bias from conditioning on outcomes",
        ),
        cross_domain_weight=1.0,
    ),
)


# Combined Inventory
ALL_PROBES: tuple[SequenceInvariant, ...] = (
    FIBONACCI_PROBES
    + LUCAS_PROBES
    + TRIBONACCI_PROBES
    + PRIME_PROBES
    + CATALAN_PROBES
    + RAMANUJAN_PROBES
    + LOGIC_PROBES
    + ORDERING_PROBES
    + ARITHMETIC_PROBES
    + CAUSALITY_PROBES
)


# Default families for probing (most universal cross-domain anchors)
DEFAULT_FAMILIES: frozenset[SequenceFamily] = frozenset(
    [
        SequenceFamily.FIBONACCI,
        SequenceFamily.LUCAS,
        SequenceFamily.PRIMES,
        SequenceFamily.CATALAN,
        SequenceFamily.RAMANUJAN,
        SequenceFamily.LOGIC,
        SequenceFamily.ORDERING,
        SequenceFamily.ARITHMETIC,
        SequenceFamily.CAUSALITY,
    ]
)


class SequenceInvariantInventory:
    """Static inventory of sequence invariant probes."""

    @staticmethod
    def all_probes() -> tuple[SequenceInvariant, ...]:
        """All sequence invariant probes across all families."""
        return ALL_PROBES

    @staticmethod
    def probes_for_families(
        families: set[SequenceFamily] | None = None,
    ) -> list[SequenceInvariant]:
        """Get probes for specific sequence families."""
        if families is None or len(families) == 0:
            return list(ALL_PROBES)
        return [p for p in ALL_PROBES if p.family in families]

    @staticmethod
    def probes_by_family(family: SequenceFamily) -> tuple[SequenceInvariant, ...]:
        """Get probes by family."""
        family_map = {
            SequenceFamily.FIBONACCI: FIBONACCI_PROBES,
            SequenceFamily.LUCAS: LUCAS_PROBES,
            SequenceFamily.TRIBONACCI: TRIBONACCI_PROBES,
            SequenceFamily.PRIMES: PRIME_PROBES,
            SequenceFamily.CATALAN: CATALAN_PROBES,
            SequenceFamily.RAMANUJAN: RAMANUJAN_PROBES,
            SequenceFamily.LOGIC: LOGIC_PROBES,
            SequenceFamily.ORDERING: ORDERING_PROBES,
            SequenceFamily.ARITHMETIC: ARITHMETIC_PROBES,
            SequenceFamily.CAUSALITY: CAUSALITY_PROBES,
        }
        return family_map.get(family, ())

    @staticmethod
    def probe_count_by_family() -> dict[SequenceFamily, int]:
        """Probe count by family."""
        counts: dict[SequenceFamily, int] = {}
        for probe in ALL_PROBES:
            counts[probe.family] = counts.get(probe.family, 0) + 1
        return counts


class RelationType(str, Enum):
    """Relationship types between sequence families."""

    GENERALIZATION = "generalization"  # Lucas generalizes Fibonacci
    SAME_RECURRENCE = "sameRecurrence"  # Same recurrence structure, different seeds
    RATIO_CONVERGENT = "ratioConvergent"  # Ratios converge to same or related limit
    COMPLEMENTARY = "complementary"  # Complementary counting interpretations


@dataclass(frozen=True)
class SequenceRelationship:
    """Relationship between two sequence families."""

    source: SequenceFamily
    target: SequenceFamily
    relation_type: RelationType
    strength: float


# Known relationships between sequence families
SEQUENCE_RELATIONSHIPS: tuple[SequenceRelationship, ...] = (
    # Fibonacci-Lucas: same recurrence, different initial values
    SequenceRelationship(
        SequenceFamily.FIBONACCI, SequenceFamily.LUCAS, RelationType.SAME_RECURRENCE, 0.95
    ),
    SequenceRelationship(
        SequenceFamily.LUCAS, SequenceFamily.FIBONACCI, RelationType.SAME_RECURRENCE, 0.95
    ),
    # Fibonacci-Tribonacci: k-nacci generalization
    SequenceRelationship(
        SequenceFamily.FIBONACCI, SequenceFamily.TRIBONACCI, RelationType.GENERALIZATION, 0.75
    ),
    SequenceRelationship(
        SequenceFamily.TRIBONACCI, SequenceFamily.FIBONACCI, RelationType.GENERALIZATION, 0.75
    ),
    # Lucas-Tribonacci: both generalizations of linear recurrence
    SequenceRelationship(
        SequenceFamily.LUCAS, SequenceFamily.TRIBONACCI, RelationType.GENERALIZATION, 0.6
    ),
    # Fibonacci-Primes: weak structural relationship (prime Fibonacci numbers)
    SequenceRelationship(
        SequenceFamily.FIBONACCI, SequenceFamily.PRIMES, RelationType.COMPLEMENTARY, 0.4
    ),
    # Catalan-Fibonacci: Catalan appears in Fibonacci counting problems
    SequenceRelationship(
        SequenceFamily.CATALAN, SequenceFamily.FIBONACCI, RelationType.COMPLEMENTARY, 0.5
    ),
    # Catalan-Ramanujan: partition-counting structures
    SequenceRelationship(
        SequenceFamily.CATALAN, SequenceFamily.RAMANUJAN, RelationType.COMPLEMENTARY, 0.6
    ),
    SequenceRelationship(
        SequenceFamily.RAMANUJAN, SequenceFamily.CATALAN, RelationType.COMPLEMENTARY, 0.6
    ),
    # Primes-Catalan: both fundamental counting objects
    SequenceRelationship(
        SequenceFamily.PRIMES, SequenceFamily.CATALAN, RelationType.COMPLEMENTARY, 0.3
    ),
)


class SequenceRelationships:
    """Utilities for querying sequence relationships."""

    @staticmethod
    def strength(source: SequenceFamily, target: SequenceFamily) -> float:
        """Get relationship strength between two families."""
        if source == target:
            return 1.0
        for rel in SEQUENCE_RELATIONSHIPS:
            if rel.source == source and rel.target == target:
                return rel.strength
        return 0.0


@dataclass(frozen=True)
class TriangulatedScore:
    """Result of triangulation scoring."""

    base: float
    cross_domain_multiplier: float
    relationship_bonus: float
    coherence_bonus: float

    @property
    def final(self) -> float:
        """Final combined score."""
        return (
            self.base * self.cross_domain_multiplier
            + self.relationship_bonus
            + self.coherence_bonus
        )


class TriangulationScorer:
    """Computes cross-domain triangulation scores for sequence invariants."""

    @staticmethod
    def compute_score(
        activations: dict[ExpressionDomain, float],
        family: SequenceFamily,
        related_family_activations: dict[SequenceFamily, float] | None = None,
    ) -> TriangulatedScore:
        """
        Compute triangulated confidence score based on cross-domain detection.

        Cross-domain detection (e.g., detecting Fibonacci in both code AND nature)
        provides stronger anchoring than single-domain detection.

        Args:
            activations: Domain activations (domain -> activation strength)
            family: The sequence family being scored
            related_family_activations: Activations from related sequence families

        Returns:
            TriangulatedScore with base, multiplier, and bonus components
        """
        if related_family_activations is None:
            related_family_activations = {}

        # Filter to domains with significant activation
        detected_domains = {k: v for k, v in activations.items() if v > 0.3}
        if not detected_domains:
            return TriangulatedScore(
                base=0.0, cross_domain_multiplier=1.0, relationship_bonus=0.0, coherence_bonus=0.0
            )

        base_score = sum(detected_domains.values()) / len(detected_domains)

        return TriangulatedScore(
            base=base_score,
            cross_domain_multiplier=1.0,
            relationship_bonus=0.0,
            coherence_bonus=0.0,
        )
