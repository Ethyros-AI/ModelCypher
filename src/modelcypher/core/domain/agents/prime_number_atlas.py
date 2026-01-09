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
Prime Number Atlas - Complete prime number knowledge space for geometric analysis.

This atlas is designed for COMPLETE COVERAGE of the prime number space to enable
geometric analysis of how LLMs encode prime structure. The key insight is that to
see the full geometric structure, you need probes >= model hidden dimension.

Contents:
- 1000 prime numerals (2 to 7919) - matches/exceeds typical hidden dimensions
- Prime words for first 10 primes
- Primality concepts (prime, composite, indivisible)
- Number theory concepts (divisor, factor, coprime, modulo)
- Mathematical operations (factorize, sieve, primality test)
- Famous primes (Mersenne, twin primes, Sophie Germain)
- CRYPTOGRAPHY concepts that DEPEND on primes:
  - RSA, Diffie-Hellman, ElGamal, DSA
  - Elliptic curve cryptography
  - Primality testing (Miller-Rabin, AKS)
  - Discrete logarithm, modular exponentiation

NO COMPOSITES - they are noise for prime structure analysis.

CKA = 1.0 guarantees the relational structure is identical across models.
The question is whether primes form a distinct geometric pattern within that
invariant structure.

Total: ~1050 probes covering the prime knowledge space.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache


class PrimeCategory(str, Enum):
    """Category of prime-related concept."""

    PRIME_NUMERAL = "prime_numeral"  # Prime numbers as digits: 2, 3, 5, 7, ...
    PRIME_WORD = "prime_word"  # Prime numbers as words: two, three, five, ...
    PRIMALITY = "primality"  # Concepts: prime, composite, indivisible
    NUMBER_THEORY = "number_theory"  # Divisor, factor, multiple, coprime
    OPERATIONS = "operations"  # Factorize, divide, modulo
    FAMOUS_PRIMES = "famous_primes"  # Mersenne, twin, Sophie Germain
    SPECIAL_NUMBERS = "special_numbers"  # 0, 1, infinity
    CRYPTOGRAPHY = "cryptography"  # RSA, DH, ECC - algorithms that depend on primes


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


def _generate_primes(n: int) -> list[int]:
    """Generate the first n prime numbers using a sieve."""
    if n < 1:
        return []

    # Upper bound for nth prime: p_n < n * (ln(n) + ln(ln(n))) for n >= 6
    import math
    if n < 6:
        upper_bound = 15
    else:
        ln_n = math.log(n)
        upper_bound = int(n * (ln_n + math.log(ln_n)) * 1.3) + 100

    # Sieve of Eratosthenes
    is_prime = [True] * (upper_bound + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(upper_bound**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, upper_bound + 1, i):
                is_prime[j] = False

    primes = [i for i, prime in enumerate(is_prime) if prime]
    return primes[:n]


# Generate first 1000 primes at module load time
_FIRST_1000_PRIMES = _generate_primes(1000)


def _ordinal_suffix(n: int) -> str:
    """Return ordinal suffix for a number (1st, 2nd, 3rd, 4th, etc.)."""
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _generate_prime_numeral_concepts() -> tuple[PrimeConcept, ...]:
    """Generate PrimeConcept entries for first 1000 primes."""
    concepts = []
    for idx, prime in enumerate(_FIRST_1000_PRIMES, start=1):
        concepts.append(
            PrimeConcept(
                id=f"p{prime}",
                name=str(prime),
                category=PrimeCategory.PRIME_NUMERAL,
                description=f"{_ordinal_suffix(idx)} prime",
                support_texts=(str(prime), f"the number {prime}"),
                is_prime=True,
            )
        )
    return tuple(concepts)


class PrimeConceptInventory:
    """Inventory of prime-related concepts."""

    # First 1000 prime numbers as numerals (2 to 7919)
    PRIME_NUMERALS = _generate_prime_numeral_concepts()

    # Prime numbers as words (first 10)
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
        PrimeConcept("euler_phi", "Euler's totient", PrimeCategory.NUMBER_THEORY, "Count of coprimes less than n", ("Euler totient", "phi function", "totient")),
        PrimeConcept("carmichael", "Carmichael function", PrimeCategory.NUMBER_THEORY, "Smallest m where a^m = 1 mod n", ("Carmichael", "lambda function", "reduced totient")),
    )

    # Mathematical operations
    OPERATIONS = (
        PrimeConcept("factorize", "factorize", PrimeCategory.OPERATIONS, "Find prime factors", ("factorize", "factor", "decompose")),
        PrimeConcept("divide", "divide", PrimeCategory.OPERATIONS, "Split into equal parts", ("divide", "division", "quotient")),
        PrimeConcept("remainder", "remainder", PrimeCategory.OPERATIONS, "What's left after division", ("remainder", "leftover", "residue")),
        PrimeConcept("sieve", "sieve", PrimeCategory.OPERATIONS, "Filter primes from composites", ("sieve", "filter", "Eratosthenes")),
        PrimeConcept("test_primality", "primality test", PrimeCategory.OPERATIONS, "Check if prime", ("primality test", "is prime", "check prime")),
        PrimeConcept("mod_exp", "modular exponentiation", PrimeCategory.OPERATIONS, "Compute a^b mod n efficiently", ("modular exponentiation", "fast exponentiation", "square and multiply")),
        PrimeConcept("mod_inverse", "modular inverse", PrimeCategory.OPERATIONS, "Find x where ax = 1 mod n", ("modular inverse", "multiplicative inverse", "inverse mod")),
    )

    # Famous primes and prime types
    FAMOUS_PRIMES = (
        PrimeConcept("mersenne", "Mersenne prime", PrimeCategory.FAMOUS_PRIMES, "Prime of form 2^n - 1", ("Mersenne", "Mersenne prime", "2^n minus 1")),
        PrimeConcept("twin_prime", "twin prime", PrimeCategory.FAMOUS_PRIMES, "Primes differing by 2", ("twin prime", "twin primes", "prime pair")),
        PrimeConcept("sophie_germain", "Sophie Germain prime", PrimeCategory.FAMOUS_PRIMES, "p where 2p+1 is also prime", ("Sophie Germain", "safe prime", "Germain")),
        PrimeConcept("fermat", "Fermat prime", PrimeCategory.FAMOUS_PRIMES, "Prime of form 2^(2^n) + 1", ("Fermat prime", "Fermat number", "Fermat")),
        PrimeConcept("perfect_number", "perfect number", PrimeCategory.FAMOUS_PRIMES, "Sum of divisors equals itself", ("perfect number", "perfect", "6 28 496")),
        PrimeConcept("prime_gap", "prime gap", PrimeCategory.FAMOUS_PRIMES, "Distance between consecutive primes", ("prime gap", "gap", "distance between primes")),
        PrimeConcept("safe_prime", "safe prime", PrimeCategory.FAMOUS_PRIMES, "p where (p-1)/2 is also prime", ("safe prime", "safe", "strong prime")),
        PrimeConcept("wieferich", "Wieferich prime", PrimeCategory.FAMOUS_PRIMES, "p where p^2 divides 2^(p-1) - 1", ("Wieferich", "Wieferich prime", "Fermat quotient")),
        PrimeConcept("wilson", "Wilson prime", PrimeCategory.FAMOUS_PRIMES, "p where p^2 divides (p-1)! + 1", ("Wilson", "Wilson prime", "factorial")),
        PrimeConcept("palindromic_prime", "palindromic prime", PrimeCategory.FAMOUS_PRIMES, "Prime that is a palindrome", ("palindromic prime", "palindrome", "11 101 131")),
    )

    # Special numbers (edge cases related to primality)
    SPECIAL_NUMBERS = (
        PrimeConcept("zero", "0", PrimeCategory.SPECIAL_NUMBERS, "Zero - neither prime nor composite", ("0", "zero", "nothing"), is_prime=None),
        PrimeConcept("one", "1", PrimeCategory.SPECIAL_NUMBERS, "One - the unit, not prime", ("1", "one", "unity"), is_prime=None),
        PrimeConcept("infinity", "infinity", PrimeCategory.SPECIAL_NUMBERS, "Unbounded - infinitely many primes", ("infinity", "infinite", "unbounded")),
    )

    # Cryptography concepts that DEPEND on primes
    CRYPTOGRAPHY = (
        # Public key cryptography (relies on prime factorization being hard)
        PrimeConcept("rsa", "RSA", PrimeCategory.CRYPTOGRAPHY, "Public key crypto using prime factorization", ("RSA", "RSA encryption", "Rivest Shamir Adleman")),
        PrimeConcept("rsa_modulus", "RSA modulus", PrimeCategory.CRYPTOGRAPHY, "Product of two large primes n=pq", ("RSA modulus", "modulus", "n equals p times q")),
        PrimeConcept("rsa_exponent", "RSA exponent", PrimeCategory.CRYPTOGRAPHY, "Public/private exponent e, d", ("RSA exponent", "public exponent", "private exponent")),
        PrimeConcept("rsa_key_generation", "RSA key generation", PrimeCategory.CRYPTOGRAPHY, "Generate p, q, compute n, phi, e, d", ("RSA keygen", "key generation", "generate RSA keys")),

        # Diffie-Hellman and discrete log
        PrimeConcept("diffie_hellman", "Diffie-Hellman", PrimeCategory.CRYPTOGRAPHY, "Key exchange using discrete log", ("Diffie-Hellman", "DH", "key exchange")),
        PrimeConcept("dh_prime", "DH prime", PrimeCategory.CRYPTOGRAPHY, "Large prime p for DH modulus", ("DH prime", "Diffie-Hellman prime", "group prime")),
        PrimeConcept("dh_generator", "DH generator", PrimeCategory.CRYPTOGRAPHY, "Generator g of multiplicative group", ("DH generator", "primitive root", "generator")),
        PrimeConcept("discrete_log", "discrete logarithm", PrimeCategory.CRYPTOGRAPHY, "Find x where g^x = h mod p", ("discrete log", "DLOG", "discrete logarithm problem")),
        PrimeConcept("elgamal", "ElGamal", PrimeCategory.CRYPTOGRAPHY, "Public key crypto using discrete log", ("ElGamal", "ElGamal encryption", "ElGamal signature")),

        # Digital signatures
        PrimeConcept("dsa", "DSA", PrimeCategory.CRYPTOGRAPHY, "Digital Signature Algorithm using primes", ("DSA", "Digital Signature Algorithm", "DSA signature")),
        PrimeConcept("dsa_prime", "DSA prime", PrimeCategory.CRYPTOGRAPHY, "Prime p for DSA group", ("DSA prime", "DSA p", "signature prime")),
        PrimeConcept("schnorr", "Schnorr signature", PrimeCategory.CRYPTOGRAPHY, "Signature scheme using prime groups", ("Schnorr", "Schnorr signature", "Schnorr protocol")),

        # Elliptic curve cryptography (uses primes as field characteristic)
        PrimeConcept("ecc", "elliptic curve cryptography", PrimeCategory.CRYPTOGRAPHY, "Crypto over elliptic curves mod p", ("ECC", "elliptic curve", "curve cryptography")),
        PrimeConcept("ecc_prime", "ECC prime", PrimeCategory.CRYPTOGRAPHY, "Prime p defining the finite field", ("ECC prime", "curve prime", "field characteristic")),
        PrimeConcept("ecdsa", "ECDSA", PrimeCategory.CRYPTOGRAPHY, "Elliptic Curve Digital Signature", ("ECDSA", "EC DSA", "elliptic curve signature")),
        PrimeConcept("ecdh", "ECDH", PrimeCategory.CRYPTOGRAPHY, "Elliptic Curve Diffie-Hellman", ("ECDH", "EC DH", "elliptic curve key exchange")),
        PrimeConcept("curve25519", "Curve25519", PrimeCategory.CRYPTOGRAPHY, "Curve over prime 2^255 - 19", ("Curve25519", "X25519", "curve25519")),
        PrimeConcept("secp256k1", "secp256k1", PrimeCategory.CRYPTOGRAPHY, "Bitcoin curve over prime field", ("secp256k1", "Bitcoin curve", "Koblitz curve")),
        PrimeConcept("p256", "P-256", PrimeCategory.CRYPTOGRAPHY, "NIST curve over 256-bit prime", ("P-256", "secp256r1", "prime256v1")),

        # Primality testing algorithms (used in key generation)
        PrimeConcept("miller_rabin", "Miller-Rabin", PrimeCategory.CRYPTOGRAPHY, "Probabilistic primality test", ("Miller-Rabin", "Miller Rabin test", "probabilistic primality")),
        PrimeConcept("fermat_test", "Fermat primality test", PrimeCategory.CRYPTOGRAPHY, "Test using Fermat's little theorem", ("Fermat test", "Fermat primality", "a^(p-1) = 1")),
        PrimeConcept("aks", "AKS primality test", PrimeCategory.CRYPTOGRAPHY, "Deterministic polynomial-time test", ("AKS", "Agrawal Kayal Saxena", "deterministic primality")),
        PrimeConcept("lucas_lehmer", "Lucas-Lehmer test", PrimeCategory.CRYPTOGRAPHY, "Test for Mersenne primes", ("Lucas-Lehmer", "Mersenne test", "Lucas Lehmer")),
        PrimeConcept("baillie_psw", "Baillie-PSW test", PrimeCategory.CRYPTOGRAPHY, "Combined Miller-Rabin and Lucas test", ("Baillie-PSW", "Baillie PSW", "combined primality")),

        # Prime generation
        PrimeConcept("prime_generation", "prime generation", PrimeCategory.CRYPTOGRAPHY, "Generate large random primes", ("prime generation", "generate prime", "random prime")),
        PrimeConcept("probable_prime", "probable prime", PrimeCategory.CRYPTOGRAPHY, "Number passing probabilistic tests", ("probable prime", "pseudoprime", "industrial-grade prime")),
        PrimeConcept("provable_prime", "provable prime", PrimeCategory.CRYPTOGRAPHY, "Prime with proof of primality", ("provable prime", "certified prime", "proven prime")),

        # Hash-based and other applications
        PrimeConcept("quadratic_residue", "quadratic residue", PrimeCategory.CRYPTOGRAPHY, "x^2 = a mod p has solution", ("quadratic residue", "QR", "square mod p")),
        PrimeConcept("legendre_symbol", "Legendre symbol", PrimeCategory.CRYPTOGRAPHY, "Is a a quadratic residue mod p?", ("Legendre", "Legendre symbol", "(a/p)")),
        PrimeConcept("jacobi_symbol", "Jacobi symbol", PrimeCategory.CRYPTOGRAPHY, "Generalized Legendre symbol", ("Jacobi", "Jacobi symbol", "(a/n)")),
        PrimeConcept("blum_integer", "Blum integer", PrimeCategory.CRYPTOGRAPHY, "Product of two primes both = 3 mod 4", ("Blum integer", "Blum number", "3 mod 4 primes")),
        PrimeConcept("rabin_cryptosystem", "Rabin cryptosystem", PrimeCategory.CRYPTOGRAPHY, "Crypto based on square roots mod n", ("Rabin", "Rabin cryptosystem", "square root modular")),
        PrimeConcept("paillier", "Paillier cryptosystem", PrimeCategory.CRYPTOGRAPHY, "Homomorphic encryption using primes", ("Paillier", "Paillier encryption", "additive homomorphic")),

        # Key sizes and security
        PrimeConcept("rsa_2048", "RSA-2048", PrimeCategory.CRYPTOGRAPHY, "RSA with 2048-bit modulus", ("RSA-2048", "2048-bit RSA", "2048 bit")),
        PrimeConcept("rsa_4096", "RSA-4096", PrimeCategory.CRYPTOGRAPHY, "RSA with 4096-bit modulus", ("RSA-4096", "4096-bit RSA", "4096 bit")),
        PrimeConcept("dh_2048", "DH-2048", PrimeCategory.CRYPTOGRAPHY, "Diffie-Hellman with 2048-bit prime", ("DH-2048", "2048-bit DH", "2048 bit prime")),

        # Factorization attacks (why primes matter)
        PrimeConcept("prime_factorization", "prime factorization", PrimeCategory.CRYPTOGRAPHY, "Decompose n into prime factors", ("prime factorization", "integer factorization", "factor")),
        PrimeConcept("gnfs", "GNFS", PrimeCategory.CRYPTOGRAPHY, "General Number Field Sieve factoring", ("GNFS", "number field sieve", "NFS")),
        PrimeConcept("pollard_rho", "Pollard's rho", PrimeCategory.CRYPTOGRAPHY, "Factoring algorithm for small factors", ("Pollard rho", "Pollard's rho", "rho algorithm")),
        PrimeConcept("quadratic_sieve", "quadratic sieve", PrimeCategory.CRYPTOGRAPHY, "Factoring via smooth numbers", ("quadratic sieve", "QS", "smooth numbers")),
    )

    @classmethod
    @lru_cache(maxsize=1)
    def all_concepts(cls) -> list[PrimeConcept]:
        """Get all prime-related concepts."""
        concepts: list[PrimeConcept] = []
        concepts.extend(cls.PRIME_NUMERALS)
        concepts.extend(cls.PRIME_WORDS)
        concepts.extend(cls.PRIMALITY)
        concepts.extend(cls.NUMBER_THEORY)
        concepts.extend(cls.OPERATIONS)
        concepts.extend(cls.FAMOUS_PRIMES)
        concepts.extend(cls.SPECIAL_NUMBERS)
        concepts.extend(cls.CRYPTOGRAPHY)
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
    def numerals_only(cls) -> list[PrimeConcept]:
        """Get only prime numeral concepts."""
        return [
            c for c in cls.all_concepts()
            if c.category == PrimeCategory.PRIME_NUMERAL
        ]

    @classmethod
    def words_only(cls) -> list[PrimeConcept]:
        """Get only word concepts."""
        return [
            c for c in cls.all_concepts()
            if c.category == PrimeCategory.PRIME_WORD
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
                PrimeCategory.CRYPTOGRAPHY,
            )
        ]

    @classmethod
    def cryptography_only(cls) -> list[PrimeConcept]:
        """Get only cryptography concepts."""
        return [
            c for c in cls.all_concepts()
            if c.category == PrimeCategory.CRYPTOGRAPHY
        ]

    @classmethod
    def count(cls) -> int:
        """Total number of concepts."""
        return len(cls.all_concepts())

    @classmethod
    def prime_count(cls) -> int:
        """Number of prime numeral concepts."""
        return len(cls.PRIME_NUMERALS)

    @classmethod
    def get_prime_list(cls) -> list[int]:
        """Get the list of prime numbers covered by this atlas."""
        return list(_FIRST_1000_PRIMES)
