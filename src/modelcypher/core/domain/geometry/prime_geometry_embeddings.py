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

"""Prime number generation and embedding utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import log_scalar, sqrt_scalar

from .prime_geometry_types import PrimeSequence
from .prime_geometry_utils import _array_to_list

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


def generate_primes(n: int, backend: "Backend | None" = None) -> PrimeSequence:
    """Generate the first n prime numbers using Sieve of Eratosthenes.

    Args:
        n: Number of primes to generate.
        backend: Compute backend (defaults to system default).

    Returns:
        PrimeSequence with primes and gaps.
    """
    backend = backend or get_default_backend()

    if n < 1:
        raise ValueError("n must be at least 1")

    # Upper bound for nth prime: p_n < n * (ln(n) + ln(ln(n))) for n >= 6
    if n < 6:
        limit = 15
    else:
        ln_n = log_scalar(float(n), backend)
        ln_ln_n = log_scalar(ln_n, backend)
        limit = int(n * (ln_n + ln_ln_n)) + 100

    # Sieve of Eratosthenes (backend-only)
    indices = backend.arange(limit + 1, dtype="int32")
    ones = backend.ones((limit + 1,), dtype="int32")
    zeros = backend.zeros((limit + 1,), dtype="int32")
    is_prime = backend.where(indices < 2, zeros, ones)

    max_factor = int(sqrt_scalar(float(limit), backend))
    for i in range(2, max_factor + 1):
        idx_arr = backend.array([i], dtype="int32")
        is_prime_i = backend.take(is_prime, idx_arr, axis=0)
        backend.eval(is_prime_i)
        if int(backend.to_scalar(is_prime_i)) == 0:
            continue

        start = i * i
        mask_range = indices >= start
        is_multiple = backend.mod(indices, i) == 0
        composite_mask = mask_range & is_multiple
        is_prime = backend.where(composite_mask, zeros, is_prime)

    prime_count_arr = backend.sum(is_prime)
    backend.eval(prime_count_arr)
    prime_count = int(backend.to_scalar(prime_count_arr))
    if prime_count < n:
        # Recursively increase limit if needed
        return generate_primes(n, backend)

    non_prime = ones - is_prime
    keys = indices + non_prime * (limit + 1)
    sorted_idx = backend.argsort(keys)
    prime_indices = backend.take(indices, sorted_idx, axis=0)
    primes = prime_indices[:n]

    gaps = primes[1:] - primes[:-1]

    max_prime_arr = backend.take(primes, backend.array([n - 1], dtype="int32"), axis=0)
    backend.eval(max_prime_arr)
    max_prime = int(backend.to_scalar(max_prime_arr))

    return PrimeSequence(
        primes=primes,
        gaps=gaps,
        count=n,
        max_prime=max_prime,
    )


def time_delay_embedding(
    sequence: "Array",
    embedding_dim: int,
    delay: int = 1,
    backend: "Backend | None" = None,
) -> "Array":
    """Create time-delay (Takens) embedding of a sequence.

    Transforms a 1D sequence into a matrix where each row is a sliding
    window of `embedding_dim` consecutive values.

    Args:
        sequence: 1D array of values [n].
        embedding_dim: Dimension of each embedded vector.
        delay: Time delay between consecutive dimensions (default 1).
        backend: Compute backend.

    Returns:
        Embedded matrix [n_windows, embedding_dim].

    Note:
        Time-delay embedding preserves the topology of the underlying
        dynamical system (Takens' theorem). If prime gaps have structure,
        it should be visible in this embedding.
    """
    backend = backend or get_default_backend()

    n = int(backend.shape(sequence)[0])
    n_windows = n - (embedding_dim - 1) * delay

    if n_windows < 1:
        raise ValueError(
            f"Sequence length {n} too short for embedding_dim={embedding_dim}, delay={delay}"
        )

    # Build embedding matrix with vectorized indexing
    starts = backend.arange(n_windows)
    offsets = backend.arange(0, embedding_dim * delay, delay)
    starts_2d = backend.reshape(starts, (-1, 1))
    offsets_2d = backend.reshape(offsets, (1, -1))
    indices = starts_2d + offsets_2d
    return backend.take(sequence, indices, axis=0)


def residue_embedding(
    primes: "Array",
    moduli: list[int] | None = None,
    backend: "Backend | None" = None,
) -> "Array":
    """Create residue class embedding of primes.

    Embeds each prime as a vector of its residues modulo various moduli.
    Uses primorials (2, 6, 30, 210) by default since primes distribute
    non-uniformly across residue classes of primorials.

    Args:
        primes: Array of prime numbers [n_primes].
        moduli: List of moduli for residue computation.
                Default: [2, 6, 30, 210] (primorial sequence).
        backend: Compute backend.

    Returns:
        Residue embedding matrix [n_primes, len(moduli)].

    Note:
        For p > 2, p ≡ 1 or 5 (mod 6) - primes only hit 2 of 6 residue classes.
        This non-uniform distribution encodes prime structure.
    """
    backend = backend or get_default_backend()

    if moduli is None:
        moduli = [2, 6, 30, 210]  # Primorials: 2, 2*3, 2*3*5, 2*3*5*7

    primes_list = _array_to_list(backend, primes)
    rows = []

    for p in primes_list:
        residues = [float(int(p) % m) for m in moduli]
        rows.append(backend.array(residues))

    return backend.stack(rows, axis=0)


def digit_embedding(
    sequence: "Array",
    base: int = 10,
    max_digits: int = 10,
    backend: "Backend | None" = None,
) -> "Array":
    """Create digit pattern embedding of a sequence.

    Embeds each number as a vector of its digits in the specified base.
    Useful for detecting digit-based patterns (e.g., Benford's law).

    Args:
        sequence: Array of numbers [n].
        base: Number base for digit representation (default 10).
        max_digits: Maximum number of digits to consider.
        backend: Compute backend.

    Returns:
        Digit embedding matrix [n, max_digits].

    Note:
        Numbers are padded with leading zeros to ensure uniform dimension.
        The digit sequence is from most significant to least significant.
    """
    backend = backend or get_default_backend()

    seq_list = _array_to_list(backend, sequence)
    rows = []

    for num in seq_list:
        n = int(num)
        digits = []

        if n == 0:
            digits = [0.0] * max_digits
        else:
            while n > 0 and len(digits) < max_digits:
                digits.append(float(n % base))
                n //= base
            # Pad with zeros
            while len(digits) < max_digits:
                digits.append(0.0)
            # Reverse to get MSB first
            digits = digits[::-1]

        rows.append(backend.array(digits))

    return backend.stack(rows, axis=0)


def binary_digit_embedding(
    sequence: "Array",
    max_bits: int = 20,
    backend: "Backend | None" = None,
) -> "Array":
    """Create binary representation embedding.

    Args:
        sequence: Array of numbers [n].
        max_bits: Maximum number of bits to consider.
        backend: Compute backend.

    Returns:
        Binary embedding matrix [n, max_bits].
    """
    return digit_embedding(sequence, base=2, max_digits=max_bits, backend=backend)
