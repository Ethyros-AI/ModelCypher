#!/usr/bin/env python3
"""SHA-256 GF(2) Linear Structure Analysis.

The message schedule of SHA-256 is LINEAR over GF(2):
  W[i] = σ1(W[i-2]) ⊕ W[i-7] ⊕ σ0(W[i-15]) ⊕ W[i-16]

This means the 64 message words live in a linear subspace of dimension 16
(the original message words), not dimension 64.

This script checks if any of this linear structure survives into the output.
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def bytes_to_gf2(data: bytes) -> np.ndarray:
    """Convert bytes to GF(2) vector (numpy array of 0/1)."""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def gf2_rank(matrix: np.ndarray) -> int:
    """Compute rank of a binary matrix over GF(2) using Gaussian elimination."""
    m = matrix.copy()
    rows, cols = m.shape
    rank = 0

    for col in range(cols):
        # Find pivot
        pivot_row = None
        for row in range(rank, rows):
            if m[row, col] == 1:
                pivot_row = row
                break

        if pivot_row is None:
            continue

        # Swap with rank row
        m[[rank, pivot_row]] = m[[pivot_row, rank]]

        # Eliminate
        for row in range(rows):
            if row != rank and m[row, col] == 1:
                m[row] = (m[row] + m[rank]) % 2

        rank += 1

    return rank


def compute_linear_dependencies(outputs: np.ndarray) -> Tuple[int, int]:
    """Check for linear dependencies among output bits over GF(2).

    For truly random outputs, the GF(2) rank should be min(n_samples, 256).
    Lower rank indicates linear structure.

    Returns (rank, expected_rank)
    """
    n_samples, n_bits = outputs.shape
    expected_rank = min(n_samples, n_bits)

    # Transpose so rows are bit positions, columns are samples
    # We're looking for linear relationships among bits across samples
    actual_rank = gf2_rank(outputs.T.astype(np.uint8))

    return actual_rank, expected_rank


def main():
    import hashlib

    np.random.seed(42)

    header = b"GF2 Analysis"
    n_samples = 300  # Keep small for GF(2) elimination speed

    print("SHA-256 GF(2) Linear Structure Analysis")
    print("=" * 60)

    for num_rounds in [8, 16, 32, 64]:
        print(f"\nRounds: {num_rounds}")

        # Generate SHA-256 outputs
        outputs = []
        for _ in range(n_samples):
            nonce = np.random.bytes(32)
            if num_rounds == 64:
                digest = hashlib.sha256(header + nonce).digest()
            else:
                from modelcypher.core.domain.geometry.hash_analyzer import sha256_reduced_rounds
                digest = sha256_reduced_rounds(header + nonce, num_rounds)

            outputs.append(bytes_to_gf2(digest))

        outputs = np.array(outputs)

        # Compute GF(2) rank
        sha_rank, expected = compute_linear_dependencies(outputs)

        # Generate random baseline
        random_outputs = np.random.randint(0, 2, size=(n_samples, 256), dtype=np.uint8)
        rand_rank, _ = compute_linear_dependencies(random_outputs)

        print(f"  SHA-256 GF(2) rank: {sha_rank} / {expected}")
        print(f"  Random GF(2) rank:  {rand_rank} / {expected}")

        if sha_rank < rand_rank:
            print(f"  *** LINEAR STRUCTURE DETECTED: {rand_rank - sha_rank} fewer independent dimensions")
        elif sha_rank > rand_rank:
            print(f"  SHA-256 has MORE independence than random (??)")
        else:
            print(f"  No linear structure detected")

    print("\n" + "=" * 60)
    print("Expected: Random should have full rank (256).")
    print("Lower SHA-256 rank indicates linear dependencies among output bits.")


if __name__ == "__main__":
    main()
