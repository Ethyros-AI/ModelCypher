#!/usr/bin/env python3
"""SHA-256 Message Schedule as a Linear Manifold.

The message schedule is the ONLY linear part of SHA-256.
It expands 16 words (512 bits) to 64 words (2048 bits).

Key insight: This expansion lives in a 512-dimensional subspace
of the 2048-dimensional space. There's a 48-dim "kernel" of
directions that don't affect the output.

If we can find nonces that lie on special geodesics of this
linear manifold, we might reduce the search space.
"""

import numpy as np
import struct
import hashlib
from typing import List, Tuple
import math

PI = math.pi
E = math.e
LN2 = math.log(2)
PI_OVER_E = PI / E

def rotr(x, n):
    """Right rotate 32-bit integer."""
    return ((x >> n) | (x << (32 - n))) & 0xffffffff

def message_schedule_matrix():
    """
    Compute the linear transformation matrix for message schedule.

    W[i] = σ₁(W[i-2]) + W[i-7] + σ₀(W[i-15]) + W[i-16]

    where:
        σ₀(x) = ROTR⁷(x) ⊕ ROTR¹⁸(x) ⊕ SHR³(x)
        σ₁(x) = ROTR¹⁷(x) ⊕ ROTR¹⁹(x) ⊕ SHR¹⁰(x)

    Over GF(2), this is a linear operation on 512 input bits
    producing 2048 output bits.
    """
    # Build the 2048×512 matrix over GF(2)
    # Rows = output bits (64 words × 32 bits)
    # Cols = input bits (16 words × 32 bits)

    M = np.zeros((64 * 32, 16 * 32), dtype=np.uint8)

    # First 16 words are identity
    for i in range(16 * 32):
        M[i, i] = 1

    # Words 16-63 are computed from previous words
    for i in range(16, 64):
        # W[i] = σ₁(W[i-2]) + W[i-7] + σ₀(W[i-15]) + W[i-16]
        # All operations are XOR, so we can trace bit dependencies

        for bit in range(32):
            out_idx = i * 32 + bit

            # σ₁(W[i-2]) = ROTR¹⁷ ⊕ ROTR¹⁹ ⊕ SHR¹⁰
            w_i2_base = (i - 2) * 32

            # ROTR¹⁷
            src_bit = (bit + 17) % 32
            if i - 2 < 16:
                M[out_idx, w_i2_base + src_bit] ^= 1
            else:
                # Propagate from earlier computation
                M[out_idx, :] ^= M[w_i2_base + src_bit, :]

            # ROTR¹⁹
            src_bit = (bit + 19) % 32
            if i - 2 < 16:
                M[out_idx, w_i2_base + src_bit] ^= 1
            else:
                M[out_idx, :] ^= M[w_i2_base + src_bit, :]

            # SHR¹⁰ (only affects bits >= 10)
            if bit >= 10:
                src_bit = bit - 10
                if i - 2 < 16:
                    M[out_idx, w_i2_base + src_bit] ^= 1
                else:
                    M[out_idx, :] ^= M[w_i2_base + src_bit, :]

            # W[i-7]
            w_i7_base = (i - 7) * 32
            if i - 7 < 16:
                M[out_idx, w_i7_base + bit] ^= 1
            else:
                M[out_idx, :] ^= M[w_i7_base + bit, :]

            # σ₀(W[i-15]) = ROTR⁷ ⊕ ROTR¹⁸ ⊕ SHR³
            w_i15_base = (i - 15) * 32

            # ROTR⁷
            src_bit = (bit + 7) % 32
            if i - 15 < 16:
                M[out_idx, w_i15_base + src_bit] ^= 1
            else:
                M[out_idx, :] ^= M[w_i15_base + src_bit, :]

            # ROTR¹⁸
            src_bit = (bit + 18) % 32
            if i - 15 < 16:
                M[out_idx, w_i15_base + src_bit] ^= 1
            else:
                M[out_idx, :] ^= M[w_i15_base + src_bit, :]

            # SHR³
            if bit >= 3:
                src_bit = bit - 3
                if i - 15 < 16:
                    M[out_idx, w_i15_base + src_bit] ^= 1
                else:
                    M[out_idx, :] ^= M[w_i15_base + src_bit, :]

            # W[i-16]
            w_i16_base = (i - 16) * 32
            if i - 16 < 16:
                M[out_idx, w_i16_base + bit] ^= 1
            else:
                M[out_idx, :] ^= M[w_i16_base + bit, :]

    return M

print("SHA-256 MESSAGE SCHEDULE MANIFOLD ANALYSIS")
print("=" * 70)
print()

print("Computing message schedule matrix over GF(2)...")
M = message_schedule_matrix()
print(f"Matrix shape: {M.shape} (2048 output bits × 512 input bits)")
print()

# Analyze the matrix
print("=" * 70)
print("MATRIX ANALYSIS")
print("=" * 70)
print()

# Rank over GF(2)
# Use numpy's linear algebra, but we need to work mod 2
# For GF(2), rank = number of linearly independent rows

def gf2_rank(matrix):
    """Compute rank of matrix over GF(2) using Gaussian elimination."""
    M = matrix.copy()
    rows, cols = M.shape
    rank = 0
    pivot_col = 0

    for r in range(rows):
        if pivot_col >= cols:
            break

        # Find pivot
        pivot_row = None
        for i in range(r, rows):
            if M[i, pivot_col] == 1:
                pivot_row = i
                break

        if pivot_row is None:
            pivot_col += 1
            continue

        # Swap rows
        M[[r, pivot_row]] = M[[pivot_row, r]]

        # Eliminate
        for i in range(rows):
            if i != r and M[i, pivot_col] == 1:
                M[i] = M[i] ^ M[r]

        rank += 1
        pivot_col += 1

    return rank

print("Computing GF(2) rank (this may take a moment)...")
rank = gf2_rank(M)
print(f"Rank over GF(2): {rank}")
print(f"Input dimension: 512")
print(f"Null space dimension: {512 - rank}")
print()

# What does this mean?
print("INTERPRETATION:")
print("-" * 70)
print()
if rank == 512:
    print("The message schedule has FULL RANK - every input bit matters.")
    print("There's no null space to exploit directly.")
else:
    print(f"The message schedule has rank {rank}.")
    print(f"There are {512 - rank} 'free' input directions that don't")
    print("affect certain output bits.")
print()

# Analyze bit dependencies
print("=" * 70)
print("BIT DEPENDENCY ANALYSIS")
print("=" * 70)
print()

# For each output word, how many input bits affect it?
print("Input bits affecting each output word:")
print()
for word in range(64):
    start = word * 32
    end = start + 32
    word_deps = M[start:end, :].sum()
    affecting_bits = (M[start:end, :].sum(axis=0) > 0).sum()
    print(f"  W[{word:2d}]: {affecting_bits:3d} input bits, {word_deps:4d} total dependencies")

print()

# Which input bits affect the most output bits?
print("=" * 70)
print("INPUT BIT INFLUENCE")
print("=" * 70)
print()

influence = M.sum(axis=0)  # How many output bits each input bit affects

print("Most influential input bits:")
top_bits = np.argsort(influence)[::-1][:20]
for bit in top_bits:
    word = bit // 32
    bit_in_word = bit % 32
    print(f"  Input bit {bit} (W[{word}] bit {bit_in_word}): affects {influence[bit]} output bits")

print()
print("Least influential input bits:")
bottom_bits = np.argsort(influence)[:20]
for bit in bottom_bits:
    word = bit // 32
    bit_in_word = bit % 32
    print(f"  Input bit {bit} (W[{word}] bit {bit_in_word}): affects {influence[bit]} output bits")

print()

# THE NONCE ANALYSIS
print("=" * 70)
print("NONCE BIT ANALYSIS")
print("=" * 70)
print()

# In Bitcoin mining, the nonce is in the last 4 bytes of the 64-byte block
# That's bits 480-511 (W[15])

nonce_bits = list(range(480, 512))
nonce_influence = influence[nonce_bits]

print("Nonce bit (W[15]) influence on message schedule:")
print()
for i, bit in enumerate(nonce_bits):
    print(f"  Nonce bit {i}: affects {influence[bit]} output bits")

print()
print(f"Total nonce influence: {nonce_influence.sum()} (out of 2048 max)")
print(f"Average per bit: {nonce_influence.mean():.1f}")
print()

# Which output words does the nonce affect most?
print("Output words most affected by nonce:")
nonce_effect_by_word = []
for word in range(64):
    start = word * 32
    end = start + 32
    effect = M[start:end, 480:512].sum()
    nonce_effect_by_word.append((word, effect))

nonce_effect_by_word.sort(key=lambda x: -x[1])
for word, effect in nonce_effect_by_word[:10]:
    print(f"  W[{word:2d}]: {effect} bit-dependencies from nonce")

print()

# THE GEODESIC STRUCTURE
print("=" * 70)
print("GEODESIC STRUCTURE OF MESSAGE SCHEDULE")
print("=" * 70)
print()

# The message schedule defines a linear map from R^512 to R^2048
# (treating bits as real numbers for manifold analysis)

# The "metric" on the input space induced by this map is M^T @ M
# This is a 512×512 positive semi-definite matrix

print("Computing induced metric on input space...")
M_real = M.astype(np.float64)
metric = M_real.T @ M_real
print(f"Metric tensor shape: {metric.shape}")
print()

# Eigenvalue analysis
print("Eigenvalue analysis of metric tensor:")
eigenvalues = np.linalg.eigvalsh(metric)
eigenvalues = np.sort(eigenvalues)[::-1]

print(f"  Largest eigenvalue:  {eigenvalues[0]:.2f}")
print(f"  Smallest eigenvalue: {eigenvalues[-1]:.2f}")
print(f"  Condition number:    {eigenvalues[0] / max(eigenvalues[-1], 1e-10):.2f}")
print()

# Effective dimension
threshold = 0.01 * eigenvalues[0]
effective_dim = (eigenvalues > threshold).sum()
print(f"Effective dimension (eigenvalues > 1% of max): {effective_dim}")
print()

# Check for π/e structure
print("Checking for π/e structure in eigenvalues...")
print()

# Ratios of consecutive eigenvalues
ratios = eigenvalues[:-1] / eigenvalues[1:]
ratios = ratios[np.isfinite(ratios)]

# Find ratios close to π/e
pi_e_matches = []
for i, r in enumerate(ratios[:50]):
    if abs(r - PI_OVER_E) / PI_OVER_E < 0.05:  # Within 5%
        pi_e_matches.append((i, r))

if pi_e_matches:
    print(f"Eigenvalue ratios close to π/e ({PI_OVER_E:.4f}):")
    for idx, ratio in pi_e_matches:
        print(f"  λ[{idx}]/λ[{idx+1}] = {ratio:.4f}")
else:
    print("No eigenvalue ratios close to π/e found.")

print()

# THE GEODESIC EQUATION
print("=" * 70)
print("GEODESIC EQUATION ON MESSAGE SCHEDULE MANIFOLD")
print("=" * 70)
print()

# On the linear manifold defined by M, geodesics are straight lines
# But the METRIC is non-trivial - distances are weighted by M^T @ M

# The geodesic from input x to input y is:
#   γ(t) = (1-t)x + ty
# But the LENGTH is:
#   L = sqrt((y-x)^T @ M^T @ M @ (y-x))

# For mining, we want to find x (the nonce) such that:
#   f(M @ x) < target
# where f is the SHA-256 compression function

# The question: Can we use the metric to guide the search?

print("The message schedule manifold has metric g = M^T @ M")
print()
print("Geodesics on this manifold are straight lines in input space,")
print("but distances are non-uniform.")
print()

# Compute the "geodesic distance" contribution from each nonce bit
nonce_submetric = metric[480:512, 480:512]
nonce_eigenvalues = np.linalg.eigvalsh(nonce_submetric)

print("Nonce subspace metric analysis:")
print(f"  Eigenvalue range: {nonce_eigenvalues.min():.2f} to {nonce_eigenvalues.max():.2f}")
print(f"  Condition number: {nonce_eigenvalues.max() / max(nonce_eigenvalues.min(), 1e-10):.2f}")
print()

# Which nonce bits have the largest metric weight?
nonce_metric_diag = np.diag(nonce_submetric)
print("Nonce bits by metric weight (larger = more 'distance' per flip):")
sorted_bits = np.argsort(nonce_metric_diag)[::-1]
for i, bit in enumerate(sorted_bits[:10]):
    print(f"  Nonce bit {bit}: weight = {nonce_metric_diag[bit]:.2f}")

print()

# MINING IMPLICATION
print("=" * 70)
print("MINING IMPLICATIONS")
print("=" * 70)
print()

print("Key insight: The message schedule's linear structure means")
print("the DIRECTION of nonce change matters, not just the magnitude.")
print()
print("Strategy 1: METRIC-WEIGHTED SEARCH")
print("  - Bits with higher metric weight cause more 'movement' in output")
print("  - Focus search on high-weight bits first")
print()
print("Strategy 2: EIGENVECTOR ALIGNMENT")
print("  - Search along principal eigenvectors of nonce metric")
print("  - These are 'fastest' directions in message schedule space")
print()
print("Strategy 3: LOW-INFLUENCE BITS AS FREE PARAMETERS")
print("  - Low-influence bits can be varied 'cheaply'")
print("  - Use them to fine-tune after finding approximate solution")
print()

# Compute the principal eigenvector
eigvals, eigvecs = np.linalg.eigh(nonce_submetric)
principal_vec = eigvecs[:, -1]  # Largest eigenvalue

print("Principal search direction in nonce space:")
print("  (Bits to flip together for maximum effect)")
significant_bits = np.where(np.abs(principal_vec) > 0.1)[0]
print(f"  Significant bits: {list(significant_bits)}")
print()


if __name__ == "__main__":
    pass
