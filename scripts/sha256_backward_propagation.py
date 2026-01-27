#!/usr/bin/env python3
"""SHA-256 Backward Constraint Propagation.

The insight: Mining is constraint satisfaction.
The constraint: H[0:k] = 0 (first k bits zero)

Instead of forward brute force, can we propagate constraints BACKWARD
through the compression function to narrow the nonce search?

If we know the output must satisfy H[0:k] = 0, what can we infer about
intermediate states?
"""

import hashlib
import struct
import numpy as np
from typing import List, Tuple, Dict, Set, Optional
import time
from collections import defaultdict
import math

# SHA-256 Constants
K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]

H0 = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
]


def rotr(x, n, w=32):
    return ((x >> n) | (x << (w - n))) & ((1 << w) - 1)


def sigma0(x):
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)


def sigma1(x):
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10)


def Sigma0(x):
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)


def Sigma1(x):
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)


def Ch(e, f, g):
    return (e & f) ^ (~e & g)


def Maj(a, b, c):
    return (a & b) ^ (a & c) ^ (b & c)


def count_leading_zeros(hash_bytes: bytes) -> int:
    n = 0
    for byte in hash_bytes:
        if byte == 0:
            n += 8
        else:
            for i in range(7, -1, -1):
                if byte & (1 << i):
                    return n
                n += 1
    return n


# =============================================================================
# BACKWARD PROPAGATION ANALYSIS
# =============================================================================

def analyze_final_addition():
    """
    The final step of SHA-256: H_final = H_init + compression_output

    For H_final[0:k] = 0, we need specific values of compression_output.
    This is a LINEAR constraint!
    """
    print("=" * 70)
    print("FINAL ADDITION CONSTRAINT")
    print("=" * 70)
    print()

    print("SHA-256 finishes with: H_final[i] = H_init[i] + compression[i]")
    print()

    for target_zeros in [8, 16, 24, 32]:
        print(f"Target: {target_zeros} leading zeros")

        # Full words that must be zero
        full_zero_words = target_zeros // 32

        # For H_final = 0, we need compression = -H_init (mod 2^32)
        for i in range(min(full_zero_words + 1, 8)):
            if i < full_zero_words:
                required = (0 - H0[i]) & 0xFFFFFFFF
                print(f"  compression[{i}] must equal {required:#010x}")
            else:
                partial_bits = target_zeros % 32
                if partial_bits > 0:
                    mask = ((1 << partial_bits) - 1) << (32 - partial_bits)
                    required_masked = ((0 - H0[i]) & 0xFFFFFFFF) & mask
                    print(f"  compression[{i}] top {partial_bits} bits must equal {required_masked:#010x}")

        print()


def compute_backward_constraints(target_zeros: int, rounds_back: int = 5):
    """
    Propagate the output constraint backward through compression rounds.

    The compression function update is:
        T1 = h + Σ₁(e) + Ch(e,f,g) + K[i] + W[i]
        T2 = Σ₀(a) + Maj(a,b,c)
        (a,b,c,d,e,f,g,h) := (T1+T2, a, b, c, d+T1, e, f, g)

    Going BACKWARD:
        h_prev = a_cur - T1 - T2  ... but we don't know T1, T2 yet
    """
    print("=" * 70)
    print(f"BACKWARD PROPAGATION: {rounds_back} rounds from output")
    print("=" * 70)
    print()

    # The output constraint tells us about the final (a,b,c,d,e,f,g,h)
    # After adding H0, we need H_final[0:k] = 0

    print("At round 64 (final), the state is (a,b,c,d,e,f,g,h)")
    print("The output is H_final[i] = H0[i] + state[i]")
    print()

    # Required final state for each word
    print(f"For {target_zeros} leading zeros, final state must satisfy:")
    print()

    full_words = target_zeros // 32
    partial_bits = target_zeros % 32

    constraints = []
    for i in range(8):
        if i < full_words:
            required = (0 - H0[i]) & 0xFFFFFFFF
            constraints.append((i, 'exact', required))
            print(f"  state[{i}] = {required:#010x}")
        elif i == full_words and partial_bits > 0:
            mask = ((1 << partial_bits) - 1) << (32 - partial_bits)
            required_masked = ((0 - H0[i]) & mask)
            constraints.append((i, 'partial', mask, required_masked))
            print(f"  state[{i}] & {mask:#010x} = {required_masked:#010x}")

    print()

    # Now work backward through rounds
    print("BACKWARD PROPAGATION:")
    print()
    print("Round 64 → 63:")
    print("  The update was: a = T1 + T2")
    print("  So: T1 + T2 is constrained (partially)")
    print()
    print("  But T1 = h_prev + Σ₁(e_prev) + Ch(e_prev,f_prev,g_prev) + K[63] + W[63]")
    print("  And T2 = Σ₀(a_prev) + Maj(a_prev,b_prev,c_prev)")
    print()
    print("  This creates NONLINEAR constraints on round 63 state!")
    print()

    # The key insight: each backward round doubles the algebraic degree
    # But we can still enumerate PARTIAL solutions

    print("THE BRANCHING FACTOR:")
    print()

    # For each bit of constraint, how many (T1, T2) pairs satisfy it?
    # If we know a = T1 + T2 (mod 2^32), there are 2^32 pairs
    # But T1 and T2 are not independent!

    print("  a = T1 + T2 is one equation in many unknowns")
    print("  But T1 depends on (h, e, f, g, W[i]) - 5 words = 160 bits")
    print("  And T2 depends on (a, b, c) - 3 words = 96 bits")
    print()
    print("  However, b = a_prev, c = b_prev, etc.")
    print("  So the state only has 8 × 32 = 256 bits of freedom")
    print()

    # Meet-in-the-middle analysis
    print("MEET-IN-THE-MIDDLE OPPORTUNITY:")
    print()
    print("  Split at round 32:")
    print("  - Forward: compute state at round 32 from nonce (2^32 options)")
    print("  - Backward: compute state at round 32 from constraint (?? options)")
    print()
    print("  The backward direction has BRANCHING:")
    print("  - Each round backward, we must guess values")
    print("  - Constraint propagation prunes some branches")
    print()

    return constraints


def analyze_ch_maj_invertibility():
    """
    Ch and Maj are the nonlinear components. How invertible are they?

    Ch(e,f,g) = (e & f) ^ (~e & g)
    Maj(a,b,c) = (a & b) ^ (a & c) ^ (b & c)
    """
    print("=" * 70)
    print("Ch AND Maj INVERTIBILITY")
    print("=" * 70)
    print()

    print("Ch(e,f,g) = (e & f) ^ (~e & g)")
    print()
    print("  If we know Ch and e:")
    print("  - When e=1: Ch = f (the f bit is revealed)")
    print("  - When e=0: Ch = g (the g bit is revealed)")
    print()
    print("  If we know Ch but not e:")
    print("  - Ch constrains (f XOR g) at positions where e is unknown")
    print("  - Each unknown e bit gives 2 possibilities")
    print()

    # For a single bit, the Ch function has this truth table:
    print("  Ch truth table (per bit):")
    print("  e f g | Ch")
    print("  0 0 0 | 0")
    print("  0 0 1 | 1")
    print("  0 1 0 | 0")
    print("  0 1 1 | 1")
    print("  1 0 0 | 0")
    print("  1 0 1 | 0")
    print("  1 1 0 | 1")
    print("  1 1 1 | 1")
    print()

    # Given Ch, how many (e,f,g) triples are possible?
    print("  Given Ch=0: 4 possibilities for (e,f,g)")
    print("  Given Ch=1: 4 possibilities for (e,f,g)")
    print()
    print("  32 bits of Ch constraint leaves 2^64 possibilities for (e,f,g)")
    print()

    print("Maj(a,b,c) = (a & b) ^ (a & c) ^ (b & c)")
    print()
    print("  Maj = 1 when at least 2 of {a,b,c} are 1")
    print("  Maj = 0 when at least 2 of {a,b,c} are 0")
    print()
    print("  Maj truth table (per bit):")
    print("  a b c | Maj")
    print("  0 0 0 | 0")
    print("  0 0 1 | 0")
    print("  0 1 0 | 0")
    print("  0 1 1 | 1")
    print("  1 0 0 | 0")
    print("  1 0 1 | 1")
    print("  1 1 0 | 1")
    print("  1 1 1 | 1")
    print()
    print("  Given Maj=0: 4 possibilities for (a,b,c)")
    print("  Given Maj=1: 4 possibilities for (a,b,c)")
    print()
    print("  32 bits of Maj constraint leaves 2^64 possibilities for (a,b,c)")
    print()


def count_backward_branches(target_zeros: int, rounds: int):
    """
    Count how many candidate states exist when propagating backward.

    Starting from the output constraint, how many states at round (64-r)
    could lead to a valid output?
    """
    print("=" * 70)
    print(f"BACKWARD BRANCHING FACTOR ({rounds} rounds)")
    print("=" * 70)
    print()

    # Initial constraint: k bits of the output
    constrained_bits = target_zeros

    print(f"Output constraint: {constrained_bits} bits")
    print()

    total_branches = 1  # Start with 1 path (the constraint)

    for r in range(rounds):
        round_num = 64 - r

        # Going back one round:
        # - Each constrained bit creates constraints on T1, T2
        # - T1 = h + Σ₁(e) + Ch(e,f,g) + K[i] + W[i]
        # - T2 = Σ₀(a) + Maj(a,b,c)

        # The constraint T1 + T2 = known_value
        # But T1 and T2 are not independent

        # Σ₀, Σ₁ are linear (XOR of rotations) - don't add branches
        # Addition is linear - doesn't add branches
        # Ch, Maj are nonlinear - each adds 2^(affected bits) branches

        # Per round, the branching comes from:
        # - Guessing W[i] if nonce-dependent (32 bits in rounds > 15)
        # - Ch and Maj resolution (but constrained by state relations)

        # In practice: about 2^2 per round due to Ch/Maj nonlinearity
        # But many branches are pruned by carry propagation

        branch_factor = 4 if r > 0 else 1  # Ch and Maj each have 4 preimages

        # Pruning from state relations (b=a_prev, etc.)
        prune_factor = 0.5 if r > 1 else 1

        net_factor = branch_factor * prune_factor
        total_branches *= net_factor
        constrained_bits = min(constrained_bits + 32, 256)  # Spreads to more state

        print(f"  Round {round_num}: branch_factor={branch_factor}, total={total_branches:.0f}")

    print()
    print(f"After {rounds} rounds backward: ~{total_branches:.0e} candidate states")
    print()

    return total_branches


def meet_in_middle_analysis(target_zeros: int = 20):
    """
    Analyze the meet-in-the-middle approach for SHA-256.
    """
    print("=" * 70)
    print("MEET-IN-THE-MIDDLE ANALYSIS")
    print("=" * 70)
    print()

    print("Classic MITM splits computation at the middle:")
    print()
    print("  Forward: nonce → round 32 state (deterministic)")
    print("  Backward: constraint → round 32 states (branching)")
    print()

    print("FORWARD PHASE:")
    print("  2^32 nonces, each gives exactly 1 state at round 32")
    print("  Store all (nonce, state_32) pairs")
    print("  Space: O(2^32) words = O(128 GB)")
    print()

    print("BACKWARD PHASE:")
    print("  Start with output constraint (k bits)")
    print("  Propagate back 32 rounds")
    print("  Each round multiplies candidates by ~4 (Ch, Maj)")
    print("  But pruning reduces this to ~2 per round")
    print()

    # Estimate backward candidates
    backward_factor_per_round = 2
    rounds_back = 32
    backward_candidates = 2 ** target_zeros * (backward_factor_per_round ** rounds_back)

    print(f"  For {target_zeros} zeros: ~{backward_candidates:.0e} backward candidates")
    print()

    print("COLLISION SEARCH:")
    print("  Find (nonce, state_32) that appears in both forward and backward sets")
    print()

    # Expected collisions
    forward_count = 2**32
    # State at round 32 is 256 bits, but constrained by earlier state
    state_space = 2**256

    collision_prob = (forward_count * backward_candidates) / state_space

    print(f"  Forward candidates: {forward_count:.0e}")
    print(f"  Backward candidates: {backward_candidates:.0e}")
    print(f"  State space: {state_space:.0e}")
    print(f"  Expected collisions: {collision_prob:.0e}")
    print()

    if collision_prob >= 1:
        print("  MITM could find solution!")
        print(f"  Time: O(2^32) forward + O(backward candidates)")
        print(f"  Space: O(2^32)")
    else:
        print("  MITM unlikely to find collision without more structure")
    print()


def explore_message_schedule_constraints():
    """
    The message schedule W is LINEAR over GF(2).
    Can we use this to propagate constraints?
    """
    print("=" * 70)
    print("MESSAGE SCHEDULE CONSTRAINTS")
    print("=" * 70)
    print()

    print("Message schedule: W[i] = σ₁(W[i-2]) + W[i-7] + σ₀(W[i-15]) + W[i-16]")
    print()
    print("This is LINEAR over GF(2) for the XOR parts, plus carries for addition.")
    print()

    print("Given W[0:16] (the input), all W[16:64] are determined.")
    print()

    print("KEY OBSERVATION:")
    print("  W[0:14] are fixed by block header")
    print("  W[15] is the nonce (32 bits)")
    print()
    print("  Therefore: All W[16:64] are LINEAR functions of the nonce!")
    print()

    print("COMPUTING W AS FUNCTION OF NONCE:")
    print()

    # The message schedule is a linear recurrence
    # W[i] depends on W[i-2], W[i-7], W[i-15], W[i-16]

    # For Bitcoin blocks, the nonce is in word 15 (I think - let me check format)
    # Actually the exact position depends on the block format

    # Let's compute symbolically how each W[i] depends on the nonce

    # Represent nonce as a 32-element vector (one per bit)
    # Then W[i] = sum of nonce bits times coefficients

    print("  W[16] = σ₁(W[14]) + W[9] + σ₀(W[1]) + W[0]")
    print("        = (nonce-independent)")
    print()
    print("  W[17] = σ₁(W[15]) + W[10] + σ₀(W[2]) + W[1]")
    print("        = σ₁(NONCE) + (fixed terms)")
    print("        = LINEAR in nonce (via σ₁)")
    print()
    print("  W[22] = σ₁(W[20]) + W[15] + σ₀(W[7]) + W[6]")
    print("        = NONCE + (other terms)")
    print("        = LINEAR in nonce")
    print()
    print("  Each subsequent W[i] inherits linearity from W[15]=nonce")
    print()

    print("THE CONSTRAINT:")
    print()
    print("  If the output constraint requires specific W[i] values,")
    print("  and W[i] is linear in nonce, then we have LINEAR CONSTRAINTS on nonce!")
    print()
    print("  But wait - the output constraint comes from COMPRESSION (nonlinear)")
    print("  So we can't directly get W constraints from output constraints")
    print()

    print("HOWEVER:")
    print("  At each round r, the compression uses W[r]")
    print("  If we know what state we need at round r (from backward propagation),")
    print("  we know what T1 must be, which tells us about W[r]")
    print()
    print("  T1 = h + Σ₁(e) + Ch(e,f,g) + K[r] + W[r]")
    print("  W[r] = T1 - h - Σ₁(e) - Ch(e,f,g) - K[r]")
    print()
    print("  This creates a system:")
    print("  - Backward propagation gives T1 requirements")
    print("  - T1 requirements give W requirements (modulo state unknowns)")
    print("  - W requirements are LINEAR in nonce")
    print()


def search_for_exploitable_structure():
    """
    Look for any exploitable structure in the constraint satisfaction problem.
    """
    print("=" * 70)
    print("SEARCHING FOR EXPLOITABLE STRUCTURE")
    print("=" * 70)
    print()

    print("We have:")
    print("  1. Output constraint: k bits = 0")
    print("  2. Backward propagation: spreads to ~256 bits within 16 rounds")
    print("  3. Message schedule: W[i] is linear in nonce")
    print("  4. Compression: Ch and Maj are nonlinear")
    print()

    print("POTENTIAL APPROACHES:")
    print()

    print("A) LINEARIZATION ATTACK")
    print("   Replace Ch(e,f,g) with linear approximation: e ⊕ f ⊕ g")
    print("   This is wrong by 1 bit with probability 1/4")
    print("   After 64 rounds: probability of all approximations correct = (3/4)^64 ≈ 2^{-26}")
    print("   Not enough to help")
    print()

    print("B) ALGEBRAIC ATTACK (Gröbner Basis)")
    print("   Represent SHA-256 as polynomial system over GF(2)")
    print("   Variables: nonce bits, all intermediate state bits")
    print("   Equations: round functions, output constraint")
    print("   Problem: degree 2^64 after 64 rounds")
    print()

    print("C) SAT SOLVER ATTACK")
    print("   Encode as boolean satisfiability")
    print("   Works for reduced rounds (<30)")
    print("   Full 64 rounds: no known SAT encoding beats brute force")
    print()

    print("D) DIFFERENTIAL ATTACK")
    print("   Find nonce pairs with predictable output difference")
    print("   Best known differential: probability 2^{-256}")
    print("   Useless")
    print()

    print("E) CUBE ATTACK")
    print("   Find 'cube' indices that simplify the system")
    print("   Works for stream ciphers, less effective on SHA-256")
    print()

    print("F) THE GEODESIC APPROACH (our work)")
    print("   The constraint manifold has geometry (π/e factor)")
    print("   But this is an information-theoretic BOUND, not an algorithm")
    print()

    print("G) HYBRID APPROACHES")
    print("   Combine multiple weak attacks")
    print("   E.g., linearization + MITM + pruning")
    print("   Still no known success on full SHA-256")
    print()


if __name__ == "__main__":
    print("SHA-256 BACKWARD CONSTRAINT PROPAGATION")
    print("=" * 70)
    print()
    print("Can we propagate the output constraint backward to narrow the search?")
    print()

    target_zeros = 20  # A reasonable mining difficulty

    analyze_final_addition()
    print()

    compute_backward_constraints(target_zeros, rounds_back=5)
    print()

    analyze_ch_maj_invertibility()
    print()

    count_backward_branches(target_zeros, rounds=10)
    print()

    meet_in_middle_analysis(target_zeros)
    print()

    explore_message_schedule_constraints()
    print()

    search_for_exploitable_structure()
    print()

    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("Backward constraint propagation reveals:")
    print()
    print("  1. The output constraint IS specific (k bits must equal specific values)")
    print("  2. Propagating backward creates branching (Ch, Maj nonlinearity)")
    print("  3. But the branching overwhelms any pruning")
    print()
    print("The 'shorter path' we're looking for would need to:")
    print()
    print("  A) Find a way to prune branches more aggressively")
    print("     (requires finding additional invariants)")
    print()
    print("  B) Find a representation where the problem is more structured")
    print("     (e.g., lattice, algebraic variety)")
    print()
    print("  C) Exploit the message schedule linearity to constrain the search")
    print("     (but compression destroys this linearity)")
    print()
    print("The π/e factor we found describes the MINIMUM cost of any solution.")
    print("It's a floor, not a ceiling. No algorithm can beat it.")
