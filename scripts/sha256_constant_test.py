#!/usr/bin/env python3
"""Test if fundamental constants appear consistently in SHA-256 state evolution.

The entropy flow analysis found π/e ≈ 1.1557 appearing in SVD ratios during
the mixing phase (rounds 8-15) with 0.02% error.

This script tests:
1. Does this appear consistently across different random seeds?
2. Does it appear at the same rounds with different inputs?
3. Does it appear in random state trajectories? (null hypothesis)
"""

import sys
from pathlib import Path
import struct
import math

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


PI_OVER_E = math.pi / math.e  # 1.155727...

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
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]

H_INIT = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
]


def rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF


def get_states_at_round(messages: list[bytes], target_round: int) -> np.ndarray:
    """Get state vectors at a specific round for multiple messages."""
    states = []

    for message in messages:
        # Pad message
        msg_len = len(message)
        padded = message + b'\x80'
        padded = padded + b'\x00' * ((56 - (msg_len + 1) % 64) % 64)
        padded = padded + struct.pack('>Q', msg_len * 8)

        chunk = padded[:64]

        # Message schedule
        w = list(struct.unpack('>16I', chunk))
        for i in range(16, 64):
            s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3)
            s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10)
            w.append((w[i-16] + s0 + w[i-7] + s1) & 0xFFFFFFFF)

        # Initialize and run to target round
        a, b, c, d, e, f, g, h = H_INIT

        for i in range(target_round):
            S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
            ch = (e & f) ^ (~e & g)
            temp1 = (h + S1 + ch + K[i] + w[i]) & 0xFFFFFFFF
            S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
            maj = (a & b) ^ (a & c) ^ (b & c)
            temp2 = (S0 + maj) & 0xFFFFFFFF

            h = g
            g = f
            f = e
            e = (d + temp1) & 0xFFFFFFFF
            d = c
            c = b
            b = a
            a = (temp1 + temp2) & 0xFFFFFFFF

        states.append([a/0xFFFFFFFF, b/0xFFFFFFFF, c/0xFFFFFFFF, d/0xFFFFFFFF,
                       e/0xFFFFFFFF, f/0xFFFFFFFF, g/0xFFFFFFFF, h/0xFFFFFFFF])

    return np.array(states)


def get_svd_ratios(states: np.ndarray) -> list[float]:
    """Compute SVD and return consecutive singular value ratios."""
    centered = states - np.mean(states, axis=0)
    try:
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        ratios = []
        for i in range(len(S) - 1):
            if S[i+1] > 1e-10:
                ratios.append(S[i] / S[i+1])
        return ratios
    except:
        return []


def count_pi_over_e_matches(ratios: list[float], tolerance: float = 0.02) -> int:
    """Count how many ratios match π/e within tolerance."""
    count = 0
    for r in ratios:
        rel_error = abs(r - PI_OVER_E) / PI_OVER_E
        if rel_error < tolerance:
            count += 1
    return count


def main():
    print("Testing Fundamental Constant Consistency in SHA-256")
    print("=" * 70)
    print(f"Target constant: π/e = {PI_OVER_E:.6f}")
    print(f"Tolerance: 2%")
    print()

    n_messages = 100  # messages per trial
    n_trials = 10     # independent trials

    # Test SHA-256 at different rounds
    print("SHA-256 State SVD Ratios:")
    print("-" * 70)
    print(f"{'Round':<8} {'Matches/Trial':<15} {'Mean Best Error':<18} {'Consistency':<15}")

    for target_round in [8, 10, 12, 14, 16, 20, 32, 64]:
        matches_per_trial = []
        best_errors = []

        for trial in range(n_trials):
            np.random.seed(trial * 1000)
            messages = [b"test" + np.random.bytes(32) for _ in range(n_messages)]
            states = get_states_at_round(messages, target_round)
            ratios = get_svd_ratios(states)

            matches = count_pi_over_e_matches(ratios, tolerance=0.02)
            matches_per_trial.append(matches)

            if ratios:
                errors = [abs(r - PI_OVER_E) / PI_OVER_E for r in ratios]
                best_errors.append(min(errors))

        mean_matches = np.mean(matches_per_trial)
        std_matches = np.std(matches_per_trial)
        mean_best_error = np.mean(best_errors) * 100 if best_errors else 100

        # Consistency: how often do we see at least one match?
        consistency = sum(1 for m in matches_per_trial if m > 0) / n_trials * 100

        print(f"{target_round:<8} {mean_matches:<15.2f} {mean_best_error:<18.2f}% {consistency:<15.0f}%")

    # Null hypothesis: random states (not from SHA-256)
    print()
    print("Random State Trajectories (Null Hypothesis):")
    print("-" * 70)

    for target_round in [8, 12, 16, 32]:
        matches_per_trial = []
        best_errors = []

        for trial in range(n_trials):
            np.random.seed(trial * 1000 + 500)
            # Random states in [0, 1]^8
            states = np.random.random((n_messages, 8))
            ratios = get_svd_ratios(states)

            matches = count_pi_over_e_matches(ratios, tolerance=0.02)
            matches_per_trial.append(matches)

            if ratios:
                errors = [abs(r - PI_OVER_E) / PI_OVER_E for r in ratios]
                best_errors.append(min(errors))

        mean_matches = np.mean(matches_per_trial)
        mean_best_error = np.mean(best_errors) * 100 if best_errors else 100
        consistency = sum(1 for m in matches_per_trial if m > 0) / n_trials * 100

        print(f"Random    {mean_matches:<15.2f} {mean_best_error:<18.2f}% {consistency:<15.0f}%")

    print()
    print("=" * 70)
    print("If SHA-256 consistently shows more π/e matches than random,")
    print("this indicates the constant is embedded in the algorithm's geometry.")


if __name__ == "__main__":
    main()
