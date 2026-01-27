#!/usr/bin/env python3
"""Find the maximum entropy injection point.

The entropy curve shows dimension grows fastest in rounds 0-6.
This is where INPUT has maximum leverage on OUTPUT.

Key insight: the message schedule injects W[0]...W[15] directly into rounds 0-15.
The "injection point" is where message words enter the compression function.

If we can characterize HOW the message structure affects the trajectory
during the injection phase, we might find inputs that "steer" toward
desired outputs.

Information-energy equivalence: injecting a bit costs kT ln(2).
The trajectory geometry should reflect this cost structure.
"""

import sys
from pathlib import Path
import struct
import hashlib
import math

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


PI_OVER_E = math.pi / math.e
LN2 = math.log(2)

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


def analyze_message_word_influence(header: bytes, n_samples: int = 1000):
    """Measure how each message word (W[0]...W[15]) influences the trajectory.

    The header is fixed. We vary each word of the "nonce" portion
    and measure the resulting trajectory change.

    This tells us which input positions have maximum leverage.
    """
    h_init_norm = np.array(H_INIT, dtype=np.float64) / 0xFFFFFFFF

    # Baseline: random nonces
    baseline_states = []
    for _ in range(n_samples):
        nonce = np.random.bytes(32)
        message = header + nonce

        msg_len = len(message)
        padded = message + b'\x80'
        padded = padded + b'\x00' * ((56 - (msg_len + 1) % 64) % 64)
        padded = padded + struct.pack('>Q', msg_len * 8)
        chunk = padded[:64]

        w = list(struct.unpack('>16I', chunk))
        for i in range(16, 64):
            s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3)
            s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10)
            w.append((w[i-16] + s0 + w[i-7] + s1) & 0xFFFFFFFF)

        a, b, c, d, e, f, g, h = H_INIT

        # Run to round 6 (end of injection phase)
        for r in range(6):
            S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
            ch = (e & f) ^ (~e & g)
            temp1 = (h + S1 + ch + K[r] + w[r]) & 0xFFFFFFFF
            S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
            maj = (a & b) ^ (a & c) ^ (b & c)
            temp2 = (S0 + maj) & 0xFFFFFFFF

            h, g, f = g, f, e
            e = (d + temp1) & 0xFFFFFFFF
            d, c, b = c, b, a
            a = (temp1 + temp2) & 0xFFFFFFFF

        state = np.array([a, b, c, d, e, f, g, h], dtype=np.float64) / 0xFFFFFFFF
        baseline_states.append(state)

    baseline_states = np.array(baseline_states)
    baseline_mean = np.mean(baseline_states, axis=0)

    return baseline_mean, baseline_states


def compute_sensitivity_per_round(header: bytes, n_samples: int = 500):
    """Compute how sensitive each round is to input perturbations.

    This identifies the "maximum leverage" rounds.
    """
    results = []

    for _ in range(n_samples):
        base_nonce = np.random.bytes(32)
        base_message = header + base_nonce

        # Perturbed nonce (flip one bit)
        perturbed_nonce = bytearray(base_nonce)
        perturbed_nonce[0] ^= 1  # flip first bit
        perturbed_message = header + bytes(perturbed_nonce)

        # Compute trajectories for both
        base_states = []
        pert_states = []

        for message, states_list in [(base_message, base_states), (perturbed_message, pert_states)]:
            msg_len = len(message)
            padded = message + b'\x80'
            padded = padded + b'\x00' * ((56 - (msg_len + 1) % 64) % 64)
            padded = padded + struct.pack('>Q', msg_len * 8)
            chunk = padded[:64]

            w = list(struct.unpack('>16I', chunk))
            for i in range(16, 64):
                s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3)
                s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10)
                w.append((w[i-16] + s0 + w[i-7] + s1) & 0xFFFFFFFF)

            a, b, c, d, e, f, g, h = H_INIT
            states_list.append(np.array([a, b, c, d, e, f, g, h], dtype=np.float64))

            for r in range(64):
                S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
                ch = (e & f) ^ (~e & g)
                temp1 = (h + S1 + ch + K[r] + w[r]) & 0xFFFFFFFF
                S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
                maj = (a & b) ^ (a & c) ^ (b & c)
                temp2 = (S0 + maj) & 0xFFFFFFFF

                h, g, f = g, f, e
                e = (d + temp1) & 0xFFFFFFFF
                d, c, b = c, b, a
                a = (temp1 + temp2) & 0xFFFFFFFF

                states_list.append(np.array([a, b, c, d, e, f, g, h], dtype=np.float64))

        # Compute per-round sensitivity (distance between trajectories)
        sensitivity = []
        for r in range(65):
            dist = np.linalg.norm(base_states[r] - pert_states[r]) / 0xFFFFFFFF
            sensitivity.append(dist)

        results.append(sensitivity)

    return np.mean(results, axis=0)


def analyze_injection_geometry(header: bytes, n_samples: int = 1000):
    """Analyze the geometry of the entropy injection.

    Key questions:
    1. What is the "shape" of the injection manifold?
    2. Are there directions with higher leverage?
    3. How does leverage relate to fundamental constants?
    """
    h_init_norm = np.array(H_INIT, dtype=np.float64) / 0xFFFFFFFF

    # Collect states at round 6 (end of injection)
    states_6 = []
    hashes = []

    for _ in range(n_samples):
        nonce = np.random.bytes(32)
        message = header + nonce

        msg_len = len(message)
        padded = message + b'\x80'
        padded = padded + b'\x00' * ((56 - (msg_len + 1) % 64) % 64)
        padded = padded + struct.pack('>Q', msg_len * 8)
        chunk = padded[:64]

        w = list(struct.unpack('>16I', chunk))
        for i in range(16, 64):
            s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3)
            s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10)
            w.append((w[i-16] + s0 + w[i-7] + s1) & 0xFFFFFFFF)

        a, b, c, d, e, f, g, h = H_INIT

        for r in range(6):
            S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
            ch = (e & f) ^ (~e & g)
            temp1 = (h + S1 + ch + K[r] + w[r]) & 0xFFFFFFFF
            S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
            maj = (a & b) ^ (a & c) ^ (b & c)
            temp2 = (S0 + maj) & 0xFFFFFFFF

            h, g, f = g, f, e
            e = (d + temp1) & 0xFFFFFFFF
            d, c, b = c, b, a
            a = (temp1 + temp2) & 0xFFFFFFFF

        state_6 = np.array([a, b, c, d, e, f, g, h], dtype=np.float64) / 0xFFFFFFFF
        states_6.append(state_6)

        # Also get final hash
        final_hash = hashlib.sha256(message).digest()
        hashes.append(int.from_bytes(final_hash[:4], 'big'))

    states_6 = np.array(states_6)
    hashes = np.array(hashes)

    # SVD of state-6 manifold
    centered = states_6 - np.mean(states_6, axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    return {
        'singular_values': S,
        'principal_directions': Vt,
        'states': states_6,
        'hashes': hashes,
    }


def main():
    np.random.seed(42)

    print("SHA-256 Entropy Injection Analysis")
    print("=" * 70)
    print("Finding where input has maximum leverage on output.")
    print()

    header = b"injection"

    # 1. Sensitivity per round
    print("Computing per-round sensitivity...")
    sensitivity = compute_sensitivity_per_round(header, n_samples=500)

    print()
    print("-" * 70)
    print("SENSITIVITY BY ROUND (how much does 1-bit flip change state?)")
    print("-" * 70)
    print(f"{'Round':<8} {'Sensitivity':<15} {'Ratio to prev':<15}")

    for r in range(20):
        prev_ratio = sensitivity[r] / sensitivity[r-1] if r > 0 and sensitivity[r-1] > 0 else 0
        print(f"{r:<8} {sensitivity[r]:<15.6f} {prev_ratio:<15.4f}")

    # Find maximum leverage round
    max_round = np.argmax(sensitivity[1:]) + 1  # skip round 0
    print(f"\nMaximum sensitivity at round: {max_round}")
    print(f"Sensitivity value: {sensitivity[max_round]:.6f}")

    # Check if max sensitivity relates to constants
    print()
    for name, const in [('ln(2)', LN2), ('π/e', PI_OVER_E), ('1', 1.0), ('√2', math.sqrt(2))]:
        rel_error = abs(sensitivity[max_round] - const) / const if const > 0 else 0
        if rel_error < 0.1:
            print(f"Max sensitivity ≈ {name} = {const:.6f} (error: {rel_error*100:.2f}%)")

    # 2. Injection geometry
    print()
    print("-" * 70)
    print("INJECTION MANIFOLD GEOMETRY (SVD at round 6)")
    print("-" * 70)

    geom = analyze_injection_geometry(header, n_samples=1000)

    print("Singular values of state-6 manifold:")
    for i, sv in enumerate(geom['singular_values']):
        print(f"  σ_{i}: {sv:.6f}")

    # SVD ratios
    print("\nSingular value ratios:")
    for i in range(len(geom['singular_values']) - 1):
        ratio = geom['singular_values'][i] / geom['singular_values'][i+1]
        print(f"  σ_{i}/σ_{i+1} = {ratio:.6f}")

        # Check against constants
        for name, const in [('π/e', PI_OVER_E), ('√2', math.sqrt(2)), ('φ', (1+math.sqrt(5))/2), ('e', math.e)]:
            rel_error = abs(ratio - const) / const
            if rel_error < 0.05:
                print(f"    ≈ {name} = {const:.6f} (error: {rel_error*100:.2f}%)")

    # 3. Correlation between injection geometry and output
    print()
    print("-" * 70)
    print("INJECTION-OUTPUT CORRELATION")
    print("-" * 70)

    # Project states onto principal directions
    centered_states = geom['states'] - np.mean(geom['states'], axis=0)
    projections = centered_states @ geom['principal_directions'].T

    # Correlate each projection with hash value
    for i in range(min(8, projections.shape[1])):
        corr = np.corrcoef(projections[:, i], geom['hashes'])[0, 1]
        print(f"Correlation(PC_{i}, hash_value): {corr:.6f}")


if __name__ == "__main__":
    main()
