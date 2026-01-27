#!/usr/bin/env python3
"""SHA-256 as a dynamical system - entropy flow analysis.

Instead of measuring final output statistics, we model the state evolution
and look for geometric structure in how entropy flows through rounds.

Key insights to explore:
1. The message schedule is LINEAR over GF(2) - a 16-dim manifold in 64-dim space
2. The compression function is nonlinear but deterministic
3. Entropy is injected at round 0 and flows through the system
4. The Jacobian of state evolution may have structure related to fundamental constants

This is about finding WHERE entropy concentrates, not just measuring it.
"""

import sys
from pathlib import Path
import struct
import math

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Fundamental constants from ModelCypher
PI = math.pi
E = math.e
PHI = (1 + math.sqrt(5)) / 2
SQRT2 = math.sqrt(2)
PI_OVER_E = PI / E
E_OVER_PI = E / PI

FUNDAMENTAL_CONSTANTS = {
    'π': PI,
    'e': E,
    'φ': PHI,
    '√2': SQRT2,
    'π/e': PI_OVER_E,
    'e/π': E_OVER_PI,
    'ln(2)': math.log(2),
    '1/φ': 1/PHI,
    'π²/6': PI**2/6,  # Basel problem - sum of 1/n²
}

# SHA-256 constants
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


def sha256_state_trajectory(message: bytes, max_rounds: int = 64) -> list[tuple]:
    """Compute the full state trajectory through SHA-256 rounds.

    Returns list of (round, state, W[round]) tuples.
    State is (a, b, c, d, e, f, g, h) as 8 x 32-bit words.
    """
    # Pad message
    msg_len = len(message)
    message = message + b'\x80'
    message = message + b'\x00' * ((56 - (msg_len + 1) % 64) % 64)
    message = message + struct.pack('>Q', msg_len * 8)

    trajectory = []
    h = list(H_INIT)

    for chunk_start in range(0, len(message), 64):
        chunk = message[chunk_start:chunk_start + 64]

        # Message schedule
        w = list(struct.unpack('>16I', chunk))
        for i in range(16, 64):
            s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3)
            s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10)
            w.append((w[i-16] + s0 + w[i-7] + s1) & 0xFFFFFFFF)

        # Initialize working variables
        a, b, c, d, e, f, g, hh = h

        # Record initial state
        trajectory.append((0, (a, b, c, d, e, f, g, hh), w[0]))

        # Compression rounds
        for i in range(min(max_rounds, 64)):
            S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
            ch = (e & f) ^ (~e & g)
            temp1 = (hh + S1 + ch + K[i] + w[i]) & 0xFFFFFFFF
            S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
            maj = (a & b) ^ (a & c) ^ (b & c)
            temp2 = (S0 + maj) & 0xFFFFFFFF

            hh = g
            g = f
            f = e
            e = (d + temp1) & 0xFFFFFFFF
            d = c
            c = b
            b = a
            a = (temp1 + temp2) & 0xFFFFFFFF

            trajectory.append((i + 1, (a, b, c, d, e, f, g, hh), w[min(i+1, 63)]))

    return trajectory


def state_to_float_vector(state: tuple) -> np.ndarray:
    """Convert 8 x 32-bit state to normalized float vector."""
    # Normalize each word to [0, 1]
    return np.array([s / 0xFFFFFFFF for s in state], dtype=np.float64)


def compute_state_entropy(state: tuple) -> float:
    """Compute bit-level entropy of state."""
    bits = []
    for word in state:
        for bit in range(32):
            bits.append((word >> bit) & 1)

    # Count 1s
    ones = sum(bits)
    p1 = ones / 256
    p0 = 1 - p1

    if p0 <= 0 or p1 <= 0:
        return 0.0

    return -p0 * math.log2(p0) - p1 * math.log2(p1)


def compute_inter_round_distance(traj: list) -> list[float]:
    """Compute Euclidean distance between consecutive states."""
    distances = []
    for i in range(1, len(traj)):
        s1 = state_to_float_vector(traj[i-1][1])
        s2 = state_to_float_vector(traj[i][1])
        dist = np.linalg.norm(s2 - s1)
        distances.append(dist)
    return distances


def compute_lyapunov_approximation(trajectories: list[list]) -> list[float]:
    """Approximate Lyapunov exponent by measuring divergence of nearby trajectories.

    This measures how fast perturbations grow - the "chaos rate".
    """
    if len(trajectories) < 2:
        return []

    n_rounds = min(len(t) for t in trajectories)
    divergences = []

    # Compare pairs of trajectories
    for r in range(1, n_rounds):
        dists = []
        for i in range(len(trajectories)):
            for j in range(i + 1, len(trajectories)):
                s1 = state_to_float_vector(trajectories[i][r][1])
                s2 = state_to_float_vector(trajectories[j][r][1])
                dists.append(np.linalg.norm(s2 - s1))

        if dists:
            divergences.append(np.mean(dists))

    # Lyapunov exponent is the log rate of divergence
    lyapunov = []
    for i in range(1, len(divergences)):
        if divergences[i-1] > 1e-10:
            lyapunov.append(math.log(divergences[i] / divergences[i-1] + 1e-10))

    return lyapunov


def find_constant_matches(values: list[float], tolerance: float = 0.05) -> list[tuple]:
    """Find values that match fundamental constants within tolerance."""
    matches = []
    for val in values:
        if abs(val) < 1e-10:
            continue
        for name, const in FUNDAMENTAL_CONSTANTS.items():
            # Check val, 1/val, val², √val
            for transform, transform_name in [
                (val, ''),
                (1/val if val != 0 else 0, '1/'),
                (val**2, '²'),
                (math.sqrt(abs(val)), '√'),
            ]:
                if abs(transform) < 1e-10:
                    continue
                rel_error = abs(transform - const) / const
                if rel_error < tolerance:
                    matches.append((val, f"{transform_name}{name}", const, rel_error))
    return matches


def analyze_state_space_geometry(trajectories: list[list]) -> dict:
    """Analyze the geometric structure of state space trajectories.

    Look for:
    1. Dimensionality collapse (states clustering on a manifold)
    2. Characteristic distances/angles related to fundamental constants
    3. Entropy concentration patterns
    """
    n_rounds = min(len(t) for t in trajectories)

    results = {
        'round_entropies': [],
        'round_distances': [],
        'round_dimensions': [],
        'constant_matches': [],
    }

    for r in range(n_rounds):
        # Collect states at this round
        states = [state_to_float_vector(t[r][1]) for t in trajectories]
        states = np.array(states)

        # Entropy of each state
        entropies = [compute_state_entropy(t[r][1]) for t in trajectories]
        results['round_entropies'].append(np.mean(entropies))

        # Mean pairwise distance at this round
        if len(states) > 1:
            dists = []
            for i in range(len(states)):
                for j in range(i + 1, len(states)):
                    dists.append(np.linalg.norm(states[i] - states[j]))
            results['round_distances'].append(np.mean(dists))

        # SVD to estimate effective dimensionality
        if len(states) > 8:
            centered = states - np.mean(states, axis=0)
            try:
                _, S, _ = np.linalg.svd(centered, full_matrices=False)
                # Effective dimension via entropy
                S_norm = S / (np.sum(S) + 1e-10)
                S_norm = S_norm[S_norm > 1e-10]
                eff_dim = np.exp(-np.sum(S_norm * np.log(S_norm + 1e-10)))
                results['round_dimensions'].append(eff_dim)

                # Check SVD ratios against constants
                for i in range(len(S) - 1):
                    if S[i+1] > 1e-10:
                        ratio = S[i] / S[i+1]
                        matches = find_constant_matches([ratio], tolerance=0.02)
                        for m in matches:
                            results['constant_matches'].append((r, i, m))
            except:
                pass

    return results


def main():
    np.random.seed(42)

    print("SHA-256 Entropy Flow Analysis")
    print("=" * 70)
    print("Modeling SHA-256 as a dynamical system, looking for geometric structure.")
    print()

    # Generate trajectories from similar starting points
    header = b"Entropy flow test"
    n_trajectories = 50

    print(f"Generating {n_trajectories} state trajectories...")
    trajectories = []
    for i in range(n_trajectories):
        nonce = np.random.bytes(32)
        traj = sha256_state_trajectory(header + nonce, max_rounds=64)
        trajectories.append(traj)

    print("Analyzing state space geometry...")
    results = analyze_state_space_geometry(trajectories)

    # Print entropy flow
    print("\n" + "-" * 70)
    print("ENTROPY FLOW BY ROUND")
    print("-" * 70)
    print(f"{'Round':<8} {'Entropy':<12} {'Distance':<12} {'Eff Dim':<12}")
    for r in range(min(20, len(results['round_entropies']))):
        entropy = results['round_entropies'][r] if r < len(results['round_entropies']) else 0
        dist = results['round_distances'][r] if r < len(results['round_distances']) else 0
        dim = results['round_dimensions'][r] if r < len(results['round_dimensions']) else 0
        print(f"{r:<8} {entropy:<12.6f} {dist:<12.6f} {dim:<12.4f}")

    # Print constant matches
    if results['constant_matches']:
        print("\n" + "-" * 70)
        print("FUNDAMENTAL CONSTANT MATCHES IN SVD RATIOS")
        print("-" * 70)
        for round_num, sv_idx, (val, match_name, const, error) in results['constant_matches'][:20]:
            print(f"Round {round_num:2d}, SV ratio {sv_idx}: {val:.6f} ≈ {match_name} = {const:.6f} (error: {error*100:.2f}%)")

    # Compute Lyapunov exponent
    print("\n" + "-" * 70)
    print("LYAPUNOV EXPONENT (divergence rate)")
    print("-" * 70)
    lyapunov = compute_lyapunov_approximation(trajectories)
    if lyapunov:
        mean_lyap = np.mean(lyapunov)
        print(f"Mean Lyapunov exponent: {mean_lyap:.6f}")

        # Check if it matches any constants
        matches = find_constant_matches([mean_lyap, abs(mean_lyap)])
        if matches:
            for val, match_name, const, error in matches:
                print(f"  ≈ {match_name} = {const:.6f} (error: {error*100:.2f}%)")

        # Also check per-round Lyapunov
        print("\nPer-round Lyapunov:")
        for r, l in enumerate(lyapunov[:15]):
            print(f"  Round {r+1}: {l:.6f}")

    # Key geometric quantities to check against constants
    print("\n" + "-" * 70)
    print("KEY GEOMETRIC QUANTITIES")
    print("-" * 70)

    # Ratio of final to initial entropy
    if results['round_entropies']:
        entropy_ratio = results['round_entropies'][-1] / (results['round_entropies'][0] + 1e-10)
        print(f"Entropy ratio (final/initial): {entropy_ratio:.6f}")
        matches = find_constant_matches([entropy_ratio])
        for val, match_name, const, error in matches:
            print(f"  ≈ {match_name} = {const:.6f} (error: {error*100:.2f}%)")

    # Ratio of final to initial pairwise distance
    if results['round_distances'] and len(results['round_distances']) > 1:
        dist_ratio = results['round_distances'][-1] / (results['round_distances'][0] + 1e-10)
        print(f"Distance expansion ratio: {dist_ratio:.6f}")
        matches = find_constant_matches([dist_ratio])
        for val, match_name, const, error in matches:
            print(f"  ≈ {match_name} = {const:.6f} (error: {error*100:.2f}%)")

    # Dimension collapse ratio
    if results['round_dimensions'] and len(results['round_dimensions']) > 1:
        dim_ratio = results['round_dimensions'][-1] / (results['round_dimensions'][0] + 1e-10)
        print(f"Dimension ratio (final/initial): {dim_ratio:.6f}")
        matches = find_constant_matches([dim_ratio])
        for val, match_name, const, error in matches:
            print(f"  ≈ {match_name} = {const:.6f} (error: {error*100:.2f}%)")


if __name__ == "__main__":
    main()
