#!/usr/bin/env python3
"""Model SHA-256 entropy flow as a curve.

Instead of measuring entropy, we model it:
- Entropy as a function of round number
- Fit to fundamental forms (exponential, logistic, power law)
- Extract the constants that govern the flow
- Use the model to identify maximum entropy injection points

If SHA-256 has structure, the entropy curve should have predictable form.
If the form involves fundamental constants, we have a lever.
"""

import sys
from pathlib import Path
import struct
import math
from scipy.optimize import curve_fit
from scipy.stats import entropy as scipy_entropy

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Fundamental constants
PI = math.pi
E = math.e
PHI = (1 + math.sqrt(5)) / 2
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


def collect_round_statistics(messages: list[bytes], max_rounds: int = 64) -> dict:
    """Collect entropy statistics at each round for multiple messages.

    Returns dict with:
    - bit_entropy: Shannon entropy of bit distribution at each round
    - word_entropy: Entropy of word value distribution
    - pairwise_mutual_info: MI between consecutive rounds
    - effective_dim: SVD-based effective dimension
    """
    n_messages = len(messages)

    # Collect states at each round
    round_states = {r: [] for r in range(max_rounds + 1)}

    for message in messages:
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
        round_states[0].append((a, b, c, d, e, f, g, h))

        for r in range(max_rounds):
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

            round_states[r + 1].append((a, b, c, d, e, f, g, h))

    # Compute statistics
    results = {
        'rounds': list(range(max_rounds + 1)),
        'bit_entropy': [],
        'effective_dim': [],
        'mean_distance': [],
    }

    for r in range(max_rounds + 1):
        states = np.array(round_states[r], dtype=np.float64)

        # Bit entropy: treat state as 256 bits, compute entropy of bit distribution
        bit_counts = np.zeros(256)
        for state in round_states[r]:
            for word_idx, word in enumerate(state):
                for bit in range(32):
                    bit_idx = word_idx * 32 + bit
                    bit_counts[bit_idx] += (word >> bit) & 1

        bit_probs = bit_counts / n_messages
        bit_probs = bit_probs[bit_probs > 0]
        bit_probs = bit_probs[bit_probs < 1]
        if len(bit_probs) > 0:
            # Entropy per bit, averaged
            h = -np.mean(bit_probs * np.log2(bit_probs + 1e-10) +
                         (1 - bit_probs) * np.log2(1 - bit_probs + 1e-10))
        else:
            h = 1.0
        results['bit_entropy'].append(h)

        # Effective dimension via SVD
        states_norm = states / 0xFFFFFFFF
        centered = states_norm - np.mean(states_norm, axis=0)
        if n_messages > 1:
            try:
                _, S, _ = np.linalg.svd(centered, full_matrices=False)
                S = S[S > 1e-10]
                if len(S) > 0:
                    S_norm = S / np.sum(S)
                    eff_dim = np.exp(-np.sum(S_norm * np.log(S_norm + 1e-10)))
                else:
                    eff_dim = 0
            except:
                eff_dim = 0
        else:
            eff_dim = 0
        results['effective_dim'].append(eff_dim)

        # Mean pairwise distance
        if n_messages > 1:
            dists = []
            for i in range(min(100, n_messages)):
                for j in range(i + 1, min(100, n_messages)):
                    d = np.linalg.norm(states_norm[i] - states_norm[j])
                    dists.append(d)
            results['mean_distance'].append(np.mean(dists))
        else:
            results['mean_distance'].append(0)

    return results


# Candidate curve models
def logistic(r, L, k, r0):
    """Logistic growth: entropy saturates to L."""
    return L / (1 + np.exp(-k * (r - r0)))


def exponential_saturation(r, L, tau):
    """Exponential approach to saturation."""
    return L * (1 - np.exp(-r / tau))


def power_law(r, a, b):
    """Power law: entropy ~ r^b."""
    return a * (r + 1) ** b


def ln_growth(r, a, b):
    """Logarithmic growth: entropy ~ ln(r)."""
    return a * np.log(r + 1) + b


def fit_and_analyze(rounds, values, name):
    """Fit multiple models and report best fit with extracted constants."""
    rounds = np.array(rounds, dtype=np.float64)
    values = np.array(values, dtype=np.float64)

    # Skip if not enough variation
    if np.std(values) < 1e-6:
        return None

    results = {}

    # Try logistic
    try:
        popt, _ = curve_fit(logistic, rounds, values,
                            p0=[np.max(values), 0.5, 10],
                            maxfev=5000)
        pred = logistic(rounds, *popt)
        mse = np.mean((pred - values) ** 2)
        results['logistic'] = {'params': popt, 'mse': mse,
                               'L': popt[0], 'k': popt[1], 'r0': popt[2]}
    except:
        pass

    # Try exponential saturation
    try:
        popt, _ = curve_fit(exponential_saturation, rounds, values,
                            p0=[np.max(values), 5],
                            maxfev=5000)
        pred = exponential_saturation(rounds, *popt)
        mse = np.mean((pred - values) ** 2)
        results['exp_sat'] = {'params': popt, 'mse': mse,
                              'L': popt[0], 'tau': popt[1]}
    except:
        pass

    # Try power law
    try:
        popt, _ = curve_fit(power_law, rounds[1:], values[1:],  # skip r=0
                            p0=[1, 0.5],
                            maxfev=5000)
        pred = power_law(rounds, *popt)
        mse = np.mean((pred - values) ** 2)
        results['power'] = {'params': popt, 'mse': mse,
                            'a': popt[0], 'b': popt[1]}
    except:
        pass

    # Try ln growth
    try:
        popt, _ = curve_fit(ln_growth, rounds, values,
                            p0=[1, 1],
                            maxfev=5000)
        pred = ln_growth(rounds, *popt)
        mse = np.mean((pred - values) ** 2)
        results['ln'] = {'params': popt, 'mse': mse,
                         'a': popt[0], 'b': popt[1]}
    except:
        pass

    return results


def check_constant_matches(value, name):
    """Check if a fitted parameter matches a fundamental constant."""
    matches = []
    constants = {
        'π': PI, 'e': E, 'φ': PHI, 'ln(2)': LN2,
        'π/e': PI/E, 'e/π': E/PI, '1/φ': 1/PHI, '√2': math.sqrt(2),
        '2': 2, '8': 8, '16': 16, '32': 32, '64': 64,
    }

    for cname, cval in constants.items():
        if abs(cval) < 1e-10:
            continue
        rel_error = abs(value - cval) / cval
        if rel_error < 0.05:  # 5% tolerance
            matches.append((cname, cval, rel_error))

    return matches


def main():
    np.random.seed(42)

    print("SHA-256 Entropy Curve Modeling")
    print("=" * 70)
    print("Fitting entropy flow to fundamental curve forms.")
    print()

    # Generate messages
    n_messages = 200
    messages = [b"curve" + np.random.bytes(32) for _ in range(n_messages)]

    print(f"Collecting statistics from {n_messages} message trajectories...")
    stats = collect_round_statistics(messages, max_rounds=64)

    # Fit effective dimension curve (this is the key one)
    print()
    print("-" * 70)
    print("EFFECTIVE DIMENSION CURVE")
    print("-" * 70)

    dim_fits = fit_and_analyze(stats['rounds'], stats['effective_dim'], 'eff_dim')
    if dim_fits:
        best_model = min(dim_fits.items(), key=lambda x: x[1]['mse'])
        print(f"Best model: {best_model[0]} (MSE: {best_model[1]['mse']:.6f})")
        print(f"Parameters:")
        for k, v in best_model[1].items():
            if k not in ['params', 'mse']:
                print(f"  {k} = {v:.6f}")
                matches = check_constant_matches(v, k)
                for cname, cval, err in matches:
                    print(f"      ≈ {cname} = {cval:.6f} (error: {err*100:.2f}%)")

    # Fit distance curve
    print()
    print("-" * 70)
    print("MEAN DISTANCE CURVE")
    print("-" * 70)

    dist_fits = fit_and_analyze(stats['rounds'], stats['mean_distance'], 'distance')
    if dist_fits:
        best_model = min(dist_fits.items(), key=lambda x: x[1]['mse'])
        print(f"Best model: {best_model[0]} (MSE: {best_model[1]['mse']:.6f})")
        print(f"Parameters:")
        for k, v in best_model[1].items():
            if k not in ['params', 'mse']:
                print(f"  {k} = {v:.6f}")
                matches = check_constant_matches(v, k)
                for cname, cval, err in matches:
                    print(f"      ≈ {cname} = {cval:.6f} (error: {err*100:.2f}%)")

    # Key insight: where does entropy saturate?
    print()
    print("-" * 70)
    print("ENTROPY SATURATION ANALYSIS")
    print("-" * 70)

    # Find the round where effective dimension reaches 95% of max
    max_dim = max(stats['effective_dim'])
    saturation_round = None
    for r, d in enumerate(stats['effective_dim']):
        if d > 0.95 * max_dim:
            saturation_round = r
            break

    if saturation_round:
        print(f"Effective dimension saturates at round {saturation_round}")
        print(f"Saturation value: {max_dim:.4f}")

        # Check if saturation round relates to constants
        matches = check_constant_matches(saturation_round, 'saturation_round')
        for cname, cval, err in matches:
            print(f"  Saturation round ≈ {cname} = {cval:.0f}")

    # Print the curve data for inspection
    print()
    print("-" * 70)
    print("EFFECTIVE DIMENSION BY ROUND")
    print("-" * 70)
    print(f"{'Round':<8} {'Eff Dim':<12} {'Distance':<12}")
    for r in [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 48, 64]:
        if r < len(stats['rounds']):
            print(f"{r:<8} {stats['effective_dim'][r]:<12.4f} {stats['mean_distance'][r]:<12.6f}")


if __name__ == "__main__":
    main()
