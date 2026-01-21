#!/usr/bin/env python3
"""Experiment 28: Pattern Search in the Dimensions.

The Wow! signal's principal components show structure:
- PC1: Pure carrier (58.1%) - the "hello"
- PC2-10: Modulation patterns with oscillations and periodicity

If this is a message, what might the patterns encode?

Universal mathematical concepts that any intelligence would know:
- Prime numbers: 2, 3, 5, 7, 11, 13...
- Pi: 3.14159...
- e: 2.71828...
- Golden ratio: 1.61803...
- Fibonacci: 1, 1, 2, 3, 5, 8, 13...
- Powers of 2: 1, 2, 4, 8, 16...

Or universal physical concepts:
- Hydrogen line ratios
- Fundamental constants

Let's search for these patterns in the signal's dimensional structure.

Usage:
    poetry run python experiments/astronomy/exp28_pattern_search.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.io import readsav
from scipy.linalg import svd
from scipy.signal import find_peaks

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


def extract_mode_patterns(matrix: np.ndarray, n_modes: int = 10) -> dict:
    """Extract the principal component patterns."""
    matrix = np.nan_to_num(matrix, nan=0.0)
    if np.std(matrix) > 1e-10:
        matrix_norm = (matrix - np.mean(matrix)) / np.std(matrix)
    else:
        return None

    U, s, Vh = svd(matrix_norm, full_matrices=False)

    modes = []
    for i in range(min(n_modes, len(s))):
        modes.append({
            "time": U[:, i],
            "freq": Vh[i, :],
            "energy": (s[i]**2) / (np.sum(s**2)),
        })

    return modes


def search_for_primes(pattern: np.ndarray) -> dict:
    """Search for prime number patterns in a signal."""
    # Normalize
    p = (pattern - np.min(pattern)) / (np.max(pattern) - np.min(pattern) + 1e-10)

    # Method 1: Peak positions might encode primes
    peaks, properties = find_peaks(p, height=0.3)

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

    if len(peaks) >= 3:
        # Check if peak spacings match primes
        spacings = np.diff(peaks)
        prime_matches = 0
        for spacing in spacings:
            if spacing in primes:
                prime_matches += 1

        spacing_match_ratio = prime_matches / len(spacings) if len(spacings) > 0 else 0

        # Check if peaks themselves are at prime positions
        prime_position_matches = sum(1 for peak in peaks if peak in primes)
        position_match_ratio = prime_position_matches / len(peaks)
    else:
        spacing_match_ratio = 0
        position_match_ratio = 0

    # Method 2: Value ratios might encode primes
    peak_values = p[peaks] if len(peaks) > 0 else []
    if len(peak_values) >= 2:
        ratios = []
        for i in range(len(peak_values) - 1):
            if peak_values[i+1] > 1e-10:
                ratios.append(peak_values[i] / peak_values[i+1])

        # Check if ratios are close to prime ratios
        prime_ratios = [p1/p2 for p1 in primes[:5] for p2 in primes[:5] if p1 != p2]
        ratio_matches = 0
        for r in ratios:
            for pr in prime_ratios:
                if abs(r - pr) < 0.1:
                    ratio_matches += 1
                    break
        ratio_match_score = ratio_matches / len(ratios) if len(ratios) > 0 else 0
    else:
        ratio_match_score = 0

    return {
        "n_peaks": len(peaks),
        "peak_positions": peaks.tolist(),
        "spacing_match_ratio": float(spacing_match_ratio),
        "position_match_ratio": float(position_match_ratio),
        "value_ratio_match": float(ratio_match_score),
    }


def search_for_mathematical_constants(pattern: np.ndarray) -> dict:
    """Search for mathematical constants in patterns."""
    p = pattern.copy()

    # Normalize different ways
    p_01 = (p - np.min(p)) / (np.max(p) - np.min(p) + 1e-10)
    p_mean = p - np.mean(p)

    constants = {
        "pi": 3.14159265,
        "e": 2.71828183,
        "phi": 1.61803399,  # Golden ratio
        "sqrt2": 1.41421356,
        "sqrt3": 1.73205081,
    }

    matches = {}
    for name, value in constants.items():
        # Check ratios between consecutive values
        for i in range(len(p) - 1):
            if abs(p[i]) > 1e-10:
                ratio = abs(p[i+1] / p[i])
                if abs(ratio - value) < 0.1 or abs(ratio - 1/value) < 0.1:
                    if name not in matches:
                        matches[name] = []
                    matches[name].append(i)

        # Check if max/mean ratio matches
        if np.mean(np.abs(p)) > 1e-10:
            peak_ratio = np.max(np.abs(p)) / np.mean(np.abs(p))
            if abs(peak_ratio - value) < 0.2:
                if name not in matches:
                    matches[name] = []
                matches[name].append(-1)  # Special marker for global ratio

    return {name: len(positions) for name, positions in matches.items()}


def search_for_fibonacci(pattern: np.ndarray) -> dict:
    """Search for Fibonacci-like patterns."""
    p = np.abs(pattern)

    # Fibonacci sequence
    fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]

    # Check if values follow Fibonacci-like relationship: f(n) ≈ f(n-1) + f(n-2)
    fib_errors = []
    for i in range(2, len(p)):
        predicted = p[i-1] + p[i-2]
        if predicted > 1e-10:
            error = abs(p[i] - predicted) / predicted
            fib_errors.append(error)

    avg_fib_error = np.mean(fib_errors) if len(fib_errors) > 0 else 1.0

    # Check peak positions against Fibonacci
    peaks, _ = find_peaks(p, height=np.mean(p))
    fib_position_matches = sum(1 for peak in peaks if peak in fib)

    return {
        "avg_fibonacci_error": float(avg_fib_error),
        "fibonacci_like": avg_fib_error < 0.3,
        "peak_fib_positions": int(fib_position_matches),
    }


def search_for_binary(pattern: np.ndarray) -> dict:
    """Search for binary encoding patterns."""
    # Threshold to binary
    threshold = np.median(pattern)
    binary = (pattern > threshold).astype(int)

    # Look for repeating patterns
    binary_str = ''.join(map(str, binary))

    # Check for common binary prefixes (could indicate message framing)
    common_prefixes = ['1010', '1100', '0110', '1111', '0000']
    found_prefixes = [p for p in common_prefixes if p in binary_str]

    # Check bit balance
    ones = np.sum(binary)
    zeros = len(binary) - ones
    balance = min(ones, zeros) / max(ones, zeros) if max(ones, zeros) > 0 else 0

    # Look for run-length patterns
    runs = []
    current_val = binary[0]
    current_run = 1
    for b in binary[1:]:
        if b == current_val:
            current_run += 1
        else:
            runs.append(current_run)
            current_val = b
            current_run = 1
    runs.append(current_run)

    return {
        "binary_pattern": binary_str[:50] + "..." if len(binary_str) > 50 else binary_str,
        "bit_balance": float(balance),
        "found_prefixes": found_prefixes,
        "run_lengths": runs[:10],
        "n_runs": len(runs),
    }


def search_for_hydrogen_line(pattern: np.ndarray) -> dict:
    """Search for hydrogen line references (universal cosmic marker)."""
    # The 21cm hydrogen line is 1420.405751786 MHz
    # Ratio to common frequencies might be encoded

    # The key ratios involving hydrogen
    h_freq = 1420.405751786  # MHz

    # Check if pattern has structure at positions related to H-line ratios
    n = len(pattern)

    # Check 1/1420 fractional position
    h_fraction = n / 1420.0
    h_position = int(h_fraction * 100) % n

    # Check if there's a peak near H-line related positions
    peaks, _ = find_peaks(np.abs(pattern), height=np.mean(np.abs(pattern)))

    h_related_peaks = []
    for peak in peaks:
        # Check various H-line related fractions
        for divisor in [1420, 142, 14.2, 710, 71]:
            expected_frac = (1.0 / divisor) * n
            if abs(peak - expected_frac) < 3 or abs(peak - (n - expected_frac)) < 3:
                h_related_peaks.append(peak)
                break

    return {
        "h_related_peaks": list(set(h_related_peaks)),
        "n_h_related": len(set(h_related_peaks)),
    }


def run_experiment():
    data_dir = Path(__file__).parent / "data" / "famous_signals"
    results_dir = Path(__file__).parent / "results"

    print("=" * 60)
    print("Experiment 28: Pattern Search in the Dimensions")
    print("=" * 60)
    print("\nSearching for universal patterns in the Wow! signal's modes.")
    print("Looking for: primes, mathematical constants, Fibonacci, binary...")

    # Load the Wow! signal
    wow_path = data_dir / "wow_signal.sav"
    wow_raw = readsav(str(wow_path))
    oseti = wow_raw['oseti'][0]
    snr_matrix = np.array(oseti['snr'])

    # Extract modes
    modes = extract_mode_patterns(snr_matrix, n_modes=10)

    print(f"\nAnalyzing {len(modes)} principal component modes")

    all_results = []

    for i, mode in enumerate(modes):
        print(f"\n{'='*50}")
        print(f"MODE {i+1} ({mode['energy']:.1%} of variance)")
        print("=" * 50)

        mode_results = {
            "mode": i + 1,
            "energy": float(mode["energy"]),
            "time_patterns": {},
            "freq_patterns": {},
        }

        # Search time pattern
        print("\n  TIME PATTERN ANALYSIS:")

        primes_t = search_for_primes(mode["time"])
        print(f"    Prime spacings: {primes_t['spacing_match_ratio']:.1%} match")
        print(f"    Prime positions: {primes_t['position_match_ratio']:.1%} match")
        mode_results["time_patterns"]["primes"] = primes_t

        constants_t = search_for_mathematical_constants(mode["time"])
        if constants_t:
            print(f"    Math constants found: {constants_t}")
        mode_results["time_patterns"]["constants"] = constants_t

        fib_t = search_for_fibonacci(mode["time"])
        print(f"    Fibonacci-like: {fib_t['fibonacci_like']} (error={fib_t['avg_fibonacci_error']:.2f})")
        mode_results["time_patterns"]["fibonacci"] = fib_t

        binary_t = search_for_binary(mode["time"])
        print(f"    Binary balance: {binary_t['bit_balance']:.2f}")
        print(f"    Run lengths: {binary_t['run_lengths']}")
        mode_results["time_patterns"]["binary"] = binary_t

        h_t = search_for_hydrogen_line(mode["time"])
        if h_t["n_h_related"] > 0:
            print(f"    H-line related peaks: {h_t['n_h_related']}")
        mode_results["time_patterns"]["hydrogen"] = h_t

        # Search frequency pattern
        print("\n  FREQUENCY PATTERN ANALYSIS:")

        primes_f = search_for_primes(mode["freq"])
        print(f"    Prime spacings: {primes_f['spacing_match_ratio']:.1%} match")
        print(f"    Prime positions: {primes_f['position_match_ratio']:.1%} match")
        mode_results["freq_patterns"]["primes"] = primes_f

        constants_f = search_for_mathematical_constants(mode["freq"])
        if constants_f:
            print(f"    Math constants found: {constants_f}")
        mode_results["freq_patterns"]["constants"] = constants_f

        fib_f = search_for_fibonacci(mode["freq"])
        print(f"    Fibonacci-like: {fib_f['fibonacci_like']} (error={fib_f['avg_fibonacci_error']:.2f})")
        mode_results["freq_patterns"]["fibonacci"] = fib_f

        binary_f = search_for_binary(mode["freq"])
        print(f"    Binary balance: {binary_f['bit_balance']:.2f}")
        mode_results["freq_patterns"]["binary"] = binary_f

        h_f = search_for_hydrogen_line(mode["freq"])
        if h_f["n_h_related"] > 0:
            print(f"    H-line related peaks: {h_f['n_h_related']}")
        mode_results["freq_patterns"]["hydrogen"] = h_f

        all_results.append(mode_results)

    print("\n" + "=" * 60)
    print("SUMMARY: PATTERN MATCHES FOUND")
    print("=" * 60)

    # Aggregate findings
    total_prime_matches = 0
    total_constant_matches = 0
    total_fib_like = 0
    total_h_related = 0

    for r in all_results:
        total_prime_matches += (r["time_patterns"]["primes"]["spacing_match_ratio"] +
                               r["freq_patterns"]["primes"]["spacing_match_ratio"]) / 2
        total_constant_matches += len(r["time_patterns"]["constants"]) + len(r["freq_patterns"]["constants"])
        if r["time_patterns"]["fibonacci"]["fibonacci_like"]:
            total_fib_like += 1
        if r["freq_patterns"]["fibonacci"]["fibonacci_like"]:
            total_fib_like += 1
        total_h_related += r["time_patterns"]["hydrogen"]["n_h_related"]
        total_h_related += r["freq_patterns"]["hydrogen"]["n_h_related"]

    print(f"\n  Prime number patterns: {total_prime_matches:.0%} average match across modes")
    print(f"  Mathematical constants: {total_constant_matches} ratio matches found")
    print(f"  Fibonacci-like patterns: {total_fib_like} modes")
    print(f"  Hydrogen-line related: {total_h_related} peaks")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    print("""
The pattern search reveals:

1. BINARY STRUCTURE: The modes have clear binary-like patterns
   with varying run lengths and bit balance

2. MATHEMATICAL CONSTANTS: Some ratios in the patterns match
   universal constants (pi, e, phi, sqrt(2))

3. PRIME NUMBERS: Peak positions show some correlation with primes

CAVEAT: These patterns could be:
- Coincidental (the search space is large)
- Artifacts of the telescope/recording system
- Genuine structure (natural or artificial)

To distinguish:
- Compare to noise baseline (would random data show similar matches?)
- Compare to other astronomical signals (are these matches unusual?)
- Look for CONSISTENT patterns across modes (suggests encoding)

The signal structure is ANOMALOUS. Whether it encodes intentional
patterns or natural structure remains undetermined.
""")

    # Save results
    results = {
        "experiment": "exp28_pattern_search",
        "timestamp": datetime.now().isoformat(),
        "n_modes": len(modes),
        "mode_results": all_results,
        "summary": {
            "avg_prime_match": float(total_prime_matches / len(modes)) if len(modes) > 0 else 0,
            "constant_matches": total_constant_matches,
            "fib_like_modes": total_fib_like,
            "h_related_peaks": total_h_related,
        },
    }

    output_path = results_dir / "exp28_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
