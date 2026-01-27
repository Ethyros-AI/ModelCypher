#!/usr/bin/env python3
"""SHA-256 Differential Analysis.

Measures how single-bit input changes propagate through different round counts.
For a perfect hash function, flipping one input bit should flip ~128 output bits.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcypher.backends import initialize_default_backend
initialize_default_backend()

from modelcypher.core.domain.geometry.hash_analyzer import compute_differential_propagation


def main():
    header = b"ModelCypher Differential Analysis"
    rounds_to_test = [4, 8, 12, 16, 20, 24, 28, 32, 48, 64]

    print("SHA-256 Differential Propagation Analysis")
    print("=" * 70)
    print(f"{'Rounds':>8} {'Mean Hamming':>14} {'Std':>10} {'Min':>8} {'Max':>8} {'Sensitivity Range':>18}")
    print("-" * 70)

    results = {}
    for num_rounds in rounds_to_test:
        print(f"Analyzing {num_rounds} rounds...", end=" ", flush=True)
        diff = compute_differential_propagation(header, num_rounds=num_rounds, n_samples=50)
        results[num_rounds] = diff
        print(
            f"\r{num_rounds:>8} {diff['mean_hamming']:>14.2f} {diff['std_hamming']:>10.2f} "
            f"{diff['min_hamming']:>8.0f} {diff['max_hamming']:>8.0f} "
            f"{diff['bit_sensitivity_range']:>18.2f}"
        )

    print("=" * 70)
    print("\nExpected for random oracle: Mean ≈ 128, Std ≈ 8")
    print("Deviation from 128 indicates differential bias.")
    print()

    # Check for anomalies
    for num_rounds, diff in results.items():
        mean_dev = abs(diff['mean_hamming'] - 128)
        if mean_dev > 5:
            print(f"*** Round {num_rounds}: Mean Hamming = {diff['mean_hamming']:.1f} (deviation: {mean_dev:.1f})")
        if diff['bit_sensitivity_range'] > 20:
            print(f"*** Round {num_rounds}: High sensitivity range = {diff['bit_sensitivity_range']:.1f}")


if __name__ == "__main__":
    main()
