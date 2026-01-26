#!/usr/bin/env python3
"""
Experiment 2.2: Prime Number Distributions - Pure Mathematics

This is a critical experiment: testing pure mathematics with no physical substrate.

If primes show π/e → Mathematics IS information (profound implication)
If primes show φ/√3 → Mathematics IS geometry
If balanced → Mathematics spans both regimes

METHODOLOGY:
- Generate prime gap sequences for first N primes
- Build multiple matrix representations
- SVD analysis with same constants/thresholds
- Null hypothesis: random gaps with same distribution
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

# Constants - identical to all other analyses
PI = math.pi
E = math.e
PHI = (1 + math.sqrt(5)) / 2
SQRT2 = math.sqrt(2)
SQRT3 = math.sqrt(3)

CONSTANTS = {
    "pi/e": PI / E,
    "e/pi": E / PI,
    "phi": PHI,
    "1/phi": 1 / PHI,
    "sqrt2": SQRT2,
    "1/sqrt2": 1 / SQRT2,
    "sqrt3": SQRT3,
    "e": E,
    "pi": PI,
}

MATCH_THRESHOLD = 0.05


def count_constant_matches(S: np.ndarray, bidirectional: bool = True) -> Dict[str, int]:
    """Count matches for each constant in singular value ratios."""
    matches = {name: 0 for name in CONSTANTS}

    for i in range(min(len(S) - 1, 20)):
        for j in range(i + 1, min(len(S), i + 6)):
            if S[j] > 1e-10:
                ratio1 = S[i] / S[j]
                ratio2 = S[j] / S[i] if bidirectional else None

                for const_name, const_val in CONSTANTS.items():
                    error1 = abs(ratio1 - const_val) / const_val
                    if error1 < MATCH_THRESHOLD:
                        matches[const_name] += 1

                    if bidirectional and ratio2 is not None:
                        error2 = abs(ratio2 - const_val) / const_val
                        if error2 < MATCH_THRESHOLD:
                            matches[const_name] += 1

    return matches


def sieve_of_eratosthenes(limit: int) -> List[int]:
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False

    return [i for i in range(limit + 1) if is_prime[i]]


def get_primes(n_primes: int) -> List[int]:
    """Get first n_primes prime numbers."""
    # Estimate upper bound using prime number theorem
    if n_primes < 6:
        limit = 15
    else:
        limit = int(n_primes * (math.log(n_primes) + math.log(math.log(n_primes))) * 1.3)

    primes = sieve_of_eratosthenes(limit)

    while len(primes) < n_primes:
        limit = int(limit * 1.5)
        primes = sieve_of_eratosthenes(limit)

    return primes[:n_primes]


def compute_prime_gaps(primes: List[int]) -> np.ndarray:
    """Compute gaps between consecutive primes: g(n) = p(n+1) - p(n)."""
    return np.array([primes[i+1] - primes[i] for i in range(len(primes)-1)])


def build_gap_position_matrix(gaps: np.ndarray, window_size: int = 100) -> np.ndarray:
    """Build matrix of [gap × position] windows.

    Each row is a window of consecutive gaps.
    This captures local structure in the gap sequence.
    """
    n_windows = len(gaps) - window_size + 1
    matrix = np.zeros((n_windows, window_size))

    for i in range(n_windows):
        matrix[i] = gaps[i:i+window_size]

    return matrix


def build_gap_frequency_matrix(gaps: np.ndarray, max_gap: int = 100) -> np.ndarray:
    """Build matrix of gap frequencies in windows.

    Rows: windows of the prime sequence
    Columns: gap sizes (2, 4, 6, ..., max_gap)
    Values: frequency of each gap in that window
    """
    window_size = 1000
    n_windows = len(gaps) // window_size

    # Gap sizes are always even (except 1 between 2 and 3)
    gap_sizes = list(range(2, max_gap + 1, 2))
    n_gaps = len(gap_sizes)

    matrix = np.zeros((n_windows, n_gaps))

    for w in range(n_windows):
        window_gaps = gaps[w*window_size:(w+1)*window_size]
        for i, g in enumerate(gap_sizes):
            matrix[w, i] = np.sum(window_gaps == g)

    return matrix


def build_gap_autocorr_matrix(gaps: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """Build autocorrelation matrix of gaps at different lags.

    Tests if there's periodic structure in the gap sequence.
    """
    n = len(gaps)
    window_size = 10000
    n_windows = n // window_size

    matrix = np.zeros((n_windows, max_lag))

    for w in range(n_windows):
        window = gaps[w*window_size:(w+1)*window_size]
        window_mean = np.mean(window)
        window_std = np.std(window)

        if window_std > 0:
            window_norm = (window - window_mean) / window_std
            for lag in range(max_lag):
                if lag < len(window_norm) - 1:
                    matrix[w, lag] = np.mean(window_norm[:-lag-1] * window_norm[lag+1:])

    return matrix


def analyze_matrix(matrix: np.ndarray, name: str) -> Dict:
    """Analyze a matrix for constant matches."""
    print(f"\n--- {name} ---")
    print(f"Shape: {matrix.shape}")

    # SVD
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    # Count matches
    matches = count_constant_matches(S, bidirectional=True)
    total = sum(matches.values())

    # Compute fractions
    pi_e = matches["pi/e"] + matches["e/pi"]
    phi_sqrt3 = matches["phi"] + matches["1/phi"] + matches["sqrt3"]

    pi_e_frac = pi_e / total if total > 0 else 0
    phi_sqrt3_frac = phi_sqrt3 / total if total > 0 else 0

    print(f"Total matches: {total}")
    print(f"π/e fraction: {pi_e_frac*100:.1f}%")
    print(f"φ/√3 fraction: {phi_sqrt3_frac*100:.1f}%")

    for const_name, count in sorted(matches.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {const_name}: {count}")

    return {
        "name": name,
        "shape": matrix.shape,
        "matches": matches,
        "total_matches": total,
        "pi_e_matches": pi_e,
        "phi_sqrt3_matches": phi_sqrt3,
        "pi_e_fraction": float(pi_e_frac),
        "phi_sqrt3_fraction": float(phi_sqrt3_frac),
        "top_singular_values": list(S[:15]),
    }


def null_hypothesis_test(matrix: np.ndarray, n_samples: int = 100) -> Dict:
    """Compare to random matrices with same distribution."""

    # Get actual matches
    _, S, _ = np.linalg.svd(matrix, full_matrices=False)
    actual_matches = count_constant_matches(S, bidirectional=True)
    actual_total = sum(actual_matches.values())

    # Generate random matrices
    random_totals = []
    random_by_const = {name: [] for name in CONSTANTS}

    for _ in range(n_samples):
        # Random matrix with same shape and similar distribution
        random_matrix = np.random.permutation(matrix.flatten()).reshape(matrix.shape)
        _, S_random, _ = np.linalg.svd(random_matrix, full_matrices=False)
        random_matches = count_constant_matches(S_random, bidirectional=True)
        random_totals.append(sum(random_matches.values()))

        for name, count in random_matches.items():
            random_by_const[name].append(count)

    # Statistics
    random_mean = np.mean(random_totals)
    random_std = np.std(random_totals)

    if random_std > 0:
        z_score = (actual_total - random_mean) / random_std
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    else:
        z_score = float('inf') if actual_total > random_mean else 0
        p_value = 0.0

    return {
        "actual_total": actual_total,
        "random_mean": float(random_mean),
        "random_std": float(random_std),
        "z_score": float(z_score),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }


def main():
    """Run prime number geometry experiment."""

    print("=" * 70)
    print("EXPERIMENT 2.2: PRIME NUMBER DISTRIBUTIONS - PURE MATHEMATICS")
    print("=" * 70)
    print("\nThis tests pure mathematics with NO physical substrate.")
    print("If primes show π/e → Mathematics IS information")
    print("If primes show φ/√3 → Mathematics IS geometry")

    results = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "2.2_prime_numbers",
        "hypothesis": "Test if pure mathematics shows π/e or φ/√3",
        "analyses": {},
    }

    # Generate primes
    print("\n" + "=" * 70)
    print("Generating primes...")
    print("=" * 70)

    n_primes = 1_000_000
    print(f"Generating first {n_primes:,} primes...")
    primes = get_primes(n_primes)
    print(f"Largest prime: {primes[-1]:,}")

    # Compute gaps
    gaps = compute_prime_gaps(primes)
    print(f"Number of gaps: {len(gaps):,}")
    print(f"Gap statistics: mean={np.mean(gaps):.2f}, max={np.max(gaps)}")

    results["data"] = {
        "n_primes": n_primes,
        "largest_prime": int(primes[-1]),
        "n_gaps": len(gaps),
        "mean_gap": float(np.mean(gaps)),
        "max_gap": int(np.max(gaps)),
    }

    # Analysis 1: Gap-Position Matrix
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Gap-Position Windows")
    print("=" * 70)

    gap_pos_matrix = build_gap_position_matrix(gaps, window_size=100)
    analysis1 = analyze_matrix(gap_pos_matrix, "Gap-Position Matrix")
    results["analyses"]["gap_position"] = analysis1

    # Analysis 2: Gap Frequency Matrix
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Gap Frequency Distribution")
    print("=" * 70)

    gap_freq_matrix = build_gap_frequency_matrix(gaps, max_gap=100)
    analysis2 = analyze_matrix(gap_freq_matrix, "Gap Frequency Matrix")
    results["analyses"]["gap_frequency"] = analysis2

    # Analysis 3: Autocorrelation Matrix
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Gap Autocorrelation")
    print("=" * 70)

    autocorr_matrix = build_gap_autocorr_matrix(gaps, max_lag=50)
    analysis3 = analyze_matrix(autocorr_matrix, "Autocorrelation Matrix")
    results["analyses"]["autocorrelation"] = analysis3

    # Null hypothesis test (on gap-position matrix)
    print("\n" + "=" * 70)
    print("NULL HYPOTHESIS TEST")
    print("=" * 70)

    print("\nComparing to shuffled gap sequences...")
    null_result = null_hypothesis_test(gap_pos_matrix, n_samples=100)
    results["null_hypothesis"] = null_result

    print(f"Actual matches: {null_result['actual_total']}")
    print(f"Random: {null_result['random_mean']:.1f} ± {null_result['random_std']:.1f}")
    print(f"Z-score: {null_result['z_score']:.2f}")
    print(f"P-value: {null_result['p_value']:.6f}")
    print(f"Significant: {null_result['significant']}")

    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    all_pi_e = [a["pi_e_fraction"] for a in results["analyses"].values()]
    all_phi_sqrt3 = [a["phi_sqrt3_fraction"] for a in results["analyses"].values()]

    mean_pi_e = np.mean(all_pi_e)
    mean_phi_sqrt3 = np.mean(all_phi_sqrt3)

    print(f"\nAcross all analyses:")
    print(f"  Mean π/e fraction: {mean_pi_e*100:.1f}%")
    print(f"  Mean φ/√3 fraction: {mean_phi_sqrt3*100:.1f}%")

    results["aggregate"] = {
        "mean_pi_e": float(mean_pi_e),
        "mean_phi_sqrt3": float(mean_phi_sqrt3),
    }

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if mean_pi_e > mean_phi_sqrt3 * 1.5:
        print(f"\n✓ π/e DOMINATES ({mean_pi_e*100:.1f}% vs {mean_phi_sqrt3*100:.1f}%)")
        print("  → Mathematics IS information")
        print("  → Pure number theory shares structure with neural networks")
        verdict = "MATHEMATICS_IS_INFORMATION"
    elif mean_phi_sqrt3 > mean_pi_e * 1.5:
        print(f"\n✓ φ/√3 DOMINATES ({mean_phi_sqrt3*100:.1f}% vs {mean_pi_e*100:.1f}%)")
        print("  → Mathematics IS geometry")
        print("  → Pure number theory shares structure with crystals/spacetime")
        verdict = "MATHEMATICS_IS_GEOMETRY"
    else:
        print(f"\n~ BALANCED ({mean_pi_e*100:.1f}% π/e vs {mean_phi_sqrt3*100:.1f}% φ/√3)")
        print("  → Mathematics spans BOTH information and geometry")
        print("  → Primes exist at the boundary, like biology")
        verdict = "MATHEMATICS_SPANS_BOTH"

    results["verdict"] = verdict

    # Additional analysis: specific mathematical relationships
    print("\n" + "-" * 40)
    print("Notable mathematical observations:")

    # Check if 21 appears in prime gap statistics
    gap_21_count = np.sum(np.abs(gaps - 21) < 1)  # Gaps of 20 or 22
    print(f"  Gaps near 21 (20-22): {gap_21_count:,}")

    # First gap of exactly 20
    first_20 = np.where(gaps == 20)[0]
    if len(first_20) > 0:
        print(f"  First gap of 20 at prime index: {first_20[0]}")

    # Twin prime count (gap = 2)
    twin_count = np.sum(gaps == 2)
    print(f"  Twin prime pairs (gap=2): {twin_count:,}")

    results["mathematical_notes"] = {
        "gaps_near_21": int(gap_21_count),
        "twin_prime_pairs": int(twin_count),
    }

    # Save results
    output_dir = Path(__file__).parent.parent.parent / "data" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"prime_geometry_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    results = main()
