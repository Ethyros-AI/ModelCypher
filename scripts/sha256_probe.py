#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# SHA-256 Structure Probe
#
# Uses ModelCypher's manifold geometry tools to detect structure in SHA-256.
# If structure exists, it manifests as deviations from random oracle baseline.

"""SHA-256 Structure Probe.

Hypothesis: If SHA-256 has exploitable structure, it will show as:
- Lower intrinsic dimension than 256
- Non-zero CKA between input and output
- Lower effective rank than random oracle
- Non-uniform SVD spectrum

Control: Reduced-round SHA-256 (known to have structure for < ~30 rounds)
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Initialize backend before importing domain code
from modelcypher.backends import initialize_default_backend
initialize_default_backend()

from modelcypher.core.domain.geometry.hash_analyzer import (
    generate_sha256_dataset,
    generate_random_oracle_dataset,
    analyze_structure,
    StructureMetrics,
)


@dataclass
class StatisticalResult:
    """Statistical comparison between SHA-256 and random oracle."""

    metric_name: str
    sha256_mean: float
    sha256_std: float
    random_mean: float
    random_std: float
    effect_size: float  # Cohen's d
    is_significant: bool  # |d| > 0.5


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std < 1e-10:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def run_experiment(
    n_samples: int = 1000,
    num_rounds: int = 64,
    n_trials: int = 5,
    seed: int = 42,
) -> tuple[List[StructureMetrics], List[StructureMetrics]]:
    """Run multiple trials of SHA-256 vs random oracle.

    Args:
        n_samples: Samples per trial
        num_rounds: SHA-256 rounds (64 = full)
        n_trials: Number of independent trials
        seed: Base random seed

    Returns:
        (sha256_results, random_results)
    """
    sha256_results = []
    random_results = []

    for trial in range(n_trials):
        trial_seed = seed + trial * 1000
        print(f"  Trial {trial + 1}/{n_trials}...", end=" ", flush=True)

        # Generate SHA-256 data
        inputs, outputs = generate_sha256_dataset(
            n_samples=n_samples,
            num_rounds=num_rounds,
            seed=trial_seed,
        )
        sha_metrics = analyze_structure(inputs, outputs, num_rounds=num_rounds)
        sha256_results.append(sha_metrics)

        # Generate random oracle baseline
        rand_inputs, rand_outputs = generate_random_oracle_dataset(
            n_samples=n_samples,
            seed=trial_seed + 500,
        )
        rand_metrics = analyze_structure(rand_inputs, rand_outputs, num_rounds=0)
        random_results.append(rand_metrics)

        print(f"ID={sha_metrics.intrinsic_dim_output:.1f} vs {rand_metrics.intrinsic_dim_output:.1f}")

    return sha256_results, random_results


def compare_metrics(
    sha_results: List[StructureMetrics],
    rand_results: List[StructureMetrics],
) -> List[StatisticalResult]:
    """Compare SHA-256 vs random oracle with effect sizes."""

    metrics_to_compare = [
        ("intrinsic_dim_output", "Intrinsic Dimension (output)"),
        ("intrinsic_dim_joint", "Intrinsic Dimension (joint)"),
        ("cka_input_output", "CKA (input-output)"),
        ("effective_rank_output", "Effective Rank (output)"),
        ("svd_ratio_mean", "SVD Ratio Mean"),
        ("svd_ratio_std", "SVD Ratio Std"),
        # Local structure metrics - these should reveal hidden structure
        ("local_hamming_correlation", "Local Hamming Correlation"),
        ("bit_bias", "Bit Bias"),
        ("pairwise_bit_correlation", "Pairwise Bit Correlation"),
    ]

    results = []
    for attr, name in metrics_to_compare:
        sha_values = [getattr(m, attr) for m in sha_results]
        rand_values = [getattr(m, attr) for m in rand_results]

        d = cohens_d(sha_values, rand_values)

        results.append(StatisticalResult(
            metric_name=name,
            sha256_mean=float(np.mean(sha_values)),
            sha256_std=float(np.std(sha_values)),
            random_mean=float(np.mean(rand_values)),
            random_std=float(np.std(rand_values)),
            effect_size=d,
            is_significant=abs(d) > 0.5,
        ))

    return results


def print_results(
    round_results: dict[int, List[StatisticalResult]],
):
    """Print formatted results."""

    print("\n" + "=" * 80)
    print("SHA-256 STRUCTURE PROBE RESULTS")
    print("=" * 80)

    for num_rounds, stats in sorted(round_results.items()):
        print(f"\n{'─' * 80}")
        print(f"ROUNDS: {num_rounds}")
        print(f"{'─' * 80}")
        print(f"{'Metric':<35} {'SHA-256':>12} {'Random':>12} {'Cohen d':>10} {'Sig?':>6}")
        print(f"{'─' * 35} {'─' * 12} {'─' * 12} {'─' * 10} {'─' * 6}")

        for stat in stats:
            sig = "YES" if stat.is_significant else "no"
            print(
                f"{stat.metric_name:<35} "
                f"{stat.sha256_mean:>12.4f} "
                f"{stat.random_mean:>12.4f} "
                f"{stat.effect_size:>10.3f} "
                f"{sig:>6}"
            )

    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("  - Cohen's d > 0.5 (or < -0.5) indicates meaningful effect")
    print("  - Significant deviations from random oracle suggest structure")
    print("  - Expected: structure at low rounds, disappears by round 64")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Probe SHA-256 for exploitable structure using manifold analysis"
    )
    parser.add_argument(
        "--samples", "-n", type=int, default=500,
        help="Number of samples per trial (default: 500)"
    )
    parser.add_argument(
        "--trials", "-t", type=int, default=3,
        help="Number of independent trials (default: 3)"
    )
    parser.add_argument(
        "--rounds", "-r", type=int, nargs="+", default=[8, 16, 32, 64],
        help="SHA-256 round counts to test (default: 8 16 32 64)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output", "-o", type=str,
        help="Output JSON file for results"
    )
    args = parser.parse_args()

    print("SHA-256 Structure Probe")
    print(f"  Samples: {args.samples}")
    print(f"  Trials: {args.trials}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Seed: {args.seed}")
    print()

    all_results = {}
    all_raw = {}

    for num_rounds in args.rounds:
        print(f"\nAnalyzing {num_rounds}-round SHA-256...")
        sha_results, rand_results = run_experiment(
            n_samples=args.samples,
            num_rounds=num_rounds,
            n_trials=args.trials,
            seed=args.seed,
        )
        stats = compare_metrics(sha_results, rand_results)
        all_results[num_rounds] = stats
        def to_serializable(d):
            """Convert numpy types to Python native types."""
            result = {}
            for k, v in d.items():
                if hasattr(v, 'item'):  # numpy scalar
                    result[k] = v.item()
                elif isinstance(v, (np.bool_, bool)):
                    result[k] = bool(v)
                else:
                    result[k] = v
            return result

        all_raw[num_rounds] = {
            "sha256": [to_serializable(asdict(r)) for r in sha_results],
            "random": [to_serializable(asdict(r)) for r in rand_results],
            "comparison": [to_serializable(asdict(s)) for s in stats],
        }

    print_results(all_results)

    # Summary: any significant findings at 64 rounds?
    if 64 in all_results:
        significant_at_64 = [s for s in all_results[64] if s.is_significant]
        if significant_at_64:
            print("\n*** POTENTIAL STRUCTURE DETECTED AT 64 ROUNDS ***")
            print("Significant metrics:")
            for s in significant_at_64:
                print(f"  - {s.metric_name}: d={s.effect_size:.3f}")
            print("\nThis warrants further investigation.")
        else:
            print("\nNo significant structure detected at 64 rounds.")
            print("SHA-256 appears to behave as a random oracle (within detection limits).")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_raw, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
