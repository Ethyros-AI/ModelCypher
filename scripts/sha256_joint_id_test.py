#!/usr/bin/env python3
"""Test if joint intrinsic dimension difference persists without fixed header."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcypher.backends import initialize_default_backend
initialize_default_backend()

from modelcypher.core.domain.geometry.hash_analyzer import (
    generate_sha256_dataset,
    generate_random_oracle_dataset,
)
from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
from modelcypher.core.domain._backend import get_default_backend


def main():
    b = get_default_backend()
    id_estimator = IntrinsicDimension(backend=b)

    print("Joint Intrinsic Dimension: SHA-256 vs Random Oracle")
    print("=" * 70)
    print(f"{'Condition':<30} {'SHA-256 Joint ID':>18} {'Random Joint ID':>18}")
    print("-" * 70)

    n_samples = 1000
    n_trials = 3

    for use_header in [True, False]:
        for seed in range(n_trials):
            # SHA-256 data
            sha_inputs, sha_outputs = generate_sha256_dataset(
                n_samples=n_samples,
                num_rounds=64,
                seed=42 + seed,
                use_header=use_header,
            )
            sha_joint = np.concatenate([sha_inputs, sha_outputs], axis=1)
            sha_joint_arr = b.array(sha_joint)
            sha_id = id_estimator.compute_two_nn(sha_joint_arr)

            # Random oracle data
            rand_inputs, rand_outputs = generate_random_oracle_dataset(
                n_samples=n_samples,
                seed=42 + seed + 1000,
            )
            rand_joint = np.concatenate([rand_inputs, rand_outputs], axis=1)
            rand_joint_arr = b.array(rand_joint)
            rand_id = id_estimator.compute_two_nn(rand_joint_arr)

            condition = f"{'With header' if use_header else 'No header'}, trial {seed+1}"
            print(f"{condition:<30} {sha_id.intrinsic_dimension:>18.2f} {rand_id.intrinsic_dimension:>18.2f}")

    print("=" * 70)
    print("\nIf the difference persists without header, it's real structure.")
    print("If it disappears, it was an artifact of the fixed header.")


if __name__ == "__main__":
    main()
