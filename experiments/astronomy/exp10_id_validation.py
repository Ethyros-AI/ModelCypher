#!/usr/bin/env python3
"""Experiment 10: Intrinsic Dimension Validation.

Exp1 found FRB intrinsic dimension = 4.48D from 26D features.
Is this a property of FRBs, or an artifact of the feature extraction?

Test: Run the same ID analysis on white noise processed through
the identical feature extraction pipeline.

If noise also shows ~4-5D: The ID is from feature extraction
If noise shows ~26D: FRBs genuinely have low intrinsic dimension

Usage:
    poetry run python experiments/astronomy/exp10_id_validation.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from modelcypher.backends import initialize_default_backend
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

from shared.data_loader import load_frb_batch
from shared.feature_extraction import batch_extract_features, extract_frb_features


def generate_noise_features(n_samples: int, backend, seed: int = 42):
    """Generate features from white noise through same pipeline as FRBs."""
    rng = np.random.default_rng(seed)
    features = []

    for i in range(n_samples):
        # Same dimensions as real FRB data
        n_freq, n_time = 256, 1024
        waterfall = rng.standard_normal((n_freq, n_time)).astype(np.float32)
        waterfall = backend.array(waterfall)
        time_series = backend.array(rng.standard_normal(n_time).astype(np.float32))
        spectrum = backend.array(rng.standard_normal(n_freq).astype(np.float32))

        frb_feat = extract_frb_features(
            waterfall, time_series, spectrum, backend,
            tns_name=f"noise_{i}"
        )
        features.append(backend.tolist(frb_feat.features))

    return np.array(features)


def generate_random_26d(n_samples: int, seed: int = 42):
    """Generate random 26D vectors (no feature extraction)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, 26)).astype(np.float32)


def run_experiment():
    data_dir = Path(__file__).parent / "data" / "raw"
    results_dir = Path(__file__).parent / "results"

    initialize_default_backend()
    backend = get_default_backend()

    print("=" * 60)
    print("Experiment 10: Intrinsic Dimension Validation")
    print("=" * 60)
    print("\nQuestion: Is FRB ID=4.48D a property of FRBs,")
    print("or an artifact of the feature extraction pipeline?")

    # Load FRBs
    frb_files = sorted(data_dir.glob("FRB*_waterfall.h5"))
    waterfalls = load_frb_batch([str(f) for f in frb_files])
    n_frbs = len(waterfalls)
    print(f"\nLoaded {n_frbs} FRBs")

    # Extract FRB features
    frb_features_arr = batch_extract_features(waterfalls, backend)
    frb_np = np.array(backend.tolist(frb_features_arr))

    # Generate noise features (same pipeline)
    print("Generating noise features through same pipeline...")
    noise_np = generate_noise_features(n_frbs, backend)

    # Generate raw random 26D (no pipeline)
    print("Generating raw random 26D vectors...")
    random_26d = generate_random_26d(n_frbs)

    print("\n" + "=" * 40)
    print("INTRINSIC DIMENSION COMPARISON")
    print("=" * 40)

    id_estimator = IntrinsicDimension(backend)

    # FRB intrinsic dimension
    print("\n--- FRB Features ---")
    frb_id_result = id_estimator.compute(backend.array(frb_np), with_ci=True)
    frb_ci_lower = frb_id_result.ci.lower if frb_id_result.ci else None
    frb_ci_upper = frb_id_result.ci.upper if frb_id_result.ci else None
    print(f"  Intrinsic dimension: {frb_id_result.intrinsic_dimension:.2f}")
    if frb_ci_lower and frb_ci_upper:
        print(f"  95% CI: [{frb_ci_lower:.2f}, {frb_ci_upper:.2f}]")

    # Noise features intrinsic dimension
    print("\n--- Noise Features (same pipeline) ---")
    noise_id_result = id_estimator.compute(backend.array(noise_np), with_ci=True)
    noise_ci_lower = noise_id_result.ci.lower if noise_id_result.ci else None
    noise_ci_upper = noise_id_result.ci.upper if noise_id_result.ci else None
    print(f"  Intrinsic dimension: {noise_id_result.intrinsic_dimension:.2f}")
    if noise_ci_lower and noise_ci_upper:
        print(f"  95% CI: [{noise_ci_lower:.2f}, {noise_ci_upper:.2f}]")

    # Raw random 26D intrinsic dimension
    print("\n--- Raw Random 26D (no pipeline) ---")
    random_id_result = id_estimator.compute(backend.array(random_26d), with_ci=True)
    random_ci_lower = random_id_result.ci.lower if random_id_result.ci else None
    random_ci_upper = random_id_result.ci.upper if random_id_result.ci else None
    print(f"  Intrinsic dimension: {random_id_result.intrinsic_dimension:.2f}")
    if random_ci_lower and random_ci_upper:
        print(f"  95% CI: [{random_ci_lower:.2f}, {random_ci_upper:.2f}]")

    print("\n" + "=" * 40)
    print("FEATURE VARIANCE ANALYSIS")
    print("=" * 40)

    # Check variance per feature dimension
    frb_var = np.var(frb_np, axis=0)
    noise_var = np.var(noise_np, axis=0)
    random_var = np.var(random_26d, axis=0)

    print(f"\nFRB feature variances (top 5):")
    frb_var_sorted = sorted(enumerate(frb_var), key=lambda x: x[1], reverse=True)
    for idx, var in frb_var_sorted[:5]:
        print(f"  Feature {idx}: {var:.4f}")

    print(f"\nNoise feature variances (top 5):")
    noise_var_sorted = sorted(enumerate(noise_var), key=lambda x: x[1], reverse=True)
    for idx, var in noise_var_sorted[:5]:
        print(f"  Feature {idx}: {var:.4f}")

    # Count "active" dimensions (variance > threshold)
    threshold = 0.01
    frb_active = sum(frb_var > threshold)
    noise_active = sum(noise_var > threshold)
    random_active = sum(random_var > threshold)

    print(f"\nActive dimensions (variance > {threshold}):")
    print(f"  FRB: {frb_active}/26")
    print(f"  Noise (pipeline): {noise_active}/26")
    print(f"  Random 26D: {random_active}/26")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    frb_id = frb_id_result.intrinsic_dimension
    noise_id = noise_id_result.intrinsic_dimension
    random_id = random_id_result.intrinsic_dimension

    print(f"\nIntrinsic Dimensions:")
    print(f"  FRB features: {frb_id:.2f}D")
    print(f"  Noise (pipeline): {noise_id:.2f}D")
    print(f"  Raw random 26D: {random_id:.2f}D")

    if abs(frb_id - noise_id) < 2.0:
        print("\n** FRB ID ≈ Noise ID: The low ID is from feature extraction, not FRBs **")
    elif frb_id < noise_id - 2.0:
        print("\n** FRB ID << Noise ID: FRBs have genuinely lower intrinsic dimension! **")
    else:
        print("\n** FRB ID >> Noise ID: Unexpected - FRBs have higher ID than noise **")

    if random_id > noise_id + 5.0:
        print("\n** Pipeline causes dimension collapse regardless of input **")

    results = {
        "experiment": "exp10_id_validation",
        "timestamp": datetime.now().isoformat(),
        "n_samples": n_frbs,
        "feature_dimension": 26,
        "intrinsic_dimensions": {
            "frb": {
                "id": float(frb_id_result.intrinsic_dimension),
                "ci_lower": float(frb_ci_lower) if frb_ci_lower else None,
                "ci_upper": float(frb_ci_upper) if frb_ci_upper else None,
            },
            "noise_pipeline": {
                "id": float(noise_id_result.intrinsic_dimension),
                "ci_lower": float(noise_ci_lower) if noise_ci_lower else None,
                "ci_upper": float(noise_ci_upper) if noise_ci_upper else None,
            },
            "random_26d": {
                "id": float(random_id_result.intrinsic_dimension),
                "ci_lower": float(random_ci_lower) if random_ci_lower else None,
                "ci_upper": float(random_ci_upper) if random_ci_upper else None,
            },
        },
        "active_dimensions": {
            "frb": int(frb_active),
            "noise_pipeline": int(noise_active),
            "random_26d": int(random_active),
        },
    }

    output_path = results_dir / "exp10_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
