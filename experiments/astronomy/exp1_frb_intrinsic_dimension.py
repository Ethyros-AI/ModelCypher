#!/usr/bin/env python3
"""Experiment 1: Measure intrinsic dimension of FRB feature space.

Tests whether FRB features compress to a low-dimensional manifold
(like semantic representations, ~5-15D) or remain high-dimensional (noise).

Hypothesis:
    - If FRBs encode information, ID should be 5-15D
    - If FRBs are natural noise, ID ≈ feature dimension (26D)

Usage:
    poetry run python experiments/astronomy/exp1_frb_intrinsic_dimension.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from modelcypher.core.domain._backend import get_default_backend, initialize_default_backend
from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
from modelcypher.core.domain.geometry.effective_rank import EffectiveRank

from shared.data_loader import load_frb_batch, get_frb_metadata
from shared.feature_extraction import batch_extract_features, get_feature_dimension


def find_frb_files(data_dir: Path, limit: int | None = None) -> list[Path]:
    """Find all FRB waterfall files in data directory."""
    files = sorted(data_dir.glob("FRB*_waterfall.h5"))
    if limit:
        files = files[:limit]
    return files


def run_experiment() -> dict:
    """Run intrinsic dimension experiment on FRB features."""
    # Setup
    data_dir = Path(__file__).parent / "data" / "raw"
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Initialize backend
    initialize_default_backend()
    backend = get_default_backend()

    print("=" * 60)
    print("Experiment 1: FRB Intrinsic Dimension Profile")
    print("=" * 60)
    print()

    # Find FRB files
    frb_files = find_frb_files(data_dir)

    if not frb_files:
        print(f"ERROR: No FRB files found in {data_dir}")
        print("Run the download script first:")
        print("  poetry run python experiments/astronomy/data/download_chime_frb.py --limit 20")
        return {"error": "No data files found"}

    print(f"Found {len(frb_files)} FRB files")
    print()

    # Load FRBs
    print("Loading FRB waterfalls...")
    waterfalls = load_frb_batch([str(f) for f in frb_files])
    print(f"  Successfully loaded: {len(waterfalls)} FRBs")

    if len(waterfalls) < 3:
        print("ERROR: Need at least 3 FRBs for intrinsic dimension estimation")
        return {"error": "Insufficient data"}

    # Extract features
    print()
    print("Extracting features...")
    features = batch_extract_features(waterfalls, backend)
    feature_dim = get_feature_dimension()
    print(f"  Feature dimension: {feature_dim}")
    print(f"  Feature matrix shape: [{len(waterfalls)}, {feature_dim}]")

    # Compute intrinsic dimension
    print()
    print("Computing intrinsic dimension (TwoNN)...")
    id_estimator = IntrinsicDimension(backend)

    try:
        id_result = id_estimator.compute(features, with_ci=True)
        intrinsic_dim = id_result.intrinsic_dimension
        ci_lower = id_result.ci.lower if id_result.ci else None
        ci_upper = id_result.ci.upper if id_result.ci else None

        print(f"  Intrinsic dimension: {intrinsic_dim:.2f}")
        if ci_lower and ci_upper:
            print(f"  95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
        print(f"  Usable samples: {id_result.usable_count}/{id_result.sample_count}")

    except Exception as e:
        print(f"  ERROR computing ID: {e}")
        intrinsic_dim = None
        ci_lower = None
        ci_upper = None

    # Compute effective rank (spectral)
    print()
    print("Computing effective rank (spectral)...")
    rank_estimator = EffectiveRank(backend)

    try:
        # Center the features
        features_centered = features - backend.mean(features, axis=0)
        rank_result = rank_estimator.compute(features_centered)

        print(f"  Shannon effective rank: {rank_result.shannon_effective_rank:.2f}")
        print(f"  Renyi effective rank: {rank_result.renyi_effective_rank:.2f}")
        print(f"  Spectral entropy: {rank_result.spectral_entropy:.4f}")
        print(f"  Non-zero singular values: {rank_result.n_singular_values}")

    except Exception as e:
        print(f"  ERROR computing rank: {e}")
        rank_result = None

    # Compute local dimension map if enough samples
    print()
    print("Computing local dimension map...")
    try:
        if len(waterfalls) >= 10:
            local_map = id_estimator.local_dimension_map(features)
            modal_dim = local_map.modal_dimension
            mean_dim = local_map.mean_dimension
            std_dim = local_map.std_dimension
            n_deficient = len(local_map.deficient_indices)

            print(f"  Modal dimension: {modal_dim:.2f}")
            print(f"  Mean dimension: {mean_dim:.2f}")
            print(f"  Std dimension: {std_dim:.2f}")
            print(f"  Deficient points: {n_deficient}")
        else:
            print("  Skipped (need >= 10 samples for local map)")
            modal_dim = None
            mean_dim = None
            std_dim = None
            n_deficient = None

    except Exception as e:
        print(f"  ERROR computing local map: {e}")
        modal_dim = None
        mean_dim = None
        std_dim = None
        n_deficient = None

    # Interpretation
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    if intrinsic_dim is not None:
        compression_ratio = feature_dim / intrinsic_dim if intrinsic_dim > 0 else float("inf")
        print(f"Feature dimension: {feature_dim}")
        print(f"Intrinsic dimension: {intrinsic_dim:.2f}")
        print(f"Compression ratio: {compression_ratio:.1f}x")
        print()

        # Compare to semantic manifold expectation
        if intrinsic_dim < 15:
            print("OBSERVATION: ID < 15 suggests low-dimensional structure")
            print("  This is consistent with information-encoding systems")
        else:
            print("OBSERVATION: ID >= 15 suggests high-dimensional structure")
            print("  This is more consistent with noise or uncorrelated features")

    # Build results dict
    results = {
        "experiment": "exp1_frb_intrinsic_dimension",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_frbs": len(waterfalls),
            "feature_dimension": feature_dim,
            "frb_files": [str(f) for f in frb_files],
        },
        "metrics": {
            "intrinsic_dimension": intrinsic_dim,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "modal_dimension": modal_dim,
            "mean_dimension": mean_dim,
            "std_dimension": std_dim,
            "n_deficient_points": n_deficient,
            "shannon_effective_rank": (
                rank_result.shannon_effective_rank if rank_result else None
            ),
            "renyi_effective_rank": (
                rank_result.renyi_effective_rank if rank_result else None
            ),
            "spectral_entropy": (
                rank_result.spectral_entropy if rank_result else None
            ),
            "compression_ratio": (
                feature_dim / intrinsic_dim if intrinsic_dim and intrinsic_dim > 0 else None
            ),
        },
        "frb_metadata": [
            {
                "tns_name": w.metadata.tns_name,
                "dm": w.metadata.dm,
                "snr": w.metadata.snr,
            }
            for w in waterfalls
        ],
    }

    # Save results
    output_path = results_dir / "exp1_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print()
    print(f"Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
