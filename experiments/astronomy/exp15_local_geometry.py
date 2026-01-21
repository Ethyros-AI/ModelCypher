#!/usr/bin/env python3
"""Experiment 15: Local Geometry Analysis.

The "semantic highway" pattern in neural networks:
- Entry ramp: high local ID (expanding into manifold)
- Highway: low local ID (compressed information flow)
- Exit ramp: high local ID (expanding to output)

Do FRBs have similar local structure?
- Are there "highways" - low-ID paths through FRB space?
- Are some FRBs on compressed paths, others on expanded regions?

This could reveal:
- Information bottlenecks in FRB emission
- Regions of shared structure vs unique signatures

Usage:
    poetry run python experiments/astronomy/exp15_local_geometry.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from modelcypher.backends import initialize_default_backend
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

from shared.data_loader import load_frb_batch
from shared.feature_extraction import batch_extract_features


def compute_local_intrinsic_dimension(features: np.ndarray, k: int, backend):
    """Compute local intrinsic dimension for each point using k-NN.

    Uses the TwoNN estimator on local neighborhoods.
    """
    n_samples = features.shape[0]
    local_ids = []

    # Compute pairwise distances
    dist_matrix = squareform(pdist(features, metric='euclidean'))

    id_estimator = IntrinsicDimension(backend)

    for i in range(n_samples):
        # Get k nearest neighbors (excluding self)
        distances = dist_matrix[i, :]
        distances[i] = np.inf  # Exclude self
        neighbor_idx = np.argsort(distances)[:k]

        # Get local neighborhood
        local_points = features[neighbor_idx]

        # Compute local ID
        if len(local_points) >= 5:  # Need minimum samples
            try:
                local_id_result = id_estimator.compute(backend.array(local_points), with_ci=False)
                local_ids.append(float(local_id_result.intrinsic_dimension))
            except Exception:
                local_ids.append(np.nan)
        else:
            local_ids.append(np.nan)

    return np.array(local_ids)


def compute_local_density(features: np.ndarray, k: int = 5):
    """Compute local density for each point using k-NN distances."""
    dist_matrix = squareform(pdist(features, metric='euclidean'))
    n_samples = features.shape[0]

    densities = []
    for i in range(n_samples):
        distances = dist_matrix[i, :].copy()
        distances[i] = np.inf
        k_nearest = np.sort(distances)[:k]
        avg_dist = np.mean(k_nearest)
        density = 1.0 / (avg_dist + 1e-10)
        densities.append(density)

    return np.array(densities)


def compute_local_curvature_proxy(features: np.ndarray, k: int = 10):
    """Compute a proxy for local curvature using PCA on neighborhoods.

    High curvature = eigenvalues more spread out (non-flat local region)
    Low curvature = eigenvalues concentrated (locally flat)
    """
    dist_matrix = squareform(pdist(features, metric='euclidean'))
    n_samples = features.shape[0]

    curvatures = []
    for i in range(n_samples):
        # Get k nearest neighbors
        distances = dist_matrix[i, :].copy()
        distances[i] = np.inf
        neighbor_idx = np.argsort(distances)[:k]

        # Get local neighborhood
        local_points = features[neighbor_idx]

        # Center the points
        centered = local_points - np.mean(local_points, axis=0)

        # Compute covariance
        cov = np.cov(centered.T)

        # Eigenvalues
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.maximum(eigenvalues, 0)  # Numerical stability
        eigenvalues = np.sort(eigenvalues)[::-1]

        # Curvature proxy: ratio of sum of small eigenvalues to largest
        # High ratio = more curved (energy spread across dimensions)
        # Low ratio = flat (energy in one direction)
        if eigenvalues[0] > 1e-10:
            curvature = np.sum(eigenvalues[1:]) / eigenvalues[0]
        else:
            curvature = 0.0

        curvatures.append(curvature)

    return np.array(curvatures)


def run_experiment():
    data_dir = Path(__file__).parent / "data" / "raw"
    results_dir = Path(__file__).parent / "results"

    initialize_default_backend()
    backend = get_default_backend()

    print("=" * 60)
    print("Experiment 15: Local Geometry Analysis")
    print("=" * 60)
    print("\nQuestion: Do FRBs have 'semantic highway' structure?")
    print("(Low-ID paths = compressed information flow)")

    # Load FRBs
    frb_files = sorted(data_dir.glob("FRB*_waterfall.h5"))
    waterfalls = load_frb_batch([str(f) for f in frb_files])
    n_frbs = len(waterfalls)
    print(f"\nLoaded {n_frbs} FRBs")

    # Get physical properties
    dms = np.array([w.metadata.dm for w in waterfalls])
    snrs = np.array([w.metadata.snr for w in waterfalls])
    names = [w.metadata.tns_name for w in waterfalls]

    # Extract features
    frb_features_arr = batch_extract_features(waterfalls, backend)
    frb_np = np.array(backend.tolist(frb_features_arr))

    print("\n" + "=" * 40)
    print("PART 1: LOCAL INTRINSIC DIMENSION")
    print("=" * 40)

    # Compute local ID with k=15 neighbors (about 1/3 of data)
    k_local = min(15, n_frbs // 3)
    print(f"\nComputing local ID with k={k_local} neighbors...")

    local_ids = compute_local_intrinsic_dimension(frb_np, k=k_local, backend=backend)

    # Filter out NaN values for statistics
    valid_mask = ~np.isnan(local_ids)
    valid_ids = local_ids[valid_mask]

    if len(valid_ids) > 0:
        print(f"\nLocal ID statistics:")
        print(f"  Mean: {np.mean(valid_ids):.2f}")
        print(f"  Std: {np.std(valid_ids):.2f}")
        print(f"  Range: [{np.min(valid_ids):.2f}, {np.max(valid_ids):.2f}]")

        # Find FRBs on "highways" (low local ID)
        low_id_threshold = np.percentile(valid_ids, 25)
        high_id_threshold = np.percentile(valid_ids, 75)

        low_id_mask = local_ids < low_id_threshold
        high_id_mask = local_ids > high_id_threshold

        n_low_id = np.sum(low_id_mask)
        n_high_id = np.sum(high_id_mask)

        print(f"\n'Highway' FRBs (low local ID < {low_id_threshold:.1f}): {n_low_id}")
        print(f"'Expanded' FRBs (high local ID > {high_id_threshold:.1f}): {n_high_id}")
    else:
        print("Warning: Could not compute local ID (not enough data)")
        low_id_mask = np.zeros(n_frbs, dtype=bool)
        high_id_mask = np.zeros(n_frbs, dtype=bool)

    print("\n" + "=" * 40)
    print("PART 2: LOCAL DENSITY")
    print("=" * 40)

    local_density = compute_local_density(frb_np, k=5)

    print(f"\nLocal density statistics:")
    print(f"  Mean: {np.mean(local_density):.4f}")
    print(f"  Std: {np.std(local_density):.4f}")
    print(f"  Range: [{np.min(local_density):.4f}, {np.max(local_density):.4f}]")

    # Dense vs sparse regions
    dense_threshold = np.percentile(local_density, 75)
    sparse_threshold = np.percentile(local_density, 25)

    dense_mask = local_density > dense_threshold
    sparse_mask = local_density < sparse_threshold

    print(f"\nDense regions (top 25%): {np.sum(dense_mask)} FRBs")
    print(f"Sparse regions (bottom 25%): {np.sum(sparse_mask)} FRBs")

    print("\n" + "=" * 40)
    print("PART 3: LOCAL CURVATURE")
    print("=" * 40)

    local_curvature = compute_local_curvature_proxy(frb_np, k=10)

    print(f"\nLocal curvature proxy statistics:")
    print(f"  Mean: {np.mean(local_curvature):.4f}")
    print(f"  Std: {np.std(local_curvature):.4f}")
    print(f"  Range: [{np.min(local_curvature):.4f}, {np.max(local_curvature):.4f}]")

    # Flat vs curved regions
    flat_threshold = np.percentile(local_curvature, 25)
    curved_threshold = np.percentile(local_curvature, 75)

    flat_mask = local_curvature < flat_threshold
    curved_mask = local_curvature > curved_threshold

    print(f"\nLocally flat regions: {np.sum(flat_mask)} FRBs")
    print(f"Locally curved regions: {np.sum(curved_mask)} FRBs")

    print("\n" + "=" * 40)
    print("PART 4: PHYSICS vs LOCAL GEOMETRY")
    print("=" * 40)

    # Do low-ID (highway) FRBs have different physics?
    if len(valid_ids) > 0:
        from scipy import stats

        # Local ID vs DM
        valid_dms = dms[valid_mask]
        r_id_dm, p_id_dm = stats.pearsonr(valid_ids, valid_dms)
        print(f"\nLocal ID vs DM: r={r_id_dm:.3f} (p={p_id_dm:.3f})")

        # Local ID vs SNR
        valid_snrs = snrs[valid_mask]
        r_id_snr, p_id_snr = stats.pearsonr(valid_ids, valid_snrs)
        print(f"Local ID vs SNR: r={r_id_snr:.3f} (p={p_id_snr:.3f})")

    # Local density vs physics
    r_dens_dm, p_dens_dm = stats.pearsonr(local_density, dms)
    r_dens_snr, p_dens_snr = stats.pearsonr(local_density, snrs)
    print(f"\nLocal density vs DM: r={r_dens_dm:.3f} (p={p_dens_dm:.3f})")
    print(f"Local density vs SNR: r={r_dens_snr:.3f} (p={p_dens_snr:.3f})")

    # Local curvature vs physics
    r_curv_dm, p_curv_dm = stats.pearsonr(local_curvature, dms)
    r_curv_snr, p_curv_snr = stats.pearsonr(local_curvature, snrs)
    print(f"\nLocal curvature vs DM: r={r_curv_dm:.3f} (p={p_curv_dm:.3f})")
    print(f"Local curvature vs SNR: r={r_curv_snr:.3f} (p={p_curv_snr:.3f})")

    print("\n" + "=" * 40)
    print("PART 5: HIGHWAY STRUCTURE")
    print("=" * 40)

    # Combination: Low ID + Flat + Dense = information highway
    if len(valid_ids) > 0:
        highway_mask = low_id_mask & flat_mask & dense_mask
        n_highway = np.sum(highway_mask)
        print(f"\n'Information highway' FRBs (low ID + flat + dense): {n_highway}")

        if n_highway > 0:
            highway_names = [names[i] for i in range(n_frbs) if highway_mask[i]]
            highway_dms = dms[highway_mask]
            highway_snrs = snrs[highway_mask]
            print(f"  FRBs: {highway_names}")
            print(f"  DM range: [{np.min(highway_dms):.0f}, {np.max(highway_dms):.0f}]")
            print(f"  SNR range: [{np.min(highway_snrs):.1f}, {np.max(highway_snrs):.1f}]")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if len(valid_ids) > 0 and np.std(valid_ids) > 0.5:
        print("\n** LOCAL ID VARIES across FRB space **")
        print("→ Some regions are more 'compressed' than others")
        print("→ This suggests non-uniform information structure")
    else:
        print("\n** LOCAL ID is relatively UNIFORM **")
        print("→ FRB space has consistent local dimensionality")

    if np.std(local_density) / np.mean(local_density) > 0.5:
        print("\n** DENSITY VARIES significantly **")
        print("→ FRBs cluster in some regions, sparse in others")
    else:
        print("\n** DENSITY is relatively UNIFORM **")

    results = {
        "experiment": "exp15_local_geometry",
        "timestamp": datetime.now().isoformat(),
        "n_samples": n_frbs,
        "k_local_id": k_local,
        "local_intrinsic_dimension": {
            "values": local_ids.tolist(),
            "mean": float(np.nanmean(local_ids)),
            "std": float(np.nanstd(local_ids)),
            "n_valid": int(np.sum(~np.isnan(local_ids))),
        },
        "local_density": {
            "values": local_density.tolist(),
            "mean": float(np.mean(local_density)),
            "std": float(np.std(local_density)),
        },
        "local_curvature": {
            "values": local_curvature.tolist(),
            "mean": float(np.mean(local_curvature)),
            "std": float(np.std(local_curvature)),
        },
        "correlations": {
            "local_id_dm": {"r": float(r_id_dm) if len(valid_ids) > 0 else None,
                           "p": float(p_id_dm) if len(valid_ids) > 0 else None},
            "local_id_snr": {"r": float(r_id_snr) if len(valid_ids) > 0 else None,
                            "p": float(p_id_snr) if len(valid_ids) > 0 else None},
            "density_dm": {"r": float(r_dens_dm), "p": float(p_dens_dm)},
            "density_snr": {"r": float(r_dens_snr), "p": float(p_dens_snr)},
            "curvature_dm": {"r": float(r_curv_dm), "p": float(p_curv_dm)},
            "curvature_snr": {"r": float(r_curv_snr), "p": float(p_curv_snr)},
        },
        "frb_names": names,
    }

    output_path = results_dir / "exp15_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
