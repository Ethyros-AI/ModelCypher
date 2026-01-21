#!/usr/bin/env python3
"""Experiment 11: Internal Structure of FRBs.

Stop asking "do FRBs speak English?"
Start asking "what language do FRBs speak?"

If FRBs have structure (they repeat!), they encode information.
What IS that information?

Analysis:
1. Clustering - what natural groupings exist in FRB feature space?
2. Manifold structure - what is the geometry of the FRB space?
3. Feature decomposition - which features carry the signal?
4. Physical interpretation - what do the structures mean?

Usage:
    poetry run python experiments/astronomy/exp11_internal_structure.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from modelcypher.backends import initialize_default_backend
from modelcypher.core.domain._backend import get_default_backend

from shared.data_loader import load_frb_batch
from shared.feature_extraction import batch_extract_features


def compute_pca(data: np.ndarray, n_components: int = None):
    """Simple PCA implementation."""
    # Center the data
    mean = np.mean(data, axis=0)
    centered = data - mean

    # Compute covariance matrix
    cov = np.cov(centered.T)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Project data
    if n_components:
        eigenvectors = eigenvectors[:, :n_components]
        eigenvalues = eigenvalues[:n_components]

    projected = centered @ eigenvectors

    # Compute explained variance
    total_var = np.sum(eigenvalues)
    explained_var = eigenvalues / total_var if total_var > 0 else eigenvalues

    return {
        "projected": projected,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "explained_variance_ratio": explained_var,
        "mean": mean,
    }


def analyze_clusters(features: np.ndarray, physical_props: dict, n_clusters: int = 3):
    """Perform hierarchical clustering and analyze physical meaning."""
    # Compute linkage
    Z = linkage(features, method='ward')

    # Cut into n clusters
    labels = fcluster(Z, n_clusters, criterion='maxclust')

    # Analyze each cluster
    cluster_analysis = []
    for i in range(1, n_clusters + 1):
        mask = labels == i
        n_members = np.sum(mask)

        cluster_info = {
            "cluster_id": i,
            "n_members": int(n_members),
            "member_indices": np.where(mask)[0].tolist(),
        }

        # Physical property statistics for this cluster
        for prop_name, prop_values in physical_props.items():
            prop_array = np.array(prop_values)
            cluster_vals = prop_array[mask]
            if len(cluster_vals) > 0:
                cluster_info[f"{prop_name}_mean"] = float(np.mean(cluster_vals))
                cluster_info[f"{prop_name}_std"] = float(np.std(cluster_vals))
                cluster_info[f"{prop_name}_min"] = float(np.min(cluster_vals))
                cluster_info[f"{prop_name}_max"] = float(np.max(cluster_vals))

        cluster_analysis.append(cluster_info)

    return {
        "labels": labels.tolist(),
        "linkage": Z.tolist(),
        "cluster_analysis": cluster_analysis,
    }


def find_feature_importance(features: np.ndarray, physical_props: dict):
    """Find which features best predict physical properties."""
    n_features = features.shape[1]
    feature_names = [
        "band_0_mean", "band_0_std", "band_1_mean", "band_1_std",
        "band_2_mean", "band_2_std", "band_3_mean", "band_3_std",
        "band_4_mean", "band_4_std", "band_5_mean", "band_5_std",
        "band_6_mean", "band_6_std", "band_7_mean", "band_7_std",
        "ts_mean", "ts_std", "ts_max", "ts_peak_loc",
        "spec_entropy", "dm", "width", "fluence", "total_intensity", "snr_proxy"
    ]

    importance = {}
    for prop_name, prop_values in physical_props.items():
        prop_array = np.array(prop_values)
        correlations = []

        for i in range(min(n_features, len(feature_names))):
            r, p = stats.pearsonr(features[:, i], prop_array)
            correlations.append({
                "feature": feature_names[i] if i < len(feature_names) else f"feature_{i}",
                "correlation": float(r) if not np.isnan(r) else 0.0,
                "p_value": float(p) if not np.isnan(p) else 1.0,
            })

        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        importance[prop_name] = correlations

    return importance


def compute_manifold_properties(features: np.ndarray):
    """Analyze the manifold structure of FRB feature space."""
    # Pairwise distances
    distances = pdist(features, metric='euclidean')
    dist_matrix = squareform(distances)

    # Distance statistics
    upper_tri = distances  # pdist returns upper triangle

    # Nearest neighbor distances (k=1)
    nn_distances = []
    for i in range(len(features)):
        row = dist_matrix[i, :]
        row[i] = np.inf  # Exclude self
        nn_distances.append(np.min(row))

    # Second nearest neighbor (k=2)
    nn2_distances = []
    for i in range(len(features)):
        row = dist_matrix[i, :].copy()
        row[i] = np.inf
        sorted_row = np.sort(row)
        nn2_distances.append(sorted_row[1])  # Second smallest

    # Compute local density (inverse of avg k-NN distance)
    k = 5
    local_densities = []
    for i in range(len(features)):
        row = dist_matrix[i, :].copy()
        row[i] = np.inf
        k_nearest = np.sort(row)[:k]
        avg_dist = np.mean(k_nearest)
        local_densities.append(1.0 / (avg_dist + 1e-10))

    return {
        "distance_stats": {
            "mean": float(np.mean(upper_tri)),
            "std": float(np.std(upper_tri)),
            "min": float(np.min(upper_tri)),
            "max": float(np.max(upper_tri)),
        },
        "nearest_neighbor": {
            "mean": float(np.mean(nn_distances)),
            "std": float(np.std(nn_distances)),
        },
        "second_nearest": {
            "mean": float(np.mean(nn2_distances)),
            "std": float(np.std(nn2_distances)),
        },
        "mu_ratio": float(np.mean(nn2_distances) / np.mean(nn_distances)),
        "local_density_variance": float(np.var(local_densities)),
        "density_range": float(np.max(local_densities) / (np.min(local_densities) + 1e-10)),
    }


def run_experiment():
    data_dir = Path(__file__).parent / "data" / "raw"
    results_dir = Path(__file__).parent / "results"

    initialize_default_backend()
    backend = get_default_backend()

    print("=" * 60)
    print("Experiment 11: Internal Structure of FRBs")
    print("=" * 60)
    print("\nQuestion: What language do FRBs speak?")
    print("(Not: do they speak English?)")

    # Load FRBs
    frb_files = sorted(data_dir.glob("FRB*_waterfall.h5"))
    waterfalls = load_frb_batch([str(f) for f in frb_files])
    n_frbs = len(waterfalls)
    print(f"\nLoaded {n_frbs} FRBs")

    # Get physical properties
    physical_props = {
        "dm": [w.metadata.dm for w in waterfalls],
        "snr": [w.metadata.snr for w in waterfalls],
    }

    # Also get names for later
    names = [w.metadata.tns_name for w in waterfalls]

    # Extract features
    frb_features_arr = batch_extract_features(waterfalls, backend)
    frb_np = np.array(backend.tolist(frb_features_arr))

    print(f"Feature matrix: {frb_np.shape}")

    print("\n" + "=" * 40)
    print("PART 1: PRINCIPAL COMPONENT ANALYSIS")
    print("=" * 40)

    pca_result = compute_pca(frb_np)
    cumulative_var = np.cumsum(pca_result["explained_variance_ratio"])

    print("\nExplained variance by PC:")
    for i in range(min(10, len(pca_result["explained_variance_ratio"]))):
        print(f"  PC{i+1}: {pca_result['explained_variance_ratio'][i]*100:.1f}% "
              f"(cumulative: {cumulative_var[i]*100:.1f}%)")

    # How many PCs for 90% variance?
    n_90 = np.argmax(cumulative_var >= 0.90) + 1
    n_95 = np.argmax(cumulative_var >= 0.95) + 1
    print(f"\nPCs needed for 90% variance: {n_90}")
    print(f"PCs needed for 95% variance: {n_95}")

    print("\n" + "=" * 40)
    print("PART 2: CLUSTERING ANALYSIS")
    print("=" * 40)

    for n_clusters in [2, 3, 4]:
        print(f"\n--- {n_clusters} Clusters ---")
        cluster_result = analyze_clusters(frb_np, physical_props, n_clusters)

        for cluster in cluster_result["cluster_analysis"]:
            cid = cluster["cluster_id"]
            n = cluster["n_members"]
            dm_mean = cluster.get("dm_mean", 0)
            dm_std = cluster.get("dm_std", 0)
            snr_mean = cluster.get("snr_mean", 0)
            snr_std = cluster.get("snr_std", 0)
            print(f"  Cluster {cid} (n={n}): DM={dm_mean:.0f}±{dm_std:.0f}, SNR={snr_mean:.1f}±{snr_std:.1f}")

    # Use 3 clusters for detailed analysis
    cluster_result = analyze_clusters(frb_np, physical_props, n_clusters=3)

    print("\n" + "=" * 40)
    print("PART 3: FEATURE IMPORTANCE")
    print("=" * 40)

    importance = find_feature_importance(frb_np, physical_props)

    print("\nFeatures most predictive of DM:")
    for item in importance["dm"][:5]:
        sig = "*" if item["p_value"] < 0.05 else ""
        print(f"  {item['feature']}: r={item['correlation']:.3f} {sig}")

    print("\nFeatures most predictive of SNR:")
    for item in importance["snr"][:5]:
        sig = "*" if item["p_value"] < 0.05 else ""
        print(f"  {item['feature']}: r={item['correlation']:.3f} {sig}")

    print("\n" + "=" * 40)
    print("PART 4: MANIFOLD GEOMETRY")
    print("=" * 40)

    manifold = compute_manifold_properties(frb_np)

    print(f"\nPairwise distances: {manifold['distance_stats']['mean']:.3f} ± {manifold['distance_stats']['std']:.3f}")
    print(f"Nearest neighbor: {manifold['nearest_neighbor']['mean']:.3f} ± {manifold['nearest_neighbor']['std']:.3f}")
    print(f"μ ratio (NN2/NN1): {manifold['mu_ratio']:.3f}")
    print(f"Local density variance: {manifold['local_density_variance']:.4f}")
    print(f"Density range (max/min): {manifold['density_range']:.1f}x")

    # μ ratio interpretation
    # For uniform distribution in D dimensions: μ ≈ 2^(1/D)
    # Solve: 2^(1/D) = μ → D = log(2)/log(μ)
    estimated_dim = np.log(2) / np.log(manifold['mu_ratio']) if manifold['mu_ratio'] > 1 else float('inf')
    print(f"\nEstimated dimension from μ-ratio: {estimated_dim:.1f}D")

    print("\n" + "=" * 60)
    print("INTERPRETATION: THE FRB VOCABULARY")
    print("=" * 60)

    print("\nThe FRB feature space has structure:")
    print(f"  - {n_90} principal components capture 90% of variance")
    print(f"  - Clusters separate by physical properties (DM, SNR)")
    print(f"  - Local density varies {manifold['density_range']:.1f}x across the space")

    # What do the clusters represent?
    print("\nCluster interpretation:")
    for cluster in cluster_result["cluster_analysis"]:
        cid = cluster["cluster_id"]
        dm_mean = cluster.get("dm_mean", 0)
        snr_mean = cluster.get("snr_mean", 0)

        if dm_mean < 300:
            dm_label = "NEARBY"
        elif dm_mean < 700:
            dm_label = "INTERMEDIATE"
        else:
            dm_label = "DISTANT"

        if snr_mean < 15:
            snr_label = "FAINT"
        elif snr_mean < 30:
            snr_label = "MODERATE"
        else:
            snr_label = "BRIGHT"

        print(f"  Cluster {cid}: {dm_label} + {snr_label} FRBs")

    results = {
        "experiment": "exp11_internal_structure",
        "timestamp": datetime.now().isoformat(),
        "n_samples": n_frbs,
        "pca": {
            "explained_variance": pca_result["explained_variance_ratio"].tolist(),
            "cumulative_variance": cumulative_var.tolist(),
            "n_components_90pct": int(n_90),
            "n_components_95pct": int(n_95),
        },
        "clustering": cluster_result,
        "feature_importance": importance,
        "manifold": manifold,
        "estimated_dimension_from_mu": float(estimated_dim),
        "frb_names": names,
    }

    output_path = results_dir / "exp11_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
