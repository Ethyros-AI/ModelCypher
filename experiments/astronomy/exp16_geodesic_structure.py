#!/usr/bin/env python3
"""Experiment 16: Geodesic Structure Analysis.

Previous experiments used Euclidean distance.
But if FRBs lie on a curved manifold, geodesic distance is the right metric.

Questions:
1. Does geodesic clustering differ from Euclidean clustering?
2. Are there geodesic "paths" connecting FRBs of similar physics?
3. Does the k-NN graph reveal hidden structure?

Usage:
    poetry run python experiments/astronomy/exp16_geodesic_structure.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import shortest_path, connected_components
from scipy.cluster.hierarchy import linkage, fcluster

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from modelcypher.backends import initialize_default_backend
from modelcypher.core.domain._backend import get_default_backend

from shared.data_loader import load_frb_batch
from shared.feature_extraction import batch_extract_features


def build_knn_graph(features: np.ndarray, k: int):
    """Build k-NN graph and return adjacency matrix."""
    n = features.shape[0]
    dist_matrix = squareform(pdist(features, metric='euclidean'))

    # Build k-NN adjacency
    adjacency = np.zeros((n, n))

    for i in range(n):
        distances = dist_matrix[i, :].copy()
        distances[i] = np.inf  # Exclude self

        # Find k nearest neighbors
        k_nearest = np.argsort(distances)[:k]

        # Symmetric k-NN (edge if either is in other's k-NN)
        for j in k_nearest:
            adjacency[i, j] = dist_matrix[i, j]
            adjacency[j, i] = dist_matrix[j, i]

    return adjacency, dist_matrix


def compute_geodesic_distances(adjacency: np.ndarray):
    """Compute geodesic distances using shortest paths on k-NN graph."""
    # Replace zeros with infinity for shortest path computation
    graph = adjacency.copy()
    graph[graph == 0] = np.inf
    np.fill_diagonal(graph, 0)

    # Compute shortest paths
    geodesic = shortest_path(graph, method='D', directed=False)

    return geodesic


def analyze_graph_structure(adjacency: np.ndarray):
    """Analyze the structure of the k-NN graph."""
    # Convert to binary adjacency for component analysis
    binary_adj = (adjacency > 0).astype(int)

    # Connected components
    n_components, labels = connected_components(binary_adj, directed=False)

    # Degree distribution
    degrees = np.sum(binary_adj, axis=1)

    return {
        "n_connected_components": int(n_components),
        "component_labels": labels.tolist(),
        "degree_mean": float(np.mean(degrees)),
        "degree_std": float(np.std(degrees)),
        "degree_min": int(np.min(degrees)),
        "degree_max": int(np.max(degrees)),
    }


def run_experiment():
    data_dir = Path(__file__).parent / "data" / "raw"
    results_dir = Path(__file__).parent / "results"

    initialize_default_backend()
    backend = get_default_backend()

    print("=" * 60)
    print("Experiment 16: Geodesic Structure Analysis")
    print("=" * 60)
    print("\nQuestion: Does geodesic distance reveal structure that")
    print("Euclidean distance misses?")

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
    print("PART 1: k-NN GRAPH CONSTRUCTION")
    print("=" * 40)

    # Build k-NN graph with different k values
    k_values = [3, 5, 10]
    graph_results = {}

    for k in k_values:
        adjacency, euclidean_dist = build_knn_graph(frb_np, k=k)
        graph_info = analyze_graph_structure(adjacency)

        print(f"\nk={k} graph:")
        print(f"  Connected components: {graph_info['n_connected_components']}")
        print(f"  Average degree: {graph_info['degree_mean']:.1f} ± {graph_info['degree_std']:.1f}")

        graph_results[k] = graph_info

    # Use k=5 for main analysis
    k_main = 5
    adjacency, euclidean_dist = build_knn_graph(frb_np, k=k_main)

    print("\n" + "=" * 40)
    print("PART 2: GEODESIC vs EUCLIDEAN DISTANCES")
    print("=" * 40)

    geodesic_dist = compute_geodesic_distances(adjacency)

    # Compare Euclidean and geodesic distances
    # Get upper triangle (excluding diagonal)
    upper_tri = np.triu_indices(n_frbs, k=1)
    euclidean_flat = euclidean_dist[upper_tri]
    geodesic_flat = geodesic_dist[upper_tri]

    # Filter out infinite geodesic distances (disconnected pairs)
    finite_mask = np.isfinite(geodesic_flat)
    euclidean_finite = euclidean_flat[finite_mask]
    geodesic_finite = geodesic_flat[finite_mask]

    print(f"\nPairwise comparisons: {len(euclidean_flat)}")
    print(f"Finite geodesic pairs: {len(euclidean_finite)} ({100*len(euclidean_finite)/len(euclidean_flat):.0f}%)")

    if len(geodesic_finite) > 0:
        # Correlation between Euclidean and geodesic
        r_eg, p_eg = stats.pearsonr(euclidean_finite, geodesic_finite)
        print(f"\nCorrelation (Euclidean vs Geodesic): r={r_eg:.3f} (p={p_eg:.2e})")

        # Ratio statistics
        ratios = geodesic_finite / (euclidean_finite + 1e-10)
        print(f"\nGeodesic/Euclidean ratio:")
        print(f"  Mean: {np.mean(ratios):.2f}")
        print(f"  Std: {np.std(ratios):.2f}")
        print(f"  Range: [{np.min(ratios):.2f}, {np.max(ratios):.2f}]")

        # Pairs where geodesic >> Euclidean (path through manifold much longer)
        detour_mask = ratios > 2.0
        n_detours = np.sum(detour_mask)
        print(f"\nPairs with geodesic > 2x Euclidean: {n_detours} ({100*n_detours/len(ratios):.1f}%)")

    print("\n" + "=" * 40)
    print("PART 3: CLUSTERING COMPARISON")
    print("=" * 40)

    # Euclidean clustering
    Z_euclidean = linkage(frb_np, method='ward')
    labels_euclidean_3 = fcluster(Z_euclidean, 3, criterion='maxclust')

    # Geodesic clustering (using finite pairs only)
    # Replace infinite with large value for linkage
    geodesic_for_cluster = geodesic_dist.copy()
    geodesic_for_cluster[~np.isfinite(geodesic_for_cluster)] = 1e10
    geodesic_condensed = geodesic_for_cluster[upper_tri]

    Z_geodesic = linkage(geodesic_condensed, method='average')  # Use average for precomputed distances
    labels_geodesic_3 = fcluster(Z_geodesic, 3, criterion='maxclust')

    # Compare clusterings
    from scipy.stats import chi2_contingency

    # Create contingency table
    contingency = np.zeros((3, 3), dtype=int)
    for i in range(n_frbs):
        contingency[labels_euclidean_3[i] - 1, labels_geodesic_3[i] - 1] += 1

    print("\nContingency table (Euclidean vs Geodesic clusters):")
    print(contingency)

    # Chi-square test of independence
    chi2, p_chi2, dof, expected = chi2_contingency(contingency)
    print(f"\nChi-square test: χ²={chi2:.2f}, p={p_chi2:.3f}")

    if p_chi2 < 0.05:
        print("→ Euclidean and geodesic clusterings are DIFFERENT")
    else:
        print("→ Euclidean and geodesic clusterings are SIMILAR")

    # Analyze what differs
    same_cluster = labels_euclidean_3 == labels_geodesic_3
    n_same = np.sum(same_cluster)
    print(f"\nFRBs in same cluster (both methods): {n_same}/{n_frbs} ({100*n_same/n_frbs:.0f}%)")

    print("\n" + "=" * 40)
    print("PART 4: PHYSICS vs GEODESIC STRUCTURE")
    print("=" * 40)

    # Does physical similarity correlate with geodesic proximity?
    dm_diffs_flat = np.abs(np.subtract.outer(dms, dms)[upper_tri])
    snr_diffs_flat = np.abs(np.subtract.outer(snrs, snrs)[upper_tri])

    # Euclidean vs physical
    r_euc_dm, p_euc_dm = stats.pearsonr(euclidean_flat, dm_diffs_flat)
    r_euc_snr, p_euc_snr = stats.pearsonr(euclidean_flat, snr_diffs_flat)

    print(f"\nEuclidean distance vs physical differences:")
    print(f"  vs DM difference: r={r_euc_dm:.3f} (p={p_euc_dm:.2e})")
    print(f"  vs SNR difference: r={r_euc_snr:.3f} (p={p_euc_snr:.2e})")

    if len(geodesic_finite) > 0:
        # Geodesic vs physical (finite pairs only)
        dm_finite = dm_diffs_flat[finite_mask]
        snr_finite = snr_diffs_flat[finite_mask]

        r_geo_dm, p_geo_dm = stats.pearsonr(geodesic_finite, dm_finite)
        r_geo_snr, p_geo_snr = stats.pearsonr(geodesic_finite, snr_finite)

        print(f"\nGeodesic distance vs physical differences:")
        print(f"  vs DM difference: r={r_geo_dm:.3f} (p={p_geo_dm:.2e})")
        print(f"  vs SNR difference: r={r_geo_snr:.3f} (p={p_geo_snr:.2e})")

        # Does geodesic capture physics BETTER than Euclidean?
        print("\n" + "=" * 40)
        print("PART 5: GEODESIC IMPROVEMENT")
        print("=" * 40)

        dm_improvement = abs(r_geo_dm) - abs(r_euc_dm)
        snr_improvement = abs(r_geo_snr) - abs(r_euc_snr)

        print(f"\nImprovement in physical correlation:")
        print(f"  DM: {dm_improvement:+.3f} ({'geodesic better' if dm_improvement > 0 else 'euclidean better'})")
        print(f"  SNR: {snr_improvement:+.3f} ({'geodesic better' if snr_improvement > 0 else 'euclidean better'})")

    print("\n" + "=" * 40)
    print("PART 6: GEODESIC NEIGHBORS")
    print("=" * 40)

    # For each FRB, find geodesic nearest neighbor
    geodesic_nn = []
    for i in range(n_frbs):
        row = geodesic_dist[i, :].copy()
        row[i] = np.inf
        if np.any(np.isfinite(row)):
            nn_idx = np.argmin(row)
            nn_dist = row[nn_idx]
            geodesic_nn.append({
                "frb": names[i],
                "dm": float(dms[i]),
                "snr": float(snrs[i]),
                "geodesic_nn": names[nn_idx],
                "geodesic_nn_dm": float(dms[nn_idx]),
                "geodesic_nn_snr": float(snrs[nn_idx]),
                "geodesic_distance": float(nn_dist),
            })

    # Are geodesic neighbors physically similar?
    dm_diffs_nn = [abs(g["dm"] - g["geodesic_nn_dm"]) for g in geodesic_nn]
    snr_diffs_nn = [abs(g["snr"] - g["geodesic_nn_snr"]) for g in geodesic_nn]

    print(f"\nGeodesic nearest neighbor analysis:")
    print(f"  Mean DM difference: {np.mean(dm_diffs_nn):.0f}")
    print(f"  Mean SNR difference: {np.mean(snr_diffs_nn):.1f}")

    # Compare to random pairs
    random_dm_diffs = np.random.choice(dm_diffs_flat, size=len(dm_diffs_nn), replace=False)
    random_snr_diffs = np.random.choice(snr_diffs_flat, size=len(snr_diffs_nn), replace=False)

    print(f"\nRandom pair differences:")
    print(f"  Mean DM difference: {np.mean(random_dm_diffs):.0f}")
    print(f"  Mean SNR difference: {np.mean(random_snr_diffs):.1f}")

    # Test significance
    t_dm, p_dm = stats.ttest_ind(dm_diffs_nn, random_dm_diffs)
    t_snr, p_snr = stats.ttest_ind(snr_diffs_nn, random_snr_diffs)

    print(f"\nSignificance tests:")
    print(f"  DM: t={t_dm:.2f}, p={p_dm:.3f}")
    print(f"  SNR: t={t_snr:.2f}, p={p_snr:.3f}")

    if p_dm < 0.05 and np.mean(dm_diffs_nn) < np.mean(random_dm_diffs):
        print("  ** Geodesic neighbors have SIMILAR DM **")
    if p_snr < 0.05 and np.mean(snr_diffs_nn) < np.mean(random_snr_diffs):
        print("  ** Geodesic neighbors have SIMILAR SNR **")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if len(geodesic_finite) > 0 and r_eg < 0.95:
        print("\n** GEODESIC ≠ EUCLIDEAN **")
        print("→ The FRB manifold is CURVED")
        print("→ Geodesic distance captures different structure")
    else:
        print("\n** GEODESIC ≈ EUCLIDEAN **")
        print("→ The FRB manifold is approximately FLAT")
        print("→ No significant curvature detected")

    results = {
        "experiment": "exp16_geodesic_structure",
        "timestamp": datetime.now().isoformat(),
        "n_samples": n_frbs,
        "k_main": k_main,
        "graph_structures": graph_results,
        "distance_comparison": {
            "n_pairs": int(len(euclidean_flat)),
            "n_finite_geodesic": int(len(euclidean_finite)) if len(geodesic_finite) > 0 else 0,
            "euclidean_geodesic_correlation": float(r_eg) if len(geodesic_finite) > 0 else None,
            "geodesic_euclidean_ratio_mean": float(np.mean(ratios)) if len(geodesic_finite) > 0 else None,
        },
        "clustering_comparison": {
            "chi_square": float(chi2),
            "p_value": float(p_chi2),
            "same_cluster_fraction": float(n_same / n_frbs),
        },
        "physics_correlations": {
            "euclidean_dm": {"r": float(r_euc_dm), "p": float(p_euc_dm)},
            "euclidean_snr": {"r": float(r_euc_snr), "p": float(p_euc_snr)},
            "geodesic_dm": {"r": float(r_geo_dm), "p": float(p_geo_dm)} if len(geodesic_finite) > 0 else None,
            "geodesic_snr": {"r": float(r_geo_snr), "p": float(p_geo_snr)} if len(geodesic_finite) > 0 else None,
        },
        "geodesic_neighbors": geodesic_nn,
        "frb_names": names,
    }

    output_path = results_dir / "exp16_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
