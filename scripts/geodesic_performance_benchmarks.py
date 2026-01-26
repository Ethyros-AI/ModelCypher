#!/usr/bin/env python3
"""Experiment 37: Performance Benchmarks.

Measure computational overhead of geodesic vs Euclidean distance computation.

Key question: What is the performance cost of using geodesic metrics?
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def compute_euclidean_distances(points: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances."""
    n = points.shape[0]
    dists = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        diff = points[i] - points
        dists[i] = np.sqrt(np.sum(diff * diff, axis=1))
    return dists


def compute_geodesic_distances(
    points: np.ndarray,
    k_neighbors: int = 5,
) -> np.ndarray:
    """Compute geodesic distances via k-NN graph."""
    n = points.shape[0]

    # Chord distances
    chord_dists = compute_euclidean_distances(points)

    # Build k-NN graph
    adjacency = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        neighbors = np.argsort(chord_dists[i])[1:k_neighbors + 1]
        for j in neighbors:
            adjacency[i, j] = chord_dists[i, j]
            adjacency[j, i] = chord_dists[j, i]

    # Geodesic via shortest path
    sparse_adj = csr_matrix(adjacency)
    geo_dists = shortest_path(sparse_adj, directed=False)

    # Handle inf
    finite_mask = np.isfinite(geo_dists)
    if not np.all(finite_mask):
        max_finite = np.max(geo_dists[finite_mask]) if np.any(finite_mask) else 1.0
        geo_dists[~finite_mask] = max_finite * 2

    return geo_dists.astype(np.float32)


def compute_linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA."""
    X_c = X - X.mean(axis=0, keepdims=True)
    Y_c = Y - Y.mean(axis=0, keepdims=True)

    K_X = X_c @ X_c.T
    K_Y = Y_c @ Y_c.T

    n = X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n

    K_X_c = H @ K_X @ H
    K_Y_c = H @ K_Y @ H

    hsic_xy = np.trace(K_X_c @ K_Y_c) / ((n - 1) ** 2)
    hsic_xx = np.trace(K_X_c @ K_X_c) / ((n - 1) ** 2)
    hsic_yy = np.trace(K_Y_c @ K_Y_c) / ((n - 1) ** 2)

    if hsic_xx > 1e-10 and hsic_yy > 1e-10:
        cka = hsic_xy / np.sqrt(hsic_xx * hsic_yy)
    else:
        cka = 0.0

    return float(np.clip(cka, 0.0, 1.0))


def compute_geodesic_cka(X: np.ndarray, Y: np.ndarray, k_neighbors: int = 5) -> float:
    """Compute geodesic CKA."""

    def geodesic_gram(points: np.ndarray) -> np.ndarray:
        geo_dists = compute_geodesic_distances(points, k_neighbors)
        geo_sq = geo_dists ** 2
        n = points.shape[0]
        valid_sq = geo_sq[np.triu_indices(n, k=1)]
        sigma = np.median(valid_sq[valid_sq > 1e-10]) if np.any(valid_sq > 1e-10) else 1.0
        sigma = max(sigma, 1e-10)
        K = np.exp(-geo_sq / (2 * sigma))
        return K

    K_X = geodesic_gram(X)
    K_Y = geodesic_gram(Y)

    n = X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n

    K_X_c = H @ K_X @ H
    K_Y_c = H @ K_Y @ H

    hsic_xy = np.trace(K_X_c @ K_Y_c) / ((n - 1) ** 2)
    hsic_xx = np.trace(K_X_c @ K_X_c) / ((n - 1) ** 2)
    hsic_yy = np.trace(K_Y_c @ K_Y_c) / ((n - 1) ** 2)

    if hsic_xx > 1e-10 and hsic_yy > 1e-10:
        cka = hsic_xy / np.sqrt(hsic_xx * hsic_yy)
    else:
        cka = 0.0

    return float(np.clip(cka, 0.0, 1.0))


def run_benchmark(
    n_samples: int,
    d_hidden: int,
    n_trials: int = 3,
) -> Dict:
    """Run benchmark for given dimensions."""
    results = {
        "n_samples": n_samples,
        "d_hidden": d_hidden,
        "euclidean_distance_times": [],
        "geodesic_distance_times": [],
        "linear_cka_times": [],
        "geodesic_cka_times": [],
    }

    for trial in range(n_trials):
        # Generate random data
        X = np.random.randn(n_samples, d_hidden).astype(np.float32)
        Y = np.random.randn(n_samples, d_hidden).astype(np.float32)

        # Euclidean distance
        start = time.perf_counter()
        _ = compute_euclidean_distances(X)
        results["euclidean_distance_times"].append(time.perf_counter() - start)

        # Geodesic distance
        start = time.perf_counter()
        _ = compute_geodesic_distances(X, k_neighbors=min(5, n_samples - 1))
        results["geodesic_distance_times"].append(time.perf_counter() - start)

        # Linear CKA
        start = time.perf_counter()
        _ = compute_linear_cka(X, Y)
        results["linear_cka_times"].append(time.perf_counter() - start)

        # Geodesic CKA
        start = time.perf_counter()
        _ = compute_geodesic_cka(X, Y, k_neighbors=min(5, n_samples - 1))
        results["geodesic_cka_times"].append(time.perf_counter() - start)

    # Compute statistics
    results["euclidean_distance_mean"] = float(np.mean(results["euclidean_distance_times"]))
    results["geodesic_distance_mean"] = float(np.mean(results["geodesic_distance_times"]))
    results["linear_cka_mean"] = float(np.mean(results["linear_cka_times"]))
    results["geodesic_cka_mean"] = float(np.mean(results["geodesic_cka_times"]))

    results["distance_overhead"] = results["geodesic_distance_mean"] / max(results["euclidean_distance_mean"], 1e-10)
    results["cka_overhead"] = results["geodesic_cka_mean"] / max(results["linear_cka_mean"], 1e-10)

    return results


def main():
    logger.info("=" * 60)
    logger.info("EXPERIMENT 37: PERFORMANCE BENCHMARKS")
    logger.info("=" * 60)

    # Test configurations
    configs = [
        (25, 1024),   # Small (typical for single prompt batch)
        (50, 1024),   # Medium
        (100, 1024),  # Larger batch
        (200, 1024),  # Large batch
        (25, 2048),   # Higher dimension
        (25, 4096),   # Very high dimension
        (50, 2048),   # Medium batch, high dim
    ]

    results = {
        "benchmarks": [],
        "summary": {},
    }

    for n_samples, d_hidden in configs:
        logger.info(f"\nBenchmarking n={n_samples}, d={d_hidden}...")
        benchmark = run_benchmark(n_samples, d_hidden, n_trials=3)

        logger.info(f"  Euclidean distance: {benchmark['euclidean_distance_mean']*1000:.2f}ms")
        logger.info(f"  Geodesic distance:  {benchmark['geodesic_distance_mean']*1000:.2f}ms ({benchmark['distance_overhead']:.1f}x)")
        logger.info(f"  Linear CKA:         {benchmark['linear_cka_mean']*1000:.2f}ms")
        logger.info(f"  Geodesic CKA:       {benchmark['geodesic_cka_mean']*1000:.2f}ms ({benchmark['cka_overhead']:.1f}x)")

        results["benchmarks"].append(benchmark)

    # Summary
    distance_overheads = [b["distance_overhead"] for b in results["benchmarks"]]
    cka_overheads = [b["cka_overhead"] for b in results["benchmarks"]]

    results["summary"] = {
        "mean_distance_overhead": float(np.mean(distance_overheads)),
        "max_distance_overhead": float(np.max(distance_overheads)),
        "mean_cka_overhead": float(np.mean(cka_overheads)),
        "max_cka_overhead": float(np.max(cka_overheads)),
    }

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Mean distance overhead: {results['summary']['mean_distance_overhead']:.1f}x")
    logger.info(f"Max distance overhead:  {results['summary']['max_distance_overhead']:.1f}x")
    logger.info(f"Mean CKA overhead:      {results['summary']['mean_cka_overhead']:.1f}x")
    logger.info(f"Max CKA overhead:       {results['summary']['max_cka_overhead']:.1f}x")

    # Interpretation
    if results["summary"]["max_cka_overhead"] < 10:
        logger.info("\nINTERPRETATION: Geodesic overhead is ACCEPTABLE (<10x)")
        results["conclusion"] = "acceptable"
    elif results["summary"]["max_cka_overhead"] < 50:
        logger.info("\nINTERPRETATION: Geodesic overhead is MODERATE (10-50x)")
        results["conclusion"] = "moderate"
    else:
        logger.info("\nINTERPRETATION: Geodesic overhead is HIGH (>50x)")
        results["conclusion"] = "high"

    output_path = "data/experiments/geodesic_performance_benchmarks.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert(results), f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
