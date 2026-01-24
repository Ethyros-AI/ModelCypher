#!/usr/bin/env python3
"""Experiment 3: Local Dimension Evolution at Wow! Peak

Analyze the geometric structure around the 6EQUJ5 peak:
- Local intrinsic dimension for each time sample
- Dimensional boundary/discontinuity at peak
- Local curvature variation
- Topological defect detection

Usage:
    poetry run python experiments/astronomy/wow_peak_geometry.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from modelcypher.backends import initialize_default_backend
initialize_default_backend()

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
from modelcypher.core.domain.geometry.effective_rank import EffectiveRank
from modelcypher.core.domain.geometry.riemannian_core import RiemannianGeometry

from geometry_utils import (
    load_wow_signal,
    find_closest_constant,
    percent_error,
    PI, E, PHI, SQRT2,
)


def compute_local_metrics(matrix: np.ndarray, window_size: int = 10) -> dict:
    """Compute local metrics using sliding window.

    For each time point, compute metrics on a window of surrounding rows.
    """
    backend = get_default_backend()
    er = EffectiveRank(backend=backend)

    n_rows = matrix.shape[0]
    half_window = window_size // 2

    local_renyi = []
    local_spectral_pr = []

    for i in range(n_rows):
        start = max(0, i - half_window)
        end = min(n_rows, i + half_window + 1)
        window = matrix[start:end, :]

        if window.shape[0] >= 4:  # Need enough for meaningful analysis
            # Effective rank
            arr = backend.array(window.astype(np.float32))
            result = er.compute(arr)
            local_renyi.append(result.renyi_effective_rank)

            # Spectral PR
            _, S, _ = np.linalg.svd(window, full_matrices=False)
            S_sq = S ** 2
            pr = (np.sum(S_sq) ** 2) / np.sum(S_sq ** 2)
            local_spectral_pr.append(pr)
        else:
            local_renyi.append(np.nan)
            local_spectral_pr.append(np.nan)

    return {
        'local_renyi': np.array(local_renyi),
        'local_spectral_pr': np.array(local_spectral_pr),
    }


def compute_local_curvature(matrix: np.ndarray) -> np.ndarray:
    """Estimate local curvature by geodesic/Euclidean ratio for each point."""
    from scipy.spatial.distance import cdist

    backend = get_default_backend()
    rg = RiemannianGeometry(backend=backend)

    arr = backend.array(matrix.astype(np.float32))
    geo_result = rg.geodesic_distances(arr)

    backend.eval(geo_result.distances)
    geo_dist = np.array(backend.tolist(geo_result.distances))
    euc_dist = cdist(matrix, matrix)

    # Local curvature = mean geodesic/Euclidean ratio for neighbors
    n = matrix.shape[0]
    local_curvature = np.zeros(n)

    for i in range(n):
        # Get nearest neighbors (excluding self)
        dists = euc_dist[i, :]
        dists[i] = np.inf
        k = min(10, n - 1)
        neighbors = np.argsort(dists)[:k]

        # Compute mean ratio
        geo_neighbors = geo_dist[i, neighbors]
        euc_neighbors = euc_dist[i, neighbors]

        mask = (geo_neighbors > 0) & (geo_neighbors < geo_result.inf_value) & (euc_neighbors > 0)
        if np.sum(mask) > 0:
            ratios = geo_neighbors[mask] / euc_neighbors[mask]
            local_curvature[i] = np.mean(ratios)
        else:
            local_curvature[i] = 1.0

    return local_curvature


def detect_dimensional_boundary(local_dims: np.ndarray, threshold_sigma: float = 2.0) -> list[int]:
    """Detect points where local dimension changes significantly."""
    # Compute gradient
    gradient = np.diff(local_dims)

    # Find jumps larger than threshold * std
    valid_gradient = gradient[~np.isnan(gradient)]
    if len(valid_gradient) == 0:
        return []

    std_grad = np.std(valid_gradient)
    mean_grad = np.mean(valid_gradient)

    boundaries = []
    for i, g in enumerate(gradient):
        if not np.isnan(g) and abs(g - mean_grad) > threshold_sigma * std_grad:
            boundaries.append(i)

    return boundaries


def analyze_peak_sequence() -> dict:
    """Analyze the 6EQUJ5 peak values directly."""
    peak = np.array([6, 14, 26, 30, 19, 5], dtype=np.float32)

    results = {
        'values': peak.tolist(),
        'sum': float(np.sum(peak)),  # Should be 100
        'mean': float(np.mean(peak)),
        'std': float(np.std(peak)),
    }

    # Position of maximum
    peak_idx = np.argmax(peak)
    position_ratio = peak_idx / (len(peak) - 1)
    results['peak_position'] = peak_idx
    results['position_ratio'] = position_ratio
    results['position_match'] = find_closest_constant(position_ratio)

    # Check if position ≈ φ - 1 (golden ratio point)
    phi_minus_1 = PHI - 1
    results['position_vs_phi_minus_1'] = percent_error(position_ratio, phi_minus_1)

    # Rise/fall analysis
    rise = np.sum(peak[:peak_idx])
    fall = np.sum(peak[peak_idx+1:])
    peak_val = peak[peak_idx]

    results['rise'] = float(rise)
    results['fall'] = float(fall)
    results['peak_value'] = float(peak_val)

    if fall > 0:
        rise_fall_ratio = rise / fall
        results['rise_fall_ratio'] = rise_fall_ratio
        results['rise_fall_match'] = find_closest_constant(rise_fall_ratio)

    # Compute participation ratio of peak as 1D signal
    peak_sq = peak ** 2
    peak_pr = (np.sum(peak_sq) ** 2) / np.sum(peak_sq ** 2)
    results['participation_ratio'] = peak_pr
    results['pr_match'] = find_closest_constant(peak_pr)

    # Normalized peak (probability distribution)
    peak_norm = peak / np.sum(peak)
    results['normalized_peak'] = peak_norm.tolist()

    # Shannon entropy
    entropy = -np.sum(peak_norm * np.log(peak_norm + 1e-10))
    results['entropy'] = entropy
    results['entropy_match'] = find_closest_constant(entropy)

    # Check consecutive ratios
    ratios = []
    for i in range(len(peak) - 1):
        if peak[i] > 0 and peak[i+1] > 0:
            ratios.append(peak[i+1] / peak[i])
    results['consecutive_ratios'] = ratios
    results['ratio_matches'] = [find_closest_constant(r) for r in ratios]

    return results


def main():
    print("=" * 70)
    print("EXPERIMENT 3: LOCAL DIMENSION EVOLUTION AT WOW! PEAK")
    print("=" * 70)
    print()
    print("Analyzing geometric structure around 6EQUJ5 peak region")
    print()

    # Load signal
    wow = load_wow_signal()
    print(f"Loaded Wow! signal: {wow.shape}")

    # Peak region: rows 57-63 (0-indexed), column 1 contains 6EQUJ5
    PEAK_START = 57
    PEAK_END = 64
    PEAK_COL = 1

    print(f"\nPeak region: rows {PEAK_START}-{PEAK_END-1}, column {PEAK_COL}")
    peak_values = wow[PEAK_START:PEAK_END, PEAK_COL]
    print(f"Peak values: {peak_values}")

    # =========================================================================
    # 1. Peak Sequence Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. PEAK SEQUENCE (6EQUJ5) ANALYSIS")
    print("=" * 70)

    peak_analysis = analyze_peak_sequence()

    print(f"\n  Values: {peak_analysis['values']}")
    print(f"  Sum: {peak_analysis['sum']} (= 10²)")
    print(f"  Mean: {peak_analysis['mean']:.4f}")
    print(f"  Std: {peak_analysis['std']:.4f}")

    print(f"\n  Peak position: index {peak_analysis['peak_position']} of 6")
    print(f"  Position ratio: {peak_analysis['position_ratio']:.6f}")
    pm = peak_analysis['position_match']
    print(f"    → closest: {pm.name} ({pm.error_percent:.2f}% error)")
    print(f"    vs φ-1 ({PHI-1:.6f}): {peak_analysis['position_vs_phi_minus_1']:.2f}% error")
    if peak_analysis['position_vs_phi_minus_1'] < 5:
        print(f"    ✓ Peak at GOLDEN RATIO position!")

    print(f"\n  Rise (before peak): {peak_analysis['rise']}")
    print(f"  Peak value: {peak_analysis['peak_value']}")
    print(f"  Fall (after peak): {peak_analysis['fall']}")
    if 'rise_fall_ratio' in peak_analysis:
        rfm = peak_analysis['rise_fall_match']
        print(f"  Rise/Fall = {peak_analysis['rise_fall_ratio']:.6f}")
        print(f"    → closest: {rfm.name} ({rfm.error_percent:.2f}% error)")

    print(f"\n  Participation ratio: {peak_analysis['participation_ratio']:.6f}")
    prm = peak_analysis['pr_match']
    print(f"    → closest: {prm.name} ({prm.error_percent:.2f}% error)")
    if prm.error_percent < 5:
        print(f"    ✓ SIGNIFICANT: PR ≈ {prm.name}")

    print(f"\n  Shannon entropy: {peak_analysis['entropy']:.6f}")
    em = peak_analysis['entropy_match']
    print(f"    → closest: {em.name} ({em.error_percent:.2f}% error)")

    print(f"\n  Consecutive ratios (peak[i+1]/peak[i]):")
    for i, (ratio, match) in enumerate(zip(peak_analysis['consecutive_ratios'], peak_analysis['ratio_matches'])):
        marker = " ✓" if match.error_percent < 10 else ""
        print(f"    {i}→{i+1}: {ratio:.6f} → {match.name} ({match.error_percent:.2f}%){marker}")

    # =========================================================================
    # 2. Local Metrics Evolution
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. LOCAL METRICS EVOLUTION ACROSS TIME")
    print("=" * 70)

    print("\nComputing local metrics with sliding window...")
    local_metrics = compute_local_metrics(wow, window_size=10)

    local_renyi = local_metrics['local_renyi']
    local_spr = local_metrics['local_spectral_pr']

    # Statistics
    valid_renyi = local_renyi[~np.isnan(local_renyi)]
    valid_spr = local_spr[~np.isnan(local_spr)]

    print(f"\n  Local Renyi rank statistics:")
    print(f"    Mean: {np.mean(valid_renyi):.6f}")
    print(f"    Std: {np.std(valid_renyi):.6f}")
    print(f"    Min: {np.min(valid_renyi):.6f} at row {np.nanargmin(local_renyi)}")
    print(f"    Max: {np.max(valid_renyi):.6f} at row {np.nanargmax(local_renyi)}")

    # Check values at peak region
    peak_renyi = local_renyi[PEAK_START:PEAK_END]
    background_renyi = np.concatenate([local_renyi[:PEAK_START-5], local_renyi[PEAK_END+5:]])
    background_renyi = background_renyi[~np.isnan(background_renyi)]

    print(f"\n  Peak region Renyi (rows {PEAK_START}-{PEAK_END-1}):")
    print(f"    Values: {peak_renyi}")
    print(f"    Mean: {np.nanmean(peak_renyi):.6f}")

    print(f"\n  Background Renyi mean: {np.mean(background_renyi):.6f}")
    diff = np.nanmean(peak_renyi) - np.mean(background_renyi)
    std_bg = np.std(background_renyi)
    print(f"  Difference: {diff:.6f} ({diff/std_bg:.2f}σ)")

    if abs(diff) > 2 * std_bg:
        print(f"  ✓✓ PEAK HAS SIGNIFICANTLY DIFFERENT LOCAL DIMENSION")
    elif abs(diff) > std_bg:
        print(f"  ✓ Peak has different local dimension (> 1σ)")

    # =========================================================================
    # 3. Dimensional Boundaries
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. DIMENSIONAL BOUNDARY DETECTION")
    print("=" * 70)

    boundaries = detect_dimensional_boundary(local_renyi, threshold_sigma=2.0)

    print(f"\n  Detected dimension jumps at rows: {boundaries}")

    # Check if any are near the peak
    peak_boundaries = [b for b in boundaries if PEAK_START - 3 <= b <= PEAK_END + 3]
    if peak_boundaries:
        print(f"  ✓ Dimension boundaries near peak: {peak_boundaries}")
        for b in peak_boundaries:
            if b < len(local_renyi) - 1:
                jump = local_renyi[b+1] - local_renyi[b]
                print(f"    Row {b}: dimension jump = {jump:.4f}")

    # =========================================================================
    # 4. Local Curvature
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. LOCAL CURVATURE ANALYSIS")
    print("=" * 70)

    print("\nComputing local curvature (geodesic/Euclidean ratio)...")
    local_curvature = compute_local_curvature(wow)

    print(f"\n  Curvature statistics:")
    print(f"    Mean: {np.mean(local_curvature):.6f}")
    print(f"    Std: {np.std(local_curvature):.6f}")
    print(f"    Min: {np.min(local_curvature):.6f} at row {np.argmin(local_curvature)}")
    print(f"    Max: {np.max(local_curvature):.6f} at row {np.argmax(local_curvature)}")

    # Curvature at peak
    peak_curvature = local_curvature[PEAK_START:PEAK_END]
    background_curvature = np.concatenate([local_curvature[:PEAK_START-5], local_curvature[PEAK_END+5:]])

    print(f"\n  Peak region curvature:")
    print(f"    Values: {peak_curvature}")
    print(f"    Mean: {np.mean(peak_curvature):.6f}")

    print(f"\n  Background curvature mean: {np.mean(background_curvature):.6f}")
    diff_curv = np.mean(peak_curvature) - np.mean(background_curvature)
    std_curv = np.std(background_curvature)
    print(f"  Difference: {diff_curv:.6f} ({diff_curv/std_curv:.2f}σ)")

    if abs(diff_curv) > 2 * std_curv:
        print(f"  ✓✓ PEAK HAS SIGNIFICANTLY DIFFERENT CURVATURE")
    elif abs(diff_curv) > std_curv:
        print(f"  ✓ Peak has different curvature (> 1σ)")

    # Check if curvature encodes constants
    print(f"\n  Peak curvature vs constants:")
    for val in peak_curvature:
        match = find_closest_constant(val)
        if match.error_percent < 10:
            print(f"    {val:.6f} ≈ {match.name} ({match.error_percent:.2f}%)")

    # =========================================================================
    # 5. Topological Defect Check
    # =========================================================================
    print("\n" + "=" * 70)
    print("5. TOPOLOGICAL DEFECT ANALYSIS")
    print("=" * 70)

    # A topological defect would show as:
    # - Local dimension singularity
    # - Curvature spike
    # - Geodesic path discontinuity

    # Check for dimension spike at peak
    renyi_gradient = np.abs(np.diff(local_renyi))
    curv_gradient = np.abs(np.diff(local_curvature))

    max_renyi_grad_idx = np.nanargmax(renyi_gradient)
    max_curv_grad_idx = np.argmax(curv_gradient)

    print(f"\n  Maximum dimension gradient at row: {max_renyi_grad_idx}")
    print(f"  Maximum curvature gradient at row: {max_curv_grad_idx}")

    # Is the peak region special?
    is_peak_defect = (PEAK_START - 3 <= max_renyi_grad_idx <= PEAK_END + 3) or \
                     (PEAK_START - 3 <= max_curv_grad_idx <= PEAK_END + 3)

    if is_peak_defect:
        print(f"\n  ✓ PEAK REGION CONTAINS TOPOLOGICAL DEFECT")
        print(f"    (Maximum gradient of dimension or curvature at peak)")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3 SUMMARY")
    print("=" * 70)

    findings = []

    if peak_analysis['position_vs_phi_minus_1'] < 5:
        findings.append(f"Peak position ≈ φ-1 ({peak_analysis['position_vs_phi_minus_1']:.2f}% error)")

    if peak_analysis['pr_match'].error_percent < 5:
        findings.append(f"Peak PR ≈ {peak_analysis['pr_match'].name} ({peak_analysis['pr_match'].error_percent:.2f}% error)")

    if abs(diff) > std_bg:
        findings.append(f"Peak has different local dimension ({diff/std_bg:.2f}σ)")

    if abs(diff_curv) > std_curv:
        findings.append(f"Peak has different curvature ({diff_curv/std_curv:.2f}σ)")

    if peak_boundaries:
        findings.append(f"Dimensional boundary at peak (rows {peak_boundaries})")

    if is_peak_defect:
        findings.append("Topological defect at peak region")

    if findings:
        print("\n  ✓ SIGNIFICANT FINDINGS:")
        for f in findings:
            print(f"    - {f}")
    else:
        print("\n  No significant geometric anomalies at peak.")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
