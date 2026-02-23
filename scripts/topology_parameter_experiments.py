#!/usr/bin/env python3
"""Experiments to resolve parameter guesses in the topology pipeline.

Three experiments, each testing a parameter that currently lacks derivation:

1. H1 Cycle Cap Convergence (topological_fingerprint.py:456)
   The VR H1 detection caps at `possible_cycles[:20]`. No derivation.
   Test: Run with caps {10, 20, 50, 100, ALL} on synthetic and real attention
   distance matrices. Measure whether capped and uncapped persistence diagrams
   produce different discriminative signals (Cohen's d for correct vs incorrect).

2. Graph β₁ Threshold Sensitivity (attention_topology.py)
   The graph β₁ proxy uses median nonzero distance when threshold=None.
   Test: Compare {25th percentile, median, 75th percentile, mean} thresholds.
   Measure stability: if β₁ changes wildly across thresholds, the measurement
   is threshold-dependent (bad). If stable, median is fine.

3. Ollivier-Ricci Curvature on Attention Graphs
   The Ricci module (ollivier_ricci.py) has clean derived math but was only
   tested on hidden-state point clouds. The question: does Ricci curvature
   on attention distance matrices discriminate correct from incorrect?
   Test: Compute κ(i,j) = 1 - W₁(mᵢ,mⱼ)/d(i,j) on attention distance
   matrices (same D[i,j] used for VR persistence). Measure Cohen's d.

Usage:
    # All experiments (requires model + benchmark)
    poetry run python scripts/topology_parameter_experiments.py \
        --model LFM2-350M --samples 25 --max-tokens 256 \
        --output results/topology_parameters/

    # H1 cap only (synthetic, no model needed)
    poetry run python scripts/topology_parameter_experiments.py \
        --experiment h1-cap-synthetic \
        --output results/topology_parameters/

    # Graph β₁ threshold only (synthetic)
    poetry run python scripts/topology_parameter_experiments.py \
        --experiment beta1-threshold-synthetic \
        --output results/topology_parameters/
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Experiment 1: H1 Cycle Cap Convergence
# =============================================================================


def _vr_with_cap(
    dist: list[list[float]],
    min_filtration: float,
    max_filtration: float,
    h1_cap: int | None,
) -> list[tuple[float, float, int]]:
    """Run VR filtration with a configurable H1 cycle cap.

    Returns list of (birth, death, dimension) tuples.
    Reimplements the core logic from TopologicalFingerprint._vietoris_rips_filtration
    but with a configurable cap instead of hardcoded 20.
    """
    n = len(dist)
    if n < 2:
        return []

    # Build sorted edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i][j] > 0:
                edges.append((i, j, dist[i][j]))
    edges.sort(key=lambda e: e[2])

    # Union-Find for H0
    parent = list(range(n))
    uf_rank = [0] * n
    component_birth = [0.0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y, births):
        px, py = find(x), find(y)
        if px == py:
            return None
        if uf_rank[px] < uf_rank[py]:
            px, py = py, px
        parent[py] = px
        if uf_rank[px] == uf_rank[py]:
            uf_rank[px] += 1
        dying = py
        births[px] = min(births[px], births[py])
        return dying

    # H0 persistence
    persistence_points = []
    for i, j, d in edges:
        if d > max_filtration:
            break
        dying = union(i, j, component_birth)
        if dying is not None:
            birth = component_birth[dying]
            if d > birth:
                persistence_points.append((birth, d, 0))

    # Surviving components
    for i in range(n):
        if find(i) == i:
            persistence_points.append((0.0, max_filtration, 0))

    # H1 persistence (triangle approximation) — same as topological_fingerprint.py
    if n >= 3:
        parent2 = list(range(n))
        uf_rank2 = [0] * n

        def find2(x):
            while parent2[x] != x:
                parent2[x] = parent2[parent2[x]]
                x = parent2[x]
            return x

        def union2(x, y):
            px, py = find2(x), find2(y)
            if px == py:
                return True  # cycle
            if uf_rank2[px] < uf_rank2[py]:
                px, py = py, px
            parent2[py] = px
            if uf_rank2[px] == uf_rank2[py]:
                uf_rank2[px] += 1
            return False

        possible_cycles = []
        for i, j, d in edges:
            if d > max_filtration:
                break
            if union2(i, j):
                # Cycle detected — find minimum triangle fill
                min_fill = float("inf")
                for k in range(n):
                    if k == i or k == j:
                        continue
                    fill = max(dist[i][k], dist[j][k], d)
                    if fill < min_fill:
                        min_fill = fill
                if min_fill < max_filtration and min_fill > d:
                    possible_cycles.append((d, min_fill, 1))

        # Apply cap
        if h1_cap is not None:
            persistence_points.extend(possible_cycles[:h1_cap])
        else:
            persistence_points.extend(possible_cycles)

    return persistence_points


def _persistence_stats(points: list[tuple[float, float, int]]) -> dict:
    """Compute barcode statistics from persistence points."""
    persistences = [d - b for b, d, dim in points if d > b]
    n_h0 = sum(1 for _, _, dim in points if dim == 0)
    n_h1 = sum(1 for _, _, dim in points if dim == 1)

    if not persistences:
        return {
            "total_persistence": 0.0,
            "max_persistence": 0.0,
            "persistence_entropy": 0.0,
            "n_h0": n_h0,
            "n_h1": n_h1,
        }

    total = sum(persistences)
    max_p = max(persistences)
    entropy = 0.0
    if total > 0:
        for p in persistences:
            prob = p / total
            if prob > 0:
                entropy -= prob * math.log(prob)

    return {
        "total_persistence": total,
        "max_persistence": max_p,
        "persistence_entropy": entropy,
        "n_h0": n_h0,
        "n_h1": n_h1,
    }


def _make_synthetic_attention(n: int, seed: int) -> list[list[float]]:
    """Create synthetic attention matrix for testing.

    Uses a simple LCG-based pseudo-random to create attention-like patterns
    without importing numpy. Values in [0, 1], rows approximately sum to 1.
    """
    # LCG: x_{n+1} = (a*x_n + c) mod m
    a, c, m = 1664525, 1013904223, 2**32
    state = seed
    attn = [[0.0] * n for _ in range(n)]
    for i in range(n):
        row_vals = []
        for j in range(n):
            state = (a * state + c) % m
            row_vals.append(state / m)
        # Softmax-like normalization (rough)
        total = sum(row_vals)
        if total > 0:
            for j in range(n):
                attn[i][j] = row_vals[j] / total
    return attn


def _attention_to_distance(attn: list[list[float]]) -> list[list[float]]:
    """D[i,j] = 1 - max(A[i,j], A[j,i])."""
    n = len(attn)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = 1.0 - max(attn[i][j], attn[j][i])
            dist[i][j] = d
            dist[j][i] = d
    return dist


def run_h1_cap_synthetic(output_dir: Path):
    """Experiment 1: H1 cap convergence on synthetic data.

    Tests whether capping H1 cycles at 20 (vs unlimited) changes the
    persistence statistics. If stats converge by cap=20, the cap is safe.
    If they diverge, the cap is destroying signal.
    """
    logger.info("=== Experiment 1: H1 Cycle Cap Convergence (Synthetic) ===")

    caps = [10, 20, 50, 100, None]  # None = unlimited
    seq_lengths = [8, 16, 32, 64]
    n_seeds = 10

    results = []

    for seq_len in seq_lengths:
        for seed in range(n_seeds):
            attn = _make_synthetic_attention(seq_len, seed=seed * 1000 + seq_len)
            dist = _attention_to_distance(attn)

            # Find filtration range from data
            nonzero = []
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    if dist[i][j] > 0:
                        nonzero.append(dist[i][j])
            if not nonzero:
                continue
            nonzero.sort()
            min_f = nonzero[0]
            max_f = nonzero[-1]

            row = {"seq_len": seq_len, "seed": seed}
            for cap in caps:
                cap_label = str(cap) if cap is not None else "unlimited"
                points = _vr_with_cap(dist, min_f, max_f, h1_cap=cap)
                stats = _persistence_stats(points)
                for k, v in stats.items():
                    row[f"{k}_cap{cap_label}"] = v

            results.append(row)

    # Analysis: compare cap=20 vs unlimited
    logger.info("\nResults by sequence length:")
    report_lines = [
        "# H1 Cycle Cap Convergence",
        "",
        "Tests whether `possible_cycles[:20]` loses H1 signal vs unlimited.",
        "",
        "| seq_len | metric | cap=20 (mean) | unlimited (mean) | delta | delta% |",
        "|---------|--------|---------------|------------------|-------|--------|",
    ]

    for seq_len in seq_lengths:
        rows = [r for r in results if r["seq_len"] == seq_len]
        for metric in ["total_persistence", "persistence_entropy", "n_h1"]:
            cap20_vals = [r.get(f"{metric}_cap20", 0.0) for r in rows]
            unlim_vals = [r.get(f"{metric}_capunlimited", 0.0) for r in rows]
            mean_20 = sum(cap20_vals) / len(cap20_vals) if cap20_vals else 0.0
            mean_u = sum(unlim_vals) / len(unlim_vals) if unlim_vals else 0.0
            delta = mean_u - mean_20
            pct = (delta / mean_u * 100) if mean_u != 0 else 0.0
            report_lines.append(
                f"| {seq_len} | {metric} | {mean_20:.4f} | {mean_u:.4f} "
                f"| {delta:.4f} | {pct:.1f}% |"
            )
            logger.info(
                f"  seq={seq_len} {metric}: cap20={mean_20:.4f} "
                f"unlimited={mean_u:.4f} delta={delta:.4f} ({pct:.1f}%)"
            )

    report_lines.extend([
        "",
        "## Interpretation",
        "",
        "If delta% < 5% for all metrics at seq_len=64, cap=20 is safe.",
        "If delta% > 10%, the cap is destroying signal and needs to be increased.",
        "If n_h1 increases monotonically with seq_len for unlimited but not cap=20,",
        "the cap is saturating and masking real topology.",
    ])

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "h1_cap_convergence.md").write_text("\n".join(report_lines) + "\n")
    (output_dir / "h1_cap_raw.json").write_text(json.dumps(results, indent=2))
    logger.info(f"Results saved to {output_dir}/h1_cap_convergence.md")

    return results


# =============================================================================
# Experiment 2: Graph β₁ Threshold Sensitivity
# =============================================================================


def _compute_graph_beta1(
    dist: list[list[float]], threshold: float
) -> tuple[int, int, int, int]:
    """β₁ = |E| - |V| + C at given threshold. Returns (β₁, V, E, C)."""
    n = len(dist)
    if n < 2:
        return 0, n, 0, n

    # Build adjacency
    n_edges = 0
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i][j] < threshold:
                adj[i][j] = True
                adj[j][i] = True
                n_edges += 1

    # Count components via DFS
    n_components = 0
    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            n_components += 1
            stack = [i]
            while stack:
                v = stack.pop()
                if visited[v]:
                    continue
                visited[v] = True
                for j in range(n):
                    if adj[v][j] and not visited[j]:
                        stack.append(j)

    beta1 = max(0, n_edges - n + n_components)
    return beta1, n, n_edges, n_components


def run_beta1_threshold_synthetic(output_dir: Path):
    """Experiment 2: Graph β₁ threshold sensitivity on synthetic data.

    Tests whether β₁ is stable across reasonable threshold choices.
    """
    logger.info("=== Experiment 2: Graph β₁ Threshold Sensitivity (Synthetic) ===")

    seq_lengths = [8, 16, 32, 64]
    n_seeds = 10

    results = []

    for seq_len in seq_lengths:
        for seed in range(n_seeds):
            attn = _make_synthetic_attention(seq_len, seed=seed * 1000 + seq_len)
            dist = _attention_to_distance(attn)

            # Collect nonzero distances
            nonzero = []
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    if dist[i][j] > 0:
                        nonzero.append(dist[i][j])
            if not nonzero:
                continue
            nonzero.sort()

            # Compute threshold variants
            p25 = nonzero[len(nonzero) // 4]
            median = nonzero[len(nonzero) // 2]
            p75 = nonzero[3 * len(nonzero) // 4]
            mean = sum(nonzero) / len(nonzero)

            thresholds = {
                "p25": p25,
                "median": median,
                "p75": p75,
                "mean": mean,
            }

            row = {"seq_len": seq_len, "seed": seed}
            for name, thresh in thresholds.items():
                beta1, n_v, n_e, n_c = _compute_graph_beta1(dist, thresh)
                row[f"beta1_{name}"] = beta1
                row[f"edges_{name}"] = n_e
                row[f"components_{name}"] = n_c
                row[f"threshold_{name}"] = thresh

            results.append(row)

    # Analysis
    report_lines = [
        "# Graph β₁ Threshold Sensitivity",
        "",
        "Tests whether β₁ is stable across threshold choices.",
        "Current default: median nonzero distance.",
        "",
        "| seq_len | β₁(p25) | β₁(median) | β₁(p75) | β₁(mean) | CV |",
        "|---------|---------|------------|---------|----------|-----|",
    ]

    for seq_len in seq_lengths:
        rows = [r for r in results if r["seq_len"] == seq_len]
        for name in ["p25", "median", "p75", "mean"]:
            vals = [r[f"beta1_{name}"] for r in rows]
            mean_v = sum(vals) / len(vals) if vals else 0.0
            # Store for the row
            if name == "p25":
                row_data = {"seq_len": seq_len, "p25": mean_v}
            else:
                row_data[name] = mean_v

        # CV across threshold choices (per-sample, then average)
        cvs = []
        for r in rows:
            vals = [r["beta1_p25"], r["beta1_median"], r["beta1_p75"], r["beta1_mean"]]
            m = sum(vals) / len(vals) if vals else 0.0
            if m > 0:
                std = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
                cvs.append(std / m)
        mean_cv = sum(cvs) / len(cvs) if cvs else 0.0

        report_lines.append(
            f"| {seq_len} | {row_data['p25']:.1f} | {row_data['median']:.1f} "
            f"| {row_data['p75']:.1f} | {row_data['mean']:.1f} | {mean_cv:.3f} |"
        )
        logger.info(
            f"  seq={seq_len}: β₁(p25)={row_data['p25']:.1f} "
            f"β₁(med)={row_data['median']:.1f} β₁(p75)={row_data['p75']:.1f} "
            f"β₁(mean)={row_data['mean']:.1f} CV={mean_cv:.3f}"
        )

    report_lines.extend([
        "",
        "## Interpretation",
        "",
        "CV (coefficient of variation) measures stability across thresholds.",
        "- CV < 0.2: β₁ is robust to threshold choice → median is fine",
        "- CV 0.2-0.5: moderate sensitivity → threshold matters, need derivation",
        "- CV > 0.5: highly sensitive → β₁ at fixed threshold is unreliable",
        "",
        "If β₁ is threshold-dependent, consider reporting the persistent β₁",
        "(β₁ that survives across a range of thresholds), which is the",
        "topological definition of a 'real' feature.",
    ])

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "beta1_threshold_sensitivity.md").write_text(
        "\n".join(report_lines) + "\n"
    )
    (output_dir / "beta1_threshold_raw.json").write_text(json.dumps(results, indent=2))
    logger.info(f"Results saved to {output_dir}/beta1_threshold_sensitivity.md")

    return results


# =============================================================================
# Experiment 3: Ollivier-Ricci on Attention Graphs
# =============================================================================


def _sinkhorn_wasserstein(
    mu: list[float], nu: list[float], cost: list[list[float]],
    max_iter: int | None = None, reg: float = 0.1,
) -> float:
    """Sinkhorn-regularized W₁ between two discrete distributions.

    max_iter derived from convergence theory: n * ceil(log(n+1))
    where n = support size (Altschuler et al. 2017).
    reg = 0.1 is the only guess here — experiment will test sensitivity.
    """
    n = len(mu)
    if n == 0:
        return 0.0

    if max_iter is None:
        max_iter = n * math.ceil(math.log(n + 1))

    # Gibbs kernel K[i,j] = exp(-C[i,j] / reg)
    K = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            K[i][j] = math.exp(-cost[i][j] / reg)

    u = [1.0] * n
    v = [1.0] * n

    for _ in range(max_iter):
        # u = mu / (K @ v)
        for i in range(n):
            Kv_i = sum(K[i][j] * v[j] for j in range(n))
            u[i] = mu[i] / max(Kv_i, 1e-30)
        # v = nu / (K^T @ u)
        for j in range(n):
            Ku_j = sum(K[i][j] * u[i] for i in range(n))
            v[j] = nu[j] / max(Ku_j, 1e-30)

    # Transport cost = sum_ij u_i K_ij v_j C_ij
    total = 0.0
    for i in range(n):
        for j in range(n):
            total += u[i] * K[i][j] * v[j] * cost[i][j]
    return total


def _compute_ricci_on_attention(
    dist: list[list[float]],
    threshold: float | None = None,
) -> dict:
    """Compute Ollivier-Ricci curvature on an attention distance graph.

    For each edge (i,j) with d(i,j) < threshold:
        κ(i,j) = 1 - W₁(mᵢ, mⱼ) / d(i,j)

    where mᵢ is a lazy random walk: mass α on self, (1-α)/degree on neighbors.
    α = 1/(1 + degree_i) — derived from graph density (same as ollivier_ricci.py).
    """
    n = len(dist)
    if n < 2:
        return {"mean_curvature": 0.0, "std_curvature": 0.0, "n_edges": 0}

    # Derive threshold from data if not given
    if threshold is None:
        nonzero = []
        for i in range(n):
            for j in range(i + 1, n):
                if dist[i][j] > 0:
                    nonzero.append(dist[i][j])
        if not nonzero:
            return {"mean_curvature": 0.0, "std_curvature": 0.0, "n_edges": 0}
        nonzero.sort()
        threshold = nonzero[len(nonzero) // 2]

    # Build adjacency
    neighbors: dict[int, list[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i][j] < threshold:
                neighbors[i].append(j)
                neighbors[j].append(i)

    # Compute curvature for each edge
    curvatures = []
    for i in range(n):
        for j in neighbors[i]:
            if j <= i:
                continue  # avoid double counting

            d_ij = dist[i][j]
            if d_ij <= 0:
                continue

            # Lazy random walk measures
            deg_i = len(neighbors[i])
            deg_j = len(neighbors[j])
            alpha_i = 1.0 / (1.0 + deg_i)
            alpha_j = 1.0 / (1.0 + deg_j)

            # Build distributions over all n nodes
            mu = [0.0] * n
            nu = [0.0] * n
            mu[i] = alpha_i
            for k in neighbors[i]:
                mu[k] += (1.0 - alpha_i) / deg_i
            nu[j] = alpha_j
            for k in neighbors[j]:
                nu[k] += (1.0 - alpha_j) / deg_j

            # W₁ via Sinkhorn
            w1 = _sinkhorn_wasserstein(mu, nu, dist)
            kappa = 1.0 - w1 / d_ij
            curvatures.append(kappa)

    if not curvatures:
        return {"mean_curvature": 0.0, "std_curvature": 0.0, "n_edges": 0}

    mean_k = sum(curvatures) / len(curvatures)
    var_k = sum((k - mean_k) ** 2 for k in curvatures) / len(curvatures)
    std_k = var_k ** 0.5

    return {
        "mean_curvature": mean_k,
        "std_curvature": std_k,
        "min_curvature": min(curvatures),
        "max_curvature": max(curvatures),
        "n_edges": len(curvatures),
    }


def run_ricci_synthetic(output_dir: Path):
    """Experiment 3: Ricci curvature on synthetic attention graphs.

    Tests whether Ricci curvature produces finite, non-degenerate values
    on attention distance matrices. Full discriminative power test
    requires real model data (see run_ricci_real).
    """
    logger.info("=== Experiment 3: Ricci on Attention Graphs (Synthetic) ===")

    seq_lengths = [8, 16, 32]
    n_seeds = 5

    results = []
    for seq_len in seq_lengths:
        for seed in range(n_seeds):
            t0 = time.time()
            attn = _make_synthetic_attention(seq_len, seed=seed * 1000 + seq_len)
            dist = _attention_to_distance(attn)
            ricci = _compute_ricci_on_attention(dist)
            elapsed = time.time() - t0

            row = {
                "seq_len": seq_len,
                "seed": seed,
                "elapsed_s": elapsed,
                **ricci,
            }
            results.append(row)
            logger.info(
                f"  seq={seq_len} seed={seed}: κ_mean={ricci['mean_curvature']:.4f} "
                f"κ_std={ricci['std_curvature']:.4f} edges={ricci['n_edges']} "
                f"({elapsed:.2f}s)"
            )

    report_lines = [
        "# Ollivier-Ricci on Attention Graphs (Synthetic)",
        "",
        "Verifies Ricci curvature produces finite, non-degenerate values.",
        "",
        "| seq_len | κ_mean | κ_std | κ_min | κ_max | edges | time(s) |",
        "|---------|--------|-------|-------|-------|-------|---------|",
    ]

    for seq_len in seq_lengths:
        rows = [r for r in results if r["seq_len"] == seq_len]
        means = [r["mean_curvature"] for r in rows]
        stds = [r["std_curvature"] for r in rows]
        edges = [r["n_edges"] for r in rows]
        times = [r["elapsed_s"] for r in rows]
        avg_mean = sum(means) / len(means)
        avg_std = sum(stds) / len(stds)
        avg_edges = sum(edges) / len(edges)
        avg_time = sum(times) / len(times)
        min_k = min(r.get("min_curvature", 0.0) for r in rows)
        max_k = max(r.get("max_curvature", 0.0) for r in rows)
        report_lines.append(
            f"| {seq_len} | {avg_mean:.4f} | {avg_std:.4f} "
            f"| {min_k:.4f} | {max_k:.4f} | {avg_edges:.0f} | {avg_time:.2f} |"
        )

    report_lines.extend([
        "",
        "## Feasibility Assessment",
        "",
        "- κ values should be in [-1, 1] (negative = hyperbolic, positive = spherical)",
        "- std > 0 means curvature varies across edges (non-degenerate)",
        "- Time scaling: O(n³) per head due to Sinkhorn — check if feasible at seq_len=64+",
        "",
        "If κ is near-constant across all attention heads: Ricci adds no signal.",
        "If κ varies meaningfully: proceed to real-model discriminative test.",
    ])

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ricci_attention_synthetic.md").write_text(
        "\n".join(report_lines) + "\n"
    )
    (output_dir / "ricci_attention_raw.json").write_text(json.dumps(results, indent=2))
    logger.info(f"Results saved to {output_dir}/ricci_attention_synthetic.md")

    return results


# =============================================================================
# Full experiment with real model (requires model loading)
# =============================================================================


def run_all_real(model_name: str, n_samples: int, max_tokens: int, output_dir: Path):
    """Run all experiments on real attention data from model inference.

    Requires model loading. Tests discriminative power of each parameter
    choice for predicting correct vs incorrect answers.
    """
    from modelcypher.adapters.activation_provider import ActivationProviderAdapter
    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain.statistics import cohens_d_two_groups
    from modelcypher.core.use_cases.curriculum.benchmark_loader import BenchmarkLoader

    MODEL_PATHS = {
        "LFM2-350M": "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
        "LFM2-700M": "/Volumes/CodeCypher/models/mlx-community/LFM2-700M-bf16",
        "LFM2-1.2B": "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16",
    }

    model_path = MODEL_PATHS.get(model_name)
    if not model_path:
        raise ValueError(f"Unknown model: {model_name}")

    logger.info(f"Loading model from {model_path}")
    backend = initialize_default_backend()
    model, tokenizer = load_model_for_training(model_path)
    provider = ActivationProviderAdapter(backend=backend, model_path=model_path)
    logger.info("Model loaded")

    # Load benchmark
    loader = BenchmarkLoader()
    bm = loader.load("gsm8k", split="test", limit=n_samples)
    samples = [(s.prompt, s.answer) for s in bm.samples]
    logger.info(f"Loaded {len(samples)} gsm8k samples")

    import re

    def extract_answer(text: str) -> str:
        boxed = re.search(r"\\boxed\{([^}]+)\}", text)
        if boxed:
            return boxed.group(1).strip()
        sep = re.search(r"####\s*(.+)", text)
        if sep:
            return sep.group(1).strip()
        numbers = re.findall(r"-?\d+\.?\d*", text)
        return numbers[-1] if numbers else text.strip()

    def normalize(ans: str) -> str:
        ans = ans.strip().replace(",", "").replace("$", "").replace("%", "")
        try:
            val = float(ans)
            return str(int(val)) if val == int(val) else str(val)
        except ValueError:
            return ans.lower()

    # Collect attention data + answers
    @dataclass
    class Sample:
        prompt: str
        expected: str
        is_correct: bool
        dist_matrices: list[list[list[list[float]]]]  # per-layer, per-head
        ricci_per_head: list[dict]
        beta1_per_threshold: dict  # {threshold_name: [per-head β₁]}
        h1_per_cap: dict  # {cap: persistence_stats}

    collected: list[Sample] = []

    for idx, (prompt, answer) in enumerate(samples):
        t0 = time.time()
        formatted = f"Q: {prompt}\nA:"

        # Extract attention matrices
        try:
            attn_raw = provider.collect_attention_matrices(model, tokenizer, formatted)
        except Exception as e:
            logger.error(f"[{idx+1}] Attention extraction failed: {e}")
            raise

        if not attn_raw:
            logger.error(f"[{idx+1}] Empty attention matrices")
            raise RuntimeError("Empty attention matrices")

        # Generate answer
        prompt_tokens = tokenizer.encode(formatted)
        if not isinstance(prompt_tokens, list):
            prompt_tokens = backend.tolist(prompt_tokens)
        current_ids = backend.array([prompt_tokens])
        generated_tokens = []

        for _ in range(max_tokens):
            logits = model(current_ids)
            backend.eval(logits)
            next_logits = logits[0, -1, :] if logits.ndim == 3 else logits[-1, :]
            next_id = int(backend.to_scalar(backend.argmax(next_logits)))
            if next_id == tokenizer.eos_token_id:
                break
            generated_tokens.append(next_id)
            current_ids = backend.array([prompt_tokens + generated_tokens])

        gen_text = tokenizer.decode(generated_tokens)
        extracted = extract_answer(gen_text)
        is_correct = normalize(extracted) == normalize(answer)

        # Convert attention to Python lists + distance matrices
        all_dist_matrices: list[list[list[list[float]]]] = []
        all_ricci: list[dict] = []
        all_beta1: dict[str, list[int]] = {"p25": [], "median": [], "p75": [], "mean": []}
        all_h1: dict[str, dict] = {}

        for layer_idx in sorted(attn_raw.keys()):
            head_arrays = attn_raw[layer_idx]
            layer_dists = []
            for head_arr in head_arrays:
                backend.eval(head_arr)
                head_list = backend.tolist(head_arr)
                dist = _attention_to_distance(head_list)
                layer_dists.append(dist)

                # Ricci per head
                ricci = _compute_ricci_on_attention(dist)
                all_ricci.append(ricci)

                # β₁ at different thresholds
                nonzero = []
                nn = len(dist)
                for i in range(nn):
                    for j in range(i + 1, nn):
                        if dist[i][j] > 0:
                            nonzero.append(dist[i][j])
                if nonzero:
                    nonzero.sort()
                    thresholds = {
                        "p25": nonzero[len(nonzero) // 4],
                        "median": nonzero[len(nonzero) // 2],
                        "p75": nonzero[3 * len(nonzero) // 4],
                        "mean": sum(nonzero) / len(nonzero),
                    }
                    for name, thresh in thresholds.items():
                        b1, _, _, _ = _compute_graph_beta1(dist, thresh)
                        all_beta1[name].append(b1)

                # H1 at different caps (use first head of first layer only — expensive)
                if layer_idx == min(attn_raw.keys()) and len(layer_dists) == 1:
                    min_f = nonzero[0] if nonzero else 0.0
                    max_f = nonzero[-1] if nonzero else 1.0
                    for cap in [10, 20, 50, 100, None]:
                        cap_label = str(cap) if cap is not None else "unlimited"
                        points = _vr_with_cap(dist, min_f, max_f, h1_cap=cap)
                        all_h1[cap_label] = _persistence_stats(points)

            all_dist_matrices.append(layer_dists)

        collected.append(Sample(
            prompt=prompt,
            expected=answer,
            is_correct=is_correct,
            dist_matrices=all_dist_matrices,
            ricci_per_head=all_ricci,
            beta1_per_threshold=all_beta1,
            h1_per_cap=all_h1,
        ))

        elapsed = time.time() - t0
        status = "CORRECT" if is_correct else "WRONG"
        logger.info(f"[{idx+1}/{len(samples)}] {status} ({elapsed:.1f}s)")

    # Analysis
    correct = [s for s in collected if s.is_correct]
    incorrect = [s for s in collected if not s.is_correct]

    report_lines = [
        f"# Topology Parameter Experiments — {model_name}",
        f"",
        f"**Samples:** {len(collected)} ({len(correct)} correct, {len(incorrect)} incorrect)",
        f"",
    ]

    # Ricci discriminative power
    report_lines.extend(["## Experiment 3: Ricci Curvature Discriminative Power", ""])
    if len(correct) >= 2 and len(incorrect) >= 2:
        c_means = [
            sum(r["mean_curvature"] for r in s.ricci_per_head) / max(len(s.ricci_per_head), 1)
            for s in correct
        ]
        i_means = [
            sum(r["mean_curvature"] for r in s.ricci_per_head) / max(len(s.ricci_per_head), 1)
            for s in incorrect
        ]
        d = cohens_d_two_groups(c_means, i_means)
        report_lines.extend([
            f"| Metric | Cohen's d | Mean(correct) | Mean(incorrect) |",
            f"|--------|-----------|---------------|-----------------|",
            f"| mean_ricci_curvature | {d:.4f} | {sum(c_means)/len(c_means):.4f} | {sum(i_means)/len(i_means):.4f} |",
            f"",
            f"|d| < 0.2: negligible effect → Ricci adds nothing on attention graphs",
            f"|d| 0.2-0.5: small effect → worth exploring further",
            f"|d| > 0.5: medium+ effect → wire into pipeline",
            f"",
        ])
    else:
        report_lines.append("Insufficient samples for Cohen's d (need >= 2 per group).\n")

    # β₁ threshold discriminative power
    report_lines.extend(["## Experiment 2: β₁ Threshold — Discriminative Power", ""])
    if len(correct) >= 2 and len(incorrect) >= 2:
        report_lines.extend([
            "| Threshold | Cohen's d | Mean(correct) | Mean(incorrect) |",
            "|-----------|-----------|---------------|-----------------|",
        ])
        for tname in ["p25", "median", "p75", "mean"]:
            c_vals = [sum(s.beta1_per_threshold[tname]) for s in correct]
            i_vals = [sum(s.beta1_per_threshold[tname]) for s in incorrect]
            if c_vals and i_vals:
                d = cohens_d_two_groups(c_vals, i_vals)
                mc = sum(c_vals) / len(c_vals)
                mi = sum(i_vals) / len(i_vals)
                report_lines.append(
                    f"| {tname} | {d:.4f} | {mc:.2f} | {mi:.2f} |"
                )
        report_lines.append("")

    # H1 cap discriminative power
    report_lines.extend(["## Experiment 1: H1 Cap — Discriminative Power", ""])
    if len(correct) >= 2 and len(incorrect) >= 2:
        report_lines.extend([
            "| Cap | Cohen's d (n_h1) | Cohen's d (total_pers) | Cohen's d (entropy) |",
            "|-----|-------------------|----------------------|---------------------|",
        ])
        for cap_label in ["10", "20", "50", "100", "unlimited"]:
            c_h1 = [
                s.h1_per_cap.get(cap_label, {}).get("n_h1", 0.0) for s in correct
                if cap_label in s.h1_per_cap
            ]
            i_h1 = [
                s.h1_per_cap.get(cap_label, {}).get("n_h1", 0.0) for s in incorrect
                if cap_label in s.h1_per_cap
            ]
            c_tp = [
                s.h1_per_cap.get(cap_label, {}).get("total_persistence", 0.0) for s in correct
                if cap_label in s.h1_per_cap
            ]
            i_tp = [
                s.h1_per_cap.get(cap_label, {}).get("total_persistence", 0.0) for s in incorrect
                if cap_label in s.h1_per_cap
            ]
            c_ent = [
                s.h1_per_cap.get(cap_label, {}).get("persistence_entropy", 0.0) for s in correct
                if cap_label in s.h1_per_cap
            ]
            i_ent = [
                s.h1_per_cap.get(cap_label, {}).get("persistence_entropy", 0.0) for s in incorrect
                if cap_label in s.h1_per_cap
            ]

            if len(c_h1) >= 2 and len(i_h1) >= 2:
                d_h1 = cohens_d_two_groups(c_h1, i_h1)
                d_tp = cohens_d_two_groups(c_tp, i_tp)
                d_ent = cohens_d_two_groups(c_ent, i_ent)
                report_lines.append(
                    f"| {cap_label} | {d_h1:.4f} | {d_tp:.4f} | {d_ent:.4f} |"
                )
            else:
                report_lines.append(f"| {cap_label} | insufficient data | | |")
        report_lines.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "EXPERIMENT_REPORT.md").write_text("\n".join(report_lines) + "\n")

    # Raw data
    raw = []
    for s in collected:
        raw.append({
            "prompt": s.prompt,
            "expected": s.expected,
            "is_correct": s.is_correct,
            "ricci_per_head": s.ricci_per_head,
            "beta1_per_threshold": s.beta1_per_threshold,
            "h1_per_cap": s.h1_per_cap,
        })
    (output_dir / "raw_experiment_data.json").write_text(json.dumps(raw, indent=2))
    logger.info(f"Full report saved to {output_dir}/EXPERIMENT_REPORT.md")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Topology parameter experiments — resolve guesses",
    )
    parser.add_argument(
        "--experiment",
        choices=[
            "h1-cap-synthetic",
            "beta1-threshold-synthetic",
            "ricci-synthetic",
            "all-synthetic",
            "all-real",
        ],
        default="all-synthetic",
        help="Which experiment to run (default: all-synthetic)",
    )
    parser.add_argument("--model", default="LFM2-350M")
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Max tokens for generation (required for all-real)",
    )
    parser.add_argument(
        "--output", type=str, default="results/topology_parameters/",
    )

    args = parser.parse_args()
    output_dir = Path(args.output)

    if args.experiment == "h1-cap-synthetic":
        run_h1_cap_synthetic(output_dir)
    elif args.experiment == "beta1-threshold-synthetic":
        run_beta1_threshold_synthetic(output_dir)
    elif args.experiment == "ricci-synthetic":
        run_ricci_synthetic(output_dir)
    elif args.experiment == "all-synthetic":
        run_h1_cap_synthetic(output_dir)
        run_beta1_threshold_synthetic(output_dir)
        run_ricci_synthetic(output_dir)
    elif args.experiment == "all-real":
        if args.max_tokens is None:
            parser.error("--max-tokens is required for all-real experiment")
        run_all_real(args.model, args.samples, args.max_tokens, output_dir)


if __name__ == "__main__":
    main()
