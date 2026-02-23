# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""Attention topology: persistent homology on attention-graph distance matrices.

The attention matrix IS the computation graph. Token i attending to token j
with weight A[i,j] is a directed weighted edge. Converting to distance via
D[i,j] = 1 - max(A[i,j], A[j,i]) and computing VR persistence measures the
topological structure of information flow directly — the mechanism, not a
symptom.

Every measurement in this module operates directly on persistence diagrams
(exact birth/death pairs). No discretization, no arbitrary parameters.

Literature:
  - TOHA (Bazarova 2025): AUROC 0.89 on hallucination detection
  - Kostenok 2024: +16% AUC-ARC with persistence features
  - Tan et al. 2025: R²=0.236 vs 0.064 (hidden-state baseline)
  - HalluZig (Samaga, EACL 2026): zigzag persistence over layerwise evolution

This module is pure domain code — no framework imports. Operates on Python
lists/floats and reuses existing PersistenceDiagram / VR filtration code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from modelcypher.core.domain.geometry.topological_fingerprint import (
    PersistenceDiagram,
    TopologicalFingerprint,
)


@dataclass
class GraphBeta1:
    """Graph-theoretic first Betti number (cycle rank) of a thresholded graph.

    β₁ = |E| - |V| + C where E = edges, V = vertices, C = connected components.
    This is the exact first Betti number of the 1-skeleton — not an approximation.
    O(n²) vs O(n³) for full VR filtration.
    """

    beta1: int
    n_vertices: int
    n_edges: int
    n_components: int


@dataclass
class AttentionTopologySignal:
    """Topological features extracted from one inference pass.

    All fields are direct measurements from persistence diagrams
    (birth/death pairs from VR filtration) or Wasserstein distances
    between diagrams. No discretization.
    """

    # Per-head, per-layer persistence diagrams (the raw topological data)
    diagrams: dict[int, list[PersistenceDiagram]] = field(default_factory=dict)

    # Barcode statistics — direct from persistence diagram birth/death pairs
    # (Edelsbrunner & Harer 2010, Computational Topology)
    total_persistence: float = 0.0
    max_persistence: float = 0.0
    persistence_entropy: float = 0.0
    n_bars_h0: int = 0
    n_bars_h1: int = 0

    # Cross-head agreement — Wasserstein metric within each layer
    cross_head_wasserstein: dict[int, float] = field(default_factory=dict)

    # Cross-layer evolution — Wasserstein between consecutive attention layers
    cross_layer_wasserstein: list[float] = field(default_factory=list)

    # Expansion ratio (existing pipeline integration point)
    expansion_ratio: float | None = None


# ---------------------------------------------------------------------------
# Distance conversion
# ---------------------------------------------------------------------------


def attention_to_distance(attn: list[list[float]]) -> list[list[float]]:
    """Convert attention weight matrix to distance matrix.

    D[i,j] = 1 - max(A[i,j], A[j,i])  — symmetrize and invert.
    Diagonal set to 0 (self-distance).

    Derivation: A[i,j] in [0,1] is the softmax-normalized edge weight from
    token i to token j. max(A[i,j], A[j,i]) symmetrizes the directed graph.
    1 - weight inverts to distance: strong attention = short distance.
    Result is a valid metric: D[i,j] in [0,1], D[i,i]=0, D[i,j]=D[j,i].

    Args:
        attn: Attention weight matrix [seq_len, seq_len]. Row i sums to ~1
              (softmax output). Values in [0, 1].

    Returns:
        Symmetric distance matrix [seq_len, seq_len] with values in [0, 1].
    """
    n = len(attn)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = 1.0 - max(attn[i][j], attn[j][i])
            dist[i][j] = d
            dist[j][i] = d
    return dist


# ---------------------------------------------------------------------------
# Graph β₁ proxy (exact graph theory)
# ---------------------------------------------------------------------------


def compute_attention_graph_beta1(
    dist: list[list[float]],
    threshold: float,
) -> GraphBeta1:
    """Compute graph-theoretic β₁ (cycle rank) from an attention distance matrix.

    Given a distance matrix D[i,j] = 1 - max(A[i,j], A[j,i]), build a graph
    where edge (i,j) exists iff D[i,j] < threshold, then compute:

        β₁ = |E| - |V| + C

    This is the exact first Betti number of the 1-skeleton (Hatcher, Ch. 2).
    O(n²) vs O(n³) for full VR filtration.

    NOT wired into compute_attention_topology() because β₁ is highly sensitive
    to the threshold choice (CV 0.37–0.63 across {p25, median, p75, mean} on
    synthetic attention matrices — see scripts/topology_parameter_experiments.py).
    Callers must supply a threshold derived from their specific context.
    VR persistence (which handles thresholds via the filtration) is preferred
    for the automated pipeline.

    Args:
        dist: Symmetric distance matrix [n, n] from attention_to_distance().
        threshold: Distance threshold for edge inclusion. Required — no default,
            because no single default is geometrically justified.

    Returns:
        GraphBeta1 with cycle rank and graph statistics.
    """
    n = len(dist)
    if n < 2:
        return GraphBeta1(beta1=0, n_vertices=n, n_edges=0, n_components=n)

    # Build adjacency and count edges
    adj: list[list[bool]] = [[False] * n for _ in range(n)]
    n_edges = 0
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i][j] < threshold:
                adj[i][j] = True
                adj[j][i] = True
                n_edges += 1

    # Count connected components via DFS
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

    # Cycle rank: |E| - |V| + C — exact first Betti number of 1-skeleton
    beta1 = max(0, n_edges - n + n_components)

    return GraphBeta1(
        beta1=beta1,
        n_vertices=n,
        n_edges=n_edges,
        n_components=n_components,
    )


# ---------------------------------------------------------------------------
# Barcode statistics
# ---------------------------------------------------------------------------


def barcode_statistics(
    diagram: PersistenceDiagram,
) -> tuple[float, float, float, int, int]:
    """Extract barcode statistics from a persistence diagram.

    All quantities are computed directly from birth/death pairs —
    no discretization, no arbitrary parameters.

    Returns:
        (total_persistence, max_persistence, persistence_entropy, n_bars_h0, n_bars_h1)
        - total_persistence: sum(death - birth) — total lifetime of all features
        - max_persistence: max(death - birth) — most persistent feature
        - persistence_entropy: -sum(p_i log p_i) where p_i = pers_i / total
          (Shannon entropy of normalized bar lengths)
        - n_bars_h0: count of H0 features (connected components)
        - n_bars_h1: count of H1 features (loops)
    """
    persistences = [p.persistence for p in diagram.points if p.persistence > 0]
    n_h0 = sum(1 for p in diagram.points if p.dimension == 0)
    n_h1 = sum(1 for p in diagram.points if p.dimension == 1)

    if not persistences:
        return 0.0, 0.0, 0.0, n_h0, n_h1

    total = sum(persistences)
    max_p = max(persistences)

    # Shannon entropy of normalized bar lengths
    entropy = 0.0
    if total > 0:
        for p in persistences:
            prob = p / total
            if prob > 0:
                entropy -= prob * math.log(prob)

    return total, max_p, entropy, n_h0, n_h1


# ---------------------------------------------------------------------------
# Per-head persistence computation
# ---------------------------------------------------------------------------


def _compute_head_diagram(
    attn_head: list[list[float]],
) -> PersistenceDiagram:
    """Compute persistence diagram for a single attention head.

    Converts attention to distance, then runs VR filtration via the existing
    TopologicalFingerprint static method. Filtration range is derived from
    the data (min/max nonzero distances in the matrix).
    """
    dist = attention_to_distance(attn_head)
    n = len(dist)
    if n < 2:
        return PersistenceDiagram([])

    # Find min/max filtration values from distance matrix — data-derived
    max_dist = 0.0
    min_dist = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            d = dist[i][j]
            if d > max_dist:
                max_dist = d
            if 0 < d < min_dist:
                min_dist = d

    if max_dist <= 0:
        return PersistenceDiagram([])
    if min_dist == float("inf"):
        min_dist = 0.0

    return TopologicalFingerprint._vietoris_rips_filtration(
        distances=dist,
        min_filtration=min_dist,
        max_filtration=max_dist,
    )


# ---------------------------------------------------------------------------
# Cross-head and cross-layer agreement
# ---------------------------------------------------------------------------


def _mean_pairwise_wasserstein(diagrams: list[PersistenceDiagram]) -> float:
    """Mean pairwise Wasserstein distance between a list of diagrams.

    Wasserstein distance is a proven metric on persistence diagrams
    (Edelsbrunner & Harer 2010). Mean over all pairs gives a single
    scalar measuring intra-group topological agreement.
    """
    n = len(diagrams)
    if n < 2:
        return 0.0

    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += TopologicalFingerprint._wasserstein_distance(
                diagrams[i], diagrams[j]
            )
            count += 1
    return total / count if count > 0 else 0.0


def _aggregate_diagrams(diagrams: list[PersistenceDiagram]) -> PersistenceDiagram:
    """Merge all points from multiple diagrams into one."""
    all_points = []
    for d in diagrams:
        all_points.extend(d.points)
    return PersistenceDiagram(all_points)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_attention_topology(
    attention_matrices: dict[int, list[list[list[float]]]],
    expansion_ratio: float | None = None,
) -> AttentionTopologySignal:
    """Compute topological signal from attention matrices.

    Args:
        attention_matrices: {layer_idx: [head_0_attn, head_1_attn, ...]}
            where each head_k_attn is [seq_len, seq_len] attention weights.
        expansion_ratio: Optional expansion ratio from existing pipeline.

    Returns:
        AttentionTopologySignal with all computed features.
    """
    if not attention_matrices:
        return AttentionTopologySignal(expansion_ratio=expansion_ratio)

    # Compute per-head persistence diagrams via VR filtration
    # (proven: TOHA AUROC 0.89, Kostenok +16% AUC-ARC, Tan et al. R²=0.236)
    all_diagrams: dict[int, list[PersistenceDiagram]] = {}

    for layer_idx in sorted(attention_matrices.keys()):
        heads = attention_matrices[layer_idx]
        layer_diagrams = []
        for head in heads:
            layer_diagrams.append(_compute_head_diagram(head))
        all_diagrams[layer_idx] = layer_diagrams

    # Aggregate all diagrams for global statistics
    every_diagram = []
    for layer_diagrams in all_diagrams.values():
        every_diagram.extend(layer_diagrams)
    merged = _aggregate_diagrams(every_diagram)

    # Barcode statistics (aggregate) — direct from birth/death pairs
    total_p, max_p, ent, n_h0, n_h1 = barcode_statistics(merged)

    # Cross-head Wasserstein (within each layer)
    cross_head: dict[int, float] = {}
    for layer_idx, layer_diagrams in all_diagrams.items():
        cross_head[layer_idx] = _mean_pairwise_wasserstein(layer_diagrams)

    # Cross-layer Wasserstein (between consecutive layers)
    sorted_layers = sorted(all_diagrams.keys())
    cross_layer: list[float] = []
    for i in range(len(sorted_layers) - 1):
        agg_a = _aggregate_diagrams(all_diagrams[sorted_layers[i]])
        agg_b = _aggregate_diagrams(all_diagrams[sorted_layers[i + 1]])
        cross_layer.append(
            TopologicalFingerprint._wasserstein_distance(agg_a, agg_b)
        )

    return AttentionTopologySignal(
        diagrams=all_diagrams,
        total_persistence=total_p,
        max_persistence=max_p,
        persistence_entropy=ent,
        n_bars_h0=n_h0,
        n_bars_h1=n_h1,
        cross_head_wasserstein=cross_head,
        cross_layer_wasserstein=cross_layer,
        expansion_ratio=expansion_ratio,
    )
