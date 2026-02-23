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

Discriminative features (Betti curve statistics, barcode statistics) are
proven in literature:
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
class AttentionTopologySignal:
    """Topological features extracted from one inference pass."""

    # Per-head, per-layer persistence diagrams
    diagrams: dict[int, list[PersistenceDiagram]] = field(default_factory=dict)

    # Betti curve statistics (proven discriminative — Tan et al. 2025)
    betti_curve_width: float = 0.0
    betti_curve_centroid: float = 0.0
    betti_curve_peak: float = 0.0
    betti_curve_spread: float = 0.0

    # Barcode statistics (proven — HalluZig, TOHA)
    total_persistence: float = 0.0
    max_persistence: float = 0.0
    persistence_entropy: float = 0.0
    n_bars_h0: int = 0
    n_bars_h1: int = 0

    # Cross-head agreement (within each layer)
    cross_head_wasserstein: dict[int, float] = field(default_factory=dict)

    # Cross-layer evolution (between consecutive attention layers)
    cross_layer_wasserstein: list[float] = field(default_factory=list)

    # Expansion ratio (existing — integration point)
    expansion_ratio: float | None = None

    def feature_vector(self) -> list[float]:
        """Flat feature vector for calibration / classification."""
        cross_head_mean = (
            sum(self.cross_head_wasserstein.values())
            / len(self.cross_head_wasserstein)
            if self.cross_head_wasserstein
            else 0.0
        )
        cross_layer_mean = (
            sum(self.cross_layer_wasserstein) / len(self.cross_layer_wasserstein)
            if self.cross_layer_wasserstein
            else 0.0
        )
        return [
            self.betti_curve_width,
            self.betti_curve_centroid,
            self.betti_curve_peak,
            self.betti_curve_spread,
            self.total_persistence,
            self.max_persistence,
            self.persistence_entropy,
            float(self.n_bars_h0),
            float(self.n_bars_h1),
            cross_head_mean,
            cross_layer_mean,
            self.expansion_ratio if self.expansion_ratio is not None else 0.0,
        ]

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "betti_curve_width",
            "betti_curve_centroid",
            "betti_curve_peak",
            "betti_curve_spread",
            "total_persistence",
            "max_persistence",
            "persistence_entropy",
            "n_bars_h0",
            "n_bars_h1",
            "cross_head_wasserstein_mean",
            "cross_layer_wasserstein_mean",
            "expansion_ratio",
        ]


# ---------------------------------------------------------------------------
# Distance conversion
# ---------------------------------------------------------------------------


def attention_to_distance(attn: list[list[float]]) -> list[list[float]]:
    """Convert attention weight matrix to distance matrix.

    D[i,j] = 1 - max(A[i,j], A[j,i])  — symmetrize and invert.
    Diagonal set to 0 (self-distance).

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
# Betti curve
# ---------------------------------------------------------------------------


def compute_betti_curve(
    diagram: PersistenceDiagram,
    dimension: int = 1,
    n_steps: int = 100,
) -> list[float]:
    """Compute the Betti curve from a persistence diagram.

    For each filtration value t in [0, max_death], counts the number of
    features alive at t. A feature (b, d) is alive at t if b <= t < d.

    Args:
        diagram: Persistence diagram.
        dimension: Which homological dimension to track (default H1 — loops).
        n_steps: Number of evenly-spaced filtration values.

    Returns:
        List of length n_steps with Betti number at each filtration value.
    """
    points = [p for p in diagram.points if p.dimension == dimension]
    if not points:
        return [0.0] * n_steps

    max_death = max(p.death for p in points)
    min_birth = min(p.birth for p in points)
    span = max_death - min_birth
    if span <= 0:
        return [0.0] * n_steps

    curve = [0.0] * n_steps
    for step in range(n_steps):
        t = min_birth + (step / (n_steps - 1)) * span if n_steps > 1 else min_birth
        count = 0
        for p in points:
            if p.birth <= t < p.death:
                count += 1
        curve[step] = float(count)
    return curve


def betti_curve_statistics(
    curve: list[float],
) -> tuple[float, float, float, float]:
    """Extract discriminative statistics from a Betti curve.

    Returns:
        (width, centroid, peak, spread) where:
        - width: fraction of steps where curve > 0
        - centroid: weighted average position (center of mass)
        - peak: maximum value of curve
        - spread: std dev of nonzero positions
    """
    n = len(curve)
    if n == 0 or max(curve) == 0:
        return 0.0, 0.0, 0.0, 0.0

    nonzero_count = sum(1 for v in curve if v > 0)
    width = nonzero_count / n

    peak = max(curve)

    total_weight = sum(curve)
    if total_weight > 0:
        centroid = sum(i * curve[i] for i in range(n)) / (total_weight * (n - 1)) if n > 1 else 0.0
    else:
        centroid = 0.0

    # Spread: std dev of positions weighted by curve value
    if total_weight > 0 and n > 1:
        mean_pos = sum(i * curve[i] for i in range(n)) / total_weight
        variance = sum(curve[i] * (i - mean_pos) ** 2 for i in range(n)) / total_weight
        spread = math.sqrt(variance) / (n - 1)  # normalize to [0, 1]
    else:
        spread = 0.0

    return width, centroid, peak, spread


# ---------------------------------------------------------------------------
# Barcode statistics
# ---------------------------------------------------------------------------


def barcode_statistics(
    diagram: PersistenceDiagram,
) -> tuple[float, float, float, int, int]:
    """Extract barcode statistics from a persistence diagram.

    Returns:
        (total_persistence, max_persistence, persistence_entropy, n_bars_h0, n_bars_h1)
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
    TopologicalFingerprint static method.
    """
    dist = attention_to_distance(attn_head)
    n = len(dist)
    if n < 2:
        return PersistenceDiagram([])

    # Find min/max filtration values from distance matrix
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
    """Mean pairwise Wasserstein distance between a list of diagrams."""
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
    """Merge all points from multiple diagrams into one (for cross-layer comparison)."""
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

    # Compute per-head persistence diagrams
    all_diagrams: dict[int, list[PersistenceDiagram]] = {}
    for layer_idx in sorted(attention_matrices.keys()):
        heads = attention_matrices[layer_idx]
        layer_diagrams = [_compute_head_diagram(head) for head in heads]
        all_diagrams[layer_idx] = layer_diagrams

    # Aggregate all diagrams for global statistics
    every_diagram = []
    for layer_diagrams in all_diagrams.values():
        every_diagram.extend(layer_diagrams)
    merged = _aggregate_diagrams(every_diagram)

    # Betti curve statistics (aggregate across all heads)
    all_curves: list[list[float]] = []
    for diag in every_diagram:
        curve = compute_betti_curve(diag, dimension=1, n_steps=100)
        all_curves.append(curve)

    if all_curves:
        # Mean curve across all heads
        n_steps = len(all_curves[0])
        mean_curve = [0.0] * n_steps
        for curve in all_curves:
            for i in range(n_steps):
                mean_curve[i] += curve[i]
        for i in range(n_steps):
            mean_curve[i] /= len(all_curves)
        bc_width, bc_centroid, bc_peak, bc_spread = betti_curve_statistics(mean_curve)
    else:
        bc_width, bc_centroid, bc_peak, bc_spread = 0.0, 0.0, 0.0, 0.0

    # Barcode statistics (aggregate)
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
        betti_curve_width=bc_width,
        betti_curve_centroid=bc_centroid,
        betti_curve_peak=bc_peak,
        betti_curve_spread=bc_spread,
        total_persistence=total_p,
        max_persistence=max_p,
        persistence_entropy=ent,
        n_bars_h0=n_h0,
        n_bars_h1=n_h1,
        cross_head_wasserstein=cross_head,
        cross_layer_wasserstein=cross_layer,
        expansion_ratio=expansion_ratio,
    )
