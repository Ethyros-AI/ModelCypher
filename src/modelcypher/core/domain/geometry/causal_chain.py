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

"""Causal chain profile for model layer analysis.

Computes the validated causal chain per layer:
    Entropy → Curvature (angular) → Cumulative curvature → ID → Phase

All measurements are deterministic geometric properties of the forward pass.
Phase classification uses data-derived boundaries (median curvature,
monotonicity of ID trajectory) — no arbitrary thresholds.

Validated on 6 models: LFM2-350M, LFM2-1.2B, Qwen2.5-3B, Qwen3-8B,
Llama-3.2-3B, Qwen3-1.7B. Key cross-link correlations:
    Entropy ↔ Curvature:         Spearman r ≈ 0.507 (range 0.4-0.6)
    Cumulative curvature ↔ ID:   Family-dependent (Qwen 0.55-0.77, Llama -0.38)
    Attention fraction:          Universal ~0.37 (range 0.36-0.38)

References:
    - Facco et al. (2017) TwoNN intrinsic dimension, Sci. Rep. 7:12140
    - Angular change = arccos(cos_sim): geodesic distance on the unit sphere
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class Phase(str, Enum):
    """Layer phase in the processing pipeline.

    Classification uses data-derived boundaries:
        HIGHWAY:    ID monotonically non-decreasing AND curvature < median
        PROCESSING: curvature >= median
        EXIT:       layers after peak ID where ID monotonically decreases
    """

    HIGHWAY = "highway"
    PROCESSING = "processing"
    EXIT = "exit"


@dataclass
class LayerChainMeasurement:
    """Per-layer measurements in the causal chain."""

    layer_idx: int
    entropy: float
    total_curvature: float  # Angular change in radians
    cumulative_curvature: float  # Sum of total_curvature up to this layer
    attn_curvature: float | None  # None for non-attention layers (LFM2 conv)
    mlp_curvature: float | None
    attn_fraction: float | None  # attn_curvature / (attn + mlp)
    intrinsic_dimension: float  # TwoNN estimate (NaN if insufficient samples)
    phase: Phase


@dataclass
class ChainCorrelations:
    """Cross-link Spearman correlations in the causal chain."""

    entropy_to_curvature: float  # Validated range: 0.4-0.6
    cumulative_curvature_to_id: float  # Family-dependent
    mean_attn_fraction: float | None  # Universal: ~0.36-0.38

    def as_dict(self) -> dict:
        return {
            "entropyToCurvature": self.entropy_to_curvature,
            "cumulativeCurvatureToId": self.cumulative_curvature_to_id,
            "meanAttnFraction": self.mean_attn_fraction,
        }


@dataclass
class ChainProfile:
    """Complete causal chain profile for a model."""

    model_path: str
    num_layers: int
    hidden_dim: int
    probe_count: int
    layers: list[LayerChainMeasurement]
    correlations: ChainCorrelations

    def as_dict(self) -> dict:
        return {
            "modelPath": self.model_path,
            "numLayers": self.num_layers,
            "hiddenDim": self.hidden_dim,
            "probeCount": self.probe_count,
            "layers": [
                {
                    "layer": m.layer_idx,
                    "entropy": m.entropy,
                    "totalCurvature": m.total_curvature,
                    "cumulativeCurvature": m.cumulative_curvature,
                    "attnCurvature": m.attn_curvature,
                    "mlpCurvature": m.mlp_curvature,
                    "attnFraction": m.attn_fraction,
                    "intrinsicDimension": m.intrinsic_dimension,
                    "phase": m.phase.value,
                }
                for m in self.layers
            ],
            "correlations": self.correlations.as_dict(),
        }


# ---------------------------------------------------------------------------
# Pure Python math helpers (no numpy — hexagonal boundary)
# ---------------------------------------------------------------------------


def _vec_norm(v: list[float]) -> float:
    """L2 norm of a flat vector."""
    return math.sqrt(sum(x * x for x in v))


def _vec_dot(v1: list[float], v2: list[float]) -> float:
    """Dot product of two flat vectors."""
    return sum(a * b for a, b in zip(v1, v2))


def _mean(values: list[float]) -> float:
    """Arithmetic mean."""
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    """Median of a list of floats."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return s[mid]


def _std(values: list[float]) -> float:
    """Population standard deviation."""
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


def angular_change(v1: list[float], v2: list[float]) -> float:
    """Compute angular change between two vectors in radians.

    arccos(cosine_similarity) — geodesic distance on the unit sphere.
    Returns 0 for parallel, pi/2 for orthogonal, pi for antiparallel.
    """
    n1 = _vec_norm(v1)
    n2 = _vec_norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    cos_sim = _vec_dot(v1, v2) / (n1 * n2)
    # Clamp for numerical stability
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return math.acos(cos_sim)


# ---------------------------------------------------------------------------
# Curvature computation
# ---------------------------------------------------------------------------


def compute_layer_curvatures(
    sublayer_data: list[dict],
) -> list[dict]:
    """Compute per-layer curvature from sublayer activations.

    Args:
        sublayer_data: List of dicts per layer, each with:
            h_in: list[list[float]] [N, d] — input to layer
            h_out: list[list[float]] [N, d] — output of layer
            h_post_attn: list[list[float]] [N, d] or None
            has_decomposition: bool

    Returns:
        List of dicts with curvature measurements per layer.
    """
    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

    measurements = []
    cumulative = 0.0

    for i, act in enumerate(sublayer_data):
        h_in = act["h_in"]  # list[list[float]] [N, d]
        h_out = act["h_out"]  # list[list[float]] [N, d]
        n_probes = len(h_in)

        # Total curvature: mean angular change h_in → h_out
        total_angles = [angular_change(h_in[j], h_out[j]) for j in range(n_probes)]
        total_curvature = _mean(total_angles)
        cumulative += total_curvature

        # ID via TwoNN
        if n_probes < IntrinsicDimension.local_dimension_min_samples():
            id_val = float("nan")
        else:
            try:
                estimate = IntrinsicDimension.compute_two_nn(h_out)
                id_val = estimate.intrinsic_dimension
            except Exception:
                id_val = float("nan")

        layer_result: dict = {
            "layer_idx": i,
            "total_curvature": total_curvature,
            "cumulative_curvature": cumulative,
            "id_two_nn": id_val,
            "attn_curvature": None,
            "mlp_curvature": None,
            "attn_fraction": None,
        }

        if act["has_decomposition"]:
            h_post_attn = act["h_post_attn"]
            attn_angles = [angular_change(h_in[j], h_post_attn[j]) for j in range(n_probes)]
            mlp_angles = [angular_change(h_post_attn[j], h_out[j]) for j in range(n_probes)]
            attn_curv = _mean(attn_angles)
            mlp_curv = _mean(mlp_angles)

            layer_result["attn_curvature"] = attn_curv
            layer_result["mlp_curvature"] = mlp_curv
            layer_result["attn_fraction"] = (
                attn_curv / (attn_curv + mlp_curv)
                if (attn_curv + mlp_curv) > 1e-10
                else 0.5
            )

        measurements.append(layer_result)

    return measurements


# ---------------------------------------------------------------------------
# Phase classification
# ---------------------------------------------------------------------------


def classify_phases(
    ids: list[float], curvatures: list[float]
) -> list[Phase]:
    """Classify layers into highway / processing / exit phases.

    Uses data-derived boundaries — no arbitrary thresholds:
        - Median curvature splits high/low curvature (data-derived)
        - Monotonicity of ID trajectory is boolean
        - Peak ID position is measured

    Algorithm:
        1. Find peak ID layer (argmax, ignoring NaN)
        2. Layers before peak with curvature < median AND ID non-decreasing → HIGHWAY
        3. Layers with curvature >= median → PROCESSING
        4. Layers after peak ID where ID monotonically decreases → EXIT
    """
    n = len(ids)
    if n == 0:
        return []

    # Filter valid IDs for peak detection
    valid_ids = [(i, v) for i, v in enumerate(ids) if not math.isnan(v)]
    median_curv = _median(curvatures)

    if not valid_ids:
        # No valid IDs — classify purely by curvature
        return [
            Phase.PROCESSING if c >= median_curv else Phase.HIGHWAY
            for c in curvatures
        ]

    peak_idx = max(valid_ids, key=lambda x: x[1])[0]

    phases: list[Phase] = [Phase.PROCESSING] * n

    # EXIT: layers after peak ID where ID monotonically decreases
    # Walk backwards from end to find the monotonic decreasing tail
    exit_start = n  # exclusive start of exit region
    for i in range(n - 1, peak_idx, -1):
        if math.isnan(ids[i]):
            continue
        if exit_start == n:
            # First valid layer from the end
            exit_start = i
        else:
            # Check monotonicity: this layer's ID should be >= next layer's ID
            next_valid = next(
                (ids[j] for j in range(i + 1, n) if not math.isnan(ids[j])),
                None,
            )
            if next_valid is not None and ids[i] >= next_valid:
                exit_start = i
            else:
                break

    for i in range(exit_start, n):
        phases[i] = Phase.EXIT

    # HIGHWAY: layers before peak with curvature < median AND ID non-decreasing
    for i in range(min(peak_idx + 1, exit_start)):
        if curvatures[i] < median_curv:
            # Check ID non-decreasing from start to this point
            is_nondecreasing = True
            prev_id = None
            for j in range(i + 1):
                if not math.isnan(ids[j]):
                    if prev_id is not None and ids[j] < prev_id - 1e-6:
                        is_nondecreasing = False
                        break
                    prev_id = ids[j]
            if is_nondecreasing:
                phases[i] = Phase.HIGHWAY

    return phases


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------


def _spearman_r(xs: list[float], ys: list[float]) -> float:
    """Compute Spearman rank correlation between two sequences.

    Implements the standard rank-transform + Pearson formula.
    Returns NaN if either sequence is constant or too short.
    """
    n = len(xs)
    if n < 3:
        return float("nan")

    def _rank(vals: list[float]) -> list[float]:
        indexed = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and vals[indexed[j + 1]] == vals[indexed[j]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank for ties
            for k in range(i, j + 1):
                ranks[indexed[k]] = avg_rank
            i = j + 1
        return ranks

    rx = _rank(xs)
    ry = _rank(ys)

    # Pearson on ranks
    mean_rx = _mean(rx)
    mean_ry = _mean(ry)
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    denom_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    denom_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))

    if denom_x < 1e-15 or denom_y < 1e-15:
        return float("nan")

    return num / (denom_x * denom_y)


def compute_chain_correlations(
    measurements: list[dict],
) -> ChainCorrelations:
    """Compute Spearman correlations between chain components.

    Returns correlations for the validated causal links:
        entropy ↔ curvature: r ≈ 0.507 (validated range 0.4-0.6)
        cumulative curvature ↔ ID: family-dependent
        mean attention fraction: universal ~0.37
    """
    # Filter to layers with valid ID (not NaN)
    valid = [m for m in measurements if not math.isnan(m["id_two_nn"])]

    # entropy ↔ curvature (use all layers that have entropy)
    entropies = [m.get("entropy", 0.0) for m in measurements]
    curvatures = [m["total_curvature"] for m in measurements]
    if len(entropies) >= 3 and _std(entropies) > 1e-10 and _std(curvatures) > 1e-10:
        ent_curv_r = _spearman_r(entropies, curvatures)
    else:
        ent_curv_r = float("nan")

    # cumulative curvature ↔ ID (valid layers only)
    if len(valid) >= 3:
        cum_curvs = [m["cumulative_curvature"] for m in valid]
        id_vals = [m["id_two_nn"] for m in valid]
        if _std(cum_curvs) > 1e-10 and _std(id_vals) > 1e-10:
            cum_id_r = _spearman_r(cum_curvs, id_vals)
        else:
            cum_id_r = float("nan")
    else:
        cum_id_r = float("nan")

    # Mean attention fraction (layers with decomposition)
    attn_fracs = [
        m["attn_fraction"]
        for m in measurements
        if m["attn_fraction"] is not None
    ]
    mean_attn = _mean(attn_fracs) if attn_fracs else None

    return ChainCorrelations(
        entropy_to_curvature=ent_curv_r,
        cumulative_curvature_to_id=cum_id_r,
        mean_attn_fraction=mean_attn,
    )


# ---------------------------------------------------------------------------
# Profile assembly
# ---------------------------------------------------------------------------


def assemble_chain_profile(
    model_path: str,
    num_layers: int,
    hidden_dim: int,
    probe_count: int,
    curvature_measurements: list[dict],
    entropies: list[float],
) -> ChainProfile:
    """Assemble a complete chain profile from measurements.

    Args:
        model_path: Path to model directory.
        num_layers: Number of transformer layers.
        hidden_dim: Model hidden dimension.
        probe_count: Number of probe texts used.
        curvature_measurements: Output of compute_layer_curvatures().
        entropies: Per-layer entropy values (from BehavioralAnalyzer).
    """
    # Attach entropy to curvature measurements
    for i, m in enumerate(curvature_measurements):
        m["entropy"] = entropies[i] if i < len(entropies) else 0.0

    # Classify phases
    ids = [m["id_two_nn"] for m in curvature_measurements]
    curvatures_list = [m["total_curvature"] for m in curvature_measurements]
    phases = classify_phases(ids, curvatures_list)

    # Compute correlations
    correlations = compute_chain_correlations(curvature_measurements)

    # Build layer measurements
    layers = []
    for m, phase in zip(curvature_measurements, phases):
        layers.append(
            LayerChainMeasurement(
                layer_idx=m["layer_idx"],
                entropy=m["entropy"],
                total_curvature=m["total_curvature"],
                cumulative_curvature=m["cumulative_curvature"],
                attn_curvature=m["attn_curvature"],
                mlp_curvature=m["mlp_curvature"],
                attn_fraction=m["attn_fraction"],
                intrinsic_dimension=m["id_two_nn"],
                phase=phase,
            )
        )

    return ChainProfile(
        model_path=model_path,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        probe_count=probe_count,
        layers=layers,
        correlations=correlations,
    )
