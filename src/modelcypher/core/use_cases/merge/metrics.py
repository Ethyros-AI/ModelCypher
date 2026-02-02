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

"""Geometric metric aggregation for merge operations.

This module extracts raw geometric measurements from transplant metrics.
No interpretation strings, no heuristics - just computed values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

from .models import FingerprintComparison


def compute_geometric_metrics_from_transplant(
    transplant_metrics: dict[str, Any],
) -> dict[str, float]:
    """Aggregate geometric measurements from transplant stage metrics.

    The transplant stage already computes rich geometric measurements:
    - preserved_fractions: How much knowledge survived per layer
    - cka_after: Post-alignment CKA scores
    - projection_losses: Loss during null-space projection
    - weights_transplanted/considered: Transplant success rate
    - core_distance_reductions: Core distance reduction ratios

    This function aggregates raw measurements for downstream use.

    Args:
        transplant_metrics: Metrics dict from stage_3_transplant

    Returns:
        Dict of geometric measurements (all floats, no strings):
        - mean_preserved_fraction: Average preservation across layers
        - mean_cka_after: Average post-alignment CKA
        - mean_projection_loss: Average projection loss (lower indicates less loss)
        - transplant_ratio: Fraction of weights successfully transplanted
        - mean_null_dim: Average null space dimension found
        - mean_shared_subspace_dim: Average shared subspace dimension
        - mean_core_distance_reduction: Average reduction of core distance to source
    """
    preserved = transplant_metrics.get("preserved_fractions", [])
    cka_after = transplant_metrics.get("cka_after", [])
    proj_losses = transplant_metrics.get("projection_losses", [])
    null_dims = transplant_metrics.get("null_dims", [])
    shared_dims = transplant_metrics.get("shared_subspace_dimensions", [])
    core_distance_reductions = transplant_metrics.get("core_distance_reductions", [])

    weights_transplanted = transplant_metrics.get("weights_transplanted", 0)
    weights_considered = transplant_metrics.get("weights_considered", 1)

    return {
        # Core preservation signal
        "mean_preserved_fraction": (
            sum(preserved) / len(preserved) if preserved else 0.0
        ),
        # Alignment quality signal
        "mean_cka_after": sum(cka_after) / len(cka_after) if cka_after else 0.0,
        # Projection quality signal
        "mean_projection_loss": (
            sum(proj_losses) / len(proj_losses) if proj_losses else 0.0
        ),
        # Transplant success signal
        "transplant_ratio": weights_transplanted / max(weights_considered, 1),
        # Structural signals
        "mean_null_dim": sum(null_dims) / len(null_dims) if null_dims else 0.0,
        "mean_shared_subspace_dim": (
            sum(shared_dims) / len(shared_dims) if shared_dims else 0.0
        ),
        # Core distance reduction signal
        "mean_core_distance_reduction": (
            sum(core_distance_reductions) / len(core_distance_reductions)
            if core_distance_reductions
            else 0.0
        ),
        # Raw counts for transparency
        "layers_transplanted": transplant_metrics.get("layers_transplanted", 0),
        "layers_considered": transplant_metrics.get("layers_considered", 0),
    }


def compute_fingerprint_from_activations(
    activations: dict[int, "Array"],
    backend: "Backend",
) -> dict[str, float | str]:
    """Compute geometric fingerprint from layer activations.

    Args:
        activations: Dict mapping layer_idx -> activation array.
            Arrays should be [n_samples, hidden_dim].
        backend: Backend for tensor operations.

    Returns:
        Dict containing:
        - gram_hash: SHA-256 of flattened Gram matrix
        - condition_number: κ = λ_max / λ_min
        - effective_dim: (Σλ)² / Σλ² (participation ratio)
    """
    from modelcypher.core.domain.geometry.geometry_fingerprint import GeometricFingerprint

    if not activations:
        return {
            "gram_hash": "",
            "condition_number": float("inf"),
            "effective_dim": 0.0,
        }

    # Stack all layer activations into a single matrix
    # Each layer contributes its mean-pooled activation vector
    vectors = []
    for layer_idx in sorted(activations.keys()):
        act = activations[layer_idx]
        arr = backend.array(act) if not hasattr(act, "shape") else act
        backend.eval(arr)

        # Mean-pool if 2D (n_samples, hidden_dim) -> (hidden_dim,)
        if len(arr.shape) == 2:
            arr = backend.mean(arr, axis=0)
            backend.eval(arr)

        vectors.append(arr)

    if not vectors:
        return {
            "gram_hash": "",
            "condition_number": float("inf"),
            "effective_dim": 0.0,
        }

    # Stack into matrix [n_layers, hidden_dim]
    stacked = backend.stack(vectors, axis=0)
    backend.eval(stacked)

    # Compute Gram matrix G = X @ X.T where X is [n_layers, hidden_dim]
    gram = backend.matmul(stacked, backend.transpose(stacked))
    backend.eval(gram)

    # Convert to flat list for GeometricFingerprint utilities
    n = int(gram.shape[0])
    gram_flat = backend.tolist(backend.reshape(gram, (-1,)))

    # Compute statistics using existing utilities
    mean, std, gram_hash = GeometricFingerprint.gram_statistics(gram_flat, n)
    condition_number = GeometricFingerprint.estimate_condition_number(gram_flat, n)
    effective_dim = GeometricFingerprint.estimate_effective_dimensionality(gram_flat, n)

    return {
        "gram_hash": gram_hash,
        "condition_number": condition_number,
        "effective_dim": effective_dim,
    }


def compute_fingerprint_comparison(
    source_activations: dict[int, "Array"],
    target_activations: dict[int, "Array"],
    backend: "Backend",
) -> FingerprintComparison:
    """Compute fingerprint comparison between source and target models.

    Args:
        source_activations: Dict mapping layer_idx -> activation array for source.
        target_activations: Dict mapping layer_idx -> activation array for target.
        backend: Backend for tensor operations.

    Returns:
        FingerprintComparison with geometric metrics for both models.
    """
    source_fp = compute_fingerprint_from_activations(source_activations, backend)
    target_fp = compute_fingerprint_from_activations(target_activations, backend)

    source_cond = source_fp["condition_number"]
    target_cond = target_fp["condition_number"]

    # target / source
    if source_cond > 0 and source_cond != float("inf"):
        cond_ratio = target_cond / source_cond
    else:
        cond_ratio = 1.0

    # target - source
    eff_dim_delta = target_fp["effective_dim"] - source_fp["effective_dim"]

    return FingerprintComparison(
        source_gram_hash=source_fp["gram_hash"],
        target_gram_hash=target_fp["gram_hash"],
        source_condition_number=source_cond,
        target_condition_number=target_cond,
        source_effective_dim=source_fp["effective_dim"],
        target_effective_dim=target_fp["effective_dim"],
        condition_number_ratio=cond_ratio,
        effective_dim_delta=eff_dim_delta,
    )
