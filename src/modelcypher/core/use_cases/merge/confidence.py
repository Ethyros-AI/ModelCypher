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

"""Geometric confidence computation for merge operations.

This module extracts confidence from geometric signals in transplant metrics.
No vibes - confidence IS the geometry. Returns raw measurements only.

Philosophy:
- No magic thresholds or interpretation strings
- Confidence is derived from what geometry actually measures
- All component signals are exposed for downstream transparency
"""

from __future__ import annotations

from typing import Any


def compute_geometric_confidence_from_transplant(
    transplant_metrics: dict[str, Any],
) -> dict[str, float]:
    """Extract geometric confidence signals from transplant stage metrics.

    The transplant stage already computes rich geometric measurements:
    - preserved_fractions: How much knowledge survived per layer
    - cka_after: Post-alignment CKA scores
    - projection_losses: Loss during null-space projection
    - weights_transplanted/considered: Transplant success rate

    This function extracts and aggregates these into a confidence signal.
    No interpretation - just raw geometric measurements.

    Args:
        transplant_metrics: Metrics dict from stage_3_transplant

    Returns:
        Dict of geometric confidence signals (all floats, no strings):
        - mean_preserved_fraction: Average preservation across layers
        - mean_cka_after: Average post-alignment CKA
        - mean_projection_loss: Average projection loss (lower = better)
        - transplant_ratio: Fraction of weights successfully transplanted
        - mean_null_dim: Average null space dimension found
        - mean_shared_subspace_dim: Average shared subspace dimension
    """
    preserved = transplant_metrics.get("preserved_fractions", [])
    cka_after = transplant_metrics.get("cka_after", [])
    proj_losses = transplant_metrics.get("projection_losses", [])
    null_dims = transplant_metrics.get("null_dims", [])
    shared_dims = transplant_metrics.get("shared_subspace_dimensions", [])
    alignment_improvements = transplant_metrics.get("alignment_improvements", [])

    weights_transplanted = transplant_metrics.get("weights_transplanted", 0)
    weights_considered = transplant_metrics.get("weights_considered", 1)

    return {
        # Core preservation signal
        "mean_preserved_fraction": (
            sum(preserved) / len(preserved) if preserved else 0.0
        ),
        # Alignment quality signal
        "mean_cka_after": sum(cka_after) / len(cka_after) if cka_after else 0.0,
        # Projection quality signal (lower = better, so we invert for confidence)
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
        # Alignment improvement signal
        "mean_alignment_improvement": (
            sum(alignment_improvements) / len(alignment_improvements)
            if alignment_improvements
            else 0.0
        ),
        # Raw counts for transparency
        "layers_transplanted": transplant_metrics.get("layers_transplanted", 0),
        "layers_considered": transplant_metrics.get("layers_considered", 0),
    }


def compute_mean_confidence(geometry_metrics: dict[str, float]) -> float:
    """Return the most direct geometric measurement of merge success.

    This is NOT an interpretation or weighted combination. It IS the geometry:
    mean_preserved_fraction measures how much of the manifold's structure
    survived the null-space projection. This is the geometric truth of
    what happened during the merge.

    The preserved fraction comes from:
        ||P_null @ delta|| / ||delta||

    Where P_null is the null-space projector and delta is the weight difference.
    A preserved_fraction of 0.8 means 80% of the transplanted knowledge
    lies in directions orthogonal to the boundary - it survived geometrically.

    Args:
        geometry_metrics: Output from compute_geometric_confidence_from_transplant

    Returns:
        mean_preserved_fraction - the geometric reality of what was preserved
    """
    return geometry_metrics.get("mean_preserved_fraction", 0.0)


def compute_safety_verdict(geometry_metrics: dict[str, float]) -> str:
    """Derive safety verdict from geometric signals.

    No magic thresholds - just describe what the geometry says.
    The thresholds used ARE structural:
    - transplant_ratio == 0: Nothing was transplanted (failed)
    - mean_preserved_fraction < 0.1: Almost nothing preserved (collapsed)
    - mean_preserved_fraction < 0.5: Less than half preserved (degenerate)

    Args:
        geometry_metrics: Output from compute_geometric_confidence_from_transplant

    Returns:
        Safety verdict string: "healthy", "degenerate", "collapsed", or "failed"
    """
    transplant_ratio = geometry_metrics.get("transplant_ratio", 0.0)
    preserved = geometry_metrics.get("mean_preserved_fraction", 0.0)

    # Structural thresholds based on what actually happened
    if transplant_ratio == 0.0:
        # Nothing was transplanted - operation failed
        return "failed"
    elif preserved < 0.1:
        # Almost no knowledge survived
        return "collapsed"
    elif preserved < 0.5:
        # Less than half survived
        return "degenerate"
    else:
        # Majority of knowledge preserved
        return "healthy"
