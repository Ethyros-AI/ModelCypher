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

from typing import Any


def compute_geometric_metrics_from_transplant(
    transplant_metrics: dict[str, Any],
) -> dict[str, float]:
    """Aggregate geometric measurements from transplant stage metrics.

    The transplant stage already computes rich geometric measurements:
    - preserved_fractions: How much knowledge survived per layer
    - cka_after: Post-alignment CKA scores
    - projection_losses: Loss during null-space projection
    - weights_transplanted/considered: Transplant success rate

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
