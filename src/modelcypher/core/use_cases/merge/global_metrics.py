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

from __future__ import annotations

import logging

from .data_models import MergeGeometry

logger = logging.getLogger(__name__)


def compute_global_metrics(geometry: MergeGeometry) -> None:
    """Compute global summary metrics."""
    layer_geoms = list(geometry.layer_geometries.values())
    if not layer_geoms:
        return

    # Mean intrinsic dimension
    dims = [lg.intrinsic_dimension for lg in layer_geoms if lg.intrinsic_dimension > 0]
    geometry.mean_intrinsic_dimension = sum(dims) / len(dims) if dims else 0.0

    # Mean shared dimension
    shared = [lg.shared_dimension for lg in layer_geoms if lg.shared_dimension > 0]
    geometry.mean_shared_dimension = sum(shared) / len(shared) if shared else 0.0

    # Update overall_cka from per-layer alignment quality (POST-alignment CKA)
    # This replaces the pre-alignment CKA with the actual achieved alignment
    aligned_ckas = [lg.alignment_quality for lg in layer_geoms if lg.alignment_quality > 0]
    if aligned_ckas:
        geometry.overall_cka = min(aligned_ckas)  # Use min to ensure ALL layers are aligned
        logger.debug(
            "Updated overall_cka from per-layer alignment: min=%.8f, mean=%.8f",
            min(aligned_ckas),
            sum(aligned_ckas) / len(aligned_ckas),
        )

    # Aggregate Ollivier-Ricci curvature and manifold health
    ricci_values = [
        lg.ollivier_ricci_mean for lg in layer_geoms if lg.manifold_health != "unknown"
    ]
    if ricci_values:
        geometry.mean_ollivier_ricci = sum(ricci_values) / len(ricci_values)

        # Overall health is determined by the worst layer
        # Collapsed > Degenerate > Healthy (ordered by severity)
        health_counts = {"collapsed": 0, "degenerate": 0, "healthy": 0}
        for lg in layer_geoms:
            if lg.manifold_health in health_counts:
                health_counts[lg.manifold_health] += 1

        if health_counts["collapsed"] > 0:
            geometry.overall_manifold_health = "collapsed"
        elif health_counts["degenerate"] > len(layer_geoms) // 2:
            geometry.overall_manifold_health = "degenerate"
        else:
            geometry.overall_manifold_health = "healthy"

        logger.info(
            "MANIFOLD HEALTH: %s (mean_ricci=%.4f, healthy=%d, degenerate=%d, collapsed=%d)",
            geometry.overall_manifold_health,
            geometry.mean_ollivier_ricci,
            health_counts["healthy"],
            health_counts["degenerate"],
            health_counts["collapsed"],
        )

    # Compute curvature compatibility
    # This compares curvature profiles of source and target to inform merge confidence
    _compute_curvature_compatibility(geometry, layer_geoms)

    logger.info(
        "MERGE GEOMETRY: %d layers, mean_intrinsic_dim=%.1f, mean_shared_dim=%.1f, CKA=%.4f, health=%s, curv_compat=%.3f",
        len(layer_geoms),
        geometry.mean_intrinsic_dimension,
        geometry.mean_shared_dimension,
        geometry.overall_cka,
        geometry.overall_manifold_health,
        geometry.curvature_compatibility,
    )


def _compute_curvature_compatibility(geometry: MergeGeometry, layer_geoms: list) -> None:
    """Compute curvature compatibility score from per-layer geometry.

    Curvature compatibility measures how geometrically similar the source and
    target representations are. Higher compatibility suggests easier merging.

    The score is computed from:
    - Consistency of curvature signs across layers
    - Variance in curvature values (low = more compatible)
    - Overall health distribution

    Returns a score from 0.0 (incompatible) to 1.0 (highly compatible).
    """
    import math

    # Collect valid curvature values
    sectional = [lg.curvature for lg in layer_geoms if lg.curvature != 0]
    ricci = [lg.ollivier_ricci_mean for lg in layer_geoms if lg.manifold_health != "unknown"]
    dims = [lg.intrinsic_dimension for lg in layer_geoms if lg.intrinsic_dimension > 0]

    if not ricci:
        # No curvature data available
        geometry.curvature_compatibility = 0.0
        geometry.curvature_compatibility_details = {"error": "no_curvature_data"}
        return

    # Component 1: Curvature consistency (0-1)
    # Healthy LLMs have negative Ricci curvature - how consistent is this?
    negative_ratio = sum(1 for r in ricci if r < 0) / len(ricci)
    consistency_score = negative_ratio  # 1.0 = all negative (healthy)

    # Component 2: Curvature stability (0-1)
    # Low variance in curvature = more stable manifold = higher compatibility
    if len(ricci) > 1:
        mean_ricci = sum(ricci) / len(ricci)
        variance = sum((r - mean_ricci) ** 2 for r in ricci) / (len(ricci) - 1)
        std_ricci = math.sqrt(variance)
        # Normalize: 0.1 std = 1.0 score, 0.5+ std = 0.0 score
        stability_score = max(0.0, 1.0 - std_ricci / 0.5)
    else:
        stability_score = 0.5  # Unknown stability with single sample

    # Component 3: Dimension consistency (0-1)
    if len(dims) > 1:
        mean_dim = sum(dims) / len(dims)
        dim_variance = sum((d - mean_dim) ** 2 for d in dims) / (len(dims) - 1)
        dim_std = math.sqrt(dim_variance)
        # Normalize: low relative std = high score
        relative_dim_std = dim_std / (mean_dim + 1e-6)
        dimension_score = max(0.0, 1.0 - relative_dim_std / 0.5)
    else:
        dimension_score = 0.5

    # Overall score: weighted combination
    # Ricci consistency is most important (50%), stability (30%), dimension (20%)
    overall = 0.5 * consistency_score + 0.3 * stability_score + 0.2 * dimension_score

    geometry.curvature_compatibility = overall
    geometry.curvature_compatibility_details = {
        "consistency_score": consistency_score,
        "stability_score": stability_score,
        "dimension_score": dimension_score,
        "mean_ricci": sum(ricci) / len(ricci),
        "negative_ratio": negative_ratio,
    }

    logger.debug(
        "Curvature compatibility: %.3f (consistency=%.2f, stability=%.2f, dimension=%.2f)",
        overall,
        consistency_score,
        stability_score,
        dimension_score,
    )
