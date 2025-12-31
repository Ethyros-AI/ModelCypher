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

    logger.info(
        "MERGE GEOMETRY: %d layers, mean_intrinsic_dim=%.1f, mean_shared_dim=%.1f, CKA=%.4f, health=%s",
        len(layer_geoms),
        geometry.mean_intrinsic_dimension,
        geometry.mean_shared_dimension,
        geometry.overall_cka,
        geometry.overall_manifold_health,
    )
