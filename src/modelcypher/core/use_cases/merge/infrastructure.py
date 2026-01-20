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

"""Merge infrastructure utilities.

This module provides setup and anchor selection for merge operations.
Core rank selection functions are imported from the domain layer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

# Re-export rank selection from domain for backward compatibility
from modelcypher.core.domain.geometry.rank_selection import (
    select_full_rank_indices,
    select_shared_full_rank_indices,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)

# Explicit re-exports for type checkers
__all__ = [
    "setup_infrastructure",
    "select_anchor_indices_by_coverage",
    "select_full_rank_indices",
    "select_shared_full_rank_indices",
]


def setup_infrastructure() -> tuple[float, bool, Any | None]:
    """Set up geometry infrastructure settings."""
    # numerical_stability - compute data-driven epsilons
    # These functions compute appropriate epsilon based on dtype
    backend = get_default_backend()
    epsilon = machine_epsilon(backend, backend.array([0.0]))
    # SVD is NEVER disabled. We use geodesic_svd from numerical_stability.py
    # which computes SVD via power iteration - GPU-only, stable on all backends.
    avoid_svd = False

    # geometry_metrics_cache - available for caching
    try:
        from modelcypher.core.domain.geometry.geometry_metrics_cache import (
            GeometryMetricsCache,
        )

        metrics_cache = GeometryMetricsCache()
    except Exception:
        metrics_cache = None

    logger.debug("Infrastructure: epsilon=%e", epsilon)
    return epsilon, avoid_svd, metrics_cache


def select_anchor_indices_by_coverage(
    points: "Array",
    n_anchors: int,
    backend: "Backend",
) -> list[int]:
    """Select anchor indices using farthest point sampling for coverage.

    Args:
        points: Point cloud [n_samples, dim].
        n_anchors: Number of anchors to select.
        backend: Compute backend.

    Returns:
        List of selected anchor indices.
    """
    n = int(points.shape[0])
    if n_anchors <= 0 or n <= n_anchors:
        return list(range(n))

    from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

    rg = RiemannianGeometry(backend)
    fps_result = rg.farthest_point_sampling(
        points,
        n_samples=n_anchors,
    )
    return fps_result.selected_indices
