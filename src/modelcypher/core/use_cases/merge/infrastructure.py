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
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.core.domain.geometry.vector_math import geodesic_norms

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


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
    n = int(points.shape[0])
    if n_anchors <= 0 or n <= n_anchors:
        return list(range(n))

    from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

    # k_neighbors=None triggers connectivity-based selection (geometry determines the value)
    norms = geodesic_norms(points, backend)
    backend.eval(norms)
    seed_idx_arr = backend.argmax(norms)
    backend.eval(seed_idx_arr)
    seed_idx = int(backend.to_scalar(seed_idx_arr))

    rg = RiemannianGeometry(backend)
    fps_result = rg.farthest_point_sampling(
        points,
        n_samples=n_anchors,
        seed_idx=seed_idx,
        # k_neighbors=None: connectivity-based selection (geometry determines k)
    )
    return fps_result.selected_indices


def select_shared_full_rank_indices(
    source_points: "Array",
    target_points: "Array",
    max_count: int,
    backend: "Backend",
    *,
    center: bool = True,
) -> list[int]:
    from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

    source_data = source_points
    target_data = target_points
    if center:
        source_data = source_points - backend.mean(source_points, axis=0, keepdims=True)
        target_data = target_points - backend.mean(target_points, axis=0, keepdims=True)
    backend.eval(source_data, target_data)

    n = int(source_points.shape[0])
    if max_count <= 0 or n == 0:
        return []
    if n <= max_count:
        return list(range(n))

    combined = backend.concatenate([source_data, target_data], axis=1)
    norms = geodesic_norms(combined, backend)
    backend.eval(norms)
    # Sort by norm descending using backend argsort on negated values
    neg_norms = -norms
    sorted_indices = backend.argsort(neg_norms)
    backend.eval(sorted_indices)
    ranked = [int(x) for x in backend.tolist(sorted_indices)]

    eps = machine_epsilon(backend, combined) * 100.0

    def _orthonormalize(
        vec: "Array",
        basis: list["Array"],
    ) -> tuple[bool, "Array"]:
        if not basis:
            res_norm_arr = geodesic_norms(backend.reshape(vec, (1, -1)), backend)
            backend.eval(res_norm_arr)
            res_norm = float(backend.to_scalar(res_norm_arr))
            if res_norm <= eps:
                return False, vec
            return True, vec / res_norm

        basis_matrix = backend.stack(basis, axis=0)
        vec_col = backend.reshape(vec, (-1, 1))
        proj_coeffs = backend.matmul(basis_matrix, vec_col)
        proj = backend.matmul(backend.transpose(basis_matrix), proj_coeffs)
        residual = vec_col - proj
        res_norm_arr = geodesic_norms(backend.reshape(residual, (1, -1)), backend)
        backend.eval(res_norm_arr)
        res_norm = float(backend.to_scalar(res_norm_arr))
        if res_norm <= eps:
            return False, vec
        return True, backend.reshape(residual / res_norm, (-1,))

    selected: list[int] = []
    basis_src: list["Array"] = []
    basis_tgt: list["Array"] = []

    for idx in ranked:
        vec_src = source_data[idx]
        vec_tgt = target_data[idx]
        ok_src, norm_src = _orthonormalize(vec_src, basis_src)
        ok_tgt, norm_tgt = _orthonormalize(vec_tgt, basis_tgt)
        if not (ok_src and ok_tgt):
            continue
        basis_src.append(norm_src)
        basis_tgt.append(norm_tgt)
        selected.append(idx)
        if len(selected) >= max_count:
            break

    return selected


def select_full_rank_indices(
    points: "Array",
    max_count: int,
    backend: "Backend",
    *,
    center: bool = True,
) -> list[int]:
    from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

    data = points
    if center:
        data = points - backend.mean(points, axis=0, keepdims=True)
        backend.eval(data)

    n = int(points.shape[0])
    if max_count <= 0 or n == 0:
        return []
    if n <= max_count:
        return list(range(n))

    norms = geodesic_norms(data, backend)
    backend.eval(norms)
    # Sort by norm descending using backend argsort on negated values
    neg_norms = -norms
    sorted_indices = backend.argsort(neg_norms)
    backend.eval(sorted_indices)
    ranked = [int(x) for x in backend.tolist(sorted_indices)]

    eps = machine_epsilon(backend, data) * 100.0
    selected: list[int] = []
    basis: list["Array"] = []

    for idx in ranked:
        vec = data[idx]
        if basis:
            basis_matrix = backend.stack(basis, axis=0)
            vec_col = backend.reshape(vec, (-1, 1))
            proj_coeffs = backend.matmul(basis_matrix, vec_col)
            proj = backend.matmul(backend.transpose(basis_matrix), proj_coeffs)
            residual = vec_col - proj
            res_norm_arr = geodesic_norms(backend.reshape(residual, (1, -1)), backend)
            backend.eval(res_norm_arr)
            res_norm = float(backend.to_scalar(res_norm_arr))
            if res_norm <= eps:
                continue
            vec = backend.reshape(residual / res_norm, (-1,))
        else:
            res_norm_arr = geodesic_norms(backend.reshape(vec, (1, -1)), backend)
            backend.eval(res_norm_arr)
            res_norm = float(backend.to_scalar(res_norm_arr))
            if res_norm <= eps:
                continue
            vec = vec / res_norm

        basis.append(vec)
        selected.append(idx)
        if len(selected) >= max_count:
            break

    return selected
