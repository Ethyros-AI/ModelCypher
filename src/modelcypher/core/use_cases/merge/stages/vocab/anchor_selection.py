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

from modelcypher.core.domain.geometry.alignment_diagnostic import AlignmentSignal
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    regularization_epsilon,
)

from .matrix_ops import _dynamic_condition_threshold, _matrix_rank_for_alignment


def _compute_anchor_weights(signal: AlignmentSignal | None) -> list[float] | None:
    if signal is None:
        return None
    divergences = signal.anchor_divergence
    if not divergences:
        return None
    mean_div = signal.metadata.get("mean_divergence", 0.0)
    if mean_div <= 0.0:
        return [1.0 for _ in divergences]

    weights = [float(mean_div) / (float(d) + 1e-12) for d in divergences]
    mean_weight = sum(weights) / len(weights) if weights else 1.0
    if mean_weight > 0:
        weights = [w / mean_weight for w in weights]

    balance_ratio = max(1.0, float(signal.metadata.get("balance_ratio", 1.0)))
    min_weight = 1.0 / balance_ratio
    max_weight = balance_ratio
    weights = [min(max(w, min_weight), max_weight) for w in weights]
    return weights


def _apply_anchor_weights(
    matrix: "object",
    anchor_weights: list[float] | None,
    backend: "object",
) -> "object":
    if anchor_weights is None:
        return matrix
    if matrix.shape[0] != len(anchor_weights):
        return matrix
    weights = backend.array(anchor_weights)
    weights = backend.reshape(weights, (-1, 1))
    scaled = matrix * backend.sqrt(weights)
    backend.eval(scaled)
    return scaled


def _uniform_subset(values: list[int], max_count: int) -> list[int]:
    if max_count <= 0:
        return []
    if len(values) <= max_count:
        return values
    step = len(values) / float(max_count)
    selected = []
    for idx in range(max_count):
        pos = int(idx * step)
        selected.append(values[min(pos, len(values) - 1)])
    return selected


def _select_coverage_indices(
    points: "object",
    max_count: int,
    backend: "object",
    k_neighbors: int | None = None,
) -> tuple[list[int], dict[str, float]]:
    n = int(points.shape[0])
    if max_count <= 0 or n <= max_count:
        return list(range(n)), {"coverage_applied": 0.0}

    from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

    k_neighbors = k_neighbors if k_neighbors is not None else min(10, n - 1)
    k_neighbors = max(1, min(int(k_neighbors), n - 1))

    # Seed with the largest-norm anchor to maximize initial coverage.
    norms = backend.norm(points, axis=1)
    backend.eval(norms)
    seed_idx = int(backend.to_numpy(backend.argmax(norms)))

    rg = RiemannianGeometry(backend)
    fps_result = rg.farthest_point_sampling(
        points,
        n_samples=max_count,
        seed_idx=seed_idx,
        k_neighbors=k_neighbors,
    )

    return fps_result.selected_indices, {
        "coverage_applied": 1.0,
        "coverage_radius": float(fps_result.coverage_radius),
        "k_neighbors": float(k_neighbors),
    }


def _select_full_rank_indices(
    points: "object",
    max_count: int,
    backend: "object",
) -> tuple[list[int], dict[str, float]]:
    mean = backend.mean(points, axis=0, keepdims=True)
    centered = points - mean
    backend.eval(centered)
    n = int(points.shape[0])
    if max_count <= 0 or n == 0:
        return [], {"rank": 0.0, "selected_count": 0.0}
    if n <= max_count:
        rank = _matrix_rank_for_alignment(centered, backend)
        return list(range(n)), {"rank": float(rank), "selected_count": float(n)}

    norms = backend.norm(centered, axis=1)
    backend.eval(norms)
    norm_list = backend.to_numpy(norms).tolist()
    ranked = sorted(range(n), key=lambda idx: norm_list[idx], reverse=True)

    # Dtype-derived epsilon for rank-deficiency detection
    eps = regularization_epsilon(backend, points)
    selected: list[int] = []
    basis: list["object"] = []

    for idx in ranked:
        vec = centered[idx]
        if basis:
            basis_matrix = backend.stack(basis, axis=0)
            vec_col = backend.reshape(vec, (-1, 1))
            proj_coeffs = backend.matmul(basis_matrix, vec_col)
            proj = backend.matmul(backend.transpose(basis_matrix), proj_coeffs)
            residual = vec_col - proj
            res_norm = backend.norm(residual)
            backend.eval(res_norm)
            if float(backend.to_numpy(res_norm)) <= eps:
                continue
            vec = backend.reshape(residual / res_norm, (-1,))
        else:
            res_norm = backend.norm(vec)
            backend.eval(res_norm)
            if float(backend.to_numpy(res_norm)) <= eps:
                continue
            vec = vec / res_norm

        basis.append(vec)
        selected.append(idx)
        if len(selected) >= max_count:
            break

    return selected, {"rank": float(len(basis)), "selected_count": float(len(selected))}


def _select_shared_full_rank_indices(
    source_points: "object",
    target_points: "object",
    max_count: int,
    backend: "object",
    *,
    center: bool = True,
) -> tuple[list[int], dict[str, float]]:
    """Select indices where BOTH source and target are linearly independent.

    CRITICAL: Always filter for linear independence. CKA=1.0 requires full-rank
    matrices for the closed-form solve F = A_s^T (A_s A_s^T)^-1 A_t.
    If we pass rank-deficient matrices, the support-space inverse loses information
    and CKA < 1.0.
    """
    source_data = source_points
    target_data = target_points
    if center:
        source_data = source_points - backend.mean(source_points, axis=0, keepdims=True)
        target_data = target_points - backend.mean(target_points, axis=0, keepdims=True)
    backend.eval(source_data, target_data)
    n = int(source_points.shape[0])
    if max_count <= 0 or n == 0:
        return [], {"rank_source": 0.0, "rank_target": 0.0, "selected_count": 0.0}

    # ALWAYS filter for linear independence - never skip this step!
    # The old code had: if n <= max_count: return all indices
    # This was WRONG - it passed rank-deficient matrices to GramAligner

    combined = backend.concatenate([source_data, target_data], axis=1)
    norms = backend.norm(combined, axis=1)
    backend.eval(norms)
    norm_list = backend.to_numpy(norms).tolist()
    ranked = sorted(range(n), key=lambda idx: norm_list[idx], reverse=True)

    # Dtype-derived epsilon for rank-deficiency detection
    eps = division_epsilon(backend, combined)

    def _orthonormalize(
        vec: "object",
        basis: list["object"],
    ) -> tuple[bool, "object"]:
        if not basis:
            res_norm = backend.norm(vec)
            backend.eval(res_norm)
            if float(backend.to_numpy(res_norm)) <= eps:
                return False, vec
            return True, vec / res_norm

        basis_matrix = backend.stack(basis, axis=0)
        vec_col = backend.reshape(vec, (-1, 1))
        proj_coeffs = backend.matmul(basis_matrix, vec_col)
        proj = backend.matmul(backend.transpose(basis_matrix), proj_coeffs)
        residual = vec_col - proj
        res_norm = backend.norm(residual)
        backend.eval(res_norm)
        if float(backend.to_numpy(res_norm)) <= eps:
            return False, vec
        return True, backend.reshape(residual / res_norm, (-1,))

    selected: list[int] = []
    basis_src: list["object"] = []
    basis_tgt: list["object"] = []

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

    def _condition_number(matrix: "object") -> float:
        gram = backend.matmul(matrix, backend.transpose(matrix))
        # Cast to float32 for eigendecomposition (MLX doesn't support bfloat16 for eigh)
        gram_dtype = str(backend.dtype(gram))
        if "bfloat16" in gram_dtype:
            gram_f32 = backend.astype(gram, "float32")
            backend.eval(gram_f32)
            eigvals, _ = backend.eigh(gram_f32)
        else:
            eigvals, _ = backend.eigh(gram)
        backend.eval(eigvals)
        values = [float(v) for v in backend.to_numpy(eigvals).tolist() if float(v) > eps]
        if not values:
            return float("inf")
        return (max(values) / min(values)) ** 0.5

    max_condition = _dynamic_condition_threshold(combined, backend)
    while selected and len(selected) > 2:
        idx_arr = backend.array(selected)
        src_sel = backend.take(source_points, idx_arr, axis=0)
        tgt_sel = backend.take(target_points, idx_arr, axis=0)
        backend.eval(src_sel, tgt_sel)
        cond_src = _condition_number(src_sel)
        cond_tgt = _condition_number(tgt_sel)
        if cond_src <= max_condition and cond_tgt <= max_condition:
            break
        drop_count = max(1, len(selected) // 10)
        selected = selected[:-drop_count]

    # CRITICAL: Verify numerical rank via SVD and drop anchors until rank = n_selected
    # Use dtype-derived threshold: max_s * max(m,n) * machine_epsilon
    def _numerical_rank_svd(matrix: "object") -> int:
        """Compute numerical rank via SVD with dtype-derived threshold."""
        try:
            _, s, _ = backend.svd(matrix, full_matrices=False)
            backend.eval(s)
            s_vals = backend.to_numpy(s)
            if len(s_vals) == 0:
                return 0
            max_s = float(s_vals[0])
            eps_matrix = machine_epsilon(backend, matrix)
            if max_s < eps_matrix:
                return 0
            # Standard numerical rank threshold: max_s * max(m,n) * eps
            max_dim = max(matrix.shape[0], matrix.shape[1])
            threshold = max_s * max_dim * eps_matrix
            return int(sum(1 for sv in s_vals if float(sv) > threshold))
        except Exception:
            return len(selected)  # Fallback to full count if SVD fails

    # Drop anchors until BOTH matrices have numerical rank >= n_selected
    while selected and len(selected) > 2:
        idx_arr = backend.array(selected)
        src_sel = backend.take(source_points, idx_arr, axis=0)
        tgt_sel = backend.take(target_points, idx_arr, axis=0)
        backend.eval(src_sel, tgt_sel)

        rank_src = _numerical_rank_svd(src_sel)
        rank_tgt = _numerical_rank_svd(tgt_sel)
        min_rank = min(rank_src, rank_tgt)

        if min_rank >= len(selected):
            # Both matrices are full-rank for the closed-form solve
            break

        # Drop to match numerical rank (cannot exceed actual rank)
        if min_rank < len(selected):
            # Drop extra anchors - keep only min_rank indices
            drop_to = max(2, min_rank)
            if drop_to < len(selected):
                selected = selected[:drop_to]
        else:
            break

    cond_src = float("inf")
    cond_tgt = float("inf")
    rank_src_final = 0
    rank_tgt_final = 0
    if selected:
        idx_arr = backend.array(selected)
        src_sel = backend.take(source_points, idx_arr, axis=0)
        tgt_sel = backend.take(target_points, idx_arr, axis=0)
        backend.eval(src_sel, tgt_sel)
        cond_src = float(_condition_number(src_sel))
        cond_tgt = float(_condition_number(tgt_sel))
        rank_src_final = _numerical_rank_svd(src_sel)
        rank_tgt_final = _numerical_rank_svd(tgt_sel)

    return selected, {
        "rank_source": float(len(basis_src)),
        "rank_target": float(len(basis_tgt)),
        "selected_count": float(len(selected)),
        "cond_source": cond_src,
        "cond_target": cond_tgt,
        "svd_rank_source": float(rank_src_final),
        "svd_rank_target": float(rank_tgt_final),
    }


def _balanced_anchor_subset(anchor_ids: list[str], max_count: int) -> list[str]:
    if max_count <= 0:
        return []
    if len(anchor_ids) <= max_count:
        return anchor_ids

    buckets: dict[str, list[str]] = {}
    order: list[str] = []
    for anchor_id in anchor_ids:
        probe_id = anchor_id.split(":", 1)[0]
        if probe_id not in buckets:
            buckets[probe_id] = []
            order.append(probe_id)
        buckets[probe_id].append(anchor_id)

    selected: list[str] = []
    round_idx = 0
    while len(selected) < max_count:
        progressed = False
        for probe_id in order:
            bucket = buckets[probe_id]
            if round_idx < len(bucket):
                selected.append(bucket[round_idx])
                progressed = True
                if len(selected) >= max_count:
                    break
        if not progressed:
            break
        round_idx += 1

    return selected
