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

"""Shared-manifold diagnostics, model-diff bases, and transfer harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.alignment_validation import (
    AlignmentGeneralizationReport,
    _even_odd_split,
    alignment_generalization_report,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    invariant_alignment,
)
from modelcypher.core.domain.geometry.rank_selection import (
    select_full_rank_indices,
    select_shared_full_rank_indices,
)
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms
from modelcypher.core.domain.geometry.shared_subspace_projector import (
    SharedSubspaceProjector,
)
from modelcypher.core.domain.geometry.transplant import compute_transplant_delta

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class SharedManifoldResidual:
    probe_id: str
    domain: str | None
    residual_norm: float
    residual_relative: float


@dataclass(frozen=True)
class SharedManifoldReport:
    alignment: AlignmentGeneralizationReport
    residual_mean: float
    residual_max: float
    residual_std: float
    residuals: list[SharedManifoldResidual]
    sorted_probe_ids: list[str]


@dataclass(frozen=True)
class DiffBasis:
    rank: int
    basis_vectors: "Array"  # [d, k]
    singular_values: "Array"  # [k]
    explained_variance_ratio: float


@dataclass(frozen=True)
class DiffTransferReport:
    merged_weight: "Array"
    shared_indices: list[int]
    diff_indices: list[int]
    diff_residual_mean_before: float
    diff_residual_mean_after: float
    boundary_max_relative_diff: float
    boundary_mean_relative_diff: float
    boundary_tolerance: float
    preserved_fraction: float
    projection_loss: float


def derive_alignment_indices(
    source: "Array",
    target: "Array",
    backend: "Backend",
    max_train: int | None = None,
) -> tuple[list[int], list[int]]:
    n = int(source.shape[0])
    if n != int(target.shape[0]):
        raise ValueError("Source and target must have the same sample count.")
    if n < 4:
        return _even_odd_split(list(range(n)))

    d_source = int(source.shape[1])
    d_target = int(target.shape[1])
    if max_train is None:
        max_train = min(n, min(d_source, d_target))

    train = select_shared_full_rank_indices(
        source,
        target,
        max_count=max_train,
        backend=backend,
    )
    holdout = [i for i in range(n) if i not in set(train)]

    if len(train) < 2 or len(holdout) < 2:
        return _even_odd_split(list(range(n)))
    return train, holdout


def compute_alignment_transform(
    source: "Array",
    target: "Array",
    train_indices: Sequence[int],
    backend: "Backend",
) -> "Array":
    train_idx = backend.array(list(train_indices), dtype="int32")
    source_train = backend.take(source, train_idx, axis=0)
    target_train = backend.take(target, train_idx, axis=0)
    backend.eval(source_train, target_train)
    transform = invariant_alignment(backend, source_train, target_train)
    backend.eval(transform)
    return transform


def compute_residual_matrix(
    source: "Array",
    target: "Array",
    transform: "Array",
    backend: "Backend",
) -> "Array":
    aligned = backend.matmul(source, transform)
    residuals = aligned - target
    backend.eval(aligned, residuals)
    return residuals


def compute_residual_norms(
    residuals: "Array",
    targets: "Array",
    backend: "Backend",
) -> tuple[list[float], list[float]]:
    residual_norms_arr = geodesic_norms(residuals, backend)
    target_norms_arr = geodesic_norms(targets, backend)
    backend.eval(residual_norms_arr, target_norms_arr)

    eps = division_epsilon(backend, targets)
    denom = backend.maximum(target_norms_arr, backend.full(target_norms_arr.shape, eps))
    rel_arr = residual_norms_arr / denom
    backend.eval(rel_arr)

    residual_norms = [float(x) for x in backend.tolist(residual_norms_arr)]
    residual_relative = [float(x) for x in backend.tolist(rel_arr)]
    return residual_norms, residual_relative


def compute_shared_manifold_report(
    source: "Array",
    target: "Array",
    probe_ids: Sequence[str],
    probe_domains: Sequence[str] | None = None,
    train_indices: Sequence[int] | None = None,
    holdout_indices: Sequence[int] | None = None,
    backend: "Backend | None" = None,
) -> SharedManifoldReport:
    b = backend or get_default_backend()
    source_arr = b.array(source)
    target_arr = b.array(target)
    b.eval(source_arr, target_arr)

    if len(probe_ids) != int(source_arr.shape[0]):
        raise ValueError("probe_ids must match the number of samples.")

    if probe_domains is not None and len(probe_domains) != len(probe_ids):
        raise ValueError("probe_domains must match probe_ids length.")

    if train_indices is None or holdout_indices is None:
        train_idx, holdout_idx = derive_alignment_indices(source_arr, target_arr, b)
    else:
        train_idx = list(train_indices)
        holdout_idx = list(holdout_indices)

    alignment = alignment_generalization_report(
        source_arr,
        target_arr,
        train_indices=train_idx,
        holdout_indices=holdout_idx,
        backend=b,
    )

    transform = compute_alignment_transform(source_arr, target_arr, train_idx, b)
    residuals = compute_residual_matrix(source_arr, target_arr, transform, b)
    residual_norms, residual_relative = compute_residual_norms(residuals, target_arr, b)

    if residual_norms:
        res_arr = b.array(residual_norms)
        b.eval(res_arr)
        mean_arr = b.mean(res_arr)
        max_arr = b.max(res_arr)
        b.eval(mean_arr, max_arr)
        mean_val = float(b.to_scalar(mean_arr))
        max_val = float(b.to_scalar(max_arr))

        diff = res_arr - mean_arr
        var_arr = b.mean(diff * diff)
        std_arr = b.sqrt(var_arr)
        b.eval(std_arr)
        std_val = float(b.to_scalar(std_arr))
    else:
        mean_val = 0.0
        max_val = 0.0
        std_val = 0.0

    residual_items: list[SharedManifoldResidual] = []
    for idx, probe_id in enumerate(probe_ids):
        domain = probe_domains[idx] if probe_domains is not None else None
        residual_items.append(
            SharedManifoldResidual(
                probe_id=str(probe_id),
                domain=domain,
                residual_norm=residual_norms[idx],
                residual_relative=residual_relative[idx],
            )
        )

    sorted_probe_ids = [
        item.probe_id
        for item in sorted(residual_items, key=lambda r: r.residual_norm, reverse=True)
    ]

    return SharedManifoldReport(
        alignment=alignment,
        residual_mean=mean_val,
        residual_max=max_val,
        residual_std=std_val,
        residuals=residual_items,
        sorted_probe_ids=sorted_probe_ids,
    )


def compute_diff_basis(
    residuals: "Array",
    backend: "Backend | None" = None,
) -> DiffBasis:
    from modelcypher.core.domain.geometry.numerical_stability import geodesic_svd

    b = backend or get_default_backend()
    residuals_arr = b.array(residuals)
    b.eval(residuals_arr)

    u, s, v_t = geodesic_svd(b, residuals_arr)
    b.eval(s, v_t)

    variances = s * s
    b.eval(variances)
    k = SharedSubspaceProjector._select_component_count(variances, None, backend=b)
    k = min(k, int(v_t.shape[0]))
    if k <= 0:
        k = 1

    basis = b.transpose(v_t)[:, :k]
    b.eval(basis)

    total_var_arr = b.sum(variances)
    kept_var_arr = b.sum(variances[:k])
    b.eval(total_var_arr, kept_var_arr)
    total_var = float(b.to_scalar(total_var_arr))
    kept_var = float(b.to_scalar(kept_var_arr))
    if total_var <= 0.0:
        total_var = 1.0
    ratio = kept_var / total_var

    return DiffBasis(
        rank=int(k),
        basis_vectors=basis,
        singular_values=s[:k],
        explained_variance_ratio=float(ratio),
    )


def derive_diff_indices(
    residuals: "Array",
    backend: "Backend",
    max_count: int | None = None,
) -> tuple[list[int], list[int]]:
    n = int(residuals.shape[0])
    if n == 0:
        return [], []
    if max_count is None:
        max_count = min(n, int(residuals.shape[1]))

    diff_indices = select_full_rank_indices(
        residuals,
        max_count=max_count,
        backend=backend,
        center=True,
    )
    diff_set = set(diff_indices)
    shared_indices = [i for i in range(n) if i not in diff_set]
    if len(shared_indices) < 2 or len(diff_indices) < 2:
        diff_indices, shared_indices = _even_odd_split(list(range(n)))
    return diff_indices, shared_indices


def compute_diff_transfer(
    target_weight: "Array",
    input_activations: "Array",
    source_outputs: "Array",
    target_outputs: "Array",
    train_indices: Sequence[int] | None = None,
    holdout_indices: Sequence[int] | None = None,
    diff_indices: Sequence[int] | None = None,
    shared_indices: Sequence[int] | None = None,
    backend: "Backend | None" = None,
) -> DiffTransferReport:
    b = backend or get_default_backend()
    target_w = b.array(target_weight)
    inputs = b.array(input_activations)
    source_out = b.array(source_outputs)
    target_out = b.array(target_outputs)
    b.eval(target_w, inputs, source_out, target_out)

    if train_indices is None or holdout_indices is None:
        train_idx, holdout_idx = derive_alignment_indices(source_out, target_out, b)
    else:
        train_idx = list(train_indices)
        holdout_idx = list(holdout_indices)
        if len(train_idx) < 2 or len(holdout_idx) < 2:
            raise ValueError("Need at least 2 samples in both train and holdout splits.")

    transform = compute_alignment_transform(source_out, target_out, train_idx, b)
    residuals = compute_residual_matrix(source_out, target_out, transform, b)

    if diff_indices is None and shared_indices is None:
        diff_idx_list, shared_idx_list = derive_diff_indices(residuals, b)
    else:
        diff_idx_list = list(diff_indices or [])
        if shared_indices is None:
            diff_set = set(diff_idx_list)
            shared_idx_list = [i for i in range(int(residuals.shape[0])) if i not in diff_set]
        else:
            shared_idx_list = list(shared_indices)

        if len(diff_idx_list) < 2 or len(shared_idx_list) < 2:
            raise ValueError("Need at least 2 samples in both diff and shared splits.")

    diff_idx_arr = b.array(diff_idx_list, dtype="int32")
    shared_idx_arr = b.array(shared_idx_list, dtype="int32")
    core_inputs = b.take(inputs, diff_idx_arr, axis=0)
    core_delta = b.take(residuals, diff_idx_arr, axis=0)
    boundary_inputs = b.take(inputs, shared_idx_arr, axis=0)
    b.eval(core_inputs, core_delta, boundary_inputs)

    transplant = compute_transplant_delta(
        weight_target=target_w,
        activations_core=core_inputs,
        delta_activations=core_delta,
        boundary_activations=boundary_inputs,
        backend=b,
    )

    merged = b.array(transplant.merged_weight)
    b.eval(merged)
    outputs_before = b.matmul(inputs, b.transpose(target_w))
    outputs_after = b.matmul(inputs, b.transpose(merged))
    b.eval(outputs_before, outputs_after)

    core_before = b.take(outputs_before, diff_idx_arr, axis=0)
    core_after = b.take(outputs_after, diff_idx_arr, axis=0)
    core_target = b.take(target_out, diff_idx_arr, axis=0)
    core_source = b.take(source_out, diff_idx_arr, axis=0)
    b.eval(core_before, core_after, core_target, core_source)

    before_residuals = core_source - core_target
    after_residuals = core_after - core_target
    b.eval(before_residuals, after_residuals)

    before_norms = geodesic_norms(before_residuals, b)
    after_norms = geodesic_norms(after_residuals, b)
    b.eval(before_norms, after_norms)
    diff_before_mean = float(b.to_scalar(b.mean(before_norms)))
    diff_after_mean = float(b.to_scalar(b.mean(after_norms)))

    boundary_before = b.take(outputs_before, shared_idx_arr, axis=0)
    boundary_after = b.take(outputs_after, shared_idx_arr, axis=0)
    b.eval(boundary_before, boundary_after)

    boundary_residuals = boundary_after - boundary_before
    boundary_norms = geodesic_norms(boundary_before, b)
    boundary_residual_norms = geodesic_norms(boundary_residuals, b)
    b.eval(boundary_norms, boundary_residual_norms)

    eps = division_epsilon(b, boundary_before)
    denom = b.maximum(boundary_norms, b.full(boundary_norms.shape, eps))
    rel = boundary_residual_norms / denom
    b.eval(rel)
    boundary_max = float(b.to_scalar(b.max(rel))) if shared_idx_list else 0.0
    boundary_mean = float(b.to_scalar(b.mean(rel))) if shared_idx_list else 0.0

    return DiffTransferReport(
        merged_weight=merged,
        shared_indices=shared_idx_list,
        diff_indices=diff_idx_list,
        diff_residual_mean_before=diff_before_mean,
        diff_residual_mean_after=diff_after_mean,
        boundary_max_relative_diff=boundary_max,
        boundary_mean_relative_diff=boundary_mean,
        boundary_tolerance=float(division_epsilon(b, target_w)),
        preserved_fraction=float(transplant.preserved_fraction),
        projection_loss=float(transplant.projection_loss),
    )


__all__ = [
    "DiffBasis",
    "DiffTransferReport",
    "SharedManifoldReport",
    "SharedManifoldResidual",
    "compute_alignment_transform",
    "compute_diff_basis",
    "compute_diff_transfer",
    "compute_residual_matrix",
    "compute_residual_norms",
    "compute_shared_manifold_report",
    "derive_alignment_indices",
    "derive_diff_indices",
]
