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

"""Tests for shared-manifold diagnostics and diff transfer utilities."""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms
from modelcypher.core.domain.geometry.shared_manifold import (
    compute_alignment_transform,
    compute_diff_basis,
    compute_diff_transfer,
    compute_residual_matrix,
    compute_residual_norms,
    compute_shared_manifold_report,
)
from modelcypher.core.domain.geometry.transplant import compute_transplant_delta


@pytest.fixture
def backend():
    return get_default_backend()


def _make_shared_diff_data(backend, n: int = 24, dim: int = 8, diff_count: int = 6):
    backend.random_seed(123)
    source = backend.random_normal((n, dim))
    transform = backend.random_normal((dim, dim))
    target = backend.matmul(source, transform)
    backend.eval(source, transform, target)

    diff_start = n - diff_count
    row_idx = backend.arange(n)
    diff_mask = backend.reshape(row_idx >= diff_start, (n, 1))
    diff_mask = backend.astype(diff_mask, "float32")
    delta = backend.random_normal((n, dim)) * 4.0
    delta = delta * diff_mask
    target = target + delta
    backend.eval(target)

    diff_indices = list(range(diff_start, n))
    shared_indices = list(range(0, diff_start))
    return source, target, diff_indices, shared_indices


def test_shared_manifold_report_separates_diff(backend):
    source, target, diff_indices, shared_indices = _make_shared_diff_data(backend)
    probe_ids = [f"probe-{i}" for i in range(int(source.shape[0]))]

    train_indices = shared_indices[::2]
    holdout_indices = shared_indices[1::2]

    report = compute_shared_manifold_report(
        source,
        target,
        probe_ids,
        train_indices=train_indices,
        holdout_indices=holdout_indices,
        backend=backend,
    )

    transform = compute_alignment_transform(source, target, train_indices, backend)
    residuals = compute_residual_matrix(source, target, transform, backend)
    residual_norms, residual_relative = compute_residual_norms(residuals, target, backend)

    res_arr = backend.array(residual_norms)
    backend.eval(res_arr)
    mean_arr = backend.mean(res_arr)
    max_arr = backend.max(res_arr)
    diff = res_arr - mean_arr
    var_arr = backend.mean(diff * diff)
    std_arr = backend.sqrt(var_arr)
    backend.eval(mean_arr, max_arr, std_arr)
    expected_mean = float(backend.to_scalar(mean_arr))
    expected_max = float(backend.to_scalar(max_arr))
    expected_std = float(backend.to_scalar(std_arr))

    expected_sorted = [
        probe_ids[idx]
        for idx, _ in sorted(
            enumerate(residual_norms), key=lambda pair: pair[1], reverse=True
        )
    ]

    eps = division_epsilon(backend, source) * float(source.shape[1])
    assert abs(report.residual_mean - expected_mean) <= eps
    assert abs(report.residual_max - expected_max) <= eps
    assert abs(report.residual_std - expected_std) <= eps
    assert report.sorted_probe_ids == expected_sorted

    for idx, item in enumerate(report.residuals):
        assert abs(item.residual_norm - residual_norms[idx]) <= eps
        assert abs(item.residual_relative - residual_relative[idx]) <= eps


def test_compute_diff_basis_shapes(backend):
    source, target, _diff_indices, shared_indices = _make_shared_diff_data(backend)
    train_indices = shared_indices[::2]

    transform = compute_alignment_transform(source, target, train_indices, backend)
    residuals = compute_residual_matrix(source, target, transform, backend)
    basis = compute_diff_basis(residuals, backend)

    dim = int(source.shape[1])
    assert basis.rank >= 1
    assert basis.basis_vectors.shape == (dim, basis.rank)
    assert basis.singular_values.shape[0] == basis.rank
    tol = division_epsilon(backend, basis.singular_values)
    assert 0.0 < basis.explained_variance_ratio <= 1.0 + tol


def test_diff_transfer_reduces_core_error(backend):
    backend.random_seed(321)
    n = 32
    in_dim = 8
    out_dim = 6
    diff_count = 10

    inputs = backend.random_normal((n, in_dim))
    target_weight = backend.random_normal((out_dim, in_dim))
    target_outputs = backend.matmul(inputs, backend.transpose(target_weight))
    backend.eval(inputs, target_weight, target_outputs)

    diff_start = n - diff_count
    row_idx = backend.arange(n)
    diff_mask = backend.reshape(row_idx >= diff_start, (n, 1))
    diff_mask = backend.astype(diff_mask, "float32")
    delta = backend.random_normal((n, out_dim)) * 5.0
    delta = delta * diff_mask
    source_outputs = target_outputs + delta
    backend.eval(source_outputs)

    diff_indices = list(range(diff_start, n))
    shared_indices = list(range(0, diff_start))
    train_indices = shared_indices[::2]
    holdout_indices = shared_indices[1::2]

    report = compute_diff_transfer(
        target_weight=target_weight,
        input_activations=inputs,
        source_outputs=source_outputs,
        target_outputs=target_outputs,
        train_indices=train_indices,
        holdout_indices=holdout_indices,
        diff_indices=diff_indices,
        shared_indices=shared_indices,
        backend=backend,
    )

    transform = compute_alignment_transform(source_outputs, target_outputs, train_indices, backend)
    residuals = compute_residual_matrix(source_outputs, target_outputs, transform, backend)

    diff_idx_arr = backend.array(diff_indices, dtype="int32")
    shared_idx_arr = backend.array(shared_indices, dtype="int32")
    core_inputs = backend.take(inputs, diff_idx_arr, axis=0)
    core_delta = backend.take(residuals, diff_idx_arr, axis=0)
    boundary_inputs = backend.take(inputs, shared_idx_arr, axis=0)
    backend.eval(core_inputs, core_delta, boundary_inputs)

    transplant = compute_transplant_delta(
        weight_target=target_weight,
        activations_core=core_inputs,
        delta_activations=core_delta,
        boundary_activations=boundary_inputs,
        backend=backend,
    )

    merged = backend.array(transplant.merged_weight)
    outputs_before = backend.matmul(inputs, backend.transpose(target_weight))
    outputs_after = backend.matmul(inputs, backend.transpose(merged))
    backend.eval(merged, outputs_before, outputs_after)

    core_before = backend.take(outputs_before, diff_idx_arr, axis=0)
    core_after = backend.take(outputs_after, diff_idx_arr, axis=0)
    core_target = backend.take(target_outputs, diff_idx_arr, axis=0)
    core_source = backend.take(source_outputs, diff_idx_arr, axis=0)
    backend.eval(core_before, core_after, core_target, core_source)

    before_residuals = core_source - core_target
    after_residuals = core_after - core_target
    backend.eval(before_residuals, after_residuals)

    before_norms = geodesic_norms(before_residuals, backend)
    after_norms = geodesic_norms(after_residuals, backend)
    backend.eval(before_norms, after_norms)
    expected_before_mean = float(backend.to_scalar(backend.mean(before_norms)))
    expected_after_mean = float(backend.to_scalar(backend.mean(after_norms)))

    boundary_before = backend.take(outputs_before, shared_idx_arr, axis=0)
    boundary_after = backend.take(outputs_after, shared_idx_arr, axis=0)
    backend.eval(boundary_before, boundary_after)

    boundary_residuals = boundary_after - boundary_before
    boundary_norms = geodesic_norms(boundary_before, backend)
    boundary_residual_norms = geodesic_norms(boundary_residuals, backend)
    backend.eval(boundary_norms, boundary_residual_norms)

    eps = division_epsilon(backend, boundary_before)
    denom = backend.maximum(boundary_norms, backend.full(boundary_norms.shape, eps))
    rel = boundary_residual_norms / denom
    backend.eval(rel)
    expected_boundary_max = float(backend.to_scalar(backend.max(rel)))
    expected_boundary_mean = float(backend.to_scalar(backend.mean(rel)))

    tol = division_epsilon(backend, target_weight) * float(out_dim)
    assert abs(report.diff_residual_mean_before - expected_before_mean) <= tol
    assert abs(report.diff_residual_mean_after - expected_after_mean) <= tol
    assert abs(report.boundary_max_relative_diff - expected_boundary_max) <= tol
    assert abs(report.boundary_mean_relative_diff - expected_boundary_mean) <= tol
    assert abs(report.boundary_tolerance - division_epsilon(backend, target_weight)) <= tol
