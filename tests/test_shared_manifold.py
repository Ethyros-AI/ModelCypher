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
from modelcypher.core.domain.geometry.shared_manifold import (
    compute_alignment_transform,
    compute_diff_basis,
    compute_diff_transfer,
    compute_residual_matrix,
    compute_shared_manifold_report,
)


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

    residuals = [r.residual_norm for r in report.residuals]
    diff_mean = sum(residuals[i] for i in diff_indices) / len(diff_indices)
    shared_mean = sum(residuals[i] for i in shared_indices) / len(shared_indices)

    eps = division_epsilon(backend, source) * float(source.shape[1])
    assert diff_mean > shared_mean + eps

    diff_probe_ids = {probe_ids[i] for i in diff_indices}
    assert report.sorted_probe_ids[0] in diff_probe_ids


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
    assert 0.0 < basis.explained_variance_ratio <= 1.0 + 1e-6


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

    eps = division_epsilon(backend, target_weight) * float(out_dim)
    assert report.diff_residual_mean_after <= report.diff_residual_mean_before - eps
    assert report.boundary_max_relative_diff <= report.boundary_tolerance + eps
