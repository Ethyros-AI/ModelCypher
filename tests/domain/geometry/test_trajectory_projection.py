# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import pytest

from modelcypher.core.domain.geometry.null_space import VarianceNullSpaceResult
from modelcypher.core.domain.geometry.trajectory_analysis import (
    TrajectoryResult,
    TrajectorySubspaceResult,
)
from modelcypher.core.domain.geometry.trajectory_projection import (
    TrajectoryTangentResult,
    compute_trajectory_tangent_null_space,
    project_delta_to_trajectory_tangent,
    project_delta_to_variance_null_space,
)


def test_compute_trajectory_tangent_null_space_empty_and_full_rank(any_backend) -> None:
    b = any_backend

    assert compute_trajectory_tangent_null_space([], b) is None

    full_rank_subspace = TrajectorySubspaceResult(
        Vt=b.array([[1.0, 0.0], [0.0, 1.0]]),
        singular_values=b.array([1.0, 1.0]),
        rank=2,
        hidden_dim=2,
        total_samples=2,
        position_contribution=1,
        velocity_contribution=1,
    )
    traj = TrajectoryResult(
        positions=b.zeros((2, 2)),
        velocities=b.array([[1.0, 0.0]]),
        accelerations=None,
        seq_len=2,
        hidden_dim=2,
        text="x",
    )
    assert compute_trajectory_tangent_null_space([traj], b, subspace_result=full_rank_subspace) is None


def test_compute_trajectory_tangent_null_space_success(any_backend) -> None:
    b = any_backend

    subspace = TrajectorySubspaceResult(
        Vt=b.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        singular_values=b.array([2.0, 1.0, 0.5]),
        rank=1,
        hidden_dim=3,
        total_samples=3,
        position_contribution=1,
        velocity_contribution=2,
    )
    traj = TrajectoryResult(
        positions=b.zeros((3, 3)),
        velocities=b.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]),
        accelerations=None,
        seq_len=3,
        hidden_dim=3,
        text="trajectory",
    )

    result = compute_trajectory_tangent_null_space([traj], b, subspace_result=subspace)

    assert result is not None
    assert result.null_rank == 2
    assert 1 <= result.tangent_rank <= 2
    assert tuple(result.U_null.shape) == (3, 2)
    assert tuple(result.U_tangent.shape)[0] == 3
    assert 0.0 <= result.velocity_alignment <= 1.0


def test_project_delta_to_trajectory_tangent_vector_matrix_and_mismatch(any_backend) -> None:
    b = any_backend
    tangent_result = TrajectoryTangentResult(
        U_null=b.array([[1.0, 0.0], [0.0, 1.0]]),
        U_tangent=b.array([[1.0], [0.0]]),
        null_rank=2,
        tangent_rank=1,
        velocity_alignment=1.0,
        hidden_dim=2,
    )

    vec = b.array([3.0, 4.0])
    mat = b.array([[1.0, 2.0], [3.0, 4.0]])

    vec_proj = project_delta_to_trajectory_tangent(vec, tangent_result, b)
    mat_proj = project_delta_to_trajectory_tangent(mat, tangent_result, b)
    vec_full = project_delta_to_trajectory_tangent(vec, tangent_result, b, use_full_null=True)
    mismatch = project_delta_to_trajectory_tangent(b.array([1.0, 2.0, 3.0]), tangent_result, b)

    assert b.tolist(vec_proj) == pytest.approx([3.0, 0.0], abs=1e-6)
    assert b.tolist(mat_proj)[0] == pytest.approx([1.0, 0.0], abs=1e-6)
    assert b.tolist(mat_proj)[1] == pytest.approx([3.0, 0.0], abs=1e-6)
    assert b.tolist(vec_full) == pytest.approx([3.0, 4.0], abs=1e-6)
    assert b.tolist(mismatch) == pytest.approx([1.0, 2.0, 3.0], abs=1e-6)


def test_project_delta_to_variance_null_space_zero_projection_and_mismatch(any_backend) -> None:
    b = any_backend
    delta_vec = b.array([2.0, 5.0])
    delta_mat = b.array([[1.0, 2.0], [3.0, 4.0]])

    none_available = VarianceNullSpaceResult(
        utilized_basis=b.array([[1.0, 0.0], [0.0, 1.0]]),
        available_basis=b.zeros((2, 0)),
        utilized_rank=2,
        available_rank=0,
        eigenvalues=b.array([1.0, 0.0]),
        variance_threshold=0.0,
    )
    vec_zero = project_delta_to_variance_null_space(delta_vec, none_available, b)
    assert b.tolist(vec_zero) == pytest.approx([0.0, 0.0], abs=1e-6)

    one_available = VarianceNullSpaceResult(
        utilized_basis=b.array([[0.0], [1.0]]),
        available_basis=b.array([[1.0], [0.0]]),
        utilized_rank=1,
        available_rank=1,
        eigenvalues=b.array([1.0, 0.0]),
        variance_threshold=0.0,
    )
    vec_proj = project_delta_to_variance_null_space(delta_vec, one_available, b)
    mat_proj = project_delta_to_variance_null_space(delta_mat, one_available, b)
    mismatch = project_delta_to_variance_null_space(b.array([1.0, 2.0, 3.0]), one_available, b)

    assert b.tolist(vec_proj) == pytest.approx([2.0, 0.0], abs=1e-6)
    assert b.tolist(mat_proj)[0] == pytest.approx([1.0, 0.0], abs=1e-6)
    assert b.tolist(mat_proj)[1] == pytest.approx([3.0, 0.0], abs=1e-6)
    assert b.tolist(mismatch) == pytest.approx([1.0, 2.0, 3.0], abs=1e-6)

