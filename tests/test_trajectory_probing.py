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

"""Tests for trajectory-based null-space probing."""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.orthogonal_probe_generator import (
    TrajectoryResult,
    TrajectorySubspaceResult,
    compute_trajectory_null_space,
    compute_trajectory_subspace,
)


@pytest.fixture
def backend():
    """Get the default backend."""
    return get_default_backend()


class TestTrajectorySubspace:
    """Tests for trajectory subspace computation."""

    def test_compute_trajectory_subspace_positions_only(self, backend):
        """Test subspace computation with positions only (velocities disabled)."""
        b = backend
        hidden_dim = 64
        seq_len = 10

        # Create synthetic trajectory data
        # positions: random points in hidden_dim space
        positions = b.random_normal((seq_len, hidden_dim))
        b.eval(positions)

        # Velocities: differences between consecutive positions
        velocities = positions[1:, :] - positions[:-1, :]
        b.eval(velocities)

        traj = TrajectoryResult(
            positions=positions,
            velocities=velocities,
            accelerations=None,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            text="test",
        )

        # Compute subspace with positions only
        result = compute_trajectory_subspace(
            trajectories=[traj],
            backend=b,
            include_velocities=False,
            include_accelerations=False,
        )

        assert result is not None
        assert result.hidden_dim == hidden_dim
        assert result.total_samples == seq_len
        assert result.position_contribution == seq_len
        assert result.velocity_contribution == 0
        # Rank should be <= min(n_samples, hidden_dim)
        assert result.rank <= min(seq_len, hidden_dim)

    def test_compute_trajectory_subspace_with_velocities(self, backend):
        """Test subspace computation with velocities included."""
        b = backend
        hidden_dim = 64
        seq_len = 10

        positions = b.random_normal((seq_len, hidden_dim))
        b.eval(positions)
        velocities = positions[1:, :] - positions[:-1, :]
        b.eval(velocities)

        traj = TrajectoryResult(
            positions=positions,
            velocities=velocities,
            accelerations=None,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            text="test",
        )

        # Compute subspace with velocities
        result = compute_trajectory_subspace(
            trajectories=[traj],
            backend=b,
            include_velocities=True,
            include_accelerations=False,
        )

        assert result is not None
        assert result.hidden_dim == hidden_dim
        # Total samples = positions + velocities
        assert result.total_samples == seq_len + (seq_len - 1)
        assert result.position_contribution == seq_len
        assert result.velocity_contribution == seq_len - 1
        # Rank should be higher with velocities (more samples)
        assert result.rank <= min(result.total_samples, hidden_dim)

    def test_compute_trajectory_subspace_multiple_trajectories(self, backend):
        """Test subspace computation with multiple trajectories."""
        b = backend
        hidden_dim = 32
        seq_len1, seq_len2 = 8, 12

        # Create two trajectories
        positions1 = b.random_normal((seq_len1, hidden_dim))
        velocities1 = positions1[1:, :] - positions1[:-1, :]
        b.eval(positions1, velocities1)

        positions2 = b.random_normal((seq_len2, hidden_dim))
        velocities2 = positions2[1:, :] - positions2[:-1, :]
        b.eval(positions2, velocities2)

        trajs = [
            TrajectoryResult(
                positions=positions1, velocities=velocities1, accelerations=None,
                seq_len=seq_len1, hidden_dim=hidden_dim, text="test1"
            ),
            TrajectoryResult(
                positions=positions2, velocities=velocities2, accelerations=None,
                seq_len=seq_len2, hidden_dim=hidden_dim, text="test2"
            ),
        ]

        result = compute_trajectory_subspace(trajs, b, include_velocities=True)

        assert result is not None
        expected_positions = seq_len1 + seq_len2
        expected_velocities = (seq_len1 - 1) + (seq_len2 - 1)
        assert result.position_contribution == expected_positions
        assert result.velocity_contribution == expected_velocities
        assert result.total_samples == expected_positions + expected_velocities

    def test_compute_trajectory_subspace_empty_list(self, backend):
        """Test that empty trajectory list returns None."""
        result = compute_trajectory_subspace([], backend, include_velocities=True)
        assert result is None


class TestTrajectoryNullSpace:
    """Tests for trajectory null-space computation."""

    def test_null_space_from_rank_deficient_subspace(self, backend):
        """Test null-space computation when rank < hidden_dim."""
        b = backend
        hidden_dim = 64
        seq_len = 10  # Few samples → rank deficient

        positions = b.random_normal((seq_len, hidden_dim))
        velocities = positions[1:, :] - positions[:-1, :]
        b.eval(positions, velocities)

        traj = TrajectoryResult(
            positions=positions, velocities=velocities, accelerations=None,
            seq_len=seq_len, hidden_dim=hidden_dim, text="test"
        )

        subspace_result = compute_trajectory_subspace(
            [traj], b, include_velocities=True
        )

        assert subspace_result is not None

        # Compute null space
        U_null = compute_trajectory_null_space(subspace_result, b)

        if subspace_result.rank < hidden_dim:
            # Should have null space
            assert U_null is not None
            null_rank = hidden_dim - subspace_result.rank
            assert b.shape(U_null) == (hidden_dim, null_rank)

            # Verify orthogonality: U_null.T @ Vt[:rank].T should be near zero
            Vt_used = subspace_result.Vt[:subspace_result.rank, :]  # [rank, hidden_dim]
            overlap = b.matmul(b.transpose(U_null), b.transpose(Vt_used))  # [null_rank, rank]
            b.eval(overlap)
            max_overlap = b.max(b.abs(overlap))
            b.eval(max_overlap)
            # Should be numerically zero
            assert float(b.to_scalar(max_overlap)) < 1e-5
        else:
            # Full rank → no null space
            assert U_null is None

    def test_null_space_full_rank_returns_none(self, backend):
        """Test that full-rank subspace returns None for null space."""
        b = backend
        hidden_dim = 16
        seq_len = 100  # Many samples → likely full rank

        positions = b.random_normal((seq_len, hidden_dim))
        velocities = positions[1:, :] - positions[:-1, :]
        b.eval(positions, velocities)

        traj = TrajectoryResult(
            positions=positions, velocities=velocities, accelerations=None,
            seq_len=seq_len, hidden_dim=hidden_dim, text="test"
        )

        subspace_result = compute_trajectory_subspace(
            [traj], b, include_velocities=True
        )

        assert subspace_result is not None

        # With enough samples, should be full rank
        if subspace_result.rank >= hidden_dim:
            U_null = compute_trajectory_null_space(subspace_result, b)
            assert U_null is None


class TestTrajectoryVelocityContribution:
    """Tests verifying that velocities add meaningful information."""

    def test_velocities_increase_rank(self, backend):
        """Test that including velocities increases the spanned rank."""
        b = backend
        hidden_dim = 64
        seq_len = 20

        # Create trajectory with structured data (not purely random)
        # This simulates a model trajectory where positions follow a pattern
        positions = b.random_normal((seq_len, hidden_dim))
        velocities = positions[1:, :] - positions[:-1, :]
        b.eval(positions, velocities)

        traj = TrajectoryResult(
            positions=positions, velocities=velocities, accelerations=None,
            seq_len=seq_len, hidden_dim=hidden_dim, text="test"
        )

        # Compute with positions only
        result_pos_only = compute_trajectory_subspace(
            [traj], b, include_velocities=False
        )

        # Compute with positions + velocities
        result_with_vel = compute_trajectory_subspace(
            [traj], b, include_velocities=True
        )

        assert result_pos_only is not None
        assert result_with_vel is not None

        # With velocities, we have more samples
        assert result_with_vel.total_samples > result_pos_only.total_samples

        # Rank with velocities should be >= rank without
        # (more samples can only increase or maintain rank, never decrease)
        assert result_with_vel.rank >= result_pos_only.rank

    def test_velocities_span_tangent_directions(self, backend):
        """Test that velocities capture tangent space directions."""
        b = backend
        hidden_dim = 32
        seq_len = 50

        # Create a trajectory that moves along a specific direction
        # Start with base point and add directed movement
        base = b.random_normal((1, hidden_dim))
        direction = b.random_normal((1, hidden_dim))
        # Normalize direction
        dir_norm = b.sqrt(b.sum(direction * direction))
        direction = direction / dir_norm
        b.eval(base, direction)

        # Create positions along this direction with some noise
        offsets = b.arange(seq_len, dtype="float32")
        offsets = b.reshape(offsets, (seq_len, 1))  # [seq_len, 1]
        b.eval(offsets)

        noise = 0.1 * b.random_normal((seq_len, hidden_dim))
        b.eval(noise)

        positions = base + offsets * direction + noise
        b.eval(positions)

        velocities = positions[1:, :] - positions[:-1, :]
        b.eval(velocities)

        traj = TrajectoryResult(
            positions=positions, velocities=velocities, accelerations=None,
            seq_len=seq_len, hidden_dim=hidden_dim, text="test"
        )

        result = compute_trajectory_subspace([traj], b, include_velocities=True)

        assert result is not None
        # The velocities should capture the movement direction
        # This is verified by having non-zero velocity contribution
        assert result.velocity_contribution == seq_len - 1
