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

"""Tests for null-space constrained transplant (AlphaEdit-style).

Mathematical guarantees to verify:
    1. A_boundary @ W' = A_boundary @ W_target  (boundary preserved)
    2. Core functionality transplanted via null-space filtered delta
    3. preserved_fraction measures how much delta survived projection
"""

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.null_space_filter import NullSpaceFilterConfig
from modelcypher.core.domain.geometry.transplant import (
    CoreBoundaryPartition,
    TransplantDeltaResult,
    compute_transplant_delta,
    partition_core_boundary,
)


class TestPartitionCoreBoundary:
    """Tests for partition_core_boundary function."""

    def test_partition_empty_activations(self) -> None:
        """Empty activations should return empty partition."""
        backend = get_default_backend()
        activations = backend.zeros((0, 64))
        backend.eval(activations)

        result = partition_core_boundary(
            activations=activations,
            probe_ids=[],
            core_probe_ids=set(),
            backend=backend,
        )

        assert result.core_indices == []
        assert result.boundary_indices == []

    def test_partition_no_core_probes(self) -> None:
        """No core probes should return empty partition."""
        backend = get_default_backend()
        backend.random_seed(42)
        activations = backend.random_normal((10, 64))
        backend.eval(activations)

        result = partition_core_boundary(
            activations=activations,
            probe_ids=[f"probe_{i}" for i in range(10)],
            core_probe_ids=set(),  # empty core
            backend=backend,
        )

        assert result.core_indices == []
        assert result.boundary_indices == []

    def test_partition_finds_boundary_neighbors(self) -> None:
        """Boundary should include geodesic neighbors of core probes."""
        backend = get_default_backend()
        backend.random_seed(42)

        # Create activations where some probes are close together
        activations = backend.random_normal((10, 64))
        backend.eval(activations)

        probe_ids = [f"probe_{i}" for i in range(10)]
        core_probe_ids = {"probe_0", "probe_1"}  # first two are core

        result = partition_core_boundary(
            activations=activations,
            probe_ids=probe_ids,
            core_probe_ids=core_probe_ids,
            boundary_k=3,  # find 3 nearest neighbors per core
            backend=backend,
        )

        assert result.core_indices == [0, 1]
        assert len(result.boundary_indices) > 0
        # Boundary should not include core indices
        assert 0 not in result.boundary_indices
        assert 1 not in result.boundary_indices

    def test_partition_boundary_k_limits_neighbors(self) -> None:
        """boundary_k should limit number of neighbors per core probe."""
        backend = get_default_backend()
        backend.random_seed(42)
        activations = backend.random_normal((20, 64))
        backend.eval(activations)

        probe_ids = [f"probe_{i}" for i in range(20)]
        core_probe_ids = {"probe_5"}  # single core probe

        result = partition_core_boundary(
            activations=activations,
            probe_ids=probe_ids,
            core_probe_ids=core_probe_ids,
            boundary_k=5,
            backend=backend,
        )

        assert result.core_indices == [5]
        assert len(result.boundary_indices) == 5


class TestComputeTransplantDelta:
    """Tests for compute_transplant_delta function."""

    def test_boundary_output_preserved(self) -> None:
        """Core guarantee: A_boundary @ W' = A_boundary @ W_target."""
        backend = get_default_backend()
        backend.random_seed(42)

        # Create test data
        in_dim, out_dim = 64, 32
        n_core, n_boundary = 5, 10

        weight_target = backend.random_normal((out_dim, in_dim))
        weight_source = backend.random_normal((out_dim, in_dim))
        activations_core = backend.random_normal((n_core, in_dim))
        activations_boundary = backend.random_normal((n_boundary, in_dim))
        backend.eval(weight_target, weight_source, activations_core, activations_boundary)

        # Compute transplant
        config = NullSpaceFilterConfig(rank_threshold=1e-6)
        result = compute_transplant_delta(
            weight_target=weight_target,
            weight_source_aligned=weight_source,
            activations_core=activations_core,
            activations_boundary=activations_boundary,
            backend=backend,
            nullspace_config=config,
        )

        assert result.applied is True

        # Verify boundary preservation: A_boundary @ W' = A_boundary @ W_target
        # W is [out, in], so output = A @ W^T
        merged_weight = backend.array(result.merged_weight)
        backend.eval(merged_weight)

        output_before = backend.matmul(activations_boundary, backend.transpose(weight_target))
        output_after = backend.matmul(activations_boundary, backend.transpose(merged_weight))
        backend.eval(output_before, output_after)

        diff = backend.norm(output_after - output_before)
        backend.eval(diff)
        diff_val = float(backend.to_numpy(diff))

        # Boundary output should be preserved (within numerical tolerance)
        assert diff_val < 1e-4, f"Boundary not preserved: diff={diff_val}"

    def test_non_2d_weight_skipped(self) -> None:
        """Non-2D weights should be skipped (bias vectors, etc)."""
        backend = get_default_backend()
        backend.random_seed(42)

        weight_1d = backend.random_normal((64,))  # 1D bias
        weight_source = backend.random_normal((64,))
        activations_core = backend.random_normal((5, 64))
        activations_boundary = backend.random_normal((10, 64))
        backend.eval(weight_1d, weight_source, activations_core, activations_boundary)

        result = compute_transplant_delta(
            weight_target=weight_1d,
            weight_source_aligned=weight_source,
            activations_core=activations_core,
            activations_boundary=activations_boundary,
            backend=backend,
        )

        assert result.applied is False
        assert result.null_dim == 0

    def test_dimension_mismatch_skipped(self) -> None:
        """Dimension mismatch between weight and activations should skip."""
        backend = get_default_backend()
        backend.random_seed(42)

        weight_target = backend.random_normal((32, 64))
        weight_source = backend.random_normal((32, 64))
        activations_core = backend.random_normal((5, 128))  # wrong dim
        activations_boundary = backend.random_normal((10, 64))
        backend.eval(weight_target, weight_source, activations_core, activations_boundary)

        result = compute_transplant_delta(
            weight_target=weight_target,
            weight_source_aligned=weight_source,
            activations_core=activations_core,
            activations_boundary=activations_boundary,
            backend=backend,
        )

        assert result.applied is False

    def test_insufficient_core_samples_skipped(self) -> None:
        """Less than 2 core samples should skip transplant."""
        backend = get_default_backend()
        backend.random_seed(42)

        weight_target = backend.random_normal((32, 64))
        weight_source = backend.random_normal((32, 64))
        activations_core = backend.random_normal((1, 64))  # only 1 sample
        activations_boundary = backend.random_normal((10, 64))
        backend.eval(weight_target, weight_source, activations_core, activations_boundary)

        result = compute_transplant_delta(
            weight_target=weight_target,
            weight_source_aligned=weight_source,
            activations_core=activations_core,
            activations_boundary=activations_boundary,
            backend=backend,
        )

        assert result.applied is False

    def test_preserved_fraction_computed(self) -> None:
        """preserved_fraction should measure how much delta survived projection."""
        backend = get_default_backend()
        backend.random_seed(42)

        in_dim, out_dim = 64, 32
        weight_target = backend.random_normal((out_dim, in_dim))
        weight_source = backend.random_normal((out_dim, in_dim))
        activations_core = backend.random_normal((5, in_dim))
        activations_boundary = backend.random_normal((10, in_dim))
        backend.eval(weight_target, weight_source, activations_core, activations_boundary)

        config = NullSpaceFilterConfig(rank_threshold=1e-6)
        result = compute_transplant_delta(
            weight_target=weight_target,
            weight_source_aligned=weight_source,
            activations_core=activations_core,
            activations_boundary=activations_boundary,
            backend=backend,
            nullspace_config=config,
        )

        assert result.applied is True
        # preserved_fraction should be between 0 and 1
        assert 0.0 <= result.preserved_fraction <= 1.0
        # projection_loss = 1 - preserved_fraction
        assert abs(result.projection_loss + result.preserved_fraction - 1.0) < 1e-6

    def test_zero_delta_has_full_preservation(self) -> None:
        """When source == target, delta is zero, preserved_fraction should be 1.0."""
        backend = get_default_backend()
        backend.random_seed(42)

        in_dim, out_dim = 64, 32
        weight = backend.random_normal((out_dim, in_dim))
        activations_core = backend.random_normal((5, in_dim))
        activations_boundary = backend.random_normal((10, in_dim))
        backend.eval(weight, activations_core, activations_boundary)

        result = compute_transplant_delta(
            weight_target=weight,
            weight_source_aligned=weight,  # same as target
            activations_core=activations_core,
            activations_boundary=activations_boundary,
            backend=backend,
        )

        assert result.applied is True
        assert result.delta_norm < 1e-6
        assert result.preserved_fraction == 1.0
        assert result.projection_loss == 0.0


class TestTransplantEndToEnd:
    """End-to-end tests for the transplant pipeline."""

    def test_transplant_with_large_null_space(self) -> None:
        """When boundary has large null space, more delta should survive."""
        backend = get_default_backend()
        backend.random_seed(42)

        in_dim, out_dim = 128, 64
        n_core, n_boundary = 10, 5  # few boundary = large null space

        weight_target = backend.random_normal((out_dim, in_dim))
        weight_source = backend.random_normal((out_dim, in_dim))
        activations_core = backend.random_normal((n_core, in_dim))
        activations_boundary = backend.random_normal((n_boundary, in_dim))
        backend.eval(weight_target, weight_source, activations_core, activations_boundary)

        config = NullSpaceFilterConfig(rank_threshold=1e-6)
        result = compute_transplant_delta(
            weight_target=weight_target,
            weight_source_aligned=weight_source,
            activations_core=activations_core,
            activations_boundary=activations_boundary,
            backend=backend,
            nullspace_config=config,
        )

        assert result.applied is True
        # With only 5 boundary samples in 128-dim space, null space is large
        # Expect high preserved_fraction (most of delta survives)
        assert result.null_dim > 100  # 128 - 5 = 123 null dims expected
        assert result.preserved_fraction > 0.5  # majority survives

    def test_transplant_with_small_null_space(self) -> None:
        """When boundary has small null space, less delta survives."""
        backend = get_default_backend()
        backend.random_seed(42)

        in_dim, out_dim = 64, 32
        n_core, n_boundary = 5, 60  # many boundary = small null space

        weight_target = backend.random_normal((out_dim, in_dim))
        weight_source = backend.random_normal((out_dim, in_dim))
        activations_core = backend.random_normal((n_core, in_dim))
        activations_boundary = backend.random_normal((n_boundary, in_dim))
        backend.eval(weight_target, weight_source, activations_core, activations_boundary)

        config = NullSpaceFilterConfig(rank_threshold=1e-6)
        result = compute_transplant_delta(
            weight_target=weight_target,
            weight_source_aligned=weight_source,
            activations_core=activations_core,
            activations_boundary=activations_boundary,
            backend=backend,
            nullspace_config=config,
        )

        assert result.applied is True
        # With 60 boundary samples in 64-dim space, null space is small
        # Expect lower preserved_fraction (less delta survives)
        assert result.null_dim < 10  # 64 - min(60, 64) = 4 null dims expected
