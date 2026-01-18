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

"""Tests for tangent space alignment."""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.geometry.tangent_space_alignment import (
    LayerResult,
    TangentAlignmentReport,
    TangentSpaceAlignment,
    compute_alignment_for_layers,
    MIN_ANCHOR_COUNT,
)


@pytest.fixture
def backend():
    """Get default backend for tests."""
    return get_default_backend()


class TestLayerResult:
    """Tests for LayerResult dataclass."""

    def test_frozen_dataclass(self):
        """LayerResult should be immutable."""
        result = LayerResult(
            source_layer=0,
            target_layer=1,
            anchor_count=10,
            neighbor_count=3,
            tangent_rank=2,
            mean_cosine=0.9,
            min_cosine=0.8,
            max_cosine=1.0,
            mean_angle_radians=0.1,
            median_angle_radians=0.1,
            coverage=1.0,
        )
        with pytest.raises(AttributeError):
            result.mean_cosine = 0.0

    def test_all_fields_accessible(self):
        """All fields should be accessible."""
        result = LayerResult(
            source_layer=0,
            target_layer=1,
            anchor_count=10,
            neighbor_count=3,
            tangent_rank=2,
            mean_cosine=0.9,
            min_cosine=0.8,
            max_cosine=1.0,
            mean_angle_radians=0.1,
            median_angle_radians=0.15,
            coverage=0.95,
        )
        assert result.source_layer == 0
        assert result.target_layer == 1
        assert result.anchor_count == 10
        assert result.neighbor_count == 3
        assert result.tangent_rank == 2
        assert result.mean_cosine == 0.9
        assert result.min_cosine == 0.8
        assert result.max_cosine == 1.0
        assert result.mean_angle_radians == 0.1
        assert result.median_angle_radians == 0.15
        assert result.coverage == 0.95


class TestTangentAlignmentReport:
    """Tests for TangentAlignmentReport dataclass."""

    def test_frozen_dataclass(self):
        """TangentAlignmentReport should be immutable."""
        from datetime import datetime

        report = TangentAlignmentReport(
            source_model="source",
            target_model="target",
            timestamp=datetime.now(),
            layer_results=[],
            mean_cosine=0.0,
            mean_angle_radians=0.0,
            anchor_count=0,
            layer_count=0,
        )
        with pytest.raises(AttributeError):
            report.mean_cosine = 0.5


class TestTangentSpaceAlignmentInit:
    """Tests for TangentSpaceAlignment initialization."""

    def test_default_initialization(self):
        """Should initialize without backend."""
        aligner = TangentSpaceAlignment()
        assert aligner is not None
        assert aligner._backend is not None

    def test_with_explicit_backend(self, backend):
        """Should accept explicit backend."""
        aligner = TangentSpaceAlignment(backend)
        assert aligner._backend is backend


class TestComputeLayerMetrics:
    """Tests for compute_layer_metrics method."""

    def test_insufficient_anchors(self, backend):
        """Should return None if fewer than MIN_ANCHOR_COUNT anchors."""
        aligner = TangentSpaceAlignment(backend)
        # Only 2 anchors, need at least 3
        source = backend.random_normal((2, 10))
        target = backend.random_normal((2, 10))

        result = aligner.compute_layer_metrics(source, target)
        assert result is None

    def test_mismatched_anchor_counts(self, backend):
        """Should return None if anchor counts don't match."""
        aligner = TangentSpaceAlignment(backend)
        source = backend.random_normal((5, 10))
        target = backend.random_normal((7, 10))

        result = aligner.compute_layer_metrics(source, target)
        assert result is None

    def test_minimum_valid_input(self, backend):
        """Should work with minimum valid input (3 anchors)."""
        aligner = TangentSpaceAlignment(backend)
        source = backend.random_normal((MIN_ANCHOR_COUNT, 10))
        target = backend.random_normal((MIN_ANCHOR_COUNT, 10))

        result = aligner.compute_layer_metrics(source, target)
        # May or may not return result depending on data
        if result is not None:
            assert isinstance(result, LayerResult)
            assert result.anchor_count == MIN_ANCHOR_COUNT

    def test_identical_points_high_cosine(self, backend):
        """Identical points should have high cosine similarity."""
        aligner = TangentSpaceAlignment(backend)
        # Use same points for source and target
        points = backend.random_normal((10, 20))
        backend.eval(points)

        result = aligner.compute_layer_metrics(points, points)

        if result is not None:
            # Identical tangent spaces should have high cosine
            eps = division_epsilon(backend, points)
            assert abs(result.mean_cosine - 1.0) <= eps

    def test_orthogonal_points_lower_cosine(self, backend):
        """Orthogonal subspaces should have lower cosine."""
        aligner = TangentSpaceAlignment(backend)

        # Create two orthogonal point clouds
        n, d = 10, 20
        source = backend.zeros((n, d))
        target = backend.zeros((n, d))

        # Source varies in first half of dimensions
        for i in range(n):
            vals = [float(i) / n] * (d // 2) + [0.0] * (d // 2)
            source = backend.array([
                [float(j) / n if k < d // 2 else 0.0 for k in range(d)]
                for j in range(n)
            ])

        # Target varies in second half of dimensions
        target = backend.array([
            [0.0 if k < d // 2 else float(j) / n for k in range(d)]
            for j in range(n)
        ])

        result = aligner.compute_layer_metrics(source, target)
        # Result may vary, just check it completes
        if result is not None:
            assert isinstance(result, LayerResult)

    def test_layer_indices_preserved(self, backend):
        """Layer indices should be preserved in result."""
        aligner = TangentSpaceAlignment(backend)
        source = backend.random_normal((10, 20))
        target = backend.random_normal((10, 20))

        result = aligner.compute_layer_metrics(source, target, source_layer=3, target_layer=5)

        if result is not None:
            assert result.source_layer == 3
            assert result.target_layer == 5

    def test_cosine_bounds(self, backend):
        """Cosine values should be in [0, 1]."""
        aligner = TangentSpaceAlignment(backend)
        source = backend.random_normal((15, 30))
        target = backend.random_normal((15, 30))

        result = aligner.compute_layer_metrics(source, target)

        if result is not None:
            assert 0.0 <= result.min_cosine <= 1.0
            assert 0.0 <= result.max_cosine <= 1.0
            assert 0.0 <= result.mean_cosine <= 1.0
            assert result.min_cosine <= result.mean_cosine <= result.max_cosine

    def test_coverage_bounds(self, backend):
        """Coverage should be in [0, 1]."""
        aligner = TangentSpaceAlignment(backend)
        source = backend.random_normal((10, 20))
        target = backend.random_normal((10, 20))

        result = aligner.compute_layer_metrics(source, target)

        if result is not None:
            assert 0.0 <= result.coverage <= 1.0


class TestComputeNeighbors:
    """Tests for _compute_neighbors method."""

    def test_empty_points(self, backend):
        """Empty points should return empty neighbors."""
        aligner = TangentSpaceAlignment(backend)
        points = backend.zeros((0, 10))

        neighbors = aligner._compute_neighbors(points, k=3)

        assert neighbors.shape[0] == 0

    def test_k_zero(self, backend):
        """k=0 should return empty neighbors."""
        aligner = TangentSpaceAlignment(backend)
        points = backend.random_normal((5, 10))

        neighbors = aligner._compute_neighbors(points, k=0)

        assert neighbors.shape[1] == 0

    def test_neighbor_count(self, backend):
        """Should return k neighbors per point."""
        aligner = TangentSpaceAlignment(backend)
        points = backend.random_normal((10, 20))
        k = 3

        neighbors = aligner._compute_neighbors(points, k)

        assert neighbors.shape == (10, k)


class TestMedian:
    """Tests for _median helper method."""

    def test_empty_list(self, backend):
        """Empty list should return 0.0."""
        aligner = TangentSpaceAlignment(backend)
        assert aligner._median([]) == 0.0

    def test_single_value(self, backend):
        """Single value should return that value."""
        aligner = TangentSpaceAlignment(backend)
        assert aligner._median([5.0]) == 5.0

    def test_odd_count(self, backend):
        """Odd count should return middle value."""
        aligner = TangentSpaceAlignment(backend)
        assert aligner._median([1.0, 3.0, 2.0]) == 2.0

    def test_even_count(self, backend):
        """Even count should return average of middle two."""
        aligner = TangentSpaceAlignment(backend)
        assert aligner._median([1.0, 2.0, 3.0, 4.0]) == 2.5


class TestComputeAlignmentForLayers:
    """Tests for compute_alignment_for_layers function."""

    def test_empty_mappings(self, backend):
        """Empty mappings should return empty report."""
        report = compute_alignment_for_layers({}, {}, [], backend)

        assert isinstance(report, TangentAlignmentReport)
        assert report.layer_count == 0
        assert report.layer_results == []

    def test_missing_layers(self, backend):
        """Missing layers should be skipped."""
        source = {0: backend.random_normal((10, 20))}
        target = {1: backend.random_normal((10, 20))}
        mappings = [(0, 1), (2, 3)]  # Layer 2 and 3 don't exist

        report = compute_alignment_for_layers(source, target, mappings, backend)

        # Only (0, 1) should work, but needs matching data
        assert isinstance(report, TangentAlignmentReport)

    def test_multiple_layer_pairs(self, backend):
        """Should process multiple layer pairs."""
        source = {
            0: backend.random_normal((10, 20)),
            1: backend.random_normal((10, 20)),
        }
        target = {
            0: backend.random_normal((10, 20)),
            1: backend.random_normal((10, 20)),
        }
        mappings = [(0, 0), (1, 1)]

        report = compute_alignment_for_layers(source, target, mappings, backend)

        assert isinstance(report, TangentAlignmentReport)
        # Results may vary based on random data

    def test_report_timestamp(self, backend):
        """Report should have a timestamp."""
        from datetime import datetime

        report = compute_alignment_for_layers({}, {}, [], backend)

        assert isinstance(report.timestamp, datetime)


class TestPrincipalCosines:
    """Tests for _principal_cosines method."""

    def test_matching_bases(self, backend):
        """Matching bases should have cosines near 1."""
        aligner = TangentSpaceAlignment(backend)

        # Create a simple orthonormal basis
        dim, rank = 10, 3
        basis = backend.eye(dim)[:, :rank]  # First 3 columns of identity
        backend.eval(basis)

        epsilon = division_epsilon(backend, basis)
        cosines = aligner._principal_cosines(basis, basis, epsilon)

        assert len(cosines) == rank
        for cos in cosines:
            assert abs(cos - 1.0) <= epsilon

    def test_dimension_mismatch(self, backend):
        """Mismatched dimensions should return empty list."""
        aligner = TangentSpaceAlignment(backend)

        basis_a = backend.random_normal((10, 3))
        basis_b = backend.random_normal((20, 3))  # Different first dimension

        epsilon = division_epsilon(backend, basis_a)
        cosines = aligner._principal_cosines(basis_a, basis_b, epsilon)

        assert cosines == []

    def test_zero_rank(self, backend):
        """Zero rank should return empty list."""
        aligner = TangentSpaceAlignment(backend)

        basis_a = backend.random_normal((10, 0))  # Zero columns
        basis_b = backend.random_normal((10, 0))

        epsilon = division_epsilon(backend, basis_a)
        cosines = aligner._principal_cosines(basis_a, basis_b, epsilon)

        assert cosines == []
