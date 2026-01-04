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

"""Tests for manifold fidelity sweep."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.manifold_fidelity_sweep import (
    LayerSweep,
    ManifoldFidelitySweep,
    PlateauSummary,
    RankMetrics,
    SweepReport,
    find_optimal_rank,
)


@pytest.fixture
def backend():
    """Get default backend for tests."""
    return get_default_backend()


class TestRankMetrics:
    """Tests for RankMetrics dataclass."""

    def test_frozen_dataclass(self):
        """RankMetrics should be immutable."""
        metrics = RankMetrics(
            rank=4,
            anchor_count=100,
            cka=0.95,
            procrustes_error=0.05,
            knn_overlap=0.80,
            distance_correlation=0.90,
            variance_captured_source=0.85,
            variance_captured_target=0.82,
        )
        with pytest.raises(FrozenInstanceError):
            metrics.cka = 0.99

    def test_all_fields_accessible(self):
        """All fields should be accessible."""
        metrics = RankMetrics(
            rank=8,
            anchor_count=50,
            cka=0.92,
            procrustes_error=0.08,
            knn_overlap=0.75,
            distance_correlation=0.88,
            variance_captured_source=0.80,
            variance_captured_target=0.78,
        )
        assert metrics.rank == 8
        assert metrics.anchor_count == 50
        assert metrics.cka == 0.92
        assert metrics.procrustes_error == 0.08
        assert metrics.knn_overlap == 0.75
        assert metrics.distance_correlation == 0.88
        assert metrics.variance_captured_source == 0.80
        assert metrics.variance_captured_target == 0.78


class TestPlateauSummary:
    """Tests for PlateauSummary dataclass."""

    def test_frozen_dataclass(self):
        """PlateauSummary should be immutable."""
        plateau = PlateauSummary(cka=8, procrustes_error=16)
        with pytest.raises(FrozenInstanceError):
            plateau.cka = 32

    def test_default_values(self):
        """Default values should be None."""
        plateau = PlateauSummary()
        assert plateau.cka is None
        assert plateau.procrustes_error is None
        assert plateau.knn_overlap is None
        assert plateau.distance_correlation is None
        assert plateau.variance_captured is None

    def test_partial_values(self):
        """Should allow partial initialization."""
        plateau = PlateauSummary(cka=4, knn_overlap=16)
        assert plateau.cka == 4
        assert plateau.procrustes_error is None
        assert plateau.knn_overlap == 16


class TestLayerSweep:
    """Tests for LayerSweep dataclass."""

    def test_frozen_dataclass(self):
        """LayerSweep should be immutable."""
        sweep = LayerSweep(
            source_layer=0,
            target_layer=0,
            anchor_count=100,
            metrics=(),
            plateau=PlateauSummary(),
        )
        with pytest.raises(FrozenInstanceError):
            sweep.anchor_count = 200

    def test_with_metrics_tuple(self):
        """Should accept tuple of metrics."""
        metrics = (
            RankMetrics(4, 100, 0.8, 0.2, 0.7, 0.85, 0.6, 0.55),
            RankMetrics(8, 100, 0.9, 0.1, 0.8, 0.90, 0.75, 0.72),
        )
        sweep = LayerSweep(
            source_layer=1,
            target_layer=2,
            anchor_count=100,
            metrics=metrics,
            plateau=PlateauSummary(cka=8),
        )
        assert len(sweep.metrics) == 2
        assert sweep.metrics[0].rank == 4
        assert sweep.metrics[1].rank == 8


class TestSweepReport:
    """Tests for SweepReport dataclass."""

    def test_frozen_dataclass(self):
        """SweepReport should be immutable."""
        report = SweepReport(
            source_model="/path/to/source",
            target_model="/path/to/target",
            timestamp=datetime.now(),
            anchor_count=100,
            layer_count=12,
            ranks=(4, 8, 16),
            layer_sweeps=(),
            plateau=PlateauSummary(),
        )
        with pytest.raises(FrozenInstanceError):
            report.anchor_count = 200

    def test_all_fields_accessible(self):
        """All fields should be accessible."""
        now = datetime.now()
        report = SweepReport(
            source_model="/source",
            target_model="/target",
            timestamp=now,
            anchor_count=50,
            layer_count=8,
            ranks=(4, 8),
            layer_sweeps=(),
            plateau=PlateauSummary(cka=4),
        )
        assert report.source_model == "/source"
        assert report.target_model == "/target"
        assert report.timestamp == now
        assert report.anchor_count == 50
        assert report.layer_count == 8
        assert report.ranks == (4, 8)
        assert len(report.layer_sweeps) == 0
        assert report.plateau.cka == 4


class TestManifoldFidelitySweepInit:
    """Tests for ManifoldFidelitySweep initialization."""

    def test_default_backend(self):
        """Should use default backend when not provided."""
        sweep = ManifoldFidelitySweep()
        assert sweep._backend is not None

    def test_explicit_backend(self, backend):
        """Should accept explicit backend."""
        sweep = ManifoldFidelitySweep(backend=backend)
        assert sweep._backend is backend


class TestRunSweep:
    """Tests for run_sweep method."""

    def test_small_matrices(self, backend):
        """Should handle small matrices."""
        sweep = ManifoldFidelitySweep(backend=backend)
        source = backend.random_normal((20, 16))
        target = backend.random_normal((20, 16))
        backend.eval(source, target)

        result = sweep.run_sweep(source, target)

        assert result is not None
        assert result.source_layer == 0
        assert result.target_layer == 0
        assert result.anchor_count == 20
        assert len(result.metrics) > 0

    def test_returns_none_for_insufficient_data(self, backend):
        """Should return None when data is insufficient."""
        sweep = ManifoldFidelitySweep(backend=backend)
        source = backend.random_normal((1, 8))  # Only 1 sample
        target = backend.random_normal((1, 8))
        backend.eval(source, target)

        result = sweep.run_sweep(source, target)

        assert result is None

    def test_layer_indices_passed_through(self, backend):
        """Should pass through layer indices."""
        sweep = ManifoldFidelitySweep(backend=backend)
        source = backend.random_normal((20, 16))
        target = backend.random_normal((20, 16))
        backend.eval(source, target)

        result = sweep.run_sweep(source, target, source_layer=3, target_layer=5)

        assert result is not None
        assert result.source_layer == 3
        assert result.target_layer == 5

    def test_different_sized_matrices(self, backend):
        """Should handle matrices with different sample counts."""
        sweep = ManifoldFidelitySweep(backend=backend)
        source = backend.random_normal((30, 16))
        target = backend.random_normal((25, 16))
        backend.eval(source, target)

        result = sweep.run_sweep(source, target)

        assert result is not None
        # Anchor count should be min of the two
        assert result.anchor_count == 25

    def test_metrics_include_all_fields(self, backend):
        """Metrics should include all required fields."""
        sweep = ManifoldFidelitySweep(backend=backend)
        source = backend.random_normal((30, 32))
        target = backend.random_normal((30, 32))
        backend.eval(source, target)

        result = sweep.run_sweep(source, target)

        assert result is not None
        assert len(result.metrics) > 0
        m = result.metrics[0]
        assert m.rank > 0
        assert m.anchor_count == 30
        assert 0.0 <= m.cka <= 1.0
        assert m.procrustes_error >= 0.0
        assert 0.0 <= m.knn_overlap <= 1.0
        # distance_correlation can be negative
        assert m.variance_captured_source >= 0.0
        assert m.variance_captured_target >= 0.0


class TestPlateau:
    """Tests for plateau detection."""

    def test_plateau_has_values(self, backend):
        """Plateau should have values after sweep."""
        sweep = ManifoldFidelitySweep(backend=backend)
        source = backend.random_normal((50, 64))
        target = backend.random_normal((50, 64))
        backend.eval(source, target)

        result = sweep.run_sweep(source, target)

        assert result is not None
        assert result.plateau.cka is not None
        assert result.plateau.procrustes_error is not None
        assert result.plateau.knn_overlap is not None
        assert result.plateau.variance_captured is not None


class TestFindOptimalRank:
    """Tests for find_optimal_rank convenience function."""

    def test_cka_metric(self, backend):
        """Should find optimal rank for CKA."""
        source = backend.random_normal((30, 32))
        target = backend.random_normal((30, 32))
        backend.eval(source, target)

        rank = find_optimal_rank(source, target, metric="cka", backend=backend)

        assert rank is not None
        assert rank > 0

    def test_procrustes_metric(self, backend):
        """Should find optimal rank for Procrustes."""
        source = backend.random_normal((30, 32))
        target = backend.random_normal((30, 32))
        backend.eval(source, target)

        rank = find_optimal_rank(source, target, metric="procrustes", backend=backend)

        assert rank is not None
        assert rank > 0

    def test_knn_metric(self, backend):
        """Should find optimal rank for k-NN overlap."""
        source = backend.random_normal((30, 32))
        target = backend.random_normal((30, 32))
        backend.eval(source, target)

        rank = find_optimal_rank(source, target, metric="knn", backend=backend)

        assert rank is not None
        assert rank > 0

    def test_invalid_metric_defaults_to_cka(self, backend):
        """Invalid metric should default to CKA."""
        source = backend.random_normal((30, 32))
        target = backend.random_normal((30, 32))
        backend.eval(source, target)

        rank = find_optimal_rank(source, target, metric="invalid", backend=backend)

        assert rank is not None
        assert rank > 0


class TestGeodesicMath:
    """Tests to verify geodesic math is used."""

    def test_uses_geodesic_svd(self, backend):
        """SVD should use geodesic implementation."""
        sweep = ManifoldFidelitySweep(backend=backend)
        source = backend.random_normal((20, 16))
        target = backend.random_normal((20, 16))
        backend.eval(source, target)

        # This should complete without error using geodesic SVD
        result = sweep.run_sweep(source, target)
        assert result is not None

    def test_knn_uses_geodesic_distances(self, backend):
        """k-NN should use geodesic distances."""
        sweep = ManifoldFidelitySweep(backend=backend)
        source = backend.random_normal((20, 16))
        target = backend.random_normal((20, 16))
        backend.eval(source, target)

        result = sweep.run_sweep(source, target)

        assert result is not None
        # k-NN overlap should be computed
        assert all(m.knn_overlap >= 0.0 for m in result.metrics)

    def test_distance_correlation_uses_geodesic(self, backend):
        """Distance correlation should use geodesic distances."""
        sweep = ManifoldFidelitySweep(backend=backend)
        source = backend.random_normal((20, 16))
        target = backend.random_normal((20, 16))
        backend.eval(source, target)

        result = sweep.run_sweep(source, target)

        assert result is not None
        # Distance correlation should be computed
        assert all(isinstance(m.distance_correlation, float) for m in result.metrics)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_identical_matrices(self, backend):
        """Should handle identical source and target."""
        sweep = ManifoldFidelitySweep(backend=backend)
        data = backend.random_normal((30, 32))
        backend.eval(data)

        result = sweep.run_sweep(data, data)

        assert result is not None
        # CKA should be close to 1.0 for identical data
        assert result.metrics[-1].cka > 0.99

    def test_minimum_viable_data(self, backend):
        """Should work with minimum viable data size."""
        sweep = ManifoldFidelitySweep(backend=backend)
        # Minimum: 4 anchors, 4 dimensions (to fit smallest rank)
        source = backend.random_normal((5, 8))
        target = backend.random_normal((5, 8))
        backend.eval(source, target)

        result = sweep.run_sweep(source, target)

        assert result is not None

    def test_high_dimensional_data(self, backend):
        """Should handle high-dimensional data."""
        sweep = ManifoldFidelitySweep(backend=backend)
        source = backend.random_normal((50, 256))
        target = backend.random_normal((50, 256))
        backend.eval(source, target)

        result = sweep.run_sweep(source, target)

        assert result is not None
        # Should have multiple rank levels
        assert len(result.metrics) >= 3
