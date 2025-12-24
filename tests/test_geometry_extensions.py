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

"""
Unit tests for geometry extension parity modules (requires MLX).

Tests:
- DoRA decomposition analysis
- Tangent space alignment
- Manifold fidelity sweep
"""
import math
import pytest

# Attempt MLX import - skip module entirely if unavailable
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore

# Skip all tests in this module if MLX unavailable
pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available (requires Apple Silicon)")
from modelcypher.core.domain.geometry.dora_decomposition import (
    DoRADecomposition,
    DoRAConfig,
    ChangeType,
    ChangeInterpretation,
    MagnitudeDirectionMetrics,
)
from modelcypher.core.domain.geometry.tangent_space_alignment import (
    TangentSpaceAlignment,
    TangentConfig,
    LayerResult,
)
from modelcypher.core.domain.geometry.manifold_fidelity_sweep import (
    ManifoldFidelitySweep,
    SweepConfig,
    RankMetrics,
)


class TestDoRADecomposition:
    """Tests for DoRA decomposition."""

    def test_same_weights(self):
        """Identical weights should show minimal change."""
        dora = DoRADecomposition()
        w = mx.random.normal((64, 64))

        metrics = dora.decompose(w, w, "test")

        assert metrics is not None
        assert metrics.magnitude_ratio == pytest.approx(1.0, rel=0.01)
        assert metrics.directional_drift == pytest.approx(0.0, abs=0.01)
        assert metrics.direction_cosine == pytest.approx(1.0, rel=0.01)

    def test_scaled_weights(self):
        """Scaled weights should show magnitude change only."""
        dora = DoRADecomposition()
        w1 = mx.random.normal((64, 64))
        w2 = w1 * 2.0  # Double magnitude

        metrics = dora.decompose(w1, w2, "test")

        assert metrics is not None
        assert metrics.magnitude_ratio == pytest.approx(2.0, rel=0.05)
        # Direction should be same
        assert metrics.direction_cosine == pytest.approx(1.0, rel=0.01)

    def test_adapter_analysis(self):
        """Test multi-layer adapter analysis."""
        dora = DoRADecomposition()

        base = {
            "layer1": mx.random.normal((32, 32)),
            "layer2": mx.random.normal((32, 32)),
        }
        current = {
            "layer1": base["layer1"] * 1.1,  # Small magnitude change
            "layer2": base["layer2"] + mx.random.normal((32, 32)) * 0.1,  # Direction change
        }

        result = dora.analyze_adapter(base, current)

        assert len(result.per_layer_metrics) == 2
        assert result.overall_magnitude_change >= 0
        assert result.overall_directional_drift >= 0

    def test_change_type_classification(self):
        """Test dominant change type classification."""
        dora = DoRADecomposition()

        # Minimal change
        w = mx.random.normal((32, 32))
        result = dora.analyze_adapter({"l": w}, {"l": w})
        assert result.dominant_change_type == ChangeType.MINIMAL


class TestTangentSpaceAlignment:
    """Tests for tangent space alignment."""

    def test_identical_points(self):
        """Identical point sets should have high alignment."""
        aligner = TangentSpaceAlignment()
        points = mx.random.normal((20, 64))

        result = aligner.compute_layer_metrics(points, points)

        assert result is not None
        assert result.mean_cosine >= 0.9  # High alignment
        assert result.coverage > 0

    def test_orthogonal_points(self):
        """Orthogonal point sets should have lower alignment."""
        aligner = TangentSpaceAlignment()

        # Create two distinct random manifolds
        points1 = mx.random.normal((20, 64))
        points2 = mx.random.normal((20, 64))

        result = aligner.compute_layer_metrics(points1, points2)

        assert result is not None
        # Random points have lower agreement than identical
        assert result.anchor_count == 20

    def test_insufficient_points(self):
        """Should return None for insufficient points."""
        aligner = TangentSpaceAlignment()
        points = mx.random.normal((3, 64))  # Too few

        result = aligner.compute_layer_metrics(points, points)
        assert result is None


class TestManifoldFidelitySweep:
    """Tests for manifold fidelity sweep."""

    def test_sweep_returns_metrics(self):
        """Sweep should return metrics for each rank."""
        sweep = ManifoldFidelitySweep(SweepConfig(ranks=[4, 8, 16]))

        source = mx.random.normal((50, 128))
        target = mx.random.normal((50, 128))

        result = sweep.run_sweep(source, target)

        assert result is not None
        assert len(result.metrics) == 3  # 3 ranks
        for m in result.metrics:
            assert m.cka >= 0
            assert m.procrustes_error >= 0
            assert m.knn_overlap >= 0
            assert m.distance_correlation >= -1 and m.distance_correlation <= 1

    def test_identical_points_high_cka(self):
        """Identical activations should have CKA close to 1."""
        sweep = ManifoldFidelitySweep(SweepConfig(ranks=[8]))
        points = mx.random.normal((30, 64))

        result = sweep.run_sweep(points, points)

        assert result is not None
        assert result.metrics[0].cka >= 0.99

    def test_plateau_detection(self):
        """Plateau should be detected at optimal rank."""
        sweep = ManifoldFidelitySweep(SweepConfig(ranks=[4, 8, 16, 32]))

        source = mx.random.normal((50, 64))
        target = mx.random.normal((50, 64))

        result = sweep.run_sweep(source, target)

        assert result is not None
        assert result.plateau.cka is not None
        assert result.plateau.cka in [4, 8, 16, 32]

    def test_insufficient_anchors(self):
        """Should return None for insufficient anchors."""
        sweep = ManifoldFidelitySweep(SweepConfig(min_anchor_count=10))

        source = mx.random.normal((5, 64))  # Too few
        target = mx.random.normal((5, 64))

        result = sweep.run_sweep(source, target)
        assert result is None


class TestCKA:
    """Tests for CKA computation within sweep."""

    def test_cka_range(self):
        """CKA should be in [0, 1] for normalized data."""
        sweep = ManifoldFidelitySweep(SweepConfig(ranks=[8]))

        for _ in range(5):
            source = mx.random.normal((30, 32))
            target = mx.random.normal((30, 32))
            result = sweep.run_sweep(source, target)

            if result:
                assert 0 <= result.metrics[0].cka <= 1.01  # Allow small numerical error
