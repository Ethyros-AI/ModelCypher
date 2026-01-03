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

"""Manifold regularity tests (requires MLX)."""

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
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.manifold_dimensionality import ManifoldDimensionality
from modelcypher.core.domain.geometry.manifold_fidelity_sweep import (
    ManifoldFidelitySweep,
    SweepConfig,
)
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon


def _sweep_config_for(points) -> SweepConfig:
    backend = get_default_backend()
    n = int(points.shape[0])
    d = int(points.shape[1])
    max_rank = max(1, min(n, d))
    ranks = [max_rank]
    neighbor_count = max(1, n - 1)
    eps = division_epsilon(backend, backend.array([1.0]))
    return SweepConfig.with_parameters(
        ranks=ranks,
        neighbor_count=neighbor_count,
        min_anchor_count=max(2, min(n, 2)),
        plateau_epsilon=eps,
    )


def test_manifold_regularity_cka_identity():
    """CKA should be 1.0 for identical manifold representations."""
    x = mx.random.normal((32, 64))
    sweep = ManifoldFidelitySweep(_sweep_config_for(x))

    cka = sweep._compute_cka(x, x)
    backend = get_default_backend()
    eps = division_epsilon(backend, backend.array([1.0]))
    assert abs(float(cka) - 1.0) <= eps


def test_manifold_regularity_distance_correlation():
    """Distance correlation should be high for linearly related manifolds."""
    mx.random.seed(42)  # Seed for reproducibility
    x = mx.random.normal((20, 32))
    # Linear transformation preserves distances up to scale
    y = x @ mx.random.normal((32, 32))

    sweep = ManifoldFidelitySweep(_sweep_config_for(x))
    dist_corr = sweep._compute_distance_correlation(x, y)

    backend = get_default_backend()
    eps = division_epsilon(backend, backend.array([1.0]))
    assert 0.0 - eps <= float(dist_corr) <= 1.0 + eps


def test_manifold_regularity_intrinsic_dimension():
    """Test intrinsic dimension regularity."""
    import random

    random.seed(42)
    # Points on a 2D manifold embedded in 10D
    # Use sufficient noise (0.3) to break grid structure - TwoNN requires
    # continuous manifold data, not discrete grids. With small noise (0.01),
    # all nearest neighbor ratios μ ≈ 1, making regression unstable.
    n = 200
    points = [
        [float(i % 14) + random.gauss(0, 0.3), float(i // 14) + random.gauss(0, 0.3)]
        + [random.gauss(0, 0.01)] * 8
        for i in range(n)
    ]

    summary = ManifoldDimensionality.estimate_id(points)

    backend = get_default_backend()
    eps = division_epsilon(backend, backend.array([1.0]))
    assert summary.intrinsic_dimension >= eps
    assert summary.intrinsic_dimension <= len(points[0]) + eps


def test_manifold_regularity_variance_captured():
    """Test rank-based variance capture regularity."""
    x = mx.random.normal((50, 64))
    # Zero out some dimensions to control variance
    x_low_rank = x * mx.array([1.0] * 10 + [0.0] * 54)

    sweep = ManifoldFidelitySweep(_sweep_config_for(x_low_rank))
    centered = sweep._center(x_low_rank)
    svd = sweep._compute_svd(centered)

    var_ratio = sweep._variance_ratio(svd[0], rank=10)

    # Rank 10 should capture all variance
    backend = get_default_backend()
    eps = division_epsilon(backend, backend.array([1.0]))
    assert abs(float(var_ratio) - 1.0) <= eps


def test_manifold_regularity_procrustes_error():
    """Procrustes error should be low for rotated manifolds."""
    x = mx.random.normal((30, 16))
    # Random rotation matrix - use CPU stream for QR decomposition
    q, _ = mx.linalg.qr(mx.random.normal((16, 16)), stream=mx.cpu)
    y = x @ q

    sweep = ManifoldFidelitySweep(_sweep_config_for(x))
    error = sweep._compute_procrustes_error(x, y)

    backend = get_default_backend()
    eps = division_epsilon(backend, backend.array([1.0]))
    assert error <= eps
