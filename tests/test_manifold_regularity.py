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

import mlx.core as mx
import pytest
import math
from modelcypher.core.domain.geometry.manifold_fidelity_sweep import ManifoldFidelitySweep
from modelcypher.core.domain.geometry.manifold_dimensionality import ManifoldDimensionality


def test_manifold_regularity_cka_identity():
    """CKA should be 1.0 for identical manifold representations."""
    x = mx.random.normal((32, 64))
    sweep = ManifoldFidelitySweep()
    
    cka = sweep._compute_cka(x, x)
    assert float(cka) == pytest.approx(1.0)


def test_manifold_regularity_distance_correlation():
    """Distance correlation should be high for linearly related manifolds."""
    mx.random.seed(42)  # Seed for reproducibility
    x = mx.random.normal((20, 32))
    # Linear transformation preserves distances up to scale
    y = x @ mx.random.normal((32, 32))

    sweep = ManifoldFidelitySweep()
    dist_corr = sweep._compute_distance_correlation(x, y)

    # Linear projection should preserve most pairwise distance relations
    # Using 0.6 threshold to account for random variance in small samples
    assert float(dist_corr) > 0.6


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

    summary = ManifoldDimensionality.estimate_id(points, use_regression=True)

    # ID should be close to 2.0 (2D manifold in 10D space)
    assert 1.5 < summary.intrinsic_dimension < 3.5


def test_manifold_regularity_variance_captured():
    """Test rank-based variance capture regularity."""
    x = mx.random.normal((50, 64))
    # Zero out some dimensions to control variance
    x_low_rank = x * mx.array([1.0]*10 + [0.0]*54)
    
    sweep = ManifoldFidelitySweep()
    centered = sweep._center(x_low_rank)
    svd = sweep._compute_svd(centered)
    
    var_ratio = sweep._variance_ratio(svd[0], rank=10)
    
    # Rank 10 should capture all variance
    assert var_ratio == pytest.approx(1.0)


def test_manifold_regularity_procrustes_error():
    """Procrustes error should be low for rotated manifolds."""
    x = mx.random.normal((30, 16))
    # Random rotation matrix - use CPU stream for QR decomposition
    q, _ = mx.linalg.qr(mx.random.normal((16, 16)), stream=mx.cpu)
    y = x @ q
    
    sweep = ManifoldFidelitySweep()
    error = sweep._compute_procrustes_error(x, y)
    
    # Error should be near zero after optimal rotation
    assert error < 1e-5
