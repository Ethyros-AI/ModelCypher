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
    x = mx.random.normal((20, 32))
    # Linear transformation preserves distances up to scale
    y = x @ mx.random.normal((32, 32))
    
    sweep = ManifoldFidelitySweep()
    dist_corr = sweep._compute_distance_correlation(x, y)
    
    # Linear projection should preserve most pairwise distance relations
    assert float(dist_corr) > 0.7


def test_manifold_regularity_intrinsic_dimension():
    """Test intrinsic dimension regularity."""
    # Points on a 2D plane embedded in 10D
    n = 100
    points = [[float(i), float(j)] + [0.0]*8 for i in range(10) for j in range(10)]
    
    summary = ManifoldDimensionality.estimate_id(points, use_regression=True)
    
    # ID should be close to 2.0
    assert 1.5 < summary.intrinsic_dimension < 2.5


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
    # Random rotation matrix
    q, _ = mx.linalg.qr(mx.random.normal((16, 16)))
    y = x @ q
    
    sweep = ManifoldFidelitySweep()
    error = sweep._compute_procrustes_error(x, y)
    
    # Error should be near zero after optimal rotation
    assert error < 1e-5
