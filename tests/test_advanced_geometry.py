
import mlx.core as mx
import numpy as np

from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimensionEstimator, TwoNNConfiguration
from modelcypher.core.domain.geometry.manifold_clusterer import ManifoldClusterer, ManifoldPoint

def test_intrinsic_dimension_estimator_mle():
    # Generate points on a 2D plane in 10D space
    # z = [x, y, 0, 0, ...]
    N = 100
    D = 10
    
    # Use random points which have less degeneracy than grid
    # 2D subspace in 10D
    x = mx.random.normal((N, 2))
    grid_points = mx.zeros((N, D))
    grid_points[:, :2] = x * 10.0 # Scale up
    
    # Estimate ID. Should be roughly 2.0 (MLE is biased for small N but approx 2)
    config = TwoNNConfiguration(use_regression=False)
    estimator = IntrinsicDimensionEstimator()
    est = estimator.estimate_two_nn(grid_points, config)
    
    # TwoNN on 100 random points in 2D is fairly consistent
    assert est.intrinsic_dimension > 1.0
    assert est.intrinsic_dimension < 3.0

def test_manifold_clusterer_simple():
    # Creates two distinct clusters of ManifoldPoints
    
    def create_point(base_entropy, mean_gate):
        return ManifoldPoint(
            mean_entropy=base_entropy,
            entropy_variance=0.1,
            first_token_entropy=base_entropy,
            gate_count=5,
            mean_gate_confidence=mean_gate,
            dominant_gate_category=0,
            entropy_path_correlation=0.5,
            assessment_strength=0.5,
            prompt_hash="hash"
        )
        
    cluster1 = [create_point(1.0, 0.9) for _ in range(5)]
    cluster2 = [create_point(5.0, 0.2) for _ in range(5)]
    
    all_points = cluster1 + cluster2
    
    clusterer = ManifoldClusterer(ManifoldClusterer.Configuration(epsilon=1.0, min_points=3))
    result = clusterer.cluster(all_points)
    
    # Expect 2 regions
    assert len(result.regions) == 2
    assert result.noise_points == []
    
    # Check region centroids
    centroids = sorted([r.centroid.mean_entropy for r in result.regions])
    assert abs(centroids[0] - 1.0) < 0.1
    assert abs(centroids[1] - 5.0) < 0.1

def test_manifold_clusterer_noise():
    # 5 points in cluster, 1 outlier far away
    fn = lambda e: ManifoldPoint(e, 0, e, 0, 0, 0, 0, 0, "h")
    points = [fn(1.0) for _ in range(5)]
    outlier = fn(100.0)
    
    clusterer = ManifoldClusterer(ManifoldClusterer.Configuration(epsilon=0.5, min_points=3))
    result = clusterer.cluster(points + [outlier])
    
    assert len(result.regions) == 1
    assert len(result.noise_points) == 1
    assert result.noise_points[0].mean_entropy == 100.0
