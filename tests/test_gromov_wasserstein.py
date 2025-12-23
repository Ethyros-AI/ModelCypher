from __future__ import annotations

import pytest

from modelcypher.core.domain.geometry.gromov_wasserstein import Config, GromovWassersteinDistance


def test_gw_identity_distance() -> None:
    """Self-comparison should give zero GW distance with uniform coupling."""
    points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    distances = GromovWassersteinDistance.compute_pairwise_distances(points)
    result = GromovWassersteinDistance.compute(distances, distances)
    assert result.distance == 0.0
    assert result.converged is True
    assert result.iterations == 0
    assert len(result.coupling) == 3
    assert result.coupling[0][0] == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert result.coupling[1][1] == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert result.coupling[2][2] == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_gw_permutation_distance_small() -> None:
    """Permuted points should have near-zero GW distance (same shape)."""
    points_a = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    permutation = [2, 0, 1]
    points_b = [points_a[idx] for idx in permutation]
    dist_a = GromovWassersteinDistance.compute_pairwise_distances(points_a)
    dist_b = GromovWassersteinDistance.compute_pairwise_distances(points_b)
    config = Config(
        epsilon=0.05,
        epsilon_min=0.005,
        epsilon_decay=0.97,
        max_outer_iterations=60,
        min_outer_iterations=4,
        max_inner_iterations=150,
        convergence_threshold=1e-6,
        relative_objective_threshold=1e-6,
        use_squared_loss=True,
    )
    result = GromovWassersteinDistance.compute(dist_a, dist_b, config=config)
    assert result.distance < 0.02
    # Verify coupling marginals sum to uniform distribution
    row_mass = [sum(row) for row in result.coupling]
    col_mass = [sum(result.coupling[i][j] for i in range(len(result.coupling))) for j in range(len(result.coupling[0]))]
    assert max(abs(value - 1.0 / 3.0) for value in row_mass) < 0.02
    assert max(abs(value - 1.0 / 3.0) for value in col_mass) < 0.02
