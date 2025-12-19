from __future__ import annotations

import numpy as np

from modelcypher.core.domain.geometry import DoRADecomposition
from modelcypher.core.use_cases.geometry_engine import GeometryEngine, SinkhornSolver, SinkhornSolverConfig
from tests.conftest import NumpyBackend


def test_dora_decomposition_balanced():
    base = {"layer": [1.0, 0.0, 0.0]}
    current = {"layer": [0.0, 1.0, 0.0]}
    result = DoRADecomposition.analyze_adapter(base, current)
    assert result.dominant_change_type.value in {"directionDominated", "balanced", "minimal"}


def test_procrustes_alignment_recovers_rotation():
    backend = NumpyBackend()
    engine = GeometryEngine(backend)
    rng = np.random.default_rng(0)
    source = rng.standard_normal((10, 4)).astype(np.float32)
    theta = 0.3
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0, 0.0],
            [np.sin(theta), np.cos(theta), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    target = source @ rot
    result = engine.orthogonal_procrustes(source, target, np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32))
    aligned = source @ result.omega
    assert np.allclose(aligned, target, atol=1e-3)


def test_sinkhorn_plan_marginals():
    backend = NumpyBackend()
    solver = SinkhornSolver(backend)
    cost = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    result = solver.solve(cost, config=SinkhornSolverConfig(max_iterations=200, epsilon=0.1))
    plan = result.plan
    assert np.allclose(plan.sum(axis=0), np.array([0.5, 0.5]), atol=1e-2)
    assert np.allclose(plan.sum(axis=1), np.array([0.5, 0.5]), atol=1e-2)


def test_lora_geometry_metrics():
    backend = NumpyBackend()
    engine = GeometryEngine(backend)
    params = {
        \"layer.lora_a\": np.ones((4, 2), dtype=np.float32),
        \"layer.lora_b\": np.ones((2, 3), dtype=np.float32),
    }
    metrics = engine.compute_lora_geometry(params, None, scale=1.0)
    assert metrics.trainable_scalar_count == 4 * 2 + 2 * 3
    assert metrics.parameter_l2 > 0
