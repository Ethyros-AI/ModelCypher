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

"""Geometry tests that require MLX (Apple Silicon).

These tests verify geometric operations using Metal GPU acceleration.
They are automatically skipped on non-Apple machines.
"""

from __future__ import annotations

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

from modelcypher.core.domain.geometry import DoRADecomposition
from modelcypher.core.use_cases.geometry_engine import (
    GeometryEngine,
    SinkhornSolver,
    SinkhornSolverConfig,
)
from modelcypher.core.domain._backend import get_default_backend


def test_dora_decomposition_direction_change():
    """90° rotation with same magnitude should be direction-dominated."""
    # Unit vectors: x-axis to y-axis is pure directional change
    base = {"layer": mx.array([1.0, 0.0, 0.0])}
    current = {"layer": mx.array([0.0, 1.0, 0.0])}
    decomposer = DoRADecomposition()
    result = decomposer.analyze_adapter(base, current)
    # Same magnitude (both unit vectors), different direction -> direction_dominated
    assert result.dominant_change_type.value == "direction_dominated"
    # Magnitude change should be ~0.0 (both are unit vectors, no magnitude change)
    assert result.overall_magnitude_change == pytest.approx(0.0, abs=0.01)
    # Directional drift should be ~1.0 (orthogonal vectors = cosine similarity 0)
    assert result.overall_directional_drift == pytest.approx(1.0, abs=0.01)


def test_procrustes_alignment_recovers_rotation():
    backend = get_default_backend()
    engine = GeometryEngine(backend)
    backend.random_seed(0)
    source = backend.random_randn((10, 4))
    theta = 0.3
    import math
    rot = backend.array(
        [
            [math.cos(theta), -math.sin(theta), 0.0, 0.0],
            [math.sin(theta), math.cos(theta), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    target = backend.matmul(source, rot)
    backend.eval(target)
    result = engine.orthogonal_procrustes(
        source, target, backend.eye(4), backend.eye(4)
    )
    aligned = backend.matmul(source, result.omega)
    backend.eval(aligned)
    diff = backend.abs(aligned - target)
    assert backend.max(diff).item() < 1e-3


def test_sinkhorn_plan_marginals():
    backend = get_default_backend()
    solver = SinkhornSolver(backend)
    cost = backend.array([[0.0, 1.0], [1.0, 0.0]])
    result = solver.solve(cost, config=SinkhornSolverConfig(max_iterations=200, epsilon=0.1))
    plan = result.plan
    backend.eval(plan)
    marginal_0 = backend.sum(plan, axis=0)
    marginal_1 = backend.sum(plan, axis=1)
    backend.eval(marginal_0)
    backend.eval(marginal_1)
    expected = backend.array([0.5, 0.5])
    diff_0 = backend.abs(marginal_0 - expected)
    diff_1 = backend.abs(marginal_1 - expected)
    assert backend.max(diff_0).item() < 1e-2
    assert backend.max(diff_1).item() < 1e-2


def test_lora_geometry_metrics():
    backend = get_default_backend()
    engine = GeometryEngine(backend)
    params = {
        "layer.lora_a": backend.ones((4, 2)),
        "layer.lora_b": backend.ones((2, 3)),
    }
    metrics = engine.compute_lora_geometry(params, None, scale=1.0)
    assert metrics.trainable_scalar_count == 4 * 2 + 2 * 3
    assert metrics.parameter_l2 > 0
