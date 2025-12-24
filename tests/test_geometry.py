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

import numpy as np
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
from modelcypher.core.use_cases.geometry_engine import GeometryEngine, SinkhornSolver, SinkhornSolverConfig
from tests.conftest import NumpyBackend


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
        "layer.lora_a": np.ones((4, 2), dtype=np.float32),
        "layer.lora_b": np.ones((2, 3), dtype=np.float32),
    }
    metrics = engine.compute_lora_geometry(params, None, scale=1.0)
    assert metrics.trainable_scalar_count == 4 * 2 + 2 * 3
    assert metrics.parameter_l2 > 0
