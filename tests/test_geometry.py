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
)
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.geometry.vector_math import geodesic_norms


def _eps(backend, *values: float) -> float:
    return division_epsilon(backend, backend.array(list(values) or [1.0]))


def test_dora_decomposition_direction_change():
    """90° rotation with same magnitude should be direction-dominated."""
    # Unit vectors: x-axis to y-axis is pure directional change
    base = {"layer": mx.array([1.0, 0.0, 0.0])}
    current = {"layer": mx.array([0.0, 1.0, 0.0])}
    decomposer = DoRADecomposition()
    result = decomposer.analyze_adapter(base, current)
    backend = get_default_backend()
    eps = _eps(backend, result.overall_magnitude_change, result.overall_directional_drift)
    # Same magnitude (both unit vectors), different direction -> direction_dominated
    assert result.dominant_change_type.value == "direction_dominated"
    # Magnitude change should be ~0.0 (both are unit vectors, no magnitude change)
    assert abs(result.overall_magnitude_change - 0.0) <= eps
    # Directional drift should be ~1.0 (orthogonal vectors = cosine similarity 0)
    assert abs(result.overall_directional_drift - 1.0) <= eps


def test_procrustes_alignment_recovers_rotation():
    backend = get_default_backend()
    engine = GeometryEngine(backend)
    backend.random_seed(0)
    source = backend.random_normal((10, 4), dtype="float32")
    theta = 0.3
    angle = backend.array([theta], dtype="float32")
    cos_val = backend.cos(angle)
    sin_val = backend.sin(angle)
    backend.eval(cos_val, sin_val)
    c = float(backend.to_scalar(cos_val))
    s = float(backend.to_scalar(sin_val))
    rot = backend.array(
        [
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype="float32"
    )
    target = backend.matmul(source, rot)
    backend.eval(target)
    result = engine.orthogonal_procrustes(
        source, target, backend.eye(4, dtype="float32"), backend.eye(4, dtype="float32")
    )
    aligned = backend.matmul(source, result.omega)
    backend.eval(aligned)
    diff = backend.abs(aligned - target)
    rss = geodesic_norms(backend.reshape(diff, (1, -1)), backend)
    denom = geodesic_norms(backend.reshape(target, (1, -1)), backend)
    backend.eval(rss, denom)
    rss_val = float(backend.to_scalar(rss))
    denom_val = float(backend.to_scalar(denom))
    eps = _eps(backend, rss_val, denom_val)
    ratio = rss_val / max(denom_val, eps)
    assert abs(result.error - ratio) <= eps


def test_sinkhorn_plan_marginals():
    backend = get_default_backend()
    solver = SinkhornSolver(backend)
    cost = backend.array([[0.0, 1.0], [1.0, 0.0]], dtype="float32")
    result = solver.solve(cost)
    plan = result.plan
    backend.eval(plan)
    marginal_0 = backend.sum(plan, axis=0)
    marginal_1 = backend.sum(plan, axis=1)
    backend.eval(marginal_0)
    backend.eval(marginal_1)
    expected = backend.array([0.5, 0.5], dtype="float32")
    diff_0 = backend.abs(marginal_0 - expected)
    diff_1 = backend.abs(marginal_1 - expected)
    eps = _eps(backend, result.marginal_error)
    tolerance = result.marginal_error + eps
    max_diff_0 = backend.max(diff_0)
    max_diff_1 = backend.max(diff_1)
    backend.eval(max_diff_0, max_diff_1)
    assert float(backend.to_scalar(max_diff_0)) <= tolerance
    assert float(backend.to_scalar(max_diff_1)) <= tolerance


def test_lora_geometry_metrics():
    backend = get_default_backend()
    engine = GeometryEngine(backend)
    params = {
        "layer.lora_a": backend.ones((4, 2), dtype="float32"),
        "layer.lora_b": backend.ones((2, 3), dtype="float32"),
    }
    metrics = engine.compute_lora_geometry(params, None, scale=1.0)
    assert metrics.trainable_scalar_count == 4 * 2 + 2 * 3
    assert metrics.parameter_l2 > _eps(backend, metrics.parameter_l2)
