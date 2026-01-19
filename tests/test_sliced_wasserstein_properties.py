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

"""Hypothesis property tests for sliced Wasserstein distance."""

from __future__ import annotations

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
)
from modelcypher.core.domain.geometry.sliced_wasserstein import (
    random_unit_vectors,
    sliced_wasserstein_distance,
)


def _eps(backend, *values: float) -> float:
    arr = backend.array(list(values) or [1.0])
    backend.eval(arr)
    return machine_epsilon(backend, arr)


@st.composite
def _point_clouds(draw, min_points: int = 2, max_points: int = 6, min_dim: int = 1, max_dim: int = 4):
    n = draw(st.integers(min_value=min_points, max_value=max_points))
    d = draw(st.integers(min_value=min_dim, max_value=max_dim))
    point = st.lists(
        st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False, width=32),
        min_size=d,
        max_size=d,
    )
    points = draw(st.lists(point, min_size=n, max_size=n))
    return points, d


@settings(max_examples=10, deadline=None)
@given(
    n_slices=st.integers(min_value=2, max_value=8),
    dimension=st.integers(min_value=1, max_value=6),
    seed=st.integers(min_value=0, max_value=1_000_000),
)
def test_random_unit_vectors_have_unit_norm(n_slices: int, dimension: int, seed: int) -> None:
    backend = get_default_backend()
    vectors = random_unit_vectors(n_slices, dimension, backend, seed=seed)
    backend.eval(vectors)
    norms = backend.sqrt(backend.sum(vectors * vectors, axis=1))
    diff = backend.abs(norms - 1.0)
    max_diff = backend.max(diff)
    backend.eval(max_diff)
    eps = division_epsilon(backend, vectors)
    assert float(backend.to_scalar(max_diff)) <= eps


@settings(max_examples=10, deadline=None)
@given(points_and_dim=_point_clouds(), seed=st.integers(min_value=0, max_value=1_000_000))
def test_sliced_wasserstein_identity(points_and_dim, seed: int) -> None:
    points, _ = points_and_dim
    backend = get_default_backend()
    result = sliced_wasserstein_distance(points, points, backend=backend, seed=seed)
    eps = _eps(backend, result.distance)
    assert result.distance <= eps


@settings(max_examples=10, deadline=None)
@given(
    points_a=_point_clouds(),
    points_b=_point_clouds(),
    seed=st.integers(min_value=0, max_value=1_000_000),
)
def test_sliced_wasserstein_symmetry(points_a, points_b, seed: int) -> None:
    pts_a, dim_a = points_a
    pts_b, dim_b = points_b
    assume(dim_a == dim_b)
    backend = get_default_backend()
    res_ab = sliced_wasserstein_distance(pts_a, pts_b, backend=backend, seed=seed)
    res_ba = sliced_wasserstein_distance(pts_b, pts_a, backend=backend, seed=seed)
    eps = _eps(backend, res_ab.distance, res_ba.distance)
    assert abs(res_ab.distance - res_ba.distance) <= eps
