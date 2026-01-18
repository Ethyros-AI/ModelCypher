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

"""Tests for SO(n) Lie-group rotation utilities."""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.lie_rotation import (
    so_exp,
    so_geodesic_distance,
    so_log,
    so_scale_rotation,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    pi_value,
)


@pytest.fixture
def backend():
    return get_default_backend()


def _eps(backend, *values: float) -> float:
    return division_epsilon(backend, backend.array(list(values) or [1.0]))


def _skew_random(backend, dim: int) -> "object":
    mat = backend.random_normal((dim, dim))
    mat = 0.5 * (mat - backend.transpose(mat))
    backend.eval(mat)
    return mat


def test_so_geodesic_distance_identity_zero(backend):
    dim = 6
    identity = backend.eye(dim)
    dist = so_geodesic_distance(identity, identity, backend=backend)
    eps = _eps(backend, dist)
    assert dist <= eps


def test_so_geodesic_distance_matches_2d_angle(backend):
    angle = 0.7
    angle_arr = backend.array([angle])
    cos_val = backend.cos(angle_arr)
    sin_val = backend.sin(angle_arr)
    backend.eval(cos_val, sin_val)
    c = float(backend.to_scalar(cos_val))
    s = float(backend.to_scalar(sin_val))

    R = backend.array([[c, -s], [s, c]])
    I = backend.eye(2)
    dist = so_geodesic_distance(I, R, backend=backend)
    eps = _eps(backend, angle)
    assert abs(dist - angle) <= eps


def test_so_log_exp_round_trip(backend):
    dim = 5
    A = _skew_random(backend, dim)
    pi_val = pi_value(backend)
    eps = _eps(backend, pi_val)

    # Scale to stay within principal branch (angles < pi)
    norm_arr = backend.sqrt(backend.sum(A * A))
    backend.eval(norm_arr)
    norm_val = float(backend.to_scalar(norm_arr))
    target = max(pi_val - eps, eps)
    scale = target / max(norm_val, target)
    A_scaled = A * scale

    R = so_exp(A_scaled, backend=backend)
    A_rec = so_log(R, backend=backend)

    diff = A_rec - A_scaled
    diff_norm_arr = backend.sqrt(backend.sum(diff * diff))
    scaled_norm_arr = backend.sqrt(backend.sum(A_scaled * A_scaled))
    backend.eval(diff_norm_arr, scaled_norm_arr)
    diff_norm = float(backend.to_scalar(diff_norm_arr))
    scaled_norm = float(backend.to_scalar(scaled_norm_arr))
    denom = max(scaled_norm, eps)
    rel_err = diff_norm / denom
    assert rel_err <= eps


def test_so_scale_rotation_half_angle(backend):
    dim = 4
    A = _skew_random(backend, dim)
    R = so_exp(A, backend=backend)
    R_half = so_scale_rotation(R, 0.5, backend=backend)
    I = backend.eye(dim)
    full = so_geodesic_distance(I, R, backend=backend)
    half = so_geodesic_distance(I, R_half, backend=backend)
    eps = _eps(backend, full, half)
    assert abs(half * 2.0 - full) <= eps
