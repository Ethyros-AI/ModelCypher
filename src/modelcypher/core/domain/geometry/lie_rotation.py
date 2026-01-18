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

"""Lie-group utilities for SO(n) rotations.

Provides log/exp maps and geodesic distances for rotation matrices using
backend-only operations. These are more accurate than Euclidean surrogates
when measuring or scaling rotations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon, geodesic_svd

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


def _project_to_so(matrix: "Array", backend: "Backend") -> "Array":
    """Project a square matrix to the nearest proper rotation (det=+1)."""
    U, _, Vt = geodesic_svd(backend, matrix)
    R = backend.matmul(U, Vt)
    det_arr = backend.det(R)
    backend.eval(det_arr)
    if float(backend.to_scalar(det_arr)) < 0:
        U_fixed = backend.concatenate([U[:, :-1], -U[:, -1:]], axis=1)
        R = backend.matmul(U_fixed, Vt)
    backend.eval(R)
    return R


def _sym_part(matrix: "Array", backend: "Backend") -> "Array":
    """Symmetric part: (A + A^T) / 2."""
    return 0.5 * (matrix + backend.transpose(matrix))


def _skew_part(matrix: "Array", backend: "Backend") -> "Array":
    """Skew-symmetric part: (A - A^T) / 2."""
    return 0.5 * (matrix - backend.transpose(matrix))


def so_log(
    rotation: "Array",
    backend: "Backend | None" = None,
    project: bool = True,
) -> "Array":
    """Log map from SO(n) to so(n).

    Uses the identity: log(R) = S * f(C), where
        C = (R + R^T)/2, S = (R - R^T)/2,
        f(c) = arccos(c) / sqrt(1 - c^2).
    This is exact for orthogonal matrices without eigenvalue -1 and
    uses stable limits near c=1.
    """
    b = backend or get_default_backend()
    R = rotation if hasattr(rotation, "shape") else b.array(rotation)
    if project:
        R = _project_to_so(R, b)

    C = _sym_part(R, b)
    S = _skew_part(R, b)
    eigvals, eigvecs = b.eigh(C)
    b.eval(eigvals, eigvecs)

    eigvals = b.clip(eigvals, -1.0, 1.0)
    eps = division_epsilon(b, eigvals)
    eps_arr = b.full(b.shape(eigvals), eps)

    theta = b.arccos(eigvals)
    sin_theta = b.sqrt(b.maximum(1.0 - eigvals * eigvals, eps_arr))
    near_one = (1.0 - eigvals) <= eps_arr
    factor = b.where(near_one, b.ones_like(theta), theta / sin_theta)

    fC = b.matmul(eigvecs, b.matmul(b.diag(factor), b.transpose(eigvecs)))
    log_R = b.matmul(S, fC)
    log_R = _skew_part(log_R, b)
    b.eval(log_R)
    return log_R


def so_exp(
    algebra: "Array",
    backend: "Backend | None" = None,
    project: bool = True,
) -> "Array":
    """Exponential map from so(n) to SO(n).

    For skew-symmetric A, exp(A) is computed via spectral functions of K = -A^2:
        exp(A) = I + f1(K) A + f2(K) A^2
    where f1(λ) = sin(sqrt(λ)) / sqrt(λ),
          f2(λ) = (1 - cos(sqrt(λ))) / λ,
    with stable limits for λ→0.
    """
    b = backend or get_default_backend()
    A = algebra if hasattr(algebra, "shape") else b.array(algebra)
    A = _skew_part(A, b)
    A2 = b.matmul(A, A)
    K = -A2

    eigvals, eigvecs = b.eigh(K)
    b.eval(eigvals, eigvecs)
    eigvals = b.maximum(eigvals, b.zeros_like(eigvals))

    eps = division_epsilon(b, eigvals)
    eps_arr = b.full(b.shape(eigvals), eps)
    sqrt_vals = b.sqrt(eigvals)
    near_zero = sqrt_vals <= eps_arr

    f1 = b.where(near_zero, b.ones_like(sqrt_vals), b.sin(sqrt_vals) / sqrt_vals)
    f2 = b.where(
        near_zero,
        b.full(b.shape(eigvals), 0.5),
        (1.0 - b.cos(sqrt_vals)) / eigvals,
    )

    f1K = b.matmul(eigvecs, b.matmul(b.diag(f1), b.transpose(eigvecs)))
    f2K = b.matmul(eigvecs, b.matmul(b.diag(f2), b.transpose(eigvecs)))

    n = int(b.shape(A)[0])
    I = b.eye(n)
    exp_A = I + b.matmul(f1K, A) + b.matmul(f2K, A2)
    if project:
        exp_A = _project_to_so(exp_A, b)
    b.eval(exp_A)
    return exp_A


def so_geodesic_distance(
    rotation_a: "Array",
    rotation_b: "Array",
    backend: "Backend | None" = None,
    project: bool = True,
) -> float:
    """Geodesic distance on SO(n) with the canonical bi-invariant metric."""
    b = backend or get_default_backend()
    R_a = rotation_a if hasattr(rotation_a, "shape") else b.array(rotation_a)
    R_b = rotation_b if hasattr(rotation_b, "shape") else b.array(rotation_b)
    if project:
        R_a = _project_to_so(R_a, b)
        R_b = _project_to_so(R_b, b)

    R_rel = b.matmul(b.transpose(R_a), R_b)
    C = _sym_part(R_rel, b)
    eigvals = b.eigvalsh(C)
    b.eval(eigvals)
    eigvals = b.clip(eigvals, -1.0, 1.0)
    theta = b.arccos(eigvals)
    theta_sq = theta * theta
    dist_arr = b.sqrt(0.5 * b.sum(theta_sq))
    b.eval(dist_arr)
    return float(b.to_scalar(dist_arr))


def so_scale_rotation(
    rotation: "Array",
    scale: float,
    backend: "Backend | None" = None,
    project: bool = True,
) -> "Array":
    """Scale a rotation geodesically: exp(scale * log(R))."""
    b = backend or get_default_backend()
    R = rotation if hasattr(rotation, "shape") else b.array(rotation)
    n = int(b.shape(R)[0])
    if scale <= 0.0:
        return b.eye(n)
    if scale >= 1.0:
        return _project_to_so(R, b) if project else R

    log_R = so_log(R, backend=b, project=project)
    scaled = log_R * scale
    return so_exp(scaled, backend=b, project=project)


def so_geodesic_interpolate(
    rotation_a: "Array",
    rotation_b: "Array",
    t: float,
    backend: "Backend | None" = None,
) -> "Array":
    """Geodesic interpolation on SO(n)."""
    b = backend or get_default_backend()
    R_a = rotation_a if hasattr(rotation_a, "shape") else b.array(rotation_a)
    R_b = rotation_b if hasattr(rotation_b, "shape") else b.array(rotation_b)
    if t <= 0.0:
        return _project_to_so(R_a, b)
    if t >= 1.0:
        return _project_to_so(R_b, b)

    R_rel = b.matmul(b.transpose(_project_to_so(R_a, b)), _project_to_so(R_b, b))
    step = so_scale_rotation(R_rel, t, backend=b, project=True)
    return b.matmul(_project_to_so(R_a, b), step)
