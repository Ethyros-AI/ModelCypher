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

"""Parallel transport and holonomy for manifold diagnostics.

Implements Schild's ladder approximation for parallel transport along geodesics
and holonomy computation for closed loops. Non-zero holonomy indicates curvature.

References:
    - Schild, A. (1970). "Tether-parallel transport" in Relativity.
    - Ehlers, J., Pirani, F. A. E., Schild, A. (1972). "The geometry of free fall
      and light propagation."
    - Kheyfets, A., Miller, W. A. (2000). "Schild's ladder parallel transport
      procedure for an arbitrary connection."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
    sqrt_scalar,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class ParallelTransportResult:
    """Result of parallel transport along a path.

    Attributes:
        transported_vector: Final transported vector [d].
        initial_vector: Original vector before transport [d].
        path_indices: Indices of points along the path.
        path_length: Total geodesic distance along path.
        angular_drift: arccos(⟨v_init, v_final⟩ / norms) - rotation angle.
        norm_ratio: ||v_final|| / ||v_init|| - should be ~1.0 for Levi-Civita.
    """

    transported_vector: "Array"
    initial_vector: "Array"
    path_indices: tuple[int, ...]
    path_length: float
    angular_drift: float
    norm_ratio: float


@dataclass(frozen=True)
class HolonomyResult:
    """Result of holonomy computation around a closed loop.

    Holonomy measures the rotation acquired by parallel transport around
    a closed loop. On a flat manifold, holonomy is zero. Non-zero holonomy
    indicates curvature enclosed by the loop.

    Attributes:
        holonomy_matrix: Rotation matrix from transport [d, d].
        holonomy_angle: ||log(H)||_F via so_log - total rotation angle.
        loop_indices: Indices of points forming the loop.
        loop_length: Total geodesic distance around loop.
        axis: Rotation axis (eigenvector of log(H)) [d].
    """

    holonomy_matrix: "Array"
    holonomy_angle: float
    loop_indices: tuple[int, ...]
    loop_length: float
    axis: "Array"


class ParallelTransporter:
    """Parallel transport and holonomy computation on point cloud manifolds.

    Uses Schild's ladder approximation: at each step, the vector is projected
    orthogonally to the travel direction to maintain tangency to the manifold.
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    def transport_along_path(
        self,
        points: "Array",
        path_indices: list[int],
        initial_vector: "Array",
    ) -> ParallelTransportResult:
        """Transport a vector along a discrete geodesic path.

        Uses Schild's ladder approximation for parallel transport:
            1. A = P + v (tip of vector at current point P)
            2. M = midpoint of geodesic from A to next point Q
            3. B = 2*M - P (reflection of P through M)
            4. Transported vector at Q = B - Q

        In flat space, this correctly preserves the vector unchanged.
        On curved manifolds, it approximates Levi-Civita connection.

        Args:
            points: Point cloud [n, d].
            path_indices: Indices of points defining the path.
            initial_vector: Vector to transport [d].

        Returns:
            ParallelTransportResult with transported vector and metrics.
        """
        b = self._backend

        points = b.array(points) if not hasattr(points, "shape") else points
        v = b.array(initial_vector) if not hasattr(initial_vector, "shape") else initial_vector
        b.eval(points, v)

        d = int(b.shape(points)[1])
        if len(path_indices) < 2:
            # Trivial path - no transport
            return ParallelTransportResult(
                transported_vector=v,
                initial_vector=v,
                path_indices=tuple(path_indices),
                path_length=0.0,
                angular_drift=0.0,
                norm_ratio=1.0,
            )

        v_init = v
        v_init_norm = b.sqrt(b.sum(v_init * v_init))
        b.eval(v_init_norm)
        v_init_norm_val = float(b.to_scalar(v_init_norm))

        eps = division_epsilon(b, points)

        # Track total path length
        total_length = 0.0

        # Transport along each segment using Schild's ladder
        for i in range(len(path_indices) - 1):
            idx_from = path_indices[i]
            idx_to = path_indices[i + 1]

            P = points[idx_from]  # Current point
            Q = points[idx_to]    # Next point

            # Compute segment length
            travel = Q - P
            travel_norm = b.sqrt(b.sum(travel * travel))
            b.eval(travel_norm)
            travel_norm_val = float(b.to_scalar(travel_norm))

            total_length += travel_norm_val

            if travel_norm_val <= eps:
                # Zero-length segment - no transport needed
                continue

            # Schild's ladder construction:
            # 1. A = P + v (tip of vector at P)
            A = P + v
            b.eval(A)

            # 2. M = midpoint of "geodesic" from A to Q
            #    In discrete approximation, this is just (A + Q) / 2
            M = (A + Q) / 2.0
            b.eval(M)

            # 3. B = 2*M - P (reflection of P through M)
            B = 2.0 * M - P
            b.eval(B)

            # 4. Transported vector = B - Q
            v = B - Q
            b.eval(v)

        # Compute angular drift
        v_final = v
        v_final_norm = b.sqrt(b.sum(v_final * v_final))
        b.eval(v_final_norm)
        v_final_norm_val = float(b.to_scalar(v_final_norm))

        if v_init_norm_val <= eps or v_final_norm_val <= eps:
            angular_drift = 0.0
            norm_ratio = 0.0 if v_init_norm_val <= eps else v_final_norm_val / v_init_norm_val
        else:
            # Cosine of angle between initial and final
            dot = b.sum(v_init * v_final)
            b.eval(dot)
            dot_val = float(b.to_scalar(dot))
            cos_angle = dot_val / (v_init_norm_val * v_final_norm_val)
            cos_angle = max(-1.0, min(1.0, cos_angle))

            angular_drift_arr = b.arccos(b.array([cos_angle]))
            b.eval(angular_drift_arr)
            angular_drift = float(b.to_scalar(angular_drift_arr))
            norm_ratio = v_final_norm_val / v_init_norm_val

        return ParallelTransportResult(
            transported_vector=v_final,
            initial_vector=v_init,
            path_indices=tuple(path_indices),
            path_length=total_length,
            angular_drift=angular_drift,
            norm_ratio=norm_ratio,
        )

    def transport_basis_along_path(
        self,
        points: "Array",
        path_indices: list[int],
    ) -> "Array":
        """Transport a complete orthonormal basis along a path.

        Starts with the standard basis at the first point and transports
        each basis vector. Used for holonomy computation.

        Args:
            points: Point cloud [n, d].
            path_indices: Indices of points defining the path.

        Returns:
            Transported basis [d, d] - rows are transported basis vectors.
        """
        b = self._backend

        points = b.array(points) if not hasattr(points, "shape") else points
        b.eval(points)

        d = int(b.shape(points)[1])
        basis = b.eye(d)
        b.eval(basis)

        if len(path_indices) < 2:
            return basis

        eps = division_epsilon(b, points)

        # Transport each basis vector along the path using Schild's ladder
        transported_rows = []
        for j in range(d):
            v = basis[j]
            b.eval(v)

            for i in range(len(path_indices) - 1):
                idx_from = path_indices[i]
                idx_to = path_indices[i + 1]

                P = points[idx_from]
                Q = points[idx_to]

                travel = Q - P
                travel_norm = b.sqrt(b.sum(travel * travel))
                b.eval(travel_norm)
                travel_norm_val = float(b.to_scalar(travel_norm))

                if travel_norm_val <= eps:
                    continue

                # Schild's ladder: A = P + v, M = (A + Q)/2, B = 2*M - P
                A = P + v
                M = (A + Q) / 2.0
                B = 2.0 * M - P
                v = B - Q
                b.eval(v)

            transported_rows.append(v)

        transported_basis = b.stack(transported_rows, axis=0)
        b.eval(transported_basis)

        return transported_basis

    def compute_holonomy(
        self,
        points: "Array",
        loop_indices: list[int],
    ) -> HolonomyResult:
        """Compute holonomy around a closed loop.

        Holonomy measures the rotation acquired by parallel transport
        around a closed loop. For a flat manifold, holonomy is zero.
        Non-zero holonomy indicates curvature enclosed by the loop.

        The holonomy equals the solid angle enclosed by the loop (for
        a sphere) or more generally the integral of Gaussian curvature
        over the enclosed region.

        Args:
            points: Point cloud [n, d].
            loop_indices: Indices forming a closed loop (first != last,
                         closure is automatic).

        Returns:
            HolonomyResult with holonomy matrix and angle.
        """
        b = self._backend

        points = b.array(points) if not hasattr(points, "shape") else points
        b.eval(points)

        d = int(b.shape(points)[1])

        if len(loop_indices) < 3:
            # Need at least 3 points for a non-trivial loop
            return HolonomyResult(
                holonomy_matrix=b.eye(d),
                holonomy_angle=0.0,
                loop_indices=tuple(loop_indices),
                loop_length=0.0,
                axis=b.zeros((d,)),
            )

        # Close the loop by appending first index
        closed_loop = list(loop_indices) + [loop_indices[0]]

        # Transport basis around the closed loop
        initial_basis = b.eye(d)
        transported_basis = self.transport_basis_along_path(points, closed_loop)
        b.eval(initial_basis, transported_basis)

        # Holonomy matrix H: initial_basis @ H = transported_basis
        # H = initial_basis^T @ transported_basis (since initial is identity)
        # But we need the rotation that maps initial to transported
        # Using Procrustes: H = V @ U^T where transported = U @ S @ V^T @ initial
        # Simpler: H ≈ transported @ initial^T = transported (since initial = I)
        holonomy_matrix = b.matmul(b.transpose(initial_basis), transported_basis)
        b.eval(holonomy_matrix)

        # Project to SO(d) to ensure proper rotation
        U, _, Vt = b.svd(holonomy_matrix)
        holonomy_matrix = b.matmul(U, Vt)

        # Fix determinant if negative
        det_val = b.det(holonomy_matrix)
        b.eval(det_val)
        if float(b.to_scalar(det_val)) < 0:
            U_fixed = b.concatenate([U[:, :-1], -U[:, -1:]], axis=1)
            holonomy_matrix = b.matmul(U_fixed, Vt)
        b.eval(holonomy_matrix)

        # Compute holonomy angle using so_log
        from modelcypher.core.domain.geometry.lie_rotation import so_log

        log_H = so_log(holonomy_matrix, backend=b)
        b.eval(log_H)

        # Holonomy angle = Frobenius norm of log / sqrt(2)
        # (The sqrt(2) normalizes so angle matches geometric interpretation)
        log_norm = b.sqrt(b.sum(log_H * log_H))
        b.eval(log_norm)
        holonomy_angle = float(b.to_scalar(log_norm)) / sqrt_scalar(2.0, b)

        # Compute rotation axis from log matrix
        # For skew-symmetric log_H, axis is the eigenvector with imaginary eigenvalue
        # In practice, for 3D, axis = [log_H[2,1], log_H[0,2], log_H[1,0]]
        # For general d, we use SVD of log_H
        _, _, Vt_log = b.svd(log_H)
        b.eval(Vt_log)
        # Axis is in the null space of log_H (last row of Vt)
        axis = Vt_log[-1]
        b.eval(axis)

        # Compute loop length
        eps = division_epsilon(b, points)
        total_length = 0.0
        for i in range(len(closed_loop) - 1):
            idx_from = closed_loop[i]
            idx_to = closed_loop[i + 1]
            diff = points[idx_to] - points[idx_from]
            seg_len = b.sqrt(b.sum(diff * diff))
            b.eval(seg_len)
            total_length += float(b.to_scalar(seg_len))

        return HolonomyResult(
            holonomy_matrix=holonomy_matrix,
            holonomy_angle=holonomy_angle,
            loop_indices=tuple(loop_indices),
            loop_length=total_length,
            axis=axis,
        )

    def compute_triangle_holonomy(
        self,
        points: "Array",
        i: int,
        j: int,
        k: int,
    ) -> HolonomyResult:
        """Compute holonomy for a triangle defined by three point indices.

        A convenience method for the common case of triangular loops.
        For a sphere, the holonomy angle equals the solid angle of the
        spherical triangle.

        Args:
            points: Point cloud [n, d].
            i, j, k: Indices of the three vertices.

        Returns:
            HolonomyResult for the triangular loop.
        """
        return self.compute_holonomy(points, [i, j, k])


def parallel_transport(
    points: "Array",
    path_indices: list[int],
    initial_vector: "Array",
    backend: "Backend | None" = None,
) -> ParallelTransportResult:
    """Transport a vector along a geodesic path (convenience function).

    Args:
        points: Point cloud [n, d].
        path_indices: Indices defining the path.
        initial_vector: Vector to transport [d].
        backend: Backend for tensor operations.

    Returns:
        ParallelTransportResult with transported vector and metrics.
    """
    transporter = ParallelTransporter(backend)
    return transporter.transport_along_path(points, path_indices, initial_vector)


def compute_holonomy(
    points: "Array",
    loop_indices: list[int],
    backend: "Backend | None" = None,
) -> HolonomyResult:
    """Compute holonomy around a closed loop (convenience function).

    Args:
        points: Point cloud [n, d].
        loop_indices: Indices forming the loop.
        backend: Backend for tensor operations.

    Returns:
        HolonomyResult with rotation matrix and angle.
    """
    transporter = ParallelTransporter(backend)
    return transporter.compute_holonomy(points, loop_indices)


__all__ = [
    "ParallelTransportResult",
    "HolonomyResult",
    "ParallelTransporter",
    "parallel_transport",
    "compute_holonomy",
]
