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

Implements parallel transport via local tangent frame rotation. On a discrete
point cloud approximating a manifold, curvature is detected by how the local
tangent space rotates from point to point. Parallel transport applies this
rotation to vectors.

Algorithm:
    1. At each point, estimate local tangent frame from k-NN covariance (PCA)
    2. When moving from P to Q, compute the optimal rotation R that aligns
       the tangent frame at P to the tangent frame at Q
    3. Apply R to the vector

This captures manifold curvature: on a flat region, tangent frames are
parallel and transport is identity. On curved regions, tangent frames rotate
and so do transported vectors.

References:
    - Pennec, X. (2006). "Intrinsic statistics on Riemannian manifolds."
    - Singer, A., Wu, H.-T. (2012). "Vector diffusion maps and the connection
      Laplacian." Communications on Pure and Applied Mathematics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.lie_rotation import so_log
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
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

    Uses local tangent frame rotation: at each step, compute the rotation
    that aligns the source tangent frame to the destination tangent frame,
    and apply that rotation to the transported vector.

    This captures manifold curvature through the rotation of local geometry.

    All parameters are derived from the data:
    - k: d + 1 (algebraic minimum to span d-dimensional tangent space)
    - intrinsic_dim: from eigenvalue spectrum (significant eigenvalues only)

    No configuration needed. The geometry determines everything.
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        """Initialize transporter.

        Args:
            backend: Compute backend.
        """
        self._backend = backend or get_default_backend()

    def transport_along_path(
        self,
        points: "Array",
        path_indices: list[int],
        initial_vector: "Array",
    ) -> ParallelTransportResult:
        """Transport a vector along a discrete geodesic path.

        Uses local tangent frame rotation: at each step P -> Q, compute
        the rotation R that aligns the local tangent frame at P to the
        local tangent frame at Q, then apply R to the vector.

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

        n = int(b.shape(points)[0])
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

        # Precompute distance matrix for k-NN
        distances = self._compute_distance_matrix(points, b)

        # Compute centroid as global reference for consistent frame orientation
        centroid = b.mean(points, axis=0)
        b.eval(centroid)

        # Track total path length
        total_length = 0.0

        # Transport along each segment
        for i in range(len(path_indices) - 1):
            idx_from = path_indices[i]
            idx_to = path_indices[i + 1]

            P = points[idx_from]
            Q = points[idx_to]

            # Compute segment length
            travel = Q - P
            travel_norm = b.sqrt(b.sum(travel * travel))
            b.eval(travel_norm)
            travel_norm_val = float(b.to_scalar(travel_norm))

            total_length += travel_norm_val

            if travel_norm_val <= eps:
                continue

            # Use direction from centroid to each point as reference
            # This gives consistent frame orientation across the manifold
            ref_P = P - centroid
            ref_Q = Q - centroid
            b.eval(ref_P, ref_Q)

            # Compute local tangent frames at P and Q with consistent orientation
            result_P = self._compute_local_frame(points, idx_from, distances, b, ref_P)
            result_Q = self._compute_local_frame(points, idx_to, distances, b, ref_Q)

            if result_P is None or result_Q is None:
                # Can't compute frames - fall back to identity
                continue

            frame_P, eigenvalues_P = result_P
            frame_Q, eigenvalues_Q = result_Q

            # Compute optimal rotation from frame_P to frame_Q
            R = self._compute_frame_rotation(frame_P, frame_Q, eigenvalues_P, eigenvalues_Q, b)

            # Apply rotation to vector
            v = b.matmul(R, v)
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

    def _compute_distance_matrix(self, points: "Array", backend: "Backend") -> "Array":
        """Compute pairwise distance matrix."""
        b = backend
        n = int(b.shape(points)[0])

        p1 = b.expand_dims(points, 1)  # [n, 1, d]
        p2 = b.expand_dims(points, 0)  # [1, n, d]
        diff = p1 - p2  # [n, n, d]
        distances = b.sqrt(b.sum(diff * diff, axis=2))  # [n, n]
        b.eval(distances)

        return distances

    def _compute_local_frame(
        self,
        points: "Array",
        idx: int,
        distances: "Array",
        backend: "Backend",
        reference_direction: "Array | None" = None,
    ) -> "tuple[Array, Array] | None":
        """Compute local tangent frame at a point via PCA of k-NN.

        The eigenvectors are oriented consistently using a reference direction.
        This resolves the sign ambiguity inherent in PCA/SVD eigenvectors.

        Args:
            points: Point cloud [n, d].
            idx: Index of the center point.
            distances: Precomputed distance matrix [n, n].
            backend: Compute backend.
            reference_direction: Direction to align the first eigenvector with.
                If None, uses the direction from origin to the point.

        Returns:
            Tuple of (frame, eigenvalues) where frame is [d, d] with columns as
            principal directions (consistently oriented), and eigenvalues is [d]
            sorted descending. Returns None if computation fails.
        """
        b = backend
        n = int(b.shape(points)[0])
        d = int(b.shape(points)[1])

        # Dtype-derived threshold
        eps = machine_epsilon(b, points)
        sqrt_arr = b.sqrt(b.array([eps]))
        b.eval(sqrt_arr)
        sqrt_eps = float(b.to_scalar(sqrt_arr))

        # k = d + 1: algebraic minimum to span a d-dimensional tangent space
        # This is the smallest k that guarantees non-degenerate covariance in d dimensions
        k = min(d + 1, n - 1)
        if k < 2:
            return None

        # Get k nearest neighbors
        row = distances[idx]
        b.eval(row)

        # Set self-distance to infinity
        row_mod = b.where(
            b.arange(0, n) == idx,
            b.full((n,), float("inf")),
            row,
        )
        b.eval(row_mod)

        # Get k smallest
        sorted_idx = b.argsort(row_mod)
        b.eval(sorted_idx)

        neighbor_indices = []
        for j in range(k):
            neighbor_indices.append(int(b.to_scalar(sorted_idx[j])))

        # Compute difference vectors from center point
        center = points[idx]
        b.eval(center)

        diffs = []
        for j in neighbor_indices:
            diff = points[j] - center
            b.eval(diff)
            diffs.append(diff)

        diff_matrix = b.stack(diffs, axis=0)  # [k, d]
        b.eval(diff_matrix)

        # Covariance matrix
        cov = b.matmul(b.transpose(diff_matrix), diff_matrix)  # [d, d]
        b.eval(cov)

        # SVD for principal directions
        try:
            U, S, Vt = b.svd(cov, compute_uv=True)
            b.eval(U, S)

            # Orient eigenvectors consistently
            # Use reference direction (or point position) to resolve sign ambiguity
            if reference_direction is None:
                # Use direction from origin to point as reference
                ref = center
            else:
                ref = reference_direction
            b.eval(ref)

            ref_norm = b.sqrt(b.sum(ref * ref))
            b.eval(ref_norm)
            ref_norm_val = float(b.to_scalar(ref_norm))

            if ref_norm_val > sqrt_eps:
                # Orient each eigenvector to have non-negative dot with reference
                # This gives consistent signs across the manifold
                oriented_cols = []
                for j in range(d):
                    col = U[:, j]
                    b.eval(col)
                    dot = b.sum(col * ref)
                    b.eval(dot)
                    dot_val = float(b.to_scalar(dot))

                    if dot_val < 0:
                        col = -col
                    oriented_cols.append(col)

                U = b.stack(oriented_cols, axis=1)
                b.eval(U)

            # Ensure right-handedness (det = +1) by checking determinant
            # and flipping last column if needed
            det = b.det(U)
            b.eval(det)
            det_val = float(b.to_scalar(det))

            if det_val < 0:
                # Flip last column to make det positive
                flip_cols = []
                for j in range(d):
                    col = U[:, j]
                    b.eval(col)
                    if j == d - 1:
                        col = -col
                    flip_cols.append(col)
                U = b.stack(flip_cols, axis=1)
                b.eval(U)

            # Return frame with eigenvalues for rank detection
            return (U, S)  # ([d, d], [d])
        except Exception:
            return None

    def _estimate_intrinsic_dim_from_eigenvalues(
        self, eigenvalues: "Array", backend: "Backend"
    ) -> int:
        """Estimate intrinsic dimension from eigenvalue spectrum.

        Uses the ratio threshold: eigenvalue > max_eigenvalue * sqrt(machine_epsilon)
        is considered significant.
        """
        b = backend
        b.eval(eigenvalues)

        eps = machine_epsilon(b, eigenvalues)
        sqrt_arr = b.sqrt(b.array([eps]))
        b.eval(sqrt_arr)
        sqrt_eps = float(b.to_scalar(sqrt_arr))

        n_eigs = int(b.shape(eigenvalues)[0])
        max_eig = float(b.to_scalar(eigenvalues[0]))  # Eigenvalues sorted descending

        if max_eig < sqrt_eps:
            return 0  # All degenerate

        threshold = max_eig * sqrt_eps
        count = 0
        for i in range(n_eigs):
            eig_val = float(b.to_scalar(eigenvalues[i]))
            if eig_val > threshold:
                count += 1
            else:
                break
        return max(1, count)

    def _compute_frame_rotation(
        self,
        frame_P: "Array",
        frame_Q: "Array",
        eigenvalues_P: "Array",
        eigenvalues_Q: "Array",
        backend: "Backend",
    ) -> "Array":
        """Compute the parallel transport rotation from frame_P to frame_Q.

        Uses the manifold structure to determine the rotation:
        - For 1D curves: identity (no rotation in 1D tangent space)
        - For co-dimension 1 in 3D (r=2, d=3): Rodrigues' formula on normals
        - For r < d: Procrustes on tangent subspace only
        - For r = d: full Procrustes

        The key insight: on a curved manifold, the tangent frame rotates as we
        move along the surface. The transport rotation aligns these frames.
        """
        b = backend
        d = int(b.shape(frame_P)[0])
        eps = machine_epsilon(b, frame_P)
        sqrt_arr = b.sqrt(b.array([eps]))
        b.eval(sqrt_arr)
        sqrt_eps = float(b.to_scalar(sqrt_arr))

        # Intrinsic dimension from eigenvalue spectrum of BOTH frames
        # Use minimum to ensure we only use eigenvectors that are significant in both
        r_P = self._estimate_intrinsic_dim_from_eigenvalues(eigenvalues_P, b)
        r_Q = self._estimate_intrinsic_dim_from_eigenvalues(eigenvalues_Q, b)
        r = min(r_P, r_Q)

        # For 1D curves, parallel transport is identity (tangent space is 1D, no rotation)
        if r <= 1:
            return b.eye(d)

        # For d=3 with co-dimension 1 (r = 2), use Rodrigues formula on normals
        # This is most accurate for 2D surfaces in 3D - check BEFORE general r < d case
        if d == 3 and r == 2:
            # The normal direction is the last column (smallest eigenvalue from PCA)
            normal_P = frame_P[:, d - 1]
            normal_Q = frame_Q[:, d - 1]
            b.eval(normal_P, normal_Q)

            # Normalize
            norm_P = b.sqrt(b.sum(normal_P * normal_P))
            norm_Q = b.sqrt(b.sum(normal_Q * normal_Q))
            b.eval(norm_P, norm_Q)

            div_eps = division_epsilon(b, frame_P)

            if float(b.to_scalar(norm_P)) <= div_eps or float(b.to_scalar(norm_Q)) <= div_eps:
                return b.eye(d)

            n_P = normal_P / norm_P
            n_Q = normal_Q / norm_Q
            b.eval(n_P, n_Q)

            # cos(θ) = n_P · n_Q
            cos_theta = b.sum(n_P * n_Q)
            b.eval(cos_theta)
            cos_theta_val = float(b.to_scalar(cos_theta))

            # If normals are nearly parallel, return identity
            if abs(cos_theta_val - 1.0) < sqrt_eps:
                return b.eye(d)

            # If normals are nearly anti-parallel, compute 180° rotation
            if abs(cos_theta_val + 1.0) < sqrt_eps:
                # Find perpendicular axis for 180° rotation
                # Use whichever standard basis vector is least parallel to n_P
                dots = [abs(float(b.to_scalar(n_P[i]))) for i in range(d)]
                min_idx = dots.index(min(dots))
                perp_list = [0.0] * d
                perp_list[min_idx] = 1.0
                perp = b.array(perp_list)
                # Gram-Schmidt to make perpendicular
                perp = perp - b.sum(perp * n_P) * n_P
                perp = perp / b.sqrt(b.sum(perp * perp))
                b.eval(perp)
                # 180° rotation around perp: R = 2*perp*perp^T - I
                # Use expand_dims + matmul for outer product
                perp_col = b.expand_dims(perp, 1)  # [d, 1]
                perp_row = b.expand_dims(perp, 0)  # [1, d]
                outer = b.matmul(perp_col, perp_row)  # [d, d]
                R = 2.0 * outer - b.eye(d)
                b.eval(R)
                return R

            # k = n_P × n_Q (cross product)
            k0 = n_P[1] * n_Q[2] - n_P[2] * n_Q[1]
            k1 = n_P[2] * n_Q[0] - n_P[0] * n_Q[2]
            k2 = n_P[0] * n_Q[1] - n_P[1] * n_Q[0]
            k = b.stack([k0, k1, k2], axis=0)
            b.eval(k)

            k_norm = b.sqrt(b.sum(k * k))
            b.eval(k_norm)
            k_norm_val = float(b.to_scalar(k_norm))

            if k_norm_val < div_eps:
                return b.eye(d)

            # Normalize rotation axis
            k = k / k_norm
            b.eval(k)

            # sin(θ) = ||n_P × n_Q|| = k_norm (since n_P and n_Q are unit)
            sin_theta = k_norm

            # Skew-symmetric matrix K from axis k
            k0_val = k[0]
            k1_val = k[1]
            k2_val = k[2]
            b.eval(k0_val, k1_val, k2_val)

            zero = b.array([0.0])
            K_row0 = b.stack([zero[0], -k2_val, k1_val], axis=0)
            K_row1 = b.stack([k2_val, zero[0], -k0_val], axis=0)
            K_row2 = b.stack([-k1_val, k0_val, zero[0]], axis=0)
            K = b.stack([K_row0, K_row1, K_row2], axis=0)
            b.eval(K)

            # Rodrigues: R = I + sin(θ)*K + (1-cos(θ))*K²
            I = b.eye(d)
            K2 = b.matmul(K, K)
            b.eval(K2)

            one_minus_cos = 1.0 - cos_theta_val
            R = I + sin_theta * K + one_minus_cos * K2
            b.eval(R)

            return R

        # For r < d (but not the d=3, r=2 case handled above), construct rotation that:
        # 1. Rotates tangent subspace (first r columns of frame) via Procrustes
        # 2. Maps normal subspace (remaining d-r columns) consistently
        #
        # Since frame_P and frame_Q are full orthonormal bases:
        #   R = frame_Q @ block_diag(R_tangent, I_{d-r}) @ frame_P.T
        # This ensures normal components map normal-to-normal without arbitrary rotation.
        if r < d:
            # Extract tangent frames (first r columns = largest eigenvalues)
            tangent_P = frame_P[:, :r]  # [d, r]
            tangent_Q = frame_Q[:, :r]  # [d, r]
            b.eval(tangent_P, tangent_Q)

            try:
                # Procrustes on tangent subspace: M_small = tangent_Q.T @ tangent_P is [r, r]
                M_small = b.matmul(b.transpose(tangent_Q), tangent_P)
                b.eval(M_small)

                U_small, S_small, Vt_small = b.svd(M_small, compute_uv=True)
                b.eval(U_small, Vt_small)

                R_tangent = b.matmul(U_small, Vt_small)  # [r, r]
                b.eval(R_tangent)

                # Ensure proper rotation (det = +1) in tangent subspace
                det_tangent = b.det(R_tangent)
                b.eval(det_tangent)
                det_tangent_val = float(b.to_scalar(det_tangent))

                if det_tangent_val < 0:
                    flip_list = [
                        [1.0 if i == j else 0.0 for j in range(r)] for i in range(r)
                    ]
                    flip_list[-1][-1] = -1.0
                    flip = b.array(flip_list)
                    R_tangent = b.matmul(b.matmul(U_small, flip), Vt_small)
                    b.eval(R_tangent)

                # Construct block diagonal: block_diag(R_tangent, I_{d-r})
                # Build as a full d×d matrix
                block_rows = []
                for i in range(d):
                    row_vals = []
                    for j in range(d):
                        if i < r and j < r:
                            # Upper-left r×r block: R_tangent
                            val = R_tangent[i, j]
                        elif i >= r and j >= r and i == j:
                            # Lower-right (d-r)×(d-r) block: identity
                            val = b.array([1.0])
                        else:
                            # Off-diagonal blocks: zeros
                            val = b.array([0.0])
                        b.eval(val)
                        row_vals.append(float(b.to_scalar(val)))
                    block_rows.append(row_vals)
                block_diag = b.array(block_rows)
                b.eval(block_diag)

                # R = frame_Q @ block_diag @ frame_P.T
                R = b.matmul(b.matmul(frame_Q, block_diag), b.transpose(frame_P))
                b.eval(R)

                return R

            except Exception:
                return b.eye(d)

        # For r == d, full Procrustes on entire frame
        M = b.matmul(frame_Q, b.transpose(frame_P))
        b.eval(M)

        try:
            U, S, Vt = b.svd(M, compute_uv=True)
            b.eval(U, Vt)

            R = b.matmul(U, Vt)
            b.eval(R)

            # Ensure proper rotation (det = +1)
            det = b.det(R)
            b.eval(det)
            det_val = float(b.to_scalar(det))

            if det_val < 0:
                flip_list = [
                    [1.0 if i == j else 0.0 for j in range(d)] for i in range(d)
                ]
                flip_list[-1][-1] = -1.0
                flip = b.array(flip_list)
                R = b.matmul(b.matmul(U, flip), Vt)
                b.eval(R)

            return R

        except Exception:
            return b.eye(d)

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

        # Transport each basis vector
        transported_rows = []
        for j in range(d):
            v = basis[j]
            result = self.transport_along_path(points, path_indices, v)
            transported_rows.append(result.transported_vector)

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

        # Dtype-derived threshold
        eps = machine_epsilon(b, points)
        sqrt_arr = b.sqrt(b.array([eps]))
        b.eval(sqrt_arr)
        sqrt_eps = float(b.to_scalar(sqrt_arr))

        if len(loop_indices) < 3:
            # Need at least 3 points for a non-trivial loop
            return HolonomyResult(
                holonomy_matrix=b.eye(d),
                holonomy_angle=0.0,
                loop_indices=tuple(loop_indices),
                loop_length=0.0,
                axis=b.zeros((d,)),
            )

        # Close the loop
        closed_loop = list(loop_indices) + [loop_indices[0]]

        # Transport basis around the loop
        transported_basis = self.transport_basis_along_path(points, closed_loop)
        b.eval(transported_basis)

        # Holonomy matrix: H = transported_basis (rows are transported vectors)
        # This should be close to identity for flat space
        H = transported_basis

        # Compute loop length
        total_length = 0.0
        for i in range(len(closed_loop) - 1):
            p1 = points[closed_loop[i]]
            p2 = points[closed_loop[i + 1]]
            diff = p2 - p1
            dist = b.sqrt(b.sum(diff * diff))
            b.eval(dist)
            total_length += float(b.to_scalar(dist))

        # Holonomy angle via so_log (Lie algebra)
        # log(H) is skew-symmetric; its Frobenius norm / sqrt(2) is the rotation angle
        try:
            log_H = so_log(H, backend=b, project=True)
            b.eval(log_H)

            # ||log(H)||_F / sqrt(2) = rotation angle for SO(n)
            log_norm_sq = b.sum(log_H * log_H)
            b.eval(log_norm_sq)
            holonomy_angle = float(b.to_scalar(b.sqrt(log_norm_sq / 2.0)))

            # For 3D, extract rotation axis from skew-symmetric log(H)
            # log(H) = θ * [0, -k3, k2; k3, 0, -k1; -k2, k1, 0]
            # where k = (k1, k2, k3) is the unit axis
            if d == 3 and holonomy_angle > sqrt_eps:
                # Extract axis from skew-symmetric matrix
                axis_vals = [
                    float(b.to_scalar(log_H[2, 1] - log_H[1, 2])) / 2.0,
                    float(b.to_scalar(log_H[0, 2] - log_H[2, 0])) / 2.0,
                    float(b.to_scalar(log_H[1, 0] - log_H[0, 1])) / 2.0,
                ]
                axis_norm = (sum(v * v for v in axis_vals)) ** 0.5
                if axis_norm > sqrt_eps:
                    axis = b.array([v / axis_norm for v in axis_vals])
                else:
                    axis = b.zeros((d,))
            else:
                axis = b.zeros((d,))
            b.eval(axis)

        except Exception:
            # Fallback to Frobenius norm of (H - I) if so_log fails
            I = b.eye(d)
            diff_matrix = H - I
            angle_sq = b.sum(diff_matrix * diff_matrix)
            b.eval(angle_sq)
            holonomy_angle = float(b.to_scalar(b.sqrt(angle_sq)))
            axis = b.zeros((d,))

        return HolonomyResult(
            holonomy_matrix=H,
            holonomy_angle=holonomy_angle,
            loop_indices=tuple(loop_indices),
            loop_length=total_length,
            axis=axis,
        )

    def compute_triangle_holonomy(
        self,
        points: "Array",
        idx_a: int,
        idx_b: int,
        idx_c: int,
    ) -> HolonomyResult:
        """Compute holonomy around a triangle.

        Convenience method for computing holonomy around three points.

        Args:
            points: Point cloud [n, d].
            idx_a, idx_b, idx_c: Indices of triangle vertices.

        Returns:
            HolonomyResult for the triangle loop.
        """
        return self.compute_holonomy(points, [idx_a, idx_b, idx_c])


def parallel_transport(
    points: "Array",
    path_indices: list[int],
    initial_vector: "Array",
    backend: "Backend | None" = None,
) -> ParallelTransportResult:
    """Transport a vector along a path (convenience function).

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
        loop_indices: Indices forming a closed loop.
        backend: Backend for tensor operations.

    Returns:
        HolonomyResult with holonomy matrix and angle.
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
