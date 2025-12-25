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

"""
Gromov-Wasserstein distance computation for representation space comparison.

GPU-accelerated implementation using the Backend protocol (MLX/JAX/CUDA).

Mathematical Foundation:
    The Gromov-Wasserstein distance measures structural similarity between
    metric spaces without requiring point-to-point correspondence. Given
    source (X, dX) and target (Y, dY) metric spaces with probability measures
    μ and ν, the GW objective minimizes:

        GW(μ, ν) = min_γ ∑_{i,j,k,l} L(dX(xi, xk), dY(yj, yl)) · γij · γkl

    where γ is a coupling matrix with marginals μ and ν.

Algorithm:
    This implementation uses the Conditional Gradient (Frank-Wolfe) algorithm
    following Peyré, Cuturi, and Solomon (2016) "Gromov-Wasserstein Averaging
    of Kernel and Distance Matrices" (ICML).

    Key insight: For squared loss L(a,b) = (a-b)², the objective decomposes as:
        L(a,b) = a² + b² - 2ab = f₁(a) + f₂(b) - h₁(a)·h₂(b)

    where f₁(a) = a², f₂(b) = b², h₁(a) = a, h₂(b) = 2b.

    This allows O(n²m + nm²) tensor product computation instead of O(n²m²).

    Frank-Wolfe iteration:
        1. Compute gradient via tensor product
        2. Solve LINEAR OT problem (not full GW) to get descent direction
        3. Line search for optimal step size (analytic for quadratic)
        4. Update coupling: T ← (1-α)T + αG

Complexity:
    O(n²m + nm²) per outer iteration for gradient computation.
    Linear OT subproblem: O(nm log nm) with Sinkhorn.

References:
    - Peyré, Cuturi, Solomon (2016) "GW Averaging" ICML
    - Peyré & Cuturi (2019) "Computational Optimal Transport"
    - POT library: https://pythonot.github.io/

See also: docs/geometry/gromov_wasserstein.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class Result:
    distance: float
    coupling: "Array"
    converged: bool
    iterations: int

    @property
    def normalized_distance(self) -> float:
        return 1.0 - math.exp(-self.distance) if math.isfinite(self.distance) else 1.0

    @property
    def compatibility_score(self) -> float:
        return math.exp(-self.distance) if math.isfinite(self.distance) else 0.0


@dataclass(frozen=True)
class Config:
    # Frank-Wolfe parameters
    max_outer_iterations: int = 100
    min_outer_iterations: int = 5
    convergence_threshold: float = 1e-7
    relative_objective_threshold: float = 1e-7

    # Linear OT subproblem (Sinkhorn)
    # Small epsilon approximates exact EMD better
    sinkhorn_epsilon: float = 0.001
    sinkhorn_iterations: int = 200
    sinkhorn_threshold: float = 1e-8

    # Loss function
    use_squared_loss: bool = True

    # Random restarts to escape local minima (GW is non-convex)
    # More restarts = better chance of finding global minimum
    num_restarts: int = 10
    seed: int | None = 42  # Fixed seed for reproducibility

    # Symmetry: GW is mathematically symmetric (GW(A,B) = GW(B,A)), but
    # non-convex optimization may find different local minima.
    # When True, computes both directions and returns the minimum.
    ensure_symmetry: bool = True

    # k for geodesic k-NN graph (None = auto). Geodesic distance is always used
    # because curvature is inherent in high-dimensional spaces.
    geodesic_k_neighbors: int | None = None

    # Legacy parameters (kept for backward compatibility)
    epsilon: float = 0.05
    epsilon_min: float = 0.005
    epsilon_decay: float = 0.9
    max_inner_iterations: int = 100


class GromovWassersteinDistance:
    """GPU-accelerated Gromov-Wasserstein distance using Frank-Wolfe algorithm."""

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    def compute(
        self,
        source_distances: "Array",
        target_distances: "Array",
        config: Config = Config(),
    ) -> Result:
        """
        Compute Gromov-Wasserstein distance between two metric spaces.

        Uses Conditional Gradient (Frank-Wolfe) algorithm with multiple restarts
        to escape local minima (GW is non-convex).

        Args:
            source_distances: Pairwise distance matrix for source [n, n]
            target_distances: Pairwise distance matrix for target [m, m]
            config: Algorithm configuration

        Returns:
            Result with distance, coupling matrix, and convergence info
        """
        import numpy as np

        backend = self._backend

        # Convert inputs to backend arrays if needed
        C1 = backend.array(source_distances)
        C2 = backend.array(target_distances)
        backend.eval(C1, C2)

        n = int(C1.shape[0])
        m = int(C2.shape[0])

        if n == 0 or m == 0:
            return Result(
                distance=float("inf"),
                coupling=backend.zeros((0, 0)),
                converged=False,
                iterations=0,
            )

        # Check for identical matrices (distance = 0)
        if n == m:
            diff = backend.abs(C1 - C2)
            max_diff = backend.max(diff)
            backend.eval(max_diff)
            if float(backend.to_numpy(max_diff)) < 1e-10:
                # Identical - return identity coupling
                coupling = backend.eye(n) / n
                return Result(distance=0.0, coupling=coupling, converged=True, iterations=0)

        # For small square matrices, exhaustively search permutations
        # GW with uniform marginals is equivalent to the Quadratic Assignment Problem (QAP)
        # which has n! solutions. For n≤8, exhaustive search is tractable and exact.
        if n == m and n <= 8:
            return self._solve_by_permutation_search(C1, C2, n, backend)

        # Uniform marginals
        p = backend.ones((n,)) / n
        q = backend.ones((m,)) / m

        # Multiple restarts to escape local minima
        best_result: Result | None = None
        total_iterations = 0
        num_restarts = max(1, config.num_restarts)

        rng = np.random.default_rng(config.seed)

        for restart in range(num_restarts):
            # Generate initial coupling
            if restart == 0:
                # First: uniform coupling (outer product of marginals)
                T0 = backend.matmul(backend.reshape(p, (n, 1)), backend.reshape(q, (1, m)))
            else:
                # Random perturbation of uniform, projected to valid transport polytope
                T0 = self._random_coupling(n, m, rng, backend)

            result = self._frank_wolfe(C1, C2, p, q, T0, config)
            total_iterations += result.iterations

            if best_result is None or result.distance < best_result.distance:
                best_result = result

            # Early termination if we found near-zero distance
            if result.distance < 1e-8:
                break

        assert best_result is not None
        return Result(
            distance=best_result.distance,
            coupling=best_result.coupling,
            converged=best_result.converged,
            iterations=total_iterations,
        )

    def _random_coupling(self, n: int, m: int, rng, backend: "Backend") -> "Array":
        """Generate a random valid coupling matrix with uniform marginals."""
        # Random positive matrix
        coupling = rng.uniform(0.1, 1.0, (n, m))

        # Project onto transport polytope via Sinkhorn iterations
        for _ in range(20):
            coupling = coupling / coupling.sum(axis=1, keepdims=True) * (1.0 / n)
            coupling = coupling / coupling.sum(axis=0, keepdims=True) * (1.0 / m)

        return backend.array(coupling)

    def _solve_by_permutation_search(
        self,
        C1: "Array",
        C2: "Array",
        n: int,
        backend: "Backend",
    ) -> Result:
        """
        Solve GW by exhaustive permutation search for small matrices.

        For uniform marginals, GW simplifies to the Quadratic Assignment Problem:
            min_P sum_{i,j,k,l} (C1[i,k] - C2[j,l])^2 * P[i,j] * P[k,l]

        where P is a permutation matrix (scaled by 1/n for proper marginals).

        This is equivalent to: min_σ sum_{i,k} (C1[i,k] - C2[σ(i),σ(k)])^2 / n^2

        Complexity: O(n! * n^2) - tractable for n ≤ 8.
        """
        import itertools

        # Convert to numpy for fast iteration
        C1_np = backend.to_numpy(C1)
        C2_np = backend.to_numpy(C2)

        best_loss = float("inf")
        best_perm = None

        # Try all n! permutations
        for perm in itertools.permutations(range(n)):
            # Compute GW loss for this permutation
            # loss = sum_{i,k} (C1[i,k] - C2[perm[i], perm[k]])^2 / n^2
            loss = 0.0
            for i in range(n):
                for k in range(n):
                    diff = C1_np[i, k] - C2_np[perm[i], perm[k]]
                    loss += diff * diff
            loss /= n * n

            if loss < best_loss:
                best_loss = loss
                best_perm = perm

        # Build coupling matrix from best permutation
        # T[i, perm[i]] = 1/n
        coupling_np = [[0.0] * n for _ in range(n)]
        for i, j in enumerate(best_perm):
            coupling_np[i][j] = 1.0 / n
        coupling = backend.array(coupling_np)

        return Result(
            distance=best_loss,
            coupling=coupling,
            converged=True,
            iterations=1,  # Single "iteration" to search all perms
        )

    def _init_loss_matrices(
        self,
        C1: "Array",
        C2: "Array",
        p: "Array",
        q: "Array",
    ) -> tuple["Array", "Array", "Array"]:
        """
        Initialize constant matrices for efficient loss decomposition.

        For squared loss L(a,b) = (a-b)² = a² + b² - 2ab:
            f₁(a) = a², f₂(b) = b², h₁(a) = a, h₂(b) = 2b

        Returns:
            constC: Constant part of tensor product [n, m]
            hC1: h₁(C1) = C1 [n, n]
            hC2: h₂(C2) = 2*C2 [m, m]
        """
        backend = self._backend
        n = C1.shape[0]
        m = C2.shape[0]

        # f(C) = C² for squared loss
        fC1 = C1 * C1  # [n, n]
        fC2 = C2 * C2  # [m, m]

        # h₁(C1) = C1, h₂(C2) = 2*C2 for squared loss
        hC1 = C1
        hC2 = 2.0 * C2

        # constC = fC1 @ p @ 1ᵀ + 1 @ qᵀ @ fC2ᵀ
        # constC[i,j] = sum_k fC1[i,k] * p[k] + sum_l fC2[j,l] * q[l]
        p_col = backend.reshape(p, (n, 1))  # [n, 1]
        q_row = backend.reshape(q, (1, m))  # [1, m]
        ones_row = backend.ones((1, m))  # [1, m]
        ones_col = backend.ones((n, 1))  # [n, 1]

        # fC1 @ p gives [n, 1], broadcast to [n, m]
        constC1 = backend.matmul(backend.matmul(fC1, p_col), ones_row)  # [n, m]

        # fC2.T @ q gives [m, 1], transpose and broadcast to [n, m]
        constC2 = backend.matmul(ones_col, backend.matmul(q_row, fC2))  # [n, m]

        constC = constC1 + constC2

        return constC, hC1, hC2

    def _tensor_product(
        self,
        constC: "Array",
        hC1: "Array",
        hC2: "Array",
        T: "Array",
    ) -> "Array":
        """
        Compute tensor product efficiently using loss decomposition.

        tens[i,j] = constC[i,j] - sum_{k,l} hC1[i,k] * T[k,l] * hC2[l,j]
                  = constC[i,j] - (hC1 @ T @ hC2ᵀ)[i,j]

        Complexity: O(n²m + nm²) instead of O(n²m²)
        """
        backend = self._backend
        # hC1 @ T @ hC2.T
        inner = backend.matmul(backend.matmul(hC1, T), backend.transpose(hC2))
        return constC - inner

    def _gw_loss(
        self,
        constC: "Array",
        hC1: "Array",
        hC2: "Array",
        T: "Array",
    ) -> float:
        """
        Compute GW loss using tensor product.

        loss = <T, tensor_product(T)> = sum_{i,j} T[i,j] * tens[i,j]
        """
        backend = self._backend
        tens = self._tensor_product(constC, hC1, hC2, T)
        loss_arr = backend.sum(tens * T)
        backend.eval(loss_arr)
        return float(backend.to_numpy(loss_arr))

    def _gw_gradient(
        self,
        constC: "Array",
        hC1: "Array",
        hC2: "Array",
        T: "Array",
    ) -> "Array":
        """
        Compute GW gradient.

        grad = 2 * tensor_product(T)

        This is the gradient of the GW objective with respect to T.
        """
        return 2.0 * self._tensor_product(constC, hC1, hC2, T)

    def _solve_linear_ot(
        self,
        cost: "Array",
        p: "Array",
        q: "Array",
        epsilon: float,
        max_iterations: int,
        threshold: float,
    ) -> "Array":
        """
        Solve linear optimal transport problem using Sinkhorn.

        Finds: argmin_G <cost, G> + ε H(G)
        subject to: G @ 1 = p, Gᵀ @ 1 = q

        This is the key step in Frank-Wolfe: we solve a LINEAR OT problem
        (not the full non-convex GW) to get the descent direction.
        """
        backend = self._backend
        n = cost.shape[0]
        m = cost.shape[1]

        if n == 0 or m == 0:
            return backend.zeros((n, m))

        # Stabilized Sinkhorn with log-domain computation
        # K = exp(-cost / epsilon)

        # Row-wise stabilization
        cost_min = backend.min(cost, axis=1, keepdims=True)
        cost_centered = cost - cost_min
        log_K = -cost_centered / max(epsilon, 1e-10)

        # Clamp to avoid underflow
        log_K = backend.maximum(log_K, backend.full(log_K.shape, -80.0))
        K = backend.exp(log_K)
        K = backend.maximum(K, backend.full(K.shape, 1e-30))

        # Initialize scaling vectors
        u = backend.ones((n,))
        v = backend.ones((m,))

        for _ in range(max_iterations):
            # Row scaling: u = p / (K @ v)
            Kv = backend.matmul(K, v)
            Kv = backend.maximum(Kv, backend.full(Kv.shape, 1e-30))
            u_new = p / Kv

            # Column scaling: v = q / (Kᵀ @ u)
            Ktu = backend.matmul(backend.transpose(K), u_new)
            Ktu = backend.maximum(Ktu, backend.full(Ktu.shape, 1e-30))
            v_new = q / Ktu

            # Check convergence
            if threshold > 0:
                u_diff = backend.max(backend.abs(u_new - u))
                v_diff = backend.max(backend.abs(v_new - v))
                backend.eval(u_diff, v_diff)
                if max(float(backend.to_numpy(u_diff)), float(backend.to_numpy(v_diff))) < threshold:
                    u = u_new
                    v = v_new
                    break

            u = u_new
            v = v_new

        # Recover transport plan: G = diag(u) @ K @ diag(v)
        G = K * backend.reshape(u, (n, 1)) * backend.reshape(v, (1, m))

        return G

    def _compute_step_size(
        self,
        constC: "Array",
        hC1: "Array",
        hC2: "Array",
        T: "Array",
        G: "Array",
    ) -> float:
        """
        Compute optimal step size for Frank-Wolfe using line search.

        For the GW objective f(T), we want to minimize:
            φ(α) = f((1-α)T + αG) for α ∈ [0, 1]

        For squared loss, this is a quadratic in α with analytic minimum.

        The formula from Peyré et al.:
            α* = -(a - b) / (2c) clamped to [0, 1]
        where the coefficients come from the quadratic expansion.
        """
        backend = self._backend

        # Delta = G - T (descent direction)
        deltaT = G - T

        # Compute coefficients of the quadratic φ(α) = a + b*α + c*α²
        # a = f(T) = <T, constC - hC1 @ T @ hC2.T>
        # For the line search, we need:
        #   b = ∂φ/∂α at α=0 = <grad_f(T), deltaT>
        #   c = coefficient of α² from the quadratic term

        # Gradient at T
        grad_T = self._gw_gradient(constC, hC1, hC2, T)

        # b = <grad, deltaT>
        b_arr = backend.sum(grad_T * deltaT)

        # For GW with squared loss, the quadratic coefficient c comes from:
        # c = 2 * <hC1 @ deltaT @ hC2.T, deltaT>
        # (This is the Hessian-vector product term)
        hessian_term = backend.matmul(backend.matmul(hC1, deltaT), backend.transpose(hC2))
        c_arr = 2.0 * backend.sum(hessian_term * deltaT)

        backend.eval(b_arr, c_arr)
        b = float(backend.to_numpy(b_arr))
        c = float(backend.to_numpy(c_arr))

        # Optimal step: minimize b*α + c*α² subject to α ∈ [0, 1]
        # If c > 0 (convex), minimum at α = -b / (2c)
        # If c ≤ 0 or very small, take full step α = 1 if b < 0, else α = 0

        if abs(c) < 1e-15:
            # Linear or nearly linear: take full step in descent direction
            alpha = 1.0 if b < 0 else 0.0
        elif c > 0:
            # Quadratic with positive curvature: analytic minimum
            alpha = -b / (2.0 * c)
            alpha = max(0.0, min(1.0, alpha))  # Clamp to [0, 1]
        else:
            # Negative curvature: check endpoints
            # f(0) = 0, f(1) = b + c
            alpha = 1.0 if (b + c) < 0 else 0.0

        return alpha

    def _frank_wolfe(
        self,
        C1: "Array",
        C2: "Array",
        p: "Array",
        q: "Array",
        T0: "Array",
        config: Config,
    ) -> Result:
        """
        Conditional Gradient (Frank-Wolfe) algorithm for GW.

        At each iteration:
        1. Compute gradient of GW objective
        2. Solve linear OT to get descent direction
        3. Line search for step size
        4. Update coupling
        """
        backend = self._backend

        # Initialize loss decomposition matrices
        constC, hC1, hC2 = self._init_loss_matrices(C1, C2, p, q)

        T = T0
        prev_loss = float("inf")
        converged = False
        iterations = 0

        for outer in range(config.max_outer_iterations):
            iterations = outer + 1

            # Current loss
            loss = self._gw_loss(constC, hC1, hC2, T)

            # Check convergence
            if iterations >= config.min_outer_iterations:
                abs_change = abs(loss - prev_loss)
                rel_change = abs_change / max(abs(prev_loss), 1e-10) if math.isfinite(prev_loss) else float("inf")

                if abs_change < config.convergence_threshold or rel_change < config.relative_objective_threshold:
                    converged = True
                    break

            prev_loss = loss

            # Step 1: Compute gradient
            grad = self._gw_gradient(constC, hC1, hC2, T)

            # Step 2: Solve linear OT to get descent direction
            # G = argmin_G <grad, G> subject to marginal constraints
            G = self._solve_linear_ot(
                grad,
                p,
                q,
                epsilon=config.sinkhorn_epsilon,
                max_iterations=config.sinkhorn_iterations,
                threshold=config.sinkhorn_threshold,
            )

            # Step 3: Line search for optimal step size
            alpha = self._compute_step_size(constC, hC1, hC2, T, G)

            # Step 4: Update coupling
            if alpha > 1e-10:  # Only update if step is meaningful
                T = (1.0 - alpha) * T + alpha * G

        # Final loss
        final_loss = self._gw_loss(constC, hC1, hC2, T)

        return Result(
            distance=final_loss,
            coupling=T,
            converged=converged,
            iterations=iterations,
        )

    def compute_pairwise_distances(
        self,
        points: "Array",
        k_neighbors: int | None = None,
    ) -> "Array":
        """
        Compute pairwise geodesic distances using GPU-accelerated operations.

        In high-dimensional spaces, curvature is inherent. Geodesic distance
        follows the manifold surface - this is the correct metric.

        Args:
            points: Point matrix [n, d]
            k_neighbors: Number of neighbors for geodesic graph (None = auto).

        Returns:
            Distance matrix [n, n]
        """
        backend = self._backend
        # Convert to backend array if needed (e.g., from Python list)
        points = backend.array(points)
        backend.eval(points)
        n = int(points.shape[0])

        if n == 0:
            return backend.zeros((0, 0))

        # Geodesic distances account for manifold curvature.
        # geodesic_distances handles all cases including n <= 2
        # (where k-NN graph has a single edge, making geodesic = Euclidean).
        from .riemannian_utils import RiemannianGeometry

        rg = RiemannianGeometry(backend)
        result = rg.geodesic_distances(points, k_neighbors=k_neighbors)
        return result.distances


# Convenience function for backward compatibility
def compute_gromov_wasserstein(
    source_points: "Array",
    target_points: "Array",
    config: Config = Config(),
    backend: "Backend | None" = None,
) -> Result:
    """
    Compute Gromov-Wasserstein distance between point sets.

    Convenience function that computes pairwise geodesic distances and GW distance.
    Geodesic distances are used because curvature is inherent in high-dimensional
    representation spaces.

    Args:
        source_points: Source point matrix [n, d]
        target_points: Target point matrix [m, d]
        config: Algorithm configuration
        backend: Backend protocol implementation. If None, uses default.

    Returns:
        Result with distance, coupling, and convergence info
    """
    if backend is None:
        backend = get_default_backend()

    gw = GromovWassersteinDistance(backend)
    source_dist = gw.compute_pairwise_distances(
        source_points,
        k_neighbors=config.geodesic_k_neighbors,
    )
    target_dist = gw.compute_pairwise_distances(
        target_points,
        k_neighbors=config.geodesic_k_neighbors,
    )

    return gw.compute(source_dist, target_dist, config)
