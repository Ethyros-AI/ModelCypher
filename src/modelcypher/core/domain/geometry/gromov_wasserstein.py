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

"""Gromov-Wasserstein distance computation for representation space comparison.

GPU-accelerated implementation using the Backend protocol,
following a Frank-Wolfe optimization approach.

Includes OGW (Orthogonal GW) lower bound for O(n³) fast-reject screening:
when the spectral lower bound exceeds a threshold, we know the true GW
distance is at least that large without running expensive iteration.

References:
    - Peyré, Cuturi, Solomon (2016) "GW Averaging" ICML
    - Peyré & Cuturi (2019) "Computational Optimal Transport"
    - Jin et al. (2022) "Orthogonal Gromov-Wasserstein Discrepancy" NeurIPS
    - POT library: https://pythonot.github.io/

See also: docs/geometry/gromov_wasserstein.md
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    compute_median,
    division_epsilon,
    exp_scalar,
    is_finite,
    machine_epsilon,
)
from modelcypher.core.domain.geometry.optimal_transport import SinkhornSolver

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

_PERM_CACHE: dict[tuple[int, int], tuple[list[tuple[int, ...]], "Array", "Array"]] = {}


@dataclass(frozen=True)
class GromovWassersteinResult:
    distance: float
    coupling: "Array | None"
    converged: bool
    iterations: int
    is_lower_bound: bool = False

    def __post_init__(self) -> None:
        # Enforce the metric invariant: GW distance is non-negative.
        backend = get_default_backend()
        if is_finite(self.distance, backend) and self.distance < 0:
            object.__setattr__(self, "distance", 0.0)

    @property
    def normalized_distance(self) -> float:
        backend = get_default_backend()
        return 1.0 - exp_scalar(-self.distance, backend) if is_finite(self.distance, backend) else 1.0

    @property
    def aligned(self) -> bool:
        backend = get_default_backend()
        if not is_finite(self.distance, backend):
            return False
        eps = float(machine_epsilon(backend, backend.array([self.distance])))
        return abs(self.distance) <= eps



class GromovWassersteinDistance:
    """GPU-accelerated Gromov-Wasserstein distance using Frank-Wolfe algorithm."""

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()
        self._sinkhorn_solver = SinkhornSolver(self._backend)

    def compute(
        self,
        source_distances: "Array",
        target_distances: "Array",
        fast_reject_threshold: float | None = None,
    ) -> GromovWassersteinResult:
        """
        Compute Gromov-Wasserstein distance between two metric spaces.

        Uses Conditional Gradient (Frank-Wolfe) algorithm with deterministic
        initialization. All parameters are derived from numerical precision
        of the input dtype - no configuration needed.

        Args:
            source_distances: Pairwise distance matrix for source [n, n]
            target_distances: Pairwise distance matrix for target [m, m]
            fast_reject_threshold: If provided, compute OGW lower bound first.
                If the lower bound exceeds this threshold, return early without
                computing the full iterative solution. Useful for screening
                when you only need to know "is this misaligned?".

        Returns:
            Result with distance, coupling matrix, and convergence info.
            If fast_reject_threshold triggered, is_lower_bound=True and coupling=None.
        """
        backend = self._backend

        # Convert inputs to backend arrays if needed
        C1 = backend.array(source_distances)
        C2 = backend.array(target_distances)
        backend.eval(C1, C2)

        n = int(C1.shape[0])
        m = int(C2.shape[0])

        if n == 0 or m == 0:
            return GromovWassersteinResult(
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
            # Use precision-aware threshold for identity check
            eps = division_epsilon(backend, C1)
            if float(backend.to_scalar(max_diff)) < eps:
                # Identical - return identity coupling
                coupling = backend.eye(n) / n
                return GromovWassersteinResult(distance=0.0, coupling=coupling, converged=True, iterations=0)

        # Fast-reject path: compute OGW lower bound in O(n³) via SVD
        # If the lower bound exceeds threshold, we know the true GW distance
        # is at least that large, so we can skip expensive iteration.
        if fast_reject_threshold is not None and n == m:
            lower_bound = self._ogw_lower_bound(C1, C2)
            if is_finite(lower_bound, backend) and lower_bound > fast_reject_threshold:
                return GromovWassersteinResult(
                    distance=lower_bound,
                    coupling=None,
                    converged=True,
                    iterations=0,
                    is_lower_bound=True,
                )

        # For square matrices, exhaustively search permutations when resolution allows it.
        # We bound n! by the number of distinguishable states at machine precision.
        if n == m:
            perm_budget = max(1, int(1.0 / machine_epsilon(backend, C1)))
            if math.factorial(n) <= perm_budget:
                return self._solve_by_permutation_search(C1, C2, n, backend)

        # Uniform marginals
        p = backend.ones((n,)) / n
        q = backend.ones((m,)) / m

        # Precompute loss decomposition matrices once (reuse across restarts)
        constC, hC1, hC2 = self._init_loss_matrices(C1, C2, p, q)

        # Deterministic initialization: uniform coupling (outer product of marginals)
        T0 = self._uniform_coupling(p, q)
        result = self._frank_wolfe(C1, C2, p, q, T0, constC=constC, hC1=hC1, hC2=hC2)

        return GromovWassersteinResult(
            distance=result.distance,
            coupling=result.coupling,
            converged=result.converged,
            iterations=result.iterations,
        )

    def _uniform_coupling(self, p: "Array", q: "Array") -> "Array":
        """Deterministic coupling with uniform marginals."""
        backend = self._backend
        n = int(p.shape[0])
        m = int(q.shape[0])
        return backend.matmul(backend.reshape(p, (n, 1)), backend.reshape(q, (1, m)))

    def _ogw_lower_bound(self, C1: "Array", C2: "Array") -> float:
        """
        Compute Orthogonal Gromov-Wasserstein lower bound in O(n³).

        OGW restricts the coupling to orthogonal matrices, yielding a closed-form
        solution via singular value alignment (Jin et al., 2022):

            OGW(C1, C2) = ||C1||²_F + ||C2||²_F - 2 × Σᵢ σᵢ(C1) × σᵢ(C2)

        Since OGW optimizes over a subset of couplings (orthogonal ones),
        it provides a lower bound on the true GW distance.

        For symmetric distance matrices, singular values equal absolute eigenvalues,
        but we use SVD for generality and numerical stability.

        References:
            Jin et al. (2022) "Orthogonal Gromov-Wasserstein Discrepancy"

        Args:
            C1: Source distance matrix [n, n]
            C2: Target distance matrix [n, n] (must be same size)

        Returns:
            Lower bound on GW distance. Always non-negative by construction.
        """
        backend = self._backend
        n = int(C1.shape[0])

        if n == 0:
            return 0.0

        # Compute Frobenius norms squared
        fro_C1_sq = backend.sum(C1 * C1)
        fro_C2_sq = backend.sum(C2 * C2)

        # Compute singular values via SVD
        # For symmetric matrices, σᵢ = |λᵢ|, but SVD handles general case
        # SVD returns singular values in descending order by convention
        _, s1, _ = backend.svd(C1)
        _, s2, _ = backend.svd(C2)

        # Ensure descending order for optimal alignment
        # sort() returns ascending, so we negate, sort, negate back
        s1_sorted = -backend.sort(-s1)
        s2_sorted = -backend.sort(-s2)
        backend.eval(s1_sorted, s2_sorted)

        # Compute spectral alignment: Σᵢ σᵢ(C1) × σᵢ(C2)
        # Both should have length n for n×n matrices
        spectral_product = backend.sum(s1_sorted * s2_sorted)

        # OGW = ||C1||²_F + ||C2||²_F - 2 × Σᵢ σᵢ(C1) × σᵢ(C2)
        ogw_arr = fro_C1_sq + fro_C2_sq - 2.0 * spectral_product
        backend.eval(ogw_arr)
        ogw = float(backend.to_scalar(ogw_arr))

        # Normalize by n² to match the GW loss scaling in _gw_loss
        ogw_normalized = ogw / (n * n) if n > 0 else 0.0

        # Enforce non-negativity (can be slightly negative due to floating point)
        return max(0.0, ogw_normalized)

    def compute_lower_bound(
        self,
        source_distances: "Array",
        target_distances: "Array",
    ) -> float:
        """
        Compute OGW lower bound on GW distance in O(n³) closed-form.

        This is a standalone method for when you only need the lower bound
        and don't want to compute the full GW distance.

        Args:
            source_distances: Pairwise distance matrix for source [n, n]
            target_distances: Pairwise distance matrix for target [n, n]

        Returns:
            Lower bound on GW distance. If this is large, the true GW
            distance is definitely at least this large.
        """
        backend = self._backend
        C1 = backend.array(source_distances)
        C2 = backend.array(target_distances)
        backend.eval(C1, C2)

        n1, n2 = int(C1.shape[0]), int(C2.shape[0])
        if n1 != n2:
            raise ValueError(
                f"OGW lower bound requires square matrices of same size. "
                f"Got {n1}x{n1} and {n2}x{n2}."
            )

        return self._ogw_lower_bound(C1, C2)

    def _solve_by_permutation_search(
        self,
        C1: "Array",
        C2: "Array",
        n: int,
        backend: "Backend",
    ) -> GromovWassersteinResult:
        """
        Solve GW by exhaustive permutation search for small matrices.

        For uniform marginals, GW simplifies to the Quadratic Assignment Problem:
            min_P sum_{i,j,k,l} (C1[i,k] - C2[j,l])^2 * P[i,j] * P[k,l]

        where P is a permutation matrix (scaled by 1/n for proper marginals).

        This is equivalent to: min_σ sum_{i,k} (C1[i,k] - C2[σ(i),σ(k)])^2 / n^2

        GPU-vectorized implementation:
            - Pre-compute all permutation matrices as a batch [n_perms, n, n]
            - Compute C2_permuted = P @ C2 @ P.T for all permutations at once
            - Vectorized Frobenius norm to find minimum

        Complexity: O(n! * n^2) but fully GPU-parallel.
        """
        import itertools

        cache_key = (id(backend), n)
        cached = _PERM_CACHE.get(cache_key)
        if cached is None:
            # Generate all permutations (n! permutations, tractable for n≤8)
            perms = list(itertools.permutations(range(n)))

            # Build all permutation matrices on GPU [n_perms, n, n]
            # P[idx, i, perm[i]] = 1.0
            P_data = []
            for perm in perms:
                P_row = [[0.0] * n for _ in range(n)]
                for i, j in enumerate(perm):
                    P_row[i][j] = 1.0
                P_data.append(P_row)
            P_all = backend.array(P_data)  # [n_perms, n, n]
            P_T = backend.transpose(P_all, axes=(0, 2, 1))  # [n_perms, n, n]
            backend.eval(P_all, P_T)
            _PERM_CACHE[cache_key] = (perms, P_all, P_T)
        else:
            perms, P_all, P_T = cached

        # Compute C2_permuted[idx] = P[idx] @ C2 @ P[idx].T for all perms
        # Step 1: P @ C2 → [n_perms, n, n]
        # Reshape for batched matmul: P_all @ C2
        # Since C2 is [n, n] and P_all is [n_perms, n, n], we need to broadcast
        C2_expanded = backend.reshape(C2, (1, n, n))  # [1, n, n]
        # P @ C2: for each perm, [n, n] @ [n, n] → [n, n]
        PC2 = backend.matmul(P_all, C2_expanded)  # [n_perms, n, n] via broadcast

        # Step 2: (P @ C2) @ P.T → [n_perms, n, n]
        # P.T is transpose of each P, which is [n_perms, n, n] with last two dims swapped
        C2_permuted = backend.matmul(PC2, P_T)  # [n_perms, n, n]

        # Compute Frobenius norm squared: ||C1 - C2_permuted||^2_F for each perm
        # Compute both directions and average for symmetry:
        # loss = 0.5 * (||C1 - P @ C2 @ P.T||² + ||C2 - P.T @ C1 @ P||²)
        # This enforces symmetry in floating point.

        # Forward: ||C1 - P @ C2 @ P.T||² using C2_permuted = P @ C2 @ P.T
        C1_expanded = backend.reshape(C1, (1, n, n))  # [1, n, n]
        diff_forward = C1_expanded - C2_permuted
        losses_forward = backend.sum(diff_forward * diff_forward, axis=(1, 2))

        # Backward: ||C2 - P.T @ C1 @ P||² using C1_permuted = P.T @ C1 @ P
        C1_expanded_for_perm = backend.reshape(C1, (1, n, n))  # [1, n, n]
        PC1 = backend.matmul(P_T, C1_expanded_for_perm)  # P.T @ C1
        C1_permuted = backend.matmul(PC1, P_all)  # (P.T @ C1) @ P
        C2_expanded = backend.reshape(C2, (1, n, n))  # [1, n, n]
        diff_backward = C2_expanded - C1_permuted
        losses_backward = backend.sum(diff_backward * diff_backward, axis=(1, 2))

        # Average for symmetry
        losses = (losses_forward + losses_backward) / (2.0 * n * n)

        # Force evaluation and find minimum
        best_idx_arr = backend.argmin(losses)
        backend.eval(best_idx_arr)
        best_idx = int(backend.to_scalar(best_idx_arr))
        best_loss_arr = backend.take(losses, backend.array([best_idx]), axis=0)
        best_loss_arr = backend.squeeze(best_loss_arr)
        backend.eval(best_loss_arr)
        best_loss = float(backend.to_scalar(best_loss_arr))
        best_perm = perms[best_idx]

        # Build coupling matrix from minimum-cost permutation
        # T[i, perm[i]] = 1/n
        coupling_data = [[0.0] * n for _ in range(n)]
        for i, j in enumerate(best_perm):
            coupling_data[i][j] = 1.0 / n
        coupling = backend.array(coupling_data)

        return GromovWassersteinResult(
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
        return float(backend.to_scalar(loss_arr))

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

    def _derive_sinkhorn_epsilon(self, cost: "Array") -> float:
        """
        Derive Sinkhorn regularization epsilon from cost matrix scale.

        Standard practice in OT: epsilon should be proportional to the scale
        of the cost matrix. Using median(cost) * sqrt(machine_epsilon) provides
        a principled balance between accuracy and numerical stability.

        Returns:
            Regularization parameter derived from cost matrix statistics.
        """
        backend = self._backend
        n = int(cost.shape[0]) * int(cost.shape[1]) if len(cost.shape) >= 2 else int(cost.shape[0])
        if n == 0:
            return float(division_epsilon(backend, cost))

        # Efficient O(n) median via shared utility
        median_val = compute_median(cost, backend)

        # Epsilon = median * sqrt(machine_eps)
        # This scales with cost and provides stable numerical behavior
        eps = float(machine_epsilon(backend, cost))
        epsilon = max(median_val * (eps ** 0.5), eps)
        return epsilon

    def _compute_step_size(
        self,
        constC: "Array",
        hC1: "Array",
        hC2: "Array",
        T: "Array",
        G: "Array",
        grad_T: "Array | None" = None,
        hC2_T: "Array | None" = None,
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

        # Gradient at T (reuse if already computed)
        if grad_T is None:
            grad_T = self._gw_gradient(constC, hC1, hC2, T)
        if hC2_T is None:
            hC2_T = backend.transpose(hC2)

        # b = <grad, deltaT>
        b_arr = backend.sum(grad_T * deltaT)

        # For GW with squared loss, the quadratic coefficient c comes from:
        # c = 2 * <hC1 @ deltaT @ hC2.T, deltaT>
        # (This is the Hessian-vector product term)
        hessian_term = backend.matmul(backend.matmul(hC1, deltaT), hC2_T)
        c_arr = 2.0 * backend.sum(hessian_term * deltaT)

        backend.eval(b_arr, c_arr)
        b = float(backend.to_scalar(b_arr))
        c = float(backend.to_scalar(c_arr))

        # Optimal step: minimize b*α + c*α² subject to α ∈ [0, 1]
        # If c > 0 (convex), minimum at α = -b / (2c)
        # If c ≤ 0 or very small, take full step α = 1 if b < 0, else α = 0

        # Use precision-aware threshold for near-zero detection
        eps = division_epsilon(self._backend, c_arr)
        if abs(c) < eps:
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
        constC: "Array | None" = None,
        hC1: "Array | None" = None,
        hC2: "Array | None" = None,
    ) -> GromovWassersteinResult:
        """
        Conditional Gradient (Frank-Wolfe) algorithm for GW.

        At each iteration:
        1. Compute gradient of GW objective
        2. Solve linear OT to get descent direction
        3. Line search for step size
        4. Update coupling

        All convergence thresholds are derived from the dtype's machine epsilon.
        """
        backend = self._backend

        # Initialize loss decomposition matrices (reuse across restarts when provided)
        if constC is None or hC1 is None or hC2 is None:
            constC, hC1, hC2 = self._init_loss_matrices(C1, C2, p, q)
        hC2_T = backend.transpose(hC2)

        # Derive convergence thresholds from dtype
        # Using sqrt(eps) as standard numerical tolerance for convergence
        dtype_eps = float(machine_epsilon(backend, T0))
        conv_threshold = dtype_eps**0.5
        rel_threshold = dtype_eps**0.5
        sink_threshold = dtype_eps**0.5
        sinkhorn_epsilon = self._derive_sinkhorn_epsilon(constC)

        T = T0
        prev_loss = float("inf")
        converged = False
        iterations = 0

        # Max iterations: O(1/sqrt(eps)) from Frank-Wolfe convergence theory
        # For float32 (eps ~ 1e-7), this gives ~3000 iterations
        # For float64 (eps ~ 1e-16), this gives ~1e8 (effectively unlimited)
        # Cap at 1000: beyond this, FW convergence is < sqrt(eps) per step
        max_iters = min(1000, int(1.0 / (dtype_eps ** 0.5)))

        while iterations < max_iters:
            iterations += 1

            # Current loss
            tens = constC - backend.matmul(backend.matmul(hC1, T), hC2_T)
            loss_arr = backend.sum(tens * T)
            backend.eval(loss_arr)
            loss = float(backend.to_scalar(loss_arr))

            # Check convergence
            if is_finite(prev_loss, backend):
                abs_change = abs(loss - prev_loss)
                # Use precision-aware epsilon for relative change
                eps = division_epsilon(backend, T)
                rel_change = abs_change / max(abs(prev_loss), eps)

                if abs_change <= conv_threshold and rel_change <= rel_threshold:
                    converged = True
                    break

            prev_loss = loss

            # Step 1: Compute gradient
            grad = 2.0 * tens

            # Step 2: Solve linear OT to get descent direction
            # G = argmin_G <grad, G> subject to marginal constraints
            # Max iterations: Sinkhorn converges in O(n log n / ε²) for entropic OT
            # Using 500 as practical limit for inner loop
            G = self._sinkhorn_solver.solve_linear_ot(
                grad,
                p,
                q,
                epsilon=sinkhorn_epsilon,
                max_iterations=500,
                threshold=sink_threshold,
            )

            # Step 3: Line search for optimal step size
            alpha = self._compute_step_size(constC, hC1, hC2, T, G, grad_T=grad, hC2_T=hC2_T)
            if not is_finite(alpha, backend):
                break

            # Step 4: Update coupling via Frank-Wolfe convex combination
            # NOTE: This is NOT arbitrary blending. The step size α is computed
            # analytically by _compute_step_size() via line search on the quadratic
            # GW objective. This is standard convex optimization from Peyré et al.
            deltaT = G - T
            T_candidate = T + alpha * deltaT
            diff_arr = backend.max(backend.abs(T_candidate - T))
            max_abs_T = backend.max(backend.abs(T))
            backend.eval(diff_arr, max_abs_T)
            diff_val = float(backend.to_scalar(diff_arr))
            max_abs_val = float(backend.to_scalar(max_abs_T))
            update_floor = machine_epsilon(backend, T) * max(max_abs_val, 1.0)
            T = T_candidate
            if not is_finite(diff_val, backend):
                break
            if diff_val <= update_floor:
                break

        # Final loss
        final_loss = self._gw_loss(constC, hC1, hC2, T)

        return GromovWassersteinResult(
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
        follows the manifold surface and accounts for curvature.

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
        # (where k-NN graph has a single edge, making geodesic = chord).
        from .riemannian_utils import RiemannianGeometry

        rg = RiemannianGeometry(backend)
        result = rg.geodesic_distances(points, k_neighbors=k_neighbors)
        return result.distances
