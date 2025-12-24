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


import mlx.core as mx
import mlx.nn as nn


class GromovWassersteinSolver:
    """
    MLX implementation of Entropic Gromov-Wasserstein solver.
    Solves for the optimal transport plan T that minimizes:
    sum_{i,j,k,l} |C1[i,k] - C2[j,l]|^2 * T[i,j] * T[k,l]
    Subject to marginal constraints.
    """
    
    @staticmethod
    def solve(
        C1: mx.array, 
        C2: mx.array, 
        p: mx.array, 
        q: mx.array, 
        epsilon: float = 0.01, 
        max_iter: int = 50,
        threshold: float = 1e-4
    ) -> tuple[mx.array, float, int]:
        """
        C1: source distance matrix [n, n]
        C2: target distance matrix [m, m]
        p: source distribution [n]
        q: target distribution [m]
        """
        # Outer loop (GW) - projected gradient descent / block coordinate descent
        # Entropic GW can be solved by iterative Sinkhorn on a cost matrix derived from current T
        
        n = C1.shape[0]
        m = C2.shape[0]
        
        # Initialize T (coupling) as outer product of marginals
        T = mx.outer(p, q)
        
        # Constants
        # Constants
        C1_sq = C1 ** 2
        C2_sq = C2 ** 2
        
        # constC1[i] = sum_k C1[i,k]^2 * p[k]
        constC1 = (C1_sq @ p).reshape(-1, 1) # [n, 1]
        
        # constC2[j] = sum_l C2[j,l]^2 * q[l]
        constC2 = (C2_sq @ q).reshape(1, -1) # [1, m]
        
        const_term = constC1 + constC2 # [n, m]
        
        prev_T = T
        
        for i in range(max_iter):
            # Compute gradient/cost for local KL problem
            # L(a, b, T) = const - 2 * C1 @ T @ C2.T
            # For entropic GW, the cost matrix for Sinkhorn is:
            # M = const_term - 2 * C1 @ T @ C2.T
            
            tens = -2 * (C1 @ T @ C2.T)
            M = const_term + tens
            
            # Solve Entropic OT (Sinkhorn) for cost M
            # K = exp(-M / epsilon)
            # Find u, v such that diag(u) K diag(v) has margins p, q
            
            # Simple Sinkhorn
            K = mx.exp(-M / epsilon)
            u = mx.ones((n,))
            
            # Inner Sinkhorn iterations
            for _ in range(10): # Fixed small number of inner iters usually sufficient for GW gradient
                v = q / (K.T @ u)
                u = p / (K @ v)
                
            T = u.reshape(-1, 1) * K * v.reshape(1, -1)
            
            # Check convergence
            diff = mx.linalg.norm(T - prev_T).item()
            if diff < threshold:
                return T, diff, i
            prev_T = T
            
        return T, 0.0, max_iter

    @staticmethod
    def compute_pairwise_distances(X: mx.array) -> mx.array:
        """
        Compute squared Euclidean distance matrix.
        """
        # (x-y)^2 = x^2 + y^2 - 2xy
        norm_sq = mx.sum(X**2, axis=1, keepdims=True) # [N, 1]
        dist = norm_sq + norm_sq.T - 2 * (X @ X.T)
        return mx.maximum(dist, 0.0) # Clip negs
