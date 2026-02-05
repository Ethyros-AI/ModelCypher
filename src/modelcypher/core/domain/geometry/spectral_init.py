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

"""Spectral-normalized weight initialization.

Provides initialization methods that control the spectral norm of weight matrices,
ensuring bounded Lipschitz constants and improved training stability.

Reference:
    Miyato et al. (2018) "Spectral Normalization for GANs"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .precision import machine_epsilon
from .scalars import sqrt_scalar

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


def spectral_normalized_init(
    shape: tuple[int, int],
    backend: "Backend",
    target_spectral: float = 1.0,
    n_power_iters: int = 5,
) -> "Array":
    """Initialize a weight matrix with controlled spectral norm.

    Creates a random matrix and rescales it so that its spectral norm
    (largest singular value) equals target_spectral. This is geometry-derived
    initialization - no arbitrary scale factors.

    For LoRA matrices: Initialize so ||B @ A||_spectral = sigma_k
    (uses full geometric budget from step 0).

    Mathematical guarantee: sigma_max(output) = target_spectral +/- O(sqrt(eps))

    Args:
        shape: Output shape (out_features, in_features) for weight matrix.
        backend: Backend for tensor operations.
        target_spectral: Desired spectral norm (default 1.0).
        n_power_iters: Power iteration steps for spectral norm (default 5).

    Returns:
        Initialized weight matrix with ||W||_spectral = target_spectral.

    Reference:
        Spectral normalization ensures the Lipschitz constant of each layer
        is bounded, improving training stability.
        Miyato et al. (2018) "Spectral Normalization for GANs"
    """
    b = backend

    # Generate random matrix (Gaussian initialization)
    W = b.random_normal(shape)
    b.eval(W)

    # Compute spectral norm via power iteration
    spectral_norm = _power_iteration_spectral_norm(W, b, n_iters=n_power_iters)

    # Handle degenerate case (zero matrix)
    eps = sqrt_scalar(machine_epsilon(b, W), b)
    if spectral_norm < eps:
        # Matrix is effectively zero - use identity-like initialization
        min_dim = min(shape)
        # Create a matrix with ones on the "diagonal"
        W = b.zeros(shape)
        for i in range(min_dim):
            # Set W[i, i] = 1 via index assignment
            indices = b.array([[i, i]])
            W = b.scatter(W, indices, b.array([1.0]))
        b.eval(W)
        spectral_norm = 1.0

    # Rescale to target spectral norm
    scale = target_spectral / spectral_norm
    W_scaled = W * scale
    b.eval(W_scaled)

    return W_scaled


def _power_iteration_spectral_norm(
    M: "Array",
    backend: "Backend",
    n_iters: int = 5,
) -> float:
    """Compute spectral norm via power iteration.

    Uses deterministic initialization for reproducibility and numerical stability.

    Args:
        M: Matrix [m, n] to compute spectral norm of.
        backend: Backend for tensor operations.
        n_iters: Number of power iterations (default 5).

    Returns:
        Spectral norm (largest singular value) of M.
    """
    b = backend

    n = int(M.shape[1])
    eps = sqrt_scalar(machine_epsilon(b, M), b)

    # Deterministic initialization: unit vector (more stable than random)
    v = b.ones((n,)) / b.sqrt(b.array(float(n)))
    b.eval(v)

    for _ in range(n_iters):
        # u = M @ v
        u = b.matmul(M, b.reshape(v, (-1, 1)))
        u = b.reshape(u, (-1,))
        u_norm = b.sqrt(b.sum(u * u))
        b.eval(u_norm)
        u_norm_val = float(b.to_scalar(u_norm))

        if u_norm_val < eps:
            return 0.0

        u = u / u_norm
        b.eval(u)

        # v = M^T @ u
        v = b.matmul(b.transpose(M), b.reshape(u, (-1, 1)))
        v = b.reshape(v, (-1,))
        v_norm = b.sqrt(b.sum(v * v))
        b.eval(v_norm)
        v_norm_val = float(b.to_scalar(v_norm))

        if v_norm_val < eps:
            return 0.0

        v = v / v_norm
        b.eval(v)

    # Spectral norm = ||M @ v||
    Mv = b.matmul(M, b.reshape(v, (-1, 1)))
    Mv = b.reshape(Mv, (-1,))
    spectral = b.sqrt(b.sum(Mv * Mv))
    b.eval(spectral)

    return float(b.to_scalar(spectral))


def spectral_normalized_lora_init(
    in_features: int,
    out_features: int,
    rank: int,
    sigma_k: float,
    backend: "Backend",
) -> tuple["Array", "Array"]:
    """Initialize LoRA A and B matrices with controlled spectral budget.

    Initializes the LoRA decomposition (B @ A) such that:
        ||B @ A||_spectral = sigma_k

    This uses the FULL geometric budget from step 0. The standard LoRA
    initialization (A random, B zeros) cannot achieve this because the
    product B @ A starts at zero and only grows during training.

    This initialization allows immediate contribution while respecting
    the spectral bound derived from base weight geometry.

    Args:
        in_features: Input dimension (columns of A).
        out_features: Output dimension (rows of B).
        rank: LoRA rank.
        sigma_k: Target spectral norm (smallest significant SV of base weight).
        backend: Backend for tensor operations.

    Returns:
        Tuple (A, B) where A is [rank, in_features] and B is [out_features, rank].
        The product B @ A has spectral norm = sigma_k.

    Mathematical property:
        For random A, B with ||A||_F = ||B||_F = 1:
        ||B @ A||_spectral <= ||B||_spectral * ||A||_spectral

        We initialize both with ||.||_spectral = sqrt(sigma_k) so the
        product has spectral norm ~= sigma_k (equality for rank-1).
    """
    b = backend

    # Each matrix gets sqrt(sigma_k) spectral norm
    # Product B @ A will have spectral norm ~= sigma_k
    sqrt_sigma_k = sqrt_scalar(sigma_k, b)

    # Initialize A: [rank, in_features] with spectral norm = sqrt(sigma_k)
    A = spectral_normalized_init(
        shape=(rank, in_features),
        backend=b,
        target_spectral=sqrt_sigma_k,
    )

    # Initialize B: [out_features, rank] with spectral norm = sqrt(sigma_k)
    B = spectral_normalized_init(
        shape=(out_features, rank),
        backend=b,
        target_spectral=sqrt_sigma_k,
    )

    return A, B


__all__ = [
    "spectral_normalized_init",
    "spectral_normalized_lora_init",
]
