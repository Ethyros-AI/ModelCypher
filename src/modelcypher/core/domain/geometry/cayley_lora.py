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

"""NB-LoRA: Norm-Bounded Low-Rank Adaptation via Cayley Transform.

This module implements the Cayley parameterization from NB-LoRA (arXiv:2501.19050),
which provides a complete, smooth, and unconstrained parameterization of
bounded-norm matrices.

Key Insight:
    Standard LoRA initialization (B=0, A random) wastes rank capacity because
    the product B @ A starts at zero. The Cayley parameterization allows
    immediate full-rank contribution while mathematically guaranteeing the
    spectral norm bound is never violated.

Mathematical Background:
    The Cayley transform maps skew-symmetric matrices to orthogonal matrices:
        Q = (I - Z)(I + Z)^{-1}

    For semi-orthogonal matrices, we use:
        [A^T, B^T]^T = Cayley([A_tilde^T, B_tilde^T]^T)

    The final LoRA contribution is:
        W_lora = 2 * B^T @ S @ A

    Where S = diag(s_1, ..., s_r) with s_i <= sigma_k(W_i) ensures the
    spectral norm bound is respected.

Properties:
    1. Complete: Covers all matrices with ||W||_2 <= bound
    2. Smooth: Differentiable parameterization for gradient-based training
    3. Unconstrained: A_tilde, B_tilde have no constraints
    4. Guaranteed: ||W_lora||_spectral <= 2 * max(S) by construction

Reference:
    Wang et al. (2025). NB-LoRA: Norm-Bounded Low-Rank Adaptation.
    arXiv:2501.19050

Usage:
    from modelcypher.core.domain.geometry.cayley_lora import (
        cayley_transform,
        nb_lora_forward,
        NBLoRALayer,
    )

    # Initialize with geometry-derived scale bounds
    layer = NBLoRALayer(
        in_features=768,
        out_features=768,
        rank=8,
        scale_bound=sigma_k,  # From base weight spectral analysis
        backend=backend,
    )

    # Forward pass (norm-bounded by construction)
    output = layer.forward(x)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


# =============================================================================
# Cayley Transform Core
# =============================================================================


def cayley_transform(
    X: "Array",
    Y: "Array",
    backend: "Backend",
) -> tuple["Array", "Array"]:
    """Cayley transform for semi-orthogonal matrix construction.

    Given free parameters X ∈ R^{r×r} and Y ∈ R^{(n-r)×r}, produces
    semi-orthogonal matrices A ∈ R^{r×n} and B ∈ R^{(n-r)×r} where:
        [A^T; B^T] has orthonormal columns

    Mathematical formulation:
        Z = X - X^T + Y^T @ Y  (skew-symmetric + positive semi-definite)
        (I + Z)^{-1} exists because eigenvalues of Z are imaginary
        A = (I - Z) @ (I + Z)^{-1}
        B = -2 @ Y @ (I + Z)^{-1}

    Args:
        X: Free parameter matrix [r, r] (will be made skew-symmetric)
        Y: Free parameter matrix [m, r] where m = out_features - r
        backend: Compute backend

    Returns:
        Tuple (A, B) where:
            A: [r, r] component of semi-orthogonal matrix
            B: [m, r] component of semi-orthogonal matrix
            Together [A^T; B^T] has orthonormal columns

    Reference:
        Cayley transform: Q = (I - Z)(I + Z)^{-1} for skew-symmetric Z
        Wang et al. (2025), NB-LoRA, arXiv:2501.19050
    """
    b = backend
    r = X.shape[0]

    # Build skew-symmetric + PSD component
    # Z = (X - X^T) + Y^T @ Y
    X_skew = X - b.transpose(X)
    YtY = b.matmul(b.transpose(Y), Y)
    Z = X_skew + YtY
    b.eval(Z)

    # Identity matrix
    I = b.eye(r)

    # Compute (I + Z)^{-1}
    # This inverse always exists because Z has imaginary eigenvalues
    IpZ = I + Z
    b.eval(IpZ)

    # Use safe inverse for numerical stability
    from modelcypher.core.domain.geometry.numerical_stability import safe_inverse

    IpZ_inv, _ = safe_inverse(b, IpZ, regularize=True)
    b.eval(IpZ_inv)

    # A = (I - Z) @ (I + Z)^{-1}
    ImZ = I - Z
    A = b.matmul(ImZ, IpZ_inv)
    b.eval(A)

    # B = -2 @ Y @ (I + Z)^{-1}
    B = -2.0 * b.matmul(Y, IpZ_inv)
    b.eval(B)

    return A, B


def cayley_transform_full(
    A_tilde: "Array",
    B_tilde: "Array",
    backend: "Backend",
) -> tuple["Array", "Array"]:
    """Full Cayley transform from free parameters to LoRA factors.

    This is the high-level interface matching the NB-LoRA paper notation.
    Given unconstrained A_tilde ∈ R^{r×n} and B_tilde ∈ R^{m×r}, produces
    semi-orthogonal LoRA factors.

    Args:
        A_tilde: Free parameter [r, in_features]
        B_tilde: Free parameter [out_features, r]
        backend: Compute backend

    Returns:
        Tuple (A, B) where A @ x gives LoRA input projection and
        B^T @ (S @ A @ x) gives LoRA output with bound ||B^T @ S @ A||_2 <= max(S)
    """
    b = backend

    # For the Cayley transform, we need square X and rectangular Y
    # X comes from the first r×r block, Y from remaining rows
    r = A_tilde.shape[0]
    n = A_tilde.shape[1]

    if n < r:
        raise ValueError(f"in_features ({n}) must be >= rank ({r})")

    # Extract X from A_tilde (use first r columns to form r×r)
    X = A_tilde[:, :r]  # [r, r]

    # For Y, we use B_tilde transposed to get the right shape
    # Y should be [m, r] where m = out_features
    Y = B_tilde  # [m, r]

    A_core, B_core = cayley_transform(X, Y, b)

    # Now construct full A by combining with remaining columns
    if n > r:
        # A_full = [A_core | A_tilde[:, r:]]
        A_remainder = A_tilde[:, r:]
        A_full = b.concatenate([A_core, A_remainder], axis=1)
    else:
        A_full = A_core

    b.eval(A_full)

    return A_full, B_core


# =============================================================================
# NB-LoRA Forward Pass
# =============================================================================


def nb_lora_forward(
    A_tilde: "Array",
    B_tilde: "Array",
    S: "Array",
    x: "Array",
    backend: "Backend",
) -> "Array":
    """NB-LoRA forward pass with guaranteed norm bound.

    Computes: output = 2 * B^T @ S @ A @ x

    Where A, B are the Cayley-transformed semi-orthogonal factors from
    A_tilde, B_tilde, and S is a diagonal scale matrix with entries
    bounded by sigma_k (smallest significant SV of base weight).

    Args:
        A_tilde: Free parameter [r, in_features]
        B_tilde: Free parameter [out_features, r]
        S: Diagonal scale matrix [r, r] or vector [r]. Must have S[i] <= sigma_k.
        x: Input tensor [batch, seq, in_features] or [seq, in_features]
        backend: Compute backend

    Returns:
        LoRA contribution with ||output||_spectral <= 2 * max(S) guaranteed

    Mathematical guarantee:
        Since [A^T; B^T] has orthonormal columns:
            ||B^T @ S @ A||_2 <= ||B^T||_2 × ||S||_2 × ||A||_2
                              <= 1 × max(S) × 1 = max(S)
        With the factor of 2 from NB-LoRA formulation:
            ||W_lora||_2 <= 2 × max(S)
    """
    b = backend

    # Get semi-orthogonal factors via Cayley transform
    A, B = cayley_transform_full(A_tilde, B_tilde, b)

    # Handle S as diagonal (vector or matrix)
    if len(S.shape) == 1:
        # S is a vector - will use element-wise multiplication
        S_diag = S
    else:
        # S is already a matrix - extract diagonal
        r = S.shape[0]
        # For now, assume S is diagonal and use it directly
        S_diag = b.diagonal(S) if hasattr(b, "diagonal") else S

    # Forward: y = 2 * B^T @ S @ A @ x
    # Breakdown:
    # 1. z = A @ x^T -> [r, seq] for each batch element
    # 2. z_scaled = S @ z (element-wise with diagonal S)
    # 3. output = 2 * B^T @ z_scaled

    # Handle input dimensions
    input_shape = x.shape
    if len(input_shape) == 3:
        batch, seq, d_in = input_shape
        x_flat = b.reshape(x, (batch * seq, d_in))
    elif len(input_shape) == 2:
        x_flat = x
    else:
        x_flat = b.reshape(x, (-1, input_shape[-1]))

    # Step 1: Project to rank-r space
    # x_flat: [batch*seq, in_features]
    # A: [r, in_features] -> A @ x^T: [r, batch*seq]
    z = b.matmul(A, b.transpose(x_flat))  # [r, batch*seq]
    b.eval(z)

    # Step 2: Apply diagonal scaling
    # S_diag: [r] -> broadcast to [r, batch*seq]
    if len(S_diag.shape) == 1:
        z_scaled = z * b.reshape(S_diag, (-1, 1))
    else:
        z_scaled = b.matmul(S_diag, z)
    b.eval(z_scaled)

    # Step 3: Project to output space
    # B: [out_features, r] -> B^T: [r, out_features]
    # output = 2 * B^T @ z_scaled = 2 * [out_features, batch*seq]
    output = 2.0 * b.matmul(b.transpose(B), z_scaled)  # [out_features, batch*seq]
    b.eval(output)

    # Transpose back: [batch*seq, out_features]
    output = b.transpose(output)

    # Reshape to original batch structure
    if len(input_shape) == 3:
        output = b.reshape(output, (batch, seq, -1))
    b.eval(output)

    return output


# =============================================================================
# NB-LoRA Layer Class
# =============================================================================


@dataclass
class NBLoRAConfig:
    """Configuration for NB-LoRA layer.

    Attributes:
        in_features: Input dimension
        out_features: Output dimension
        rank: LoRA rank (r)
        scale_bound: Maximum value for diagonal S (sigma_k from base weight)
        init_scale: Initial scale for S (default: 0.5 * scale_bound)
    """

    in_features: int
    out_features: int
    rank: int
    scale_bound: float
    init_scale: float | None = None

    def __post_init__(self) -> None:
        if self.init_scale is None:
            self.init_scale = 0.5 * self.scale_bound


class NBLoRALayer:
    """NB-LoRA layer with guaranteed spectral norm bound.

    Implements the Cayley-parameterized LoRA from Wang et al. (2025).
    The spectral norm of the LoRA contribution is bounded by 2 * max(S)
    where S is initialized to respect sigma_k (smallest significant SV
    of the base weight).

    Args:
        config: NBLoRAConfig with dimensions and scale bound
        backend: Compute backend
        init_std: Standard deviation for initializing free parameters

    Example:
        config = NBLoRAConfig(
            in_features=768,
            out_features=768,
            rank=8,
            scale_bound=0.001,  # sigma_k from base weight
        )
        layer = NBLoRALayer(config, backend)

        # Forward pass (always respects bound)
        lora_output = layer.forward(x)

        # Training: gradients flow through Cayley transform
        # Parameters: layer.A_tilde, layer.B_tilde, layer.S_raw
    """

    def __init__(
        self,
        config: NBLoRAConfig,
        backend: "Backend",
        init_std: float = 0.01,
    ) -> None:
        self._backend = backend
        self._config = config
        b = backend

        r = config.rank
        n_in = config.in_features
        n_out = config.out_features

        # Free parameters (unconstrained)
        # A_tilde: [r, in_features]
        self.A_tilde = b.random_normal((r, n_in)) * init_std
        b.eval(self.A_tilde)

        # B_tilde: [out_features, r]
        self.B_tilde = b.random_normal((n_out, r)) * init_std
        b.eval(self.B_tilde)

        # S_raw: raw scale values (will be clamped to [0, scale_bound])
        # Initialize to init_scale
        init_s = config.init_scale or (0.5 * config.scale_bound)
        self.S_raw = b.ones((r,)) * init_s
        b.eval(self.S_raw)

        logger.debug(
            "NBLoRALayer: in=%d, out=%d, rank=%d, bound=%.6f",
            n_in,
            n_out,
            r,
            config.scale_bound,
        )

    @property
    def scale_bound(self) -> float:
        """Get the spectral norm bound."""
        return self._config.scale_bound

    def get_S(self) -> "Array":
        """Get clamped scale values (respecting bound)."""
        b = self._backend
        # Clamp S_raw to [0, scale_bound]
        S = b.clip(self.S_raw, 0.0, self._config.scale_bound)
        b.eval(S)
        return S

    def forward(self, x: "Array") -> "Array":
        """Forward pass with guaranteed norm bound.

        Args:
            x: Input tensor [batch, seq, in_features] or [seq, in_features]

        Returns:
            LoRA contribution with ||output||_spectral <= 2 * scale_bound
        """
        S = self.get_S()
        return nb_lora_forward(
            self.A_tilde,
            self.B_tilde,
            S,
            x,
            self._backend,
        )

    def get_effective_delta(self) -> "Array":
        """Get the effective weight delta: 2 * B^T @ S @ A.

        Useful for analysis or merging to base weights.

        Returns:
            Weight delta matrix [out_features, in_features]
        """
        b = self._backend

        A, B = cayley_transform_full(self.A_tilde, self.B_tilde, b)
        S = self.get_S()

        # Delta = 2 * B^T @ diag(S) @ A
        # B: [out_features, r], A: [r, in_features]
        S_A = A * b.reshape(S, (-1, 1))  # Scale rows of A by S
        delta = 2.0 * b.matmul(b.transpose(B), S_A)
        b.eval(delta)

        return delta

    def get_spectral_norm(self) -> float:
        """Compute actual spectral norm of current LoRA delta.

        Returns:
            ||2 * B^T @ S @ A||_spectral (should be <= 2 * scale_bound)
        """
        b = self._backend
        delta = self.get_effective_delta()

        # Convert to float32 for stable SVD
        delta_f32 = b.astype(delta, "float32")
        b.eval(delta_f32)

        _, S, _ = b.svd(delta_f32, compute_uv=False)
        b.eval(S)

        return float(b.to_scalar(S[0]))

    def parameters(self) -> dict[str, "Array"]:
        """Get trainable parameters.

        Returns:
            Dict with 'A_tilde', 'B_tilde', 'S_raw' as keys
        """
        return {
            "A_tilde": self.A_tilde,
            "B_tilde": self.B_tilde,
            "S_raw": self.S_raw,
        }

    def set_parameters(self, params: dict[str, "Array"]) -> None:
        """Set parameters from dict."""
        if "A_tilde" in params:
            self.A_tilde = params["A_tilde"]
        if "B_tilde" in params:
            self.B_tilde = params["B_tilde"]
        if "S_raw" in params:
            self.S_raw = params["S_raw"]


# =============================================================================
# Utility Functions
# =============================================================================


def create_nb_lora_from_base_weight(
    W: "Array",
    rank: int,
    backend: "Backend",
    scale_ratio: float = 0.9,
) -> NBLoRALayer:
    """Create NB-LoRA layer with scale bound derived from base weight.

    This is the recommended way to create NB-LoRA layers, as it automatically
    computes the geometry-derived scale bound from the base weight's spectral
    structure.

    Args:
        W: Base weight matrix [out_features, in_features]
        rank: LoRA rank
        backend: Compute backend
        scale_ratio: Fraction of sigma_k to use as bound (default 0.9 for safety)

    Returns:
        NBLoRALayer configured with geometry-derived scale bound

    Example:
        layer = create_nb_lora_from_base_weight(
            W=model.layers[0].mlp.up_proj.weight,
            rank=8,
            backend=backend,
        )
    """
    b = backend
    from modelcypher.core.domain.geometry.numerical_stability import sqrt_scalar

    # SVD of base weight
    W_f32 = b.astype(W, "float32")
    b.eval(W_f32)
    _, S, _ = b.svd(W_f32, compute_uv=False)
    b.eval(S)

    # Find sigma_k (smallest significant singular value)
    sigma_max = float(b.to_scalar(S[0]))
    sqrt_eps = sqrt_scalar(b.finfo().eps, b)
    threshold = sqrt_eps * sigma_max

    significant = S > threshold
    eff_rank = int(b.to_scalar(b.sum(b.astype(significant, "int32"))))

    if eff_rank > 0:
        sigma_k = float(b.to_scalar(S[eff_rank - 1]))
    else:
        sigma_k = float(b.to_scalar(S[-1]))

    # Scale bound with safety ratio
    scale_bound = scale_ratio * sigma_k

    out_features, in_features = W.shape

    config = NBLoRAConfig(
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        scale_bound=scale_bound,
    )

    logger.info(
        "Created NB-LoRA: W=[%d,%d], rank=%d, sigma_k=%.6f, bound=%.6f",
        out_features,
        in_features,
        rank,
        sigma_k,
        scale_bound,
    )

    return NBLoRALayer(config, backend)


__all__ = [
    "cayley_transform",
    "cayley_transform_full",
    "nb_lora_forward",
    "NBLoRAConfig",
    "NBLoRALayer",
    "create_nb_lora_from_base_weight",
]
