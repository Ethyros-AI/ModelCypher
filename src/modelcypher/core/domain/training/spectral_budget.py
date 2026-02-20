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

"""Spectral budget monitoring for LoRA training.

Tracks ||scale * B @ A||_spectral / sigma_k per layer. When the budget
ratio exceeds a per-layer Weyl-derived crossing threshold, the adapter
risks singular value crossing (catastrophic forgetting).

Per-layer threshold derivation (Weyl 1912):
    Singular value crossing at rank k occurs when ||E||_2 > gap_k / 2,
    where gap_k = σ_{k-1} - σ_k. In terms of the budget ratio r = ||E||_2 / σ_k:
        crossing_ratio = gap_k / (2 × σ_k)
    Training should stop when ANY layer's ratio exceeds its crossing_ratio.

When spectral gaps are unavailable, callers pass a scalar threshold derived
from dtype precision and/or calibration.

References:
    Weyl, H. (1912). Das asymptotische Verteilungsgesetz der Eigenwerte
        linearer partieller Differentialgleichungen.
    Shuttleworth et al. (2025). LoRA perturbations exceeding Weyl gap create
        intruder dimensions causing catastrophic forgetting. arXiv:2410.21228.

The SVD computation uses the Backend protocol. The exhaustion check is
pure Python.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

# sqrt(eps) for float32 forward error accumulation.
# The budget ratio is computed via: spectral_norm(power_iter(A,B)) / sigma_k.
# Power iteration and the ratio computation each introduce O(eps) error;
# accumulated relative error is bounded by O(sqrt(eps)) via standard
# IEEE 754 error propagation (Higham, "Accuracy and Stability of
# Numerical Algorithms", 2002, Ch. 3). This is NOT a numerical rank cutoff
# (see svd_rank_threshold in precision.py for the LAPACK convention:
# max(m,n) * eps * sigma_max).
_SQRT_EPS_F32 = math.sqrt(math.ldexp(1.0, -23))  # ~3.45e-4

# Dtype-derived default threshold: capacity exhausted when ratio exceeds
# 1 - sqrt(eps_f32) (~0.9997). Beyond this, remaining headroom
# (1.0 - ratio) is indistinguishable from accumulated numerical error.
DTYPE_THRESHOLD_F32 = 1.0 - _SQRT_EPS_F32

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


def _spectral_norm_power_iter(
    lora_a: Any,
    lora_b: Any,
    scale: float,
    backend: "Backend",
    n_iters: int = 10,
) -> float:
    """Estimate ||scale * lora_a @ lora_b||_2 via power iteration.

    Uses implicit matrix-vector products through the factors to avoid
    forming the full [in, out] product matrix. Computes the top singular
    value of (lora_a @ lora_b) via alternating left/right multiplications.

    This is numerically stable (no SVD) and avoids MLX sgesvdx_ crashes
    on ill-conditioned matrices.

    Args:
        lora_a: A factor [in, r].
        lora_b: B factor [r, out].
        scale: Scalar multiplier.
        backend: Backend for matmul/norm.
        n_iters: Power iteration steps (10 is sufficient for convergence).

    Returns:
        Estimated spectral norm (float).
    """
    # Start with random vector in output space
    out_dim = lora_b.shape[1] if len(lora_b.shape) > 1 else lora_b.shape[0]
    v = backend.random_normal((out_dim, 1))
    v = backend.astype(v, "float32")
    backend.eval(v)

    lora_a_f32 = backend.astype(lora_a, "float32")
    lora_b_f32 = backend.astype(lora_b, "float32")
    backend.eval(lora_a_f32, lora_b_f32)

    sigma = 0.0
    for _ in range(n_iters):
        # u = (A @ B) @ v  →  A @ (B @ v)
        u = backend.matmul(lora_a_f32, backend.matmul(lora_b_f32, v))
        backend.eval(u)

        u_norm = float(backend.to_scalar(backend.norm(u)))
        if u_norm < 1e-30:
            break
        u = u * (1.0 / u_norm)

        # v = (A @ B)^T @ u  →  B^T @ (A^T @ u)
        v = backend.matmul(
            backend.transpose(lora_b_f32),
            backend.matmul(backend.transpose(lora_a_f32), u),
        )
        backend.eval(v)

        sigma = float(backend.to_scalar(backend.norm(v)))
        if sigma < 1e-30:
            break
        v = v * (1.0 / sigma)
        backend.eval(v)

    return abs(scale) * sigma


def compute_budget_ratios(
    lora_products: list[tuple[float, Any, Any, float]],
    backend: "Backend",
) -> list[float]:
    """Compute spectral budget ratios for a set of LoRA layers.

    Each entry in ``lora_products`` is (scale, lora_a, lora_b, sigma_k) where:
    - scale: LoRA scale factor
    - lora_a: A factor array [in, rank]
    - lora_b: B factor array [rank, out]
    - sigma_k: Spectral bound for this layer

    The effective LoRA product in weight space is ``scale * (lora_a @ lora_b)``.
    The ratio is ``||product||_spectral / sigma_k``.

    Uses power iteration to estimate the spectral norm, avoiding full SVD
    which can crash on ill-conditioned matrices (MLX sgesvdx_ failure).

    Args:
        lora_products: List of (scale, lora_a, lora_b, sigma_k) tuples.
        backend: Backend for computation.

    Returns:
        List of budget ratios (one per valid entry).
    """
    ratios: list[float] = []

    for scale, lora_a, lora_b, sigma_k in lora_products:
        if sigma_k <= 0:
            continue

        try:
            spectral_norm = _spectral_norm_power_iter(
                lora_a, lora_b, scale, backend,
            )
            ratios.append(spectral_norm / sigma_k)
        except Exception:
            continue

    return ratios


def is_budget_exhausted(
    ratios: list[float],
    threshold: float,
    spectral_gaps: list[float] | None = None,
    sigma_ks: list[float] | None = None,
) -> tuple[bool, float]:
    """Check if spectral budget is exhausted.

    Pure Python — no framework dependencies.

    When ``spectral_gaps`` and ``sigma_ks`` are provided, uses per-layer
    Weyl-derived crossing thresholds: crossing_ratio_i = gap_i / (2 * sigma_k_i).
    Budget is exhausted if ANY layer's ratio exceeds its own crossing bound.
    Reports the median ratio for monitoring.

    When ``spectral_gaps``/``sigma_ks`` are None, compares the median ratio
    against the caller-supplied scalar ``threshold``.

    Args:
        ratios: List of per-layer budget ratios from compute_budget_ratios().
        threshold: Scalar threshold used when spectral_gaps/sigma_ks are not
            provided. Should be derived from dtype precision and/or calibration.
        spectral_gaps: Per-layer spectral gaps (σ_{k-1} - σ_k). Same order
            as ratios.
        sigma_ks: Per-layer sigma_k values. Same order as ratios.

    Returns:
        (is_exhausted, median_ratio). Returns (False, 0.0) for empty input.

    References:
        Weyl (1912): perturbation crossing at ||E||_2 > gap_k / 2.
        Shuttleworth et al. (arXiv:2410.21228): empirical confirmation for LoRA.
    """
    if not ratios:
        return False, 0.0

    sorted_ratios = sorted(ratios)
    median_ratio = sorted_ratios[len(sorted_ratios) // 2]

    # Per-layer Weyl-derived thresholds
    if spectral_gaps is not None and sigma_ks is not None:
        any_crossed = False
        for i, ratio in enumerate(ratios):
            if i >= len(spectral_gaps) or i >= len(sigma_ks):
                break
            sk = sigma_ks[i]
            if sk <= 0:
                continue
            crossing_ratio = spectral_gaps[i] / (2.0 * sk)
            if ratio > crossing_ratio:
                any_crossed = True
                break
        return any_crossed, median_ratio

    # Scalar threshold fallback
    return median_ratio > threshold, median_ratio


def compute_initialization_vectors(
    weight: Any,
    structural_rank: int,
    backend: "Backend",
) -> tuple[Any, Any]:
    """Extract k-th left and right singular vectors of a base weight.

    These vectors define the direction most sensitive to perturbation at
    the structural rank boundary. Storing them at initialization allows
    efficient projected residual monitoring during training.

    Args:
        weight: Base weight matrix [out, in].
        structural_rank: 1-indexed Shannon effective rank boundary. The
            returned vectors correspond to index (structural_rank - 1).
        backend: Backend for SVD computation.

    Returns:
        (u_k, v_k) where u_k is [out, 1] and v_k is [in, 1].

    Note:
        Uses ``compute_uv=True`` SVD, which is slower and more crash-prone
        on MLX than ``compute_uv=False``. Call once at initialization only.
    """
    b = backend
    W = b.astype(weight, "float32")
    b.eval(W)

    U, _S, Vt = b.svd(W, compute_uv=True)
    b.eval(U, Vt)

    k = max(0, min(structural_rank - 1, int(U.shape[1]) - 1))

    # u_k: k-th column of U → [out, 1]
    u_k = b.reshape(U[:, k], (-1, 1))
    # v_k: k-th row of Vt transposed → [in, 1]
    v_k = b.reshape(Vt[k, :], (-1, 1))
    b.eval(u_k, v_k)

    return u_k, v_k


def compute_projected_residuals(
    lora_products: list[tuple[float, Any, Any, float]],
    base_u_ks: list[Any],
    base_v_ks: list[Any],
    backend: "Backend",
) -> list[float]:
    """Compute projected residual |u_k^T @ delta @ v_k| per layer.

    The projected residual measures the component of the LoRA delta that
    directly perturbs sigma_k of the base weight. This is a tighter
    diagnostic than the spectral norm ratio ||BA||_2 / sigma_k, because
    Weyl's bound is tight only when the perturbation is aligned with the
    k-th singular direction.

    Each entry in ``lora_products`` is (scale, lora_a, lora_b, sigma_k).
    The effective delta in weight space is (scale * lora_a @ lora_b)^T,
    so:
        u_k^T @ delta @ v_k = scale * (lora_b @ u_k)^T @ (lora_a^T @ v_k)

    where lora_a is [in, r], lora_b is [r, out], u_k is [out, 1],
    v_k is [in, 1]. The intermediate products are r-vectors, so this is
    O(r) per layer — negligible cost.

    Args:
        lora_products: List of (scale, lora_a, lora_b, sigma_k) tuples.
        base_u_ks: List of k-th left singular vectors (one per layer).
        base_v_ks: List of k-th right singular vectors (one per layer).
        backend: Backend for computation.

    Returns:
        List of projected residual magnitudes (one per valid entry).

    Reference:
        Weyl (1912): |sigma_k(W+E) - sigma_k(W)| <= ||E||_2, with equality
        when E = ||E||_2 * u_k @ v_k^T. The projected residual
        |u_k^T E v_k| gives a first-order estimate of the actual sigma_k
        shift (Stewart, "Matrix Perturbation Theory", 1990, §2.4).
    """
    b = backend
    residuals: list[float] = []

    for i, (scale, lora_a, lora_b, sigma_k) in enumerate(lora_products):
        if sigma_k <= 0 or i >= len(base_u_ks) or i >= len(base_v_ks):
            continue

        try:
            u_k = b.astype(base_u_ks[i], "float32")
            v_k = b.astype(base_v_ks[i], "float32")
            lora_a_f32 = b.astype(lora_a, "float32")
            lora_b_f32 = b.astype(lora_b, "float32")

            # lora_b @ u_k: [r, out] @ [out, 1] = [r, 1]
            bu = b.matmul(lora_b_f32, u_k)
            # lora_a^T @ v_k: [r, in] @ [in, 1] = [r, 1]
            av = b.matmul(b.transpose(lora_a_f32), v_k)
            b.eval(bu, av)

            # dot product: bu^T @ av = [1, r] @ [r, 1] = scalar
            dot = b.matmul(b.transpose(bu), av)
            b.eval(dot)

            residual = abs(scale) * abs(float(b.to_scalar(dot)))
            residuals.append(residual)
        except Exception:
            continue

    return residuals


__all__ = [
    "DTYPE_THRESHOLD_F32",
    "compute_budget_ratios",
    "compute_initialization_vectors",
    "compute_projected_residuals",
    "is_budget_exhausted",
]
