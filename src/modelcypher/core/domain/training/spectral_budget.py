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
    # Power method on (AB)^T(AB): dominant-direction error decays as
    # (σ2/σ1)^(2 * n_iters). Ten iterations suppresses even modest gaps strongly.
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
        n_iters: Power iteration steps. For gap ratio ρ=σ2/σ1<1, direction
            error contracts as ρ^(2*n_iters) (standard power-method bound).

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
    # Normalization guard: 1/x overflows when x < 1/fmax (IEEE 754).
    # tiny = smallest normal float = 2^-126; 1/tiny = 2^126 < fmax = 2^128*(1-2^-24).
    # So norms >= tiny are safe for reciprocal.  L2 norms in float32 are either
    # exactly 0.0 or >= sqrt(min_subnormal) ≈ 2^-74.5 >> tiny, so this is
    # effectively a zero-check with IEEE 754 reciprocal safety margin.
    _norm_floor = float(backend.finfo().tiny)
    for _ in range(n_iters):
        # u = (A @ B) @ v  →  A @ (B @ v)
        u = backend.matmul(lora_a_f32, backend.matmul(lora_b_f32, v))
        backend.eval(u)

        u_norm = float(backend.to_scalar(backend.norm(u)))
        if u_norm < _norm_floor:
            break
        u = u * (1.0 / u_norm)

        # v = (A @ B)^T @ u  →  B^T @ (A^T @ u)
        v = backend.matmul(
            backend.transpose(lora_b_f32),
            backend.matmul(backend.transpose(lora_a_f32), u),
        )
        backend.eval(v)

        sigma = float(backend.to_scalar(backend.norm(v)))
        if sigma < _norm_floor:
            break
        v = v * (1.0 / sigma)
        backend.eval(v)

    return abs(scale) * sigma


def _pissa_delta_spectral_norm_power_iter(
    a_curr: Any,
    b_curr: Any,
    a_init: Any,
    b_init: Any,
    scale: float,
    backend: "Backend",
    n_iters: int = 10,
) -> float:
    """Estimate ||scale * (a_curr @ b_curr - a_init @ b_init)||_2 via power iteration.

    Computes the top singular value of the PiSSA displacement operator
    D = a_curr @ b_curr - a_init @ b_init without forming the full [in, out]
    matrix. Each power iteration step uses implicit matrix-vector products:

        D @ v  = a_curr @ (b_curr @ v) - a_init @ (b_init @ v)
        D^T @ u = b_curr^T @ (a_curr^T @ u) - b_init^T @ (a_init^T @ u)

    This costs 4 matmuls per direction per iteration (8 total), all through
    rank-r intermediates.

    Args:
        a_curr: Current A factor [in, r].
        b_curr: Current B factor [r, out].
        a_init: Initial A factor [in, r] (frozen at PiSSA injection).
        b_init: Initial B factor [r, out] (frozen at PiSSA injection).
        scale: LoRA scale factor (1.0 for PiSSA).
        backend: Backend for matmul/norm.
        n_iters: Power iteration steps.

    Returns:
        Estimated spectral norm of the displacement (float).
    """
    out_dim = b_curr.shape[1] if len(b_curr.shape) > 1 else b_curr.shape[0]
    v = backend.random_normal((out_dim, 1))
    v = backend.astype(v, "float32")
    backend.eval(v)

    ac_f32 = backend.astype(a_curr, "float32")
    bc_f32 = backend.astype(b_curr, "float32")
    ai_f32 = backend.astype(a_init, "float32")
    bi_f32 = backend.astype(b_init, "float32")
    backend.eval(ac_f32, bc_f32, ai_f32, bi_f32)

    sigma = 0.0
    _norm_floor = float(backend.finfo().tiny)
    for _ in range(n_iters):
        # u = D @ v = a_curr @ (b_curr @ v) - a_init @ (b_init @ v)
        u = backend.matmul(ac_f32, backend.matmul(bc_f32, v))
        u_init = backend.matmul(ai_f32, backend.matmul(bi_f32, v))
        u = u - u_init
        backend.eval(u)

        u_norm = float(backend.to_scalar(backend.norm(u)))
        if u_norm < _norm_floor:
            break
        u = u * (1.0 / u_norm)

        # v = D^T @ u = b_curr^T @ (a_curr^T @ u) - b_init^T @ (a_init^T @ u)
        v = backend.matmul(
            backend.transpose(bc_f32),
            backend.matmul(backend.transpose(ac_f32), u),
        )
        v_init = backend.matmul(
            backend.transpose(bi_f32),
            backend.matmul(backend.transpose(ai_f32), u),
        )
        v = v - v_init
        backend.eval(v)

        sigma = float(backend.to_scalar(backend.norm(v)))
        if sigma < _norm_floor:
            break
        v = v * (1.0 / sigma)
        backend.eval(v)

    return abs(scale) * sigma


def compute_pissa_budget_ratios(
    pissa_layers: list[tuple[float, Any, Any, Any, Any, float]],
    backend: "Backend",
) -> list[float]:
    """Compute spectral budget ratios for PiSSA displacement from initialization.

    Each entry in ``pissa_layers`` is
    (scale, a_curr, b_curr, a_init, b_init, sigma_k) where the displacement
    is ``scale * (a_curr @ b_curr - a_init @ b_init)`` and the ratio is
    ``||displacement||_spectral / sigma_k``.

    Uses implicit power iteration on the delta operator to avoid forming
    the full [in, out] displacement matrix.

    Args:
        pissa_layers: List of (scale, a_curr, b_curr, a_init, b_init, sigma_k).
        backend: Backend for computation.

    Returns:
        List of displacement ratios (one per valid entry).
    """
    ratios: list[float] = []

    for scale, a_curr, b_curr, a_init, b_init, sigma_k in pissa_layers:
        if sigma_k <= 0:
            continue

        try:
            spectral_norm = _pissa_delta_spectral_norm_power_iter(
                a_curr, b_curr, a_init, b_init, scale, backend,
            )
            ratios.append(spectral_norm / sigma_k)
        except Exception:
            continue

    return ratios


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


def compute_stable_rank(
    matrix: Any,
    backend: "Backend",
) -> float:
    """Compute stable rank of a matrix: ||A||²_F / ||A||²_2.

    Stable rank measures how distributed vs concentrated the matrix's energy
    is across its singular values. For an isotropic matrix (all singular
    values equal), stable_rank = rank. When energy concentrates in few
    directions, stable_rank approaches 1.0.

    Uses Frobenius norm (exact, O(mn)) and power iteration for spectral
    norm (O(mn * n_iters)), avoiding SVD crashes on ill-conditioned matrices.

    Args:
        matrix: Input matrix (any shape with 2 dimensions).
        backend: Backend for computation.

    Returns:
        Stable rank (float). Returns 0.0 if the matrix is zero.
    """
    b = backend
    M = b.astype(matrix, "float32")
    b.eval(M)

    # Frobenius norm: sqrt(sum of squared elements)
    frob_sq = float(b.to_scalar(b.sum(M * M)))
    if frob_sq <= 0.0:
        return 0.0

    # Spectral norm via power iteration
    m, n = int(M.shape[0]), int(M.shape[1])
    v = b.random_normal((n, 1))
    v = b.astype(v, "float32")
    b.eval(v)

    _norm_floor = float(b.finfo().tiny)
    sigma = 0.0
    for _ in range(10):
        u = b.matmul(M, v)
        b.eval(u)
        u_norm = float(b.to_scalar(b.norm(u)))
        if u_norm < _norm_floor:
            break
        u = u * (1.0 / u_norm)
        v = b.matmul(b.transpose(M), u)
        b.eval(v)
        sigma = float(b.to_scalar(b.norm(v)))
        if sigma < _norm_floor:
            break
        v = v * (1.0 / sigma)
        b.eval(v)

    spectral_sq = sigma * sigma
    if spectral_sq <= 0.0:
        return 0.0

    return frob_sq / spectral_sq


def compute_initialization_vectors(
    weight: Any,
    structural_rank: int,
    backend: "Backend",
    oversampling: int = 5,
    power_iters: int = 2,
    seed: int | None = None,
) -> tuple[Any, Any, float]:
    """Extract k-th left and right singular vectors of a base weight.

    These vectors define the direction most sensitive to perturbation at
    the structural rank boundary. Storing them at initialization allows
    efficient projected residual monitoring during training.

    Uses randomized truncated SVD: O(m·n·(k+p)·(2q+1)) via matrix-vector
    products only, never forming the full decomposition. This is both
    faster and more numerically stable than full SVD on large matrices,
    and avoids MLX sgesvdx_ crashes on ill-conditioned weights.

    When ``structural_rank + oversampling >= min(m, n)``, the randomized
    approach would compute a near-full decomposition, so full SVD is
    used instead (same asymptotic cost, exact result).

    Args:
        weight: Base weight matrix [out, in].
        structural_rank: 1-indexed Shannon effective rank boundary. The
            returned vectors correspond to index (structural_rank - 1).
        backend: Backend for computation.
        oversampling: Extra columns in random projection. Halko et al.
            (2011), §10.3: p ≥ 2 suffices for the bound; p = 5 gives
            high-probability capture for float32.
        power_iters: Subspace power iterations. Each iteration squares
            the spectral gap ratio (Halko et al. 2011, Algorithm 4.3):
            error decays as (σ_{k+1}/σ_k)^{2q+1}. q = 2 is robust for
            all practical gap ratios.
        seed: RNG seed for reproducible random projection. When None,
            uses current backend RNG state (non-deterministic across runs).

    Returns:
        (u_k, v_k, quality) where u_k is [out, 1], v_k is [in, 1], and
        quality = |u_k^T W v_k| / ||W||_spectral ∈ [0, 1]. For the
        exact k-th singular direction, quality = σ_k / σ_1. Lower values
        indicate either a deep interior rank (σ_k << σ_1) or approximation
        error from the randomized projection. A quality value significantly
        below the expected σ_k / σ_1 ratio signals degraded accuracy.

    References:
        Halko, N., Martinsson, P.G. & Tropp, J.A. (2011). Finding
        Structure with Randomness: Probabilistic Algorithms for
        Constructing Approximate Matrix Decompositions. SIAM Review,
        53(2), 217-288. Theorem 10.5, Algorithm 4.3.
    """
    b = backend
    W = b.astype(weight, "float32")
    b.eval(W)

    m, n = int(W.shape[0]), int(W.shape[1])
    k = max(0, min(structural_rank - 1, min(m, n) - 1))
    target = k + 1 + oversampling  # columns in random projection

    # If target covers most of the matrix, full SVD is no more expensive.
    if target >= min(m, n):
        U, S, Vt = b.svd(W, compute_uv=True)
        b.eval(U, S, Vt)
        u_k = b.reshape(U[:, k], (-1, 1))
        v_k = b.reshape(Vt[k, :], (-1, 1))
        b.eval(u_k, v_k)
        # Exact quality: σ_k / σ_1
        sigma_1 = float(b.to_scalar(S[0])) if S.shape[0] > 0 else 1.0
        sigma_k_val = float(b.to_scalar(S[k])) if k < S.shape[0] else 0.0
        quality = sigma_k_val / sigma_1 if sigma_1 > 0 else 0.0
        return u_k, v_k, quality

    # --- Randomized truncated SVD (Halko et al. 2011, Algorithm 5.1) ---

    # Deterministic seeding for reproducibility across runs/diagnostics.
    if seed is not None:
        b.random_seed(seed)

    # Step 1: Random Gaussian projection Ω ∈ R^{n × target}
    omega = b.random_normal((n, target))
    omega = b.astype(omega, "float32")
    b.eval(omega)

    # Step 2: Sample matrix Y = W @ Ω ∈ R^{m × target}
    Y = b.matmul(W, omega)
    b.eval(Y)

    # Step 3: Power iteration Y = (W @ W^T)^q @ Y
    # Improves subspace capture; error decays as (σ_{k+1}/σ_k)^{2q+1}.
    for _ in range(power_iters):
        Y = b.matmul(b.transpose(W), Y)  # W^T @ Y: [n, target]
        b.eval(Y)
        Y = b.matmul(W, Y)               # W @ (W^T @ Y): [m, target]
        b.eval(Y)

    # Step 4: Orthonormal basis for range(Y) via thin SVD.
    # Y = Q @ diag(s) @ Vt_y; Q = U_y gives orthonormal columns.
    # Thin SVD of [m, target] is O(m·target²) — cheap since target << m.
    Q, _s_y, _vt_y = b.svd(Y, compute_uv=True)
    b.eval(Q)

    # Step 5: Project W into the low-rank subspace: B = Q^T @ W ∈ R^{target × n}
    B = b.matmul(b.transpose(Q), W)
    b.eval(B)

    # Step 6: SVD of the small matrix B.
    # B is [target, n] where target = k + 1 + p — fast and stable.
    U_B, S_B, Vt_B = b.svd(B, compute_uv=True)
    b.eval(U_B, S_B, Vt_B)

    # Step 7: Recover k-th singular vectors of W.
    # U ≈ Q @ U_B, so u_k = Q @ U_B[:, k]
    idx = min(k, int(U_B.shape[1]) - 1)
    u_b_k = b.reshape(U_B[:, idx], (-1, 1))   # [target, 1]
    u_k = b.matmul(Q, u_b_k)                   # [m, 1]
    v_k = b.reshape(Vt_B[idx, :], (-1, 1))     # [n, 1]
    b.eval(u_k, v_k)

    # Quality metric: |u_k^T W v_k| / σ_1(B).
    # For the true k-th singular direction, this equals σ_k / σ_1.
    # Degradation below this ratio signals approximation error.
    Wv = b.matmul(W, v_k)       # [m, 1]
    b.eval(Wv)
    dot = b.matmul(b.transpose(u_k), Wv)  # scalar
    b.eval(dot)
    sigma_1_approx = float(b.to_scalar(S_B[0])) if S_B.shape[0] > 0 else 1.0
    quality = abs(float(b.to_scalar(dot))) / sigma_1_approx if sigma_1_approx > 0 else 0.0

    return u_k, v_k, quality


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
    "compute_pissa_budget_ratios",
    "compute_projected_residuals",
    "compute_stable_rank",
    "is_budget_exhausted",
]
