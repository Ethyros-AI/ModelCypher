"""Tikhonov eigenvalue-weighted quantization correction.

Applies continuous spectral weighting to project quantization error into
activation-relevant directions.  The Marchenko-Pastur noise edge
(Marchenko & Pastur, 1967) separates signal from noise eigenvalues;
Tikhonov weights w_i = λ_i / (λ_i + α) downweight noisy directions
continuously — no integer rank cutoff, no magic numbers.

Every number in the projection traces to eigenvalues (measured data) or
the MP noise edge (theorem).

Key property: directions with small eigenvalues (low-usage) preserve their
quantization residuals.  This is functionally important — residuals in
unused directions act as anti-degeneration noise (discovered 2026-02-27).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain.geometry.mp_noise_estimator import estimate_mp_noise

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


@dataclass(frozen=True)
class TikhonovLayerResult:
    """Per-layer diagnostics from Tikhonov correction."""

    layer_key: str
    E_total_frob: float
    delta_frob: float
    E_residual_frob: float
    correction_fraction: float
    preserved_fraction: float
    tikhonov_effective_rank: float
    mp_noise_edge: float
    top_weight: float
    D_eff: float


@dataclass(frozen=True)
class TikhonovCorrectionResult:
    """Aggregate result from correcting multiple layers."""

    per_layer: list[TikhonovLayerResult] = field(default_factory=list)
    n_layers_corrected: int = 0
    n_layers_skipped: int = 0
    mean_correction_fraction: float = 0.0


def compute_mp_noise_edge(
    eigenvalues: Any,
    n_tokens: int,
    dimensionality: int,
    backend: "Backend",
) -> float:
    """Marchenko-Pastur noise edge from activation eigenspectrum.

    σ² = robust MP bulk mean after excluding exact zeros and signal spikes
    γ  = D / N          (aspect ratio)
    α  = σ² × (1 + √γ)²

    All inputs are measured data or the MP theorem (Marchenko & Pastur, 1967).

    Args:
        eigenvalues: Eigenvalues of X^T @ X in descending order, shape [D].
        n_tokens: Number of activation tokens N (columns of activation matrix).
        dimensionality: Hidden dimension D (rows of activation matrix after
            centering).
        backend: Backend protocol instance.

    Returns:
        MP noise edge α (float).
    """
    if n_tokens <= 0 or dimensionality <= 0:
        raise ValueError(
            f"n_tokens and dimensionality must be > 0, got {n_tokens}, {dimensionality}"
        )

    estimate = estimate_mp_noise(
        eigenvalues,
        n_samples=n_tokens,
        n_features=dimensionality,
        backend=backend,
    )
    return estimate.upper_edge


def compute_tikhonov_weights(
    eigenvalues: Any,
    mp_noise_edge: float,
    backend: "Backend",
) -> Any:
    """Tikhonov weights: w_i = λ_i / (λ_i + α).

    Continuous.  No integer rank.  Every number from data or MP theorem.

    Args:
        eigenvalues: Eigenvalues in descending order, shape [D].
        mp_noise_edge: α from compute_mp_noise_edge().
        backend: Backend protocol instance.

    Returns:
        Array of weights in [0, 1], same shape as eigenvalues.
    """
    alpha = backend.array([mp_noise_edge], dtype="float32")
    backend.eval(alpha)
    weights = eigenvalues / (eigenvalues + alpha)
    backend.eval(weights)
    return weights


def correct_projection_tikhonov(
    quantized_weight: Any,
    fp_weight: Any,
    eigenvectors: Any,
    tikhonov_weights: Any,
    backend: "Backend",
    layer_key: str = "",
    *,
    tikhonov_effective_rank: float = 0.0,
    mp_noise_edge: float = 0.0,
    D_eff: float = 0.0,
) -> tuple[Any, TikhonovLayerResult | None]:
    """Apply Tikhonov-weighted correction to a single weight matrix.

    E = W_fp - W_q
    Delta = E @ V @ diag(w) @ V^T
    W_corrected = W_q + Delta

    Directions with w_i → 1 (high eigenvalue) are fully corrected.
    Directions with w_i → 0 (low eigenvalue) preserve quantization residual.

    Args:
        quantized_weight: Quantized weight matrix W_q, shape [out, in].
        fp_weight: Full-precision weight matrix W_fp, shape [out, in].
        eigenvectors: Eigenvectors of activation covariance, shape [in, D].
        tikhonov_weights: Per-direction weights from compute_tikhonov_weights(),
            shape [D].
        backend: Backend protocol instance.
        layer_key: Identifier for diagnostics.
        tikhonov_effective_rank: Pre-computed sum of weights (diagnostic).
        mp_noise_edge: Pre-computed MP noise edge (diagnostic).
        D_eff: Pre-computed participation ratio (diagnostic).

    Returns:
        (corrected_weight, diagnostics) where corrected_weight has the same
        dtype as quantized_weight.  diagnostics is None if error is zero.
    """
    q_w = backend.astype(quantized_weight, "float32")
    fp_w = backend.astype(fp_weight, "float32")
    E = fp_w - q_w  # [out, in]
    backend.eval(E)

    E_frob_sq = float(backend.to_scalar(backend.sum(E * E)))
    if E_frob_sq <= 0:
        return quantized_weight, None

    # Delta = E @ V @ diag(w) @ V^T
    # Computed as (E @ V) * w @ V^T to avoid constructing diag matrix.
    E_V = backend.matmul(E, eigenvectors)  # [out, D]
    E_V_weighted = E_V * tikhonov_weights  # [out, D] (broadcast)
    Delta = backend.matmul(E_V_weighted, backend.transpose(eigenvectors))  # [out, in]
    backend.eval(Delta)

    Delta_frob_sq = float(backend.to_scalar(backend.sum(Delta * Delta)))
    E_residual = E - Delta
    backend.eval(E_residual)
    E_residual_frob_sq = float(backend.to_scalar(backend.sum(E_residual * E_residual)))

    correction_fraction = Delta_frob_sq / E_frob_sq
    preserved_fraction = E_residual_frob_sq / E_frob_sq

    corrected = q_w + Delta
    backend.eval(corrected)

    top_weight = 0.0
    n_weights = int(tikhonov_weights.shape[0]) if hasattr(tikhonov_weights, "shape") else 0
    if n_weights > 0:
        top_weight = float(backend.to_scalar(tikhonov_weights[0]))

    result = TikhonovLayerResult(
        layer_key=layer_key,
        E_total_frob=math.sqrt(E_frob_sq),
        delta_frob=math.sqrt(Delta_frob_sq),
        E_residual_frob=math.sqrt(E_residual_frob_sq),
        correction_fraction=correction_fraction,
        preserved_fraction=preserved_fraction,
        tikhonov_effective_rank=tikhonov_effective_rank,
        mp_noise_edge=mp_noise_edge,
        top_weight=top_weight,
        D_eff=D_eff,
    )

    return corrected, result
