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

"""Tikhonov quantization correction service.

Applies eigenvalue-weighted Tikhonov projection to partially reverse
quantization damage in weight matrices, using the activation covariance
eigenbasis and Marchenko-Pastur derived regularization.

Validated on 3 models, 2 architectures (Qwen3 + Llama):
    Qwen3-1.7B:  CKA +0.014, PPL -0.06, degen -0.05
    Qwen3-8B:    CKA +0.033, PPL -0.04, degen -0.02
    Llama-3.2-3B: PPL -0.08, degen -0.06

CLI: ``mc quantize correct``

Mathematical basis:
    E = W_fp - W_q  (quantization error)
    C = X^T X / N  (activation covariance from calibration data)
    C = V diag(lambda) V^T  (eigendecomposition)
    alpha = sigma_sq * (1 + sqrt(D/N))^2  (Marchenko-Pastur noise edge)
    w_i = lambda_i / (lambda_i + alpha)  (Tikhonov weights)
    Delta = E @ V @ diag(w) @ V^T  (correction)
    W_corrected = W_q + Delta

Citation: Marchenko & Pastur (1967), Tikhonov (1963).
"""

from __future__ import annotations

import gc
import logging
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain.geometry.marchenko_pastur import (
    marchenko_pastur_noise_edge,
)
from modelcypher.core.domain.training.tikhonov_correction import (
    compute_mp_noise_edge,
    compute_tikhonov_weights,
)
from modelcypher.core.domain.training.tikhonov_correction import (
    correct_projection_tikhonov as _domain_correct_projection,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProjectionCorrectionResult:
    """Result of correcting a single weight projection."""

    layer_key: str
    E_total_frob: float
    delta_frob: float
    E_residual_frob: float
    correction_fraction: float
    preserved_fraction: float


@dataclass(frozen=True)
class LayerCorrectionResult:
    """Per-layer correction diagnostics including MP profile."""

    layer_idx: int
    n_features: int
    n_samples: int
    D_eff: float
    mp_edge: float
    sigma_sq: float
    aspect_ratio: float
    effective_rank: float
    top_eigenvalues: list[float]
    top_tikhonov_weights: list[float]
    projections: list[ProjectionCorrectionResult]
    skipped_keys: list[str]
    correction_fraction: float
    preserved_fraction: float
    time_seconds: float


@dataclass(frozen=True)
class QuantizationCorrectionResult:
    """Full correction result across all layers."""

    n_layers: int
    n_projections_corrected: int
    aggregate_correction_fraction: float
    aggregate_preserved_fraction: float
    per_layer: list[LayerCorrectionResult] = field(default_factory=list)


def correct_projection_tikhonov(
    q_weight: "Array",
    fp_weight: "Array",
    eigvecs: "Array",
    tikhonov_weights: "Array",
    backend: "Backend",
) -> tuple["Array", ProjectionCorrectionResult | None]:
    """Apply Tikhonov-weighted correction to a single weight matrix.

    Args:
        q_weight: Quantized (or dequantized) weight [out, in].
        fp_weight: Full-precision reference weight [out, in].
        eigvecs: Eigenvectors of activation covariance [in, D].
        tikhonov_weights: Per-direction weights [D].
        backend: Computation backend.

    Returns:
        (corrected_weight, diagnostics) or (q_weight, None) if no correction needed.
    """
    b = backend
    E = fp_weight - q_weight
    b.eval(E)

    E_frob_sq = float(b.to_scalar(b.sum(E * E)))
    if E_frob_sq <= 0:
        return q_weight, None

    # Delta = E @ V @ diag(w) @ V^T  (computed without forming diag matrix)
    E_V = b.matmul(E, eigvecs)
    E_V_weighted = E_V * tikhonov_weights
    Delta = b.matmul(E_V_weighted, b.transpose(eigvecs))
    b.eval(Delta)

    Delta_frob_sq = float(b.to_scalar(b.sum(Delta * Delta)))
    E_residual = E - Delta
    b.eval(E_residual)
    E_residual_frob_sq = float(b.to_scalar(b.sum(E_residual * E_residual)))

    corrected = q_weight + Delta
    b.eval(corrected)

    result = ProjectionCorrectionResult(
        layer_key="",  # Caller fills this in
        E_total_frob=math.sqrt(E_frob_sq),
        delta_frob=math.sqrt(Delta_frob_sq),
        E_residual_frob=math.sqrt(E_residual_frob_sq),
        correction_fraction=Delta_frob_sq / E_frob_sq,
        preserved_fraction=E_residual_frob_sq / E_frob_sq,
    )
    return corrected, result


def compute_layer_tikhonov_weights(
    eigenvalues: "Array",
    n_features: int,
    n_samples: int,
    backend: "Backend",
) -> tuple["Array", float]:
    """Compute Tikhonov weights from eigenvalues using MP noise edge.

    Args:
        eigenvalues: Eigenvalues of activation covariance (descending order).
        n_features: D, dimensionality.
        n_samples: N, number of activation vectors (tokens × sequences).
        backend: Computation backend.

    Returns:
        (tikhonov_weights_array, mp_edge)
    """
    b = backend
    mp_edge = marchenko_pastur_noise_edge(
        eigenvalues,
        n_features,
        n_samples,
        backend=b,
    )
    weights = eigenvalues / (eigenvalues + mp_edge)
    b.eval(weights)
    return weights, mp_edge


# ── Projection classification ────────────────────────────────────────────

# Projections whose input is h (or layer_norm(h) — same subspace).
# These are correctable because the activation covariance eigenbasis
# captures their input space.
_H_INPUT_PROJS: dict[str, tuple[str, ...]] = {
    "self_attn": ("q_proj", "k_proj", "v_proj"),
    "mlp": ("up_proj", "gate_proj"),
}

# Projections whose input is a different subspace (attention output,
# MLP intermediate). Skipped — their input covariance differs from h.
_SKIPPED_PROJS: dict[str, tuple[str, ...]] = {
    "self_attn": ("o_proj",),
    "mlp": ("down_proj",),
}


# ── Sequential correction orchestration ──────────────────────────────────


def run_sequential_correction(
    model: Any,
    fp_weights: dict[str, "Array"],
    tokenizer: Any,
    eval_texts: list[str],
    backend: "Backend",
    *,
    n_calibration: int = 30,
    max_seq_len: int = 128,
) -> QuantizationCorrectionResult:
    """Run sequential layer-by-layer Tikhonov correction on a dequantized model.

    For each layer l (0 → L-1), sequentially:
      1. Flatten hidden activations → [N_tokens, D]
      2. Eigendecompose activation covariance C = X^T X / N
      3. Compute MP noise edge α and Tikhonov weights w_i = λ_i / (λ_i + α)
      4. Correct each h-input projection: Delta = E @ V @ diag(w) @ V^T
      5. Forward pass through corrected layer for next layer's activations

    The model is modified in-place. Caller should dequantize quantized modules
    before calling this (QuantizedLinear → Linear).

    Args:
        model: Neural network model (dequantized). Modified in-place.
        fp_weights: Full-precision reference weights as flat dict
            (key format: "model.layers.{idx}.{block}.{proj}.weight").
        tokenizer: Tokenizer for encoding calibration texts.
        eval_texts: Calibration texts for activation covariance.
        backend: Computation backend.
        n_calibration: Number of calibration samples. 30 >> D_eff~3-5
            (measured). CLI-overridable, not a decision boundary.
        max_seq_len: Token truncation length. Memory-compute tradeoff.

    Returns:
        QuantizationCorrectionResult with per-layer diagnostics.
    """
    b = backend

    base = getattr(model, "model", model)
    if not hasattr(base, "layers"):
        raise ValueError("Model has no .layers attribute — unsupported architecture")

    n_layers = len(base.layers)
    logger.info(
        "Sequential Tikhonov correction: %d layers, %d calibration samples",
        n_layers,
        n_calibration,
    )

    # Tokenize calibration data
    cal_texts = eval_texts[:n_calibration]
    all_token_ids = []
    for text in cal_texts:
        tokens = tokenizer.encode(text)
        all_token_ids.append(b.array(tokens[:max_seq_len]))

    # Pad to uniform length and stack into batch
    max_seq = max(t.shape[0] for t in all_token_ids)
    padded = []
    for t in all_token_ids:
        seq_len = int(t.shape[0])
        if seq_len < max_seq:
            pad = b.zeros((max_seq - seq_len,), dtype=t)
            t = b.concatenate([t, pad])
        padded.append(t)
    batch = b.stack(padded)
    b.eval(batch)

    # Embedding
    h = base.embed_tokens(batch)
    b.eval(h)

    per_layer: list[LayerCorrectionResult] = []
    total_corrected = 0
    total_e_sq = 0.0
    total_delta_sq = 0.0
    total_residual_sq = 0.0

    for layer_idx, layer in enumerate(base.layers):
        layer_start = time.monotonic()

        # Flatten activations: [n_samples, seq_len, D] → [N_tokens, D]
        X = b.reshape(h, (-1, int(h.shape[-1])))
        X = b.astype(X, "float32")
        N_tok, D = int(X.shape[0]), int(X.shape[1])
        b.eval(X)

        # Activation covariance
        XtX = b.matmul(b.transpose(X), X)
        XtX = XtX / N_tok
        b.eval(XtX)

        # Eigendecompose
        try:
            eigvals, eigvecs = b.eigh(XtX)
            b.eval(eigvals, eigvecs)
        except Exception as exc:
            logger.warning(
                "  eigh failed for layer %d: %s, skipping", layer_idx, exc
            )
            h = layer(h)
            b.eval(h)
            del X, XtX
            gc.collect()
            continue

        # eigh returns ascending; flip to descending
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]

        # Clamp negative eigenvalues (numerical noise from eigh)
        zero = b.array(0.0, dtype=eigvals)
        eigvals = b.maximum(eigvals, zero)
        b.eval(eigvals)

        # Participation ratio (diagnostic)
        total_var = float(b.to_scalar(b.sum(eigvals)))
        sum_sq = float(b.to_scalar(b.sum(eigvals * eigvals)))
        D_eff = total_var**2 / sum_sq if sum_sq > 0 else float(D)

        # MP noise edge + Tikhonov weights (domain functions)
        mp_edge = compute_mp_noise_edge(
            eigvals, n_tokens=N_tok, dimensionality=D, backend=b
        )
        sigma_sq = total_var / D
        aspect = D / N_tok
        tikhonov_w = compute_tikhonov_weights(eigvals, mp_edge, backend=b)

        effective_rank = float(b.to_scalar(b.sum(tikhonov_w)))

        # Top eigenvalue diagnostics
        n_report = min(10, D)
        top_eigvals = [float(b.to_scalar(eigvals[i])) for i in range(n_report)]
        top_weights = [float(b.to_scalar(tikhonov_w[i])) for i in range(n_report)]

        # Correct h-input projections
        projections: list[ProjectionCorrectionResult] = []
        skipped_keys: list[str] = []

        for block_name, proj_names in _H_INPUT_PROJS.items():
            block = getattr(layer, block_name, None)
            if block is None:
                continue
            for proj_name in proj_names:
                proj = getattr(block, proj_name, None)
                if proj is None or not hasattr(proj, "weight"):
                    continue
                key = f"model.layers.{layer_idx}.{block_name}.{proj_name}.weight"
                fp_w = fp_weights.get(key)
                if fp_w is None:
                    continue

                corrected, result = _domain_correct_projection(
                    quantized_weight=proj.weight,
                    fp_weight=fp_w,
                    eigenvectors=eigvecs,
                    tikhonov_weights=tikhonov_w,
                    backend=b,
                    layer_key=key,
                    tikhonov_effective_rank=effective_rank,
                    mp_noise_edge=mp_edge,
                    D_eff=D_eff,
                )
                if result is not None:
                    b.eval(corrected)
                    proj.weight = b.astype(corrected, proj.weight)
                    projections.append(
                        ProjectionCorrectionResult(
                            layer_key=key,
                            E_total_frob=result.E_total_frob,
                            delta_frob=result.delta_frob,
                            E_residual_frob=result.E_residual_frob,
                            correction_fraction=result.correction_fraction,
                            preserved_fraction=result.preserved_fraction,
                        )
                    )

        for block_name, proj_names in _SKIPPED_PROJS.items():
            for proj_name in proj_names:
                key = f"model.layers.{layer_idx}.{block_name}.{proj_name}.weight"
                if key in fp_weights:
                    skipped_keys.append(key)

        total_corrected += len(projections)

        # Forward pass with corrected weights
        h = layer(h)
        b.eval(h)

        layer_time = time.monotonic() - layer_start

        # Layer-level aggregation
        layer_e_sq = sum(p.E_total_frob**2 for p in projections)
        layer_delta_sq = sum(p.delta_frob**2 for p in projections)
        layer_residual_sq = sum(p.E_residual_frob**2 for p in projections)
        total_e_sq += layer_e_sq
        total_delta_sq += layer_delta_sq
        total_residual_sq += layer_residual_sq

        per_layer.append(
            LayerCorrectionResult(
                layer_idx=layer_idx,
                n_features=D,
                n_samples=N_tok,
                D_eff=D_eff,
                mp_edge=mp_edge,
                sigma_sq=sigma_sq,
                aspect_ratio=aspect,
                effective_rank=effective_rank,
                top_eigenvalues=top_eigvals,
                top_tikhonov_weights=top_weights,
                projections=projections,
                skipped_keys=skipped_keys,
                correction_fraction=(
                    layer_delta_sq / layer_e_sq if layer_e_sq > 0 else 0.0
                ),
                preserved_fraction=(
                    layer_residual_sq / layer_e_sq if layer_e_sq > 0 else 0.0
                ),
                time_seconds=layer_time,
            )
        )

        if layer_idx % 7 == 0 or layer_idx == n_layers - 1:
            mean_frac = (
                sum(p.correction_fraction for p in projections) / len(projections)
                if projections
                else 0.0
            )
            logger.info(
                "  Layer %d/%d: D_eff=%.1f, mp_edge=%.2e, eff_rank=%.1f, "
                "correction_frac=%.4f, corrected=%d, skipped=%d (%.1fs)",
                layer_idx,
                n_layers - 1,
                D_eff,
                mp_edge,
                effective_rank,
                mean_frac,
                len(projections),
                len(skipped_keys),
                layer_time,
            )

        del eigvals, eigvecs, tikhonov_w
        gc.collect()

    return QuantizationCorrectionResult(
        n_layers=n_layers,
        n_projections_corrected=total_corrected,
        aggregate_correction_fraction=(
            total_delta_sq / total_e_sq if total_e_sq > 0 else 0.0
        ),
        aggregate_preserved_fraction=(
            total_residual_sq / total_e_sq if total_e_sq > 0 else 0.0
        ),
        per_layer=per_layer,
    )
