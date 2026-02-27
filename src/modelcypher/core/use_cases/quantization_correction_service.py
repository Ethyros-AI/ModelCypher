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

"""Tikhonov quantization correction service (experimental).

Applies eigenvalue-weighted Tikhonov projection to partially reverse
quantization damage in weight matrices, using the activation covariance
eigenbasis and Marchenko-Pastur derived regularization.

NOT promoted to CLI. Available for experiments and tests only.
Promotion requires validation on multiple models (see CLAUDE.md policy).

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

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from modelcypher.core.domain.geometry.marchenko_pastur import (
    marchenko_pastur_noise_edge,
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
    eigenvalue_sum = float(b.to_scalar(b.sum(eigenvalues)))
    mp_edge = marchenko_pastur_noise_edge(eigenvalue_sum, n_features, n_samples)
    weights = eigenvalues / (eigenvalues + mp_edge)
    b.eval(weights)
    return weights, mp_edge
