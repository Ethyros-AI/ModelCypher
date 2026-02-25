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

"""K-FAC projector for behavior-preserving weight-delta projection.

Implements projection into the K-FAC null space via Kronecker eigensystem:

    Q = U_s^T @ delta_W @ U_a
    Q_projected = Q * null_mask
    delta_projected = U_s @ Q_projected @ U_a^T
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.kfac_diagnostic import (
    KFACDiagnosticResult,
    compute_kfac_diagnostic,
)
from modelcypher.core.domain.geometry.numerical_stability import precision_dtype

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class KFACFactors:
    """K-FAC eigensystem factors for matrix-free projection."""

    A_eigvals: "Array"  # [in_dim]
    A_eigvecs: "Array"  # [in_dim, in_dim]
    S_eigvals: "Array"  # [out_dim]
    S_eigvecs: "Array"  # [out_dim, out_dim]
    null_mask: "Array"  # [out_dim, in_dim]
    gain_ratio: float
    kron_threshold: float
    max_kron_eigenvalue: float
    in_dim: int
    out_dim: int


def factors_from_diagnostic(diagnostic: KFACDiagnosticResult) -> KFACFactors:
    """Convert diagnostic output into projector factors."""
    return KFACFactors(
        A_eigvals=diagnostic.activation_eigenvalues,
        A_eigvecs=diagnostic.activation_eigenvectors,
        S_eigvals=diagnostic.output_gradient_eigenvalues,
        S_eigvecs=diagnostic.output_gradient_eigenvectors,
        null_mask=diagnostic.kron_null_mask,
        gain_ratio=diagnostic.kfac_gain_ratio,
        kron_threshold=diagnostic.kron_threshold,
        max_kron_eigenvalue=diagnostic.max_kron_eigenvalue,
        in_dim=diagnostic.in_dim,
        out_dim=diagnostic.out_dim,
    )


def compute_kfac_factors(
    input_activations: "Array",
    output_gradients: "Array",
    *,
    backend: "Backend | None" = None,
) -> KFACFactors:
    """Compute K-FAC factors from activations and per-probe output gradients."""
    b = backend or get_default_backend()
    diagnostic = compute_kfac_diagnostic(
        input_activations=input_activations,
        output_gradients=output_gradients,
        backend=b,
    )
    return factors_from_diagnostic(diagnostic)


def project_kfac(
    delta_weight: "Array",
    factors: KFACFactors,
    *,
    backend: "Backend | None" = None,
) -> "Array":
    """Project delta_weight into K-FAC null space.

    Args:
        delta_weight: Weight delta matrix [out_dim, in_dim].
        factors: KFACFactors with eigensystem and null mask.
        backend: Backend for tensor operations.

    Returns:
        Projected weight delta [out_dim, in_dim].
    """
    b = backend or get_default_backend()
    delta = b.array(delta_weight)

    if len(b.shape(delta)) != 2:
        raise ValueError(f"delta_weight must be 2D [out_dim, in_dim], got {b.shape(delta)}")
    if int(delta.shape[0]) != factors.out_dim or int(delta.shape[1]) != factors.in_dim:
        raise ValueError(
            "delta_weight shape does not match K-FAC factors: "
            f"delta={b.shape(delta)}, factors=({factors.out_dim}, {factors.in_dim})",
        )

    compute_dtype = precision_dtype(b, reference=delta)
    for arr in (
        factors.A_eigvecs,
        factors.S_eigvecs,
    ):
        if hasattr(arr, "dtype"):
            try:
                if b.finfo(arr.dtype).eps < b.finfo(compute_dtype).eps:
                    compute_dtype = arr.dtype
            except Exception:
                pass

    delta = b.astype(delta, compute_dtype)
    A_eigvecs = b.astype(factors.A_eigvecs, compute_dtype)
    S_eigvecs = b.astype(factors.S_eigvecs, compute_dtype)
    null_mask = b.astype(factors.null_mask, compute_dtype)
    b.eval(delta, A_eigvecs, S_eigvecs, null_mask)

    # Transform to Kronecker eigenbasis.
    Q = b.matmul(b.transpose(S_eigvecs), delta)
    Q = b.matmul(Q, A_eigvecs)
    b.eval(Q)

    # Keep only null directions.
    Q_projected = Q * null_mask
    b.eval(Q_projected)

    # Transform back to weight coordinates.
    projected = b.matmul(S_eigvecs, Q_projected)
    projected = b.matmul(projected, b.transpose(A_eigvecs))
    b.eval(projected)
    return projected


__all__ = [
    "KFACFactors",
    "compute_kfac_factors",
    "factors_from_diagnostic",
    "project_kfac",
]

