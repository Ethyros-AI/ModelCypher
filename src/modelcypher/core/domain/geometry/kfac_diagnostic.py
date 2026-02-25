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

"""K-FAC diagnostics for behavior-null capacity analysis.

This module quantifies the gap between:
1. Activation-only null space (Null(K_cap), conservative)
2. K-FAC Gauss-Newton null space approximation (Null(G_cap), larger)

All thresholds are dtype-derived from IEEE-754 machine precision.
No fixed heuristics are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
    precision_dtype,
    svd_rank_threshold,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class KFACDiagnosticResult:
    """Raw K-FAC null-space diagnostics for one weight matrix.

    Attributes:
        n_probes: Number of probes/samples used.
        in_dim: Input dimension of the weight matrix.
        out_dim: Output dimension of the weight matrix.
        activation_rank: Numeric rank of activation covariance A^T A.
        activation_null_rank_input: Input-space null rank (in_dim - activation_rank).
        activation_null_rank_weight: Activation-only weight-space null dimensions.
            This is out_dim * activation_null_rank_input.
        kfac_null_rank: K-FAC null dimensions from Kronecker spectrum.
        kfac_gain_ratio: kfac_null_rank / activation_null_rank_weight.
        activation_rank_threshold: Rank threshold for activation eigenvalues.
        kron_threshold: Null threshold for Kronecker eigenvalue grid.
        max_activation_eigenvalue: Max eigenvalue of A^T A / N.
        max_output_gradient_eigenvalue: Max eigenvalue of S^T S / N.
        max_kron_eigenvalue: Product of max activation/output-gradient eigenvalues.
        activation_eigenvalues: Eigenvalues of A^T A / N (descending).
        activation_eigenvectors: Eigenvectors of A^T A / N (column-wise, descending).
        output_gradient_eigenvalues: Eigenvalues of S^T S / N (descending).
        output_gradient_eigenvectors: Eigenvectors of S^T S / N (column-wise, descending).
        kron_null_mask: Boolean mask [out_dim, in_dim] in Kronecker eigenbasis.
            True entries are classified null directions.
    """

    n_probes: int
    in_dim: int
    out_dim: int
    activation_rank: int
    activation_null_rank_input: int
    activation_null_rank_weight: int
    kfac_null_rank: int
    kfac_gain_ratio: float
    activation_rank_threshold: float
    kron_threshold: float
    max_activation_eigenvalue: float
    max_output_gradient_eigenvalue: float
    max_kron_eigenvalue: float
    activation_eigenvalues: "Array"
    activation_eigenvectors: "Array"
    output_gradient_eigenvalues: "Array"
    output_gradient_eigenvectors: "Array"
    kron_null_mask: "Array"


def _sorted_eigh_desc(matrix: "Array", backend: "Backend") -> tuple["Array", "Array"]:
    """Eigen-decompose a symmetric matrix and sort descending."""
    b = backend
    eigvals, eigvecs = b.eigh(matrix)
    b.eval(eigvals, eigvecs)

    n = int(eigvals.shape[0])
    idx = b.arange(n - 1, -1, -1)
    eigvals = b.take(eigvals, idx, axis=0)
    eigvecs = b.take(eigvecs, idx, axis=1)
    b.eval(eigvals, eigvecs)
    return eigvals, eigvecs


def estimate_output_gradients_from_weight_gradients(
    per_probe_weight_gradients: "Array",
    input_activations: "Array",
    *,
    weight_shape: tuple[int, int] | None = None,
    backend: "Backend | None" = None,
) -> "Array":
    """Estimate per-probe output gradients from per-probe weight gradients.

    For rank-1 per-probe gradients:
        dL/dW_i = s_i @ a_i^T
    where a_i is input activation and s_i is output gradient.

    We recover:
        s_i = (dL/dW_i @ a_i) / ||a_i||^2

    Args:
        per_probe_weight_gradients:
            Either [N, out_dim, in_dim] or flattened [N, out_dim * in_dim].
        input_activations: [N, in_dim] probe activations.
        weight_shape: Optional (out_dim, in_dim) if gradients are flattened.
        backend: Backend for tensor operations.

    Returns:
        Estimated output gradients S with shape [N, out_dim].
    """
    b = backend or get_default_backend()

    acts = b.array(input_activations)
    grads = b.array(per_probe_weight_gradients)

    if len(b.shape(acts)) != 2:
        raise ValueError(
            f"input_activations must be 2D [N, in_dim], got shape {b.shape(acts)}",
        )

    n_probes = int(acts.shape[0])
    in_dim = int(acts.shape[1])

    grad_shape = b.shape(grads)
    if len(grad_shape) == 2:
        if int(grad_shape[0]) != n_probes:
            raise ValueError(
                "per_probe_weight_gradients and input_activations must have "
                f"matching N, got {grad_shape[0]} and {n_probes}",
            )
        flat_dim = int(grad_shape[1])
        if weight_shape is not None:
            out_dim, in_dim_from_shape = int(weight_shape[0]), int(weight_shape[1])
            if in_dim_from_shape != in_dim:
                raise ValueError(
                    f"weight_shape in_dim={in_dim_from_shape} does not match "
                    f"input activation dimension {in_dim}",
                )
            if out_dim * in_dim != flat_dim:
                raise ValueError(
                    f"weight_shape {(out_dim, in_dim)} incompatible with flattened "
                    f"gradient dimension {flat_dim}",
                )
        else:
            if in_dim <= 0 or flat_dim % in_dim != 0:
                raise ValueError(
                    "Cannot infer (out_dim, in_dim) from flattened gradients. "
                    f"Got flat_dim={flat_dim}, in_dim={in_dim}.",
                )
            out_dim = flat_dim // in_dim
        grads = b.reshape(grads, (n_probes, out_dim, in_dim))
        b.eval(grads)
    elif len(grad_shape) == 3:
        if int(grad_shape[0]) != n_probes:
            raise ValueError(
                "per_probe_weight_gradients and input_activations must have "
                f"matching N, got {grad_shape[0]} and {n_probes}",
            )
        out_dim = int(grad_shape[1])
        grad_in_dim = int(grad_shape[2])
        if grad_in_dim != in_dim:
            raise ValueError(
                f"Gradient in_dim {grad_in_dim} does not match activation in_dim {in_dim}",
            )
    else:
        raise ValueError(
            "per_probe_weight_gradients must be 2D [N, out*in] "
            f"or 3D [N, out, in], got shape {grad_shape}",
        )

    compute_dtype = precision_dtype(b, reference=acts)
    for arr in (grads,):
        if hasattr(arr, "dtype"):
            try:
                if b.finfo(arr.dtype).eps < b.finfo(compute_dtype).eps:
                    compute_dtype = arr.dtype
            except Exception:
                pass

    acts = b.astype(acts, compute_dtype)
    grads = b.astype(grads, compute_dtype)
    b.eval(acts, grads)

    # numerator_i = dW_i @ a_i = sum_j dW_i[:, j] * a_i[j]
    acts_expanded = b.reshape(acts, (n_probes, 1, in_dim))
    numerator = b.sum(grads * acts_expanded, axis=2)
    b.eval(numerator)

    # denominator_i = ||a_i||^2
    denominator = b.sum(acts * acts, axis=1, keepdims=True)
    b.eval(denominator)

    # Avoid division by near-zero activation norm; zero-norm activations
    # provide no directional information about s_i.
    eps_div = division_epsilon(b, acts)
    has_signal = denominator > eps_div
    safe_denominator = b.where(has_signal, denominator, b.ones_like(denominator))
    recovered = numerator / safe_denominator
    recovered = b.where(has_signal, recovered, b.zeros_like(recovered))
    b.eval(recovered)
    return recovered


def compute_kfac_diagnostic(
    input_activations: "Array",
    output_gradients: "Array",
    *,
    backend: "Backend | None" = None,
) -> KFACDiagnosticResult:
    """Compute K-FAC null-space diagnostics from activations and output gradients.

    Args:
        input_activations: Activation matrix A [N, in_dim].
        output_gradients: Output-gradient matrix S [N, out_dim].
        backend: Backend for tensor ops.

    Returns:
        KFACDiagnosticResult with raw rank and spectrum measurements.
    """
    b = backend or get_default_backend()

    A = b.array(input_activations)
    S = b.array(output_gradients)

    if len(b.shape(A)) != 2:
        raise ValueError(f"input_activations must be 2D [N, in_dim], got {b.shape(A)}")
    if len(b.shape(S)) != 2:
        raise ValueError(f"output_gradients must be 2D [N, out_dim], got {b.shape(S)}")

    n_probes = int(A.shape[0])
    if n_probes == 0:
        raise ValueError("Cannot compute K-FAC diagnostics with 0 probes.")

    if int(S.shape[0]) != n_probes:
        raise ValueError(
            "input_activations and output_gradients must have the same number of probes, "
            f"got {n_probes} and {int(S.shape[0])}",
        )

    in_dim = int(A.shape[1])
    out_dim = int(S.shape[1])
    if in_dim == 0 or out_dim == 0:
        raise ValueError(
            f"Non-empty dimensions required, got in_dim={in_dim}, out_dim={out_dim}",
        )

    compute_dtype = precision_dtype(b, reference=A)
    for arr in (S,):
        if hasattr(arr, "dtype"):
            try:
                if b.finfo(arr.dtype).eps < b.finfo(compute_dtype).eps:
                    compute_dtype = arr.dtype
            except Exception:
                pass
    A = b.astype(A, compute_dtype)
    S = b.astype(S, compute_dtype)
    b.eval(A, S)

    # K-FAC factors:
    #   A_cov = E[a a^T] ≈ A^T A / N
    #   S_cov = E[s s^T] ≈ S^T S / N
    A_cov = b.matmul(b.transpose(A), A) / float(n_probes)
    S_cov = b.matmul(b.transpose(S), S) / float(n_probes)
    b.eval(A_cov, S_cov)

    A_eigvals, A_eigvecs = _sorted_eigh_desc(A_cov, b)
    S_eigvals, S_eigvecs = _sorted_eigh_desc(S_cov, b)

    eps = machine_epsilon(b, A_cov)
    A_eigvals_pos = b.maximum(A_eigvals, eps)
    S_eigvals_pos = b.maximum(S_eigvals, eps)
    b.eval(A_eigvals_pos, S_eigvals_pos)

    # Activation-only rank/null from A spectrum.
    max_A_arr = b.max(A_eigvals_pos)
    b.eval(max_A_arr)
    max_A = float(b.to_scalar(max_A_arr))
    max_A_safe = max(max_A, eps)

    A_rank_scale = svd_rank_threshold(b, A_eigvals_pos, in_dim)
    activation_rank_threshold = max_A_safe * A_rank_scale
    A_rank_mask = A_eigvals_pos > activation_rank_threshold
    A_rank_count_arr = b.sum(b.astype(A_rank_mask, compute_dtype))
    b.eval(A_rank_count_arr)
    activation_rank = int(round(float(b.to_scalar(A_rank_count_arr))))
    activation_rank = max(0, min(activation_rank, in_dim))
    activation_null_rank_input = max(0, in_dim - activation_rank)
    activation_null_rank_weight = out_dim * activation_null_rank_input

    # Kronecker spectrum eigenvalues are pairwise products:
    #   lambda_{ij} = lambda_S_i * lambda_A_j
    S_col = b.reshape(S_eigvals_pos, (out_dim, 1))
    A_row = b.reshape(A_eigvals_pos, (1, in_dim))
    kron_eigvals = b.matmul(S_col, A_row)  # [out_dim, in_dim]
    b.eval(kron_eigvals)

    max_S_arr = b.max(S_eigvals_pos)
    max_kron_arr = b.max(kron_eigvals)
    b.eval(max_S_arr, max_kron_arr)
    max_S = float(b.to_scalar(max_S_arr))
    max_kron = float(b.to_scalar(max_kron_arr))
    max_kron_safe = max(max_kron, eps)

    kron_rank_scale = svd_rank_threshold(b, A_cov, max(in_dim, out_dim))
    kron_threshold = max_kron_safe * kron_rank_scale
    kron_null_mask = kron_eigvals <= kron_threshold
    kfac_null_count_arr = b.sum(b.astype(kron_null_mask, compute_dtype))
    b.eval(kron_null_mask, kfac_null_count_arr)
    kfac_null_rank = int(round(float(b.to_scalar(kfac_null_count_arr))))
    kfac_null_rank = max(0, min(kfac_null_rank, in_dim * out_dim))

    if activation_null_rank_weight == 0:
        kfac_gain_ratio = float("inf") if kfac_null_rank > 0 else 1.0
    else:
        kfac_gain_ratio = kfac_null_rank / float(activation_null_rank_weight)

    return KFACDiagnosticResult(
        n_probes=n_probes,
        in_dim=in_dim,
        out_dim=out_dim,
        activation_rank=activation_rank,
        activation_null_rank_input=activation_null_rank_input,
        activation_null_rank_weight=activation_null_rank_weight,
        kfac_null_rank=kfac_null_rank,
        kfac_gain_ratio=kfac_gain_ratio,
        activation_rank_threshold=activation_rank_threshold,
        kron_threshold=kron_threshold,
        max_activation_eigenvalue=max_A,
        max_output_gradient_eigenvalue=max_S,
        max_kron_eigenvalue=max_kron,
        activation_eigenvalues=A_eigvals_pos,
        activation_eigenvectors=A_eigvecs,
        output_gradient_eigenvalues=S_eigvals_pos,
        output_gradient_eigenvectors=S_eigvecs,
        kron_null_mask=kron_null_mask,
    )


def compute_kfac_diagnostic_from_weight_gradients(
    input_activations: "Array",
    per_probe_weight_gradients: "Array",
    *,
    weight_shape: tuple[int, int] | None = None,
    backend: "Backend | None" = None,
) -> KFACDiagnosticResult:
    """Compute K-FAC diagnostics from per-probe weight gradients.

    This reuses existing per-probe CE gradients dL/dW and reconstructs output
    gradients s_i per probe, then runs K-FAC spectrum diagnostics.

    Args:
        input_activations: [N, in_dim] per-probe activations.
        per_probe_weight_gradients: [N, out*in] or [N, out, in] gradients.
        weight_shape: Optional (out_dim, in_dim) for flattened gradients.
        backend: Backend for tensor ops.

    Returns:
        KFACDiagnosticResult.
    """
    b = backend or get_default_backend()
    output_gradients = estimate_output_gradients_from_weight_gradients(
        per_probe_weight_gradients=per_probe_weight_gradients,
        input_activations=input_activations,
        weight_shape=weight_shape,
        backend=b,
    )
    return compute_kfac_diagnostic(
        input_activations=input_activations,
        output_gradients=output_gradients,
        backend=b,
    )


def activation_null_mask_in_kron_basis(
    diagnostic: KFACDiagnosticResult,
    *,
    backend: "Backend | None" = None,
) -> "Array":
    """Return activation-only null mask in [out_dim, in_dim] Kronecker basis."""
    b = backend or get_default_backend()
    activation_null_1d = diagnostic.activation_eigenvalues <= diagnostic.activation_rank_threshold
    activation_null = b.broadcast_to(
        b.reshape(activation_null_1d, (1, diagnostic.in_dim)),
        (diagnostic.out_dim, diagnostic.in_dim),
    )
    b.eval(activation_null)
    return activation_null


def count_activation_null_subset_violations(
    diagnostic: KFACDiagnosticResult,
    *,
    backend: "Backend | None" = None,
) -> int:
    """Count violations of Null(K_cap) ⊆ Null(G_cap) under current thresholds."""
    b = backend or get_default_backend()
    compute_dtype = precision_dtype(b, reference=diagnostic.kron_null_mask)

    activation_null = activation_null_mask_in_kron_basis(diagnostic, backend=b)
    activation_float = b.astype(activation_null, compute_dtype)
    kfac_float = b.astype(diagnostic.kron_null_mask, compute_dtype)
    violation_mass = activation_float * (1.0 - kfac_float)
    violation_count_arr = b.sum(violation_mass)
    b.eval(violation_count_arr)
    return int(round(float(b.to_scalar(violation_count_arr))))


__all__ = [
    "KFACDiagnosticResult",
    "activation_null_mask_in_kron_basis",
    "compute_kfac_diagnostic",
    "compute_kfac_diagnostic_from_weight_gradients",
    "count_activation_null_subset_violations",
    "estimate_output_gradients_from_weight_gradients",
]
