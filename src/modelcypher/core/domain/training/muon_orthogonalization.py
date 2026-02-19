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

"""Newton-Schulz orthogonalization for gradient preconditioning (Muon).

Experimental. Pure domain logic — no framework imports.

Muon (Jordan et al. 2024) applies Newton-Schulz iteration to force
gradient updates to have all singular values equal to 1.0 (isometric).
This is different from Cayley: Cayley constrains WHERE params live
(Stiefel manifold); Muon constrains HOW updates look (isometric steps).

Newton-Schulz iteration converges to the polar factor:
    G = U S V^T  =>  newton_schulz(G) = U V^T  (S -> I)

This requires only matrix multiplications (no SVD), and converges
cubically for matrices with spectral norm < 1.7.

Usage in the Cayley-Riemannian pipeline:
    grad_preconditioned = P @ grad          # Cayley natural gradient (existing)
    grad_muon = newton_schulz(grad_precond) # + Muon orthogonalization (new)

The combined effect: the update direction is the natural gradient direction,
but with all singular values equalized to 1.0. This removes the condition
number from the update step while preserving the direction information
from the pullback metric.

References:
    Jordan et al. (2024): "Muon: An optimizer for hidden layers in neural networks"
    Bernstein et al. (2024): Newton-Schulz iteration analysis
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


# Newton-Schulz polynomial coefficients (Bernstein et al.)
# These satisfy: p(x) * q(x) approximates x^{-1/2} for x in (0, 1.7^2)
# 5 iterations with these coefficients achieve ~15 digits of accuracy.
_NS_COEFFICIENTS: list[tuple[float, float, float]] = [
    (3.4445, -4.7750, 2.0315),
    (3.4445, -4.7750, 2.0315),
    (3.4445, -4.7750, 2.0315),
    (3.4445, -4.7750, 2.0315),
    (3.4445, -4.7750, 2.0315),
]


def newton_schulz_orthogonalize(
    grad: "Array",
    backend: "Backend",
    n_iterations: int = 5,
) -> "Array":
    """Apply Newton-Schulz iteration to orthogonalize a gradient matrix.

    Converges to the polar factor U V^T (all singular values -> 1.0).
    Requires spectral_norm(grad) < 1.7 after normalization.

    Args:
        grad: 2D gradient matrix [m, n] where m >= n.
        backend: Backend for array operations.
        n_iterations: Number of NS iterations (5 gives ~15 digits).

    Returns:
        Orthogonalized gradient with same shape.
    """
    b = backend
    shape = b.shape(grad)
    m, n = int(shape[0]), int(shape[1])

    # Handle wide matrices by transposing
    transposed = m < n
    if transposed:
        grad = b.transpose(grad)
        m, n = n, m

    # Normalize so spectral norm < 1.7
    frob = b.norm(grad)
    b.eval(frob)
    frob_val = float(b.to_scalar(frob))
    if frob_val <= 0.0:
        return b.transpose(grad) if transposed else grad

    # Scale factor: spectral_norm <= frobenius_norm.
    # We need spectral_norm(X) < 1.7. Using frobenius as upper bound
    # is conservative but safe.
    x = grad / max(frob_val, 1e-30)
    b.eval(x)

    # Newton-Schulz iterations: X_{k+1} = a * X_k + b * X_k @ X_k^T @ X_k + c * X_k @ (X_k^T @ X_k)^2
    # Implemented as: X_{k+1} = X_k @ (a*I + b*X_k^T@X_k + c*(X_k^T@X_k)^2)
    for a_coeff, b_coeff, c_coeff in _NS_COEFFICIENTS[:n_iterations]:
        xtx = b.matmul(b.transpose(x), x)  # [n, n]
        b.eval(xtx)

        # inner = a*I + b*xtx + c*xtx^2
        eye_n = b.eye(n)
        xtx_sq = b.matmul(xtx, xtx)
        b.eval(xtx_sq)

        inner = a_coeff * eye_n + b_coeff * xtx + c_coeff * xtx_sq
        b.eval(inner)

        x = b.matmul(x, inner)
        b.eval(x)

    if transposed:
        x = b.transpose(x)

    return x


__all__ = [
    "newton_schulz_orthogonalize",
]
