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

"""Format bias projection for gradient decontamination.

Derived from the gradient projection causal experiment (2026-02-19):

Theory:
  μ_narrow   = μ_invariant + μ_format     (signal + format bias)
  μ_augmented ≈ μ_invariant                (format cancels under group avg)
  μ_format   = μ_narrow - μ_augmented      (derivable from data)
  α_crit     = ‖μ_invariant‖ / ‖μ_format‖  (bias = signal threshold)

Intervention: project out v_format = μ_format / ‖μ_format‖ from each grad step.

  g_clean = g - (v_format · g) v_format

This module uses numpy for linear algebra (norm, dot product) — numpy is a
math library, not a framework (MLX/JAX/torch). The Backend protocol does not
expose arbitrary linalg operations on numpy arrays, and these computations
operate on adapter-produced numpy vectors, not framework tensors. The training
adapter is responsible for converting to/from framework arrays.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

# IEEE 754 float32 machine epsilon and smallest normal.
_EPS_F32 = math.ldexp(1.0, -23)  # ~1.19e-7
_TINY_F32 = math.ldexp(1.0, -126)  # ~1.18e-38


@dataclass
class FormatBiasDecomposition:
    """Result of format bias decomposition."""

    mu_format: "Array"       # [d] float32 — format bias vector (unnormalized)
    v_format: "Array"        # [d] float32 — unit format bias direction
    alpha_crit: float           # ‖μ_invariant‖ / ‖μ_format‖
    norm_format: float          # ‖μ_format‖
    norm_invariant: float       # ‖μ_augmented‖ ≈ ‖μ_invariant‖
    norm_narrow: float          # ‖μ_narrow‖
    cos_narrow_aug: float       # cosine(μ_narrow, μ_augmented)
    format_fraction: float      # ‖μ_format‖² / ‖μ_narrow‖²


def compute_format_bias(
    mu_narrow: "Array",
    mu_augmented: "Array",
    backend: "Backend | None" = None,
) -> FormatBiasDecomposition:
    """Derive the format bias vector from narrow and augmented mean gradients.

    Args:
        mu_narrow: [d] float32 — mean gradient over narrow-format samples
        mu_augmented: [d] float32 — mean gradient over augmented samples (≈ μ_invariant)

    Returns:
        FormatBiasDecomposition with bias vector, unit direction, and diagnostics.
    """
    b = backend or get_default_backend()

    # Format bias: the difference
    mu_format = mu_narrow - mu_augmented

    b.eval(mu_format)

    norm_format = float(b.to_scalar(b.norm(mu_format)))
    norm_invariant = float(b.to_scalar(b.norm(mu_augmented)))
    norm_narrow = float(b.to_scalar(b.norm(mu_narrow)))

    # Gradient-projection floor derivation (Higham 2002, §1.18 & Thm 3.1):
    # The relative error in a quotient a/b is rel_err(a) + rel_err(b).
    # For a Frobenius norm computed from d elements, rel_err ≈ sqrt(d) * u
    # (Higham 2002, Thm 3.1 on inner products). A component norm smaller
    # than eps * ||g_total|| is indistinguishable from roundoff in the
    # total gradient — the decomposition g = g_format + g_invariant has
    # no significant digits in that component. The absolute underflow
    # guard sqrt(d) * tiny catches the ||g_total|| ≈ 0 case.
    _dim = int(b.shape(mu_narrow)[0])
    _abs_floor = math.sqrt(_dim) * _TINY_F32  # underflow guard
    _rel_floor = _EPS_F32 * max(norm_narrow, _abs_floor)  # relative to total gradient

    # Unit format direction — zero if format component is below noise floor
    if norm_format > _rel_floor:
        v_format = mu_format / norm_format
    else:
        v_format = b.zeros_like(mu_narrow)

    b.eval(v_format)

    # Critical alpha: where injected bias equals signal strength
    alpha_crit = norm_invariant / max(norm_format, _rel_floor)

    # Cosine between narrow and augmented mean gradients
    _cos_denom = norm_narrow * norm_invariant
    cos_narrow_aug = float(
        b.to_scalar(b.dot(mu_narrow, mu_augmented))
        / max(_cos_denom, _abs_floor * _abs_floor)
    )

    # Format fraction of narrow gradient
    format_fraction = norm_format**2 / max(norm_narrow**2, _abs_floor * _abs_floor)

    return FormatBiasDecomposition(
        mu_format=mu_format,
        v_format=v_format,
        alpha_crit=alpha_crit,
        norm_format=norm_format,
        norm_invariant=norm_invariant,
        norm_narrow=norm_narrow,
        cos_narrow_aug=cos_narrow_aug,
        format_fraction=format_fraction,
    )


def project_out_bias_direction(
    grad_flat: "Array",
    v_format: "Array",
    backend: "Backend | None" = None,
) -> "Array":
    """Project out the format bias direction from a flattened gradient vector.

    g_clean = g - (v_format · g) v_format

    Args:
        grad_flat: [d] float32 — flattened gradient vector
        v_format: [d] float32 — unit format bias direction
        backend: Optional backend instance

    Returns:
        [d] float32 — decontaminated gradient
    """
    b = backend or get_default_backend()
    coeff = b.dot(v_format, grad_flat)
    g_clean = grad_flat - coeff * v_format
    b.eval(g_clean)
    return g_clean


__all__ = [
    "FormatBiasDecomposition",
    "compute_format_bias",
    "project_out_bias_direction",
]
