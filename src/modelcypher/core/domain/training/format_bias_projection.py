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

This module contains pure numpy domain logic only — no framework imports.
The training adapter is responsible for converting to/from framework arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FormatBiasDecomposition:
    """Result of format bias decomposition."""

    mu_format: np.ndarray       # [d] float32 — format bias vector (unnormalized)
    v_format: np.ndarray        # [d] float32 — unit format bias direction
    alpha_crit: float           # ‖μ_invariant‖ / ‖μ_format‖
    norm_format: float          # ‖μ_format‖
    norm_invariant: float       # ‖μ_augmented‖ ≈ ‖μ_invariant‖
    norm_narrow: float          # ‖μ_narrow‖
    cos_narrow_aug: float       # cosine(μ_narrow, μ_augmented)
    format_fraction: float      # ‖μ_format‖² / ‖μ_narrow‖²


def compute_format_bias(
    mu_narrow: np.ndarray,
    mu_augmented: np.ndarray,
) -> FormatBiasDecomposition:
    """Derive the format bias vector from narrow and augmented mean gradients.

    Args:
        mu_narrow: [d] float32 — mean gradient over narrow-format samples
        mu_augmented: [d] float32 — mean gradient over augmented samples (≈ μ_invariant)

    Returns:
        FormatBiasDecomposition with bias vector, unit direction, and diagnostics.
    """
    mu_narrow_64 = mu_narrow.astype(np.float64)
    mu_aug_64 = mu_augmented.astype(np.float64)

    # Format bias: the difference
    mu_format_64 = mu_narrow_64 - mu_aug_64

    norm_format = float(np.linalg.norm(mu_format_64))
    norm_invariant = float(np.linalg.norm(mu_aug_64))
    norm_narrow = float(np.linalg.norm(mu_narrow_64))

    # Unit format direction
    if norm_format > 1e-20:
        v_format = (mu_format_64 / norm_format).astype(np.float32)
    else:
        v_format = np.zeros_like(mu_narrow)

    mu_format = mu_format_64.astype(np.float32)

    # Critical alpha: where injected bias equals signal strength
    alpha_crit = norm_invariant / max(norm_format, 1e-20)

    # Cosine between narrow and augmented mean gradients
    cos_narrow_aug = float(
        np.dot(mu_narrow_64, mu_aug_64)
        / max(norm_narrow * norm_invariant, 1e-20)
    )

    # Format fraction of narrow gradient
    format_fraction = norm_format**2 / max(norm_narrow**2, 1e-20)

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
    grad_flat: np.ndarray,
    v_format: np.ndarray,
) -> np.ndarray:
    """Project out the format bias direction from a flattened gradient vector.

    g_clean = g - (v_format · g) v_format

    Args:
        grad_flat: [d] float32 — flattened gradient vector
        v_format: [d] float32 — unit format bias direction

    Returns:
        [d] float32 — decontaminated gradient
    """
    coeff = np.dot(v_format.astype(np.float64), grad_flat.astype(np.float64))
    g_clean = grad_flat.astype(np.float64) - coeff * v_format.astype(np.float64)
    return g_clean.astype(np.float32)


__all__ = [
    "FormatBiasDecomposition",
    "compute_format_bias",
    "project_out_bias_direction",
]
