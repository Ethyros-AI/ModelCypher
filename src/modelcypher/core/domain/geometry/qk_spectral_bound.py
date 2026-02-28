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

"""QK spectral bound: geometric replacement for logit softcapping.

Pure Python — zero framework dependencies.

Derivation
----------
For attention head h with RMSNorm-preceded input::

    logit[i,j] = (x_i @ W_Q_h^T)^T (x_j @ W_K_h^T) / sqrt(d_k)

By submultiplicativity (Horn & Johnson, Matrix Analysis 2nd ed., Thm 5.6.2)::

    |logit[i,j]| <= ||x_i|| * ||x_j|| * ||W_Q_h||_2 * ||W_K_h||_2 / sqrt(d_k)

RMSNorm normalizes to unit RMS, so ||x||_2 = sqrt(d_model) by construction
(Ba et al. 2016, arXiv:1607.06450). Therefore::

    |logit| <= d_model * sigma_Q * sigma_K / sqrt(d_k)

For softcap c (Riviere et al. 2024, arXiv:2408.00118, Eq. 2) to be guaranteed
inactive, the weight-space constraint is::

    sigma_Q * sigma_K <= c * sqrt(d_k) / d_model

This replaces tanh(logits/c)*c in the forward pass with a post-step projection
in weight space. The forward pass becomes purely linear in Q/K.

Composition bound
-----------------
When both Q and K receive LoRA adapters (or any perturbation), the per-matrix
Weyl bound (||ΔQ||_2 ≤ σ_k_Q) does NOT bound the composed QK product change.
By expansion::

    (σ_Q + δ_Q)(σ_K + δ_K) = σ_Q σ_K + σ_Q δ_K + σ_K δ_Q + δ_Q δ_K

The cross-terms (σ_Q δ_K + σ_K δ_Q + δ_Q δ_K) represent attention selectivity
change that is invisible to per-matrix monitoring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def softcap_equivalent_bound(soft_cap: float, d_k: int, d_model: int) -> float:
    """Derived spectral product bound equivalent to logit softcapping.

    Parameters
    ----------
    soft_cap : float
        Softcap value c from ``c * tanh(logits / c)``.
    d_k : int
        Per-head key dimension (hidden_size // num_attention_heads).
    d_model : int
        Model hidden dimension.

    Returns
    -------
    float
        Upper bound B on sigma_max(W_Q_h) * sigma_max(W_K_h).
    """
    return soft_cap * math.sqrt(d_k) / d_model


def qk_spectral_product(sigma_q: float, sigma_k: float) -> float:
    """Per-head QK spectral product."""
    return sigma_q * sigma_k


def qk_projection_scale(
    sigma_q: float, sigma_k: float, bound: float
) -> float:
    """Symmetric scaling factor to enforce spectral bound.

    If sigma_q * sigma_k <= bound, returns 1.0 (no projection needed).
    Otherwise returns alpha such that (alpha * sigma_q) * (alpha * sigma_k) = bound.

    Parameters
    ----------
    sigma_q, sigma_k : float
        Spectral norms of per-head Q and K weight matrices.
    bound : float
        Derived spectral product bound from :func:`softcap_equivalent_bound`.

    Returns
    -------
    float
        Scaling factor alpha to apply to both W_Q_h and W_K_h.
    """
    product = sigma_q * sigma_k
    if product <= bound:
        return 1.0
    return math.sqrt(bound / product)


def softcap_utilization(
    sigma_q: float, sigma_k: float, bound: float
) -> float:
    """How close this head is to the softcap boundary.

    Returns
    -------
    float
        0.0 = softcap never active, 1.0 = at boundary, >1.0 = softcap
        actively compressing logits.
    """
    if bound <= 0.0:
        return math.inf
    return (sigma_q * sigma_k) / bound


def max_logit_magnitude(
    sigma_q: float, sigma_k: float, d_k: int, d_model: int
) -> float:
    """Worst-case attention logit magnitude for this head.

    Assumes RMSNorm-normalized inputs (||x|| = sqrt(d_model)).
    """
    return d_model * sigma_q * sigma_k / math.sqrt(d_k)


def composition_change_bound(
    sigma_q: float, sigma_k: float, delta_q: float, delta_k: float
) -> float:
    """Worst-case cross-term change in QK product from perturbations.

    When Q is perturbed by delta_q and K by delta_k, the product changes by
    at most sigma_q * delta_k + sigma_k * delta_q + delta_q * delta_k.
    This is the quantity invisible to per-matrix Weyl monitoring.

    Parameters
    ----------
    sigma_q, sigma_k : float
        Base spectral norms of per-head Q and K.
    delta_q, delta_k : float
        Spectral norms of the perturbations (adapter, correction, etc.).

    Returns
    -------
    float
        Upper bound on |Δ(σ_Q × σ_K)|.
    """
    return sigma_q * delta_k + sigma_k * delta_q + delta_q * delta_k


def composition_relative_change(
    sigma_q: float, sigma_k: float, delta_q: float, delta_k: float
) -> float:
    """Relative change in QK product from perturbations.

    Returns the cross-term change normalized by the base product.
    A value of 0.01 means 1% change in attention logit magnitude.

    Returns 0.0 when base product is zero (no attention).
    """
    base = sigma_q * sigma_k
    if base <= 0.0:
        return 0.0
    return composition_change_bound(sigma_q, sigma_k, delta_q, delta_k) / base


def composition_significant(relative_change: float, eps: float) -> bool:
    """Whether the composition change exceeds detection threshold.

    Significance threshold is sqrt(eps), derived from IEEE 754 error
    propagation (Higham 2002, Ch. 3): the product of two floats with
    relative error eps has accumulated relative error O(sqrt(eps)).

    Parameters
    ----------
    relative_change : float
        From :func:`composition_relative_change`.
    eps : float
        Machine epsilon for the computation dtype.
    """
    return relative_change > math.sqrt(eps)


@dataclass(frozen=True)
class HeadSpectralBound:
    """Per-head QK spectral measurement and bound analysis."""

    layer_idx: int
    head_idx: int
    sigma_q: float
    sigma_k: float
    spectral_product: float
    bound: float
    utilization: float
    projection_scale: float
    max_logit: float
    softcap_active: bool


@dataclass(frozen=True)
class HeadCompositionChange:
    """Per-head QK composition change from a model modification."""

    layer_idx: int
    head_idx: int
    base_product: float
    modified_product: float
    absolute_change: float
    relative_change: float
    significant: bool
