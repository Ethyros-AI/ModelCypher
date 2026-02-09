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

# sqrt(eps) for float32: numerical significance threshold
_SQRT_EPS_F32 = math.sqrt(math.ldexp(1.0, -23))  # ~3.45e-4

# Dtype-derived default threshold: 1 - sqrt(eps_f32) (~0.9997 for float32)
DTYPE_THRESHOLD_F32 = 1.0 - _SQRT_EPS_F32

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


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

    Args:
        lora_products: List of (scale, lora_a, lora_b, sigma_k) tuples.
        backend: Backend for SVD computation.

    Returns:
        List of budget ratios (one per valid entry).
    """
    ratios: list[float] = []

    for scale, lora_a, lora_b, sigma_k in lora_products:
        if sigma_k <= 0:
            continue

        try:
            product = scale * backend.matmul(lora_a, lora_b)
            product_f32 = backend.astype(product, "float32")
            backend.eval(product_f32)

            S = backend.svd(product_f32, compute_uv=False)
            backend.eval(S)

            # S[0] is the largest singular value = spectral norm
            spectral_norm = float(backend.to_scalar(S[0]))
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


__all__ = [
    "DTYPE_THRESHOLD_F32",
    "compute_budget_ratios",
    "is_budget_exhausted",
]
