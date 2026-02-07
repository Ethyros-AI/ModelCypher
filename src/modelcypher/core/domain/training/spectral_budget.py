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

Tracks ||scale * B @ A||_spectral / sigma_k per layer. When the median
ratio exceeds a threshold (default 0.9), the adapter has consumed 90%
of its geometric budget and training should stop.

The SVD computation uses the Backend protocol. The median comparison
and exhaustion check are pure Python.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
    threshold: float = 0.9,
) -> tuple[bool, float]:
    """Check if spectral budget is exhausted based on median ratio.

    Pure Python — no framework dependencies.

    Args:
        ratios: List of per-layer budget ratios from compute_budget_ratios().
        threshold: Fraction of budget that triggers exhaustion (default 0.9).

    Returns:
        (is_exhausted, median_ratio). Returns (False, 0.0) for empty input.
    """
    if not ratios:
        return False, 0.0

    sorted_ratios = sorted(ratios)
    median_ratio = sorted_ratios[len(sorted_ratios) // 2]
    return median_ratio > threshold, median_ratio


__all__ = [
    "compute_budget_ratios",
    "is_budget_exhausted",
]
