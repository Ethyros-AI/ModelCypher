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

"""Goldilocks Quality Metrics for Training Data Analysis.

Returns raw geometric measurements for training data characterization.
Callers interpret and combine these measurements for their specific use case.

Metrics computed:
1. CKA similarity to reference activations (Kornblith et al. 2019)
2. Activation barrier height (CKA-based interpolation divergence)
3. Fisher Information mean (empirical diagonal FIM)

References:
    - Kornblith et al. (2019): CKA for representation similarity
    - exp17_soar_curriculum: Original Goldilocks concept validation
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cka import compute_linear_cka_from_activations
from modelcypher.core.domain.geometry.fisher_information import (
    compute_empirical_fisher_diagonal,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class GoldilocksQualityResult:
    """Raw geometric measurements for training data characterization.

    All fields are raw measurements. No composite scores or classifications.
    Callers combine these signals based on their specific use case.
    """

    # Raw measurements
    cka_similarity: float       # CKA to reference (1.0 = identical)
    barrier_height: float       # Activation divergence barrier
    fisher_mean: float          # Mean Fisher Information


def compute_goldilocks_quality(
    activations: "Array",
    reference_activations: "Array",
    backend: "Backend | None" = None,
) -> GoldilocksQualityResult:
    """Compute raw geometric measurements for training data characterization.

    Returns CKA similarity, activation barrier, and Fisher Information —
    three independent geometric measurements. Callers decide how to
    interpret or combine them.

    Args:
        activations: Activations from processing problems [n_samples, d]
        reference_activations: Reference activations for comparison [n_ref, d]
        backend: Optional backend (uses default if not provided)

    Returns:
        GoldilocksQualityResult with raw measurements
    """
    b = backend or get_default_backend()

    # Compute Fisher Information
    fisher_result = compute_empirical_fisher_diagonal(activations, b)
    fisher_mean = fisher_result.mean_fim

    # Compute CKA similarity and barrier
    cka_similarity, barrier_height = _compute_cka_and_barrier(
        activations, reference_activations, b
    )

    return GoldilocksQualityResult(
        cka_similarity=cka_similarity,
        barrier_height=barrier_height,
        fisher_mean=fisher_mean,
    )


def _compute_cka_and_barrier(
    activations: "Array",
    reference_activations: "Array",
    backend: "Backend",
) -> tuple[float, float]:
    """Compute CKA similarity and activation barrier.

    Args:
        activations: Problem activations [n_samples, d]
        reference_activations: Reference activations [n_ref, d]
        backend: Backend for computation

    Returns:
        (cka_similarity, barrier_height) tuple
    """
    n_samples = int(activations.shape[0])
    n_ref = int(reference_activations.shape[0])

    if n_samples == n_ref and n_samples > 1:
        # Standard CKA path - same sample counts
        source_centered = reference_activations - backend.mean(
            reference_activations, axis=0, keepdims=True
        )
        target_centered = activations - backend.mean(activations, axis=0, keepdims=True)
        backend.eval(source_centered, target_centered)

        cka_similarity = compute_linear_cka_from_activations(
            source_centered, target_centered, backend
        )
        barrier_height = _compute_activation_barrier(
            source_centered, target_centered, backend
        )
    else:
        # Use cosine similarity as proxy for mismatched sample counts
        act_mean = backend.mean(activations, axis=0)
        ref_mean = backend.mean(reference_activations, axis=0)
        backend.eval(act_mean, ref_mean)

        dot = float(backend.sum(act_mean * ref_mean))
        norm_act = float(backend.sqrt(backend.sum(act_mean * act_mean)))
        norm_ref = float(backend.sqrt(backend.sum(ref_mean * ref_mean)))
        # Division guard: sqrt(eps_f32) for product of norms
        _eps_f32 = math.ldexp(1.0, -23)
        div_guard = math.sqrt(_eps_f32)
        cka_similarity = max(0.0, dot / max(norm_act * norm_ref, div_guard))
        barrier_height = 1.0 - cka_similarity

    return cka_similarity, barrier_height


def _compute_activation_barrier(
    source_centered: "Array",
    target_centered: "Array",
    backend: "Backend",
    n_steps: int = 11,
) -> float:
    """Compute CKA-based barrier along activation interpolation path.

    Args:
        source_centered: Centered source activations
        target_centered: Centered target activations
        backend: Backend for computation
        n_steps: Number of interpolation points

    Returns:
        Maximum CKA divergence (barrier height)
    """
    losses = []
    for i in range(n_steps):
        t = i / (n_steps - 1)
        interpolated = (1 - t) * source_centered + t * target_centered
        backend.eval(interpolated)

        cka = compute_linear_cka_from_activations(source_centered, interpolated, backend)
        losses.append(1.0 - cka)

    return max(losses)


def classify_problems_by_quality(
    quality_results: list[GoldilocksQualityResult],
    n_groups: int = 3,
) -> list[list[int]]:
    """Classify problems into quality groups by CKA similarity.

    Groups problems by CKA distance from reference, using equal-size
    partitioning (no heuristic thresholds). Top group = closest to
    reference, bottom group = farthest.

    Args:
        quality_results: List of quality results for each problem
        n_groups: Number of groups (default: 3)

    Returns:
        List of lists containing problem indices for each group,
        sorted by CKA similarity descending.
    """
    # Sort by CKA similarity descending (closest to reference first)
    indexed_scores = [(i, r.cka_similarity) for i, r in enumerate(quality_results)]
    sorted_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)

    # Split into equal-size groups
    group_size = len(sorted_scores) // n_groups
    groups = []

    for g in range(n_groups):
        start = g * group_size
        if g == n_groups - 1:
            # Last group gets remainder
            end = len(sorted_scores)
        else:
            end = start + group_size
        groups.append([idx for idx, _ in sorted_scores[start:end]])

    return groups
