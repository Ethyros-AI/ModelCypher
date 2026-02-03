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

"""Goldilocks Quality Scoring for Curriculum Learning.

Implements the "Goldilocks Principle" for training data selection:
- Moderate challenge (not too easy, not too hard) produces optimal learning
- Based on exp17 validation (r=-0.955 correlation with training effectiveness)

Key insight from SOAR paper (arXiv:2601.18778):
"Structural quality matters more than solution correctness for learning progress."

The Goldilocks quality score combines three components:
1. CKA similarity to reference (peaks at ~0.9, not 1.0)
2. Activation barrier (productive difficulty: 0.02-0.10 is ideal)
3. Fisher Information (low = model needs to learn, not just recall)

References:
    - exp17_soar_curriculum: Validated Goldilocks quality with r=-0.955
    - SOAR paper: "Teaching Models to Teach Themselves" (arXiv:2601.18778)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.fisher_information import (
    compute_empirical_fisher_diagonal,
)
from modelcypher.core.domain.geometry.cka import compute_linear_cka_from_activations

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class GoldilocksQualityResult:
    """Result of Goldilocks quality scoring for training data.

    Higher quality_score indicates better training data:
    - Moderate CKA similarity (~0.85-0.95, not 0.99+)
    - Productive difficulty (barrier 0.02-0.10)
    - Learning opportunity (low Fisher = model needs to learn)
    """

    # Combined quality score (0-1, higher = better for learning)
    quality_score: float

    # Component metrics
    cka_similarity: float       # CKA to reference (1.0 = identical)
    barrier_height: float       # Activation divergence barrier
    fisher_mean: float          # Mean Fisher Information

    # Component scores (0-1, higher = better)
    cka_goldilocks: float       # Bell curve around 0.9
    barrier_score: float        # Productive difficulty score
    fisher_learning: float      # Learning opportunity score

    # Classification
    quality_level: str          # "high", "medium", "low"


def compute_goldilocks_quality(
    activations: "Array",
    reference_activations: "Array",
    backend: "Backend | None" = None,
) -> GoldilocksQualityResult:
    """Compute Goldilocks quality score for training data.

    Based on exp17 validation (r=-0.955):
    - CKA ~0.9 is optimal (not 0.99+)
    - Barrier 0.02-0.10 is optimal (productive difficulty)
    - Low Fisher = more learning opportunity

    Args:
        activations: Activations from processing problems [n_samples, d]
        reference_activations: Reference activations for comparison [n_ref, d]
        backend: Optional backend (uses default if not provided)

    Returns:
        GoldilocksQualityResult with quality metrics
    """
    b = backend or get_default_backend()

    # Compute Fisher Information
    fisher_result = compute_empirical_fisher_diagonal(activations, b)
    fisher_mean = fisher_result.mean_fim

    # Compute CKA similarity and barrier
    cka_similarity, barrier_height = _compute_cka_and_barrier(
        activations, reference_activations, b
    )

    # Compute component scores
    cka_goldilocks = _compute_cka_goldilocks_score(cka_similarity)
    barrier_score = _compute_barrier_score(barrier_height)
    fisher_learning = _compute_fisher_learning_score(fisher_mean)

    # Combined quality score (validated weights from exp17)
    quality_score = (
        0.4 * cka_goldilocks +      # Moderate similarity (not too easy)
        0.3 * barrier_score +        # Productive difficulty
        0.3 * fisher_learning        # Learning opportunity
    )

    # Classify quality level
    if quality_score >= 0.7:
        quality_level = "high"
    elif quality_score >= 0.4:
        quality_level = "medium"
    else:
        quality_level = "low"

    return GoldilocksQualityResult(
        quality_score=quality_score,
        cka_similarity=cka_similarity,
        barrier_height=barrier_height,
        fisher_mean=fisher_mean,
        cka_goldilocks=cka_goldilocks,
        barrier_score=barrier_score,
        fisher_learning=fisher_learning,
        quality_level=quality_level,
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
        cka_similarity = max(0.0, dot / max(norm_act * norm_ref, 1e-10))
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


def _compute_cka_goldilocks_score(cka_similarity: float) -> float:
    """Compute Goldilocks CKA score - peaks at 0.9, penalizes extremes.

    Too high (>0.98): Nothing to learn, data too similar to known patterns
    Optimal (~0.9): Moderate challenge, good for learning
    Too low (<0.7): Confusing, data too different

    Args:
        cka_similarity: CKA similarity to reference (0-1)

    Returns:
        Goldilocks score (0-1, 1 = optimal)
    """
    # Bell curve centered at 0.9, drops off both sides
    score = 1.0 - abs(cka_similarity - 0.90) * 5.0
    return max(0.0, min(1.0, score))


def _compute_barrier_score(barrier_height: float) -> float:
    """Compute productive difficulty score from barrier height.

    Too low (<0.02): Trivial, nothing to learn
    Optimal (0.02-0.10): Productive difficulty
    Too high (>0.10): Confusing, counterproductive

    Args:
        barrier_height: Activation barrier height

    Returns:
        Barrier score (0-1, 1 = optimal difficulty)
    """
    if barrier_height < 0.02:
        return barrier_height / 0.02  # Ramps up to optimal
    elif barrier_height <= 0.10:
        return 1.0  # Optimal zone
    else:
        return max(0.0, 1.0 - (barrier_height - 0.10) * 5)  # Drops off after


def _compute_fisher_learning_score(fisher_mean: float) -> float:
    """Compute learning opportunity score from Fisher Information.

    Low Fisher = model doesn't already know this = good for learning
    High Fisher = model already has strong gradients = less to learn

    Args:
        fisher_mean: Mean Fisher Information

    Returns:
        Learning opportunity score (0-1, 1 = high learning potential)
    """
    return 1.0 - min(fisher_mean * 100, 1.0)


def classify_problems_by_quality(
    quality_results: list[GoldilocksQualityResult],
    n_groups: int = 3,
) -> list[list[int]]:
    """Classify problems into quality groups.

    Args:
        quality_results: List of quality results for each problem
        n_groups: Number of groups (default: 3 for high/medium/low)

    Returns:
        List of lists containing problem indices for each group
        [high_quality_indices, medium_quality_indices, low_quality_indices]
    """
    # Sort by quality score descending
    indexed_scores = [(i, r.quality_score) for i, r in enumerate(quality_results)]
    sorted_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)

    # Split into groups
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
