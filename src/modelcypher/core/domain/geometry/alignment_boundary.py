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

"""Geometric alignment boundary for AI safety guardrails.

Defines mathematical boundaries in activation space that enforce alignment.
When activations leave the safe region, interventions can steer them back.

Mathematical Foundation:
    The alignment boundary is defined by two geometric constraints:
    1. Minimum projection onto refusal direction: r·x > threshold
       - Ensures the model maintains its "alignment" position on the manifold
    2. Maximum distance from safe centroid: ||x - centroid|| < radius
       - Ensures activations stay within the learned distribution

Both thresholds are data-derived (not arbitrary):
    - refusal_threshold: percentile of projections from training safe prompts
    - safe_radius: percentile of distances from training safe prompts

Usage:
    1. Compute boundary from training data (safe prompts)
    2. At inference time, check if activations are within boundary
    3. If outside, optionally steer back by adding refusal direction

References:
    - Arditi et al. (2024). "Refusal in Language Models Is Mediated by a Single Direction."
    - Zou et al. (2023). "Representation Engineering: A Top-Down Approach to AI Transparency."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    precision_dtype,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


class BoundaryViolationType(str, Enum):
    """Type of boundary violation."""

    NONE = "none"
    LOW_REFUSAL_PROJECTION = "low_refusal_projection"
    HIGH_DISTANCE = "high_distance"
    BOTH = "both"


@dataclass
class AlignmentBoundary:
    """Geometric boundary defining safe alignment region.

    Attributes:
        refusal_direction: The alignment vector (normalized) [d]
        safe_centroid: Mean of safe activations [d]
        refusal_threshold: Minimum projection onto refusal direction
        safe_radius: Maximum distance from safe centroid
        layer_index: Which transformer layer this boundary is for
    """

    refusal_direction: "Array"
    safe_centroid: "Array"
    refusal_threshold: float
    safe_radius: float
    layer_index: int


@dataclass
class BoundaryCheckResult:
    """Result of checking if activation is within boundary.

    Attributes:
        is_within_boundary: True if activation is safe
        violation_type: Type of violation if any
        refusal_projection: Projection onto refusal direction
        distance_to_centroid: Distance from safe centroid
        refusal_margin: How far above/below threshold (positive = safe)
        distance_margin: How far inside/outside radius (positive = safe)
    """

    is_within_boundary: bool
    violation_type: BoundaryViolationType
    refusal_projection: float
    distance_to_centroid: float
    refusal_margin: float
    distance_margin: float


def compute_alignment_boundary(
    refusal_direction: "Array",
    safe_activations: "Array",
    refusal_percentile: float = 5.0,
    distance_percentile: float = 95.0,
    layer_index: int = 0,
    backend: "Backend | None" = None,
) -> AlignmentBoundary:
    """Compute alignment boundary from safe training activations.

    Args:
        refusal_direction: The refusal/alignment vector (normalized) [d]
        safe_activations: Activations from safe prompts [n_samples, d]
        refusal_percentile: Percentile for refusal threshold (lower = stricter)
        distance_percentile: Percentile for distance threshold (higher = stricter)
        layer_index: Which transformer layer this boundary is for
        backend: Backend for tensor operations

    Returns:
        AlignmentBoundary with data-derived thresholds
    """
    b = backend or get_default_backend()

    # Ensure direction is normalized
    direction = refusal_direction
    dir_norm = b.sqrt(b.sum(direction * direction))
    eps = division_epsilon(b, direction)
    direction = direction / b.maximum(dir_norm, b.array([eps]))
    b.eval(direction)

    # Compute safe centroid (Fréchet mean would be better, but mean suffices here)
    safe_centroid = b.mean(safe_activations, axis=0)
    b.eval(safe_centroid)

    # Compute projections of safe activations onto refusal direction
    # projection_i = dot(activation_i, direction)
    projections = b.sum(safe_activations * direction, axis=1)  # [n_samples]
    b.eval(projections)

    # Compute distances from centroid
    centered = safe_activations - safe_centroid  # [n_samples, d]
    distances = b.sqrt(b.sum(centered * centered, axis=1))  # [n_samples]
    b.eval(distances)

    # Compute percentile-based thresholds
    n = int(b.shape(projections)[0])

    # Sort projections to find percentile
    sorted_proj = b.sort(projections, axis=0)
    b.eval(sorted_proj)
    proj_idx = int(refusal_percentile / 100.0 * n)
    proj_idx = max(0, min(proj_idx, n - 1))
    refusal_threshold = float(b.to_scalar(b.take(sorted_proj, b.array([proj_idx]), axis=0)))

    # Sort distances to find percentile
    sorted_dist = b.sort(distances, axis=0)
    b.eval(sorted_dist)
    dist_idx = int(distance_percentile / 100.0 * n)
    dist_idx = max(0, min(dist_idx, n - 1))
    safe_radius = float(b.to_scalar(b.take(sorted_dist, b.array([dist_idx]), axis=0)))

    logger.info(
        "Alignment boundary computed: refusal_threshold=%.4f (p%d), safe_radius=%.4f (p%d)",
        refusal_threshold,
        int(refusal_percentile),
        safe_radius,
        int(distance_percentile),
    )

    return AlignmentBoundary(
        refusal_direction=direction,
        safe_centroid=safe_centroid,
        refusal_threshold=refusal_threshold,
        safe_radius=safe_radius,
        layer_index=layer_index,
    )


def check_boundary(
    activation: "Array",
    boundary: AlignmentBoundary,
    backend: "Backend | None" = None,
) -> BoundaryCheckResult:
    """Check if activation is within the alignment boundary.

    Args:
        activation: Single activation vector [d]
        boundary: The alignment boundary to check against
        backend: Backend for tensor operations

    Returns:
        BoundaryCheckResult with violation details
    """
    b = backend or get_default_backend()

    # Compute projection onto refusal direction
    refusal_projection = float(
        b.to_scalar(b.sum(activation * boundary.refusal_direction))
    )

    # Compute distance from centroid
    diff = activation - boundary.safe_centroid
    distance_to_centroid = float(b.to_scalar(b.sqrt(b.sum(diff * diff))))

    # Check constraints
    refusal_ok = refusal_projection >= boundary.refusal_threshold
    distance_ok = distance_to_centroid <= boundary.safe_radius

    # Compute margins (positive = safe, negative = violation)
    refusal_margin = refusal_projection - boundary.refusal_threshold
    distance_margin = boundary.safe_radius - distance_to_centroid

    # Determine violation type
    if refusal_ok and distance_ok:
        violation_type = BoundaryViolationType.NONE
    elif not refusal_ok and not distance_ok:
        violation_type = BoundaryViolationType.BOTH
    elif not refusal_ok:
        violation_type = BoundaryViolationType.LOW_REFUSAL_PROJECTION
    else:
        violation_type = BoundaryViolationType.HIGH_DISTANCE

    return BoundaryCheckResult(
        is_within_boundary=refusal_ok and distance_ok,
        violation_type=violation_type,
        refusal_projection=refusal_projection,
        distance_to_centroid=distance_to_centroid,
        refusal_margin=refusal_margin,
        distance_margin=distance_margin,
    )


def steer_to_boundary(
    activation: "Array",
    boundary: AlignmentBoundary,
    strength: float = 1.0,
    backend: "Backend | None" = None,
) -> "Array":
    """Steer activation back within boundary if outside.

    If the activation has low projection onto the refusal direction,
    add the refusal direction to increase it. The strength parameter
    controls how aggressively to steer.

    Args:
        activation: Single activation vector [d]
        boundary: The alignment boundary
        strength: Steering strength multiplier (default 1.0)
        backend: Backend for tensor operations

    Returns:
        Steered activation vector [d]
    """
    b = backend or get_default_backend()

    check_result = check_boundary(activation, boundary, backend=b)

    if check_result.is_within_boundary:
        # Already safe, no steering needed
        return activation

    steered = activation

    # If low refusal projection, add refusal direction
    if check_result.violation_type in (
        BoundaryViolationType.LOW_REFUSAL_PROJECTION,
        BoundaryViolationType.BOTH,
    ):
        # Calculate how much to add to reach threshold
        deficit = boundary.refusal_threshold - check_result.refusal_projection
        # Add direction scaled by deficit and strength
        steered = steered + strength * deficit * boundary.refusal_direction
        b.eval(steered)

    # If too far from centroid, pull toward centroid
    if check_result.violation_type in (
        BoundaryViolationType.HIGH_DISTANCE,
        BoundaryViolationType.BOTH,
    ):
        # Calculate direction toward centroid
        to_centroid = boundary.safe_centroid - steered
        to_centroid_norm = b.sqrt(b.sum(to_centroid * to_centroid))
        eps = division_epsilon(b, to_centroid)
        to_centroid_unit = to_centroid / b.maximum(to_centroid_norm, b.array([eps]))
        b.eval(to_centroid_unit)

        # Calculate how much to move to get within radius
        excess = check_result.distance_to_centroid - boundary.safe_radius
        # Move toward centroid
        steered = steered + strength * excess * to_centroid_unit
        b.eval(steered)

    return steered


def batch_check_boundary(
    activations: "Array",
    boundary: AlignmentBoundary,
    backend: "Backend | None" = None,
) -> tuple[list[BoundaryCheckResult], float]:
    """Check multiple activations against boundary.

    Args:
        activations: Batch of activations [n_samples, d]
        boundary: The alignment boundary
        backend: Backend for tensor operations

    Returns:
        Tuple of (list of results, violation rate)
    """
    b = backend or get_default_backend()

    n = int(b.shape(activations)[0])
    results = []
    violations = 0

    for i in range(n):
        act_i = b.take(activations, b.array([i]), axis=0)
        act_i = b.reshape(act_i, (-1,))
        b.eval(act_i)

        result = check_boundary(act_i, boundary, backend=b)
        results.append(result)

        if not result.is_within_boundary:
            violations += 1

    violation_rate = violations / max(n, 1)

    return results, violation_rate


def optimize_boundary_thresholds(
    refusal_direction: "Array",
    harmless_activations: "Array",
    harmful_activations: "Array",
    layer_index: int = 0,
    target_fpr: float = 0.10,
    backend: "Backend | None" = None,
) -> AlignmentBoundary:
    """Optimize boundary thresholds directly from model geometry.

    Instead of using fixed percentiles, this function finds optimal thresholds
    by maximizing detection rate (recall) subject to a target false positive rate.

    The optimization:
    1. Compute projections and distances for both harmless and harmful
    2. For refusal threshold: find value that achieves target FPR on harmless
    3. For safe radius: find value that achieves target FPR on harmless
    4. Select the threshold combination that maximizes harmful detection

    Args:
        refusal_direction: The refusal/alignment vector (normalized) [d]
        harmless_activations: Activations from harmless prompts [n_harmless, d]
        harmful_activations: Activations from harmful prompts [n_harmful, d]
        layer_index: Which transformer layer this boundary is for
        target_fpr: Target false positive rate (default 10%)
        backend: Backend for tensor operations

    Returns:
        AlignmentBoundary with geometry-optimized thresholds
    """
    b = backend or get_default_backend()

    # Ensure direction is normalized
    direction = refusal_direction
    dir_norm = b.sqrt(b.sum(direction * direction))
    eps = division_epsilon(b, direction)
    direction = direction / b.maximum(dir_norm, b.array([eps]))
    b.eval(direction)

    # Compute safe centroid from harmless activations
    safe_centroid = b.mean(harmless_activations, axis=0)
    b.eval(safe_centroid)

    n_harmless = int(b.shape(harmless_activations)[0])
    n_harmful = int(b.shape(harmful_activations)[0])

    # Compute projections and distances for harmless
    harmless_projs = []
    harmless_dists = []
    for i in range(n_harmless):
        act = b.take(harmless_activations, b.array([i]), axis=0)
        act = b.reshape(act, (act.shape[1],))
        proj = float(b.to_scalar(b.sum(act * direction)))
        diff = act - safe_centroid
        dist = float(b.to_scalar(b.sqrt(b.sum(diff * diff))))
        harmless_projs.append(proj)
        harmless_dists.append(dist)

    # Compute projections and distances for harmful
    harmful_projs = []
    harmful_dists = []
    for i in range(n_harmful):
        act = b.take(harmful_activations, b.array([i]), axis=0)
        act = b.reshape(act, (act.shape[1],))
        proj = float(b.to_scalar(b.sum(act * direction)))
        diff = act - safe_centroid
        dist = float(b.to_scalar(b.sqrt(b.sum(diff * diff))))
        harmful_projs.append(proj)
        harmful_dists.append(dist)

    # Find optimal thresholds by grid search
    # Try percentiles from 1% to 20% for refusal (lower projection = violation)
    # Try percentiles from 80% to 99% for distance (higher distance = violation)
    best_f1 = -1.0
    best_refusal_thresh = 0.0
    best_safe_radius = 0.0

    sorted_harmless_projs = sorted(harmless_projs)
    sorted_harmless_dists = sorted(harmless_dists)

    for refusal_pct in range(1, 21, 2):  # 1%, 3%, 5%, ..., 19%
        for dist_pct in range(80, 100, 2):  # 80%, 82%, ..., 98%
            # Compute thresholds at these percentiles
            proj_idx = int(refusal_pct / 100.0 * n_harmless)
            proj_idx = max(0, min(proj_idx, n_harmless - 1))
            refusal_thresh = sorted_harmless_projs[proj_idx]

            dist_idx = int(dist_pct / 100.0 * n_harmless)
            dist_idx = max(0, min(dist_idx, n_harmless - 1))
            safe_radius = sorted_harmless_dists[dist_idx]

            # Count violations
            harmless_violations = sum(
                1 for p, d in zip(harmless_projs, harmless_dists)
                if p < refusal_thresh or d > safe_radius
            )
            harmful_violations = sum(
                1 for p, d in zip(harmful_projs, harmful_dists)
                if p < refusal_thresh or d > safe_radius
            )

            # Compute metrics
            fpr = harmless_violations / max(n_harmless, 1)
            tpr = harmful_violations / max(n_harmful, 1)

            # Skip if FPR is too high
            if fpr > target_fpr + 0.05:
                continue

            # Compute F1-like score (using TPR as recall, 1-FPR as "negative precision")
            # We want high TPR and low FPR
            if tpr + (1 - fpr) > 0:
                # Score that balances detection and false positives
                score = 2 * tpr * (1 - fpr) / (tpr + (1 - fpr))
            else:
                score = 0.0

            if score > best_f1:
                best_f1 = score
                best_refusal_thresh = refusal_thresh
                best_safe_radius = safe_radius

    # If no good threshold found, fall back to reasonable defaults
    if best_f1 < 0:
        proj_idx = int(0.05 * n_harmless)
        best_refusal_thresh = sorted_harmless_projs[max(0, min(proj_idx, n_harmless - 1))]
        dist_idx = int(0.95 * n_harmless)
        best_safe_radius = sorted_harmless_dists[max(0, min(dist_idx, n_harmless - 1))]

    logger.info(
        "Optimized boundary: refusal_threshold=%.4f, safe_radius=%.4f (score=%.4f)",
        best_refusal_thresh,
        best_safe_radius,
        best_f1,
    )

    return AlignmentBoundary(
        refusal_direction=direction,
        safe_centroid=safe_centroid,
        refusal_threshold=best_refusal_thresh,
        safe_radius=best_safe_radius,
        layer_index=layer_index,
    )


__all__ = [
    "AlignmentBoundary",
    "BoundaryCheckResult",
    "BoundaryViolationType",
    "batch_check_boundary",
    "check_boundary",
    "compute_alignment_boundary",
    "optimize_boundary_thresholds",
    "steer_to_boundary",
]
