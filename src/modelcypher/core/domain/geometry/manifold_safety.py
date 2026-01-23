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

"""Combined manifold safety analysis.

This module unifies two complementary views of the activation manifold:

1. VARIANCE (intrinsic dimension): Where does the data live?
   - High variance directions: DATA USES this space
   - Low variance directions: Data doesn't spread here

2. BOUNDARY (flood fill): Where does the model break?
   - Large radius: MODEL TOLERATES perturbations
   - Small radius: Model is sensitive here

For SAFE compression/transplant, you need directions that are BOTH:
- Low variance (data doesn't use)
- Large boundary radius (model tolerates)

The "safe subspace" is the intersection of:
- Variance null space (low variance directions)
- Directions with boundary radius above threshold

Key insight: Variance can be misleading alone. A direction might have low
variance (data doesn't spread there) but small boundary (model is sensitive).
Layer 4 in our tests showed this: 72% "available" by variance, but boundary
radius = 0.0 (model at stability edge everywhere).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.precision_utils import (
    _promote_precision_float32 as _promote_precision,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class ManifoldSafetyResult:
    """Combined safety analysis of a layer's manifold.

    Attributes:
        layer_idx: Layer index.
        hidden_dim: Hidden dimension.

        # Variance analysis
        variance_utilized_rank: Dimensions with high variance.
        variance_available_rank: Dimensions with low variance.
        variance_captured: Fraction of variance in utilized subspace.

        # Boundary analysis
        boundary_mean_radius: Mean boundary radius across probed directions.
        boundary_min_radius: Minimum boundary (tightest constraint).
        boundary_max_radius: Maximum boundary (loosest constraint).
        n_directions_probed: Number of directions probed.
        n_bounded: Number of directions where boundary was found.

        # Combined safety
        safe_compression_rank: Directions safe for compression (low var + large boundary).
        safe_compression_fraction: Fraction of dimensions safe for compression.
        safety_score: Overall safety score [0, 1]. High = safe to compress.
        is_bottleneck: True if this layer is a bottleneck (small boundary).
    """
    layer_idx: int
    hidden_dim: int

    # Variance
    variance_utilized_rank: int
    variance_available_rank: int
    variance_captured: float

    # Boundary
    boundary_mean_radius: float
    boundary_min_radius: float
    boundary_max_radius: float
    n_directions_probed: int
    n_bounded: int

    # Combined
    safe_compression_rank: int
    safe_compression_fraction: float
    safety_score: float
    is_bottleneck: bool


def analyze_layer_safety(
    activations: "Array",
    forward_fn: Callable[["Array"], "Array"],
    backend: "Backend",
    layer_idx: int = 0,
    n_directions: int = 50,
    max_radius: float = 5.0,
    min_safe_radius: float = 1.0,
) -> ManifoldSafetyResult:
    """Analyze the safety of a layer for compression/transplant.

    Combines variance-based intrinsic dimension with flood-fill boundary
    detection to find the truly safe subspace for manipulation.

    Args:
        activations: Sample activations [n_samples, hidden_dim].
        forward_fn: Function to forward activations through.
        backend: Backend for tensor operations.
        layer_idx: Layer index (for logging).
        n_directions: Number of directions to probe for boundary.
        max_radius: Maximum radius to probe.
        min_safe_radius: Minimum boundary radius considered safe.

    Returns:
        ManifoldSafetyResult with combined analysis.
    """
    from modelcypher.core.domain.geometry.orthogonal_probe_generator import (
        compute_variance_null_space,
    )
    from modelcypher.core.domain.geometry.manifold_boundary import (
        detect_manifold_boundary,
    )

    b = backend
    activations = _promote_precision(b.array(activations), b)
    b.eval(activations)

    shape = b.shape(activations)
    hidden_dim = int(shape[1])

    # Step 1: Variance analysis
    logger.info("Layer %d: Computing variance null space...", layer_idx)
    variance_result = compute_variance_null_space(activations, b)

    variance_utilized = variance_result.utilized_rank
    variance_available = variance_result.available_rank
    # Compute variance captured from eigenvalues
    eigenvalues = variance_result.eigenvalues
    total_var = float(b.to_scalar(b.sum(eigenvalues)))
    if total_var > 0 and variance_utilized > 0:
        utilized_var = float(b.to_scalar(b.sum(eigenvalues[:variance_utilized])))
        variance_captured = utilized_var / total_var
    else:
        variance_captured = 0.0

    # Step 2: Boundary analysis
    logger.info("Layer %d: Probing manifold boundary...", layer_idx)
    boundary_result = detect_manifold_boundary(
        activations=activations,
        forward_fn=forward_fn,
        backend=b,
        n_directions=n_directions,
        max_radius=max_radius,
    )

    boundary_mean = boundary_result.mean_radius
    boundary_min = boundary_result.min_radius
    boundary_max = boundary_result.max_radius
    n_bounded = boundary_result.n_bounded

    # Step 3: Combined safety analysis
    #
    # The key insight: For each direction, we need BOTH:
    # - Low variance (data doesn't use this direction)
    # - Large boundary (model tolerates perturbations)
    #
    # The "variance available" rank tells us how many directions have low variance.
    # The boundary radius tells us if the model is stable in those directions.
    #
    # If boundary_min < min_safe_radius, the model is at stability edge and
    # we can't safely compress ANY direction (even low-variance ones).

    is_bottleneck = boundary_min < min_safe_radius

    if is_bottleneck:
        # Model is at stability edge - no safe compression
        safe_compression_rank = 0
        safety_score = 0.0
    else:
        # Model is stable - low-variance directions are potentially safe
        # Scale by boundary radius: larger boundary = more safety margin
        # safety_score = (boundary_min / max_radius) * (variance_available / hidden_dim)

        # Safe rank = variance_available, scaled by how much boundary margin we have
        boundary_margin = min(1.0, boundary_min / min_safe_radius)

        # If boundary_min >= min_safe_radius, we have full margin
        # If boundary_min < min_safe_radius, we're in bottleneck (handled above)
        safe_compression_rank = variance_available

        # Safety score combines both factors
        # - variance_available / hidden_dim: what fraction of space is low-variance
        # - boundary_min / max_radius: how much stability margin we have
        variance_factor = variance_available / hidden_dim
        boundary_factor = min(1.0, boundary_min / max_radius)

        safety_score = variance_factor * boundary_factor

    safe_compression_fraction = safe_compression_rank / hidden_dim

    logger.info(
        "Layer %d: variance=%d/%d (%.1f%%), boundary=%.2f-%.2f, safe=%d (%.1f%%), bottleneck=%s",
        layer_idx,
        variance_utilized, hidden_dim, (variance_utilized / hidden_dim) * 100,
        boundary_min, boundary_max,
        safe_compression_rank, safe_compression_fraction * 100,
        is_bottleneck,
    )

    return ManifoldSafetyResult(
        layer_idx=layer_idx,
        hidden_dim=hidden_dim,
        variance_utilized_rank=variance_utilized,
        variance_available_rank=variance_available,
        variance_captured=variance_captured,
        boundary_mean_radius=boundary_mean,
        boundary_min_radius=boundary_min,
        boundary_max_radius=boundary_max,
        n_directions_probed=n_directions,
        n_bounded=n_bounded,
        safe_compression_rank=safe_compression_rank,
        safe_compression_fraction=safe_compression_fraction,
        safety_score=safety_score,
        is_bottleneck=is_bottleneck,
    )


def analyze_model_safety(
    model: Any,
    layer_activations: dict[int, "Array"],
    config: dict,
    backend: "Backend",
    n_directions: int = 30,
    max_radius: float = 5.0,
    min_safe_radius: float = 1.0,
    forward_mode: str = "mlp",
) -> dict[int, ManifoldSafetyResult]:
    """Analyze safety across all layers.

    Args:
        model: The model.
        layer_activations: Activations per layer.
        config: Model config.
        backend: Backend for tensor operations.
        n_directions: Directions to probe per layer.
        max_radius: Maximum radius to probe.
        min_safe_radius: Minimum safe boundary radius.
        forward_mode: "mlp" for local MLP sensitivity, "full_model" for
            cascade sensitivity through remaining layers + lm_head.

    Returns:
        Dict mapping layer_idx -> ManifoldSafetyResult.
    """
    from modelcypher.core.domain.geometry.manifold_boundary import (
        create_layer_forward_fn,
    )

    results = {}

    for layer_idx in sorted(layer_activations.keys()):
        activations = layer_activations[layer_idx]

        forward_fn = create_layer_forward_fn(model, layer_idx, config, mode=forward_mode)

        result = analyze_layer_safety(
            activations=activations,
            forward_fn=forward_fn,
            backend=backend,
            layer_idx=layer_idx,
            n_directions=n_directions,
            max_radius=max_radius,
            min_safe_radius=min_safe_radius,
        )

        results[layer_idx] = result

    return results


def compute_safe_transplant_directions(
    source_activations: "Array",
    target_activations: "Array",
    target_forward_fn: Callable[["Array"], "Array"],
    backend: "Backend",
    n_directions: int = 50,
    max_radius: float = 5.0,
    min_safe_radius: float = 1.0,
) -> tuple["Array", "Array", ManifoldSafetyResult]:
    """Compute directions safe for transplant from source to target.

    For transplant, we need directions where:
    1. Target has low variance (target data doesn't use)
    2. Target has large boundary (target model tolerates)
    3. Source has information to transfer (ideally high variance in source)

    The transplant delta (source_centroid - target_centroid) should be
    projected onto the intersection of these safe directions.

    Args:
        source_activations: Source model activations [n_samples, hidden_dim].
        target_activations: Target model activations [n_samples, hidden_dim].
        target_forward_fn: Forward function for target model.
        backend: Backend for tensor operations.
        n_directions: Directions to probe.
        max_radius: Maximum radius.
        min_safe_radius: Minimum safe radius.

    Returns:
        Tuple of:
        - safe_directions: [n_safe, hidden_dim] orthonormal safe directions
        - transplant_delta: [hidden_dim] delta projected onto safe subspace
        - safety_result: Full safety analysis for target layer
    """
    from modelcypher.core.domain.geometry.orthogonal_probe_generator import (
        compute_variance_null_space,
    )

    b = backend

    # Analyze target safety
    safety_result = analyze_layer_safety(
        activations=target_activations,
        forward_fn=target_forward_fn,
        backend=b,
        layer_idx=0,
        n_directions=n_directions,
        max_radius=max_radius,
        min_safe_radius=min_safe_radius,
    )

    # Get target variance null space (low variance = available for transplant)
    target_variance = compute_variance_null_space(target_activations, b)
    target_available_space = target_variance.available_basis  # [hidden_dim, available_rank]

    if safety_result.is_bottleneck:
        # Target is at stability edge - can't safely transplant anything
        logger.warning("Target layer is a bottleneck - transplant may be unsafe")
        # Return empty safe directions
        hidden_dim = safety_result.hidden_dim
        empty_directions = b.zeros((0, hidden_dim))
        b.eval(empty_directions)

        source_centroid = b.mean(source_activations, axis=0)
        target_centroid = b.mean(target_activations, axis=0)
        b.eval(source_centroid, target_centroid)
        zero_delta = b.zeros_like(source_centroid)
        b.eval(zero_delta)

        return empty_directions, zero_delta, safety_result

    # The safe directions are the target's available space
    # (low variance AND within stable boundary)
    safe_directions = b.transpose(target_available_space)  # [available_rank, hidden_dim]
    b.eval(safe_directions)

    # Compute transplant delta and project onto safe subspace
    source_centroid = b.mean(source_activations, axis=0)
    target_centroid = b.mean(target_activations, axis=0)
    b.eval(source_centroid, target_centroid)

    full_delta = source_centroid - target_centroid
    b.eval(full_delta)

    # Project delta onto safe subspace: V_avail @ V_avail.T @ delta
    # This keeps only the component in the safe directions
    projected_delta = target_available_space @ (b.transpose(target_available_space) @ full_delta)
    b.eval(projected_delta)

    return safe_directions, projected_delta, safety_result


__all__ = [
    "ManifoldSafetyResult",
    "analyze_layer_safety",
    "analyze_model_safety",
    "compute_safe_transplant_directions",
]
