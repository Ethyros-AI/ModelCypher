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

"""
Anchor-Relative Concept Grafting Pipeline.

Orchestrates the canonical pipeline for knowledge transfer:
    1. Map to anchor-relative space (dimension-agnostic)
    2. Align using Procrustes in anchor space
    3. Compute delta in anchor space
    4. Weight by density ratio (no thresholds)
    5. Decode to target activation space

Key property: Source coordinates NEVER touch target weights directly.
All transfer happens through the shared anchor space, and decoding uses
target's pseudo-inverse to stay in target's coordinate system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.anchor_decoder import (
    compute_anchor_decoder,
    decode_to_activation_space,
)
from modelcypher.core.domain.geometry.cross_grounding_transfer import (
    CrossGroundingSynthesizer,
)
from modelcypher.core.domain.geometry.knowledge_density import (
    compute_density_weights,
    compute_knn_point_cloud_density,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    find_magnitude_gap_threshold,
    sqrt_scalar,
    ulp_scalar,
)
from modelcypher.core.domain.geometry.relative_representation import (
    align_relative_representations,
    compute_relative_representation,
)
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)

__all__ = [
    "AnchorGraftingResult",
    "compute_anchor_grafting_delta",
    "compute_anchor_grafting_with_ghost_anchors",
]


@dataclass(frozen=True)
class AnchorGraftingResult:
    """Complete result of anchor-relative grafting for one layer.

    Attributes:
        delta_activations: Delta in target activation space [n, d_target]
        rotation_matrix: Procrustes rotation R [n_anchors, n_anchors]
        alignment_error: Procrustes alignment error in anchor space
        delta_relative: Delta in anchor space before decoding [n, n_anchors]
        density_weights: Per-sample transfer weights [n] in [0, 1]
        decoder_matrix: Decoder B = pinv(S_t) @ A_t [n_anchors, d_target]
        reconstruction_error: Decoder reconstruction error
        source_density_mean: Mean density of source samples
        target_density_mean: Mean density of target samples
        transfer_fraction: Mean of density weights (overall transfer strength)
    """

    delta_activations: "Array"
    rotation_matrix: "Array"
    alignment_error: float
    delta_relative: "Array"
    density_weights: "Array"
    decoder_matrix: "Array"
    reconstruction_error: float
    source_density_mean: float
    target_density_mean: float
    transfer_fraction: float


def compute_anchor_grafting_delta(
    source_activations: "Array",
    target_activations: "Array",
    source_anchors: "Array",
    target_anchors: "Array",
    backend: "Backend | None" = None,
) -> AnchorGraftingResult:
    """Compute delta_A via the canonical anchor-relative grafting pipeline.

    This is the main entry point for anchor-relative knowledge transfer.
    The pipeline ensures source coordinates never directly touch target weights:

    1. S_s = cos(A_s, C_s)  - Source relative representation
    2. S_t = cos(A_t, C_t)  - Target relative representation
    3. R = argmin ||S_s @ R - S_t||  - Procrustes alignment
    4. delta_S = S_s @ R - S_t  - Delta in anchor space
    5. densities = k-NN density comparison
    6. w = source_density / (source + target)  - Transfer weights
    7. B = pinv(S_t) @ A_t  - Decoder (uses target's pseudo-inverse)
    8. delta_A = (w * delta_S) @ B  - Decode to target activation space

    Args:
        source_activations: Source activations [n, d_source]
        target_activations: Target activations [n, d_target]
        source_anchors: Source anchor embeddings [n_anchors, d_source]
        target_anchors: Target anchor embeddings [n_anchors, d_target]
        backend: Compute backend

    Returns:
        AnchorGraftingResult with delta_A and diagnostics
    """
    b = backend or get_default_backend()

    n_samples = b.shape(source_activations)[0]
    d_source = b.shape(source_activations)[1]
    d_target = b.shape(target_activations)[1]
    n_anchors = b.shape(source_anchors)[0]

    logger.info(
        "ANCHOR GRAFTING: n=%d, d_source=%d, d_target=%d, n_anchors=%d",
        n_samples,
        d_source,
        d_target,
        n_anchors,
    )

    # Step 1-2: Compute relative representations
    # S_s = cos(A_s, C_s) -> [n, n_anchors]
    # S_t = cos(A_t, C_t) -> [n, n_anchors]
    S_s = compute_relative_representation(source_activations, source_anchors)
    S_t = compute_relative_representation(target_activations, target_anchors)
    b.eval(S_s, S_t)

    logger.debug(
        "ANCHOR GRAFTING: S_s shape=%s, S_t shape=%s",
        b.shape(S_s),
        b.shape(S_t),
    )

    # Step 3: Align in anchor space using Procrustes
    # R = argmin ||S_s @ R - S_t||
    R, alignment_error = align_relative_representations(S_s, S_t)
    b.eval(R)

    logger.info(
        "ANCHOR GRAFTING: Procrustes alignment_error=%.4f",
        alignment_error,
    )

    # Step 4: Compute delta in anchor space
    # delta_S = S_s @ R - S_t
    S_s_aligned = b.matmul(S_s, b.transpose(R))
    delta_S = S_s_aligned - S_t
    b.eval(delta_S)

    # Step 5: Compute k-NN densities in anchor space
    # We use the anchor-space representations for density comparison
    # This is dimension-agnostic and captures semantic similarity
    density_result = compute_knn_point_cloud_density(
        source_activations=S_s_aligned,  # Use aligned source
        target_activations=S_t,
        backend=b,
    )

    source_densities = density_result.source_densities
    target_densities = density_result.target_densities
    b.eval(source_densities, target_densities)

    # Step 6: Compute density weights
    # w = source / (source + target) -> [n] in [0, 1]
    density_weights = compute_density_weights(
        source_densities=source_densities,
        target_densities=target_densities,
        backend=b,
    )
    b.eval(density_weights)

    # Compute density statistics
    source_density_mean = float(b.to_scalar(b.mean(source_densities)))
    target_density_mean = float(b.to_scalar(b.mean(target_densities)))
    transfer_fraction = float(b.to_scalar(b.mean(density_weights)))

    logger.info(
        "ANCHOR GRAFTING: source_density=%.4f, target_density=%.4f, transfer_fraction=%.3f",
        source_density_mean,
        target_density_mean,
        transfer_fraction,
    )

    # Step 7: Compute decoder
    # B = pinv(S_t) @ A_t -> [n_anchors, d_target]
    decoder, reconstruction_error = compute_anchor_decoder(
        target_relative_rep=S_t,
        target_activations=target_activations,
        backend=b,
    )

    logger.info(
        "ANCHOR GRAFTING: decoder shape=%s, reconstruction_error=%.4f",
        b.shape(decoder),
        reconstruction_error,
    )

    # Step 8: Decode to target activation space
    # delta_A = (w * delta_S) @ B -> [n, d_target]
    delta_activations = decode_to_activation_space(
        delta_relative=delta_S,
        decoder=decoder,
        density_weights=density_weights,
        backend=b,
    )
    b.eval(delta_activations)

    logger.info(
        "ANCHOR GRAFTING: delta_A shape=%s, pipeline complete",
        b.shape(delta_activations),
    )

    return AnchorGraftingResult(
        delta_activations=delta_activations,
        rotation_matrix=R,
        alignment_error=alignment_error,
        delta_relative=delta_S,
        density_weights=density_weights,
        decoder_matrix=decoder,
        reconstruction_error=reconstruction_error,
        source_density_mean=source_density_mean,
        target_density_mean=target_density_mean,
        transfer_fraction=transfer_fraction,
    )


def compute_anchor_grafting_with_ghost_anchors(
    source_activations: "Array",
    target_activations: "Array",
    source_anchors: "Array",
    target_anchors: "Array",
    anchor_names: list[str] | None = None,
    backend: "Backend | None" = None,
) -> AnchorGraftingResult:
    """Anchor grafting with Ghost Anchor synthesis for novel concepts.

    This is the SINGULAR PIPELINE for perfect knowledge addition:

    1. Compute relative representations and Procrustes alignment (ONCE)
    2. Compute per-sample alignment residuals
    3. Identify samples beyond a data-derived residual gap (novel concepts)
    4. For novel samples, synthesize Ghost Anchors (coordinate-invariant positions)
    5. Replace target activations with Ghost Anchor positions for novel samples
    6. Complete the pipeline with corrected targets (reusing alignment)

    The key insight: Novel concepts (high residual) have no meaningful target
    representation. Ghost Anchors provide synthetic target positions that
    preserve the concept's Relational Stress (distance pattern to anchors).
    This is coordinate-invariant and survives rotation between models.

    Args:
        source_activations: Source activations [n, d_source]
        target_activations: Target activations [n, d_target]
        source_anchors: Source anchor embeddings [n_anchors, d_source]
        target_anchors: Target anchor embeddings [n_anchors, d_target]
        anchor_names: Optional names for anchors (for Ghost Anchor synthesis)
        backend: Compute backend

    Returns:
        AnchorGraftingResult with combined aligned + Ghost Anchor deltas
    """
    b = backend or get_default_backend()

    n_samples = int(b.shape(source_activations)[0])
    n_anchors = int(b.shape(source_anchors)[0])
    d_target = int(b.shape(target_activations)[1])

    # Generate anchor names if not provided
    if anchor_names is None:
        anchor_names = [f"anchor_{i}" for i in range(n_anchors)]

    # Step 1: Compute relative representations (ONCE - reused throughout)
    S_s = compute_relative_representation(source_activations, source_anchors)
    S_t_original = compute_relative_representation(target_activations, target_anchors)
    b.eval(S_s, S_t_original)

    # Procrustes alignment (ONCE - reused)
    R, alignment_error = align_relative_representations(S_s, S_t_original)
    b.eval(R)

    # Aligned source in anchor space
    S_s_aligned = b.matmul(S_s, b.transpose(R))
    b.eval(S_s_aligned)

    # Step 2: Compute per-sample alignment residuals (vectorized)
    residual_vectors = S_s_aligned - S_t_original
    residual_norms = geodesic_norms(residual_vectors, b)
    b.eval(residual_norms)

    residual_list = b.tolist(residual_norms)
    sorted_residuals = sorted(residual_list)
    median_residual = sorted_residuals[len(sorted_residuals) // 2]
    residual_threshold = find_magnitude_gap_threshold(sorted_residuals, backend=b)
    min_threshold = ulp_scalar(median_residual, b)
    residual_threshold = max(residual_threshold, median_residual + min_threshold)

    # Step 3: Identify novel samples using vectorized comparison
    novel_mask_arr = residual_norms > residual_threshold
    n_novel_arr = b.sum(b.astype(novel_mask_arr, "int32"))
    b.eval(n_novel_arr)
    n_novel = int(b.to_scalar(n_novel_arr))

    gap = residual_threshold - median_residual
    if gap <= min_threshold:
        logger.info(
            "GHOST ANCHORS: Skipping - no separable residual gap "
            "(median=%.4f, threshold=%.4f).",
            median_residual,
            residual_threshold,
        )
        n_novel = 0  # Skip Ghost Anchor synthesis
    else:
        novelty_ratio = n_novel / n_samples
        logger.info(
            "GHOST ANCHORS: %d/%d samples (%.1f%%) have residual > %.4f (novel concepts)",
            n_novel, n_samples, 100.0 * novelty_ratio, residual_threshold,
        )

    # Step 4: For novel samples, synthesize Ghost Anchors
    if n_novel > 0:
        # Build anchor dicts ONCE (vectorized extraction)
        source_anchors_list = b.tolist(source_anchors)
        target_anchors_list = b.tolist(target_anchors)
        source_anchor_dict = {
            name: b.array(source_anchors_list[i])
            for i, name in enumerate(anchor_names)
        }
        target_anchor_dict = {
            name: b.array(target_anchors_list[i])
            for i, name in enumerate(anchor_names)
        }

        synthesizer = CrossGroundingSynthesizer(b)

        # Precompute grounding rotation ONCE (shared across all Ghost Anchors)
        grounding_rotation = synthesizer._rotation_estimator.estimate_rotation(
            source_anchor_dict, target_anchor_dict
        )

        # Extract novel sample indices
        novel_mask_list = b.tolist(novel_mask_arr)
        novel_indices = [i for i, is_novel in enumerate(novel_mask_list) if is_novel]

        # Extract all source activations at once for novel samples
        source_acts_list = b.tolist(source_activations)

        # Synthesize Ghost Anchors for novel samples (reusing precomputed rotation)
        ghost_positions = {}
        for idx in novel_indices:
            source_act = b.array(source_acts_list[idx])
            ghost = synthesizer.synthesize_ghost_anchor(
                concept_id=f"sample_{idx}",
                source_activation=source_act,
                source_anchors=source_anchor_dict,
                target_anchors=target_anchor_dict,
                grounding_rotation=grounding_rotation,
            )
            ghost_positions[idx] = ghost.target_position

        # Build corrected target activations (vectorized where possible)
        target_acts_list = b.tolist(target_activations)
        corrected_list = []
        for i in range(n_samples):
            if i in ghost_positions:
                corrected_list.append(b.tolist(ghost_positions[i]))
            else:
                corrected_list.append(target_acts_list[i])

        corrected_target_activations = b.array(corrected_list)
        b.eval(corrected_target_activations)

        # Recompute S_t with corrected activations
        S_t = compute_relative_representation(corrected_target_activations, target_anchors)
        b.eval(S_t)

        logger.info(
            "GHOST ANCHORS: Synthesized %d Ghost Anchors",
            n_novel,
        )
    else:
        # No novel concepts - use original
        corrected_target_activations = target_activations
        S_t = S_t_original

    # Step 5: Complete pipeline REUSING alignment (no redundant computation)
    # delta_S = S_s @ R - S_t (with corrected S_t)
    delta_S = S_s_aligned - S_t
    b.eval(delta_S)

    # Density computation in anchor space
    density_result = compute_knn_point_cloud_density(
        source_activations=S_s_aligned,
        target_activations=S_t,
        backend=b,
    )
    source_densities = density_result.source_densities
    target_densities = density_result.target_densities
    b.eval(source_densities, target_densities)

    # Density weights
    density_weights = compute_density_weights(
        source_densities=source_densities,
        target_densities=target_densities,
        backend=b,
    )
    b.eval(density_weights)

    source_density_mean = float(b.to_scalar(b.mean(source_densities)))
    target_density_mean = float(b.to_scalar(b.mean(target_densities)))
    transfer_fraction = float(b.to_scalar(b.mean(density_weights)))

    logger.info(
        "ANCHOR GRAFTING: source_density=%.4f, target_density=%.4f, transfer_fraction=%.3f",
        source_density_mean,
        target_density_mean,
        transfer_fraction,
    )

    # Decoder using corrected target
    decoder, reconstruction_error = compute_anchor_decoder(
        target_relative_rep=S_t,
        target_activations=corrected_target_activations,
        backend=b,
    )

    # Decode to target activation space
    delta_activations = decode_to_activation_space(
        delta_relative=delta_S,
        decoder=decoder,
        density_weights=density_weights,
        backend=b,
    )
    b.eval(delta_activations)

    return AnchorGraftingResult(
        delta_activations=delta_activations,
        rotation_matrix=R,
        alignment_error=alignment_error,
        delta_relative=delta_S,
        density_weights=density_weights,
        decoder_matrix=decoder,
        reconstruction_error=reconstruction_error,
        source_density_mean=source_density_mean,
        target_density_mean=target_density_mean,
        transfer_fraction=transfer_fraction,
    )
