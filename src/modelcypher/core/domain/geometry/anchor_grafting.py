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
from modelcypher.core.domain.geometry.knowledge_density import (
    compute_density_weights,
    compute_knn_point_cloud_density,
)
from modelcypher.core.domain.geometry.relative_representation import (
    align_relative_representations,
    compute_relative_representation,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)

__all__ = [
    "AnchorGraftingResult",
    "compute_anchor_grafting_delta",
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
