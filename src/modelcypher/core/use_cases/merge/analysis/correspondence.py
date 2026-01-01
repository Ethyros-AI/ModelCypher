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

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..models import MergeGeometry

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


def stage_layer_correspondence(
    geometry: MergeGeometry,
    source_activations: dict[int, list["Array"]] | None,
    target_activations: dict[int, list["Array"]] | None,
    backend: "Backend",
) -> None:
    """
    STAGE 1.5: Compute layer correspondence for cross-architecture models.

    Uses CrossArchitectureLayerMatcher with CKA-based dynamic programming
    to find optimal monotonic alignment between source and target layers.

    This stage is crucial for cross-architecture merges where:
    - Models have different layer counts (e.g., 12 vs 24 layers)
    - Models have different hidden dimensions (e.g., 768 vs 4096)

    The layer correspondence tells merge_weights() which source layer
    maps to which target layer.
    """
    if not source_activations or not target_activations:
        return

    src_layers = sorted(source_activations.keys())
    tgt_layers = sorted(target_activations.keys())

    # Detect cross-architecture
    is_cross_arch = len(src_layers) != len(tgt_layers)

    # Also check dimension mismatch from activations
    if src_layers and tgt_layers:
        src_first_acts = source_activations.get(src_layers[0], [])
        tgt_first_acts = target_activations.get(tgt_layers[0], [])
        if src_first_acts and tgt_first_acts:
            src_dim = src_first_acts[0].shape[-1] if src_first_acts[0].ndim > 0 else 0
            tgt_dim = tgt_first_acts[0].shape[-1] if tgt_first_acts[0].ndim > 0 else 0
            if src_dim != tgt_dim and src_dim > 0 and tgt_dim > 0:
                is_cross_arch = True

    geometry.is_cross_architecture = is_cross_arch

    if not is_cross_arch:
        # Same architecture - simple 1:1 mapping
        geometry.layer_correspondence = {i: i for i in range(len(tgt_layers))}
        geometry.alignment_quality = 1.0
        return

    # Use CrossArchitectureLayerMatcher for different layer counts
    try:
        # Build CRM-like structures from activations
        # We need to compute CKA between all layer pairs
        # First, compute a CKA matrix manually since we don't have full CRMs
        from modelcypher.core.domain.geometry.cka import HSICEstimator, compute_cka
        from modelcypher.core.domain.geometry.cross_architecture_layer_matcher import (
            CrossArchitectureLayerMatcher,
        )

        len(src_layers)
        n_tgt = len(tgt_layers)
        cka_matrix: list[list[float]] = []

        for src_idx in src_layers:
            row: list[float] = []
            src_acts = source_activations.get(src_idx, [])
            if not src_acts:
                row = [0.0] * n_tgt
                cka_matrix.append(row)
                continue

            for tgt_idx in tgt_layers:
                tgt_acts = target_activations.get(tgt_idx, [])
                if not tgt_acts:
                    row.append(0.0)
                    continue

                # Compute CKA between activations at these layers
                n = min(len(src_acts), len(tgt_acts))
                if n < 2:
                    row.append(0.0)
                    continue

                try:
                    src_stacked = backend.stack(src_acts[:n], axis=0)
                    tgt_stacked = backend.stack(tgt_acts[:n], axis=0)
                    backend.eval(src_stacked, tgt_stacked)

                    # Handle dimension mismatch with Gram-based CKA
                    if src_stacked.shape[1] != tgt_stacked.shape[1]:
                        # Use Gram matrices for cross-dimensional CKA
                        from modelcypher.core.domain.geometry.cka import (
                            HSICEstimator,
                            compute_cka_from_grams,
                        )
                        gram_src = backend.matmul(src_stacked, backend.transpose(src_stacked))
                        gram_tgt = backend.matmul(tgt_stacked, backend.transpose(tgt_stacked))
                        backend.eval(gram_src, gram_tgt)
                        cka_val = compute_cka_from_grams(
                            gram_src,
                            gram_tgt,
                            backend=backend,
                            estimator=HSICEstimator.AUTO,
                            feature_dim_a=int(src_stacked.shape[1]),
                            feature_dim_b=int(tgt_stacked.shape[1]),
                            feature_bias_correction=True,
                        )
                    else:
                        result = compute_cka(
                            src_stacked,
                            tgt_stacked,
                            backend=backend,
                            estimator=HSICEstimator.AUTO,
                            feature_bias_correction=True,
                        )
                        if result.is_valid:
                            cka_val = (
                                result.cka_corrected
                                if result.cka_corrected is not None
                                else result.cka
                            )
                        else:
                            cka_val = 0.0

                    row.append(float(cka_val))
                except Exception:
                    row.append(0.0)

            cka_matrix.append(row)

        # Use DP alignment directly since we have the CKA matrix
        dp_path, alignment_score = CrossArchitectureLayerMatcher._dynamic_programming_alignment(
            cka_matrix,
        )

        # Build correspondence dict from DP path
        correspondence: dict[int, int] = {}
        for src_pos, tgt_pos in dp_path:
            if src_pos < len(src_layers) and tgt_pos < len(tgt_layers):
                correspondence[src_layers[src_pos]] = tgt_layers[tgt_pos]

        if not correspondence:
            raise RuntimeError("Cross-architecture alignment produced no mappings.")

        geometry.layer_correspondence = correspondence
        geometry.alignment_quality = alignment_score / len(dp_path) if dp_path else 0.0

        # NOTE: We do NOT require CKA=1.0 at this stage.
        # This stage identifies WHICH layers correspond based on activation similarity.
        # Stage 2+ will ALIGN the layers to achieve CKA=1.0.
        # The actual alignment (GramAligner) happens during weight merging.

        logger.info(
            "STAGE 1.5: Cross-architecture layer correspondence: %d -> %d layers, quality=%.4f",
            len(src_layers),
            len(tgt_layers),
            geometry.alignment_quality,
        )

    except Exception as e:
        logger.error("Cross-architecture layer matching failed: %s", e)
        raise
