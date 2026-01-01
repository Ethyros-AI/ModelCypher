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

from ..models import LayerGeometry
from ..infrastructure import select_anchor_indices_by_coverage

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


def stage_find_shared_structure(
    layer_geom: LayerGeometry,
    src_acts: list["Array"] | None,
    tgt_acts: list["Array"] | None,
    backend: "Backend",
    *,
    avoid_svd: bool,
) -> None:
    """STAGE 3: Find shared structure between source and target."""
    if not src_acts or not tgt_acts or len(src_acts) < 5 or len(tgt_acts) < 5:
        return

    n = min(len(src_acts), len(tgt_acts))

    # shared_subspace_projector - CCA-based
    if not avoid_svd:
        try:
            from modelcypher.core.domain.geometry.shared_subspace_projector import (
                AlignmentMethod,
                SharedSubspaceProjector,
            )
            from modelcypher.core.domain.geometry.shared_subspace_projector import (
                Config as SSPConfig,
            )

            # Convert activations to lists for CRM-style input
            src_stacked = backend.stack(src_acts[:n], axis=0)
            tgt_stacked = backend.stack(tgt_acts[:n], axis=0)
            backend.eval(src_stacked, tgt_stacked)

            # Use the CCA-based discovery
            # This identifies WHICH dimensions are shared
            src_list = backend.to_numpy(src_stacked).tolist()
            tgt_list = backend.to_numpy(tgt_stacked).tolist()

            result = SharedSubspaceProjector._discover_with_cca(
                source_activations=src_list,
                target_activations=tgt_list,
                weights=None,
                n=n,
                d_source=len(src_list[0]),
                d_target=len(tgt_list[0]),
                config=SSPConfig(alignment_method=AlignmentMethod.cca),
                backend=backend,
            )

            if result and result.is_valid:
                layer_geom.shared_dimension = result.shared_dimension
                layer_geom.alignment_strengths = result.alignment_strengths
                layer_geom.source_projection = backend.array(result.source_projection)
                layer_geom.target_projection = backend.array(result.target_projection)
                logger.debug(
                    "Layer %d: shared_dim=%d, top_corr=%.3f",
                    layer_geom.layer_idx,
                    layer_geom.shared_dimension,
                    layer_geom.alignment_strengths[0] if layer_geom.alignment_strengths else 0,
                )
        except Exception as e:
            logger.debug(
                "shared_subspace_projector failed for layer %d: %s",
                layer_geom.layer_idx,
                e,
            )

    # relative_representation - anchor-based alignment
    if not avoid_svd:
        try:
            from modelcypher.core.domain.geometry.relative_representation import (
                align_relative_representations,
                compute_relative_representation,
            )

            # Need anchor embeddings - use first N activations as anchors
            n_anchors = min(32, n)
            src_stacked = backend.stack(src_acts[:n], axis=0)
            tgt_stacked = backend.stack(tgt_acts[:n], axis=0)
            backend.eval(src_stacked, tgt_stacked)

            # Use coverage-selected target activations as anchors (balanced manifold coverage).
            anchor_indices = select_anchor_indices_by_coverage(
                tgt_stacked, n_anchors, backend
            )
            anchors = backend.take(tgt_stacked, backend.array(anchor_indices), axis=0)
            backend.eval(anchors)

            # Compute relative representations
            src_rel = compute_relative_representation(src_stacked, anchors)
            tgt_rel = compute_relative_representation(tgt_stacked, anchors)
            backend.eval(src_rel, tgt_rel)

            # Align in anchor space
            rotation, error = align_relative_representations(src_rel, tgt_rel)
            layer_geom.relative_rep_error = error
            logger.debug(
                "Layer %d: relative_rep_error=%.4f",
                layer_geom.layer_idx,
                error,
            )
        except Exception as e:
            logger.debug(
                "relative_representation failed for layer %d: %s",
                layer_geom.layer_idx,
                e,
            )
