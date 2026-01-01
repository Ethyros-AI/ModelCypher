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

from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from ..models import LayerGeometry

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


def stage_compute_dimension_weights(
    layer_geom: LayerGeometry,
    src_acts: list["Array"] | None,
    tgt_acts: list["Array"] | None,
    src_weights: dict[str, "Array"],
    tgt_weights: dict[str, "Array"],
    backend: "Backend",
) -> None:
    """STAGE 6: Compute per-dimension weights for blending."""
    # dimension_blender - per-dimension alpha
    try:
        pass
        # Would compute dimension-specific alphas
    except Exception:
        pass

    # verb_noun_classifier - skill vs structure
    try:
        pass
        # Would classify dimensions as verb (skill) or noun (structure)
    except Exception:
        pass

    # fisher_blending - importance weights from activation variance
    # Higher variance = more important = trust that model more
    try:
        if src_acts and tgt_acts and len(src_acts) >= 5 and len(tgt_acts) >= 5:
            n = min(len(src_acts), len(tgt_acts))
            src_stacked = backend.stack(src_acts[:n], axis=0)
            tgt_stacked = backend.stack(tgt_acts[:n], axis=0)
            backend.eval(src_stacked, tgt_stacked)

            # Estimate Fisher from activation variance (inverse variance)
            # High variance = uncertain = low Fisher = trust other model
            src_var = backend.var(src_stacked, axis=0)
            tgt_var = backend.var(tgt_stacked, axis=0)
            backend.eval(src_var, tgt_var)

            # Fisher ~ 1/variance (stable for small variance)
            epsilon = division_epsilon(backend, src_var)
            src_fisher = 1.0 / (src_var + epsilon)
            tgt_fisher = 1.0 / (tgt_var + epsilon)
            backend.eval(src_fisher, tgt_fisher)

            # Combined weights: normalize and store
            total_fisher = src_fisher + tgt_fisher
            layer_geom.fisher_weights = tgt_fisher / (total_fisher + epsilon)
            layer_geom.source_fisher = src_fisher
            layer_geom.target_fisher = tgt_fisher
            layer_geom.fisher_method = "activation_variance"
            backend.eval(layer_geom.fisher_weights)

            logger.debug(
                "Layer %d: Fisher weights computed, mean=%.4f",
                layer_geom.layer_idx,
                float(backend.mean(layer_geom.fisher_weights).item()),
            )
    except Exception as e:
        logger.debug("fisher_blending failed for layer %d: %s", layer_geom.layer_idx, e)

    # dimension_blender - per-dimension domain-based alphas
    try:
        if src_acts and tgt_acts and len(src_acts) >= 5 and len(tgt_acts) >= 5:
            n = min(len(src_acts), len(tgt_acts))
            src_stacked = backend.stack(src_acts[:n], axis=0)
            tgt_stacked = backend.stack(tgt_acts[:n], axis=0)
            backend.eval(src_stacked, tgt_stacked)

            # Compute per-dimension correlation between source and target
            # High correlation = safe to blend evenly
            # Low correlation = trust target for stability
            dot = backend.sum(src_stacked * tgt_stacked, axis=0)
            norm_src = backend.sqrt(backend.sum(src_stacked * src_stacked, axis=0))
            norm_tgt = backend.sqrt(backend.sum(tgt_stacked * tgt_stacked, axis=0))
            eps = division_epsilon(backend, src_stacked)
            corr = dot / (norm_src * norm_tgt + eps)
            corr = backend.maximum(0.0, backend.minimum(1.0, corr))
            backend.eval(corr)

            layer_geom.dimension_alphas = corr
            backend.eval(layer_geom.dimension_alphas)

            logger.debug(
                "Layer %d: dimension correlations computed, mean=%.4f",
                layer_geom.layer_idx,
                float(backend.mean(corr).item()),
            )
    except Exception as e:
        logger.debug("dimension_blender failed for layer %d: %s", layer_geom.layer_idx, e)

    # Compute base alpha from alignment quality and shared dimension
    # Higher alignment quality = more source contribution
    # Higher shared dimension = safer to blend more evenly
    alignment_factor = layer_geom.alignment_quality
    shared_factor = (
        min(1.0, layer_geom.shared_dimension / 64.0)
        if layer_geom.shared_dimension > 0
        else 0.5
    )

    # Alpha = how much to trust source. Higher quality = trust source more
    layer_geom.base_alpha = 0.5 * (1.0 - alignment_factor) + 0.5 * (1.0 - shared_factor)
    layer_geom.base_alpha = max(0.0, min(1.0, layer_geom.base_alpha))
