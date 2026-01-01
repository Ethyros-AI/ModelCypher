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
from typing import TYPE_CHECKING, Any

from ..models import LayerGeometry

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


def stage_analyze_interference(
    layer_geom: LayerGeometry,
    src_weights: dict[str, "Array"],
    tgt_weights: dict[str, "Array"],
    tgt_acts: list["Array"] | None,
    backend: "Backend",
    *,
    cache: dict[str, Any],
    avoid_svd: bool,
) -> None:
    """STAGE 5: Analyze interference patterns."""
    # interference_predictor - determine required transforms
    try:
        from modelcypher.core.domain.geometry.interference_predictor import (
            MergeAnalysisConfig,
            TransformationType,
        )

        MergeAnalysisConfig()
        # Would analyze using RiemannianDensityEstimator
        # For now, set defaults based on alignment quality
        if layer_geom.alignment_quality < 0.5:
            layer_geom.transform_requirements.append(TransformationType.PROCRUSTES_ROTATION.value)
        if layer_geom.curvature > 0.1:
            layer_geom.transform_requirements.append(TransformationType.CURVATURE_CORRECTION.value)
    except Exception as e:
        logger.debug("interference_predictor failed for layer %d: %s", layer_geom.layer_idx, e)

    # WUDI interference - data-free subspace overlap without SVD
    try:
        from modelcypher.core.domain.geometry.interference_predictor import (
            TransformationType,
        )
        from modelcypher.core.domain.geometry.wudi_interference import (
            compute_wudi_interference,
            group_task_vectors_by_shape,
        )

        cache_key = f"wudi:{layer_geom.layer_idx}:{len(src_weights)}:{len(tgt_weights)}"
        cached = cache.get(cache_key)
        if cached is None:
            groups = group_task_vectors_by_shape(src_weights, tgt_weights, backend=backend)
            if groups:
                cached = compute_wudi_interference(groups, backend=backend)
            else:
                cached = None
            cache[cache_key] = cached

        if cached is not None:
            layer_geom.wudi_loss = cached.mean_loss
            layer_geom.wudi_mean_overlap = cached.mean_overlap
            layer_geom.wudi_max_overlap = cached.max_overlap
            layer_geom.interference_score = max(
                layer_geom.interference_score,
                cached.normalized_loss,
            )
            if cached.mean_loss > 0.0:
                layer_geom.transform_requirements.append(
                    TransformationType.ALPHA_SCALING.value
                )
            logger.debug(
                "Layer %d: WUDI loss=%.6f overlap=%.4f max=%.4f",
                layer_geom.layer_idx,
                cached.mean_loss,
                cached.mean_overlap,
                cached.max_overlap,
            )
    except Exception as e:
        logger.debug("WUDI interference failed for layer %d: %s", layer_geom.layer_idx, e)

    # spectral_analysis - condition number etc
    if not avoid_svd:
        try:
            from modelcypher.core.domain.geometry.spectral_analysis import (
                SpectralConfig,
                compute_spectral_metrics,
            )

            # Find a representative weight matrix for this layer
            for key in tgt_weights:
                tgt_w = tgt_weights[key]
                if key in src_weights and tgt_w.ndim == 2:
                    src_w = src_weights[key]
                    if src_w.shape == tgt_w.shape:
                        result = compute_spectral_metrics(
                            src_w, tgt_w, SpectralConfig(), backend=backend
                        )
                        layer_geom.spectral_condition = result.condition_ratio
                        break
        except Exception as e:
            logger.debug(
                "spectral_analysis failed for layer %d: %s",
                layer_geom.layer_idx,
                e,
            )

    # null_space_filter - compute null space for this layer
    if tgt_acts and len(tgt_acts) >= 5 and not avoid_svd:
        try:
            from modelcypher.core.domain.geometry.null_space_filter import NullSpaceFilter

            stacked = backend.stack(tgt_acts, axis=0)
            backend.eval(stacked)
            # All params derived from spectral properties - no configuration
            nsf = NullSpaceFilter(backend=backend)
            projection = nsf.compute_null_space_projection(stacked)
            layer_geom.null_space_dim = projection.null_dim
            layer_geom.null_space_projection = projection.projection_matrix
            backend.eval(layer_geom.null_space_projection)
        except Exception as e:
            logger.debug("null_space_filter failed for layer %d: %s", layer_geom.layer_idx, e)
