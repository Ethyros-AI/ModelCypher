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

from ..data_models import LayerGeometry

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


def stage_analyze_geometry(
    layer_geom: LayerGeometry,
    src_acts: list["Array"] | None,
    tgt_acts: list["Array"] | None,
    backend: "Backend",
) -> None:
    """STAGE 2: Analyze geometric properties at this layer."""
    if not tgt_acts or len(tgt_acts) < 5:
        return

    # intrinsic_dimension - Two-NN method (Facco et al., 2017)
    try:
        from modelcypher.core.domain.geometry.intrinsic_dimension import (
            IntrinsicDimension,
            TwoNNConfiguration,
        )
        stacked = backend.stack(tgt_acts, axis=0)
        backend.eval(stacked)
        result = IntrinsicDimension.compute_two_nn(stacked, TwoNNConfiguration(), backend)
        layer_geom.intrinsic_dimension = result.intrinsic_dimension
        logger.debug(
            "Layer %d: intrinsic_dim=%.1f (usable=%d/%d)",
            layer_geom.layer_idx,
            layer_geom.intrinsic_dimension,
            result.usable_count,
            result.sample_count,
        )
    except Exception as e:
        logger.debug("intrinsic_dimension failed for layer %d: %s", layer_geom.layer_idx, e)

    # manifold_curvature - sectional curvature for geodesic interpolation
    try:
        from modelcypher.core.domain.geometry.manifold_curvature import (
            CurvatureConfig,
            SectionalCurvatureEstimator,
        )
        stacked = backend.stack(tgt_acts, axis=0)
        backend.eval(stacked)
        stacked_np = backend.to_numpy(stacked).tolist()
        estimator = SectionalCurvatureEstimator(CurvatureConfig())
        profile = estimator.estimate_curvature_profile(stacked_np, backend=backend)
        layer_geom.curvature = profile.global_mean
        logger.debug(
            "Layer %d: curvature=%.4f, sign=%s",
            layer_geom.layer_idx,
            layer_geom.curvature,
            profile.dominant_sign.value,
        )
    except Exception as e:
        logger.debug("manifold_curvature failed for layer %d: %s", layer_geom.layer_idx, e)

    # Ollivier-Ricci curvature - discrete Ricci for manifold health
    try:
        from modelcypher.core.domain.geometry.manifold_curvature import (
            OllivierRicciConfig,
            OllivierRicciCurvature,
        )
        stacked = backend.stack(tgt_acts, axis=0)
        backend.eval(stacked)

        # Use adaptive alpha for varying-density manifolds
        config = OllivierRicciConfig(
            adaptive_alpha=True,
            k_neighbors=min(10, len(tgt_acts) - 1),
        )
        estimator = OllivierRicciCurvature(config=config, backend=backend)
        result = estimator.compute(stacked, k_neighbors=config.k_neighbors)

        layer_geom.ollivier_ricci_mean = result.mean_edge_curvature
        layer_geom.ollivier_ricci_std = result.std_edge_curvature

        logger.debug(
            "Layer %d: Ollivier-Ricci=%.4f (std=%.4f)",
            layer_geom.layer_idx,
            layer_geom.ollivier_ricci_mean,
            layer_geom.ollivier_ricci_std,
        )
    except Exception as e:
        logger.debug("Ollivier-Ricci failed for layer %d: %s", layer_geom.layer_idx, e)

    # gromov_wasserstein - distance between source and target representations
    # A.5: Store both distance AND coupling matrix for transport-guided merge
    if src_acts and len(src_acts) >= 5:
        try:
            from modelcypher.core.domain.geometry.gromov_wasserstein import (
                Config as GWConfig,
            )
            from modelcypher.core.domain.geometry.gromov_wasserstein import (
                GromovWassersteinDistance,
            )
            n = min(len(src_acts), len(tgt_acts), 50)  # Limit for speed
            src_stacked = backend.stack(src_acts[:n], axis=0)
            tgt_stacked = backend.stack(tgt_acts[:n], axis=0)
            backend.eval(src_stacked, tgt_stacked)

            gw = GromovWassersteinDistance(backend)
            result = gw.compute(src_stacked, tgt_stacked, GWConfig())

            # Store GW distance and coupling matrix for use in merge
            layer_geom.gw_distance = result.distance
            if hasattr(result, "coupling") and result.coupling is not None:
                layer_geom.gw_coupling = result.coupling

            logger.debug(
                "Layer %d: GW_distance=%.4f, converged=%s, coupling_stored=%s",
                layer_geom.layer_idx,
                result.distance,
                result.converged,
                layer_geom.gw_coupling is not None,
            )
        except Exception as e:
            logger.debug("gromov_wasserstein failed for layer %d: %s", layer_geom.layer_idx, e)
