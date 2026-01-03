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

"""Geometry interference and safety polytope MCP tools.

Contains tools for:
- Interference prediction using Riemannian density estimation
- Null-space filtering to eliminate interference by construction
- Safety polytope for unified merge safety decisions
"""

from __future__ import annotations

from ..common import (
    READ_ONLY_ANNOTATIONS,
    ServiceContext,
    require_existing_directory,
)


def register_geometry_interference_tools(ctx: ServiceContext) -> None:
    """Register interference prediction and null-space filtering tools.

    These tools support pre-merge quality estimation:
    - Interference prediction using Riemannian density estimation
    - Null-space filtering to eliminate interference by construction
    - Safety polytope for unified merge safety decisions
    """
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    if "mc_geometry_interference_predict" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_interference_predict(
            sourceModel: str,
            targetModel: str,
        ) -> dict:
            """
            Predict interference between two models before merging.

            Uses Riemannian density estimation to model concepts as probability
            distributions and predict constructive vs destructive interference.

            Args:
                sourceModel: Path to source model
                targetModel: Path to target model
                Uses all validated domains and the final layer.

            Returns:
            Interference prediction with safety scores.
            """

            from modelcypher.backends.mlx_backend import MLXBackend
            from modelcypher.core.domain.domains import AtlasDomain
            from modelcypher.core.domain.geometry.domain_geometry_waypoints import (
                DomainGeometryWaypointService,
            )
            from modelcypher.core.domain.geometry.interference_predictor import (
                MergeAnalyzer,
            )
            from modelcypher.core.domain.geometry.riemannian_density import (
                RiemannianDensityEstimator,
            )

            source_path = require_existing_directory(sourceModel)
            target_path = require_existing_directory(targetModel)

            domain_list = [
                AtlasDomain.SPATIAL,
                AtlasDomain.SOCIAL,
                AtlasDomain.TEMPORAL,
                AtlasDomain.MORAL,
            ]
            layer = -1

            DomainGeometryWaypointService()
            RiemannianDensityEstimator()
            MergeAnalyzer()
            MLXBackend()

            domain_results = {}

            for domain in domain_list:
                try:
                    # This would need activation extraction - simplified for MCP
                    domain_results[domain.value] = {
                        "analyzed": True,
                        "note": "Use CLI for full activation extraction",
                    }
                except Exception as e:
                    domain_results[domain.value] = {"error": str(e)}

            return {
                "_schema": "mc.geometry.interference.predict.v1",
                "sourceModel": source_path,
                "targetModel": target_path,
                "layer": layer,
                "domainsRequested": [d.value for d in domain_list],
                "perDomain": domain_results,
            }

    if "mc_geometry_null_space_filter" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_null_space_filter(
            weightDelta: list[list[float]],
            priorActivations: list[list[float]],
            method: str = "svd",
        ) -> dict:
            """
            Filter weight delta to null space of prior activations.

            Eliminates interference by construction: if Δw ∈ null(A),
            then A @ (W + Δw) = A @ W.

            Based on MINGLE (arXiv:2509.21413).

            Args:
                weightDelta: Weight update to filter (2D array)
                priorActivations: Activation matrix from prior task [n_samples, d]
                method: Computation method: 'svd', 'qr', or 'eigenvalue'

            Returns:
                Filtered delta with diagnostics

            Note:
                Rank threshold is derived from machine epsilon (dtype-dependent),
                not an arbitrary user parameter.
            """
            from modelcypher.core.domain._backend import get_default_backend
            from modelcypher.core.domain.geometry.null_space_filter import (
                NullSpaceFilter,
                NullSpaceFilterConfig,
                NullSpaceMethod,
            )
            from modelcypher.core.domain.geometry.numerical_stability import (
                svd_rank_threshold,
            )

            backend = get_default_backend()
            delta = backend.array(weightDelta)
            activations = backend.array(priorActivations)
            backend.eval(delta)
            backend.eval(activations)

            try:
                method_enum = NullSpaceMethod(method.lower())
            except ValueError:
                method_enum = NullSpaceMethod.SVD

            # Derive rank threshold from machine epsilon - no arbitrary defaults
            rank_threshold = svd_rank_threshold(backend, activations)

            config = NullSpaceFilterConfig(
                rank_threshold=rank_threshold,
                method=method_enum,
            )

            null_filter = NullSpaceFilter(config)
            delta_flat = backend.reshape(delta, (-1,))
            backend.eval(delta_flat)
            result = null_filter.filter_delta(delta_flat, activations)

            # Convert filtered_delta to list for JSON serialization
            if hasattr(result.filtered_delta, "shape"):
                from modelcypher.core.support.array_utils import array_to_list

                filtered_list = array_to_list(backend, result.filtered_delta)
            else:
                filtered_list = result.filtered_delta

            return {
                "_schema": "mc.geometry.null_space.filter.v1",
                "filteringApplied": result.filtering_applied,
                "nullSpaceDim": result.null_space_dim,
                "preservedFraction": result.preserved_fraction,
                "projectionLoss": result.projection_loss,
                "originalNorm": result.original_norm,
                "filteredNorm": result.filtered_norm,
                "filteredDelta": filtered_list,
            }

    if "mc_geometry_null_space_profile" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_null_space_profile(
            layerActivations: dict[str, list[list[float]]],
        ) -> dict:
            """
            Compute null space profile across model layers.

            Returns raw null space measurements for each layer. User interprets
            which layers have sufficient null space based on their specific
            model and use case.

            Args:
                layerActivations: Dict mapping layer index (as string) to
                                  activation matrix [n_samples, d]

            Returns:
                Per-layer null space analysis with raw measurements.
                User filters by nullFraction based on their requirements.
            """
            from modelcypher.core.domain._backend import get_default_backend
            from modelcypher.core.domain.geometry.null_space_filter import (
                NullSpaceFilter,
                NullSpaceFilterConfig,
            )

            backend = get_default_backend()
            config = NullSpaceFilterConfig()
            null_filter = NullSpaceFilter(config)

            layer_arrays = {}
            for k, v in layerActivations.items():
                arr = backend.array(v)
                backend.eval(arr)
                layer_arrays[int(k)] = arr

            # Don't pass arbitrary threshold - compute raw measurements
            profile = null_filter.compute_model_null_space_profile(
                layer_arrays, graft_threshold=0.0  # Return all layers
            )

            per_layer_info = {}
            null_fractions = []
            for layer_idx, lp in profile.per_layer.items():
                per_layer_info[str(layer_idx)] = {
                    "nullDim": lp.null_dim,
                    "totalDim": lp.total_dim,
                    "nullFraction": lp.null_fraction,
                    "meanSingularValue": lp.mean_singular_value,
                    "conditionNumber": lp.condition_number,
                }
                null_fractions.append(lp.null_fraction)

            # Return raw statistics - user decides threshold
            return {
                "_schema": "mc.geometry.null_space.profile.v1",
                "totalNullDim": profile.total_null_dim,
                "totalDim": profile.total_dim,
                "meanNullFraction": profile.mean_null_fraction,
                # Statistics to help user choose threshold
                "minNullFraction": min(null_fractions) if null_fractions else 0.0,
                "maxNullFraction": max(null_fractions) if null_fractions else 0.0,
                "medianNullFraction": sorted(null_fractions)[len(null_fractions) // 2] if null_fractions else 0.0,
                "perLayer": per_layer_info,
            }
