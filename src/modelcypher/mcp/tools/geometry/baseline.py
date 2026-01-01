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

"""Geometry profile MCP tools.

Contains tools for model geometry profiling:
- Profile listing
- Profile extraction
- Profile comparison
"""

from __future__ import annotations

from ..common import (
    READ_ONLY_ANNOTATIONS,
    ServiceContext,
    require_existing_directory,
)


def register_geometry_baseline_tools(ctx: ServiceContext) -> None:
    """Register geometry profile tools for model geometry analysis."""
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    if "mc_geometry_baseline_list" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_baseline_list(family: str | None = None) -> dict:
            """
            List available model geometry profiles.

            Args:
                family: Optional model family filter (qwen, llama, mistral, etc.)

            Returns:
                List of available profiles with their metadata
            """
            from modelcypher.core.domain.geometry.model_profile import (
                ProfileRepository,
            )

            repo = ProfileRepository()
            if family:
                profiles = repo.get_profiles_for_family(family)
            else:
                profiles = repo.get_all_profiles()

            return {
                "_schema": "mc.geometry.profile.list.v1",
                "profiles": [
                    {
                        "modelFamily": p.model_family,
                        "modelPath": p.model_path,
                        "ollivierRicciMean": p.global_ollivier_ricci_mean,
                        "intrinsicDimensionMean": p.global_intrinsic_dimension_mean,
                        "computedAt": p.computed_at,
                        "layersAnalyzed": len(p.layer_profiles),
                    }
                    for p in profiles
                ],
            }

    if "mc_geometry_baseline_extract" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_baseline_extract(
            modelPath: str,
            layer: int = -1,
        ) -> dict:
            """
            Extract geometry profile from a model.

            Uses Ollivier-Ricci curvature and intrinsic dimension to create
            a geometry profile. k for k-NN is computed from the data (not guessed).

            Args:
                modelPath: Path to the model directory
                layer: Layer to analyze (-1 for sampled layers)

            Returns:
                Extracted profile with curvature and dimension metrics
            """
            from modelcypher.adapters.mlx_model_loader import MLXModelLoader
            from modelcypher.core.domain.geometry.model_profile import (
                ModelProfileExtractor,
                ProfileRepository,
            )

            model_path = require_existing_directory(modelPath)

            model_loader = MLXModelLoader()
            extractor = ModelProfileExtractor(model_loader=model_loader)
            profile = extractor.extract_profile(
                model_path=model_path,
                layers=[layer] if layer != -1 else None,
            )

            # Save profile
            repo = ProfileRepository()
            saved_path = repo.save_profile(profile)

            return {
                "_schema": "mc.geometry.profile.extract.v1",
                "modelFamily": profile.model_family,
                "modelPath": profile.model_path,
                "ollivierRicciMean": profile.global_ollivier_ricci_mean,
                "ollivierRicciStd": profile.global_ollivier_ricci_std,
                "intrinsicDimensionMean": profile.global_intrinsic_dimension_mean,
                "layersAnalyzed": len(profile.layer_profiles),
                "savedPath": str(saved_path),
            }

    if "mc_geometry_baseline_compare" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_baseline_compare(
            model1Path: str,
            model2Path: str,
            layer: int = -1,
        ) -> dict:
            """
            Compare geometry profiles of two models.

            Extracts profiles from both models and computes divergence metrics.
            Useful for pre-merge alignment assessment.

            Args:
                model1Path: Path to first model
                model2Path: Path to second model
                layer: Layer to analyze (-1 for sampled layers)

            Returns:
                Comparison results with divergence metrics
            """
            from modelcypher.adapters.mlx_model_loader import MLXModelLoader
            from modelcypher.core.domain.geometry.model_profile import (
                ModelProfileExtractor,
            )

            model1_path = require_existing_directory(model1Path)
            model2_path = require_existing_directory(model2Path)

            model_loader = MLXModelLoader()
            extractor = ModelProfileExtractor(model_loader=model_loader)

            profile1 = extractor.extract_profile(
                model_path=model1_path,
                layers=[layer] if layer != -1 else None,
            )
            profile2 = extractor.extract_profile(
                model_path=model2_path,
                layers=[layer] if layer != -1 else None,
            )

            # Compute divergence - raw measurements only
            ricci_divergence = abs(
                profile1.global_ollivier_ricci_mean - profile2.global_ollivier_ricci_mean
            )
            id_divergence = abs(
                profile1.global_intrinsic_dimension_mean
                - profile2.global_intrinsic_dimension_mean
            )

            return {
                "_schema": "mc.geometry.profile.compare.v1",
                "model1": {
                    "path": model1_path,
                    "family": profile1.model_family,
                    "ollivierRicciMean": profile1.global_ollivier_ricci_mean,
                    "intrinsicDimension": profile1.global_intrinsic_dimension_mean,
                },
                "model2": {
                    "path": model2_path,
                    "family": profile2.model_family,
                    "ollivierRicciMean": profile2.global_ollivier_ricci_mean,
                    "intrinsicDimension": profile2.global_intrinsic_dimension_mean,
                },
                "divergence": {
                    "ollivierRicci": ricci_divergence,
                    "intrinsicDimension": id_divergence,
                },
            }

    # Note: mc_geometry_baseline_validate removed
    # Validation is now just comparison against existing profiles
