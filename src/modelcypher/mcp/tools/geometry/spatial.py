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

"""Geometry spatial MCP tools.

Contains tools for 3D spatial metrology:
- Spatial anchors listing
- Euclidean consistency analysis
- Gravity gradient analysis
- Volumetric density probing
- Full 3D world model analysis
- Model probing with Spatial Prime Atlas
- Cross-grounding feasibility and transfer
"""

from __future__ import annotations

from pathlib import Path

from ..common import (
    READ_ONLY_ANNOTATIONS,
    ServiceContext,
    require_existing_directory,
)


def register_geometry_spatial_tools(ctx: ServiceContext) -> None:
    """Register 3D spatial metrology tools.

    These tools probe how language models capture 3-dimensional spatial
    relationships in their internal representations. Tests whether the latent
    manifold encodes a geometrically consistent 3D world model.
    """
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    if "mc_geometry_spatial_anchors" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_spatial_anchors(
            axis: str | None = None,
            category: str | None = None,
        ) -> dict:
            """List the Spatial Prime Atlas anchors.

            Shows the 23 spatial anchors with their expected 3D coordinates (X, Y, Z)
            and categories. These anchors probe the model's 3D world model.

            Args:
                axis: Filter by axis (x_lateral, y_vertical, z_depth)
                category: Filter by category (vertical, lateral, depth, mass, furniture)

            Returns:
                List of spatial anchors with 3D coordinates
            """
            from modelcypher.core.domain.agents.spatial_atlas import (
                SpatialCategory,
                SpatialConceptInventory,
            )
            from modelcypher.core.domain.geometry.spatial_3d import (
                SpatialAxis,
                get_spatial_anchors_by_axis,
            )

            if axis:
                try:
                    axis_enum = SpatialAxis(axis)
                    anchors = get_spatial_anchors_by_axis(axis_enum)
                except ValueError:
                    raise ValueError(f"Invalid axis: {axis}. Use: x_lateral, y_vertical, z_depth")
            else:
                anchors = SpatialConceptInventory.all_concepts()

            if category:
                try:
                    category_enum = SpatialCategory(category)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid category: {category}. Use: vertical, lateral, depth, mass, furniture"
                    ) from exc
                anchors = [a for a in anchors if a.category == category_enum]

            return {
                "_schema": "mc.geometry.spatial.anchors.v1",
                "anchors": [
                    {
                        "name": a.name,
                        "prompt": a.prompt,
                        "expectedX": a.expected_x,
                        "expectedY": a.expected_y,
                        "expectedZ": a.expected_z,
                        "category": a.category.value,
                    }
                    for a in anchors
                ],
                "count": len(anchors),
                "categories": list(set(a.category for a in anchors)),
                "axisLegend": {
                    "X": "Lateral (Left=-1, Right=+1)",
                    "Y": "Vertical (Down=-1, Up=+1) - Gravity axis",
                    "Z": "Depth (Far=-1, Near=+1) - Perspective axis",
                },
            }

    if "mc_geometry_spatial_euclidean" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_spatial_euclidean(
            anchorActivations: dict[str, list[float]],
        ) -> dict:
            """Test Euclidean consistency of spatial anchor representations.

            Checks if the Pythagorean theorem holds in latent space:
            dist(A,C)² ≈ dist(A,B)² + dist(B,C)² for right-angle triplets.

            If consistency score > 0.6 and no triangle inequality violations,
            the model has internalized Euclidean 3D geometry.

            Args:
                anchorActivations: Dict mapping anchor_name to activation vector

            Returns:
                Euclidean consistency analysis with Pythagorean error
            """
            from modelcypher.backends.mlx_backend import MLXBackend
            from modelcypher.core.domain.geometry.spatial_3d import EuclideanConsistencyAnalyzer

            backend = MLXBackend()
            activations = {name: backend.array(vec) for name, vec in anchorActivations.items()}

            analyzer = EuclideanConsistencyAnalyzer(backend=backend)
            result = analyzer.analyze(activations)

            return {
                "_schema": "mc.geometry.spatial.euclidean.v1",
                **result.to_dict(),
            }

    if "mc_geometry_spatial_gravity" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_spatial_gravity(
            anchorActivations: dict[str, list[float]],
        ) -> dict:
            """Analyze gravity gradient in latent representations.

            Tests if the model has a 'gravity gradient' where heavy objects
            are pulled toward 'down' (Floor, Ground) in latent space.

            High mass correlation (>0.5) indicates the model understands
            physical mass as a geometric property, not just a word.

            Args:
                anchorActivations: Dict mapping anchor_name to activation vector

            Returns:
                Gravity gradient analysis with mass correlation
            """
            from modelcypher.backends.mlx_backend import MLXBackend
            from modelcypher.core.domain.geometry.spatial_3d import GravityGradientAnalyzer

            backend = MLXBackend()
            activations = {name: backend.array(vec) for name, vec in anchorActivations.items()}

            analyzer = GravityGradientAnalyzer(backend=backend)
            result = analyzer.analyze(activations)

            return {
                "_schema": "mc.geometry.spatial.gravity.v1",
                **result.to_dict(),
            }

    if "mc_geometry_spatial_density" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_spatial_density(
            anchorActivations: dict[str, list[float]],
        ) -> dict:
            """Probe volumetric density of spatial representations.

            Tests if physical objects have representational densities that
            match their real-world properties:
            - Heavy objects should have 'denser' representations
            - Distant objects should have attenuated density (inverse-square law)

            Args:
                anchorActivations: Dict mapping anchor_name to activation vector

            Returns:
                Volumetric density analysis with inverse-square compliance
            """
            from modelcypher.backends.mlx_backend import MLXBackend
            from modelcypher.core.domain.geometry.spatial_3d import VolumetricDensityProber

            backend = MLXBackend()
            activations = {name: backend.array(vec) for name, vec in anchorActivations.items()}

            prober = VolumetricDensityProber(backend=backend)
            result = prober.analyze(activations)

            return {
                "_schema": "mc.geometry.spatial.density.v1",
                **result.to_dict(),
            }

    if "mc_geometry_spatial_analyze" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_spatial_analyze(
            anchorActivations: dict[str, list[float]],
        ) -> dict:
            """Run full 3D world model analysis.

            Comprehensive analysis combining:
            - Euclidean consistency (Pythagorean theorem test)
            - Gravity gradient (mass -> down correlation)
            - Volumetric density (inverse-square law)

            All models encode physics geometrically. The world_model_score measures
            Visual-Spatial Grounding Density: how concentrated probability mass is
            along human-perceptual 3D axes. Lower scores indicate physics encoded
            along alternative geometric axes (linguistic, formula-based).

            Args:
                anchorActivations: Dict mapping anchor_name to activation vector

            Returns:
                Full 3D world model analysis
            """
            from modelcypher.backends.mlx_backend import MLXBackend
            from modelcypher.core.domain.geometry.spatial_3d import Spatial3DAnalyzer

            backend = MLXBackend()
            activations = {name: backend.array(vec) for name, vec in anchorActivations.items()}

            analyzer = Spatial3DAnalyzer(backend=backend)
            report = analyzer.full_analysis(activations)

            return {
                "_schema": "mc.geometry.spatial.full_analysis.v1",
                **report.to_dict(),
            }

    if "mc_geometry_spatial_probe_model" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_spatial_probe_model(
            modelPath: str,
            layer: int = -1,
            saveActivations: str | None = None,
        ) -> dict:
            """Probe a model with the Spatial Prime Atlas.

            Runs all 23 spatial anchor prompts through the model and extracts
            activations, then performs full 3D world model analysis.

            This is the end-to-end command to test if a model has a physics engine.

            Args:
                modelPath: Path to model directory
                layer: Layer to extract activations from (-1 = last)
                saveActivations: Optional path to save activations JSON

            Returns:
                Full 3D world model analysis
            """
            import json

            import mlx.core as mx

            from modelcypher.adapters.model_loader import load_model_for_training
            from modelcypher.backends.mlx_backend import MLXBackend
            from modelcypher.core.domain.agents.spatial_atlas import SpatialConceptInventory
            from modelcypher.core.domain.geometry.spatial_3d import Spatial3DAnalyzer

            model_path = require_existing_directory(modelPath)
            model, tokenizer = load_model_for_training(str(model_path))

            backend = MLXBackend()
            anchor_activations = {}

            for anchor in SpatialConceptInventory.all_concepts():
                tokens = tokenizer.encode(anchor.prompt)
                input_ids = mx.array([tokens])

                try:
                    hidden = model.model.embed_tokens(input_ids)
                    target_layer = layer if layer >= 0 else len(model.model.layers) - 1
                    for i, layer_module in enumerate(model.model.layers):
                        hidden = layer_module(hidden, mask=None)
                        if i == target_layer:
                            break

                    activation = mx.mean(hidden[0], axis=0)
                    mx.eval(activation)
                    anchor_activations[anchor.id] = activation
                except Exception:
                    pass  # Skip anchors that fail

            if not anchor_activations:
                return {
                    "_schema": "mc.geometry.spatial.probe_model.v1",
                    "modelPath": str(model_path),
                    "error": "No activations extracted",
                }

            # Save activations if requested
            if saveActivations:
                activations_json = {
                    name: backend.to_numpy(act).tolist() for name, act in anchor_activations.items()
                }
                Path(saveActivations).write_text(json.dumps(activations_json, indent=2))

            # Run full analysis
            analyzer = Spatial3DAnalyzer(backend=backend)
            report = analyzer.full_analysis(anchor_activations)

            return {
                "_schema": "mc.geometry.spatial.probe_model.v1",
                "modelPath": str(model_path),
                "anchorsProbed": len(anchor_activations),
                "layer": layer if layer >= 0 else "last",
                **report.to_dict(),
            }

    if "mc_geometry_spatial_cross_grounding_feasibility" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_spatial_cross_grounding_feasibility(
            sourceAnchors: dict[str, list[float]],
            targetAnchors: dict[str, list[float]],
        ) -> dict:
            """Estimate feasibility of cross-grounding knowledge transfer.

            Compares the coordinate systems of two models to determine how much
            'rotation' exists between their grounding axes. Lower rotation means
            easier transfer; higher rotation requires more sophisticated mapping.

            This is a pre-flight check before running a full transfer.

            Args:
                sourceAnchors: Dict of anchor_name -> activation_vector from source model
                targetAnchors: Dict of anchor_name -> activation_vector from target model

            Returns:
            Feasibility assessment with rotation estimate
            """
            from modelcypher.backends.mlx_backend import MLXBackend
            from modelcypher.core.domain.geometry.cross_grounding_transfer import (
                CrossGroundingTransferEngine,
            )

            backend = MLXBackend()
            source = {name: backend.array(vec) for name, vec in sourceAnchors.items()}
            target = {name: backend.array(vec) for name, vec in targetAnchors.items()}

            engine = CrossGroundingTransferEngine(backend=backend)
            feasibility = engine.estimate_transfer_feasibility(source, target)

            return {
                "_schema": "mc.geometry.spatial.cross_grounding_feasibility.v1",
                **feasibility,
            }

    if "mc_geometry_spatial_cross_grounding_transfer" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_spatial_cross_grounding_transfer(
            sourceAnchors: dict[str, list[float]],
            targetAnchors: dict[str, list[float]],
            concepts: dict[str, list[float]] | None = None,
            sourceGrounding: str = "unknown",
            targetGrounding: str = "unknown",
        ) -> dict:
            """Transfer knowledge from source to target model via cross-grounding.

            Uses Density Re-mapping to transfer concepts by preserving Relational Stress
            (distances to universal anchors) rather than absolute coordinates.

            This is the '3D Printer' for high-dimensional knowledge transfer.

            Args:
                sourceAnchors: Dict of anchor_name -> activation_vector from source model
                targetAnchors: Dict of anchor_name -> activation_vector from target model
                concepts: Optional dict of concept_id -> vector to transfer
                         If not provided, uses subset of source anchors as demo
                sourceGrounding: Source grounding type (high_visual, moderate, alternative)
                targetGrounding: Target grounding type

            Returns:
                Ghost Anchors with synthesized target positions
            """
            from modelcypher.backends.mlx_backend import MLXBackend
            from modelcypher.core.domain.geometry.cross_grounding_transfer import (
                CrossGroundingTransferEngine,
            )

            backend = MLXBackend()
            source = {name: backend.array(vec) for name, vec in sourceAnchors.items()}
            target = {name: backend.array(vec) for name, vec in targetAnchors.items()}

            # Process concepts
            if concepts:
                concept_arrays = {name: backend.array(vec) for name, vec in concepts.items()}
            else:
                # Demo with subset of source anchors
                demo_keys = ["chair", "floor", "ceiling", "left_hand", "background"]
                concept_arrays = {k: v for k, v in source.items() if k in demo_keys}
                if not concept_arrays:
                    concept_arrays = dict(list(source.items())[:5])

            engine = CrossGroundingTransferEngine(backend=backend)
            result = engine.transfer_concepts(
                concepts=concept_arrays,
                source_anchors=source,
                target_anchors=target,
                source_grounding=sourceGrounding,
                target_grounding=targetGrounding,
            )

            # Serialize Ghost Anchors
            ghost_anchors_serialized = [
                {
                    "conceptId": g.concept_id,
                    "sourcePosition": g.source_position.tolist(),
                    "targetPosition": g.target_position.tolist(),
                    "stressPreservation": g.stress_preservation,
                    "synthesisConfidence": g.synthesis_confidence,
                    "warning": g.warning,
                }
                for g in result.ghost_anchors
            ]

            return {
                "_schema": "mc.geometry.spatial.cross_grounding_transfer.v1",
                "sourceGrounding": result.source_model_grounding,
                "targetGrounding": result.target_model_grounding,
                "groundingRotation": {
                    "angleDegrees": result.grounding_rotation.angle_degrees,
                    "alignmentScore": result.grounding_rotation.alignment_score,
                    "confidence": result.grounding_rotation.confidence,
                },
                "ghostAnchors": ghost_anchors_serialized,
                "meanStressPreservation": result.mean_stress_preservation,
                "minStressPreservation": result.min_stress_preservation,
                "interpretabilityGap": result.interpretability_gap,
                # Note: recommendation and nextActions removed per No Vibes rule
            }
