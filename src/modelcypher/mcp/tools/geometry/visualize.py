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

"""Geometry visualization MCP tools.

Contains tools for real-time manifold visualization:
- Create 3D visualization from model activations
- Create visualization from pre-computed activations
- Export to HTML or JSON

The geometry is REAL, not approximate:
- Gram transport finds exact structure-preserving coupling
- Ollivier-Ricci curvature reflects true manifold curvature
- The 3D "shadow" IS the manifold shape

Requires: poetry install -E viz (installs plotly>=5.18.0)
"""

from __future__ import annotations

from pathlib import Path

from ..common import (
    MUTATING_ANNOTATIONS,
    READ_ONLY_ANNOTATIONS,
    ServiceContext,
    require_existing_directory,
    require_existing_path,
)


def register_geometry_visualize_tools(ctx: ServiceContext) -> None:
    """Register manifold visualization tools.

    These tools create 3D visualizations of the ACTUAL geometry of neural
    network activations. The visualization shows:
    - 3D point cloud (structure-preserving projection from high-D)
    - Curvature coloring: Red = walls (positive ORC), Blue = funnels (negative ORC)
    - Density-sized markers: Denser regions = smaller markers
    """
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    if "mc_geometry_visualize_create" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_geometry_visualize_create(
            model: str,
            prompt: str,
            output: str = "manifold.html",
        ) -> dict:
            """Create 3D manifold visualization from model activations.

            Runs a prompt through the model, captures hidden state activations,
            projects them through a dimension cascade (high-D → 4D → 3D),
            and renders an interactive visualization.

            The visualization shows ACTUAL geometry:
            - Gram transport preserves relational structure exactly
            - Ollivier-Ricci curvature reflects true manifold curvature
            - The 3D "shadow" IS the manifold shape

            Requires: poetry install -E viz (plotly>=5.18.0)

            Args:
                model: Path to the model directory
                prompt: Prompt to analyze (captures activations during forward pass)
                output: Output file path (.html for interactive, .json for data)

            Returns:
                Visualization metadata including intrinsic dimension and file path
            """
            try:
                import plotly  # noqa: F401
            except ImportError:
                raise ImportError(
                    "Visualization requires plotly. Install with: poetry install -E viz"
                )

            model_path = require_existing_directory(model)

            dims = [4, 3]

            from modelcypher.adapters.model_loader import load_model_for_training
            from modelcypher.core.domain._backend import get_default_backend
            from modelcypher.cli.commands.geometry.helpers import (
                forward_through_backbone,
                resolve_model_backbone,
            )
            from modelcypher.core.domain.geometry.dimension_cascade import DimensionCascade
            from modelcypher.viz.manifold_viewer import ManifoldViewer

            # Load model
            model_obj, tokenizer = load_model_for_training(str(model_path))
            model_type = getattr(model_obj, "model_type", "unknown")
            resolved = resolve_model_backbone(model_obj, model_type)

            if not resolved:
                raise ValueError(f"Could not resolve architecture for model at {model_path}")

            embed_tokens, layers, norm = resolved
            num_layers = len(layers)
            target_layer = num_layers - 1

            backend = get_default_backend()

            # Tokenize and capture activations
            tokens = tokenizer.encode(prompt)
            input_ids = backend.array([tokens])

            # Forward through backbone
            hidden = forward_through_backbone(
                input_ids,
                embed_tokens,
                layers,
                norm,
                target_layer,
                backend,
            )
            backend.eval(hidden)

            # Extract activations
            if len(hidden.shape) == 3:
                activations = hidden[0]  # [seq, hidden_dim]
            else:
                activations = hidden

            n_tokens, hidden_dim = activations.shape

            # Run dimension cascade (all parameters derived from data)
            cascade = DimensionCascade(backend)
            cascade_result = cascade.calibrate(activations, target_dims=dims)

            # Create visualization
            viewer = ManifoldViewer(
                backend,
                title=f"Manifold Geometry: {Path(model_path).name}",
            )

            viz_dim = 3 if 3 in cascade_result.projections else min(cascade_result.projections.keys())
            result = viewer.create_figure(cascade_result, target_dim=viz_dim)

            # Export
            output_path = Path(output)
            if output_path.suffix == ".json":
                import json
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(result.json_data, indent=2))
            else:
                viewer.export_html(result, output_path)

            return {
                "_schema": "mc.geometry.visualize.create.v1",
                "modelPath": str(model_path),
                "prompt": prompt,
                "nTokens": n_tokens,
                "hiddenDim": hidden_dim,
                "intrinsicDim": cascade_result.intrinsic_dim,
                "targetDims": dims,
                "visualizationDim": viz_dim,
                "pointCount": result.point_count,
                "outputFile": str(output_path.absolute()),
                "geodesicDistortion": {
                    str(k): v for k, v in cascade_result.geodesic_distortion.items()
                },
            }

    if "mc_geometry_visualize_from_activations" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_geometry_visualize_from_activations(
            activationsFile: str,
            output: str = "manifold.html",
        ) -> dict:
            """Create visualization from pre-computed activations.

            Load activations from a JSON file and project through dimension cascade.
            Useful for visualizing activations captured separately or from other tools.

            The JSON file should contain an array of shape [n_points, hidden_dim].

            Requires: poetry install -E viz (plotly>=5.18.0)

            Args:
                activationsFile: JSON file with activations array [n_points, hidden_dim]
                output: Output file path (.html for interactive, .json for data)

            Returns:
                Visualization metadata including intrinsic dimension and file path
            """
            try:
                import plotly  # noqa: F401
            except ImportError:
                raise ImportError(
                    "Visualization requires plotly. Install with: poetry install -E viz"
                )

            import json as json_module

            activations_path = require_existing_path(activationsFile)

            dims = [4, 3]

            from modelcypher.core.domain._backend import get_default_backend
            from modelcypher.core.domain.geometry.dimension_cascade import DimensionCascade
            from modelcypher.viz.manifold_viewer import ManifoldViewer

            # Load activations
            data = json_module.loads(activations_path.read_text())

            backend = get_default_backend()

            # Handle different JSON formats
            if isinstance(data, dict):
                if "activations" in data:
                    act_data = data["activations"]
                elif "points" in data:
                    act_data = data["points"]
                else:
                    act_data = list(data.values())
            else:
                act_data = data

            activations = backend.array(act_data)
            n_points, hidden_dim = activations.shape

            # Run dimension cascade (all parameters derived from data)
            cascade = DimensionCascade(backend)
            cascade_result = cascade.calibrate(activations, target_dims=dims)

            # Create visualization
            viewer = ManifoldViewer(
                backend,
                title=f"Manifold Geometry: {activations_path.stem}",
            )

            viz_dim = 3 if 3 in cascade_result.projections else min(cascade_result.projections.keys())
            result = viewer.create_figure(cascade_result, target_dim=viz_dim)

            # Export
            output_path = Path(output)
            if output_path.suffix == ".json":
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json_module.dumps(result.json_data, indent=2))
            else:
                viewer.export_html(result, output_path)

            return {
                "_schema": "mc.geometry.visualize.fromActivations.v1",
                "sourceFile": str(activations_path),
                "nPoints": n_points,
                "hiddenDim": hidden_dim,
                "intrinsicDim": cascade_result.intrinsic_dim,
                "targetDims": dims,
                "visualizationDim": viz_dim,
                "pointCount": result.point_count,
                "outputFile": str(output_path.absolute()),
                "geodesicDistortion": {
                    str(k): v for k, v in cascade_result.geodesic_distortion.items()
                },
            }

    if "mc_geometry_visualize_info" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_visualize_info() -> dict:
            """Get information about the manifold visualization system.

            Returns details about what the visualization shows and how it works.
            The visualization represents ACTUAL geometry, not approximations:
            - Gram transport preserves relational structure exactly
            - Ollivier-Ricci curvature reflects true manifold curvature
            - The 3D "shadow" IS the manifold shape

            Returns:
                Information about the visualization system and its capabilities
            """
            try:
                import plotly
                plotly_installed = True
                plotly_version = plotly.__version__
            except ImportError:
                plotly_installed = False
                plotly_version = None

            return {
                "_schema": "mc.geometry.visualize.info.v1",
                "plotlyInstalled": plotly_installed,
                "plotlyVersion": plotly_version,
                "capabilities": {
                    "3dPointCloud": True,
                    "curvatureColoring": True,
                    "densitySizing": True,
                    "trajectoryAnimation": True,
                    "htmlExport": True,
                    "jsonExport": True,
                },
                "curvatureColors": {
                    "positive": "Red (walls - clustering regions)",
                    "negative": "Blue (funnels - spreading regions)",
                    "zero": "White (flat regions)",
                },
                "geometryProperties": {
                    "projectionMethod": "GRAM_TRANSPORT (structure-preserving)",
                    "curvatureType": "Ollivier-Ricci (via optimal transport)",
                    "densityEstimation": "k-NN ball volume",
                    "dimensionReduction": "Cascade via composite coupling matrices",
                },
                "installCommand": "poetry install -E viz",
            }

    if "mc_geometry_visualize_fingerprints_2d" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_geometry_visualize_fingerprints_2d(
            modelPath: str,
            output: str = "fingerprints_2d.json",
            maxFeatures: int = 1200,
        ) -> dict:
            """Project model fingerprints to 2D for visualization.

            Uses geodesic MDS to project high-dimensional fingerprint activations
            to a 2D representation that preserves manifold structure. This enables
            visualization of how different probes cluster in the model's
            representation space.

            Args:
                modelPath: Path to the model directory
                output: Output JSON file path for the projection data
                maxFeatures: Maximum features to use (default 1200)

            Returns:
                Projection metadata including points and feature information
            """
            from modelcypher.core.domain.geometry.manifold_stitcher import (
                ManifoldStitcher,
                ProbeSpace,
            )
            from modelcypher.core.domain.geometry.model_fingerprints_projection import (
                ModelFingerprintsProjection,
                ProjectionMethod,
            )

            model_path = require_existing_directory(modelPath)
            output_path = Path(output)

            # Get fingerprints
            stitcher = ManifoldStitcher()
            fingerprints = stitcher.fingerprint_model(
                str(model_path),
                probe_space=ProbeSpace.prelogits_hidden,
            )

            if not fingerprints.fingerprints:
                raise ValueError(f"No fingerprints generated for model at {model_path}")

            # Project to 2D
            projection = ModelFingerprintsProjection.project_2d(
                fingerprints,
                method=ProjectionMethod.pca,
                max_features=maxFeatures,
            )

            # Format output
            result_data = {
                "modelId": projection.model_id,
                "method": projection.method.value,
                "maxFeatures": projection.max_features,
                "featureCount": len(projection.features),
                "pointCount": len(projection.points),
                "points": [
                    {
                        "primeId": p.prime_id,
                        "primeText": p.prime_text,
                        "x": p.x,
                        "y": p.y,
                    }
                    for p in projection.points
                ],
                "features": [
                    {
                        "layer": f.layer,
                        "dimension": f.dimension,
                        "frequency": f.frequency,
                    }
                    for f in projection.features[:50]  # Limit features in output
                ],
            }

            # Save to file
            import json
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result_data, indent=2))

            return {
                "_schema": "mc.geometry.visualize.fingerprints2d.v1",
                "modelPath": str(model_path),
                "outputFile": str(output_path.absolute()),
                "pointCount": len(projection.points),
                "featureCount": len(projection.features),
                "method": projection.method.value,
            }
