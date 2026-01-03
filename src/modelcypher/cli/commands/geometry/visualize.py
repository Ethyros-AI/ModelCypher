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

"""Real-time manifold geometry visualization CLI commands.

Visualizes the ACTUAL geometry of neural network activations:
- 3D point cloud of activation positions (structure-preserving projection)
- Curvature coloring: Red = walls (positive ORC), Blue = funnels (negative ORC)
- Density-sized markers: Denser regions = smaller markers
- Animated token trajectory through concept space

The geometry you see is REAL:
- Gram transport finds exact structure-preserving coupling
- Ollivier-Ricci curvature reflects true manifold curvature
- The 3D "shadow" IS the manifold shape, not an approximation

Commands:
    mc geometry visualize create <model_path> <prompt> --output <file.html>
    mc geometry visualize from-activations <activations.json> --output <file.html>

Requires: poetry install -E viz (installs plotly>=5.18.0)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import typer

from modelcypher.cli.commands.geometry.helpers import (
    forward_through_backbone,
    resolve_model_backbone,
)
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output

app = typer.Typer(no_args_is_help=True)
logger = logging.getLogger(__name__)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


def _ensure_viz_installed():
    """Check that plotly is installed, raise helpful error if not."""
    try:
        import plotly  # noqa: F401
    except ImportError:
        raise typer.BadParameter(
            "Visualization requires plotly. Install with: poetry install -E viz"
        )


@app.command("create")
def geometry_visualize_create(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to the model directory"),
    prompt: str = typer.Argument(..., help="Prompt to analyze"),
    output: Path = typer.Option(
        Path("manifold.html"),
        "--output",
        "-o",
        help="Output file path (.html for interactive, .json for data)",
    ),
):
    """
    Create 3D manifold visualization from model activations.

    Runs a prompt through the model, captures hidden state activations,
    projects them through a dimension cascade (high-D → 4D → 3D),
    and renders an interactive visualization.

    The visualization shows ACTUAL geometry:
    - Gram transport preserves relational structure exactly
    - Ollivier-Ricci curvature reflects true manifold curvature
    - The 3D "shadow" IS the manifold shape

    Example:
        mc geometry visualize create /path/to/model "What is justice?" -o justice.html
    """
    _ensure_viz_installed()
    context = _context(ctx)

    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.dimension_cascade import DimensionCascade
    from modelcypher.viz.manifold_viewer import ManifoldViewer

    dims = [4, 3]

    typer.echo(f"Loading model from {model_path}...")
    model, tokenizer = load_model_for_training(model_path)

    model_type = getattr(model, "model_type", "unknown")
    resolved = resolve_model_backbone(model, model_type)

    if not resolved:
        raise typer.BadParameter(
            f"Could not resolve architecture for model at {model_path}"
        )

    embed_tokens, layers, norm = resolved
    num_layers = len(layers)
    target_layer = num_layers - 1
    typer.echo(f"Architecture resolved: {num_layers} layers, probing layer {target_layer}")

    backend = get_default_backend()

    # Tokenize and capture activations
    tokens = tokenizer.encode(prompt)
    input_ids = backend.array([tokens])
    typer.echo(f"Prompt tokenized: {len(tokens)} tokens")

    # Forward through backbone to capture hidden states
    hidden = forward_through_backbone(
        input_ids,
        embed_tokens,
        layers,
        norm,
        target_layer,
        backend,
    )
    backend.eval(hidden)

    # hidden is [batch, seq, hidden_dim] - extract all token positions
    if len(hidden.shape) == 3:
        activations = hidden[0]  # [seq, hidden_dim]
    else:
        activations = hidden

    n_tokens, hidden_dim = activations.shape
    typer.echo(f"Captured activations: {n_tokens} tokens × {hidden_dim} dims")

    # Run dimension cascade (all parameters derived from data)
    typer.echo(f"Running dimension cascade: {hidden_dim}D → {dims}")
    cascade = DimensionCascade(backend)
    cascade_result = cascade.calibrate(activations, target_dims=dims)
    typer.echo(
        f"Intrinsic dimension: {cascade_result.intrinsic_dim:.1f} "
        f"(ambient: {cascade_result.original_dim})"
    )

    # Create visualization
    typer.echo("Creating visualization...")
    viewer = ManifoldViewer(
        backend,
        title=f"Manifold Geometry: {Path(model_path).name}",
    )

    # Use 3D if available, otherwise use smallest available dimension
    viz_dim = 3 if 3 in cascade_result.projections else min(cascade_result.projections.keys())
    result = viewer.create_figure(cascade_result, target_dim=viz_dim)

    # Export
    output_path = Path(output)
    if output_path.suffix == ".json":
        # Export JSON data
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result.json_data, indent=2))
        typer.echo(f"Exported JSON to {output_path}")
    else:
        # Export HTML
        viewer.export_html(result, output_path)
        typer.echo(f"Exported HTML to {output_path}")

    # Output summary
    payload = {
        "model_path": model_path,
        "prompt": prompt,
        "n_tokens": n_tokens,
        "hidden_dim": hidden_dim,
        "intrinsic_dim": cascade_result.intrinsic_dim,
        "target_dims": dims,
        "visualization_dim": viz_dim,
        "point_count": result.point_count,
        "output_file": str(output_path.absolute()),
        "geodesic_distortion": {
            str(k): v for k, v in cascade_result.geodesic_distortion.items()
        },
    }

    if context.output_format == "text":
        lines = [
            "",
            "MANIFOLD VISUALIZATION COMPLETE",
            f"  Model: {Path(model_path).name}",
            f"  Tokens: {n_tokens}",
            f"  Hidden dim: {hidden_dim} → {viz_dim}D projection",
            f"  Intrinsic dim: {cascade_result.intrinsic_dim:.1f}",
            f"  Output: {output_path.absolute()}",
        ]
        for dim, distortion in cascade_result.geodesic_distortion.items():
            lines.append(f"  Distortion at {dim}D: {distortion:.4f}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("from-activations")
def geometry_visualize_from_activations(
    ctx: typer.Context,
    activations_file: Path = typer.Argument(
        ...,
        help="JSON file with activations array [n_points, hidden_dim]",
    ),
    output: Path = typer.Option(
        Path("manifold.html"),
        "--output",
        "-o",
        help="Output file path (.html for interactive, .json for data)",
    ),
):
    """
    Create visualization from pre-computed activations.

    Load activations from a JSON file and project through dimension cascade.
    Useful for visualizing activations captured separately or from other tools.

    The JSON file should contain an array of shape [n_points, hidden_dim].

    Example:
        mc geometry visualize from-activations activations.json -o manifold.html
    """
    _ensure_viz_installed()
    context = _context(ctx)

    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.dimension_cascade import DimensionCascade
    from modelcypher.viz.manifold_viewer import ManifoldViewer

    dims = [4, 3]

    # Load activations
    typer.echo(f"Loading activations from {activations_file}...")
    data = json.loads(activations_file.read_text())

    backend = get_default_backend()

    # Handle different JSON formats
    if isinstance(data, dict):
        if "activations" in data:
            act_data = data["activations"]
        elif "points" in data:
            act_data = data["points"]
        else:
            # Assume the dict values are the activations
            act_data = list(data.values())
    else:
        act_data = data

    activations = backend.array(act_data)
    n_points, hidden_dim = activations.shape
    typer.echo(f"Loaded activations: {n_points} points × {hidden_dim} dims")

    # Run dimension cascade (all parameters derived from data)
    typer.echo(f"Running dimension cascade: {hidden_dim}D → {dims}")
    cascade = DimensionCascade(backend)
    cascade_result = cascade.calibrate(activations, target_dims=dims)
    typer.echo(
        f"Intrinsic dimension: {cascade_result.intrinsic_dim:.1f} "
        f"(ambient: {cascade_result.original_dim})"
    )

    # Create visualization
    typer.echo("Creating visualization...")
    viewer = ManifoldViewer(
        backend,
        title=f"Manifold Geometry: {activations_file.stem}",
    )

    # Use 3D if available
    viz_dim = 3 if 3 in cascade_result.projections else min(cascade_result.projections.keys())
    result = viewer.create_figure(cascade_result, target_dim=viz_dim)

    # Export
    output_path = Path(output)
    if output_path.suffix == ".json":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result.json_data, indent=2))
        typer.echo(f"Exported JSON to {output_path}")
    else:
        viewer.export_html(result, output_path)
        typer.echo(f"Exported HTML to {output_path}")

    # Output summary
    payload = {
        "source_file": str(activations_file),
        "n_points": n_points,
        "hidden_dim": hidden_dim,
        "intrinsic_dim": cascade_result.intrinsic_dim,
        "target_dims": dims,
        "visualization_dim": viz_dim,
        "point_count": result.point_count,
        "output_file": str(output_path.absolute()),
        "geodesic_distortion": {
            str(k): v for k, v in cascade_result.geodesic_distortion.items()
        },
    }

    if context.output_format == "text":
        lines = [
            "",
            "MANIFOLD VISUALIZATION COMPLETE",
            f"  Source: {activations_file.name}",
            f"  Points: {n_points}",
            f"  Hidden dim: {hidden_dim} → {viz_dim}D projection",
            f"  Intrinsic dim: {cascade_result.intrinsic_dim:.1f}",
            f"  Output: {output_path.absolute()}",
        ]
        for dim, distortion in cascade_result.geodesic_distortion.items():
            lines.append(f"  Distortion at {dim}D: {distortion:.4f}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
