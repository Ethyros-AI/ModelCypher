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

"""3D Spatial Metrology CLI commands.

Probes how language models capture 3-dimensional spatial relationships
in their internal representations. Tests whether the latent manifold
encodes a geometrically consistent 3D world model.

Commands:
    mc geometry spatial analyze <model_path>
    mc geometry spatial gravity <model_path>
    mc geometry spatial euclidean <activations_file>
    mc geometry spatial anchors
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


@app.command("anchors")
def spatial_anchors(
    ctx: typer.Context,
    axis: str = typer.Option(None, "--axis", help="Filter by axis: x_lateral, y_vertical, z_depth"),
    category: str = typer.Option(
        None, "--category", help="Filter by category: vertical, lateral, depth, mass, furniture"
    ),
) -> None:
    """
    List the Spatial Prime Atlas anchors.

    Shows the 23 spatial anchors with their expected 3D coordinates (X, Y, Z)
    and categories. These anchors probe the model's 3D world model.

    Categories:
    - vertical: ceiling, floor, sky, ground (Y-axis)
    - lateral: left_hand, right_hand, west, east (X-axis)
    - depth: foreground, background, horizon (Z-axis)
    - mass: balloon, stone, feather, anvil (gravity test)
    - furniture: chair, table, lamp, rug (virtual room)
    """
    context = _context(ctx)

    from modelcypher.core.domain.geometry.spatial_3d import (
        SPATIAL_PRIME_ATLAS,
        SpatialAxis,
        get_spatial_anchors_by_axis,
    )

    # Filter anchors
    if axis:
        try:
            axis_enum = SpatialAxis(axis)
            anchors = get_spatial_anchors_by_axis(axis_enum)
        except ValueError:
            typer.echo(f"Invalid axis: {axis}. Use: x_lateral, y_vertical, z_depth", err=True)
            raise typer.Exit(1)
    else:
        anchors = SPATIAL_PRIME_ATLAS

    if category:
        anchors = [a for a in anchors if a.category == category]

    payload = {
        "_schema": "mc.geometry.spatial.anchors.v1",
        "anchors": [
            {
                "name": a.name,
                "prompt": a.prompt,
                "expected_x": a.expected_x,
                "expected_y": a.expected_y,
                "expected_z": a.expected_z,
                "category": a.category,
            }
            for a in anchors
        ],
        "count": len(anchors),
        "categories": list(set(a.category for a in anchors)),
    }

    if context.output_format == "text":
        lines = [
            "SPATIAL PRIME ATLAS",
            f"Total: {len(anchors)} anchors",
            "",
            f"{'Name':<15} {'X':>6} {'Y':>6} {'Z':>6} {'Category':<12} Prompt",
            "-" * 80,
        ]
        for a in anchors:
            lines.append(
                f"{a.name:<15} {a.expected_x:>6.2f} {a.expected_y:>6.2f} {a.expected_z:>6.2f} {a.category:<12} {a.prompt[:40]}"
            )

        lines.extend(
            [
                "",
                "Legend:",
                "  X: Lateral (Left=-1, Right=+1)",
                "  Y: Vertical (Down=-1, Up=+1) - Gravity axis",
                "  Z: Depth (Far=-1, Near=+1) - Perspective axis",
            ]
        )
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("euclidean")
def spatial_euclidean(
    ctx: typer.Context,
    activations_file: str = typer.Argument(
        ..., help="JSON file with anchor_name -> activation_vector mapping"
    ),
) -> None:
    """
    Test Euclidean consistency of spatial anchor representations.

    Checks if the Pythagorean theorem holds in latent space:
    dist(A,C)² ≈ dist(A,B)² + dist(B,C)² for right-angle triplets.

    If consistency score > 0.6 and no triangle inequality violations,
    the model has internalized Euclidean 3D geometry.

    Input: JSON file with {anchor_name: [activation_vector]} mapping.
    """
    context = _context(ctx)

    from modelcypher.backends.mlx_backend import MLXBackend
    from modelcypher.core.domain.geometry.spatial_3d import (
        EuclideanConsistencyAnalyzer,
    )

    # Load activations
    activations_data = json.loads(Path(activations_file).read_text())

    backend = MLXBackend()
    anchor_activations = {name: backend.array(vec) for name, vec in activations_data.items()}

    analyzer = EuclideanConsistencyAnalyzer(backend=backend)
    result = analyzer.analyze(anchor_activations)

    payload = {
        "_schema": "mc.geometry.spatial.euclidean.v1",
        **result.to_dict(),
        "interpretation": (
            "The model has a 3D Euclidean world model."
            if result.is_euclidean
            else "The model's spatial representation is non-Euclidean."
        ),
        "nextActions": [
            "mc geometry spatial gravity <model> to test gravity gradient",
            "mc geometry spatial analyze <model> for full 3D analysis",
        ],
    }

    if context.output_format == "text":
        lines = [
            "EUCLIDEAN CONSISTENCY ANALYSIS",
            "",
            f"Is Euclidean: {'Yes' if result.is_euclidean else 'No'}",
            f"Consistency Score: {result.consistency_score:.2f}",
            f"Pythagorean Error: {result.pythagorean_error:.4f}",
            f"Triangle Inequality Violations: {result.triangle_inequality_violations}",
            f"Dimensionality Estimate: {result.dimensionality_estimate:.1f}",
            "",
            "Axis Orthogonality:",
        ]
        for axis, score in result.axis_orthogonality.items():
            lines.append(f"  {axis}: {score:.2%}")

        lines.extend(
            [
                "",
                "Interpretation:",
                payload["interpretation"],
            ]
        )
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("gravity")
def spatial_gravity(
    ctx: typer.Context,
    activations_file: str = typer.Argument(
        ..., help="JSON file with anchor_name -> activation_vector mapping"
    ),
) -> None:
    """
    Analyze gravity gradient in latent representations.

    Tests if the model has a "gravity gradient" where heavy objects
    are pulled toward "down" (Floor, Ground) in latent space.

    High mass correlation (>0.5) indicates the model understands
    physical mass as a geometric property, not just a word.

    Input: JSON file with {anchor_name: [activation_vector]} mapping.
    """
    context = _context(ctx)

    from modelcypher.backends.mlx_backend import MLXBackend
    from modelcypher.core.domain.geometry.spatial_3d import (
        GravityGradientAnalyzer,
    )

    # Load activations
    activations_data = json.loads(Path(activations_file).read_text())

    backend = MLXBackend()
    anchor_activations = {name: backend.array(vec) for name, vec in activations_data.items()}

    analyzer = GravityGradientAnalyzer(backend=backend)
    result = analyzer.analyze(anchor_activations)

    payload = {
        "_schema": "mc.geometry.spatial.gravity.v1",
        **result.to_dict(),
        "interpretation": (
            "Gravity gradient detected - the model has a physics engine for mass."
            if result.gravity_axis_detected
            else "No gravity gradient - spatial reasoning may be surface-level."
        ),
        "nextActions": [
            "mc geometry spatial euclidean <file> to verify Euclidean structure",
            "mc geometry spatial analyze <model> for full 3D analysis",
        ],
    }

    if context.output_format == "text":
        lines = [
            "GRAVITY GRADIENT ANALYSIS",
            "",
            f"Gravity Axis Detected: {'Yes' if result.gravity_axis_detected else 'No'}",
            f"Mass Correlation: {result.mass_correlation:.2f}",
            "",
            f"Sink Anchors (Heavy): {', '.join(result.sink_anchors)}",
            f"Float Anchors (Light): {', '.join(result.float_anchors)}",
        ]

        if result.gravity_direction is not None:
            lines.extend(
                [
                    "",
                    f"Gravity Direction Vector: [{', '.join(f'{x:.3f}' for x in result.gravity_direction[:5])}...]",
                ]
            )

        lines.extend(
            [
                "",
                "Interpretation:",
                payload["interpretation"],
            ]
        )
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("density")
def spatial_density(
    ctx: typer.Context,
    activations_file: str = typer.Argument(
        ..., help="JSON file with anchor_name -> activation_vector mapping"
    ),
) -> None:
    """
    Probe volumetric density of spatial representations.

    Tests if physical objects have representational densities that
    match their real-world properties:
    - Heavy objects should have "denser" representations
    - Distant objects should have attenuated density (inverse-square law)

    High density-mass correlation suggests the model represents
    physical properties geometrically.

    Input: JSON file with {anchor_name: [activation_vector]} mapping.
    """
    context = _context(ctx)

    from modelcypher.backends.mlx_backend import MLXBackend
    from modelcypher.core.domain.geometry.spatial_3d import (
        VolumetricDensityProber,
    )

    # Load activations
    activations_data = json.loads(Path(activations_file).read_text())

    backend = MLXBackend()
    anchor_activations = {name: backend.array(vec) for name, vec in activations_data.items()}

    prober = VolumetricDensityProber(backend=backend)
    result = prober.analyze(anchor_activations)

    payload = {
        "_schema": "mc.geometry.spatial.density.v1",
        **result.to_dict(),
        "interpretation": (
            f"Density-mass correlation: {result.density_mass_correlation:.2f}. "
            f"Inverse-square compliance: {result.inverse_square_compliance:.2f}. "
            + (
                "Physical mass is encoded geometrically."
                if abs(result.density_mass_correlation) > 0.3
                else "Mass encoding is weak."
            )
        ),
    }

    if context.output_format == "text":
        lines = [
            "VOLUMETRIC DENSITY ANALYSIS",
            "",
            f"Density-Mass Correlation: {result.density_mass_correlation:.2f}",
            f"Perspective Attenuation: {result.perspective_attenuation:.2f}",
            f"Inverse-Square Compliance: {result.inverse_square_compliance:.2f}",
            "",
            "Anchor Densities:",
        ]
        for name, density in sorted(result.anchor_densities.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"  {name}: {density:.2f}")

        lines.extend(
            [
                "",
                "Interpretation:",
                payload["interpretation"],
            ]
        )
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("analyze")
def spatial_analyze(
    ctx: typer.Context,
    activations_file: str = typer.Argument(
        ..., help="JSON file with anchor_name -> activation_vector mapping"
    ),
) -> None:
    """
    Run full 3D world model analysis.

    Comprehensive analysis combining:
    - Euclidean consistency (Pythagorean theorem test)
    - Gravity gradient (mass -> down correlation)
    - Volumetric density (inverse-square law)

    All models encode physics geometrically. The world_model_score measures
    Visual-Spatial Grounding Density: how concentrated the model's probability
    mass is along human-perceptual 3D axes. Higher scores indicate alignment
    with visual experience; lower scores indicate physics encoded along
    alternative geometric axes (linguistic, formula-based, higher-dimensional).

    Input: JSON file with {anchor_name: [activation_vector]} mapping.
    """
    context = _context(ctx)

    from modelcypher.backends.mlx_backend import MLXBackend
    from modelcypher.core.domain.geometry.spatial_3d import (
        Spatial3DAnalyzer,
    )

    # Load activations
    activations_data = json.loads(Path(activations_file).read_text())

    backend = MLXBackend()
    anchor_activations = {name: backend.array(vec) for name, vec in activations_data.items()}

    analyzer = Spatial3DAnalyzer(backend=backend)
    report = analyzer.full_analysis(anchor_activations)

    payload = {
        "_schema": "mc.geometry.spatial.full_analysis.v1",
        **report.to_dict(),
        "verdict": (
            "HIGH VISUAL GROUNDING - Probability concentrated on human-perceptual 3D axes."
            if report.has_3d_world_model and report.physics_engine_detected
            else "MODERATE GROUNDING - 3D structure present, probability more diffuse."
            if report.has_3d_world_model
            else "ALTERNATIVE GROUNDING - Physics encoded geometrically along non-visual axes."
        ),
    }

    if context.output_format == "text":
        lines = [
            "=" * 60,
            "3D WORLD MODEL ANALYSIS",
            "=" * 60,
            "",
            f"Has 3D World Model: {'YES' if report.has_3d_world_model else 'NO'}",
            f"World Model Score: {report.world_model_score:.2f}",
            f"Physics Engine Detected: {'YES' if report.physics_engine_detected else 'NO'}",
            "",
            "-" * 40,
            "EUCLIDEAN CONSISTENCY",
            "-" * 40,
            f"  Consistency Score: {report.euclidean_consistency.consistency_score:.2f}",
            f"  Pythagorean Error: {report.euclidean_consistency.pythagorean_error:.4f}",
            f"  Triangle Violations: {report.euclidean_consistency.triangle_inequality_violations}",
            f"  Intrinsic Dimension: {report.euclidean_consistency.dimensionality_estimate:.1f}",
            "",
            "-" * 40,
            "GRAVITY GRADIENT",
            "-" * 40,
            f"  Gravity Detected: {'Yes' if report.gravity_gradient.gravity_axis_detected else 'No'}",
            f"  Mass Correlation: {report.gravity_gradient.mass_correlation:.2f}",
            f"  Sink Anchors: {', '.join(report.gravity_gradient.sink_anchors[:3])}...",
            "",
            "-" * 40,
            "VOLUMETRIC DENSITY",
            "-" * 40,
            f"  Density-Mass Correlation: {report.volumetric_density.density_mass_correlation:.2f}",
            f"  Inverse-Square Compliance: {report.volumetric_density.inverse_square_compliance:.2f}",
            "",
            "=" * 60,
            "VERDICT",
            "=" * 60,
            payload["verdict"],
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("probe-model")
def spatial_probe_model(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to the model directory"),
    layer: int = typer.Option(-1, help="Layer to analyze (default is last)"),
    output_file: str = typer.Option(None, "--output-file", "-o", help="File to save activations"),
) -> None:
    """Probe a model for 3D world model geometry."""
    context = _context(ctx)

    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.backends.mlx_backend import MLXBackend
    from modelcypher.core.domain.geometry.spatial_3d import (
        SPATIAL_PRIME_ATLAS,
        Spatial3DAnalyzer,
    )

    typer.echo(f"Loading model from {model_path}...")
    model, tokenizer = load_model_for_training(model_path)

    model_type = getattr(model, "model_type", "unknown")
    resolved = resolve_model_backbone(model, model_type)

    if not resolved:
        typer.echo("Error: Could not resolve architecture.", err=True)
        raise typer.Exit(1)

    embed_tokens, layers, norm = resolved
    num_layers = len(layers)
    target_layer = layer if layer >= 0 else num_layers - 1
    typer.echo(f"Architecture resolved: {num_layers} layers, probing layer {target_layer}")

    backend = MLXBackend()
    anchor_activations = {}
    pending_activations = []

    typer.echo(f"Probing {len(SPATIAL_PRIME_ATLAS)} spatial anchors...")

    for anchor in SPATIAL_PRIME_ATLAS:
        try:
            tokens = tokenizer.encode(anchor.prompt)
            input_ids = backend.array([tokens])

            hidden = forward_through_backbone(
                input_ids,
                embed_tokens,
                layers,
                norm,
                target_layer=target_layer,
                backend=backend,
            )

            activation = backend.mean(hidden[0], axis=0)
            # Use async_eval for pipeline parallelism - overlap CPU tokenization with GPU compute
            backend.async_eval(activation)
            anchor_activations[anchor.name] = activation
            pending_activations.append(activation)

        except Exception as e:
            typer.echo(f"  Warning: Failed anchor {anchor.name}: {e}", err=True)

    # Final sync - materialize all pending activations
    if pending_activations:
        backend.eval(*pending_activations)

    if not anchor_activations:
        typer.echo("Error: No activations extracted.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Extracted {len(anchor_activations)} activations.")

    # Save activations if requested
    if output_file:
        activations_json = {
            name: backend.to_numpy(act).tolist() for name, act in anchor_activations.items()
        }
        Path(output_file).write_text(json.dumps(activations_json, indent=2))
        typer.echo(f"Saved activations to {output_file}")

    # Run full analysis
    typer.echo("Running 3D world model analysis...")
    analyzer = Spatial3DAnalyzer(backend=backend)
    report = analyzer.full_analysis(anchor_activations)

    payload = {
        "_schema": "mc.geometry.spatial.probe_model.v1",
        "model_path": model_path,
        "anchors_probed": len(anchor_activations),
        "layer": layer,
        **report.to_dict(),
        "verdict": (
            "HIGH VISUAL GROUNDING - Physics probability concentrated on 3D visual axes."
            if report.has_3d_world_model and report.physics_engine_detected
            else "MODERATE GROUNDING - 3D structure detected, probability diffuse."
            if report.has_3d_world_model
            else "ALTERNATIVE GROUNDING - Physics encoded geometrically along non-visual axes."
        ),
    }

    if context.output_format == "text":
        lines = [
            "=" * 60,
            f"3D WORLD MODEL ANALYSIS: {Path(model_path).name}",
            "=" * 60,
            "",
            f"Anchors Probed: {len(anchor_activations)}/23",
            f"Layer Analyzed: {layer if layer != -1 else 'last'}",
            "",
            f"Has 3D World Model: {'YES' if report.has_3d_world_model else 'NO'}",
            f"World Model Score: {report.world_model_score:.2f}",
            f"Physics Engine: {'DETECTED' if report.physics_engine_detected else 'NOT FOUND'}",
            "",
            "-" * 40,
            "Key Metrics:",
            f"  Euclidean Consistency: {report.euclidean_consistency.consistency_score:.2f}",
            f"  Gravity Correlation: {report.gravity_gradient.mass_correlation:.2f}",
            f"  Axis Orthogonality: {list(report.euclidean_consistency.axis_orthogonality.values())[0]:.2%}"
            if report.euclidean_consistency.axis_orthogonality
            else "  Axis Orthogonality: N/A",
            "",
            "=" * 60,
            payload["verdict"],
            "=" * 60,
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("cross-grounding-feasibility")
def cross_grounding_feasibility(
    ctx: typer.Context,
    source_activations_file: str = typer.Argument(
        ..., help="JSON file with source model anchor activations"
    ),
    target_activations_file: str = typer.Argument(
        ..., help="JSON file with target model anchor activations"
    ),
) -> None:
    """
    Estimate feasibility of cross-grounding knowledge transfer.

    Compares the coordinate systems of two models to determine how much
    "rotation" exists between their grounding axes. Lower rotation means
    easier transfer; higher rotation requires more sophisticated mapping.

    This is a pre-flight check before running a full transfer.

    Input: Two JSON files with {anchor_name: [activation_vector]} mappings.
    """
    context = _context(ctx)

    from modelcypher.backends.mlx_backend import MLXBackend
    from modelcypher.core.domain.geometry.cross_grounding_transfer import (
        CrossGroundingTransferEngine,
    )

    # Load activations
    source_data = json.loads(Path(source_activations_file).read_text())
    target_data = json.loads(Path(target_activations_file).read_text())

    backend = MLXBackend()
    source_anchors = {name: backend.array(vec) for name, vec in source_data.items()}
    target_anchors = {name: backend.array(vec) for name, vec in target_data.items()}

    engine = CrossGroundingTransferEngine(backend=backend)
    feasibility = engine.estimate_transfer_feasibility(source_anchors, target_anchors)

    payload = {
        "_schema": "mc.geometry.spatial.cross_grounding_feasibility.v1",
        **feasibility,
        "nextActions": [
            "mc geometry spatial cross-grounding-transfer <source> <target> --concepts <file> to perform transfer",
            "mc geometry spatial analyze <activations> to analyze each model individually",
        ],
    }

    if context.output_format == "text":
        lines = [
            "=" * 60,
            "CROSS-GROUNDING TRANSFER FEASIBILITY",
            "=" * 60,
            "",
            f"Common Anchors: {feasibility['common_anchors']}",
            f"Grounding Rotation: {feasibility['grounding_rotation_degrees']:.1f}°",
            f"Alignment Score: {feasibility['alignment_score']:.2f}",
            f"Confidence: {feasibility['confidence']:.2f}",
            "",
            f"Feasibility: {feasibility['feasibility']}",
            "",
            "Recommendation:",
            feasibility["recommendation"],
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("cross-grounding-transfer")
def cross_grounding_transfer(
    ctx: typer.Context,
    source_activations_file: str = typer.Argument(
        ..., help="JSON file with source model anchor activations"
    ),
    target_activations_file: str = typer.Argument(
        ..., help="JSON file with target model anchor activations"
    ),
    concepts_file: str = typer.Option(
        None, "--concepts", "-c", help="JSON file with concepts to transfer {id: [vector]}"
    ),
    output_file: str = typer.Option(
        None, "--output-file", "-o", help="Output file for Ghost Anchors (JSON)"
    ),
    source_grounding: str = typer.Option(
        "unknown",
        "--source-grounding",
        help="Source grounding type: high_visual, moderate, alternative",
    ),
    target_grounding: str = typer.Option(
        "unknown",
        "--target-grounding",
        help="Target grounding type: high_visual, moderate, alternative",
    ),
) -> None:
    """
    Transfer knowledge from source to target model via cross-grounding.

    Uses Density Re-mapping to transfer concepts by preserving Relational Stress
    (distances to universal anchors) rather than absolute coordinates.

    This is the "3D Printer" for high-dimensional knowledge transfer.

    If --concepts is not provided, will synthesize Ghost Anchors for all
    source anchors as a demonstration.

    Input: Two JSON files with anchor activations, optional concepts file.
    Output: Ghost Anchors with synthesized target positions.
    """
    context = _context(ctx)

    from modelcypher.backends.mlx_backend import MLXBackend
    from modelcypher.core.domain.geometry.cross_grounding_transfer import (
        CrossGroundingTransferEngine,
    )

    # Load activations
    source_data = json.loads(Path(source_activations_file).read_text())
    target_data = json.loads(Path(target_activations_file).read_text())

    backend = MLXBackend()
    source_anchors = {name: backend.array(vec) for name, vec in source_data.items()}
    target_anchors = {name: backend.array(vec) for name, vec in target_data.items()}

    # Load concepts or use source anchors as demo
    if concepts_file:
        concepts_data = json.loads(Path(concepts_file).read_text())
        concepts = {name: backend.array(vec) for name, vec in concepts_data.items()}
    else:
        # Use a subset of source anchors as demo
        demo_concepts = ["chair", "floor", "ceiling", "left_hand", "background"]
        concepts = {k: v for k, v in source_anchors.items() if k in demo_concepts}
        if not concepts:
            # Fallback to first 5 anchors
            concepts = dict(list(source_anchors.items())[:5])

    engine = CrossGroundingTransferEngine(backend=backend)
    result = engine.transfer_concepts(
        concepts=concepts,
        source_anchors=source_anchors,
        target_anchors=target_anchors,
        source_grounding=source_grounding,
        target_grounding=target_grounding,
    )

    # Serialize Ghost Anchors
    ghost_anchors_serialized = [
        {
            "concept_id": g.concept_id,
            "source_position": g.source_position.tolist(),
            "target_position": g.target_position.tolist(),
            "stress_preservation": g.stress_preservation,
            "synthesis_confidence": g.synthesis_confidence,
            "warning": g.warning,
        }
        for g in result.ghost_anchors
    ]

    payload = {
        "_schema": "mc.geometry.spatial.cross_grounding_transfer.v1",
        "source_grounding": result.source_model_grounding,
        "target_grounding": result.target_model_grounding,
        "grounding_rotation": {
            "angle_degrees": result.grounding_rotation.angle_degrees,
            "alignment_score": result.grounding_rotation.alignment_score,
            "is_aligned": result.grounding_rotation.is_aligned,
            "confidence": result.grounding_rotation.confidence,
        },
        "ghost_anchors": ghost_anchors_serialized,
        "mean_stress_preservation": result.mean_stress_preservation,
        "min_stress_preservation": result.min_stress_preservation,
        "successful_transfers": result.successful_transfers,
        "failed_transfers": result.failed_transfers,
        "interpretability_gap": result.interpretability_gap,
        "recommendation": result.recommendation,
        "nextActions": [
            "Use Ghost Anchor target_positions for downstream tasks",
            "mc geometry spatial analyze to verify target positions",
        ],
    }

    # Save to file if requested
    if output_file:
        Path(output_file).write_text(json.dumps(payload, indent=2))

    if context.output_format == "text":
        lines = [
            "=" * 60,
            "CROSS-GROUNDING KNOWLEDGE TRANSFER",
            "=" * 60,
            "",
            f"Source Grounding: {result.source_model_grounding}",
            f"Target Grounding: {result.target_model_grounding}",
            f"Grounding Rotation: {result.grounding_rotation.angle_degrees:.1f}°",
            "",
            f"Concepts Transferred: {result.successful_transfers}",
            f"Failed Transfers: {result.failed_transfers}",
            f"Mean Stress Preservation: {result.mean_stress_preservation:.2%}",
            "",
            "-" * 40,
            "GHOST ANCHORS",
            "-" * 40,
        ]
        for g in result.ghost_anchors:
            status = "+" if g.stress_preservation >= 0.5 else "!"
            lines.append(
                f"  {status} {g.concept_id}: stress={g.stress_preservation:.2f}, conf={g.synthesis_confidence:.2f}"
            )
            if g.warning:
                lines.append(f"      Warning: {g.warning}")

        lines.extend(
            [
                "",
                "=" * 60,
                "RECOMMENDATION",
                "=" * 60,
                result.recommendation,
            ]
        )

        if output_file:
            lines.append(f"\nGhost Anchors saved to: {output_file}")

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
