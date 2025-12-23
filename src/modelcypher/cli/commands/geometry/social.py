"""
CLI commands for Social Geometry analysis.

Probes the emergent social manifold in language models - the geometric structure
encoding power hierarchies, kinship relations, and formality gradients.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

app = typer.Typer(help="Social geometry analysis commands")
logger = logging.getLogger(__name__)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


def _resolve_text_backbone(model, model_type: str):
    """Resolve text backbone components from model architecture."""
    # Strategy 1: Standard mlx_lm structure
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens, model.model.layers, getattr(model.model, "norm", None)

    # Strategy 2: Multimodal VL wrapper
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model"):
            return (
                getattr(lm.model, "embed_tokens", None),
                getattr(lm.model, "layers", None),
                getattr(lm.model, "norm", None),
            )

    # Strategy 3: Direct structure
    if hasattr(model, "embed_tokens") and hasattr(model, "layers"):
        return model.embed_tokens, model.layers, getattr(model, "norm", None)

    return None


def _forward_text_backbone(input_ids, embed_tokens, layers, norm, target_layer: int, backend: "Backend"):
    """Forward pass through text backbone."""
    hidden = embed_tokens(input_ids)
    seq_len = input_ids.shape[1]
    mask = backend.create_causal_mask(seq_len, hidden.dtype)
    actual_target = target_layer if target_layer >= 0 else len(layers) - 1

    for i, layer in enumerate(layers):
        try:
            hidden = layer(hidden, mask=mask)
        except TypeError:
            try:
                hidden = layer(hidden, mask)
            except TypeError:
                hidden = layer(hidden)
        if i == actual_target:
            break

    if norm is not None and actual_target == len(layers) - 1:
        hidden = norm(hidden)

    return hidden


@app.command("anchors")
def social_anchors(
    ctx: typer.Context,
    axis: str = typer.Option(None, "--axis", help="Filter by axis: power, kinship, formality"),
    category: str = typer.Option(None, "--category", help="Filter by category"),
) -> None:
    """
    List the Social Prime Atlas anchors.

    Shows the 23 social anchors with their axis assignments and hierarchy levels.
    These anchors probe the model's social geometry.

    Axes:
    - power: slave → servant → citizen → noble → emperor
    - kinship: enemy → stranger → acquaintance → friend → family
    - formality: hey → hi → hello → greetings → salutations
    """
    context = _context(ctx)

    from modelcypher.core.domain.geometry.social_geometry import (
        SOCIAL_PRIME_ATLAS,
        SocialAxis,
    )

    anchors = SOCIAL_PRIME_ATLAS

    if axis:
        try:
            axis_enum = SocialAxis(axis.lower())
            anchors = [a for a in anchors if a.axis == axis_enum]
        except ValueError:
            typer.echo(f"Invalid axis: {axis}. Use: power, kinship, formality", err=True)
            raise typer.Exit(1)

    if category:
        anchors = [a for a in anchors if a.category.value == category]

    if context.output_format == "text":
        lines = [
            "=" * 70,
            "SOCIAL PRIME ATLAS",
            "=" * 70,
            "",
            f"{'Name':<15} {'Axis':<12} {'Level':<6} {'Category':<25}",
            "-" * 70,
        ]
        for a in anchors:
            lines.append(f"{a.name:<15} {a.axis.value:<12} {a.level:<6} {a.category.value:<25}")
        lines.append("")
        lines.append(f"Total: {len(anchors)} anchors")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    payload = {
        "_schema": "mc.geometry.social.anchors.v1",
        "anchors": [
            {
                "name": a.name,
                "axis": a.axis.value,
                "level": a.level,
                "category": a.category.value,
                "prompt": a.prompt,
            }
            for a in anchors
        ],
        "total": len(anchors),
    }
    write_output(payload, context.output_format, context.pretty)


@app.command("probe-model")
def social_probe_model(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to the model directory"),
    layer: int = typer.Option(-1, help="Layer to analyze (default is last)"),
    output_file: str = typer.Option(None, "--output", "-o", help="File to save activations"),
) -> None:
    """Probe a model for social geometry structure."""
    context = _context(ctx)

    from modelcypher.core.domain.geometry.social_geometry import (
        SOCIAL_PRIME_ATLAS,
        SocialGeometryAnalyzer,
    )
    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.backends.mlx_backend import MLXBackend

    typer.echo(f"Loading model from {model_path}...")
    model, tokenizer = load_model_for_training(model_path)

    model_type = getattr(model, "model_type", "unknown")
    resolved = _resolve_text_backbone(model, model_type)

    if not resolved:
        typer.echo("Error: Could not resolve architecture.", err=True)
        raise typer.Exit(1)

    embed_tokens, layers, norm = resolved
    if embed_tokens is None or layers is None:
        typer.echo("Error: Could not find embedding or layers.", err=True)
        raise typer.Exit(1)

    num_layers = len(layers)
    target_layer = layer if layer >= 0 else num_layers - 1
    typer.echo(f"Architecture resolved: {num_layers} layers, probing layer {target_layer}")

    backend = MLXBackend()
    anchor_activations = {}

    typer.echo(f"Probing {len(SOCIAL_PRIME_ATLAS)} social anchors...")

    for anchor in SOCIAL_PRIME_ATLAS:
        try:
            tokens = tokenizer.encode(anchor.prompt)
            input_ids = backend.array([tokens])

            hidden = _forward_text_backbone(
                input_ids, embed_tokens, layers, norm,
                target_layer=target_layer,
                backend=backend,
            )

            activation = backend.mean(hidden[0], axis=0)
            backend.eval(activation)
            anchor_activations[anchor.name] = activation

        except Exception as e:
            typer.echo(f"  Warning: Failed anchor {anchor.name}: {e}", err=True)

    if not anchor_activations:
        typer.echo("Error: No activations extracted.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Extracted {len(anchor_activations)} activations.")

    # Save activations if requested
    if output_file:
        activations_json = {
            name: backend.to_numpy(act).tolist()
            for name, act in anchor_activations.items()
        }
        Path(output_file).write_text(json.dumps(activations_json, indent=2))
        typer.echo(f"Saved activations to {output_file}")

    # Run analysis
    typer.echo("Running social geometry analysis...")
    analyzer = SocialGeometryAnalyzer(backend=backend)
    report = analyzer.full_analysis(anchor_activations)

    payload = {
        "_schema": "mc.geometry.social.probe_model.v1",
        "model_path": model_path,
        "anchors_probed": len(anchor_activations),
        "layer": layer,
        **report.to_dict(),
        "verdict": (
            "STRONG SOCIAL MANIFOLD - Clear power/kinship/formality axes detected."
            if report.has_social_manifold and report.social_manifold_score > 0.6
            else "MODERATE SOCIAL MANIFOLD - Some social structure detected."
            if report.has_social_manifold
            else "WEAK SOCIAL MANIFOLD - Limited social geometry found."
        ),
    }

    if context.output_format == "text":
        lines = [
            "=" * 60,
            f"SOCIAL GEOMETRY ANALYSIS: {Path(model_path).name}",
            "=" * 60,
            "",
            f"Anchors Probed: {len(anchor_activations)}/23",
            f"Layer Analyzed: {layer if layer != -1 else 'last'}",
            "",
            f"Has Social Manifold: {'YES' if report.has_social_manifold else 'NO'}",
            f"Social Manifold Score: {report.social_manifold_score:.2f}",
            "",
            "-" * 40,
            "Axis Orthogonality:",
            f"  Power ⊥ Kinship:    {report.axis_orthogonality.power_kinship:.2%}",
            f"  Power ⊥ Formality:  {report.axis_orthogonality.power_formality:.2%}",
            f"  Kinship ⊥ Formality:{report.axis_orthogonality.kinship_formality:.2%}",
            "",
            "Gradient Consistency:",
            f"  Power:    {'MONOTONIC' if report.gradient_consistency.power_monotonic else 'non-monotonic'} (r={report.gradient_consistency.power_correlation:.2f})",
            f"  Kinship:  {'MONOTONIC' if report.gradient_consistency.kinship_monotonic else 'non-monotonic'} (r={report.gradient_consistency.kinship_correlation:.2f})",
            f"  Formality:{'MONOTONIC' if report.gradient_consistency.formality_monotonic else 'non-monotonic'} (r={report.gradient_consistency.formality_correlation:.2f})",
            "",
            "Power Axis:",
            f"  Detected: {'YES' if report.power_gradient.power_axis_detected else 'NO'}",
            f"  Status Correlation: {report.power_gradient.status_correlation:.2f}",
            "",
            "=" * 60,
            payload["verdict"],
            "=" * 60,
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("analyze")
def social_analyze(
    ctx: typer.Context,
    activations_file: str = typer.Argument(..., help="JSON file with anchor activations"),
) -> None:
    """
    Analyze social geometry from pre-computed activations.

    Input format: JSON with {anchor_name: [activation_vector]} mappings.
    """
    context = _context(ctx)

    from modelcypher.core.domain.geometry.social_geometry import SocialGeometryAnalyzer
    from modelcypher.backends.mlx_backend import MLXBackend

    path = Path(activations_file)
    if not path.exists():
        typer.echo(f"File not found: {activations_file}", err=True)
        raise typer.Exit(1)

    with open(path) as f:
        raw_activations = json.load(f)

    backend = MLXBackend()
    activations = {name: backend.array(vec) for name, vec in raw_activations.items()}

    analyzer = SocialGeometryAnalyzer(backend=backend)
    report = analyzer.full_analysis(activations)

    payload = {
        "_schema": "mc.geometry.social.analyze.v1",
        "source_file": activations_file,
        "anchors_analyzed": len(activations),
        **report.to_dict(),
    }

    write_output(payload, context.output_format, context.pretty)


__all__ = ["app"]
